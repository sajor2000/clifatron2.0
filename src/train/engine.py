"""Resumable training engine — single-device and DDP.

Builds DataLoader(s), runs optimizer-update-based accumulation with bf16 autocast,
clips once per update, validates periodically, saves epoch-boundary checkpoints,
and records a provenance manifest.
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.data.segments import compare_binding
from src.train.checkpoint import load_checkpoint, save_checkpoint, unwrap_compiled
from src.train.manifest import Manifest


ALLOW_CPU_DDP_FLAG = "--allow-cpu-ddp"


def is_distributed_launch() -> bool:
    """A launcher (torchrun) sets RANK per process; WORLD_SIZE > 1 asks for several."""
    return "RANK" in os.environ or int(os.environ.get("WORLD_SIZE") or 1) > 1


def setup_ddp(*, allow_cpu: bool = False) -> tuple[int, bool]:
    """Join the process group of a distributed launch; (local rank, is rank 0).

    Fails closed when a distributed launch finds no CUDA: the 2x L40 recipe would
    otherwise crawl on CPU. `allow_cpu` (`--allow-cpu-ddp`) opts into a CPU rehearsal
    over gloo. Single-process CPU / MPS runs are not distributed launches and pass."""
    if not is_distributed_launch():
        return 0, True
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "distributed launch (RANK / WORLD_SIZE set) but CUDA is unavailable: refusing "
            "to run the multi-GPU recipe on CPU. Restore CUDA (driver mismatch: reboot), "
            "launch a single process without torchrun, or pass "
            f"{ALLOW_CPU_DDP_FLAG} for a CPU (gloo) rehearsal."
        )
    if "RANK" not in os.environ:
        return 0, True
    local = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    else:
        dist.init_process_group("gloo")
    return local, dist.get_rank() == 0


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def select_device(local: int) -> torch.device:
    """CUDA for the L40 box; CPU for a `--allow-cpu-ddp` rehearsal (DDP does not run on
    MPS); MPS for Mac smoke tests (AGENTS.md dev workflow); CPU last."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local}")
    if is_distributed() or not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device("mps")


def wrap_ddp(model, dev: torch.device, local: int) -> DistributedDataParallel:
    """DDP over `model`. `find_unused_parameters` stays off (KTD2): every head touches
    its parameters on every step, so a parameter left without a gradient is a real bug
    and raises. `device_ids` is a CUDA-only argument."""
    return DistributedDataParallel(model, device_ids=[local] if dev.type == "cuda" else None)


def precision_note(dev: torch.device) -> str | None:
    """What the engine's bf16 autocast really does off CUDA (every train config sets
    `runtime.precision: bf16`, including the smoke and MPS ones)."""
    dev = torch.device(dev)
    if dev.type == "cuda":
        return None
    if dev.type == "cpu" and not torch.cuda.is_available():
        return ("precision: bf16 on cpu is CPU autocast, not the CUDA bf16 kernels of the "
                "L40 recipe")
    return (f"precision: bf16 is inert on {dev.type} (engine autocast is cuda/cpu-scoped) "
            "— training runs in fp32")


def _all_gather(obj: Any) -> list[Any]:
    """Collective all-gather; every rank must call it.  Rank 0 gets the full list."""
    if not is_distributed():
        return [obj]
    world = dist.get_world_size()
    out = [None for _ in range(world)]
    dist.all_gather_object(out, obj)
    return out


class TrainConfig:
    def __init__(self, cfg: dict, tcfg: dict, mcfg: dict, total_steps: int):
        eff_batch = tcfg["batch"]["per_gpu"] * max(1, dist.get_world_size() if is_distributed() else 1) * tcfg["batch"].get("grad_accum", 1)
        self.grad_accum = tcfg["batch"].get("grad_accum", 1)
        self.val_every = tcfg.get("eval_schedule", {}).get("val_every", 2000)
        self.ckpt_every = tcfg["runtime"].get("ckpt_every", 2000)
        self.log_every = max(1, int(tcfg["runtime"].get("log_every", 10)))
        self.ckpt_dir = Path(tcfg["runtime"].get("ckpt_dir", "checkpoints"))
        self.cache_clear_every = int(tcfg["runtime"].get("cache_clear_every", 0) or 0)
        self.warmup_steps = tcfg["schedule"].get("warmup_steps", 2000)
        self.total_steps = total_steps
        self.grad_clip = tcfg["optimizer"].get("grad_clip", 1.0)
        self.cosine = tcfg["schedule"].get("cosine_decay", True)
        self.compile = mcfg.get("compile", False)
        self.effective_batch_size = eff_batch


def _get_step(opt) -> int:
    for state in opt.state_dict().get("state", {}).values():
        if isinstance(state, dict) and "step" in state:
            s = state["step"]
            return int(s) if not isinstance(s, int) else s
    return 0


def _prepare_batch(batch: dict, dev) -> dict:
    b = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            b[k] = v.to(dev, non_blocking=True)
        else:
            b[k] = v
    if "input_ids" in b and "token" not in b:
        b["token"] = b["input_ids"]
    if "last_idx" not in b and "anchor_idx" in b:
        b["last_idx"] = b["anchor_idx"]
    if "value" not in b and "value_target" in b:
        b["value"] = b["value_target"]
    if "val_mask" not in b and "value_mask" in b:
        b["val_mask"] = b["value_mask"]
    if "cr_type" not in b and "document_labels" in b:
        labels = [_select_tte_label(group) for group in b["document_labels"]]
        b["cr_mask"] = torch.tensor([label is not None for label in labels], dtype=torch.bool, device=dev)
        b["cr_type"] = torch.tensor(
            [int(label["event_cause"]) if label is not None else -1 for label in labels],
            dtype=torch.long,
            device=dev,
        )
        b["cr_bin"] = torch.tensor(
            [int(label["observed_bins"] if label.get("censored") or label.get("event_cause", -1) < 0 else label["event_bin"]) if label is not None else 0 for label in labels],
            dtype=torch.long,
            device=dev,
        )
    if "th_target" not in b and "threshold_queries" in b:
        queries = b["threshold_queries"]
        b["th_mask"] = torch.tensor([query is not None for query in queries], dtype=torch.bool, device=dev)
        b["th_target"] = torch.tensor(
            [int(query["target_idx"]) if query is not None else 0 for query in queries],
            dtype=torch.long,
            device=dev,
        )
        b["th_tau"] = torch.tensor(
            [int(query["threshold_bin"]) if query is not None else 0 for query in queries],
            dtype=torch.long,
            device=dev,
        )
        b["th_dir"] = torch.tensor(
            [int(query["direction"]) if query is not None else 0 for query in queries],
            dtype=torch.long,
            device=dev,
        )
        b["th_crossed"] = torch.tensor(
            [int(query["threshold_crossed_bin"]) if query is not None else -1 for query in queries],
            dtype=torch.long,
            device=dev,
        )
        b["th_observed_bin"] = torch.tensor(
            [int(query.get("observed_bins", 0)) if query is not None else 0 for query in queries],
            dtype=torch.long,
            device=dev,
        )
    return b


def _select_tte_label(group):
    supervised = [label for label in group if label.get("tte_mask")]
    events = [label for label in supervised if label.get("event_cause", -1) >= 0]
    if events:
        return min(events, key=lambda label: int(label.get("event_bin", label.get("observed_bins", 0))))
    if supervised:
        return max(supervised, key=lambda label: int(label.get("observed_bins", 0)))
    return None


def _restore_rng_states(rng_states) -> None:
    if not rng_states:
        return
    rank = dist.get_rank() if is_distributed() else 0
    state = rng_states[rank] if isinstance(rng_states, list) and len(rng_states) > rank else rng_states
    if isinstance(state, dict) and state.get("cpu") is not None:
        torch.set_rng_state(torch.tensor(state["cpu"], dtype=torch.uint8))
    cuda_states = state.get("cuda", {}) if isinstance(state, dict) else {}
    if torch.cuda.is_available() and isinstance(cuda_states, dict):
        local = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        value = cuda_states.get(f"cuda_{local}") or cuda_states.get("cuda_current")
        if value is not None:
            torch.cuda.set_rng_state(torch.tensor(value, dtype=torch.uint8), local)


def _scale_grads(model, denom: int) -> None:
    if denom <= 1:
        return
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(denom)


def _train_one_epoch(
    model, dl, opt, scheduler, epoch, tcfg: TrainConfig, dev, *, rank,
    max_updates: int | None = None, boundary_cb=None,
):
    model.train()
    ml = MetricsLog()
    opt.zero_grad(set_to_none=True)
    micro = 0
    updates = 0
    started_at = time.monotonic()
    samples_seen, tokens_seen, ntp_tokens = 0, 0, 0

    # KTD2: under DDP only the microstep that precedes an optimizer update exchanges
    # gradients — an accumulation boundary, or the epoch's last microbatch (the partial
    # update below). A stop at max_updates always lands on a boundary. The gradients
    # accumulated under no_sync are reduced by that synchronized step, so every rank
    # applies the same update. no_sync must enclose the forward as well as the backward.
    ddp = model if isinstance(model, DistributedDataParallel) else None
    last_batch_idx = len(dl) - 1 if ddp is not None else None

    for batch_idx, batch in enumerate(dl):
        batch = _prepare_batch(batch, dev)

        synchronized = (
            ddp is None
            or (micro + 1) % tcfg.grad_accum == 0
            or batch_idx == last_batch_idx
        )
        with contextlib.nullcontext() if synchronized else ddp.no_sync():
            with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                losses = model(batch)
                loss = losses["total"]

            nonfinite = [
                name for name, value in losses.items()
                if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
            ]
            failed = torch.tensor(bool(nonfinite), dtype=torch.int32, device=dev)
            if is_distributed():
                dist.all_reduce(failed, op=dist.ReduceOp.MAX)
            if failed.item():
                detail = ", ".join(nonfinite) if nonfinite else "another rank"
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}, batch {batch_idx}, "
                    f"rank {rank}: {detail}"
                )

            loss.backward()
        micro += 1
        samples_seen += batch["input_ids"].size(0)
        batch_tokens = int(batch.get("attention_mask", batch["input_ids"] > 0).sum().item())
        tokens_seen += batch_tokens
        ntp_tokens += int(batch["ntp_mask"].sum().item()) if "ntp_mask" in batch else batch_tokens

        if micro % tcfg.grad_accum == 0:
            _scale_grads(model, tcfg.grad_accum)
            if tcfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)
            updates += 1
            ml.record(micro // tcfg.grad_accum, losses, scheduler.get_last_lr()[0])
            if rank == 0 and (updates <= 5 or updates % tcfg.log_every == 0):
                elapsed = max(time.monotonic() - started_at, 1e-6)
                print(
                    f"epoch={epoch} update={updates} "
                    f"loss={_as_float(losses['total']):.4f} "
                    f"ntp={_as_float(losses.get('ntp', 0)):.4f} "
                    f"cr={_as_float(losses.get('cr', 0)):.4f} "
                    f"th={_as_float(losses.get('th', 0)):.4f} "
                    f"val={_as_float(losses.get('val', 0)):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.3e} "
                    f"updates_per_min={updates / elapsed * 60:.2f}",
                    flush=True,
                )
            if boundary_cb is not None:
                # Step-granular validation + checkpointing (see train()); fires on
                # every rank at the same update, before any max_updates early return.
                # epoch_done: at this update the dataloader is exhausted, so the
                # checkpoint's epoch+1 = epochs fully consumed (resume must NOT
                # replay this epoch). Mid-epoch saves keep the current epoch and
                # resume replays it from the start (documented approximation).
                epoch_done = updates >= -(-len(dl) // tcfg.grad_accum)
                boundary_cb(updates, epoch_done)
            if max_updates is not None and updates >= max_updates:
                return ml, samples_seen, int(tokens_seen), int(ntp_tokens), updates

    # Final partial accumulation
    if micro % tcfg.grad_accum != 0 and (max_updates is None or updates < max_updates):
        partial = micro % tcfg.grad_accum
        _scale_grads(model, partial)
        if tcfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        updates += 1
    return ml, samples_seen, int(tokens_seen), int(ntp_tokens), updates


def train(model, train_dl, val_dl, opt, scheduler, tcfg: TrainConfig, dev, *,
          vocab_binding, resume_ckpt=None, seed=42, fresh_schedule=False):
    """Resumable (DDP) training loop. `vocab_binding` (`segments.artifact_binding` of the
    training vocab.json; required) is recorded in every checkpoint, and a resume
    checkpoint bound to a different (or no) vocabulary/segments is refused before any
    state loads."""
    if vocab_binding is None:
        raise ValueError("train() requires vocab_binding (segments.artifact_binding of the "
                         "training vocab.json): every checkpoint is bound to it")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    note = precision_note(dev)
    if is_main and note:
        print(note, flush=True)

    manifest = Manifest(
        model_name="clifatron2",
        config=tcfg.__dict__,
        seed=seed,
        ckpt_dir=str(tcfg.ckpt_dir),
    )
    manifest.record_env()

    start_epoch = 0
    start_step = 0
    if resume_ckpt is not None:
        manifest.lineage_parent = str(resume_ckpt)
        loaded = load_checkpoint(resume_ckpt, dev)
        compare_binding(loaded.get("vocab_binding"), vocab_binding, what="resume checkpoint")
        target_model = model.module if is_distributed() and hasattr(model, "module") else model
        unwrap_compiled(target_model).load_state_dict(loaded["model"])
        # fresh_schedule: keep the model + optimizer (Adam moments) but NOT the
        # saved LR schedule — for continuation runs with a NEW config schedule
        # (e.g. extending a finished cosine; a loaded decayed schedule would pin
        # LR at its tail value forever). The optimizer state also pins group LRs
        # at save time, so a fresh schedule restores THIS run's configured LRs.
        saved_lrs = [group["lr"] for group in opt.param_groups]
        opt.load_state_dict(loaded["optimizer"])
        if fresh_schedule:
            for group, lr in zip(opt.param_groups, saved_lrs):
                group["lr"] = lr
        else:
            scheduler.load_state_dict(loaded["scheduler"])
        start_epoch = loaded.get("epoch", 0)
        start_step = loaded.get("step", 0)
        _restore_rng_states(loaded.get("rng_states"))
        manifest.lineage_parent = loaded.get("manifest", {}).get("run_id", str(resume_ckpt))

    tcfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    previous_ledger = loaded.get("manifest", {}).get("ledger", {}) if resume_ckpt is not None else {}
    total_sm = int(previous_ledger.get("samples_seen", 0))
    total_tok = int(previous_ledger.get("tokens_seen", 0))
    total_ntp = int(previous_ledger.get("ntp_eligible_tokens", 0))
    global_step = start_step
    next_val_step = ((global_step // tcfg.val_every) + 1) * tcfg.val_every
    next_ckpt_step = ((global_step // tcfg.ckpt_every) + 1) * tcfg.ckpt_every
    epoch = start_epoch

    def _maybe_checkpoint(epoch_n: int, gs: int) -> None:
        """Save at step-granular boundaries. DDP-safe: fires on every rank at the
        same update (synchronized update counts + _all_gather). The ledger totals
        lag within an epoch (running totals land at epoch end). `epoch_n` is
        EPOCHS FULLY CONSUMED (end-of-epoch saves get epoch+1, so resume never
        replays a completed epoch; mid-epoch saves replay the partial epoch)."""
        nonlocal next_ckpt_step
        if gs < next_ckpt_step:
            return
        local = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device())) if torch.cuda.is_available() else 0
        per_rank_rng = {
            f"cuda_{local}": torch.cuda.get_rng_state(local).cpu().tolist() if torch.cuda.is_available() else None,
            "cuda_current": torch.cuda.get_rng_state().cpu().tolist() if torch.cuda.is_available() else None,
        }
        all_rng = _all_gather({"cpu": torch.get_rng_state().tolist(), "cuda": per_rank_rng})
        if is_main:
            manifest.record_ledger(total_sm, total_tok, total_ntp, gs)
            save_checkpoint(
                tcfg.ckpt_dir / f"ckpt_ep{epoch_n}_step{gs}.pt",
                model=model.module if is_distributed() else model,
                optimizer=opt,
                scheduler=scheduler,
                epoch=epoch_n,
                step=gs,
                rng_states=all_rng,
                manifest=manifest,
                vocab_binding=vocab_binding,
            )
        while next_ckpt_step <= gs:
            next_ckpt_step += tcfg.ckpt_every

    def _maybe_validate(epoch_n: int, gs: int) -> None:
        """Step-granular validation. Single-process only: a rank-0-only eval
        mid-epoch would desync DDP ranks (epoch-boundary val covers DDP, below)."""
        nonlocal next_val_step
        if val_dl is None or not is_main or gs < next_val_step or is_distributed():
            return
        validation_model = (
            model.module
            if is_distributed() and hasattr(model, "module")
            else model
        )
        validation_model.eval()
        with torch.no_grad(), torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            vlosses = []
            for vb in val_dl:
                vb = _prepare_batch(vb, dev)
                value = validation_model(vb)["total"]
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError(
                        f"non-finite validation loss at epoch {epoch_n}, rank {local_rank}"
                    )
                vlosses.append(value.item())
            validation_loss = sum(vlosses) / len(vlosses)
            manifest.record_validation(epoch_n, validation_loss)
            print(
                f"validation epoch={epoch_n} global_step={gs} "
                f"loss={validation_loss:.4f}",
                flush=True,
            )
        model.train()
        while next_val_step <= gs:
            next_val_step += tcfg.val_every

    def _maybe_clear_cache(gs: int) -> None:
        """MPS ratchet guard: math-path attention saves T^2 buffers per layer for
        backward on long batches (~40 GiB for a 4x6413 batch); the MPS caching
        allocator retains those blocks at lengths that never recur, so the
        watermark ratchets to the ceiling and the next long batch OOMs (two
        crashes overnight 2026-09-26, at updates ~205 and ~410). Periodic
        empty_cache() flattens it. Opt-in via runtime.cache_clear_every (MPS
        smoke runs); no-op on CUDA/CPU."""
        if tcfg.cache_clear_every <= 0 or gs % tcfg.cache_clear_every:
            return
        mps = getattr(torch, "mps", None)
        if mps is not None and mps.is_available() and dev.type == "mps":
            mps.empty_cache()

    while global_step < tcfg.total_steps:
        # set_epoch must reach custom samplers, including batch samplers
        # (TokenBudgetBatchSampler) — DataLoader.sampler is None when batch_sampler
        # is used.
        for sm in (getattr(train_dl, "sampler", None), getattr(train_dl, "batch_sampler", None)):
            if sm is not None and hasattr(sm, "set_epoch"):
                sm.set_epoch(epoch)
        if hasattr(train_dl, "dataset") and hasattr(train_dl.dataset, "set_epoch"):
            train_dl.dataset.set_epoch(epoch)

        epoch_start = global_step

        def _boundary(epoch_updates: int, epoch_done: bool, _start=epoch_start, _epoch=epoch):
            gs = _start + epoch_updates
            _maybe_clear_cache(gs)
            _maybe_validate(_epoch, gs)
            _maybe_checkpoint(_epoch + 1 if epoch_done else _epoch, gs)

        ml, sm, tok, ntp, updates = _train_one_epoch(
            model, train_dl, opt, scheduler, epoch, tcfg, dev, rank=local_rank,
            max_updates=tcfg.total_steps - global_step, boundary_cb=_boundary,
        )
        total_sm += sm
        total_tok += tok
        total_ntp += ntp
        epoch += 1
        global_step += updates

        if val_dl is not None and is_main and global_step >= next_val_step:
            validation_model = (
                model.module
                if is_distributed() and hasattr(model, "module")
                else model
            )
            validation_model.eval()
            with torch.no_grad(), torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                vlosses = []
                for vb in val_dl:
                    vb = _prepare_batch(vb, dev)
                    value = validation_model(vb)["total"]
                    if not bool(torch.isfinite(value)):
                        raise FloatingPointError(
                            f"non-finite validation loss at epoch {epoch}, rank {local_rank}"
                        )
                    vlosses.append(value.item())
                validation_loss = sum(vlosses) / len(vlosses)
                manifest.record_validation(epoch, validation_loss)
                print(
                    f"validation epoch={epoch} global_step={global_step} "
                    f"loss={validation_loss:.4f}",
                    flush=True,
                )
            model.train()
            while next_val_step <= global_step:
                next_val_step += tcfg.val_every
        if val_dl is not None and is_distributed():
            dist.barrier()

        if global_step >= next_ckpt_step:
            local = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device())) if torch.cuda.is_available() else 0
            per_rank_rng = {
                f"cuda_{local}": torch.cuda.get_rng_state(local).cpu().tolist() if torch.cuda.is_available() else None,
                "cuda_current": torch.cuda.get_rng_state().cpu().tolist() if torch.cuda.is_available() else None,
            }
            all_rng = _all_gather({"cpu": torch.get_rng_state().tolist(), "cuda": per_rank_rng})
            if is_main:
                manifest.record_ledger(total_sm, total_tok, total_ntp, global_step)
                save_checkpoint(
                    tcfg.ckpt_dir / f"ckpt_ep{epoch}_step{global_step}.pt",
                    model=model.module if is_distributed() else model,
                    optimizer=opt,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=global_step,
                    rng_states=all_rng,
                    manifest=manifest,
                    vocab_binding=vocab_binding,
                )
            while next_ckpt_step <= global_step:
                next_ckpt_step += tcfg.ckpt_every

    manifest.record_ledger(total_sm, total_tok, total_ntp, global_step)
    return model, manifest


@dataclass
class MetricsLog:
    step: list[int] = field(default_factory=list)
    loss_ntp: list[float] = field(default_factory=list)
    loss_cr: list[float] = field(default_factory=list)
    loss_th: list[float] = field(default_factory=list)
    loss_val: list[float] = field(default_factory=list)
    loss_total: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)

    def record(self, step_n: int, losses: dict, lr_val: float):
        self.step.append(step_n)
        self.loss_ntp.append(_as_float(losses.get("ntp", 0)))
        self.loss_cr.append(_as_float(losses.get("cr", 0)))
        self.loss_th.append(_as_float(losses.get("th", 0)))
        self.loss_val.append(_as_float(losses.get("val", 0)))
        self.loss_total.append(_as_float(losses.get("total", 0)))
        self.lr.append(lr_val)


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach())
    return float(value)
