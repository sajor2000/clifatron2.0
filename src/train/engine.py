"""Resumable training engine — single-device and DDP.

Builds DataLoader(s), runs optimizer-update-based accumulation with bf16 autocast,
clips once per update, validates periodically, saves epoch-boundary checkpoints,
and records a provenance manifest.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import yaml

from src.train.checkpoint import save_checkpoint, load_checkpoint
from src.train.manifest import Manifest


def setup_ddp() -> tuple[int, bool]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return local, dist.get_rank() == 0
    return 0, True


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


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
        self.ckpt_dir = Path(tcfg["runtime"].get("ckpt_dir", "checkpoints"))
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
    max_updates: int | None = None,
):
    model.train()
    ml = MetricsLog()
    opt.zero_grad(set_to_none=True)
    micro = 0
    updates = 0
    samples_seen, tokens_seen, ntp_tokens = 0, 0, 0

    for batch_idx, batch in enumerate(dl):
        batch = _prepare_batch(batch, dev)

        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
            losses = model(batch)
            loss = losses["total"]

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
          resume_ckpt=None, seed=42):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

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
        target_model = model.module if is_distributed() and hasattr(model, "module") else model
        target_model.load_state_dict(loaded["model"])
        opt.load_state_dict(loaded["optimizer"])
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

    while global_step < tcfg.total_steps:
        if hasattr(train_dl, "sampler") and hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(epoch)
        if hasattr(train_dl, "dataset") and hasattr(train_dl.dataset, "set_epoch"):
            train_dl.dataset.set_epoch(epoch)

        ml, sm, tok, ntp, updates = _train_one_epoch(
            model, train_dl, opt, scheduler, epoch, tcfg, dev, rank=local_rank,
            max_updates=tcfg.total_steps - global_step,
        )
        total_sm += sm
        total_tok += tok
        total_ntp += ntp
        epoch += 1
        global_step += updates

        if val_dl is not None and is_main and global_step >= next_val_step:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                vlosses = []
                for vb in val_dl:
                    vb = _prepare_batch(vb, dev)
                    vlosses.append(model(vb)["total"].item())
                manifest.record_validation(epoch, sum(vlosses) / len(vlosses))
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
