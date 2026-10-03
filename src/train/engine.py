"""Resumable training engine — single-device and DDP.

Builds DataLoader(s), runs optimizer-update-based accumulation with bf16 autocast,
clips once per update, validates periodically, saves step-granular checkpoints plus a
final one when the run ends (or stops cleanly on SIGTERM / SIGINT), and records a
provenance manifest.

The run's warm-up length and checkpoint interval scale with its length
(`resolve_schedule`): a 0.05-pass screening run is a few dozen updates, far below any
fixed interval sized for a long run.

Before each optimizer update the engine hands its update counter (restored from the
checkpoint on resume) to a model that sets its loss weights per update
(`pretrain.ObjectiveSchedule`, KTD5), and switches each head's weight decay off while
that head's weight is zero.
"""
from __future__ import annotations

import contextlib
import math
import os
import re
import signal
import threading
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


# Defaults when a config leaves the warm-up / checkpoint interval to scale with the run.
DEFAULT_WARMUP_FRAC = 0.05          # schedule.warmup_frac: share of the run's updates
DEFAULT_CHECKPOINTS_PER_RUN = 5     # runtime.checkpoints_per_run: periodic saves per run


class ScheduleError(ValueError):
    """A training schedule that cannot do its job (refused before any training)."""


@dataclass(frozen=True)
class RunSchedule:
    """The resolved schedule of one run (`resolve_schedule`)."""

    total_steps: int
    warmup_steps: int
    ckpt_every: int
    warmup_source: str      # "warmup_steps" (absolute) or "warmup_frac"
    ckpt_source: str        # "ckpt_every" (absolute) or "checkpoints_per_run"
    final_checkpoint: bool = True   # runtime.final_checkpoint

    @property
    def periodic_checkpoints(self) -> int:
        """Checkpoints the interval writes before the end (the final one is extra)."""
        return self.total_steps // self.ckpt_every

    def describe(self) -> str:
        final = " + final" if self.final_checkpoint else ", no final"
        return (f"{self.total_steps} updates, warm-up {self.warmup_steps} "
                f"({self.warmup_source}), checkpoint every {self.ckpt_every} "
                f"({self.ckpt_source}: {self.periodic_checkpoints} periodic{final})")


def resolve_schedule(tcfg: dict, total_steps: int, *, validate: bool = True) -> RunSchedule:
    """Warm-up length and checkpoint interval for a run of `total_steps` updates.

    - warm-up: `schedule.warmup_steps` when set (absolute), else
      floor(`schedule.warmup_frac` x total) (default `DEFAULT_WARMUP_FRAC`), at least one
      update and at most total - 1 (none for a one-update run);
    - checkpoint interval: `runtime.ckpt_every` when set (absolute), else
      max(1, total // `runtime.checkpoints_per_run`) (default
      `DEFAULT_CHECKPOINTS_PER_RUN`), so every run writes at least that many periodic
      checkpoints (one per update when it is shorter);
    - final checkpoint: the engine saves the run's last update unless
      `runtime.final_checkpoint` is false.

    `validate` refuses (ScheduleError) a warm-up as long as the run (the learning rate
    would never leave the linear warm-up) and a run that would write no checkpoint at all
    (an interval longer than the run with the final checkpoint switched off). Launchers
    validate; `TrainConfig` resolves without validating so hand-built test configs keep
    working. An interval longer than the run with the final checkpoint on is allowed
    (smoke runs) — the pre-flight warns about it for real runs, since a crash before the
    end would lose the whole run."""
    schedule = tcfg.get("schedule") or {}
    runtime = tcfg.get("runtime") or {}
    total = int(total_steps)
    if validate and total < 1:
        raise ScheduleError(f"a run needs at least one optimizer update, got {total}")

    if schedule.get("warmup_steps") is not None:
        warmup, warmup_source = int(schedule["warmup_steps"]), "warmup_steps"
        if validate and warmup < 0:
            raise ScheduleError(f"schedule.warmup_steps must be >= 0, got {warmup}")
    else:
        frac = schedule.get("warmup_frac")
        frac = DEFAULT_WARMUP_FRAC if frac is None else float(frac)
        if validate and not (math.isfinite(frac) and 0.0 <= frac < 1.0):
            raise ScheduleError(f"schedule.warmup_frac must be in [0, 1), got {frac}")
        # At least one warm-up update when the share rounds to zero (a few-update
        # screening run), never the whole run.
        warmup = math.floor(frac * total)
        if frac > 0 and total >= 2:
            warmup = min(max(1, warmup), total - 1)
        warmup, warmup_source = max(0, warmup), "warmup_frac"
    if validate and warmup > 0 and warmup >= total:
        raise ScheduleError(
            f"warm-up {warmup} updates ({warmup_source}) >= the run's {total} updates: the "
            "learning rate would never leave the linear warm-up. Set schedule.warmup_steps "
            "to null and use schedule.warmup_frac (scales with the run), or lower it")

    if runtime.get("ckpt_every") is not None:
        every, ckpt_source = int(runtime["ckpt_every"]), "ckpt_every"
        if validate and every < 1:
            raise ScheduleError(f"runtime.ckpt_every must be >= 1, got {every}")
    else:
        per_run = runtime.get("checkpoints_per_run")
        per_run = DEFAULT_CHECKPOINTS_PER_RUN if per_run is None else int(per_run)
        if validate and per_run < 1:
            raise ScheduleError(f"runtime.checkpoints_per_run must be >= 1, got {per_run}")
        every, ckpt_source = max(1, total // max(1, per_run)), "checkpoints_per_run"
    final = bool(runtime.get("final_checkpoint", True))
    if validate and every > total and not final:
        raise ScheduleError(
            f"runtime.ckpt_every {every} > the run's {total} updates and "
            "runtime.final_checkpoint is false: no checkpoint would be written (nothing to "
            "evaluate or resume). Set runtime.ckpt_every to null and use "
            "runtime.checkpoints_per_run (scales with the run), or keep the final checkpoint")
    return RunSchedule(total, max(0, warmup), max(1, every), warmup_source, ckpt_source,
                       final)


class TrainConfig:
    def __init__(self, cfg: dict, tcfg: dict, mcfg: dict, total_steps: int):
        eff_batch = tcfg["batch"]["per_gpu"] * max(1, dist.get_world_size() if is_distributed() else 1) * tcfg["batch"].get("grad_accum", 1)
        resolved = resolve_schedule(tcfg, total_steps, validate=False)
        self.grad_accum = tcfg["batch"].get("grad_accum", 1)
        self.val_every = tcfg.get("eval_schedule", {}).get("val_every", 2000)
        # Resolved from runtime.ckpt_every or runtime.checkpoints_per_run.
        self.ckpt_every = resolved.ckpt_every
        self.final_checkpoint = resolved.final_checkpoint
        self.log_every = max(1, int(tcfg["runtime"].get("log_every", 10)))
        self.ckpt_dir = Path(tcfg["runtime"].get("ckpt_dir", "checkpoints"))
        self.cache_clear_every = int(tcfg["runtime"].get("cache_clear_every", 0) or 0)
        # Resolved from schedule.warmup_steps or schedule.warmup_frac (the scheduler the
        # launcher builds uses the same value).
        self.warmup_steps = resolved.warmup_steps
        self.total_steps = total_steps
        # Run length in passes over the data (`schedule.passes`), when the run was sized
        # that way; `total_steps` is then `resolve_total_steps`'s update count.
        self.passes = tcfg["schedule"].get("passes")
        self.grad_clip = tcfg["optimizer"].get("grad_clip", 1.0)
        self.cosine = tcfg["schedule"].get("cosine_decay", True)
        self.compile = mcfg.get("compile", False)
        self.effective_batch_size = eff_batch


def resolve_total_steps(tcfg: dict, batches_per_pass: int) -> int:
    """The run's optimizer-update count. With `schedule.passes` (a positive number of
    passes over the training data) it is ceil(passes x updates per pass), one pass being
    ceil(`batches_per_pass` / `batch.grad_accum`) updates — the engine applies a partial
    accumulation at the end of each pass. Batches per pass are the training loader's
    `len()` on this rank (equal on every rank). Without it, `schedule.total_steps`."""
    schedule = tcfg["schedule"]
    passes = schedule.get("passes")
    if passes is None:
        return int(schedule.get("total_steps", 60000))
    passes = float(passes)
    if not math.isfinite(passes) or passes <= 0:
        raise ValueError(f"schedule.passes must be a positive number, got {passes}")
    grad_accum = int(tcfg["batch"].get("grad_accum", 1))
    updates_per_pass = -(-int(batches_per_pass) // grad_accum)
    return max(1, math.ceil(passes * updates_per_pass))


CHECKPOINT_NAME = re.compile(r"^ckpt_ep(\d+)_step(\d+)\.pt$")


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    """The checkpoint with the most optimizer updates in `ckpt_dir` (``ckpt_ep<E>_step<S>.pt``
    files written by `train`; ties by epoch), or None. Partial writes never match: the
    checkpoint writer renames a finished temporary file into place."""
    directory = Path(ckpt_dir)
    best = None
    for path in directory.glob("ckpt_ep*_step*.pt") if directory.is_dir() else ():
        match = CHECKPOINT_NAME.match(path.name)
        if match:
            key = (int(match.group(2)), int(match.group(1)))
            if best is None or key > best[0]:
                best = (key, path)
    return None if best is None else best[1]


def resolve_resume(resume: str | Path | None, ckpt_dir: str | Path) -> Path | None:
    """`--resume`: a checkpoint path, or ``latest`` for the newest checkpoint in the run's
    checkpoint directory (refused when there is none, so a mistyped run directory does
    not silently restart from scratch)."""
    if resume is None:
        return None
    if str(resume) == "latest":
        found = latest_checkpoint(ckpt_dir)
        if found is None:
            raise SystemExit(f"--resume latest: no checkpoint in {ckpt_dir}; drop --resume "
                             "to start the run from scratch")
        return found
    path = Path(resume)
    if not path.is_file():
        raise SystemExit(f"--resume {path}: no such checkpoint")
    return path


def _get_step(opt) -> int:
    for state in opt.state_dict().get("state", {}).values():
        if isinstance(state, dict) and "step" in state:
            s = state["step"]
            return int(s) if not isinstance(s, int) else s
    return 0


def _prepare_batch(batch: dict, dev) -> dict:
    """Move a collated batch to `dev` and add the names `pretrain.Model` reads.

    The per-anchor time-to-event tensors come from the collator as they are
    (`src.data.collate`): label times stay MINUTES since the anchor, and each head bins
    them on its own grid inside the model (KTD4). Nothing is binned here."""
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
    return b


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


def _objective_model(model):
    """The model under any DDP / torch.compile wrapper, when it schedules its own loss
    weights (`set_training_step`); else None."""
    module = model.module if isinstance(model, DistributedDataParallel) else model
    module = getattr(module, "_orig_mod", module)
    return module if hasattr(module, "set_training_step") else None


def _apply_objective_step(model, opt, step: int, total_steps: int) -> dict | None:
    """Loss weights for optimizer update `step`, and each head group's weight decay: off
    while the head's weight is 0 (AdamW decays a parameter whatever its gradient), the
    configured value once it trains. Every rank runs this with the same counter."""
    module = _objective_model(model)
    if module is None:
        return None
    weights = module.set_training_step(step, total_steps)
    for group in opt.param_groups:
        head = group.get("head")
        if head is not None:
            group["weight_decay"] = group["head_weight_decay"] if weights[head] > 0 else 0.0
    return weights


def _check_head_decay_groups(model, opt) -> None:
    """Fail closed when a head that can sit at zero weight shares a decaying parameter
    group: it would shrink while it is not trained. `pretrain.build_optimizer` builds
    the per-head groups."""
    module = _objective_model(model)
    if module is None or not hasattr(module, "head_parameters"):
        return
    group_of = {id(p): group for group in opt.param_groups for p in group["params"]}
    for head, params in module.head_parameters().items():
        if not module.head_weight_can_be_zero(head):
            continue
        for param in params:
            group = group_of.get(id(param))
            if group is not None and group.get("head") != head and group.get("weight_decay"):
                raise ValueError(
                    f"the {head} head can train at zero weight (curriculum or objective "
                    "arm) but its parameters share a weight-decayed optimizer group, so "
                    "AdamW would shrink them untrained; build the optimizer with "
                    "pretrain.build_optimizer (one group per head)")


def _scale_grads(model, denom: int) -> None:
    if denom <= 1:
        return
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(denom)


def _train_one_epoch(
    model, dl, opt, scheduler, epoch, tcfg: TrainConfig, dev, *, rank,
    max_updates: int | None = None, boundary_cb=None, step_offset: int = 0,
):
    """One pass over `dl`. `step_offset` is the number of optimizer updates applied
    before this epoch (the run's update counter): update `step_offset + updates` sets
    the loss weights of every microbatch it accumulates (KTD5)."""
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

    weights = None
    for batch_idx, batch in enumerate(dl):
        batch = _prepare_batch(batch, dev)
        if micro % tcfg.grad_accum == 0:
            # First microbatch of an update: the schedule advances once per update.
            weights = _apply_objective_step(model, opt, step_offset + updates,
                                            tcfg.total_steps)

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
                    + ("" if weights is None else
                       "w=" + "/".join(f"{w:.3g}" for w in weights.values()) + " ")
                    + f"lr={scheduler.get_last_lr()[0]:.3e} "
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
                # A true return is a clean stop (`StopRequest`): checkpointed, leave now.
                epoch_done = updates >= -(-len(dl) // tcfg.grad_accum)
                if boundary_cb(updates, epoch_done):
                    return ml, samples_seen, int(tokens_seen), int(ntp_tokens), updates
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


class StopRequest:
    """A clean stop asked for by SIGTERM / SIGINT (a scheduler's pre-emption, Ctrl-C).

    The first signal only sets a flag; the engine reads it at the next optimizer update,
    agrees on it across ranks (one all-reduce per update, so every rank stops at the same
    update), saves a checkpoint and returns. A second signal raises KeyboardInterrupt at
    once. Handlers are installed only from the main thread and restored on exit."""

    SIGNALS = (signal.SIGTERM, signal.SIGINT)

    def __init__(self):
        self.requested = False
        self.signum: int | None = None
        self._previous: dict = {}

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt(f"second signal {signum}: stopping without a checkpoint")
        self.requested, self.signum = True, signum
        print(f"signal {signum}: stopping cleanly after the current optimizer update "
              "(checkpoint first); send it again to abort", flush=True)

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for sig in self.SIGNALS:
                self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()
        return False

    def agreed(self, dev) -> bool:
        """Whether any rank was asked to stop (collective under DDP: every rank calls it
        at the same update)."""
        if not is_distributed():
            return self.requested
        flag = torch.tensor(int(self.requested), dtype=torch.int32, device=dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())


def _check_resume_config(saved: dict, current: dict, keys) -> None:
    """Refuse a resume whose checkpoint was trained under a different value of any of
    `keys` (manifest config; a key the checkpoint does not record is not compared)."""
    mismatched = [key for key in keys if key in saved and saved[key] != current.get(key)]
    if mismatched:
        detail = "; ".join(f"{key}: checkpoint {saved[key]!r} != this run {current.get(key)!r}"
                           for key in mismatched)
        raise ValueError(f"resume checkpoint was trained under a different run: {detail}")


def train(model, train_dl, val_dl, opt, scheduler, tcfg: TrainConfig, dev, *,
          vocab_binding, resume_ckpt=None, seed=42, fresh_schedule=False,
          resume_match: tuple[str, ...] = ()):
    """Resumable (DDP) training loop. `vocab_binding` (`segments.artifact_binding` of the
    training vocab.json; required) is recorded in every checkpoint, and a resume
    checkpoint bound to a different (or no) vocabulary/segments is refused before any
    state loads; so is one whose manifest config differs on any `resume_match` key.

    Checkpoints: every `tcfg.ckpt_every` updates, a final one when the run reaches its
    update limit (unless `runtime.final_checkpoint` is false) and one on a clean stop
    (SIGTERM / SIGINT), each skipped when that update was just saved. Only rank 0 writes; every rank takes part in the RNG gather and waits at
    a barrier after the final save."""
    if vocab_binding is None:
        raise ValueError("train() requires vocab_binding (segments.artifact_binding of the "
                         "training vocab.json): every checkpoint is bound to it")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    note = precision_note(dev)
    if is_main and note:
        print(note, flush=True)
    _check_head_decay_groups(model, opt)
    objective = _objective_model(model)
    config = dict(tcfg.__dict__)
    if objective is not None:
        config["objective"] = objective.objective_record()
        if is_main:
            print(objective.describe_schedule(tcfg.total_steps), flush=True)

    manifest = Manifest(
        model_name="clifatron2",
        config=config,
        seed=seed,
        ckpt_dir=str(tcfg.ckpt_dir),
    )
    manifest.record_env()
    manifest.record_parameters(model)

    start_epoch = 0
    start_step = 0
    if resume_ckpt is not None:
        manifest.lineage_parent = str(resume_ckpt)
        loaded = load_checkpoint(resume_ckpt, dev)
        compare_binding(loaded.get("vocab_binding"), vocab_binding, what="resume checkpoint")
        saved_manifest = loaded.get("manifest") or {}
        saved_config = dict(saved_manifest.get("config") or {})
        if "seed" in saved_manifest:
            saved_config["seed"] = saved_manifest["seed"]
        _check_resume_config(saved_config, {**config, "seed": seed}, resume_match)
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
        manifest.lineage_parent = saved_manifest.get("run_id", str(resume_ckpt))
        if is_main:
            print(f"resumed from {resume_ckpt}: update {start_step} of {tcfg.total_steps}, "
                  f"epoch {start_epoch}", flush=True)

    tcfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    previous_ledger = loaded.get("manifest", {}).get("ledger", {}) if resume_ckpt is not None else {}
    total_sm = int(previous_ledger.get("samples_seen", 0))
    total_tok = int(previous_ledger.get("tokens_seen", 0))
    total_ntp = int(previous_ledger.get("ntp_eligible_tokens", 0))
    global_step = start_step
    next_val_step = ((global_step // tcfg.val_every) + 1) * tcfg.val_every
    next_ckpt_step = ((global_step // tcfg.ckpt_every) + 1) * tcfg.ckpt_every
    # The resume checkpoint already holds the state at start_step.
    last_saved_step = start_step if resume_ckpt is not None else None
    epoch = start_epoch
    stop = StopRequest()
    stopped = False

    def _save(epoch_n: int, gs: int) -> None:
        """Write the checkpoint of update `gs`. Collective under DDP (the RNG gather):
        every rank calls it at the same update; rank 0 writes. `epoch_n` is EPOCHS FULLY
        CONSUMED (end-of-epoch saves get epoch+1, so resume never replays a completed
        epoch; mid-epoch saves replay the partial epoch). The ledger totals lag within
        an epoch (running totals land at epoch end)."""
        nonlocal last_saved_step
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
        last_saved_step = gs

    def _maybe_checkpoint(epoch_n: int, gs: int) -> None:
        """Save at step-granular boundaries. DDP-safe: fires on every rank at the
        same update (synchronized update counts + _all_gather)."""
        nonlocal next_ckpt_step
        if gs < next_ckpt_step:
            return
        _save(epoch_n, gs)
        while next_ckpt_step <= gs:
            next_ckpt_step += tcfg.ckpt_every

    def _stop_now(epoch_n: int, gs: int) -> bool:
        """Clean stop: every rank agrees, the update is checkpointed, training ends."""
        nonlocal stopped
        if not stop.agreed(dev):
            return False
        if last_saved_step != gs:
            _save(epoch_n, gs)
        stopped = True
        if is_main:
            print(f"stopped cleanly at update {gs} of {tcfg.total_steps} "
                  f"(checkpoint ckpt_ep{epoch_n}_step{gs}.pt); resume with --resume",
                  flush=True)
        return True

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

    # Epochs (passes) fully consumed at the run's last update: the final checkpoint's
    # epoch, so a resume of a stopped run neither replays nor skips a pass.
    epochs_done = epoch
    updates_per_epoch = -(-len(train_dl) // tcfg.grad_accum) if hasattr(train_dl, "__len__") else None
    with stop:
        while global_step < tcfg.total_steps and not stopped:
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
                epoch_n = _epoch + 1 if epoch_done else _epoch
                _maybe_clear_cache(gs)
                _maybe_validate(_epoch, gs)
                _maybe_checkpoint(epoch_n, gs)
                return _stop_now(epoch_n, gs)

            ml, sm, tok, ntp, updates = _train_one_epoch(
                model, train_dl, opt, scheduler, epoch, tcfg, dev, rank=local_rank,
                max_updates=tcfg.total_steps - global_step, boundary_cb=_boundary,
                step_offset=global_step,
            )
            total_sm += sm
            total_tok += tok
            total_ntp += ntp
            complete = updates_per_epoch is None or updates >= updates_per_epoch
            epochs_done = epoch + 1 if complete else epoch
            epoch += 1
            global_step += updates
            if stopped:
                break

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

            # End of a pass: the partial accumulation's update never reached the
            # boundary callback, so its checkpoint and stop check happen here.
            _maybe_checkpoint(epochs_done, global_step)
            if _stop_now(epochs_done, global_step):
                break

    # Final checkpoint: the run's last update is always saved (rank 0 writes; every rank
    # joins the gather), unless the periodic or stop save just wrote it.
    if getattr(tcfg, "final_checkpoint", True) and last_saved_step != global_step:
        _save(epochs_done, global_step)
        if is_main:
            print(f"final checkpoint: {tcfg.ckpt_dir / f'ckpt_ep{epochs_done}_step{global_step}.pt'}",
                  flush=True)
    if is_distributed():
        dist.barrier()
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
