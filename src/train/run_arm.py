"""Run a single ablation arm — unified driver for the finetune-vs-scratch comparison.

Evidence matrix: see configs/ablation.yaml

  frozen_backbone_head_only  Al Attrach 2025, Mataraso 2025    ↑↑
  joint_finetune             Al Attrach 2025 (unfreeze hurts)
  from_scratch               TOO-BERT PMC12177421
  no_pretrain_baseline       negative control

Run one arm:
    torchrun --nproc_per_node=2 -m src.train.run_arm \
        --arm frozen_backbone_head_only \
        --checkpoint /path/to/clifatron_checkpoint \
        --data /path/to/tokenized_narratives

Arms that use clif_encoder (from_scratch, no_pretrain) don't need --checkpoint.
Data loads through `pretrain.build_loaders` (`--events`, default <data>/events.parquet,
bound to `--vocab`; `--value-stats` for value-head normalization) and trains with
`engine.train`; `--dry-run` prints the arm and exits before loading data.

The arm's `curriculum` (ntp_then_tte | none) is applied by the engine per optimizer update
(KTD5); a frozen-trunk arm cannot run it (its warm-up would train nothing). Launch and
devices go through `engine.setup_ddp` / `select_device` / `wrap_ddp` like pretrain; a
torchrun launch without CUDA needs `--allow-cpu-ddp`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from src.model.encoder import count_params
from src.data.segments import artifact_binding, load_vocab_blob, n_value_bins
from src.model.head_adapter import CLIFATRONHeads, load_backbone
from src.train.engine import (
    ALLOW_CPU_DDP_FLAG,
    TrainConfig,
    is_distributed,
    resolve_schedule,
    select_device,
    setup_ddp,
    train,
    wrap_ddp,
)
from src.train.pretrain import (
    Model,
    ObjectiveSchedule,
    build_loaders,
    build_optimizer,
    build_scheduler,
    embedding_vocab_size,
    objective_weights,
)


# -------------------------------------------------------------------- models
class FromScratchModel(Model):
    """From-scratch arm: exactly `pretrain.Model` — config-driven `heads.*.weight`, the
    CR/threshold losses masked to documents that carry labels/queries, and fail-to-zero
    guards for absent head targets — so engine batches (where unlabeled documents carry
    placeholder targets) never train on false censoring/no-crossing labels."""


class AdapterModel(ObjectiveSchedule, torch.nn.Module):
    """CLIFATRON backbone + our heads, weighted like `pretrain.Model`: `heads.*.weight`
    (and grids) from the model config, the curriculum per optimizer update. A skipped
    head touches its parameters (`head_adapter.touch`), and a frozen probe freezes the
    next-event head it never trains, so DDP sees the same gradient set on every step."""

    def __init__(self, backbone, n_targets: int, freeze_trunk: bool, mcfg: dict, *,
                 n_value_bins: int):
        super().__init__()
        h = mcfg["heads"]
        self.adapter = CLIFATRONHeads(
            backbone, n_targets,
            freeze_backbone=freeze_trunk,
            cr_bins=h["competing_risk"]["n_time_bins"],
            th_bins=h["threshold_hazard"]["n_time_bins"],
            cr_horizon_hours=h["competing_risk"].get("horizon_hours", 48),
            th_horizon_hours=h["threshold_hazard"].get("horizon_hours", 48),
            n_value_bins=n_value_bins, enable_value=h["value_regression"]["enabled"],
            tie_weights=False,
        )
        weights = objective_weights(mcfg)
        if freeze_trunk:
            weights["next_event"] = 0.0   # a frozen probe has no next-token objective
        self._init_objective(weights, mcfg)
        if freeze_trunk and self.curriculum:
            raise ValueError("a frozen-trunk arm cannot run the NTP->TTE curriculum: its "
                             "next-token warm-up would train nothing; set curriculum: none")

    def _head_module(self, head: str):
        return getattr(self.adapter, self.HEAD_ATTRS[head], None)

    def forward(self, batch):
        weights = {head: self._weight(head) if self._active[head] else 0.0
                   for head in self._active}
        return self.adapter.loss(
            batch, w_ntp=weights["next_event"], w_cr=weights["competing_risk"],
            w_th=weights["threshold_hazard"], w_val=weights["value_regression"])


# -------------------------------------------------------------------- driver
def build_from_scratch(vocab_blob: dict, mcfg: dict, *, n_targets: int) -> FromScratchModel:
    """The from-scratch arm's model: embedding rows = the vocabulary's max id + 1
    (`embedding_vocab_size`, as `pretrain` sizes it), so its checkpoints and pretrain's
    load into each other and into every checkpoint consumer."""
    return FromScratchModel(embedding_vocab_size(vocab_blob, mcfg), n_targets, mcfg,
                            n_value_bins=n_value_bins(vocab_blob))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", default=None,
                    help="tokenizer-v2 vocab.json (default: <data>/vocab.json)")
    ap.add_argument("--events", default=None,
                    help="events shard (default: <data>/events.parquet; a sibling "
                         "events_with_outcomes.parquet is used when it carries the labels)")
    ap.add_argument("--value-stats", default=None,
                    help="value_stats.json bound to the vocabulary (value-head normalization)")
    ap.add_argument("--ablation-config", default="configs/ablation.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--dry-run", action="store_true",
                    help="print model params + arm config, then exit")
    ap.add_argument(ALLOW_CPU_DDP_FLAG, action="store_true",
                    help="let a distributed (torchrun) launch run on CPU over gloo when "
                         "CUDA is unavailable; without it such a launch is refused")
    args = ap.parse_args()

    local, is_main = setup_ddp(allow_cpu=args.allow_cpu_ddp)
    dev = select_device(local)

    abl = yaml.safe_load(Path(args.ablation_config).read_text())
    mcfg = yaml.safe_load(Path(args.model_config).read_text())
    tcfg = yaml.safe_load(Path(args.train_config).read_text())

    arm_cfg = abl["arms"].get(args.arm)
    if arm_cfg is None:
        raise SystemExit(f"Unknown arm: {args.arm}. Choices: {list(abl['arms'])}")
    mcfg["curriculum"] = arm_cfg.get("curriculum", "none")   # applied by the engine

    n_targets = len(
        yaml.safe_load(Path("configs/data.yaml").read_text())["target_concepts"]
    )
    total_steps = arm_cfg["total_steps"]
    # The threshold head's value-bin count comes from the vocabulary (never a default).
    vocab_blob = load_vocab_blob(
        Path(args.vocab or Path(args.data) / "vocab.json"),
        required_for="n_value_bins is derived from the frozen vocabulary's segments")
    value_bins = n_value_bins(vocab_blob)
    vocab_size = embedding_vocab_size(vocab_blob, mcfg)

    # --------------- build model
    if arm_cfg["trunk"] in ("clif_encoder",):
        model = build_from_scratch(vocab_blob, mcfg, n_targets=n_targets).to(dev)
        if is_main:
            print(f"[{args.arm}] CLIFEncoder from scratch, {count_params(model)/1e6:.1f}M params")
    else:
        if not args.checkpoint:
            raise SystemExit(f"--checkpoint required for arm {args.arm}")
        backbone = load_backbone(args.checkpoint)
        freeze = arm_cfg.get("freeze_trunk", True)
        model = AdapterModel(backbone, n_targets, freeze, mcfg,
                             n_value_bins=value_bins).to(dev)
        if is_main:
            total_p = sum(p.numel() for p in model.parameters())
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[{args.arm}] adapter on CLIFATRON backbone, {total_p/1e6:.1f}M total / {trainable_p/1e6:.1f}M trainable")

    if args.dry_run:
        if is_main:
            print(f"  description: {arm_cfg['description']}")
            print(f"  trunk: {arm_cfg['trunk']}, freeze: {arm_cfg.get('freeze_trunk')}, "
                  f"steps: {total_steps:,}, n_value_bins: {value_bins}")
        return

    # --------------- data: the shared pretrain loader path (shard rows and value stats
    # bound to the vocabulary; TargetBuilder masks; length-grouped / DDP samplers)
    binding = artifact_binding(vocab_blob)
    loaders = build_loaders(
        Path(args.events or Path(args.data) / "events.parquet"),
        binding=binding,
        vocab_blob=vocab_blob,
        tcfg=tcfg,
        mcfg=mcfg,
        vocab_size=vocab_size,
        value_stats_path=args.value_stats,
        is_main=is_main,
    )

    if mcfg.get("compile"):
        model = torch.compile(model, dynamic=True)

    if is_distributed():
        model = wrap_ddp(model, dev, local)

    # --------------- optimizer: the trunk (CLIFEncoder incl. its next-event projection, or
    # the CLIFATRON backbone) at the trunk LR; heads at the head LR, one group each so a
    # head at zero weight does not decay.
    lr_value = arm_cfg["lr"]
    if isinstance(lr_value, list):
        trunk_lr, head_lr = (float(v) for v in lr_value)
    else:
        trunk_lr = head_lr = float(lr_value)
    opt = build_optimizer(
        model, lr=trunk_lr, head_lr=head_lr,
        trunk_prefixes=("enc.", "adapter.backbone."),
        weight_decay=tcfg["optimizer"]["weight_decay"], betas=tcfg["optimizer"]["betas"],
    )
    # warmup_steps, or floor(warmup_frac x updates) when it is null (configs/train.yaml).
    warmup = min(resolve_schedule(tcfg, total_steps, validate=False).warmup_steps, total_steps)
    scheduler = build_scheduler(opt, total_steps, warmup)

    out_dir = Path(args.out) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg["runtime"]["ckpt_dir"] = str(out_dir / "checkpoints")
    if is_main:
        print(
            f"{'='*60}\n"
            f"Arm: {args.arm}\n"
            f"  Description: {arm_cfg['description']}\n"
            f"  Trunk: {arm_cfg['trunk']}, freeze: {arm_cfg.get('freeze_trunk')}\n"
            f"  LR: trunk={trunk_lr}, head={head_lr}\n"
            f"  Steps: {total_steps:,} | Curriculum: {mcfg['curriculum']} "
            "(per optimizer update)\n"
            f"  Output: {out_dir}\n"
            f"{'='*60}"
        )
    train_cfg = TrainConfig({}, tcfg, mcfg, total_steps)
    train_cfg.vocab_size = vocab_size   # recorded in every checkpoint manifest
    train_cfg.trunk = dict(mcfg["trunk"])
    _, manifest = train(
        model, loaders.train, loaders.validation, opt, scheduler,
        train_cfg, dev, seed=42, vocab_binding=binding,
    )
    if is_main:
        (out_dir / "run.json").write_text(json.dumps({
            "arm": args.arm, "run_id": manifest.run_id, "ledger": manifest.ledger,
            "validation": manifest.validation,
        }, indent=2))
        print(f"[{args.arm}] training complete. Run ID: {manifest.run_id}")
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()