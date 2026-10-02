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
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model.encoder import count_params
from src.data.segments import artifact_binding, n_value_bins
from src.model.head_adapter import CLIFATRONHeads, load_backbone
from src.train.engine import TrainConfig, train
from src.train.pretrain import Model, build_loaders, build_scheduler


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return local, dist.get_rank() == 0
    return 0, True


# -------------------------------------------------------------------- models
class FromScratchModel(Model):
    """From-scratch arm: exactly `pretrain.Model` — config-driven `heads.*.weight`, the
    CR/threshold losses masked to documents that carry labels/queries, and fail-to-zero
    guards for absent head targets — so engine batches (where unlabeled documents carry
    placeholder targets) never train on false censoring/no-crossing labels."""


class AdapterModel(torch.nn.Module):
    def __init__(self, backbone, n_targets: int, freeze_trunk: bool, *, n_value_bins: int):
        super().__init__()
        self.adapter = CLIFATRONHeads(
            backbone, n_targets,
            freeze_backbone=freeze_trunk,
            cr_bins=16, th_bins=48,
            n_value_bins=n_value_bins, enable_value=True,
            tie_weights=False,
        )

    def forward(self, batch):
        losses = self.adapter.loss(batch, w_ntp=0.2, w_cr=1.0, w_th=1.0, w_val=0.5)
        return losses


# -------------------------------------------------------------------- driver
def load_n_value_bins(path: str | Path) -> int:
    """Threshold-head value-bin count from a tokenizer-v2 vocab.json (never a default)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} is required: n_value_bins is derived from the frozen "
                         "vocabulary's segments")
    return n_value_bins(json.loads(path.read_text()))


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
    args = ap.parse_args()

    local, is_main = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    abl = yaml.safe_load(Path(args.ablation_config).read_text())
    mcfg = yaml.safe_load(Path(args.model_config).read_text())
    tcfg = yaml.safe_load(Path(args.train_config).read_text())

    arm_cfg = abl["arms"].get(args.arm)
    if arm_cfg is None:
        raise SystemExit(f"Unknown arm: {args.arm}. Choices: {list(abl['arms'])}")

    n_targets = len(
        yaml.safe_load(Path("configs/data.yaml").read_text())["target_concepts"]
    )
    vocab_size = mcfg["trunk"].get("target_vocab", 10000)
    total_steps = arm_cfg["total_steps"]
    vocab_path = Path(args.vocab or Path(args.data) / "vocab.json")
    value_bins = load_n_value_bins(vocab_path)

    # --------------- build model
    if arm_cfg["trunk"] in ("clif_encoder",):
        model = FromScratchModel(vocab_size, n_targets, mcfg,
                                 n_value_bins=value_bins).to(dev)
        if is_main:
            print(f"[{args.arm}] CLIFEncoder from scratch, {count_params(model)/1e6:.1f}M params")
    else:
        if not args.checkpoint:
            raise SystemExit(f"--checkpoint required for arm {args.arm}")
        backbone = load_backbone(args.checkpoint)
        freeze = arm_cfg.get("freeze_trunk", True)
        model = AdapterModel(backbone, n_targets, freeze, n_value_bins=value_bins).to(dev)
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
    binding = artifact_binding(json.loads(vocab_path.read_text()))
    loaders = build_loaders(
        Path(args.events or Path(args.data) / "events.parquet"),
        binding=binding,
        tcfg=tcfg,
        mcfg=mcfg,
        vocab_size=vocab_size,
        value_stats_path=args.value_stats,
        is_main=is_main,
    )

    if mcfg.get("compile"):
        model = torch.compile(model, dynamic=True)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local])

    # --------------- optimizer
    lr_value = arm_cfg["lr"]
    if isinstance(lr_value, list):
        trunk_lr, head_lr = (float(v) for v in lr_value)
    else:
        trunk_lr = head_lr = float(lr_value)

    param_groups = []
    if arm_cfg.get("freeze_trunk", True):
        param_groups.append({"params": [p for p in model.parameters() if p.requires_grad],
                             "lr": head_lr})
    else:
        param_groups.append({"params": [p for n, p in model.named_parameters()
                                        if "enc." in n and p.requires_grad],
                             "lr": trunk_lr})
        param_groups.append({"params": [p for n, p in model.named_parameters()
                                        if "enc." not in n and p.requires_grad],
                             "lr": head_lr})

    opt = torch.optim.AdamW(
        param_groups,
        weight_decay=tcfg["optimizer"]["weight_decay"],
        betas=tcfg["optimizer"]["betas"],
    )
    warmup = min(int(tcfg["schedule"].get("warmup_steps", 2000)), total_steps)
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
            f"  Steps: {total_steps:,} | Curriculum: {arm_cfg.get('curriculum', 'none')} "
            "(not applied by this driver)\n"
            f"  Output: {out_dir}\n"
            f"{'='*60}"
        )
    _, manifest = train(
        model, loaders.train, loaders.validation, opt, scheduler,
        TrainConfig({}, tcfg, mcfg, total_steps), dev,
        seed=42, vocab_binding=binding,
    )
    if is_main:
        (out_dir / "run.json").write_text(json.dumps({
            "arm": args.arm, "run_id": manifest.run_id, "ledger": manifest.ledger,
            "validation": manifest.validation,
        }, indent=2))
        print(f"[{args.arm}] training complete. Run ID: {manifest.run_id}")
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()