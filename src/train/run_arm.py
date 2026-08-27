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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model.encoder import CLIFEncoder, count_params
from src.model.heads import (
    CompetingRiskHead,
    NextEventHead,
    ThresholdHazardHead,
    ValueRegressionHead,
    TaskHead,
    next_event_loss,
)
from src.model.head_adapter import CLIFATRONHeads, load_backbone
from src.train.curriculum import curriculum_weights


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
        self.loss_ntp.append(float(losses.get("ntp", 0)))
        self.loss_cr.append(float(losses.get("cr", 0)))
        self.loss_th.append(float(losses.get("th", 0)))
        self.loss_val.append(float(losses.get("val", 0)))
        self.loss_total.append(float(losses.get("total", 0)))
        self.lr.append(lr_val)


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return local, dist.get_rank() == 0
    return 0, True


# -------------------------------------------------------------------- models
class FromScratchModel(torch.nn.Module):
    def __init__(self, vocab_size: int, n_targets: int, mcfg: dict):
        super().__init__()
        self.enc = CLIFEncoder(vocab_size, mcfg)
        d = self.enc.d_model
        h = mcfg["heads"]
        self.cr = CompetingRiskHead(d, n_targets, h["competing_risk"]["n_time_bins"])
        self.th = ThresholdHazardHead(
            d, n_targets, h["threshold_hazard"]["n_time_bins"],
            n_value_bins=10, thr_dim=h["threshold_hazard"]["threshold_embed_dim"],
        )
        self.vr = ValueRegressionHead(d, vocab_size) if h["value_regression"]["enabled"] else None

    def forward(self, batch):
        enc_token = batch.get("soft_token", batch["token"])
        H = self.enc(enc_token, batch["pos_min"], batch.get("soft_weight"))
        h_last = H[torch.arange(H.size(0)), batch["last_idx"]]
        ntp = next_event_loss(self.enc.lm_logits(H), batch["token"])
        cr = self.cr.loss(h_last, batch["cr_type"], batch["cr_bin"])
        th = self.th.loss(h_last, batch["th_target"], batch["th_tau"],
                          batch["th_dir"], batch["th_crossed"])
        val = self.vr.loss(H, batch["token"], batch["value"], batch["val_mask"]) if self.vr else None
        return {"ntp": ntp, "cr": cr, "th": th, "val": val, "total": None}


class AdapterModel(torch.nn.Module):
    def __init__(self, backbone, n_targets: int, freeze_trunk: bool):
        super().__init__()
        self.adapter = CLIFATRONHeads(
            backbone, n_targets,
            freeze_backbone=freeze_trunk,
            cr_bins=16, th_bins=48,
            n_value_bins=10, enable_value=True,
            tie_weights=False,
        )

    def forward(self, batch):
        losses = self.adapter.loss(batch, w_ntp=0.2, w_cr=1.0, w_th=1.0, w_val=0.5)
        return losses


# -------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data", required=True)
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

    # --------------- build model
    if arm_cfg["trunk"] in ("clif_encoder",):
        model = FromScratchModel(vocab_size, n_targets, mcfg).to(dev)
        if is_main:
            print(f"[{args.arm}] CLIFEncoder from scratch, {count_params(model)/1e6:.1f}M params")
    else:
        if not args.checkpoint:
            raise SystemExit(f"--checkpoint required for arm {args.arm}")
        backbone = load_backbone(args.checkpoint)
        freeze = arm_cfg.get("freeze_trunk", True)
        model = AdapterModel(backbone, n_targets, freeze).to(dev)
        if is_main:
            total_p = sum(p.numel() for p in model.parameters())
            trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[{args.arm}] adapter on CLIFATRON backbone, {total_p/1e6:.1f}M total / {trainable_p/1e6:.1f}M trainable")

    if mcfg.get("compile"):
        model = torch.compile(model, dynamic=True)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local])

    # --------------- optimizer
    lr_value = arm_cfg["lr"]
    if isinstance(lr_value, list):
        trunk_lr, head_lr = lr_value
    else:
        trunk_lr = head_lr = lr_value

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

    use_curriculum = arm_cfg.get("curriculum") != "none"
    metrics_log = MetricsLog()

    out_dir = Path(args.out) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: replace with real DataLoader. Batch must contain:
    #   token/soft_token, pos_min, soft_weight, value, val_mask, last_idx,
    #   cr_type, cr_bin, th_target, th_tau, th_dir, th_crossed.
    print(
        f"{'='*60}\n"
        f"Arm: {args.arm}\n"
        f"  Description: {arm_cfg['description']}\n"
        f"  Trunk: {arm_cfg['trunk']}, freeze: {arm_cfg.get('freeze_trunk')}\n"
        f"  LR: trunk={trunk_lr}, head={head_lr}\n"
        f"  Steps: {total_steps:,} | Curriculum: {use_curriculum}\n"
        f"  Output: {out_dir}\n"
        f"{'='*60}\n"
        "TODO: wire DataLoader.\n"
        "Expected batch keys: token, soft_token, soft_weight, pos_min,\n"
        "  value, val_mask, last_idx, cr_type, cr_bin, th_target,\n"
        "  th_tau, th_dir, th_crossed.\n"
        "The run loop will use curriculum_weights(step, total_steps) to\n"
        "adjust loss mixing and head trainability per the configured curriculum."
    )


if __name__ == "__main__":
    main()