"""Phase-2 joint pretraining: bolt our heads onto a trained CLIFATRON checkpoint.

Uses head_adapter.CLIFATRONHeads with the NTP→TTE curriculum (curriculum.py).
Produces zero-shot threshold/CR predictions — the mechanism that makes external
CLIF-federation validation possible without local labels.

Run on the 2× L40 box:
    torchrun --nproc_per_node=2 -m src.train.joint_pretrain \
        --checkpoint /path/to/clifatron_checkpoint \
        --data /path/to/tokenized_narratives \
        --config configs/train.yaml --model-config configs/model.yaml

The narrative parquet must contain input_ids (list[int], tokenized + packed),
attention_mask, and per-sample labels: in_hospital_mortality, aki_kdigo_48h, etc.
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

from src.model.head_adapter import CLIFATRONHeads, load_backbone
from src.train.curriculum import curriculum_weights


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return local, dist.get_rank() == 0
    return 0, True


class JointModel(torch.nn.Module):
    def __init__(self, backbone, n_targets: int, freeze_backbone: bool = False):
        super().__init__()
        self.adapter = CLIFATRONHeads(
            backbone,
            n_targets,
            freeze_backbone=freeze_backbone,
            cr_bins=16,
            th_bins=48,
            n_value_bins=10,
            enable_value=True,
            tie_weights=False,
        )

    def forward(self, batch, step: int, total_steps: int):
        mix = curriculum_weights(step, total_steps)
        if not mix.train_heads and self.adapter.frozen is False:
            self.adapter.frozen = True
            for p in self.adapter.backbone.parameters():
                p.requires_grad = True
            for p in self.adapter.cr.parameters():
                p.requires_grad = False
            for p in self.adapter.th.parameters():
                p.requires_grad = False
            if self.adapter.vr is not None:
                for p in self.adapter.vr.parameters():
                    p.requires_grad = False
        if mix.train_heads and self.adapter.frozen is True:
            self.adapter.frozen = False
            for p in self.adapter.backbone.parameters():
                p.requires_grad = True
            for p in self.adapter.cr.parameters():
                p.requires_grad = True
            for p in self.adapter.th.parameters():
                p.requires_grad = True
            if self.adapter.vr is not None:
                for p in self.adapter.vr.parameters():
                    p.requires_grad = True

        loss_dict = self.adapter.loss(
            batch,
            w_ntp=mix.w_ntp,
            w_cr=mix.w_cr,
            w_th=mix.w_th,
            w_val=mix.w_val,
        )
        return loss_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="trained CLIFATRON checkpoint (HF causal LM)")
    ap.add_argument("--data", required=True,
                    help="tokenized narratives parquet directory")
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--grad-accum", type=int, default=1)
    args = ap.parse_args()

    local, is_main = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    tcfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = yaml.safe_load(Path(args.model_config).read_text())
    n_targets = len(
        yaml.safe_load(Path("configs/data.yaml").read_text())["target_concepts"]
    )
    total_steps = tcfg["schedule"]["total_steps"]

    if is_main:
        print(f"Loading backbone from {args.checkpoint} ...")
    backbone = load_backbone(args.checkpoint)
    model = JointModel(backbone, n_targets, freeze_backbone=False).to(dev)

    if is_main:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    if mcfg.get("compile"):
        model = torch.compile(model, dynamic=True)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local])

    opt = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters()
                           if "adapter.backbone" in n and p.requires_grad],
                "lr": tcfg["optimizer"]["lr"],
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if "adapter.backbone" not in n and p.requires_grad],
                "lr": tcfg["optimizer"]["lr"] * 3,
            },
        ],
        weight_decay=tcfg["optimizer"]["weight_decay"],
        betas=tcfg["optimizer"]["betas"],
    )

    print(
        f"{'='*60}\n"
        f"Ready for joint pretraining on {dev}.\n"
        f"  Total steps: {total_steps:,}\n"
        f"  Warmup:       {int(total_steps*0.15):,} steps (NTP only)\n"
        f"  Transition:   {int(total_steps*0.05):,} steps (linear blend)\n"
        f"  Mixed:        {int(total_steps*0.80):,} steps (full ORA + 0.2 NTP)\n"
        f"{'='*60}\n"
        "TODO: wire up the DataLoader.\n"
        "Batch keys required: input_ids, attention_mask,\n"
        "  value (per-event continuous values), val_mask,\n"
        "  cr_type, cr_bin, th_target, th_tau, th_dir, th_crossed.\n"
        "The collate must handle packed sequences from CLIFATRON's tokenETL\n"
        "narrative assembly (same format as external/clifatron/AR/**/pack_sequences.py)."
    )


if __name__ == "__main__":
    main()