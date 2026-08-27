"""Tokenization ablation driver — runs all 5 arms from configs/tokenization_ablation.yaml.

Each arm uses the same frozen encoder architecture; only tokenization varies.
Outputs per-arm metrics for comparison reporting via src/eval/ablation_compare.py.

Usage:
    torchrun --nproc_per_node=2 -m src.train.run_tokenization_ablation \
        --arm deciles_plus_soft --data /path/to/events.parquet

Or run all arms sequentially:
    for arm in clifatron_clinical_bins global_deciles deciles_plus_soft \
               continuous_fused textcode; do
        torchrun --nproc_per_node=2 -m src.train.run_tokenization_ablation \
            --arm $arm --data /path/to/events.parquet
    done
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from src.model.encoder import CLIFEncoder, count_params
from src.model.encoder_continuous import ContinuousFusedEncoder
from src.model.heads import (
    CompetingRiskHead,
    ThresholdHazardHead,
    ValueRegressionHead,
    next_event_loss,
)


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return local, dist.get_rank() == 0
    return 0, True


class TokenizationAblationModel(torch.nn.Module):
    def __init__(self, vocab_size: int, n_targets: int, mcfg: dict,
                 arm_cfg: dict):
        super().__init__()
        tokenizer_type = arm_cfg.get("tokenizer", "decile")
        d = mcfg["trunk"]["d_model"]

        if tokenizer_type == "continuous_fused":
            self.enc = ContinuousFusedEncoder(vocab_size, mcfg)
            self.use_continuous_fused = True
        else:
            self.enc = CLIFEncoder(vocab_size, mcfg)
            self.use_continuous_fused = False

        h = mcfg["heads"]
        self.cr = CompetingRiskHead(d, n_targets, h["competing_risk"]["n_time_bins"])
        self.th = ThresholdHazardHead(
            d, n_targets, h["threshold_hazard"]["n_time_bins"],
            n_value_bins=10, thr_dim=h["threshold_hazard"]["threshold_embed_dim"],
        )
        self.vr = ValueRegressionHead(d, vocab_size) if h["value_regression"]["enabled"] else None

    def forward(self, batch):
        if self.use_continuous_fused:
            H = self.enc(
                batch["token"],
                batch["pos_min"],
                continuous_value=batch["value"],
            )
        else:
            enc_token = batch.get("soft_token", batch["token"])
            H = self.enc(enc_token, batch["pos_min"], batch.get("soft_weight"))

        h_last = H[torch.arange(H.size(0)), batch["last_idx"]]
        la = next_event_loss(self.enc.lm_logits(H), batch["token"])
        lb = self.cr.loss(h_last, batch["cr_type"], batch["cr_bin"])
        lc = self.th.loss(h_last, batch["th_target"], batch["th_tau"],
                          batch["th_dir"], batch["th_crossed"])
        ld = self.vr.loss(H, batch["token"], batch["value"], batch["val_mask"]) if self.vr else None

        total = 0.2 * la + 1.0 * lb + 1.0 * lc + (0.5 * ld if ld is not None else 0)
        return {"ntp": la, "cr": lb, "th": lc, "val": ld, "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ablation-config", default="configs/tokenization_ablation.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--out", default="results/tokenization_ablation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    local, is_main = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    abl = yaml.safe_load(Path(args.ablation_config).read_text())
    mcfg = yaml.safe_load(Path(args.model_config).read_text())

    arm_cfg = abl["arms"].get(args.arm)
    if arm_cfg is None:
        raise SystemExit(f"Unknown arm: {args.arm}. Choices: {list(abl['arms'])}")

    n_targets = len(
        yaml.safe_load(Path("configs/data.yaml").read_text())["target_concepts"]
    )
    vocab_size = mcfg["trunk"].get("target_vocab", 10000)

    if args.dry_run:
        model = TokenizationAblationModel(vocab_size, n_targets, mcfg, arm_cfg)
        total = count_params(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{args.arm}] {total/1e6:.1f}M total / {trainable/1e6:.1f}M trainable")
        print(f"  tokenizer: {arm_cfg['tokenizer']}")
        print(f"  soft_discretization: {arm_cfg.get('soft_discretization')}")
        return

    total_steps = arm_cfg["total_steps"]
    model = TokenizationAblationModel(vocab_size, n_targets, mcfg, arm_cfg).to(dev)

    if is_main:
        total = count_params(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{args.arm}] {total/1e6:.1f}M params, {trainable/1e6:.1f}M trainable")
        print(f"  tokenizer: {arm_cfg['tokenizer']}")
        print(f"  description: {arm_cfg['description']}")

    if mcfg.get("compile"):
        model = torch.compile(model, dynamic=True)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local])

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=arm_cfg["lr"],
        weight_decay=0.1,
    )

    out_dir = Path(args.out) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"{'='*60}\n"
        f"Arm: {args.arm} ({arm_cfg['tokenizer']})\n"
        f"  Steps: {total_steps:,} | LR: {arm_cfg['lr']}\n"
        f"  Output: {out_dir}\n"
        f"{'='*60}\n"
        "TODO: wire up the DataLoader. Same batch contract as run_arm.py.\n"
        "Continuous-fused: batch['token']=[B,T] concept IDs, batch['value']=[B,T] raw values.\n"
        "Discrete/soft: batch['soft_token']=[B,T,K], batch['soft_weight']=[B,T,K].\n"
        "TextCode: batch['token']=[B,T] concept indices -> cached BERT embeddings."
    )


if __name__ == "__main__":
    main()