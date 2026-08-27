"""Self-supervised pretraining on one site's event shards (2x L40, DDP).

Loss = w_A*next_event + w_B*competing_risk + w_C*threshold_hazard + w_D*value_regression
(ORA marked-TTE; RESEARCH.md §3). next_event uses tied embeddings (enc.lm_logits).
The threshold-hazard batch samples a random target concept + random τ (from that
concept's decile edges) + direction each step (ICareFM), and derives crossed_bin
from the future event stream inside the 48h horizon.

Run:
    torchrun --nproc_per_node=2 -m src.train.pretrain --config configs/train.yaml \
        --model-config configs/model.yaml --data data/mimic --site mimic
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

from src.model.encoder import CLIFEncoder, count_params
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


class Model(torch.nn.Module):
    def __init__(self, vocab_size, n_targets, mcfg):
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
        H = self.enc(batch["token"], batch["pos_min"])           # fused token + RoPE
        h_last = H[torch.arange(H.size(0)), batch["last_idx"]]
        la = next_event_loss(self.enc.lm_logits(H), batch["token"])   # tied embeddings
        lb = self.cr.loss(h_last, batch["cr_type"], batch["cr_bin"])
        lc = self.th.loss(
            h_last, batch["th_target"], batch["th_tau"], batch["th_dir"], batch["th_crossed"]
        )
        ld = self.vr.loss(H, batch["token"], batch["value"], batch["val_mask"]) if self.vr else None
        return la, lb, lc, ld


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--site", required=True)
    args = ap.parse_args()

    local, is_main = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    tcfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = yaml.safe_load(Path(args.model_config).read_text())
    vocab = json.loads((Path(args.data) / "vocab.json").read_text())
    n_targets = len(yaml.safe_load(Path("configs/data.yaml").read_text())["target_concepts"])

    model = Model(len(vocab["vocab"]), n_targets, mcfg).to(dev)
    if is_main:
        print(f"params: {count_params(model)/1e6:.1f}M")
    if mcfg.get("compile"):
        model = torch.compile(model)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local])

    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["optimizer"]["lr"],
        weight_decay=tcfg["optimizer"]["weight_decay"], betas=tcfg["optimizer"]["betas"],
    )
    wA = mcfg["heads"]["next_event"]["weight"]
    wB = mcfg["heads"]["competing_risk"]["weight"]
    wC = mcfg["heads"]["threshold_hazard"]["weight"]
    wD = mcfg["heads"]["value_regression"]["weight"]

    # TODO: replace with a real DataLoader over Path(args.data)/events.parquet.
    # Collate: pad to max_tokens (4096), build last_idx (anchor position), and for
    #   threshold_hazard sample (target, tau_bin, direction) + compute crossed_bin from
    #   the future stream. val_mask marks events that carry a numeric value (ORA target).
    # See src/data/collate.py (to be written by the coding agent).
    raise SystemExit(
        "Wire up the DataLoader/collate next — model, heads, DDP, and losses are ready.\n"
        "Batch keys required: token, pos_min, value, val_mask, last_idx,\n"
        "  cr_type, cr_bin, th_target, th_tau, th_dir, th_crossed."
    )


if __name__ == "__main__":
    main()
