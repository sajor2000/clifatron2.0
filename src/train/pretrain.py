"""Self-supervised pretraining on one site's event shards (2x L40, DDP).

Loss = w_A*next_event + w_B*competing_risk + w_C*threshold_hazard + w_D*value_regression
(ORA marked-TTE; RESEARCH.md §3). Uses the training engine for resumable DDP training.

Run:
    torchrun --nproc_per_node=2 -m src.train.pretrain --config configs/train.yaml \
        --model-config configs/model.yaml --data data/mimic --site mimic
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import polars as pl
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.model.encoder import CLIFEncoder, count_params
from src.model.heads import (
    CompetingRiskHead,
    ThresholdHazardHead,
    ValueRegressionHead,
    next_event_loss,
)
from src.data.dataset import ModelDataset
from src.data.collate import collate_model_samples
from src.data.targets import TargetBuilder
from src.train.engine import setup_ddp, is_distributed, TrainConfig, train


def _load_decile_records(path: Path, *, drop_values_without_stats: bool = False) -> list[dict]:
    """Normalize tokenizer `events.parquet` rows into ModelDataset records.

    The tokenizer produces context-only event shards. Until a cohort/outcome
    artifact is joined in, these records are valid for dry-run/NTP plumbing but
    carry no TTE outcomes.
    """
    rows = pl.read_parquet(path).to_dicts()
    records = []
    for row in rows:
        token = list(row["token"])
        pos_min = list(row["pos_min"])
        if not token:
            continue
        episode_key = str(row.get("episode_key") or row.get("hosp_id"))
        anchor_idx = int(row.get("anchor_idx", len(token) - 1))
        raw_values = list(row.get("value", [None] * len(token)))
        values = [None if drop_values_without_stats else value for value in raw_values]
        records.append({
            "episode_key": episode_key,
            "artifact_hashes": dict(row.get("artifact_hashes") or {}),
            "token": token,
            "pos_min": pos_min,
            "value": values,
            "target_eligible": list(row.get("target_eligible", [True] * len(token))),
            "anchor_idx": anchor_idx,
            "anchor_min": int(row.get("anchor_min", pos_min[anchor_idx])),
            "outcomes": list(row.get("outcomes", [])),
            "soft_token": row.get("soft_token"),
            "soft_weight": row.get("soft_weight"),
        })
    if not records:
        raise ValueError("no model-ready records found in tokenized events parquet")
    return records


def _load_value_stats(path: str | None) -> dict[int, tuple[float, float]]:
    if path is None:
        return {}
    blob = json.loads(Path(path).read_text())
    return {int(k): (float(v[0]), float(v[1])) for k, v in blob.items()}


def _has_numeric_values(records: list[dict]) -> bool:
    for record in records:
        for value in record.get("value", []):
            if value is not None and math.isfinite(float(value)):
                return True
    return False


def _has_supervised_outcomes(records: list[dict]) -> bool:
    supervised = {"positive", "negative", "censored", "competing_event"}
    return any(
        any(outcome.get("status") in supervised for outcome in record.get("outcomes", []))
        for record in records
    )


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
        if "document_ids" in batch:
            for row in range(batch["document_ids"].size(0)):
                docs = batch["document_ids"][row][batch["document_ids"][row] >= 0].unique()
                if docs.numel() > 1:
                    raise RuntimeError(
                        "CLIFEncoder dense causal path cannot train multi-document packed rows without block-diagonal attention"
                    )
        encoder_token = batch.get("soft_token", batch["token"])
        H = self.enc(encoder_token, batch["pos_min"], batch.get("soft_weight"))
        if "anchor_batch_idx" in batch:
            if len(batch["anchor_batch_idx"]):
                h_last = H[batch["anchor_batch_idx"], batch["last_idx"]]
            else:
                h_last = H.new_zeros((0, H.size(-1)))
        else:
            h_last = H[torch.arange(H.size(0), device=H.device), batch["last_idx"]]
        w = {"next_event": 0.2, "competing_risk": 1.0, "threshold_hazard": 1.0, "value_regression": 0.5}
        ntp = next_event_loss(self.enc.lm_logits(H), batch.get("ntp_target", batch["token"]), batch.get("ntp_mask"))
        cr_mask = batch.get("cr_mask")
        if cr_mask is not None and not bool(cr_mask.any()):
            cr = H.new_tensor(0.0)
        else:
            cr_h = h_last[cr_mask] if cr_mask is not None else h_last
            cr_type = batch["cr_type"][cr_mask] if cr_mask is not None else batch["cr_type"]
            cr_bin = batch["cr_bin"][cr_mask] if cr_mask is not None else batch["cr_bin"]
            cr = self.cr.loss(cr_h, cr_type, cr_bin)
        th_mask = batch.get("th_mask")
        if th_mask is not None and not bool(th_mask.any()):
            th = H.new_tensor(0.0)
        else:
            th_h = h_last[th_mask] if th_mask is not None else h_last
            th = self.th.loss(
                th_h,
                batch["th_target"][th_mask] if th_mask is not None else batch["th_target"],
                batch["th_tau"][th_mask] if th_mask is not None else batch["th_tau"],
                batch["th_dir"][th_mask] if th_mask is not None else batch["th_dir"],
                batch["th_crossed"][th_mask] if th_mask is not None else batch["th_crossed"],
                batch["th_observed_bin"][th_mask] if th_mask is not None and "th_observed_bin" in batch else batch.get("th_observed_bin"),
            )
        val = self.vr.loss_aligned(
            H,
            batch.get("ntp_target", batch["token"]),
            batch["value"],
            batch["val_mask"],
        ) if self.vr is not None else H.new_tensor(0.0)
        total = w["next_event"] * ntp + w["competing_risk"] * cr + w["threshold_hazard"] * th + w["value_regression"] * val
        return {"ntp": ntp, "cr": cr, "th": th, "val": val, "total": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--resume", default=None, help="path to checkpoint to resume")
    ap.add_argument("--dry-run", action="store_true", help="print model + loader info and exit")
    ap.add_argument("--value-stats", default=None, help="JSON token_id -> [center, scale] for value-head normalization")
    args = ap.parse_args()

    local, is_main = setup_ddp()
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    tcfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = yaml.safe_load(Path(args.model_config).read_text())
    dcfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    n_targets = len(dcfg["target_concepts"])
    vocab_size = mcfg["trunk"].get("target_vocab", 10000)

    model = Model(vocab_size, n_targets, mcfg).to(dev)
    if is_main:
        print(f"params: {count_params(model)/1e6:.1f}M")

    value_stats = _load_value_stats(args.value_stats)
    data_path = Path(args.data) / "events.parquet"
    records = _load_decile_records(data_path, drop_values_without_stats=args.dry_run and not value_stats)
    if not args.dry_run and not value_stats and _has_numeric_values(records):
        raise SystemExit(
            "value-head normalization is required before real training; pass --value-stats. "
            "Generate it from the reference site: "
            "`python -m src.data.value_stats --events <ref_events.parquet> --out value_stats.json`"
        )
    if not args.dry_run and not _has_supervised_outcomes(records):
        raise SystemExit("TTE supervision is required before real pretraining; join cohort outcome artifacts first")
    target_builder = TargetBuilder(
        vocab_size=vocab_size,
        n_time_bins=mcfg["heads"]["competing_risk"]["n_time_bins"],
        horizon_hours=mcfg["heads"]["competing_risk"].get("horizon_hours", 48),
        value_stats=value_stats,
        run_seed=42,
    )
    expected_hashes = {}
    dataset = ModelDataset(
        records,
        representation="decile",
        target_builder=target_builder,
        expected_hashes=expected_hashes,
        epoch=0,
    )

    sampler = DistributedSampler(dataset) if is_distributed() else None
    dl = DataLoader(
        dataset,
        batch_size=tcfg["batch"]["per_gpu"],
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_model_samples,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    if args.dry_run:
        if is_main:
            batch = collate_model_samples([dataset[0], dataset[min(1, len(dataset)-1)]])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {list(v.shape)}")
        return

    if mcfg.get("compile") and torch.cuda.is_available():
        model = torch.compile(model, dynamic=True)
    if is_distributed():
        model = DDP(model, device_ids=[local])

    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["optimizer"]["lr"],
        weight_decay=tcfg["optimizer"]["weight_decay"], betas=tcfg["optimizer"]["betas"],
    )

    total_steps = tcfg["schedule"].get("total_steps", 60000)
    warmup_steps = tcfg["schedule"].get("warmup_steps", 2000)
    sched1 = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    sched2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(opt, schedulers=[sched1, sched2], milestones=[warmup_steps])

    train_cfg = TrainConfig({}, tcfg, mcfg, total_steps)

    model, manifest = train(
        model, dl, None, opt, scheduler, train_cfg, dev,
        resume_ckpt=args.resume, seed=42,
    )

    if is_main:
        print(f"Training complete. Run ID: {manifest.run_id}")


if __name__ == "__main__":
    main()
