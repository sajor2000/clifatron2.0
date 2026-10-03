"""Self-supervised pretraining on one site's event shards (2x L40, DDP).

Loss = w_A*next_event + w_B*competing_risk + w_C*threshold_hazard + w_D*value_regression
(ORA marked-TTE; RESEARCH.md §3). Uses the training engine for resumable DDP training.

Run:
    torchrun --nproc_per_node=2 -m src.train.pretrain --config configs/train.yaml \
        --model-config configs/model.yaml --data data/mimic --site mimic
"""
from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
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
from src.data.dataset import LengthGroupedSampler, ModelDataset, TokenBudgetBatchSampler
from src.data.collate import collate_model_samples
from src.data.segments import (
    SAMPLE_VOCAB_REFUSAL,
    artifact_binding,
    compare_binding,
    is_sample_vocab,
    load_vocab_blob,
    n_value_bins,
)
from src.data.targets import TargetBuilder
from src.train.engine import setup_ddp, is_distributed, TrainConfig, train

logger = logging.getLogger(__name__)


def _load_decile_records(
    path: Path,
    *,
    partition: str | None = None,
    drop_values_without_stats: bool = False,
) -> list[dict]:
    """Normalize tokenizer `events.parquet` rows into ModelDataset records.

    The tokenizer produces context-only event shards. Until a cohort/outcome
    artifact is joined in, these records are valid for dry-run/NTP plumbing but
    carry no TTE outcomes.
    """
    frame = pl.read_parquet(path)
    if partition is not None:
        if "partition" not in frame.columns:
            raise ValueError("partition column is required before model fitting")
        frame = frame.filter(pl.col("partition") == partition)
        if frame.is_empty():
            raise ValueError(f"model fit partition {partition!r} has zero rows")
    rows = frame.to_dicts()
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
            "partition": row.get("partition"),
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


def _load_value_stats(
    path: str | None,
    *,
    expected_vocab_hash: str | None = None,
    expected_segments_hash: str | None = None,
    expected_fit_partition: str | None = None,
) -> dict[int, tuple[float, float]]:
    if path is None:
        return {}
    from src.data.value_stats import load_value_stats
    return load_value_stats(
        path,
        expected_vocab_hash=expected_vocab_hash,
        expected_segments_hash=expected_segments_hash,
        expected_fit_partition=expected_fit_partition,
    )


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


def objective_weights(mcfg: dict) -> dict[str, float]:
    """Objective mixture weights from `configs/model.yaml` `heads.*.weight`.

    These keys existed in the config but were dead — the trainer hardcoded its own
    values. Wiring them makes objective selection config-driven: the GEM pure-NTP base
    (docs/plans/2026-09-25-001-feat-icu-gem-generative-model-plan.md, D-G1) runs with
    ntp=1.0 and every TTE head at 0.0, with no code edits. `enabled: false` forces a
    weight to 0 regardless of the configured value.
    """
    h = mcfg.get("heads", {})
    defaults = {
        "next_event": 0.2,
        "competing_risk": 1.0,
        "threshold_hazard": 1.0,
        "value_regression": 0.5,
    }
    weights: dict[str, float] = {}
    for name, default in defaults.items():
        spec = h.get(name, {})
        weight = float(spec.get("weight", default)) if isinstance(spec, dict) else default
        if isinstance(spec, dict) and not spec.get("enabled", True):
            weight = 0.0
        weights[name] = weight
    return weights


class Model(torch.nn.Module):
    def __init__(self, vocab_size, n_targets, mcfg, *, n_value_bins: int,
                 encoder: torch.nn.Module | None = None):
        """`n_value_bins` comes from the frozen vocabulary (`segments.n_value_bins`).

        `encoder` (default: a fresh `CLIFEncoder`) lets the tokenization ablation swap
        the input representation (continuous-fused value channel, TextCode embedding
        table) while keeping this objective, its masks and its config weights (KTD8)."""
        super().__init__()
        n_causes = n_targets + 1  # +1 for global death competing-cause slot
        self.enc = encoder if encoder is not None else CLIFEncoder(vocab_size, mcfg)
        d = self.enc.d_model
        h = mcfg["heads"]
        self.w = objective_weights(mcfg)
        self.cr = CompetingRiskHead(d, n_causes, h["competing_risk"]["n_time_bins"])
        self.th = ThresholdHazardHead(
            d, n_targets, h["threshold_hazard"]["n_time_bins"],
            n_value_bins=n_value_bins, thr_dim=h["threshold_hazard"]["threshold_embed_dim"],
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
        if getattr(self.enc, "uses_value_channel", False):
            # Continuous-fused arm: edgeless concept ids + the normalized current value.
            if "input_value" not in batch or "input_value_mask" not in batch:
                raise ValueError(
                    "the continuous-fused encoder needs the input_value channel "
                    "(ModelDataset(value_channel=True))"
                )
            H = self.enc(batch["token"], batch["pos_min"],
                         continuous_value=batch["input_value"],
                         continuous_value_mask=batch["input_value_mask"])
        else:
            encoder_token = batch.get("soft_token", batch["token"])
            H = self.enc(encoder_token, batch["pos_min"], batch.get("soft_weight"))
        if "anchor_batch_idx" in batch:
            if len(batch["anchor_batch_idx"]):
                h_last = H[batch["anchor_batch_idx"], batch["last_idx"]]
            else:
                h_last = H.new_zeros((0, H.size(-1)))
        else:
            h_last = H[torch.arange(H.size(0), device=H.device), batch["last_idx"]]
        ntp = next_event_loss(self.enc.lm_logits(H), batch.get("ntp_target", batch["token"]), batch.get("ntp_mask"))
        cr = self._cr_loss(batch, h_last, H)
        th = self._th_loss(batch, h_last, H)
        val = self._val_loss(batch, H)
        w = self.w
        total = w["next_event"] * ntp + w["competing_risk"] * cr + w["threshold_hazard"] * th + w["value_regression"] * val
        return {"ntp": ntp, "cr": cr, "th": th, "val": val, "total": total}

    def _cr_loss(self, batch, h_last, H):
        """CR loss; 0 when the head is disabled/unweighted or the batch carries no CR
        targets (pure-NTP batches from shards without an outcome join)."""
        if self.w["competing_risk"] == 0.0 or "cr_type" not in batch or "cr_bin" not in batch:
            return H.new_tensor(0.0)
        cr_mask = batch.get("cr_mask")
        if cr_mask is not None and not bool(cr_mask.any()):
            return H.new_tensor(0.0)
        cr_h = h_last[cr_mask] if cr_mask is not None else h_last
        cr_type = batch["cr_type"][cr_mask] if cr_mask is not None else batch["cr_type"]
        cr_bin = batch["cr_bin"][cr_mask] if cr_mask is not None else batch["cr_bin"]
        return self.cr.loss(cr_h, cr_type, cr_bin)

    def _th_loss(self, batch, h_last, H):
        """Threshold-hazard loss with the same fail-to-zero guard as _cr_loss."""
        if self.w["threshold_hazard"] == 0.0 or "th_target" not in batch:
            return H.new_tensor(0.0)
        th_mask = batch.get("th_mask")
        if th_mask is not None and not bool(th_mask.any()):
            return H.new_tensor(0.0)
        th_h = h_last[th_mask] if th_mask is not None else h_last
        return self.th.loss(
            th_h,
            batch["th_target"][th_mask] if th_mask is not None else batch["th_target"],
            batch["th_tau"][th_mask] if th_mask is not None else batch["th_tau"],
            batch["th_dir"][th_mask] if th_mask is not None else batch["th_dir"],
            batch["th_crossed"][th_mask] if th_mask is not None else batch["th_crossed"],
            batch["th_observed_bin"][th_mask] if th_mask is not None and "th_observed_bin" in batch else batch.get("th_observed_bin"),
        )

    def _val_loss(self, batch, H):
        """Value-regression (ORA mark) loss; 0 when disabled or no values present."""
        if self.vr is None or self.w["value_regression"] == 0.0:
            return H.new_tensor(0.0)
        if "value" not in batch or "val_mask" not in batch:
            return H.new_tensor(0.0)
        return self.vr.loss_aligned(
            H,
            batch.get("ntp_target", batch["token"]),
            batch["value"],
            batch["val_mask"],
        )


@dataclass
class Loaders:
    """What `build_loaders` returns: the train/validation loaders and what built them."""

    train: DataLoader
    validation: DataLoader
    train_dataset: ModelDataset
    validation_dataset: ModelDataset
    records: list[dict]
    data_path: Path
    value_stats: dict[int, tuple[float, float]]


def _apply_soft_policy(records: list[dict], soft: bool | None) -> None:
    """`soft=None` keeps the shard's soft fields; `False` drops them (hard-token arm);
    `True` requires a shard tokenized with soft discretization (width > 1), so a soft arm
    can never silently train on hard bins."""
    if soft is None:
        return
    for record in records:
        if not soft:
            record["soft_token"] = None
            record["soft_weight"] = None
        elif not record.get("soft_token") or len(record["soft_token"][0]) < 2:
            raise SystemExit(
                "this arm uses soft discretization but the shard carries no soft bins; "
                "tokenize it with value_binning.soft_discretization: true"
            )


def build_loaders(
    events_path: str | Path,
    *,
    binding: dict[str, str],
    vocab_blob: Mapping,
    tcfg: dict,
    mcfg: dict,
    vocab_size: int,
    value_stats_path: str | Path | None = None,
    dry_run: bool = False,
    is_main: bool = True,
    soft: bool | None = None,
    value_channel: bool = False,
) -> Loaders:
    """Shard records -> TargetBuilder -> ModelDataset -> samplers -> DataLoaders.

    The one data path shared by `pretrain.py` and the tokenization ablation (KTD8).
    `binding` (`segments.artifact_binding` of the vocab.json the shard was encoded with)
    binds the value stats and every shard row. When `events_path` carries no supervised
    outcomes, its sibling `events_with_outcomes.parquet` is used (refused when stale).
    `soft` selects hard vs soft encoder inputs (see `_apply_soft_policy`);
    `value_channel` adds the normalized current-value field (continuous-fused arm).

    `vocab_blob` is that vocab.json itself: it must match `binding`, and a vocabulary
    fit on a verification sample (`provenance.sample: true`, KTD9) is refused unless
    `dry_run` — a sample vocabulary is smoke-only.
    """
    compare_binding(binding, artifact_binding(vocab_blob), what="training binding")
    if is_sample_vocab(vocab_blob) and not dry_run:
        raise SystemExit(f"refusing to train: the vocabulary {SAMPLE_VOCAB_REFUSAL}")
    # Bind value-stats to the data's vocabulary AND segments so a stale /
    # cross-vocabulary / cross-bin stats file is rejected rather than silently applying
    # unrelated centers/scales.
    value_stats = _load_value_stats(
        None if value_stats_path is None else str(value_stats_path),
        expected_vocab_hash=binding["vocabulary"],
        expected_segments_hash=binding["numeric_edges"],
        expected_fit_partition="train",
    )
    data_path = Path(events_path)
    data_dir = data_path.parent
    model_data_path = data_path
    records = _load_decile_records(
        data_path,
        partition="train",
        drop_values_without_stats=dry_run and not value_stats,
    )
    if not _has_supervised_outcomes(records):
        augmented_path = data_dir / "events_with_outcomes.parquet"
        if augmented_path.exists() and augmented_path != data_path:
            data_mtime = data_path.stat().st_mtime if data_path.exists() else 0
            augmented_mtime = augmented_path.stat().st_mtime
            if augmented_mtime < data_mtime:
                msg = (
                    f"Stale events_with_outcomes.parquet detected: augmented mtime "
                    f"({augmented_mtime}) < events.parquet mtime ({data_mtime}). "
                    f"Re-run outcome_join.py to regenerate."
                )
                if is_main:
                    logger.error(msg)
                raise SystemExit(msg)
            records = _load_decile_records(
                augmented_path,
                partition="train",
                drop_values_without_stats=dry_run and not value_stats,
            )
            model_data_path = augmented_path
            logger.info("using augmented events: %s", augmented_path)
            if is_main:
                print(f"  using augmented events: {augmented_path}")
        elif is_main:
            logger.warning(
                "events.parquet has no supervised outcomes and "
                "events_with_outcomes.parquet does not exist. "
                "Dry-run will proceed without TTE supervision; real training will fail."
            )
    if not dry_run and not value_stats and _has_numeric_values(records):
        raise SystemExit(
            "value-head normalization is required before real training; pass --value-stats. "
            "Generate it from the reference site: "
            "`python -m src.data.value_stats --events <ref_events.parquet> --out value_stats.json`"
        )
    if not dry_run and not _has_supervised_outcomes(records):
        raise SystemExit("TTE supervision is required before real pretraining; join cohort outcome artifacts first")
    target_builder = TargetBuilder(
        vocab_size=vocab_size,
        n_time_bins=mcfg["heads"]["competing_risk"]["n_time_bins"],
        horizon_hours=mcfg["heads"]["competing_risk"].get("horizon_hours", 48),
        value_stats=value_stats,
        run_seed=42,
    )
    # Every shard row must be bound to this vocabulary and these segments.
    expected_hashes = dict(binding)
    _apply_soft_policy(records, soft)
    dataset = ModelDataset(
        records,
        representation="decile",
        target_builder=target_builder,
        expected_hashes=expected_hashes,
        epoch=0,
        value_channel=value_channel,
    )
    validation_records = _load_decile_records(
        model_data_path,
        partition="validation",
        drop_values_without_stats=dry_run and not value_stats,
    )
    _apply_soft_policy(validation_records, soft)
    validation_dataset = ModelDataset(
        validation_records,
        representation="decile",
        target_builder=target_builder,
        expected_hashes=expected_hashes,
        epoch=0,
        value_channel=value_channel,
    )

    token_budget = int(tcfg["runtime"].get("token_budget", 0) or 0)
    if is_distributed():
        sampler = DistributedSampler(dataset)
        dl = DataLoader(
            dataset,
            batch_size=tcfg["batch"]["per_gpu"],
            sampler=sampler,
            collate_fn=collate_model_samples,
            num_workers=tcfg["runtime"].get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
        )
    elif token_budget > 0:
        # Token-budget batches: B x max_len <= budget — long stays batch alone,
        # short stays pack tight. Bounds MPS math-path attention transients (three
        # 86 GiB OOMs overnight) and cuts padding waste; CUDA flash doesn't need it.
        train_sampler = TokenBudgetBatchSampler(
            [len(r["token"]) for r in records],
            max_batch_tokens=token_budget,
            max_batch_size=tcfg["batch"]["per_gpu"],
            seed=42,
        )
        dl = DataLoader(
            dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_model_samples,
            num_workers=tcfg["runtime"].get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
        )
    else:
        # Length-grouped batches keep the padded shape (and its attention buffers)
        # recycling; uniform shuffle OOM'd the MPS watermark at 84 GiB overnight.
        sampler = LengthGroupedSampler(
            [len(r["token"]) for r in records],
            tcfg["batch"]["per_gpu"],
            seed=42,
        )
        dl = DataLoader(
            dataset,
            batch_size=tcfg["batch"]["per_gpu"],
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=collate_model_samples,
            num_workers=tcfg["runtime"].get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
        )
    if token_budget > 0 and not is_distributed():
        # Deterministic (sorted, unshuffled) token-budget val batches — same
        # memory bounds as training.
        val_sampler = TokenBudgetBatchSampler(
            [len(r["token"]) for r in validation_records],
            max_batch_tokens=token_budget,
            max_batch_size=tcfg["batch"]["per_gpu"],
            seed=42,
            shuffle=False,
        )
        validation_dl = DataLoader(
            validation_dataset,
            batch_sampler=val_sampler,
            collate_fn=collate_model_samples,
            num_workers=tcfg["runtime"].get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
        )
    else:
        validation_dl = DataLoader(
            validation_dataset,
            batch_size=tcfg["batch"]["per_gpu"],
            shuffle=False,
            collate_fn=collate_model_samples,
            num_workers=tcfg["runtime"].get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
        )
    return Loaders(
        train=dl,
        validation=validation_dl,
        train_dataset=dataset,
        validation_dataset=validation_dataset,
        records=records,
        data_path=model_data_path,
        value_stats=value_stats,
    )


def build_scheduler(opt, total_steps: int, warmup_steps: int):
    """Linear warmup then cosine decay (the pretrain schedule)."""
    sched1 = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    sched2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps))
    return SequentialLR(opt, schedulers=[sched1, sched2], milestones=[warmup_steps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--resume", default=None, help="path to checkpoint to resume")
    ap.add_argument("--fresh-schedule", action="store_true",
                    help="with --resume: load model+optimizer but keep THIS run's "
                         "config LR schedule (continuation runs; the saved decayed "
                         "schedule would pin LR at its tail value)")
    ap.add_argument("--dry-run", action="store_true", help="print model + loader info and exit")
    ap.add_argument("--value-stats", default=None, help="JSON token_id -> [center, scale] for value-head normalization")
    args = ap.parse_args()

    local, is_main = setup_ddp()
    # CUDA for the L40 box; MPS for Mac smoke tests (AGENTS.md dev workflow); CPU last.
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{local}")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        if is_main:
            print("device: mps (Mac smoke-test path)")
    else:
        dev = torch.device("cpu")
    tcfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = yaml.safe_load(Path(args.model_config).read_text())
    dcfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    n_targets = len(dcfg["target_concepts"])
    vocab_size = mcfg["trunk"].get("target_vocab", 10000)

    # KTD7: the data's tokenizer-v2 vocabulary binds everything below — the threshold
    # head's value-bin count, the shard rows, the value stats and every checkpoint.
    vblob = load_vocab_blob(Path(args.data) / "vocab.json",
                            required_for="training is bound to its vocabulary and segments")
    binding = artifact_binding(vblob)  # refuses a pre-v2 vocabulary (re-tokenize)

    model = Model(vocab_size, n_targets, mcfg, n_value_bins=n_value_bins(vblob)).to(dev)
    if is_main:
        print(f"params: {count_params(model)/1e6:.1f}M")

    loaders = build_loaders(
        Path(args.data) / "events.parquet",
        binding=binding,
        vocab_blob=vblob,
        tcfg=tcfg,
        mcfg=mcfg,
        vocab_size=vocab_size,
        value_stats_path=args.value_stats,
        dry_run=args.dry_run,
        is_main=is_main,
    )
    dataset, dl, validation_dl = loaders.train_dataset, loaders.train, loaders.validation

    if args.dry_run:
        if is_main:
            batch = collate_model_samples([dataset[0], dataset[min(1, len(dataset)-1)]])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {list(v.shape)}")
        if is_distributed():
            dist.destroy_process_group()
        return

    compile_enabled = tcfg["runtime"].get("compile", mcfg.get("compile", False))
    if compile_enabled and torch.cuda.is_available():
        model = torch.compile(model, dynamic=True)
    if is_distributed():
        model = DDP(model, device_ids=[local])

    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["optimizer"]["lr"],
        weight_decay=tcfg["optimizer"]["weight_decay"], betas=tcfg["optimizer"]["betas"],
    )

    total_steps = tcfg["schedule"].get("total_steps", 60000)
    warmup_steps = tcfg["schedule"].get("warmup_steps", 2000)
    scheduler = build_scheduler(opt, total_steps, warmup_steps)

    train_cfg = TrainConfig({}, tcfg, mcfg, total_steps)

    model, manifest = train(
        model, dl, validation_dl, opt, scheduler, train_cfg, dev,
        resume_ckpt=args.resume, seed=42, fresh_schedule=args.fresh_schedule,
        vocab_binding=binding,
    )

    if is_main:
        print(f"Training complete. Run ID: {manifest.run_id}")
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
