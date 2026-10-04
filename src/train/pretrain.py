"""Self-supervised pretraining on event shards (2x L40, DDP).

Loss = w_A*next_event + w_B*competing_risk + w_C*threshold_hazard + w_D*value_regression
(ORA marked-TTE; RESEARCH.md §3). Uses the training engine for resumable DDP training.

Run (24-hour representation, one site):
    uv run torchrun --nproc_per_node=2 -m src.train.pretrain --config configs/train.yaml \
        --model-config configs/model.yaml --data data/mimic --site mimic

Run (full-hospitalization representation, in-stream labels, one or more sites sharing ONE
frozen vocabulary; the first --data directory's vocab.json unless --vocab is given):
    uv run torchrun --nproc_per_node=2 -m src.train.pretrain --trajectory hospitalization \
        --data output/intermediate_phi/mimic output/intermediate_phi/rush \
        --site mimic rush --value-stats output/intermediate_phi/mimic/gem_value_stats.json

Value stats for this path are fit on the reference site's gem_events.parquet train stays
(`python -m src.data.value_stats --events <gem_events.parquet>`): the 24-hour stats lack
tokens seen only outside the ICU window, and target building refuses such a token.

A torchrun launch without CUDA is refused unless `--allow-cpu-ddp` is passed (a CPU / gloo
rehearsal); a single process (no torchrun) runs on CUDA, MPS or CPU as before.
"""
from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
import torch.distributed as dist
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.data.collate import collate_model_samples
from src.data.dataset import (
    DistributedTokenBudgetBatchSampler,
    header_token_ids,
    GemCorpus,
    LengthGroupedSampler,
    ModelDataset,
    TokenBudgetBatchSampler,
)
from src.data.segments import (
    SAMPLE_VOCAB_REFUSAL,
    artifact_binding,
    compare_binding,
    is_sample_vocab,
    load_vocab_blob,
    n_value_bins,
)
from src.data.targets import InStreamTargets, TargetBuilder
from src.data.threshold_grid import ThresholdGrid, load_thresholds
from src.data.tokenize_continuous import (
    REPRESENTATION as CONTINUOUS_FUSED,
)
from src.data.tokenize_continuous import (
    ContinuousThresholdGrid,
)
from src.model.encoder import CLIFEncoder, count_params
from src.model.heads import (
    CompetingRiskHead,
    ThresholdHazardHead,
    ValueRegressionHead,
    next_event_loss,
    time_bin,
)
from src.train.curriculum import (
    HEADS,
    apply_objective_arm,
    curriculum_enabled,
    curriculum_weights,
    describe_schedule,
    resolve_objective_arm,
)
from src.train.engine import (
    ALLOW_CPU_DDP_FLAG,
    ScheduleError,
    TrainConfig,
    is_distributed,
    resolve_resume,
    resolve_schedule,
    resolve_total_steps,
    select_device,
    setup_ddp,
    train,
    wrap_ddp,
)

logger = logging.getLogger(__name__)
# The seed of a run that passes none (`--seed`).
DEFAULT_SEED = 42


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


def in_stream_target_builder(
    *,
    vocab_blob: Mapping,
    mcfg: dict,
    dcfg: dict,
    thresholds: Mapping,
    vocab_size: int,
    value_stats: Mapping[int, tuple[float, float]],
    run_seed: int = DEFAULT_SEED,
) -> TargetBuilder:
    """The `gem_tte` TargetBuilder for one vocabulary (KTD1, KTD3).

    `thresholds` is `threshold_grid.load_thresholds()`: its `label_rule` gives the label
    horizon, lookback and ascertainment window, and its competing-risk causes the CR
    labels. `mcfg["heads"]["threshold_hazard"]["tau_sampling"]` selects how training
    thresholds are drawn (a missing or unknown value is refused), and
    `mcfg["in_stream"]` how many anchors and queries each window gets. No time grid is
    read here: the heads bin label minutes themselves (KTD4)."""
    rule = thresholds["label_rule"]
    tau = mcfg["heads"]["threshold_hazard"].get("tau_sampling")
    if vocab_blob.get("representation") == CONTINUOUS_FUSED:
        # Edgeless concept tokens; thresholds on the primary clinical segments.
        grid = ContinuousThresholdGrid(vocab_blob, dcfg["target_concepts"], thresholds,
                                       tau_sampling=tau)
    else:
        grid = ThresholdGrid(vocab_blob, dcfg["target_concepts"], thresholds,
                             tau_sampling=tau)
    # Outcome measurement filter (configs/cohort.yaml outcome_measurement_filters): OFF by
    # default; `in_stream.outcome_measurement_sensitivity: arterial_only` (model config)
    # counts only arterial MAP readings as measurements (the item-12 sensitivity arm).
    sensitivity = (mcfg.get("in_stream") or {}).get("outcome_measurement_sensitivity")
    filters, methods = None, frozenset()
    if sensitivity is not None:
        from src.data.targets import measurement_filters

        root = Path(__file__).resolve().parents[2]
        cohort_cfg = yaml.safe_load((root / dcfg["cohort_contract"]).read_text())
        filters, methods = measurement_filters(cohort_cfg, dcfg, vocab_blob["vocab"],
                                               dcfg["target_concepts"], sensitivity)
    return TargetBuilder(
        vocab_size=vocab_size,
        n_time_bins=mcfg["heads"]["threshold_hazard"]["n_time_bins"],
        horizon_hours=rule["horizon_hours"],
        value_stats=value_stats,
        run_seed=run_seed,
        mode="gem_tte",
        in_stream=InStreamTargets(
            grid=grid,
            anchors_per_window=int(mcfg["in_stream"]["anchors_per_window"]),
            queries_per_anchor=int(mcfg["in_stream"]["queries_per_anchor"]),
            baseline_lookback_hours=rule["baseline_lookback_hours"],
            required_measurement_within_hours_of_horizon=rule[
                "required_measurement_within_hours_of_horizon"],
            measurement_filters=filters, method_tokens=methods,
        ),
    )


def zero_touch(head: torch.nn.Module | None, like: torch.Tensor) -> torch.Tensor:
    """The loss of a skipped head: zero-valued, but reaching every trainable parameter.

    A head is skipped when its weight is zero or the batch has no supervised sample for
    it. Returning a bare constant would leave its parameters without a gradient on that
    rank, and DDP raises on the next step. This term gives them a zero gradient instead,
    so the parameters receiving gradients are the same on every rank and step (KTD2)."""
    params = [] if head is None else [p for p in head.parameters() if p.requires_grad]
    if not params:
        return like.new_tensor(0.0)
    return (sum(p.sum() for p in params) * 0.0).to(like.dtype)


class ObjectiveSchedule:
    """The loss weights a model applies on each optimizer update (KTD5).

    `w` holds the configured weights (`heads.*.weight`, or an objective arm's). Without
    the curriculum they apply from the first update; with it, `set_training_step` — called
    by the engine with its optimizer-update counter before each update — returns the
    `curriculum_weights` blend from next-token only to `w`. Every rank calls it with the
    same counter, so every rank applies the same weights on the same update.

    `_active` (a head is in the loss) drives the skip branches and changes only at the
    warm-up boundary; the weight values live in the `_loss_mix` buffer, so a compiled
    model does not recompile on every transition update. A head whose weight is zero
    still touches its parameters (`zero_touch`): DDP sees the same gradient set on every
    rank whatever the schedule."""

    HEAD_ATTRS = {"competing_risk": "cr", "threshold_hazard": "th",
                  "value_regression": "vr"}

    def _init_objective(self, weights: Mapping[str, float], mcfg: Mapping) -> None:
        self.w = {head: float(weights[head]) for head in HEADS}
        self.curriculum = curriculum_enabled(mcfg)
        self.objective_arm = mcfg.get("objective_arm")
        self.register_buffer("_loss_mix", torch.zeros(len(HEADS)), persistent=False)
        self._set_loss_weights(self.w)

    def _set_loss_weights(self, weights: Mapping[str, float]) -> None:
        self.loss_weights = {head: float(weights[head]) for head in HEADS}
        self._active = {head: weight > 0.0 for head, weight in self.loss_weights.items()}
        with torch.no_grad():
            self._loss_mix.copy_(torch.tensor([self.loss_weights[h] for h in HEADS]))

    def set_training_step(self, step: int, total_steps: int) -> dict[str, float]:
        """The weights for optimizer update `step` (0-based) of `total_steps` planned."""
        if self.curriculum:
            mix = curriculum_weights(step, total_steps,
                                     target=tuple(self.w[h] for h in HEADS))
            self._set_loss_weights(dict(zip(HEADS, mix[:4])))
        else:
            self._set_loss_weights(self.w)
        return dict(self.loss_weights)

    def _weight(self, head: str) -> torch.Tensor:
        return self._loss_mix[HEADS.index(head)]

    def _head_module(self, head: str):
        return getattr(self, self.HEAD_ATTRS[head], None)

    def head_parameters(self) -> dict[str, list[torch.nn.Parameter]]:
        """The parameters of each time-to-event / value head (by `HEADS` name). The engine
        switches their weight decay off while their weight is zero (`build_optimizer`)."""
        heads = {}
        for head in self.HEAD_ATTRS:
            module = self._head_module(head)
            if module is not None:
                heads[head] = list(module.parameters())
        return heads

    def head_weight_can_be_zero(self, head: str) -> bool:
        return self.w[head] == 0.0 or self.curriculum

    def objective_record(self) -> dict:
        """What this run optimizes, recorded in the manifest of every checkpoint. A head
        an arm removes is never listed as trained."""
        return {
            "objective_arm": self.objective_arm,
            "loss_balancing": "fixed",
            "curriculum": "ntp_then_tte" if self.curriculum else "none",
            "weights": dict(self.w),
            "trained_heads": [head for head in HEADS if self.w[head] > 0.0],
        }

    def describe_schedule(self, total_steps: int) -> str:
        return describe_schedule(total_steps, self.w, self.curriculum,
                                 arm=self.objective_arm)


# `trunk.rope_position` (configs/model.yaml): what the rotary positions count.
# admission_minutes = minutes since admission (primary); token_index = event order only
# (the R36 order-only ablation arm).
ROPE_POSITIONS = ("admission_minutes", "token_index")


class OrderOnlyEncoder(CLIFEncoder):
    """`CLIFEncoder` whose rotary positions are the token index, not admission minutes
    (R36: event order only). Same parameters, so checkpoints are interchangeable; only
    the position ids differ. RoPE is relative, so a window's offset does not matter."""

    order_only = True

    def forward(self, token, pos_min, token_weight=None) -> torch.Tensor:
        index = torch.arange(pos_min.size(-1), device=pos_min.device)
        return super().forward(token, index.expand_as(pos_min), token_weight)


def rope_position(mcfg: Mapping) -> str:
    position = mcfg["trunk"].get("rope_position", "admission_minutes")
    if position not in ROPE_POSITIONS:
        raise ValueError(f"unknown trunk.rope_position {position!r}; expected one of "
                         f"{ROPE_POSITIONS}")
    return position


class Model(ObjectiveSchedule, torch.nn.Module):
    def __init__(self, vocab_size, n_targets, mcfg, *, n_value_bins: int,
                 encoder: torch.nn.Module | None = None):
        """`n_value_bins` comes from the frozen vocabulary (`segments.n_value_bins`).

        `encoder` (default: a fresh `CLIFEncoder`, or `OrderOnlyEncoder` under
        `trunk.rope_position: token_index`) lets the tokenization ablation swap the input
        representation (continuous-fused value channel, TextCode embedding table) while
        keeping this objective, its masks and its config weights (KTD8)."""
        super().__init__()
        n_causes = n_targets + 1  # +1 for global death competing-cause slot
        order_only = rope_position(mcfg) == "token_index"
        if encoder is None:
            encoder = (OrderOnlyEncoder if order_only else CLIFEncoder)(vocab_size, mcfg)
        elif order_only:
            raise ValueError("trunk.rope_position token_index (order-only) is implemented "
                             "for the fused encoder only")
        self.enc = encoder
        d = self.enc.d_model
        h = mcfg["heads"]
        self.cr = CompetingRiskHead(d, n_causes, h["competing_risk"]["n_time_bins"])
        # Each head's own horizon: label minutes are binned per head (KTD4).
        self.cr_horizon_hours = float(h["competing_risk"].get("horizon_hours", 48))
        self.th_horizon_hours = float(h["threshold_hazard"].get("horizon_hours", 48))
        self.th = ThresholdHazardHead(
            d, n_targets, h["threshold_hazard"]["n_time_bins"],
            n_value_bins=n_value_bins, thr_dim=h["threshold_hazard"]["threshold_embed_dim"],
        )
        self.vr = ValueRegressionHead(d, vocab_size) if h["value_regression"]["enabled"] else None
        self._init_objective(objective_weights(mcfg), mcfg)

    def forward(self, batch):
        if "document_ids" in batch:
            # One device->host read for the whole batch (not one per row): a row is
            # multi-document when its smallest and largest valid document id differ.
            docs = batch["document_ids"]
            valid = docs >= 0
            high = torch.where(valid, docs, torch.full_like(docs, -1)).amax(dim=-1)
            low = torch.where(valid, docs, torch.full_like(docs, torch.iinfo(docs.dtype).max)
                              ).amin(dim=-1)
            if bool((valid.any(dim=-1) & (high != low)).any()):
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
        # One hidden state per ANCHOR (several per row on the full-hospitalization
        # representation), read at the anchor's own position, not at the last token.
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
        w = self._weight
        total = (w("next_event") * ntp + w("competing_risk") * cr
                 + w("threshold_hazard") * th + w("value_regression") * val)
        return {"ntp": ntp, "cr": cr, "th": th, "val": val, "total": total}

    @staticmethod
    def _on_grid(minutes, event, n_bins: int, horizon_hours: float):
        """Label minutes -> (event, slot) on ONE head's grid (KTD4). `slot` is
        `time_bin`: the event's bin, or the count of fully observed bins. An event
        after this head's horizon is not an event for it: the horizon was survived."""
        event = event & (minutes <= round(horizon_hours * 60))
        return event, time_bin(minutes, n_bins, horizon_hours).clamp(min=0, max=n_bins)

    def _cr_loss(self, batch, h_last, H):
        """CR loss; 0 when the head is disabled/unweighted or the batch carries no CR
        targets (pure-NTP batches from shards without an outcome join)."""
        timed = "cr_time_min" in batch
        if not self._active["competing_risk"] or "cr_type" not in batch or not (
                timed or "cr_bin" in batch):
            return zero_touch(self.cr, H)
        cr_type = batch["cr_type"]
        cr_mask = batch.get("cr_mask")
        if timed:
            event, cr_bin = self._on_grid(batch["cr_time_min"], cr_type >= 0,
                                          self.cr.n_bins, self.cr_horizon_hours)
            cr_type = torch.where(event, cr_type, torch.full_like(cr_type, -1))
            # Censored before one full bin of this grid: nothing was observed.
            observed = event | (cr_bin > 0)
            cr_mask = observed if cr_mask is None else cr_mask & observed
        else:
            cr_bin = batch["cr_bin"]
        if cr_mask is not None:
            # One device->host read: the selected rows' indices (an empty mask selects
            # none, and the head is touched at zero instead).
            rows = cr_mask.nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                return zero_touch(self.cr, H)
            h_last, cr_type, cr_bin = h_last[rows], cr_type[rows], cr_bin[rows]
        return self.cr.loss(h_last, cr_type, cr_bin)

    def _th_loss(self, batch, h_last, H):
        """Threshold-hazard loss with the same fail-to-zero guard as _cr_loss."""
        if not self._active["threshold_hazard"] or "th_target" not in batch:
            return zero_touch(self.th, H)
        th_mask = batch.get("th_mask")
        if "th_time_min" in batch:
            event, slot = self._on_grid(batch["th_time_min"], batch["th_event"],
                                        self.th.n_bins, self.th_horizon_hours)
            crossed = torch.where(event, slot.clamp(max=self.th.n_bins - 1),
                                  torch.full_like(slot, -1))
            observed_bin = slot
            observed = event | (slot > 0)
            th_mask = observed if th_mask is None else th_mask & observed
        else:
            crossed, observed_bin = batch["th_crossed"], batch.get("th_observed_bin")
        rows = None
        if th_mask is not None:
            rows = th_mask.nonzero(as_tuple=True)[0]     # one device->host read
            if rows.numel() == 0:
                return zero_touch(self.th, H)
        # Several threshold queries can share one anchor's hidden state.
        th_h = h_last[batch["th_anchor"]] if "th_anchor" in batch else h_last
        fields = [th_h, batch["th_target"], batch["th_tau"], batch["th_dir"], crossed]
        if rows is not None:
            fields = [field[rows] for field in fields]
            observed_bin = None if observed_bin is None else observed_bin[rows]
        return self.th.loss(*fields, observed_bin)

    def _val_loss(self, batch, H):
        """Value-regression (ORA mark) loss; 0 when disabled or no values present."""
        if self.vr is None or not self._active["value_regression"]:
            return zero_touch(self.vr, H)
        if "value" not in batch or "val_mask" not in batch:
            return zero_touch(self.vr, H)
        return self.vr.loss_aligned(
            H,
            batch.get("ntp_target", batch["token"]),
            batch["value"],
            batch["val_mask"],
        )


@dataclass
class Loaders:
    """What `build_loaders` returns: the train/validation loaders and what built them.

    On the full-hospitalization representation `validation` is None when no site has a
    validation stay, `sites` lists the site names in load order and `memory` is
    `loader_memory`'s resident-memory measurement of the loaded windows."""

    train: DataLoader
    validation: DataLoader | None
    train_dataset: ModelDataset
    validation_dataset: ModelDataset | None
    records: list[dict]
    data_path: Path
    value_stats: dict[int, tuple[float, float]]
    sites: list[str] | None = None
    memory: dict | None = None


def embedding_vocab_size(vocab_blob: Mapping, mcfg: Mapping) -> int:
    """The embedding / output-projection size: the frozen vocabulary's largest id + 1
    (R37: the reported model size is the measured one, not a padded `target_vocab`).
    `trunk.target_vocab` stays the budget cap: a vocabulary above it is refused."""
    size = max(int(i) for i in vocab_blob["vocab"].values()) + 1
    cap = int(mcfg["trunk"].get("target_vocab", 10000))
    if size > cap:
        raise SystemExit(f"the vocabulary needs {size} embedding rows, above the model "
                         f"config's trunk.target_vocab budget cap {cap}")
    return size


# Trunk keys a run may override (`--trunk KEY=VALUE`; the size sweep and the order-only
# arm, R36/R37). Anything else in `trunk` is part of the locked design.
TRUNK_OVERRIDES = {"d_model": int, "n_layers": int, "n_heads": int, "ffn_mult": int,
                   "rope_position": str}


def seed_everything(seed: int) -> None:
    """Seed model initialisation and every other draw from torch's default generators
    (CPU and every accelerator: `torch.manual_seed`). Batch order and anchor/threshold
    sampling take the seed explicitly (`build_loaders(seed=...)`)."""
    torch.manual_seed(int(seed))


def parse_trunk_overrides(items: list[str] | None) -> dict:
    """``["d_model=192", "n_layers=4"]`` -> ``{"d_model": 192, "n_layers": 4}``; an
    unknown key or a malformed item is refused."""
    out: dict = {}
    for item in items or ():
        key, sep, raw = str(item).partition("=")
        if not sep or key not in TRUNK_OVERRIDES:
            raise SystemExit(f"--trunk takes KEY=VALUE with KEY one of "
                             f"{sorted(TRUNK_OVERRIDES)}; got {item!r}")
        try:
            out[key] = TRUNK_OVERRIDES[key](raw)
        except ValueError as exc:
            raise SystemExit(f"--trunk {item!r}: {exc}") from exc
    return out


def apply_trunk_overrides(mcfg: Mapping, overrides: Mapping) -> dict:
    """A copy of the model config with `overrides` applied to `trunk` (refuses a size
    whose heads do not divide the width, and an unknown rope_position)."""
    import copy

    out = copy.deepcopy(dict(mcfg))
    out["trunk"] = {**out["trunk"], **dict(overrides)}
    if int(out["trunk"]["d_model"]) % int(out["trunk"]["n_heads"]):
        raise SystemExit(f"trunk d_model {out['trunk']['d_model']} is not divisible by "
                         f"n_heads {out['trunk']['n_heads']}")
    rope_position(out)
    return out


def checkpoint_model_config(ckpt: Mapping, mcfg: Mapping,
                            vocab_blob: Mapping | None = None) -> tuple[int, dict]:
    """``(embedding rows, model config)`` to rebuild a checkpoint's model.

    The rows come from the checkpoint manifest (`config.vocab_size`, recorded by every
    training runner); without one, from the saved token embedding, else
    `embedding_vocab_size(vocab_blob)`. The trunk recorded in the manifest (`config.trunk`,
    size-sweep and order-only overrides) replaces the given config's trunk."""
    config = ((ckpt.get("manifest") or {}).get("config") or {}) if isinstance(ckpt, Mapping) else {}
    trunk = config.get("trunk")
    out = apply_trunk_overrides(mcfg, trunk) if isinstance(trunk, Mapping) else dict(mcfg)
    size = config.get("vocab_size")
    if size is None:
        state = ckpt.get("model", ckpt) if isinstance(ckpt, Mapping) else {}
        weight = None
        if isinstance(state, Mapping):
            # The TextCode arm has no token embedding: its frozen table has one row per id.
            weight = state.get("enc.tok_emb.weight", state.get("enc.text_table"))
        if weight is not None:
            size = int(weight.shape[0])
        elif vocab_blob is not None:
            size = embedding_vocab_size(vocab_blob, out)
        else:
            size = int(out["trunk"].get("target_vocab", 10000))
    return int(size), out


def load_model_from_checkpoint(checkpoint, mcfg: Mapping, n_targets: int,
                               vocab_blob: Mapping, *, verify: bool = True) -> "Model":
    """The `Model` of a training checkpoint (path or loaded dict), sized from its
    manifest (`checkpoint_model_config`), refused unless bound to `vocab_blob`, with its
    weights loaded. `n_value_bins` comes from the vocabulary (the primary segments' for a
    continuous-fused one); the encoder is the arm's (`checkpoint_encoder`)."""
    from src.train.checkpoint import load_checkpoint, verify_checkpoint_binding

    ckpt = load_checkpoint(checkpoint) if not isinstance(checkpoint, Mapping) else checkpoint
    if verify:
        verify_checkpoint_binding(ckpt, vocab_blob)
    vocab_size, cfg = checkpoint_model_config(ckpt, mcfg, vocab_blob)
    state = ckpt.get("model", ckpt)
    model = Model(vocab_size, n_targets, cfg, n_value_bins=vocab_value_bins(vocab_blob),
                  encoder=checkpoint_encoder(state, vocab_size, cfg, vocab_blob))
    model.load_state_dict(state)
    return model


def checkpoint_encoder(state: Mapping, vocab_size: int, cfg: Mapping,
                       vocab_blob: Mapping) -> torch.nn.Module | None:
    """The input encoder a checkpoint was trained with, as the tokenization ablation
    builds it (`run_tokenization_ablation.TokenizationAblationModel`): the
    continuous-fused value channel for a continuous-fused vocabulary, the TextCode
    encoder (its frozen description table restored from the saved buffer) when the
    state carries `enc.text_table`, else None (`Model`'s fused / order-only default)."""
    if vocab_blob.get("representation") == CONTINUOUS_FUSED:
        from src.model.encoder_continuous import ContinuousFusedEncoder

        return ContinuousFusedEncoder(vocab_size, cfg)
    table = state.get("enc.text_table")
    if table is not None:
        from src.model.encoder_textcode import TextCodeEncoder

        return TextCodeEncoder(vocab_size, cfg, table.detach().cpu().float().numpy())
    return None


def vocab_value_bins(vocab_blob: Mapping) -> int:
    """The threshold head's value-bin count of any arm's vocabulary."""
    if vocab_blob.get("representation") == CONTINUOUS_FUSED:
        from src.data.tokenize_continuous import primary_n_value_bins

        return primary_n_value_bins(vocab_blob)
    return n_value_bins(vocab_blob)


def resident_set_bytes() -> int:
    """This process's current resident set size in bytes (Linux /proc, else `ps`)."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip()) * 1024


def loader_memory(before: int, datasets) -> dict:
    """Resident memory the loaded windows added (bytes, and bytes per event) — rough: it
    is a process RSS delta (`before` is measured before the load), so allocator reuse can
    hide or inflate a little."""
    gc.collect()
    after = resident_set_bytes()
    events = sum(ds.corpus.events for ds in datasets if ds is not None)
    return {"rss_before_bytes": before, "rss_after_bytes": after, "events": events,
            "bytes_per_event": (after - before) / max(events, 1)}


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
    representation: str = "decile",
    dcfg: Mapping | None = None,
    thresholds: Mapping | None = None,
    seed: int = 42,
) -> Loaders:
    """Shard records -> TargetBuilder -> ModelDataset -> samplers -> DataLoaders.

    `representation="decile"` (default) is the 24-hour path described below.
    `representation="gem"` is the full-hospitalization path (`_build_gem_loaders`):
    `events_path` is then ``{site: gem_events.parquet}`` and `dcfg` (the data config, for
    its target concepts) is required; `thresholds` defaults to `load_thresholds()`.

    The one data path shared by `pretrain.py` and the tokenization ablation (KTD8).
    `binding` (`segments.artifact_binding` of the vocab.json the shard was encoded with)
    binds the value stats and every shard row. When `events_path` carries no supervised
    outcomes, its sibling `events_with_outcomes.parquet` is used (refused when stale).
    `soft` selects hard vs soft encoder inputs (see `_apply_soft_policy`);
    `value_channel` adds the normalized current-value field (continuous-fused arm).

    `vocab_blob` is that vocab.json itself: it must match `binding`, and a vocabulary
    fit on a verification sample (`provenance.sample: true`, KTD9) is refused unless
    `dry_run` — a sample vocabulary is smoke-only.

    `seed` (the run's `--seed`) drives the batch order (every sampler) and the anchor
    and threshold sampling (`TargetBuilder.run_seed`); model initialisation is seeded
    by the caller (`seed_everything`).
    """
    if representation not in ("decile", "gem"):
        raise ValueError(f"representation must be 'decile' or 'gem', got {representation!r}")
    compare_binding(binding, artifact_binding(vocab_blob), what="training binding")
    if is_sample_vocab(vocab_blob) and not dry_run:
        raise SystemExit(f"refusing to train: the vocabulary {SAMPLE_VOCAB_REFUSAL}")
    if representation == "gem":
        if not isinstance(events_path, Mapping):
            raise ValueError("the gem representation reads {site: gem_events.parquet}")
        if dcfg is None:
            raise ValueError("the gem representation needs the data config (dcfg)")
        return _build_gem_loaders(
            events_path, binding=binding, vocab_blob=vocab_blob, tcfg=tcfg, mcfg=mcfg,
            dcfg=dcfg, thresholds=load_thresholds() if thresholds is None else thresholds,
            vocab_size=vocab_size, value_stats_path=value_stats_path, dry_run=dry_run,
            soft=soft, value_channel=value_channel, seed=seed)
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
        run_seed=seed,
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
        dl = _batched_loader(dataset, tcfg, seed=seed)
    elif token_budget > 0:
        # Token-budget batches: B x max_len <= budget — long stays batch alone,
        # short stays pack tight. Bounds MPS math-path attention transients (three
        # 86 GiB OOMs overnight) and cuts padding waste; CUDA flash doesn't need it.
        train_sampler = TokenBudgetBatchSampler(
            [len(r["token"]) for r in records],
            max_batch_tokens=token_budget,
            max_batch_size=tcfg["batch"]["per_gpu"],
            seed=seed,
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
            seed=seed,
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
            seed=seed,
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


def _window_lengths(dataset: ModelDataset) -> list[int]:
    """Events per training row: the gem corpus's window lengths (no per-window lists),
    else each record's token count."""
    return dataset.sample_lengths()


def _batched_loader(dataset: ModelDataset, tcfg: dict, *, seed: int) -> DataLoader:
    """Training loader over `DistributedTokenBudgetBatchSampler`: batches formed globally
    under `runtime.token_budget` (rows only when it is 0) and `batch.per_gpu`, dealt to
    the ranks of the process group (one rank outside DDP), every rank the same count."""
    sampler = DistributedTokenBudgetBatchSampler(
        _window_lengths(dataset),
        max_batch_tokens=int(tcfg["runtime"].get("token_budget", 0) or 0) or None,
        max_batch_size=tcfg["batch"]["per_gpu"],
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_model_samples,
        num_workers=tcfg["runtime"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def _build_gem_loaders(
    sites: Mapping[str, str | Path],
    *,
    binding: dict[str, str],
    vocab_blob: Mapping,
    tcfg: dict,
    mcfg: dict,
    dcfg: Mapping,
    thresholds: Mapping,
    vocab_size: int,
    value_stats_path: str | Path | None,
    dry_run: bool,
    soft: bool | None,
    value_channel: bool,
    seed: int,
) -> Loaders:
    """The full-hospitalization path (U5, KTD1): one or more sites' `gem_events.parquet`,
    each beside a vocab.json that must be the training vocabulary (same binding; a sample
    vocabulary is refused unless `dry_run`), read side by side into ONE dataset — sites
    are never pooled on disk; both development sites live on the node that trains. Each
    partition is one columnar `GemCorpus` (memory-mapped cache in ``gem_cache/`` beside
    each shard, shared page cache across ranks): the `gem_tte` labels of a window read
    events from later windows, so whole stays are assembled from the columns before
    labelling. Each shard's bindings are checked by `ModelDataset`."""
    names = list(sites)
    if not names or len(set(names)) != len(names) or any(
            not name or ":" in str(name) for name in names):
        raise ValueError(f"site names must be unique, non-empty and free of ':': {names}")
    for name, path in sites.items():
        site_vocab = Path(path).parent / "vocab.json"
        if site_vocab.exists():
            site_blob = load_vocab_blob(site_vocab)
            compare_binding(artifact_binding(site_blob), binding,
                            what=f"site {name} vocabulary")
            if is_sample_vocab(site_blob) and not dry_run:
                raise SystemExit(f"refusing to train on site {name}: its vocabulary "
                                 f"{SAMPLE_VOCAB_REFUSAL}")
    value_stats = _load_value_stats(
        None if value_stats_path is None else str(value_stats_path),
        expected_vocab_hash=binding["vocabulary"],
        expected_segments_hash=binding["numeric_edges"],
        expected_fit_partition="train",
    )
    if value_stats:
        # Fail before the first step, not at the first target: EVERY site's numeric
        # tokens (train + validation) must have value stats (fit on all sites' train).
        from src.data.value_stats import coverage_gaps, describe_gaps

        gaps = coverage_gaps({name: Path(path) for name, path in sites.items()}, value_stats)
        if gaps:
            names = {v: k for k, v in vocab_blob["vocab"].items()}
            raise SystemExit(describe_gaps(gaps, names) + ": refit the value stats on every "
                             "site's shard (python -m src.data.value_stats --events "
                             "mimic=... --events rush=...)")
    target_builder = in_stream_target_builder(
        vocab_blob=vocab_blob, mcfg=mcfg, dcfg=dcfg, thresholds=thresholds,
        vocab_size=vocab_size, value_stats=value_stats, run_seed=seed)
    rss_before = resident_set_bytes()
    datasets = {}
    # Continuation header (src/data/dataset.py): the shards record how they were cut
    # (`continuation_header`, `header_length` in artifact_hashes) and training refuses a
    # shard cut the other way (`require_cut_match`): re-tokenize it, never train it as is.
    header = (header_token_ids(vocab_blob["vocab"])
              if mcfg["trunk"].get("continuation_header", False) else None)
    for partition in ("train", "validation"):
        # Columnar, memory-mapped per site (cache beside each shard, gem_cache/); keys
        # are ``<site>:<hosp_id>`` so the same raw identifier at two sites stays two.
        corpus = GemCorpus.concat([
            GemCorpus.from_parquet(path, site=name, partition=partition, cache=True)
            for name, path in sites.items()])
        if partition == "train":
            if len(corpus) == 0:
                raise ValueError("no train-partition windows in the given sites")
            if not dry_run and not value_stats and corpus.has_numeric_values():
                raise SystemExit("value-head normalization is required before real "
                                 "training; pass --value-stats")
        if dry_run and not value_stats:
            corpus.drop_values()
        if soft is False:
            corpus.drop_soft()
        elif soft and len(corpus) and (corpus.soft_width() or 0) < 2:
            raise SystemExit(
                "this arm uses soft discretization but the shard carries no soft bins; "
                "tokenize it with value_binning.soft_discretization: true"
            )
        datasets[partition] = ModelDataset(
            corpus, representation="gem", target_builder=target_builder,
            expected_hashes=dict(binding), value_channel=value_channel,
            continuation_header=header, require_cut_match=True,
            max_tokens=None if header is None else int(mcfg["trunk"]["max_tokens"]),
        ) if len(corpus) else None
    memory = loader_memory(rss_before, datasets.values())
    validation = datasets["validation"]
    validation_dl = None if validation is None else DataLoader(
        validation,
        batch_sampler=TokenBudgetBatchSampler(
            validation.sample_lengths(),
            max_batch_tokens=int(tcfg["runtime"].get("token_budget", 0) or 0) or 2 ** 62,
            max_batch_size=tcfg["batch"]["per_gpu"], seed=seed, shuffle=False),
        collate_fn=collate_model_samples,
        num_workers=tcfg["runtime"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    return Loaders(
        train=_batched_loader(datasets["train"], tcfg, seed=seed),
        validation=validation_dl,
        train_dataset=datasets["train"],
        validation_dataset=validation,
        records=datasets["train"].records,
        data_path=Path(next(iter(sites.values()))),
        value_stats=value_stats,
        sites=names,
        memory=memory,
    )


def build_optimizer(model, *, lr: float, weight_decay: float, betas, head_lr: float | None = None,
                    trunk_prefixes: tuple[str, ...] | None = None) -> torch.optim.AdamW:
    """AdamW with one parameter group per time-to-event / value head (KTD5).

    AdamW's decoupled decay shrinks a parameter even when its gradient is exactly zero,
    so a head held at zero weight by the curriculum (or removed by an objective arm)
    would drift toward zero. Each head group carries `head` and `head_weight_decay`;
    the engine sets its `weight_decay` to 0 while that head's weight is 0 and back to
    `head_weight_decay` once it trains. With zero gradient, zero moments and no decay,
    AdamW leaves the parameter bit-identical.

    The remaining trainable parameters (trunk, next-event projection) form the first
    group at `lr`; with `trunk_prefixes`, only names starting with one of them do, and
    the other non-head parameters train at `head_lr` with the heads (default `lr`).
    Frozen parameters (`requires_grad=False`) are left out."""
    module = model.module if hasattr(model, "module") else model
    module = getattr(module, "_orig_mod", module)
    head_lr = lr if head_lr is None else head_lr
    owner = {}
    if hasattr(module, "head_parameters"):
        for head, params in module.head_parameters().items():
            owner.update({id(p): head for p in params})
    trunk, other, heads = [], [], {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in owner:
            heads.setdefault(owner[id(param)], []).append(param)
        elif trunk_prefixes is None or name.startswith(trunk_prefixes):
            trunk.append(param)
        else:
            other.append(param)
    groups = [{"params": trunk, "lr": lr}] if trunk else []
    if other:
        groups.append({"params": other, "lr": head_lr})
    for head in HEADS:
        if heads.get(head):
            groups.append({"params": heads[head], "lr": head_lr, "head": head,
                           "head_weight_decay": weight_decay})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, betas=betas)


def build_scheduler(opt, total_steps: int, warmup_steps: int):
    """Linear warmup then cosine decay (the pretrain schedule). `warmup_steps` comes from
    `engine.resolve_schedule` (absolute or a share of the run); with none (a run too short
    for `schedule.warmup_frac` to give one update) it is cosine decay alone."""
    if warmup_steps <= 0:
        return CosineAnnealingLR(opt, T_max=max(1, total_steps))
    sched1 = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    sched2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps))
    return SequentialLR(opt, schedulers=[sched1, sched2], milestones=[warmup_steps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--data", required=True, nargs="+",
                    help="site shard directory (several with --trajectory hospitalization)")
    ap.add_argument("--site", required=True, nargs="+",
                    help="site name per --data directory, in the same order")
    ap.add_argument("--trajectory", choices=("icu_24h", "hospitalization"), default="icu_24h",
                    help="icu_24h: events.parquet (one site); hospitalization: "
                         "gem_events.parquet with in-stream time-to-event labels (U5)")
    ap.add_argument("--vocab", default=None,
                    help="the frozen vocab.json (default: the first --data directory's)")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume, or 'latest' for the newest one in "
                         "runtime.ckpt_dir")
    ap.add_argument("--fresh-schedule", action="store_true",
                    help="with --resume: load model+optimizer but keep THIS run's "
                         "config LR schedule (continuation runs; the saved decayed "
                         "schedule would pin LR at its tail value)")
    ap.add_argument("--dry-run", action="store_true", help="print model + loader info and exit")
    ap.add_argument("--value-stats", default=None, help="JSON token_id -> [center, scale] for value-head normalization")
    ap.add_argument("--objective-arm", default=None,
                    help="objective variant from configs/objective_arms.yaml (full, "
                         "next_token_only, minus_value, minus_competing_risk, "
                         "minus_threshold, no_curriculum); default: model config as is")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="the run's seed: model initialisation, batch order, anchor and "
                         "threshold sampling (recorded in the manifest)")
    ap.add_argument("--trunk", action="append", default=[], metavar="KEY=VALUE",
                    help=f"trunk override, repeatable ({', '.join(TRUNK_OVERRIDES)}): the "
                         "size sweep and the order-only arm (recorded in the manifest)")
    ap.add_argument("--passes", type=float, default=None,
                    help="run length in passes over the training data (schedule.passes)")
    ap.add_argument(ALLOW_CPU_DDP_FLAG, action="store_true",
                    help="let a distributed (torchrun) launch run on CPU over gloo when "
                         "CUDA is unavailable; without it such a launch is refused")
    args = ap.parse_args()

    local, is_main = setup_ddp(allow_cpu=args.allow_cpu_ddp)
    dev = select_device(local)
    if is_main and dev.type == "mps":
        print("device: mps (Mac smoke-test path)")
    tcfg = yaml.safe_load(Path(args.config).read_text())
    mcfg = yaml.safe_load(Path(args.model_config).read_text())
    if args.objective_arm is not None:
        mcfg = apply_objective_arm(mcfg, resolve_objective_arm(args.objective_arm))
    mcfg = apply_trunk_overrides(mcfg, parse_trunk_overrides(args.trunk))
    if args.passes is not None:
        tcfg["schedule"]["passes"] = args.passes
    curriculum_enabled(mcfg)  # refuse an unknown loss_balancing / curriculum before work
    dcfg = yaml.safe_load(Path(args.data_config).read_text())
    n_targets = len(dcfg["target_concepts"])
    gem = args.trajectory == "hospitalization"
    if len(args.site) != len(args.data) or (not gem and len(args.data) != 1):
        raise SystemExit("pass one --site per --data directory; several sites need "
                         "--trajectory hospitalization")

    # KTD7: the data's tokenizer-v2 vocabulary binds everything below — the threshold
    # head's value-bin count, the shard rows, the value stats and every checkpoint.
    vblob = load_vocab_blob(args.vocab or Path(args.data[0]) / "vocab.json",
                            required_for="training is bound to its vocabulary and segments")
    binding = artifact_binding(vblob)  # refuses a pre-v2 vocabulary (re-tokenize)
    vocab_size = embedding_vocab_size(vblob, mcfg)

    seed_everything(args.seed)
    model = Model(vocab_size, n_targets, mcfg, n_value_bins=n_value_bins(vblob)).to(dev)
    if is_main:
        print(f"params: {count_params(model)/1e6:.2f}M (embedding rows {vocab_size})")

    loaders = build_loaders(
        {site: Path(d) / "gem_events.parquet" for site, d in zip(args.site, args.data)}
        if gem else Path(args.data[0]) / "events.parquet",
        binding=binding,
        vocab_blob=vblob,
        tcfg=tcfg,
        mcfg=mcfg,
        vocab_size=vocab_size,
        value_stats_path=args.value_stats,
        dry_run=args.dry_run,
        is_main=is_main,
        representation="gem" if gem else "decile",
        dcfg=dcfg,
        seed=args.seed,
    )
    dataset, dl, validation_dl = loaders.train_dataset, loaders.train, loaders.validation
    if is_main and loaders.memory is not None:
        memory = loaders.memory
        print(f"loader: {memory['events']:,} events, resident +"
              f"{(memory['rss_after_bytes'] - memory['rss_before_bytes']) / 2**20:.0f} MiB "
              f"(~{memory['bytes_per_event']:.0f} B/event, per rank)")

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
    # Inner module compiled, then wrapped (see engine.wrap_ddp on the two documented
    # orders); compile is off by default.
    if compile_enabled and torch.cuda.is_available():
        model = torch.compile(model, dynamic=True)
    if is_distributed():
        model = wrap_ddp(model, dev, local)

    opt = build_optimizer(
        model, lr=tcfg["optimizer"]["lr"],
        weight_decay=tcfg["optimizer"]["weight_decay"], betas=tcfg["optimizer"]["betas"],
    )

    total_steps = resolve_total_steps(tcfg, len(dl))
    try:
        schedule = resolve_schedule(tcfg, total_steps)
    except ScheduleError as exc:
        raise SystemExit(f"refusing to train: {exc}") from exc
    scheduler = build_scheduler(opt, total_steps, schedule.warmup_steps)
    if is_main:
        if tcfg["schedule"].get("passes") is not None:
            print(f"run length: {tcfg['schedule']['passes']} passes x {len(dl)} batches per "
                  f"rank -> {total_steps} optimizer updates")
        print(f"schedule: {schedule.describe()}")

    train_cfg = TrainConfig({}, tcfg, mcfg, total_steps)
    # Recorded in every checkpoint manifest (config): what a consumer needs to rebuild the
    # model, and the loader's measured memory.
    train_cfg.vocab_size = vocab_size
    train_cfg.trajectory = args.trajectory
    train_cfg.sites = list(args.site)
    train_cfg.loader_memory = loaders.memory
    train_cfg.trunk = dict(mcfg["trunk"])

    model, manifest = train(
        model, dl, validation_dl, opt, scheduler, train_cfg, dev,
        resume_ckpt=resolve_resume(args.resume, train_cfg.ckpt_dir), seed=args.seed,
        fresh_schedule=args.fresh_schedule, vocab_binding=binding,
    )

    if is_main:
        print(f"Training complete. Run ID: {manifest.run_id}")
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
