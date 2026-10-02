"""Join auto_labeler outcomes into tokenized events.parquet.

Bridge between the two independently-produced artifacts:
  - events.parquet   (tokenize.py — context tokens only, no outcomes)
  - labels.parquet   (clif_auto_labeler.py — outcome states per stay)

Produces an augmented events.parquet with an `outcomes` column so
pretrain.py can consume supervised time-to-event labels.

Design note — competing-event cause_idx:
  Death is a hospitalization-level competing event. The CompetingRiskHead
  is constructed with n_targets+1 cause types — one slot per target
  outcome plus one global DEATH slot. When a hospitalization ends in
  death, every outcome records cause_idx = n_targets (the dedicated death
  index). This keeps the causal signal unified: death is always the same
  cause regardless of which outcome's perspective it is viewed from.
  The n_targets extra index is computed at call time from the data config.

Usage:
    python -m src.data.outcome_join \
      --labels output/intermediate_phi/mimic/labels.parquet \
      --events output/intermediate_phi/mimic/events.parquet \
      --vocab output/intermediate_phi/mimic/vocab.json \
      --out output/intermediate_phi/mimic/events.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.data.segments import threshold_bin, vocab_segments

ROOT = Path(__file__).parents[2]


def _death_cause_idx(data_config: dict) -> int:
    """Dedicated global death cause slot = number of target outcome types."""
    return len(data_config["target_concepts"])


def _compute_threshold_bin(concept: str, threshold: float, segments: dict,
                           direction: str) -> int:
    """The queried threshold's value bin, via the tokenizer's own `bin_index` over the
    frozen segments (`segments.threshold_bin`): the bin of a value just on the event side
    of the threshold. -1 when the concept has no bins."""
    segs = segments.get(concept)
    if segs is None:
        return -1
    return threshold_bin(threshold, segs, direction)


def join_outcomes(
    labels: pl.DataFrame,
    events: pl.DataFrame,
    vocab: dict,
    data_config: dict,
    cohort_config: dict,
) -> pl.DataFrame:
    segments = vocab_segments(vocab)  # refuses a pre-v2 (edge-list) vocabulary
    target_concepts = data_config["target_concepts"]
    concept_index = {c["name"]: idx for idx, c in enumerate(target_concepts)}
    outcome_specs = cohort_config["outcomes"]

    outcome_name_cols: dict[str, tuple[str]] = {}
    for name in outcome_specs:
        outcome_name_cols[name] = (
            f"{name}_status",
            f"{name}_time_from_anchor_hours",
        )

    label_cols = ["hospitalization_id"]
    for status_col, time_col in outcome_name_cols.values():
        label_cols.extend([status_col, time_col])
    labels_sub = labels.select([c for c in label_cols if c in labels.columns])

    outcome_rows: dict[str, list[dict]] = {}
    death_cause_idx = _death_cause_idx(data_config)
    # Per outcome, resolved once on first use (same order, same errors as per row):
    # (target_idx, direction, threshold_bin), or None for an outcome off the target map.
    resolved: dict[str, tuple[int, str, int] | None] = {}

    def resolve(outcome_name: str) -> tuple[int, str, int] | None:
        if outcome_name not in resolved:
            spec = outcome_specs[outcome_name]
            concept = spec["concept"]
            target_idx = concept_index.get(concept)
            if target_idx is None:
                resolved[outcome_name] = None
            else:
                direction = spec["direction"]
                threshold = float(spec["threshold"])
                resolved[outcome_name] = (target_idx, direction, _compute_threshold_bin(
                    concept, threshold, segments, direction))
        return resolved[outcome_name]

    for row in labels_sub.iter_rows(named=True):
        hosp_id = row["hospitalization_id"]
        outcomes_list = []
        for outcome_name, (status_col, time_col) in outcome_name_cols.items():
            status = row[status_col]
            time_hours = row[time_col]
            plan = resolve(outcome_name)
            if plan is None:
                continue
            target_idx, direction, threshold_bin = plan

            outcome_dict: dict[str, Any] = {
                "status": status,
                "target_idx": target_idx,
                "time_from_anchor_hours": time_hours,
                "threshold_bin": threshold_bin,
                "direction": direction,
                "cause_idx": death_cause_idx if status == "competing_event" else None,
            }

            outcomes_list.append(outcome_dict)
        outcome_rows[hosp_id] = outcomes_list

    events_with_outcomes = events.with_columns(
        pl.col("hosp_id")
        .replace_strict(outcome_rows, default=[])
        .alias("outcomes")
    )
    return events_with_outcomes


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Join auto_labeler outcome states into tokenized events"
    )
    ap.add_argument("--labels", required=True, help="output/.../labels.parquet")
    ap.add_argument("--events", required=True, help="output/.../events.parquet")
    ap.add_argument("--vocab", required=True, help="output/.../vocab.json")
    ap.add_argument("--out", required=True, help="augmented events output path")
    ap.add_argument("--cohort-config", default=str(ROOT / "configs/cohort.yaml"))
    ap.add_argument("--data-config", default=str(ROOT / "configs/data.yaml"))
    args = ap.parse_args()

    labels = pl.read_parquet(args.labels)
    events = pl.read_parquet(args.events)
    vocab = json.loads(Path(args.vocab).read_text())
    data_config = yaml.safe_load(Path(args.data_config).read_text())
    cohort_config = yaml.safe_load(Path(args.cohort_config).read_text())

    augmented = join_outcomes(labels, events, vocab, data_config, cohort_config)
    augmented.write_parquet(args.out)
    positive = 0
    outcome_names = list(cohort_config["outcomes"])
    for outcomes in augmented["outcomes"].to_list():
        for o in outcomes:
            if o.get("status") == "positive":
                positive += 1
    print(
        f"Wrote {args.out} ({len(augmented):,} stays, "
        f"{positive:,} positive-outcome instances across {len(outcome_names)} outcomes)"
    )


if __name__ == "__main__":
    main()