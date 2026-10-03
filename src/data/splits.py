"""Deterministic patient- and linked-encounter-grouped partitions.

HELD-OUT STRATIFICATION (product authority, 2026-10-03, item 46; off until the split
freeze). At 60/15/10/15 the held-out NIV and HFNC arms of the extubation study are too
small, and relying on Rush alone for them would let site-specific confounding into the
held-out evaluation. `HeldOutStratification` over-samples patients whose FIRST
post-extubation device is NIV or HFNC into the held-out partitions (every partition except
the pretraining one) until each of those arms holds `share` of its patients there, and
moves the same number of other patients the other way, partition by partition, so every
partition keeps its size. Everything is deterministic by seed; other patients move only
held-out -> train (or train -> held-out when an arm needs fewer), never between held-out
partitions. The stratum is read from PRE-REGISTERED, OUTCOME-BLIND cohort membership only:
`patient_id`, `eligible` and the device-arm column of the extubation cohort artifact
(`load_held_out_strata` reads exactly those columns). Outcomes are never read.

This changes the training data's composition (fewer NIV/HFNC extubations are seen in
pretraining) and is baked into every vocabulary, shard and checkpoint, so it must be
decided before the split freeze (KTD6; docs/plans/l40-runbook.md step 7).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# The only columns the stratification may read from the extubation cohort artifact
# (besides the configured arm column): never an outcome or label column.
STRATA_ID_COLUMNS = ("patient_id", "eligible")


@dataclass(frozen=True)
class HeldOutStratification:
    """`strata` maps patient_id -> arm (only patients in a stratified arm); `arms` is the
    priority order; `share` the held-out share each arm must reach; `held_out` the
    partitions outside pretraining."""

    strata: Mapping[str, str]
    arms: tuple[str, ...]
    share: float
    held_out: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 < float(self.share) < 1.0:
            raise ValueError("held-out share must lie strictly between 0 and 1")
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("stratified arms must be a non-empty list of distinct names")
        unknown = sorted(set(self.strata.values()) - set(self.arms))
        if unknown:
            raise ValueError(f"strata name arm(s) {unknown} outside {list(self.arms)}")


def strata_from_cohort(cohort: pl.DataFrame, *, arm_column: str, arms: Sequence[str]) -> dict[str, str]:
    """patient_id -> arm for ELIGIBLE cohort patients whose `arm_column` is a stratified
    arm. Selects only `patient_id`, `eligible` and `arm_column` before anything else."""
    columns = [*STRATA_ID_COLUMNS, arm_column]
    missing = [c for c in columns if c not in cohort.columns]
    if missing:
        raise ValueError(f"strata source missing column(s): {', '.join(missing)}")
    frame = cohort.select(columns).filter(pl.col("eligible").fill_null(False)
                                          & pl.col(arm_column).is_in(list(arms)))
    order = {arm: i for i, arm in enumerate(arms)}
    strata: dict[str, str] = {}
    for patient, arm in frame.select("patient_id", arm_column).iter_rows():
        current = strata.get(str(patient))
        if current is None or order[arm] < order[current]:
            strata[str(patient)] = arm
    return strata


def load_held_out_strata(path: str | Path, *, arm_column: str, arms: Sequence[str]) -> dict[str, str]:
    """Read ONLY the outcome-blind columns of an extubation cohort artifact."""
    return strata_from_cohort(
        pl.read_parquet(path, columns=[*STRATA_ID_COLUMNS, arm_column]), arm_column=arm_column, arms=arms)


def held_out_stratification(contract: Mapping, strata: Mapping[str, str] | None) -> HeldOutStratification | None:
    """The `data_contract.held_out_stratification` block as an object, or None when it is
    disabled. Fails closed: enabled without strata, or strata without the option."""
    block = contract.get("held_out_stratification") or {}
    enabled = bool(block.get("enabled", False))
    if not enabled:
        if strata is not None:
            raise ValueError("held-out strata were given but data_contract.held_out_stratification "
                             "is not enabled")
        return None
    if strata is None:
        raise ValueError("data_contract.held_out_stratification is enabled: pass the extubation "
                         "cohort artifact as the strata source")
    partitions = list(contract["partitions"])
    held_out = tuple(block.get("held_out_partitions") or ())
    if not held_out or not set(held_out) < set(partitions):
        raise ValueError("held_out_partitions must be a proper subset of the partitions")
    return HeldOutStratification(strata=dict(strata), arms=tuple(block["arms"]),
                                 share=float(block["held_out_share"]), held_out=held_out)


def _validate_identifiers(rows: pl.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if column not in rows.columns:
            continue
        if rows.schema[column] != pl.String:
            raise ValueError(f"{column} must be a string identifier")
        if rows[column].has_nulls():
            raise ValueError(f"{column} contains null identifiers")


def _components(rows: list[dict]) -> list[list[int]]:
    parent = list(range(len(rows)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        keys = [("patient", str(row["patient_id"]))]
        linked = row.get("hospitalization_joined_id")
        if linked is not None:
            keys.append(("linked", str(linked)))
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def _unit(seed: int, purpose: str, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{purpose}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_grouped_splits(
    episodes: pl.DataFrame,
    ratios: Mapping[str, float],
    *,
    seed: int,
    held_out: HeldOutStratification | None = None,
) -> pl.DataFrame:
    """Assign connected patient/linkage groups with a stable content hash; with
    `held_out`, rebalance the stratified arms into the held-out partitions (module
    docstring)."""
    required = {"hospitalization_id", "patient_id"}
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"episodes missing split keys: {', '.join(sorted(missing))}")
    _validate_identifiers(
        episodes, ["hospitalization_id", "patient_id"]
    )
    if not ratios or any(value <= 0 for value in ratios.values()):
        raise ValueError("split ratios must all be positive")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {total}")

    rows = episodes.to_dicts()
    assignments: dict[int, str] = {}
    names = list(ratios)
    thresholds = []
    cumulative = 0.0
    for name in names:
        cumulative += ratios[name]
        thresholds.append((cumulative, name))
    groups = []
    for component in _components(rows):
        stable_ids = sorted(
            {f"patient:{rows[index]['patient_id']}" for index in component}
            | {
                f"linked:{rows[index]['hospitalization_joined_id']}"
                for index in component
                if rows[index].get("hospitalization_joined_id") is not None
            }
        )
        digest = hashlib.sha256(f"{seed}:{'|'.join(stable_ids)}".encode()).digest()
        score = int.from_bytes(digest[:8], "big") / 2**64
        partition = next(name for threshold, name in thresholds if score < threshold)
        groups.append({"members": component, "key": "|".join(stable_ids), "partition": partition,
                       "patients": {str(rows[index]["patient_id"]) for index in component}})
    if held_out is not None:
        _rebalance(groups, ratios, held_out, seed)
    for group in groups:
        for index in group["members"]:
            assignments[index] = group["partition"]
    return episodes.with_columns(
        pl.Series("partition", [assignments[index] for index in range(len(rows))])
    )


def _rebalance(groups: list[dict], ratios: Mapping[str, float], spec: HeldOutStratification,
               seed: int) -> None:
    """Move stratified groups so each arm reaches its held-out share, then move as many
    unstratified groups the other way, per partition, so every partition keeps its size."""
    unknown = sorted(set(spec.held_out) - set(ratios))
    pretraining = [name for name in ratios if name not in spec.held_out]
    if unknown or len(pretraining) != 1:
        raise ValueError("held-out partitions must be partitions, leaving exactly one pretraining partition")
    train = pretraining[0]
    held_weights = [(name, ratios[name]) for name in spec.held_out]
    held_total = sum(weight for _, weight in held_weights)

    def held_partition(group: dict) -> str:
        u = _unit(seed, "held_out_partition", group["key"]) * held_total
        cumulative = 0.0
        for name, weight in held_weights:
            cumulative += weight
            if u < cumulative:
                return name
        return held_weights[-1][0]

    order = {arm: i for i, arm in enumerate(spec.arms)}
    for group in groups:
        arms = [spec.strata[p] for p in group["patients"] if p in spec.strata]
        group["stratum"] = min(arms, key=order.__getitem__) if arms else None
        group["move_rank"] = _unit(seed, "held_out_rebalance", group["key"])
    delta = {name: 0 for name in spec.held_out}        # net groups moved INTO each partition
    for arm in spec.arms:
        members = sorted((g for g in groups if g["stratum"] == arm),
                         key=lambda g: (g["move_rank"], g["key"]))
        target = round(spec.share * len(members))
        inside = [g for g in members if g["partition"] in spec.held_out]
        if len(inside) < target:
            for group in [g for g in members if g["partition"] == train][:target - len(inside)]:
                group["partition"] = held_partition(group)
                delta[group["partition"]] += 1
        else:
            for group in inside[:len(inside) - target]:
                delta[group["partition"]] -= 1
                group["partition"] = train
    others = sorted((g for g in groups if g["stratum"] is None), key=lambda g: (g["move_rank"], g["key"]))
    for name, moved in delta.items():
        if moved > 0:
            pool = [g for g in others if g["partition"] == name][:moved]
            target_partition = train
        else:
            pool = [g for g in others if g["partition"] == train][:-moved]
            target_partition = name
        if len(pool) < abs(moved):
            raise ValueError(f"not enough unstratified patients to keep partition {name!r} at its size")
        for group in pool:
            group["partition"] = target_partition


def held_out_shares(rows: pl.DataFrame, strata: Mapping[str, str], held_out: Sequence[str]) -> dict[str, dict]:
    """Aggregate per stratified arm: patients and the share in the held-out partitions.
    Counts only; for logs and the split freeze, never row-level."""
    out: dict[str, dict] = {}
    frame = rows.select("patient_id", "partition").unique()
    lookup = dict(frame.iter_rows())
    for arm in sorted(set(strata.values())):
        patients = [p for p, a in strata.items() if a == arm and p in lookup]
        inside = sum(lookup[p] in held_out for p in patients)
        out[arm] = {"patients": len(patients),
                    "held_out_share": round(inside / len(patients), 4) if patients else None}
    return out


def validate_required_partitions(rows: pl.DataFrame, required: Sequence[str]) -> None:
    present = set(rows["partition"].drop_nulls().to_list()) if rows.height else set()
    missing = [name for name in required if name not in present]
    if missing:
        raise ValueError(f"required partitions have zero eligible episodes: {', '.join(missing)}")


def validate_training_targets(
    labels: pl.DataFrame,
    objectives: Sequence[str],
    *,
    partition: str = "train",
) -> None:
    """Reject enabled objectives with no eligible binary targets in training."""
    training = fit_partition(labels, partition)
    empty = [
        objective
        for objective in objectives
        if objective not in training.columns or training[objective].drop_nulls().is_empty()
    ]
    if empty:
        raise ValueError(f"enabled objectives have zero eligible training targets: {', '.join(empty)}")


def fit_partition(rows: pl.DataFrame, partition: str = "train") -> pl.DataFrame:
    if "partition" not in rows.columns:
        raise ValueError("partition column is required before fitting artifacts")
    selected = rows.filter(pl.col("partition") == partition)
    if selected.is_empty():
        raise ValueError(f"artifact fit partition {partition!r} has zero rows")
    return selected


def content_manifest(rows: pl.DataFrame, *, columns: Sequence[str]) -> dict[str, object]:
    """Hash sorted non-identifier artifact content without exposing row values."""
    selected = rows.select(columns).sort(columns)
    payload = json.dumps(selected.to_dicts(), sort_keys=True, separators=(",", ":"), default=str)
    return {
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "row_count": selected.height,
        "columns": list(columns),
    }


def validate_grouped_splits(rows: pl.DataFrame) -> None:
    """Reject split artifacts that leak a patient or linked chain across partitions."""
    required = {"hospitalization_id", "patient_id", "partition"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"split artifact missing columns: {', '.join(sorted(missing))}")
    _validate_identifiers(
        rows, ["hospitalization_id", "patient_id", "partition"]
    )
    for column in ["patient_id", "hospitalization_joined_id"]:
        if column not in rows.columns:
            continue
        if rows[column].null_count() == rows.height:
            continue
        leaked = rows.group_by(column).agg(pl.col("partition").n_unique().alias("n")).filter(
            pl.col("n") != 1
        )
        if leaked.height:
            raise ValueError(f"{column} occurs in multiple partitions")
