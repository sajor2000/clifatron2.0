"""Deterministic patient- and linked-encounter-grouped partitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import polars as pl


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


def assign_grouped_splits(
    episodes: pl.DataFrame,
    ratios: Mapping[str, float],
    *,
    seed: int,
) -> pl.DataFrame:
    """Assign connected patient/linkage groups with a stable content hash."""
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
        for index in component:
            assignments[index] = partition
    return episodes.with_columns(
        pl.Series("partition", [assignments[index] for index in range(len(rows))])
    )


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
