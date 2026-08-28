"""Derive explicit incident physiologic outcome states from local CLIF 2.1 tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.data.cohort import (
    QualificationError,
    derive_outcome_states,
    validate_artifact_destination,
    validate_episode_artifact,
)

ROOT = Path(__file__).parents[2]
DEFAULT_COHORT_CONFIG = ROOT / "configs/cohort.yaml"
DEFAULT_DATA_CONFIG = ROOT / "configs/data.yaml"


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _outcome_events(base: Path, table_spec: dict[str, Any]) -> pl.DataFrame | None:
    path = base / f"{table_spec['file']}.parquet"
    if not path.exists():
        return None
    frame = pl.read_parquet(path)
    required = {
        "hospitalization_id",
        table_spec["availability_col"],
        table_spec["concept_col"],
        table_spec["value_col"],
    }
    missing = required - set(frame.columns)
    if missing:
        raise QualificationError(
            f"{table_spec['file']} is missing required columns: {', '.join(sorted(missing))}"
        )
    unit_col = table_spec.get("unit_col")
    if frame.schema["hospitalization_id"] != pl.String:
        raise QualificationError(
            f"{table_spec['file']}.hospitalization_id must be a string identifier"
        )
    if frame["hospitalization_id"].has_nulls():
        raise QualificationError(
            f"{table_spec['file']}.hospitalization_id contains null identifiers"
        )
    return frame.select(
        pl.col("hospitalization_id"),
        pl.col(table_spec["availability_col"]).alias("dttm"),
        pl.col(table_spec["concept_col"]).alias("concept"),
        pl.col(table_spec["value_col"]).cast(pl.Float64).alias("value"),
        (
            pl.col(unit_col).cast(pl.String)
            if unit_col
            else pl.lit("").cast(pl.String)
        ).alias("unit"),
    )


def auto_label(
    data_dir: str,
    episode_artifact: str | Path,
    outcomes: list[str] | None = None,
    *,
    cohort_config: str | Path = DEFAULT_COHORT_CONFIG,
    data_config: str | Path = DEFAULT_DATA_CONFIG,
) -> pl.DataFrame:
    """Build eligible episodes and preserve non-binary outcome states."""
    base = Path(data_dir)
    cohort_cfg = _load_yaml(cohort_config)
    data_cfg = _load_yaml(data_config)
    episodes = pl.read_parquet(episode_artifact)
    validate_episode_artifact(episodes)
    episodes = episodes.filter(pl.col("eligible"))

    outcome_specs = cohort_cfg["outcomes"]
    names = outcomes or list(outcome_specs)
    unknown = sorted(set(names) - set(outcome_specs))
    if unknown:
        raise QualificationError(f"outcomes are absent from the frozen contract: {', '.join(unknown)}")

    result = episodes.select("hospitalization_id", "partition")
    source_cache: dict[str, pl.DataFrame | None] = {}
    for name in names:
        spec = {"name": name, **outcome_specs[name]}
        source = spec["source"]
        table_spec = data_cfg["tables"].get(source)
        if table_spec is None or table_spec.get("input_only"):
            raise QualificationError(f"outcome {name} has an invalid source: {source}")
        if source not in source_cache:
            source_cache[source] = _outcome_events(base, table_spec)
        observations = source_cache[source]
        states = derive_outcome_states(
            episodes,
            observations,
            spec,
            source_available=observations is not None,
        ).select(
            "hospitalization_id",
            pl.col("label").alias(name),
            pl.col("status").alias(f"{name}_status"),
            pl.col("event_time_hours").alias(f"{name}_event_time_hours"),
            pl.col("time_from_anchor_hours").alias(f"{name}_time_from_anchor_hours"),
        )
        result = result.join(states, on="hospitalization_id", how="left")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="directory containing CLIF 2.1 parquet files")
    parser.add_argument(
        "--out",
        default="output/intermediate_phi/labels.parquet",
        help="site-local patient-level Parquet output",
    )
    parser.add_argument("--outcomes", nargs="*", help="subset from configs/cohort.yaml")
    parser.add_argument("--episodes", required=True, help="canonical episode/split artifact")
    parser.add_argument("--cohort-config", default=str(DEFAULT_COHORT_CONFIG))
    parser.add_argument("--artifact-policy", default=str(ROOT / "configs/artifact_policy.yaml"))
    args = parser.parse_args()

    output = Path(args.out)
    policy = _load_yaml(args.artifact_policy)
    validate_artifact_destination(output, "patient_level_phi", policy)
    labels = auto_label(
        args.data, args.episodes, args.outcomes or None, cohort_config=args.cohort_config
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    labels.write_parquet(output)
    print(f"Wrote {len(labels):,} locally governed outcome rows")


if __name__ == "__main__":
    main()
