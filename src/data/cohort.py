"""Versioned ICU episode and incident physiologic outcome contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.data.splits import (
    assign_grouped_splits,
    content_manifest,
    held_out_shares,
    held_out_stratification,
    load_held_out_strata,
    validate_grouped_splits,
    validate_required_partitions,
)


class QualificationError(ValueError):
    """The source cannot safely produce the declared cohort contract."""


def load_cohort_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _flat_config(config: dict[str, Any]) -> dict[str, Any]:
    episode = config.get("episode", {})
    anchor = config.get("anchor", {})
    windows = config.get("windows", {})
    prediction = windows.get("prediction", {})
    return {
        "anchor_hours": config.get("anchor_hours", anchor.get("hours_after_icu_admission", 24)),
        "prediction_horizon_hours": config.get(
            "prediction_horizon_hours", prediction.get("horizon_hours", 48)
        ),
        "minimum_age": config.get("minimum_age", episode.get("minimum_age", 18)),
        "icu_location_category": config.get(
            "icu_location_category", episode.get("icu_location_category", "icu")
        ),
    }


def _require_columns(frame: pl.DataFrame, name: str, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise QualificationError(f"{name} is missing required columns: {', '.join(missing)}")


def _require_string(frame: pl.DataFrame, name: str, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        dtype = frame.schema[column]
        # An all-null optional column is typed Null by polars, not String — accept it
        # (nullability is enforced separately for the truly required identifiers).
        if dtype == pl.Null and frame[column].null_count() == frame.height:
            continue
        if dtype != pl.String:
            raise QualificationError(f"{name}.{column} must be a string identifier")


def _reject_null_identifiers(frame: pl.DataFrame, name: str, columns: list[str],
                              *,
                              allow_all_null: set[str] | None = None) -> None:
    allow_all_null = allow_all_null or set()
    for column in columns:
        if column in frame.columns and frame[column].has_nulls():
            if column in allow_all_null and frame[column].null_count() == frame.height:
                continue
            raise QualificationError(f"{name}.{column} contains null identifiers")


def _require_utc(frame: pl.DataFrame, name: str, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        dtype = frame.schema[column]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise QualificationError(f"{name}.{column} must be timezone-aware UTC")


def build_cohort(
    hospitalization: pl.DataFrame,
    adt: pl.DataFrame,
    config: dict[str, Any],
    *,
    return_waterfall: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, int]]:
    """Build one first-ICU episode per patient without dropping exclusion states."""
    cfg = _flat_config(config)
    _require_columns(
        hospitalization,
        "hospitalization",
        {
            "hospitalization_id",
            "patient_id",
            "hospitalization_joined_id",
            "admission_dttm",
            "discharge_dttm",
            "age_at_admission",
            "discharge_category",
        },
    )
    _require_columns(
        adt,
        "adt",
        {"hospitalization_id", "in_dttm", "out_dttm", "location_category"},
    )
    _require_string(
        hospitalization,
        "hospitalization",
        ["hospitalization_id", "patient_id", "hospitalization_joined_id"],
    )
    _require_string(adt, "adt", ["hospitalization_id"])
    _require_string(hospitalization, "hospitalization", ["hospital_id"])
    _require_string(adt, "adt", ["hospital_id"])
    # hospitalization_joined_id (CLIF `linked_encounter_identifier`) is OPTIONAL: it links
    # hospitalizations across transfers and is legitimately all-null in sources without
    # inter-hospital linkage (e.g. MIMIC-IV-Ext-CLIF). Require its *type* when present,
    # but do NOT reject nulls — only the true primary identifiers must be non-null.
    _reject_null_identifiers(
        hospitalization,
        "hospitalization",
        ["hospitalization_id", "patient_id"],
    )
    _reject_null_identifiers(adt, "adt", ["hospitalization_id"])
    _reject_null_identifiers(hospitalization, "hospitalization", ["hospital_id"])
    _reject_null_identifiers(adt, "adt", ["hospital_id"])
    _require_utc(hospitalization, "hospitalization", ["admission_dttm", "discharge_dttm"])
    _require_utc(adt, "adt", ["in_dttm", "out_dttm"])
    if hospitalization["hospitalization_id"].n_unique() != hospitalization.height:
        raise QualificationError("hospitalization_id must be unique in hospitalization")

    hospital_sources = [
        frame["hospital_id"].drop_nulls().cast(pl.String)
        for frame in (hospitalization, adt)
        if "hospital_id" in frame.columns
    ]
    if hospital_sources:
        hospitals = pl.concat(hospital_sources).n_unique()
        if hospitals != 1:
            raise QualificationError("cohort input must contain exactly one hospital")

    # Strip malformed rows before ICU-interval logic so a small number of real-world
    # data anomalies (null/invalid out_dttm on MIMIC-CLIF ADT rows) do not abort the
    # entire cohort build. The excluded rows are logged in the waterfall.
    adt = adt.with_columns(
        (pl.col("in_dttm").is_null() | pl.col("out_dttm").is_null() | (pl.col("out_dttm") <= pl.col("in_dttm"))).alias(
            "_invalid_interval"
        )
    )
    n_adt_invalid = adt.filter("_invalid_interval").height
    adt = adt.filter(~pl.col("_invalid_interval")).drop("_invalid_interval")

    icu = adt.filter(pl.col("location_category") == cfg["icu_location_category"])
    if icu.is_empty():
        raise QualificationError("adt contains no ICU intervals")

    icu = icu.sort(["hospitalization_id", "in_dttm", "out_dttm"]).with_columns(
        pl.col("out_dttm")
        .cum_max()
        .shift(1)
        .over("hospitalization_id")
        .alias("prior_max_out_dttm")
    ).with_columns(
        (
            pl.col("prior_max_out_dttm").is_null()
            | (pl.col("in_dttm") > pl.col("prior_max_out_dttm"))
        )
        .cast(pl.Int64)
        .cum_sum()
        .over("hospitalization_id")
        .alias("icu_interval_group")
    )
    coalesced = icu.group_by("hospitalization_id", "icu_interval_group").agg(
        pl.col("in_dttm").min().alias("icu_admit_dttm"),
        pl.col("out_dttm").max().alias("icu_out_dttm"),
    )
    first_icu = coalesced.sort(
        ["hospitalization_id", "icu_admit_dttm", "icu_out_dttm"]
    ).unique(subset=["hospitalization_id"], keep="first", maintain_order=True).drop(
        "icu_interval_group"
    )

    episodes = hospitalization.join(first_icu, on="hospitalization_id", how="inner").sort(
        ["patient_id", "icu_admit_dttm", "hospitalization_id"]
    )
    anchor_delta = pl.duration(hours=int(cfg["anchor_hours"]))
    horizon_delta = pl.duration(hours=int(cfg["prediction_horizon_hours"]))
    episodes = episodes.with_columns(
        (pl.col("icu_admit_dttm") + anchor_delta).alias("anchor_dttm")
    ).with_columns(
        (pl.col("anchor_dttm") + horizon_delta).alias("prediction_end_dttm")
    )
    # A stay with no discharge time is still open at the data cut: the caller censors it
    # at the site's declared extraction time (`build_cohort_artifact`); one still null here
    # (no extraction time declared) cannot be followed and is never eligible, with its
    # own status (eligibility and status always agree; a null never reads as eligible).
    eligible = (
        (pl.col("age_at_admission") >= cfg["minimum_age"])
        & (pl.col("icu_out_dttm") > pl.col("anchor_dttm"))
        & (pl.col("discharge_dttm") > pl.col("anchor_dttm"))
    ).fill_null(False)
    status = (
        pl.when(pl.col("age_at_admission").is_null()).then(pl.lit("missing_age"))
        .when(pl.col("age_at_admission") < cfg["minimum_age"]).then(pl.lit("underage"))
        .when(pl.col("discharge_dttm").is_null()).then(pl.lit("open_stay_no_extraction_time"))
        .when(pl.col("discharge_dttm") <= pl.col("anchor_dttm")).then(pl.lit("not_observed_at_anchor"))
        .when(pl.col("icu_out_dttm") <= pl.col("anchor_dttm")).then(pl.lit("not_in_icu_at_anchor"))
        .otherwise(pl.lit("eligible"))
    )
    episodes = episodes.with_columns(eligible.alias("eligible"), status.alias("eligibility_status"))
    all_episodes = episodes
    first_eligible = all_episodes.filter(pl.col("eligible")).unique(
        subset=["patient_id"], keep="first", maintain_order=True
    )
    no_eligible = all_episodes.join(
        first_eligible.select("patient_id"), on="patient_id", how="anti"
    ).unique(subset=["patient_id"], keep="first", maintain_order=True)
    episodes = pl.concat([first_eligible, no_eligible], how="vertical_relaxed").sort(
        ["patient_id", "icu_admit_dttm", "hospitalization_id"]
    )

    death = pl.col("discharge_category").str.to_lowercase() == "expired"
    terminal_time = pl.min_horizontal("icu_out_dttm", "discharge_dttm", "prediction_end_dttm")
    episodes = episodes.with_columns(
        terminal_time.alias("followup_end_dttm")
    ).with_columns(
        pl.when(
            death
            & (pl.col("discharge_dttm") == pl.col("followup_end_dttm"))
            & (pl.col("discharge_dttm") <= pl.col("prediction_end_dttm"))
        )
        .then(pl.lit("death"))
        .when(
            (pl.col("discharge_dttm") == pl.col("followup_end_dttm"))
            & (pl.col("discharge_dttm") < pl.col("prediction_end_dttm"))
        )
        .then(pl.lit("discharge"))
        .when(
            (pl.col("icu_out_dttm") == pl.col("followup_end_dttm"))
            & (pl.col("icu_out_dttm") < pl.col("prediction_end_dttm"))
        )
        .then(pl.lit("icu_transfer"))
        .otherwise(pl.lit("complete_followup"))
        .alias("terminal_event"),
    )

    source_patients = hospitalization["patient_id"].n_unique()
    patients_with_icu = all_episodes["patient_id"].n_unique()
    waterfall = {
        "episode_source": hospitalization.height,
        "episode_excluded_no_icu": hospitalization.height - all_episodes.height,
        "episode_excluded_invalid_adt": n_adt_invalid,
        "episode_candidates": all_episodes.height,
        "episode_excluded_non_index": all_episodes.height - episodes.height,
        "episode_selected": episodes.height,
        "patient_source": source_patients,
        "patient_excluded_no_icu": source_patients - patients_with_icu,
    }
    for value in ["underage", "missing_age", "open_stay_no_extraction_time",
                  "not_observed_at_anchor", "not_in_icu_at_anchor"]:
        waterfall[f"episode_excluded_{value}"] = all_episodes.filter(
            pl.col("eligibility_status") == value
        ).height
        waterfall[f"patient_excluded_{value}"] = episodes.filter(
            pl.col("eligibility_status") == value
        ).height
    waterfall["episode_eligible_at_anchor"] = all_episodes.filter(pl.col("eligible")).height
    waterfall["patient_eligible"] = episodes.filter(pl.col("eligible")).height
    return (episodes, waterfall) if return_waterfall else episodes


def derive_outcome_states(
    episodes: pl.DataFrame,
    observations: pl.DataFrame | None,
    spec: dict[str, Any],
    *,
    source_available: bool = True,
) -> pl.DataFrame:
    """Return explicit incident-outcome state and admission-relative event/censor time."""
    eligible = episodes.filter(pl.col("eligible"))
    rows: list[dict[str, Any]] = []
    empty_schema = {
        "hospitalization_id": pl.String,
        "outcome": pl.String,
        "status": pl.String,
        "label": pl.Boolean,
        "event_time_hours": pl.Float64,
        "time_from_anchor_hours": pl.Float64,
    }
    if eligible.is_empty():
        return pl.DataFrame(schema=empty_schema)
    if not source_available or observations is None:
        return pl.DataFrame(
            {
                "hospitalization_id": eligible["hospitalization_id"],
                "outcome": [spec["name"]] * eligible.height,
                "status": ["unsupported_at_site"] * eligible.height,
                "label": pl.Series([None] * eligible.height, dtype=pl.Boolean),
                "event_time_hours": pl.Series([None] * eligible.height, dtype=pl.Float64),
                "time_from_anchor_hours": pl.Series([None] * eligible.height, dtype=pl.Float64),
            }
        )

    _require_columns(
        observations,
        "outcome observations",
        {"hospitalization_id", "dttm", "concept", "value", "unit"},
    )
    _require_string(observations, "outcome observations", ["hospitalization_id"])
    _require_utc(observations, "outcome observations", ["dttm"])
    direction = spec["direction"]
    if direction not in {"above", "below"}:
        raise QualificationError(f"invalid outcome direction: {direction}")
    concept_events = observations.filter(pl.col("concept") == spec["concept"])
    if concept_events.filter(pl.col("unit").is_not_null()).height:
        wrong_units = concept_events.filter(
            pl.col("unit").is_not_null() & (pl.col("unit") != spec["unit"])
        )
        if wrong_units.height:
            raise QualificationError(f"non-canonical unit for outcome {spec['name']}")

    concept_events = concept_events.filter(pl.col("value").is_not_null())
    minimum = int(spec.get("minimum_post_anchor_measurements", 1))
    threshold = float(spec["threshold"])
    for episode in eligible.iter_rows(named=True):
        stay_events = concept_events.filter(
            pl.col("hospitalization_id") == episode["hospitalization_id"]
        ).sort("dttm")
        crosses = (
            pl.col("value") < threshold if direction == "below" else pl.col("value") > threshold
        )
        baseline_start = episode["anchor_dttm"] - timedelta(
            hours=float(spec.get("baseline_lookback_hours", 24))
        )
        baseline = stay_events.filter(
            (pl.col("dttm") >= baseline_start) & (pl.col("dttm") <= episode["anchor_dttm"])
        ).tail(1)
        prevalent = baseline.filter(crosses)
        future = stay_events.filter(
            (pl.col("dttm") > episode["anchor_dttm"])
            & (pl.col("dttm") <= episode["followup_end_dttm"])
        )
        incident = future.filter(crosses)
        event_dttm = None
        label = None
        last_measurement = future["dttm"].max() if future.height else None
        horizon_coverage = spec.get("required_measurement_within_hours_of_horizon")
        stale_measurement = (
            horizon_coverage is not None
            and last_measurement is not None
            and (episode["prediction_end_dttm"] - last_measurement).total_seconds() / 3600
            > float(horizon_coverage)
        )
        if prevalent.height:
            status = "prevalent"
        elif incident.height:
            status = "positive"
            label = True
            event_dttm = incident["dttm"].min()
        elif episode["followup_end_dttm"] < episode["prediction_end_dttm"]:
            status = "competing_event" if episode["terminal_event"] == "death" else "censored"
            event_dttm = episode["followup_end_dttm"]
        elif future.height < minimum or stale_measurement:
            status = "not_ascertainable"
        else:
            status = "negative"
            label = False
            event_dttm = episode["prediction_end_dttm"]
        rows.append(
            {
                "hospitalization_id": episode["hospitalization_id"],
                "outcome": spec["name"],
                "status": status,
                "label": label,
                "event_time_hours": (
                    (event_dttm - episode["icu_admit_dttm"]).total_seconds() / 3600
                    if event_dttm is not None
                    else None
                ),
                "time_from_anchor_hours": (
                    (event_dttm - episode["anchor_dttm"]).total_seconds() / 3600
                    if event_dttm is not None
                    else None
                ),
            }
        )
    return pl.DataFrame(rows, schema_overrides={"label": pl.Boolean})


def validate_artifact_destination(
    path: str | Path,
    classification: str,
    policy: dict[str, Any],
    *,
    for_export: bool = False,
) -> None:
    """Fail closed when an artifact violates its declared storage/export class."""
    try:
        rule = policy["classes"][classification]
    except KeyError as exc:
        raise ValueError(f"unknown artifact classification: {classification}") from exc
    destination = Path(os.path.normpath(path)).resolve()
    required = Path(rule["directory"]).resolve()
    if destination != required and required not in destination.parents:
        raise ValueError(f"{classification} artifacts must be stored under {required}")
    suffix = destination.suffix.removeprefix(".").lower()
    if suffix not in rule["formats"]:
        raise ValueError(f"{classification} artifacts must use {rule['formats']}")
    if for_export and not rule["export_allowed"]:
        raise ValueError(f"{classification} artifacts may not be exported")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cohort_artifact(
    data_dir: str | Path,
    output: str | Path,
    *,
    cohort_config: str | Path,
    train_config: str | Path,
    artifact_policy: str | Path,
    held_out_strata: str | Path | None = None,
    site: str = "mimic",
    data_config: str | Path | None = "configs/data.yaml",
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Build, validate, and persist the canonical patient-grouped episode artifact.

    `held_out_strata` is the extubation cohort artifact, required exactly when
    `data_contract.held_out_stratification.enabled` (item 46; src/data/splits.py): only its
    patient_id, eligible and device-arm columns are read (and its `site`, which must be
    `site`).

    The artifact is bound to `site` (a `site` column and the `site` of its manifest). The
    site profile (`data_config` `sites.<site>`, after the site-local merge;
    src/data/site_config.py) converts naive or non-UTC timestamps to UTC and censors a
    stay with no discharge time at the declared `extraction_dttm`."""
    from src.data.site_config import (
        SiteConfigError,
        censor_open_stays,
        site_profile,
        to_utc,
        with_site_local,
    )

    base = Path(data_dir)
    cohort_path = Path(cohort_config)
    train_path = Path(train_config)
    policy = yaml.safe_load(Path(artifact_policy).read_text())
    destination = Path(output)
    validate_artifact_destination(destination, "patient_level_phi", policy)
    cohort_cfg = load_cohort_config(cohort_path)
    train_cfg = yaml.safe_load(train_path.read_text())["data_contract"]
    hospitalization_path = base / f"{cohort_cfg['source_tables']['hospitalization']}.parquet"
    adt_path = base / f"{cohort_cfg['source_tables']['adt']}.parquet"
    for source in (hospitalization_path, adt_path):
        if not source.exists():
            raise QualificationError(f"required CLIF table is missing: {source.name}")
    try:
        data_cfg = (with_site_local(yaml.safe_load(Path(data_config).read_text()), site)
                    if data_config is not None and Path(data_config).exists() else {})
        profile = site_profile(data_cfg, site)
        hospitalization, nulled_h = to_utc(pl.read_parquet(hospitalization_path),
                                           ["admission_dttm", "discharge_dttm"],
                                           profile["site_timezone"])
        adt, nulled_a = to_utc(pl.read_parquet(adt_path), ["in_dttm", "out_dttm"],
                               profile["site_timezone"])
    except SiteConfigError as exc:
        raise QualificationError(str(exc)) from exc
    # A stay still in hospital at the data cut: censored at the extraction time.
    hospitalization, open_stays = censor_open_stays(hospitalization, profile["extraction_dttm"])
    episodes, waterfall = build_cohort(
        hospitalization,
        adt,
        cohort_cfg,
        return_waterfall=True,
    )
    waterfall["timestamps_nulled_dst"] = nulled_h + nulled_a
    waterfall["stays_open_censored_at_extraction"] = (
        open_stays if profile["extraction_dttm"] is not None else 0)
    block = train_cfg.get("held_out_stratification") or {}
    strata = None
    if held_out_strata is not None:
        if not block.get("enabled"):
            raise QualificationError("--held-out-strata given but data_contract.held_out_stratification "
                                     "is not enabled in the train config")
        # The strata must come from THIS site's extubation cohort.
        strata_columns = pl.read_parquet_schema(held_out_strata)
        if "site" in strata_columns:
            strata_sites = pl.read_parquet(held_out_strata, columns=["site"])["site"]
            recorded = strata_sites.drop_nulls().unique().to_list()
            if recorded != [site]:
                raise QualificationError(
                    f"--held-out-strata is an extubation cohort of site(s) {recorded}, not "
                    f"{site!r}: stratify each site with its own cohort")
        elif site != "mimic":
            raise QualificationError("--held-out-strata records no site; rebuild it")
        strata = load_held_out_strata(held_out_strata, arm_column=block["arm_column"], arms=block["arms"])
    try:
        stratification = held_out_stratification(train_cfg, strata)
    except ValueError as exc:
        raise QualificationError(str(exc)) from exc
    eligible = assign_grouped_splits(
        episodes.filter(pl.col("eligible")), train_cfg["partitions"], seed=train_cfg["split_seed"],
        held_out=stratification,
    )
    validate_grouped_splits(eligible)
    validate_required_partitions(eligible, train_cfg["required_partitions"])
    episodes = episodes.join(
        eligible.select("hospitalization_id", "partition"), on="hospitalization_id", how="left"
    )
    split = content_manifest(
        eligible, columns=["hospitalization_id", "patient_id", "partition"]
    )
    manifest = {
        "artifact_family": "canonical_episode_split",
        "site": site,
        "contract_version": cohort_cfg["contract_version"],
        "clif_version": cohort_cfg["clif_version"],
        "mcide_version": cohort_cfg["mcide_version"],
        "split_seed": train_cfg["split_seed"],
        "split_sha256": split["sha256"],
        "episode_sha256": content_manifest(
            episodes,
            columns=["hospitalization_id", "patient_id", "eligible", "partition"],
        )["sha256"],
        "source_provenance": {
            hospitalization_path.stem: _file_sha256(hospitalization_path),
            adt_path.stem: _file_sha256(adt_path),
        },
        "waterfall": waterfall,
        "held_out_stratification": None if stratification is None else {
            "arms": list(stratification.arms), "held_out_share": stratification.share,
            "held_out_partitions": list(stratification.held_out),
            "achieved": held_out_shares(eligible, stratification.strata, stratification.held_out),
        },
    }
    episodes = episodes.with_columns(
        pl.lit(site).alias("site"),
        pl.lit(manifest["contract_version"]).alias("cohort_contract_version"),
        pl.lit(manifest["split_sha256"]).alias("split_sha256"),
        pl.lit(manifest["episode_sha256"]).alias("episode_sha256"),
        pl.lit(json.dumps(manifest["source_provenance"], sort_keys=True)).alias(
            "source_provenance_json"
        ),
        pl.lit(json.dumps(waterfall, sort_keys=True)).alias("waterfall_json"),
    )
    validate_episode_artifact(episodes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    episodes.write_parquet(destination)
    return episodes, manifest


def validate_episode_artifact(episodes: pl.DataFrame) -> None:
    """Validate the canonical artifact before any downstream join or write."""
    required = {
        "hospitalization_id",
        "patient_id",
        "hospitalization_joined_id",
        "icu_admit_dttm",
        "anchor_dttm",
        "eligible",
        "partition",
        "cohort_contract_version",
        "split_sha256",
        "episode_sha256",
        "source_provenance_json",
    }
    _require_columns(episodes, "episode artifact", required)
    _require_string(
        episodes,
        "episode artifact",
        [
            "hospitalization_id",
            "patient_id",
            "hospitalization_joined_id",
            "cohort_contract_version",
            "split_sha256",
            "episode_sha256",
            "source_provenance_json",
        ],
    )
    # hospitalization_joined_id is optional linkage: a null is a stay linked to nothing
    # (a singleton), so it may be null on any row, not only on all of them.
    _reject_null_identifiers(
        episodes,
        "episode artifact",
        ["hospitalization_id", "patient_id"],
    )
    if "site" in episodes.columns and episodes["site"].drop_nulls().n_unique() > 1:
        raise QualificationError("episode artifact names more than one site")
    _require_utc(episodes, "episode artifact", ["icu_admit_dttm", "anchor_dttm"])
    eligible = episodes.filter(pl.col("eligible"))
    if eligible["partition"].has_nulls():
        raise QualificationError("eligible episode artifact rows contain null partitions")
    try:
        validate_grouped_splits(eligible)
    except ValueError as exc:
        raise QualificationError(str(exc)) from exc
    split_hashes = episodes["split_sha256"].unique().to_list()
    episode_hashes = episodes["episode_sha256"].unique().to_list()
    if len(split_hashes) != 1 or len(episode_hashes) != 1:
        raise QualificationError("episode artifact contains inconsistent content hashes")
    expected_split = content_manifest(
        eligible, columns=["hospitalization_id", "patient_id", "partition"]
    )["sha256"]
    expected_episode = content_manifest(
        episodes, columns=["hospitalization_id", "patient_id", "eligible", "partition"]
    )["sha256"]
    if split_hashes[0] != expected_split or episode_hashes[0] != expected_episode:
        raise QualificationError("episode artifact content hash mismatch")


_ELIGIBILITY_STATUSES = ("underage", "missing_age", "open_stay_no_extraction_time",
                         "not_observed_at_anchor", "not_in_icu_at_anchor")
WATERFALL_SUPPRESSED = "suppressed"


def waterfall_equations(waterfall: Mapping[str, int]) -> list[list[str]]:
    """The cohort waterfall's published totals and the cells summing to them,
    ``[total, *cells]`` (`build_cohort`). Any ONE unknown member of an equation is
    recoverable from the others. One episode is kept per patient, so `episode_selected`
    is also the patients with an ICU stay."""
    status = [f"episode_excluded_{v}" for v in _ELIGIBILITY_STATUSES]
    patient = [f"patient_excluded_{v}" for v in _ELIGIBILITY_STATUSES]
    equations = [
        ["episode_source", "episode_excluded_no_icu", "episode_candidates"],
        ["episode_candidates", *status, "episode_eligible_at_anchor"],
        ["episode_candidates", "episode_excluded_non_index", "episode_selected"],
        ["patient_source", "patient_excluded_no_icu", "episode_selected"],
        ["episode_selected", *patient, "patient_eligible"],
    ]
    equations = [[key for key in equation if key in waterfall] for equation in equations]
    return [equation for equation in equations if len(equation) > 2]


def suppress_cohort_waterfall(waterfall: Mapping[str, Any], min_cell: int) -> dict[str, Any]:
    """The waterfall as printed on the node: every count of 1 to `min_cell - 1` becomes
    ``"<{min_cell}"``, and wherever an equation (`waterfall_equations`) would leave exactly
    ONE hidden non-zero member recoverable by subtraction from the published others, the
    smallest released non-zero member is withheld too (``"suppressed"``). A zero discloses
    nobody and is shown."""
    def count(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    hidden = {k for k, v in waterfall.items() if count(v) and 0 < v < min_cell}
    complementary: set[str] = set()
    equations = waterfall_equations({k: v for k, v in waterfall.items() if count(v)})
    changed = True
    while changed:
        changed = False
        for equation in equations:
            exposed = [k for k in equation if k in hidden]
            released = sorted((waterfall[k], k) for k in equation
                              if k not in hidden and waterfall[k] > 0)
            if len(exposed) == 1 and released:
                hidden.add(released[0][1])
                complementary.add(released[0][1])
                changed = True
    return {k: (WATERFALL_SUPPRESSED if k in complementary
                else f"<{min_cell}" if k in hidden else v)
            for k, v in waterfall.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical ICU episode/split artifact")
    parser.add_argument("--data", required=True)
    parser.add_argument("--site", required=True,
                        help="the site these CLIF tables belong to (recorded in the artifact; "
                             "its profile is configs/data.yaml sites.<site> plus the "
                             "site-local configs/sites/<site>.local.yaml)")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--out", default="output/intermediate_phi/episodes.parquet")
    parser.add_argument("--cohort-config", default="configs/cohort.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--artifact-policy", default="configs/artifact_policy.yaml")
    parser.add_argument("--held-out-strata", default=None,
                        help="extubation cohort artifact; required when data_contract."
                             "held_out_stratification is enabled (item 46)")
    args = parser.parse_args()
    _, manifest = build_cohort_artifact(
        args.data,
        args.out,
        cohort_config=args.cohort_config,
        train_config=args.train_config,
        artifact_policy=args.artifact_policy,
        held_out_strata=args.held_out_strata,
        site=args.site,
        data_config=args.data_config,
    )
    # Aggregate counts only (suppressed below the minimum cell size).
    policy = yaml.safe_load(Path(args.artifact_policy).read_text())
    min_cell = int(policy["classes"]["aggregate_no_phi"]["minimum_cell_size"])
    print(json.dumps({"site": args.site,
                      "waterfall": suppress_cohort_waterfall(manifest["waterfall"], min_cell)},
                     indent=2))
    if manifest.get("held_out_stratification"):
        # Aggregate shares only (patients per arm and held-out share).
        print(json.dumps({"held_out_stratification": manifest["held_out_stratification"]}, indent=2))


if __name__ == "__main__":
    main()
