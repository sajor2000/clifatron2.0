"""Leakage-safe extubation cohort, device arms and trial risk factors (per site).

Respiratory support is a model input and never a trunk prediction target (hard rule
#1). This module only reads it, to define the cohort (first extubation per patient)
and the exposure (device arm). It computes no outcome.

Row-level output is `patient_level_phi` (configs/artifact_policy.yaml). The path-based
builder and the CLI return only the waterfall counts:

    uv run python -m src.data.extubation_cohort --data <clif dir> --site mimic
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl
import yaml

from src.data.cohort import (
    QualificationError,
    _file_sha256,
    _reject_null_identifiers,
    _require_columns,
    _require_string,
    _require_utc,
    validate_artifact_destination,
    validate_episode_artifact,
)
from src.data.splits import assign_grouped_splits, content_manifest

INVASIVE = "invasive"
TRACHEOSTOMY = "tracheostomy"
UNMAPPED = "unmapped"
ASSIGNMENT_RULES = ("first_device", "highest_support")
COMORBIDITY_SOURCES = ("prior_hospitalization_codes", "index_stay_poa_codes", "index_stay_codes")
# Exclusion reasons in the order they are applied; the first that holds is recorded.
REASONS = (
    "missing_age",
    "underage",
    "lookback_insufficient",
    "lookforward_insufficient",
    "tracheostomy",
    "comfort_care",
    "do_not_reintubate",
    "ventilation_under_minimum",
)
HASH_COLUMNS = [
    "hospitalization_id",
    "patient_id",
    "time_zero_dttm",
    "eligible",
    "eligibility_status",
    "arm",
    "partition",
]
SUPPRESSED = "suppressed"


class Availability(NamedTuple):
    """Where a table lives and when its rows became knowable (configs/data.yaml)."""

    file: str
    column: str | None
    lag_minutes: int


class ExtubationCohort(NamedTuple):
    cohort: pl.DataFrame  # one row per patient with a first extubation
    device_rows: pl.DataFrame  # every device row from time zero through keep_hours
    waterfall: dict[str, int]


def load_extubation_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def table_availability(
    config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> dict[str, Availability]:
    """Resolve each source table's file and availability ordering from the data config."""
    resolved: dict[str, Availability] = {}
    tables = data_config.get("tables", {})
    for name, declaration in config["source_tables"].items():
        if "file" in declaration:
            resolved[name] = Availability(declaration["file"], None, 0)
            continue
        table = tables.get(declaration["data_table"])
        if table is None:
            raise QualificationError(
                f"{name}: data config declares no table {declaration['data_table']!r}"
            )
        if table.get("availability") is None or table.get("availability_col") is None:
            raise QualificationError(f"{name}: data config does not declare availability")
        lag = table.get("availability_lag_minutes", 0)
        if not isinstance(lag, int) or isinstance(lag, bool) or lag < 0:
            raise QualificationError(
                f"{name}: availability_lag_minutes must be a non-negative integer"
            )
        resolved[name] = Availability(table["file"], table["availability_col"], lag)
    return resolved


def _norm(value: str) -> str:
    return value.strip().lower()


def _norm_expr(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.String).str.strip_chars().str.to_lowercase()


def _device_classes(config: Mapping[str, Any]) -> dict[str, str]:
    """Normalized device category -> class (invasive, tracheostomy, or an arm)."""
    groups = {
        INVASIVE: config["detection"]["invasive_categories"],
        TRACHEOSTOMY: config["exclusions"]["tracheostomy"]["device_categories"],
        **config["arms"]["device_categories"],
    }
    classes: dict[str, str] = {}
    for name, categories in groups.items():
        for category in categories:
            if classes.setdefault(_norm(category), name) != name:
                raise QualificationError(
                    f"device category {category!r} maps to more than one class"
                )
    return classes


def _validate_config(config: Mapping[str, Any]) -> None:
    """Fail closed on a definitional choice this module does not implement."""
    expected = {
        ("time_zero", "covariate_boundary"): "strictly_before",
        ("time_zero", "first_extubation_per"): "patient",
        ("detection", "null_device_rows"): "ignore",
        ("detection", "unmapped_device_rows"): "ignore",
        ("grace_window", "right_closed"): True,
    }
    for (section, key), value in expected.items():
        if config[section].get(key) != value:
            raise QualificationError(f"{section}.{key} must be {value!r}")
    if config["exclusions"]["code_status"].get("missing") != "keep_and_flag":
        raise QualificationError("exclusions.code_status.missing must be 'keep_and_flag'")
    arms = config["arms"]
    if set(arms["order"]) != set(arms["device_categories"]):
        raise QualificationError("arms.order must list exactly the arms in arms.device_categories")
    if not {"niv", "hfnc"} <= set(arms["order"]):
        raise QualificationError(
            "arms must include 'niv' and 'hfnc' (the alternating rule names them)"
        )
    if arms["assignment"]["rule"] not in ASSIGNMENT_RULES:
        raise QualificationError(
            f"unknown arm assignment rule {arms['assignment']['rule']!r}; "
            f"expected one of {ASSIGNMENT_RULES}"
        )
    if arms["assignment"]["alternating_niv_hfnc"] not in arms["order"]:
        raise QualificationError("arms.assignment.alternating_niv_hfnc must name an arm")
    comorbidities = config["risk_factors"]["comorbidities"]
    for source in comorbidities["sources"]:
        if source not in COMORBIDITY_SOURCES or source not in comorbidities["available_sources"]:
            raise QualificationError(f"unknown comorbidity source {source!r}")
        leaky = comorbidities["available_sources"][source]["leaks_post_time_zero"]
        if leaky and not comorbidities.get("allow_leaky_sources", False):
            raise QualificationError(
                f"comorbidity source {source!r} leaks post-time-zero information; "
                "set allow_leaky_sources for a declared sensitivity analysis"
            )
    _device_classes(config)


def _period_source(config: Mapping[str, Any], site: str) -> dict[str, Any] | None:
    sites = config.get("sites", {})
    if site not in sites:
        raise QualificationError(f"site {site!r} is not declared in the extubation config")
    declaration = (sites[site] or {}).get("calendar_period_source")
    if declaration is not None and declaration.get("kind") != "patient_table":
        raise QualificationError(f"site {site!r}: unknown calendar_period_source kind")
    return declaration


def _table(tables: Mapping[str, pl.DataFrame], name: str, *, required: bool = False):
    frame = tables.get(name)
    if frame is None and required:
        raise QualificationError(f"required CLIF table is missing: {name}")
    return frame


def _available(
    frame: pl.DataFrame, name: str, availability: Availability, columns: set[str], key: str
) -> pl.DataFrame:
    """Validate a source table and add `_at`, the time each row became knowable."""
    _require_columns(frame, name, columns | {key, availability.column})
    _require_string(frame, name, [key])
    _reject_null_identifiers(frame, name, [key])
    _require_utc(frame, name, [availability.column])
    return frame.with_columns(
        (pl.col(availability.column) + timedelta(minutes=availability.lag_minutes)).alias("_at")
    ).filter(pl.col("_at").is_not_null())


def _latest_before(
    cohort: pl.DataFrame,
    rows: pl.DataFrame,
    *,
    on: str,
    values: list[str],
    lookback: timedelta | None = None,
) -> pl.DataFrame:
    """Per cohort row, the latest source row available STRICTLY before time zero."""
    joined = (
        cohort.select(on, "time_zero_dttm")
        .join(rows, on=on, how="inner")
        .filter(pl.col("_at") < pl.col("time_zero_dttm"))
    )
    if lookback is not None:
        joined = joined.filter(pl.col("_at") >= pl.col("time_zero_dttm") - lookback)
    return (
        joined.sort([on, "_at", *values])
        .unique(subset=[on], keep="last", maintain_order=True)
        .select(on, *values)
    )


def _hospitalization(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    frame = _table(tables, "hospitalization", required=True)
    _require_columns(
        frame,
        "hospitalization",
        {
            "hospitalization_id",
            "patient_id",
            "admission_dttm",
            "discharge_dttm",
            "age_at_admission",
        },
    )
    _require_string(
        frame, "hospitalization", ["hospitalization_id", "patient_id", "hospitalization_joined_id"]
    )
    _reject_null_identifiers(frame, "hospitalization", ["hospitalization_id", "patient_id"])
    _require_utc(frame, "hospitalization", ["admission_dttm", "discharge_dttm"])
    if frame["hospitalization_id"].n_unique() != frame.height:
        raise QualificationError("hospitalization_id must be unique in hospitalization")
    if "hospitalization_joined_id" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("hospitalization_joined_id"))
    return frame.select(
        "hospitalization_id",
        "patient_id",
        pl.col("hospitalization_joined_id").cast(pl.String),
        "admission_dttm",
        "discharge_dttm",
        "age_at_admission",
    )


def _device_rows(
    tables: Mapping[str, pl.DataFrame], config: Mapping[str, Any], availability: Availability
) -> pl.DataFrame:
    """Respiratory rows with their availability time, device class and trach flag."""
    resp = _table(tables, "respiratory_support", required=True)
    flag = config["exclusions"]["tracheostomy"]["flag_col"]
    resp = _available(
        resp, "respiratory_support", availability, {"device_category", flag}, "hospitalization_id"
    )
    _require_string(resp, "respiratory_support", ["device_category"])
    dtype = resp.schema[flag]
    if dtype == pl.Boolean:
        tracheostomy = pl.col(flag).fill_null(False)
    elif dtype.is_numeric():
        tracheostomy = (pl.col(flag) == 1).fill_null(False)
    elif dtype == pl.Null:
        tracheostomy = pl.lit(False)
    else:
        raise QualificationError(f"respiratory_support.{flag} must be boolean or 0/1")
    classes = _device_classes(config)
    category = _norm_expr("device_category")
    return resp.select(
        "hospitalization_id",
        pl.col("_at").alias("device_dttm"),
        pl.col("device_category").cast(pl.String),
        pl.when(category.is_not_null())
        .then(category.replace_strict(classes, default=UNMAPPED, return_dtype=pl.String))
        .alias("device_class"),
        tracheostomy.alias("_trach"),
    )


def _ordered(rows: pl.DataFrame) -> pl.DataFrame:
    """Deterministic device order; at equal times an invasive row sorts first."""
    return rows.sort(
        "hospitalization_id",
        "device_dttm",
        pl.col("device_class") != INVASIVE,
        "device_category",
    ).with_row_index("_row")


def _detect_extubations(
    devices: pl.DataFrame, config: Mapping[str, Any]
) -> tuple[pl.DataFrame, int]:
    """Every invasive -> non-invasive transition that is not a stitched ventilator gap."""
    stitch = timedelta(hours=float(config["detection"]["stitch_gap_hours"]))
    sequence = _ordered(
        devices.filter(pl.col("device_class").is_not_null() & (pl.col("device_class") != UNMAPPED))
    ).with_columns((pl.col("device_class") == INVASIVE).alias("_inv"))
    # Rows are sorted, so window expressions see each stay in time order. A run is a
    # maximal stretch of consecutive invasive (or consecutive non-invasive) rows.
    sequence = sequence.with_columns(
        pl.col("_inv").shift(1).over("hospitalization_id").alias("_prev_inv")
    ).with_columns(
        (pl.col("_inv") != pl.col("_prev_inv"))
        .fill_null(True)
        .cast(pl.Int32)
        .cum_sum()
        .over("hospitalization_id")
        .alias("_run")
    )
    runs = (
        sequence.group_by("hospitalization_id", "_run")
        .agg(
            pl.col("_inv").first(),
            pl.col("device_dttm").min().alias("_start"),
            pl.col("device_dttm").max().alias("_end"),
            pl.len().alias("_rows"),
            pl.col("device_category").sort_by("_row").first().alias("time_zero_device_category"),
            pl.col("device_class").sort_by("_row").first().alias("_time_zero_class"),
        )
        .sort("hospitalization_id", "_run")
        .with_columns(pl.col("_start").shift(-1).over("hospitalization_id").alias("_next_start"))
    )
    # Runs alternate, so a non-invasive run after the first follows an invasive run and
    # `_next_start` is the next IMV row. Back on IMV sooner than the stitch gap: the
    # non-invasive rows are a ventilator gap inside one continuous episode.
    after_imv = ~pl.col("_inv") & (pl.col("_run") > 1)
    runs = runs.with_columns(
        (
            after_imv
            & pl.col("_next_start").is_not_null()
            & ((pl.col("_next_start") - pl.col("_start")) < stitch)
        ).alias("_stitched")
    ).with_columns(
        (~pl.col("_inv") & ~pl.col("_stitched"))
        .cast(pl.Int32)
        .cum_sum()
        .over("hospitalization_id")
        .alias("_episode")
    )
    ventilation = (
        runs.filter(pl.col("_inv"))
        .group_by("hospitalization_id", "_episode")
        .agg(
            pl.col("_start").min().alias("imv_start_dttm"),
            pl.col("_end").max().alias("_last_invasive_dttm"),
            pl.col("_rows").sum().alias("n_invasive_rows"),
        )
    )
    events = (
        runs.filter(after_imv & ~pl.col("_stitched"))
        .with_columns(pl.col("_episode") - 1)
        .join(ventilation, on=["hospitalization_id", "_episode"], how="inner")
        .select(
            "hospitalization_id",
            pl.col("_start").alias("time_zero_dttm"),
            "imv_start_dttm",
            "_last_invasive_dttm",
            "n_invasive_rows",
            "time_zero_device_category",
            "_time_zero_class",
        )
    )
    return events, runs.filter(pl.col("_stitched")).height


def _hours(later: str, earlier: str) -> pl.Expr:
    return (pl.col(later) - pl.col(earlier)).dt.total_seconds() / 3600.0


def _assign_arms(
    cohort: pl.DataFrame, devices: pl.DataFrame, config: Mapping[str, Any], grace_hours: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Arm, rescue and escalation flags, plus every device row from time zero on."""
    order = list(config["arms"]["order"])
    rank = {arm: index for index, arm in enumerate(order)}
    assignment = config["arms"]["assignment"]
    grace = timedelta(hours=grace_hours)
    keep = timedelta(hours=float(config["device_rows"]["keep_hours"]))
    look_forward = timedelta(hours=float(config["detection"]["look_forward"]["window_hours"]))
    rescue_hours = float(config["rescue"]["look_forward_hours"])

    post = cohort.select("hospitalization_id", "patient_id", "time_zero_dttm").join(
        devices.filter(pl.col("device_class").is_not_null()), on="hospitalization_id", how="inner"
    )
    post = _ordered(
        post.filter(
            (pl.col("device_dttm") >= pl.col("time_zero_dttm"))
            & (pl.col("device_dttm") <= pl.col("time_zero_dttm") + keep)
            # An invasive row charted at time zero itself precedes the extubation row.
            & ~(
                (pl.col("device_class") == INVASIVE)
                & (pl.col("device_dttm") == pl.col("time_zero_dttm"))
            )
        )
    ).with_columns(
        _hours("device_dttm", "time_zero_dttm").alias("hours_from_time_zero"),
        (pl.col("device_dttm") <= pl.col("time_zero_dttm") + grace).alias("in_grace_window"),
        (
            (pl.col("device_class") == INVASIVE).cast(pl.Int32).cum_sum().over("hospitalization_id")
            > 0
        ).alias("after_invasive_row"),
        pl.col("device_class")
        .replace_strict(rank, default=None, return_dtype=pl.Int32)
        .alias("_rank"),
    )
    # Arm devices inside the grace window, before any return to IMV.
    in_grace = post.filter(
        pl.col("in_grace_window") & pl.col("_rank").is_not_null() & ~pl.col("after_invasive_row")
    )
    first = in_grace.unique(
        subset=["hospitalization_id"], keep="first", maintain_order=True
    ).select("hospitalization_id", pl.col("device_class").alias("_first_arm"))
    present = in_grace.group_by("hospitalization_id").agg(
        pl.col("_rank").max().alias("_max_rank"),
        (pl.col("device_class") == "niv").any().alias("_has_niv"),
        (pl.col("device_class") == "hfnc").any().alias("_has_hfnc"),
    )
    counts = post.group_by("hospitalization_id").agg(
        pl.col("in_grace_window").sum().cast(pl.Int64).alias("n_device_rows_in_grace"),
        (
            (pl.col("device_class") != INVASIVE)
            & ~pl.col("after_invasive_row")
            & (pl.col("device_dttm") <= pl.col("time_zero_dttm") + look_forward)
        )
        .sum()
        .cast(pl.Int64)
        .alias("_look_forward_rows"),
    )
    alternating = (
        pl.col("_first_arm").is_in(["niv", "hfnc"]) & pl.col("_has_niv") & pl.col("_has_hfnc")
    )
    arms = (
        first.join(present, on="hospitalization_id", how="left")
        .with_columns(
            pl.when(alternating)
            .then(pl.lit(assignment["alternating_niv_hfnc"]))
            .otherwise(pl.col("_first_arm"))
            .alias("arm_first_device"),
            pl.col("_max_rank")
            .replace_strict({index: arm for arm, index in rank.items()}, return_dtype=pl.String)
            .alias("arm_highest_support"),
        )
        .with_columns(pl.col(f"arm_{assignment['rule']}").alias("arm"))
        .with_columns(
            pl.col("arm")
            .replace_strict(rank, default=None, return_dtype=pl.Int32)
            .alias("_arm_rank")
        )
        .select("hospitalization_id", "arm", "arm_first_device", "arm_highest_support", "_arm_rank")
    )
    post = post.join(
        arms.select("hospitalization_id", "_arm_rank"), on="hospitalization_id", how="left"
    )
    escalation = (
        pl.col("_rank").is_not_null()
        & ~pl.col("after_invasive_row")
        & (pl.col("_rank") > pl.col("_arm_rank"))
    ).fill_null(False)
    post = post.with_columns(
        (
            escalation
            & ~pl.col("in_grace_window")
            & (pl.col("hours_from_time_zero") <= rescue_hours)
        ).alias("is_rescue"),
        (escalation & pl.col("in_grace_window")).alias("_escalated"),
    )
    rescue = (
        post.filter(pl.col("is_rescue"))
        .unique(subset=["hospitalization_id"], keep="first", maintain_order=True)
        .select(
            "hospitalization_id",
            pl.col("device_class").alias("rescue_arm"),
            pl.col("hours_from_time_zero").alias("rescue_hours_from_time_zero"),
        )
    )
    escalated = post.group_by("hospitalization_id").agg(
        pl.col("_escalated").any().alias("escalated_in_grace")
    )
    cohort = (
        cohort.join(arms.drop("_arm_rank"), on="hospitalization_id", how="left")
        .join(counts, on="hospitalization_id", how="left")
        .join(rescue, on="hospitalization_id", how="left")
        .join(escalated, on="hospitalization_id", how="left")
        .with_columns(
            pl.col("rescue_arm").is_not_null().alias("rescue_support"),
            pl.col("escalated_in_grace").fill_null(False),
            pl.col("n_device_rows_in_grace").fill_null(0),
            pl.col("_look_forward_rows").fill_null(0),
            pl.lit(grace_hours).alias("grace_hours"),
        )
    )
    device_rows = post.sort("_row").select(
        "hospitalization_id",
        "patient_id",
        "device_dttm",
        "hours_from_time_zero",
        "device_category",
        "device_class",
        "in_grace_window",
        "after_invasive_row",
        "is_rescue",
    )
    return cohort, device_rows


def _add_tracheostomy(cohort: pl.DataFrame, devices: pl.DataFrame) -> pl.DataFrame:
    first = (
        devices.filter(pl.col("_trach"))
        .group_by("hospitalization_id")
        .agg(pl.col("device_dttm").min().alias("_first_trach_dttm"))
    )
    return cohort.join(first, on="hospitalization_id", how="left").with_columns(
        (
            (pl.col("_first_trach_dttm") <= pl.col("time_zero_dttm")).fill_null(False)
            | (pl.col("_time_zero_class") == TRACHEOSTOMY)
        ).alias("tracheostomy_at_time_zero")
    )


def _add_code_status(
    cohort: pl.DataFrame,
    tables: Mapping[str, pl.DataFrame],
    config: Mapping[str, Any],
    availability: Mapping[str, Availability],
) -> pl.DataFrame:
    rules = config["exclusions"]["code_status"]
    lists = {
        key: [_norm(value) for value in rules[key]]
        for key in (
            "comfort_care_categories",
            "do_not_reintubate_categories",
            "dnr_only_categories",
            "full_code_categories",
            "other_categories",
        )
    }
    frame = _table(tables, "code_status")
    if frame is None:
        cohort = cohort.with_columns(
            pl.lit(None, dtype=pl.String).alias("code_status_at_time_zero")
        )
    else:
        frame = _available(
            frame,
            "code_status",
            availability["code_status"],
            {"code_status_category"},
            "patient_id",
        ).select(
            "patient_id",
            "_at",
            _norm_expr("code_status_category").alias("code_status_at_time_zero"),
        )
        frame = frame.filter(pl.col("code_status_at_time_zero").is_not_null())
        declared = [value for values in lists.values() for value in values]
        undeclared = frame.filter(~pl.col("code_status_at_time_zero").is_in(declared)).height
        if undeclared:
            raise QualificationError(
                f"{undeclared} code status rows carry a category the config does not declare"
            )
        latest = _latest_before(cohort, frame, on="patient_id", values=["code_status_at_time_zero"])
        cohort = cohort.join(latest, on="patient_id", how="left")
    status = pl.col("code_status_at_time_zero")
    return cohort.with_columns(
        status.is_null().alias("code_status_missing"),
        status.is_in(lists["comfort_care_categories"])
        .fill_null(False)
        .alias("comfort_care_at_time_zero"),
        status.is_in(lists["do_not_reintubate_categories"])
        .fill_null(False)
        .alias("do_not_reintubate_at_time_zero"),
    )


def _add_hypercapnia(
    cohort: pl.DataFrame,
    tables: Mapping[str, pl.DataFrame],
    config: Mapping[str, Any],
    availability: Mapping[str, Availability],
) -> pl.DataFrame:
    rule = config["risk_factors"]["hypercapnia"]
    frame = _table(tables, "labs")
    if frame is None:
        cohort = cohort.with_columns(pl.lit(None, dtype=pl.Float64).alias("paco2_mmhg"))
    else:
        frame = _available(
            frame,
            "labs",
            availability["labs"],
            {"lab_category", "lab_value_numeric", "reference_unit"},
            "hospitalization_id",
        ).filter(
            (_norm_expr("lab_category") == _norm(rule["lab_category"]))
            & pl.col("lab_value_numeric").is_not_null()
        )
        units = [unit.replace(" ", "").lower() for unit in rule["units"]]
        unit = pl.col("reference_unit").cast(pl.String).str.replace_all(" ", "").str.to_lowercase()
        if frame.filter(unit.is_not_null() & ~unit.is_in(units)).height:
            raise QualificationError(f"non-canonical unit for {rule['lab_category']}")
        latest = _latest_before(
            cohort,
            frame.select(
                "hospitalization_id", "_at", pl.col("lab_value_numeric").alias("paco2_mmhg")
            ),
            on="hospitalization_id",
            values=["paco2_mmhg"],
            lookback=timedelta(hours=float(rule["lookback_hours"])),
        )
        cohort = cohort.join(latest, on="hospitalization_id", how="left")
    return cohort.with_columns(
        (pl.col("paco2_mmhg") > float(rule["threshold_mmhg"])).alias("hypercapnia")
    )


def _add_bmi(
    cohort: pl.DataFrame,
    hospitalization: pl.DataFrame,
    tables: Mapping[str, pl.DataFrame],
    config: Mapping[str, Any],
    availability: Mapping[str, Availability],
) -> pl.DataFrame:
    rule = config["risk_factors"]["bmi"]
    frame = _table(tables, "vitals")
    if frame is None:
        cohort = cohort.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("weight_kg"),
            pl.lit(None, dtype=pl.Float64).alias("height_cm"),
        )
    else:
        frame = _available(
            frame,
            "vitals",
            availability["vitals"],
            {"vital_category", "vital_value"},
            "hospitalization_id",
        ).join(hospitalization.select("hospitalization_id", "patient_id"), on="hospitalization_id")
        for measure in ("weight", "height"):
            column = "weight_kg" if measure == "weight" else "height_cm"
            scope = rule[f"{measure}_scope"]
            if scope not in ("index_hospitalization", "patient_history"):
                raise QualificationError(f"unknown risk_factors.bmi.{measure}_scope {scope!r}")
            low, high = rule[f"plausible_{column}"]
            on = "hospitalization_id" if scope == "index_hospitalization" else "patient_id"
            rows = frame.filter(
                (_norm_expr("vital_category") == _norm(rule[f"{measure}_category"]))
                & pl.col("vital_value").is_between(low, high)
            ).select(on, "_at", pl.col("vital_value").alias(column))
            cohort = cohort.join(
                _latest_before(cohort, rows, on=on, values=[column]), on=on, how="left"
            )
    low, high = rule["plausible_bmi"]
    bmi = pl.col("weight_kg") / (pl.col("height_cm") / 100.0) ** 2
    return cohort.with_columns(
        pl.when(bmi.is_between(low, high)).then(bmi).alias("bmi")
    ).with_columns((pl.col("bmi") > float(rule["threshold"])).alias("bmi_over_30"))


def _add_comorbidities(
    cohort: pl.DataFrame,
    hospitalization: pl.DataFrame,
    tables: Mapping[str, pl.DataFrame],
    config: Mapping[str, Any],
) -> pl.DataFrame:
    """Comorbidity proxies from the declared code sources; NULL when no source has history."""
    rule = config["risk_factors"]["comorbidities"]
    names = [f"comorbidity_{name}" for name in rule["definitions"]]
    leaky = any(
        rule["available_sources"][source]["leaks_post_time_zero"] for source in rule["sources"]
    )
    cohort = cohort.with_columns(pl.lit(leaky).alias("comorbidity_source_leaky"))
    frame = _table(tables, "hospital_diagnosis")
    if frame is None:
        return cohort.with_columns(
            pl.lit(False).alias("comorbidity_history_available"),
            *[pl.lit(None, dtype=pl.Boolean).alias(name) for name in names],
        )
    _require_columns(
        frame,
        "hospital_diagnosis",
        {"hospitalization_id", "diagnosis_code", "diagnosis_code_format"},
    )
    _require_string(frame, "hospital_diagnosis", ["hospitalization_id", "diagnosis_code"])
    present_on_admission = (
        pl.col("poa_present").cast(pl.Int64, strict=False)
        if "poa_present" in frame.columns
        else pl.lit(None, dtype=pl.Int64)
    )
    codes = frame.select(
        "hospitalization_id",
        pl.col("diagnosis_code")
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9]", "")
        .alias("_code"),
        pl.col("diagnosis_code_format")
        .cast(pl.String)
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]", "")
        .alias("_format"),
        present_on_admission.alias("_poa"),
    )
    index = cohort.select("patient_id", "hospitalization_id").join(codes, on="hospitalization_id")
    evidence = []
    if "prior_hospitalization_codes" in rule["sources"]:
        prior = (
            cohort.select("patient_id", pl.col("admission_dttm").alias("_index_admission"))
            .join(
                hospitalization.select("patient_id", "hospitalization_id", "discharge_dttm"),
                on="patient_id",
            )
            .filter(pl.col("discharge_dttm") < pl.col("_index_admission"))
            .select("patient_id", "hospitalization_id")
            .join(codes, on="hospitalization_id")
        )
        evidence.append(prior.with_columns(pl.lit(True).alias("_counts")))
    if "index_stay_poa_codes" in rule["sources"]:
        evidence.append(
            index.filter(pl.col("_poa").is_not_null()).with_columns(
                (pl.col("_poa") == 1).alias("_counts")
            )
        )
    if "index_stay_codes" in rule["sources"]:
        evidence.append(index.with_columns(pl.lit(True).alias("_counts")))
    history = pl.concat(
        [part.select("patient_id", "_code", "_format", "_counts") for part in evidence]
        or [
            index.head(0)
            .with_columns(pl.lit(False).alias("_counts"))
            .drop("hospitalization_id", "_poa")
        ]
    )

    def matches(definition: Mapping[str, list[str]]) -> pl.Expr:
        hit = pl.lit(False)
        for code_format, prefixes in definition.items():
            pattern = "^(?:" + "|".join(prefixes) + ")"
            hit = hit | ((pl.col("_format") == code_format) & pl.col("_code").str.contains(pattern))
        return (hit & pl.col("_counts")).fill_null(False).any()

    flags = history.group_by("patient_id").agg(
        (pl.len() > 0).alias("comorbidity_history_available"),
        *[
            matches(definition).alias(f"comorbidity_{name}")
            for name, definition in rule["definitions"].items()
        ],
    )
    return cohort.join(flags, on="patient_id", how="left").with_columns(
        pl.col("comorbidity_history_available").fill_null(False)
    )


def _add_calendar_period(
    cohort: pl.DataFrame, source: Mapping[str, Any] | None, periods: pl.DataFrame | None
) -> pl.DataFrame:
    """Join the site's declared period source on patient. Never read event dates."""
    if source is None or periods is None:
        return cohort.with_columns(pl.lit(None, dtype=pl.String).alias("calendar_period"))
    _require_columns(
        periods, "calendar period source", {source["patient_id_col"], source["period_col"]}
    )
    periods = periods.select(
        pl.col(source["patient_id_col"]).cast(pl.String).alias("patient_id"),
        pl.col(source["period_col"]).cast(pl.String).alias("calendar_period"),
    ).unique()
    if periods["patient_id"].n_unique() != periods.height:
        raise QualificationError("calendar period source has more than one period for a patient")
    return cohort.join(periods, on="patient_id", how="left")


def _add_unit(
    cohort: pl.DataFrame,
    tables: Mapping[str, pl.DataFrame],
    availability: Mapping[str, Availability],
) -> pl.DataFrame:
    frame = _table(tables, "adt")
    if frame is None:
        return cohort.with_columns(pl.lit(None, dtype=pl.String).alias("unit_at_time_zero"))
    column = "location_type" if "location_type" in frame.columns else "location_category"
    frame = _available(frame, "adt", availability["adt"], {column}, "hospitalization_id").select(
        "hospitalization_id", "_at", pl.col(column).cast(pl.String).alias("unit_at_time_zero")
    )
    latest = _latest_before(cohort, frame, on="hospitalization_id", values=["unit_at_time_zero"])
    return cohort.join(latest, on="hospitalization_id", how="left")


def _add_partition(
    cohort: pl.DataFrame, episodes: pl.DataFrame | None, split: Mapping[str, Any]
) -> pl.DataFrame:
    """Inherit the episode artifact's partition by patient; assign the rest by its rule."""
    if episodes is None:
        cohort = cohort.with_columns(
            pl.lit(None, dtype=pl.String).alias("partition"), pl.lit(False).alias("_in_artifact")
        )
    else:
        _require_columns(episodes, "episode artifact", {"patient_id", "partition"})
        _require_string(episodes, "episode artifact", ["patient_id", "partition"])
        assigned = (
            episodes.filter(pl.col("partition").is_not_null())
            .select("patient_id", "partition")
            .unique()
        )
        if assigned["patient_id"].n_unique() != assigned.height:
            raise QualificationError("episode artifact places a patient in more than one partition")
        present = (
            episodes.select("patient_id").unique().with_columns(pl.lit(True).alias("_in_artifact"))
        )
        cohort = cohort.join(assigned, on="patient_id", how="left").join(
            present, on="patient_id", how="left"
        )
    cohort = cohort.with_columns(
        pl.when(pl.col("partition").is_not_null())
        .then(pl.lit("episode_artifact"))
        .when(pl.col("_in_artifact").fill_null(False))
        .then(pl.lit("assigned_null_in_artifact"))
        .otherwise(pl.lit("assigned_absent_from_artifact"))
        .alias("partition_source")
    )
    missing = cohort.filter(pl.col("partition").is_null())
    if missing.height:
        # The episode artifact's own rule (src.data.splits.assign_grouped_splits): a
        # seeded hash of the patient id and, where a site populates it, the linked
        # encounter id of the row being assigned.
        try:
            ruled = assign_grouped_splits(
                missing.select("hospitalization_id", "patient_id", "hospitalization_joined_id"),
                split["partitions"],
                seed=split["split_seed"],
            )
        except ValueError as exc:
            raise QualificationError(str(exc)) from exc
        cohort = (
            cohort.join(
                ruled.select("patient_id", pl.col("partition").alias("_ruled")),
                on="patient_id",
                how="left",
            )
            .with_columns(pl.coalesce("partition", "_ruled").alias("partition"))
            .drop("_ruled")
        )
    return cohort


def _statuses(config: Mapping[str, Any]) -> dict[str, pl.Expr]:
    """Eligibility status per cohort: `primary` plus each sensitivity cohort."""
    look_back = config["detection"]["look_back"]
    look_forward = config["detection"]["look_forward"]
    minimums = {"primary": config["cohorts"]["primary"]["min_invasive_hours"]}
    for name, cohort in (config["cohorts"].get("sensitivity") or {}).items():
        minimums[name] = cohort["min_invasive_hours"]
    age = pl.col("age_at_admission")

    def status(min_hours: float) -> pl.Expr:
        return (
            pl.when(age.is_null())
            .then(pl.lit("missing_age"))
            .when(age < config["exclusions"]["minimum_age"])
            .then(pl.lit("underage"))
            .when(
                (pl.col("n_invasive_rows") < look_back["min_invasive_rows"])
                | (
                    pl.col("hours_since_last_invasive_row")
                    > float(look_back["max_hours_since_last_invasive_row"])
                )
            )
            .then(pl.lit("lookback_insufficient"))
            .when(pl.col("_look_forward_rows") < look_forward["min_non_invasive_rows"])
            .then(pl.lit("lookforward_insufficient"))
            .when(pl.col("tracheostomy_at_time_zero"))
            .then(pl.lit("tracheostomy"))
            .when(pl.col("comfort_care_at_time_zero"))
            .then(pl.lit("comfort_care"))
            .when(pl.col("do_not_reintubate_at_time_zero"))
            .then(pl.lit("do_not_reintubate"))
            .when(pl.col("imv_hours") < float(min_hours))
            .then(pl.lit("ventilation_under_minimum"))
            .otherwise(pl.lit("eligible"))
        )

    return {name: status(hours) for name, hours in minimums.items()}


def _waterfall(
    cohort: pl.DataFrame,
    hospitalization: pl.DataFrame,
    devices: pl.DataFrame,
    events: pl.DataFrame,
    stitched: int,
    config: Mapping[str, Any],
    split: Mapping[str, Any],
) -> dict[str, int]:
    invasive = (
        devices.filter(pl.col("device_class") == INVASIVE).select("hospitalization_id").unique()
    )
    waterfall = {
        "hospitalization_source": hospitalization.height,
        "patient_source": hospitalization["patient_id"].n_unique(),
        "resp_rows_without_device": devices.filter(pl.col("device_class").is_null()).height,
        "resp_rows_unmapped_device": devices.filter(pl.col("device_class") == UNMAPPED).height,
        "hospitalization_with_invasive_ventilation": invasive.join(
            hospitalization, on="hospitalization_id"
        ).height,
        "patient_with_invasive_ventilation": invasive.join(
            hospitalization, on="hospitalization_id"
        )["patient_id"].n_unique(),
        "extubation_gaps_stitched": stitched,
        "extubation_events": events.height,
        "hospitalization_with_extubation": events["hospitalization_id"].n_unique(),
        "extubation_excluded_not_first": events.height - cohort.height,
        "patient_first_extubation": cohort.height,
    }
    arms = list(config["arms"]["order"])
    sensitivity = list(config["cohorts"].get("sensitivity") or {})
    for name in ["primary", *sensitivity]:
        prefix = "" if name == "primary" else f"{name}_"
        suffix = "" if name == "primary" else f"_{name}"
        status = cohort[f"eligibility_status{suffix}"]
        for reason in REASONS:
            waterfall[f"{prefix}patient_excluded_{reason}"] = int((status == reason).sum())
        eligible = cohort.filter(pl.col(f"eligible{suffix}"))
        waterfall[f"{prefix}patient_eligible"] = eligible.height
        for arm in arms:
            waterfall[f"{prefix}arm_{arm}"] = eligible.filter(pl.col("arm") == arm).height
    eligible = cohort.filter(pl.col("eligible"))
    for arm in arms:
        waterfall[f"highest_support_arm_{arm}"] = eligible.filter(
            pl.col("arm_highest_support") == arm
        ).height
    for key, flag in {
        "eligible_code_status_missing": pl.col("code_status_missing"),
        "eligible_escalated_in_grace": pl.col("escalated_in_grace"),
        "eligible_rescue_support": pl.col("rescue_support"),
        "eligible_paco2_available": pl.col("paco2_mmhg").is_not_null(),
        "eligible_bmi_available": pl.col("bmi").is_not_null(),
        "eligible_comorbidity_history_available": pl.col("comorbidity_history_available"),
        "eligible_calendar_period_available": pl.col("calendar_period").is_not_null(),
        "eligible_unit_available": pl.col("unit_at_time_zero").is_not_null(),
    }.items():
        waterfall[key] = eligible.filter(flag).height
    for partition in split["partitions"]:
        waterfall[f"eligible_partition_{partition}"] = eligible.filter(
            pl.col("partition") == partition
        ).height
    for key, source in {
        "partition_inherited": "episode_artifact",
        "partition_assigned_absent_from_artifact": "assigned_absent_from_artifact",
        "partition_assigned_null_in_artifact": "assigned_null_in_artifact",
    }.items():
        waterfall[key] = cohort.filter(pl.col("partition_source") == source).height
    return waterfall


def build_extubation_cohort(
    tables: Mapping[str, pl.DataFrame],
    config: Mapping[str, Any],
    *,
    availability: Mapping[str, Availability],
    split: Mapping[str, Any],
    site: str,
    episodes: pl.DataFrame | None = None,
    calendar_periods: pl.DataFrame | None = None,
    grace_hours: float | None = None,
) -> ExtubationCohort:
    """Build one first-extubation row per patient without dropping exclusion states.

    `tables` maps the names in `config["source_tables"]` to CLIF frames;
    `hospitalization` and `respiratory_support` are required, the rest optional.
    `split` is the train config's `data_contract` (`partitions`, `split_seed`).
    `grace_hours` overrides the configured grace window for a sensitivity build.

    Covariate columns use only rows available strictly before `time_zero_dttm`. The
    exposure columns (`arm*`, `escalated_in_grace`, `n_device_rows_in_grace`) and the
    descriptive `rescue_*` flags are read after it by design and are never covariates.
    """
    _validate_config(config)
    period_source = _period_source(config, site)
    grace = float(config["grace_window"]["hours"] if grace_hours is None else grace_hours)
    if grace < 0:
        raise QualificationError("grace window must be non-negative")
    hospitalization = _hospitalization(tables)
    devices = _device_rows(tables, config, availability["respiratory_support"])
    events, stitched = _detect_extubations(devices, config)
    events = events.join(hospitalization, on="hospitalization_id", how="inner")
    cohort = events.sort("patient_id", "time_zero_dttm", "hospitalization_id").unique(
        subset=["patient_id"], keep="first", maintain_order=True
    )
    cohort = cohort.with_columns(
        _hours("time_zero_dttm", "imv_start_dttm").alias("imv_hours"),
        _hours("time_zero_dttm", "_last_invasive_dttm").alias("hours_since_last_invasive_row"),
    )
    cohort, device_rows = _assign_arms(cohort, devices, config, grace)
    cohort = _add_tracheostomy(cohort, devices)
    cohort = _add_code_status(cohort, tables, config, availability)
    cohort = _add_hypercapnia(cohort, tables, config, availability)
    cohort = _add_bmi(cohort, hospitalization, tables, config, availability)
    cohort = _add_comorbidities(cohort, hospitalization, tables, config)
    cohort = _add_calendar_period(cohort, period_source, calendar_periods)
    cohort = _add_unit(cohort, tables, availability)
    risk = config["risk_factors"]
    cohort = cohort.with_columns(
        (pl.col("age_at_admission") > risk["age_over_65"]["threshold"]).alias("age_over_65"),
        (pl.col("imv_hours") >= float(risk["prolonged_ventilation"]["min_invasive_hours"])).alias(
            "prolonged_ventilation"
        ),
    )
    for name, status in _statuses(config).items():
        suffix = "" if name == "primary" else f"_{name}"
        cohort = cohort.with_columns(status.alias(f"eligibility_status{suffix}")).with_columns(
            (pl.col(f"eligibility_status{suffix}") == "eligible").alias(f"eligible{suffix}")
        )
    cohort = _add_partition(cohort, episodes, split)
    waterfall = _waterfall(cohort, hospitalization, devices, events, stitched, config, split)

    config_sha256 = hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "grace_hours": grace,
                "site": site,
                "split": split,
                "availability": {name: list(value) for name, value in availability.items()},
            },
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    episode_hashes = (
        episodes["episode_sha256"].unique().to_list()
        if episodes is not None and "episode_sha256" in episodes.columns
        else []
    )
    # The index stay's discharge time is post-time-zero information: not carried.
    cohort = cohort.drop(
        "discharge_dttm", *[column for column in cohort.columns if column.startswith("_")]
    ).sort("patient_id")
    cohort = cohort.with_columns(
        pl.lit(site).alias("site"),
        pl.lit(config["contract_version"]).alias("extubation_contract_version"),
        pl.lit(config_sha256).alias("extubation_config_sha256"),
        pl.lit(episode_hashes[0] if len(episode_hashes) == 1 else None, dtype=pl.String).alias(
            "episode_sha256"
        ),
        pl.lit(content_manifest(cohort, columns=HASH_COLUMNS)["sha256"]).alias("extubation_sha256"),
    )
    validate_extubation_artifact(cohort)
    return ExtubationCohort(cohort, device_rows, waterfall)


def validate_extubation_artifact(cohort: pl.DataFrame) -> None:
    """Validate the extubation cohort before any downstream join or write."""
    _require_columns(
        cohort,
        "extubation cohort",
        {
            *HASH_COLUMNS,
            "extubation_contract_version",
            "extubation_config_sha256",
            "extubation_sha256",
        },
    )
    _require_string(
        cohort,
        "extubation cohort",
        [
            "hospitalization_id",
            "patient_id",
            "eligibility_status",
            "arm",
            "partition",
            "extubation_sha256",
        ],
    )
    _reject_null_identifiers(cohort, "extubation cohort", ["hospitalization_id", "patient_id"])
    _require_utc(cohort, "extubation cohort", ["time_zero_dttm"])
    if cohort["patient_id"].n_unique() != cohort.height:
        raise QualificationError("extubation cohort must hold one first extubation per patient")
    eligible = cohort.filter(pl.col("eligible"))
    if eligible["arm"].has_nulls() or eligible["partition"].has_nulls():
        raise QualificationError("eligible extubation rows must carry an arm and a partition")
    if cohort["time_zero_dttm"].has_nulls():
        raise QualificationError("extubation cohort rows must carry a time zero")
    if cohort.is_empty():
        return
    hashes = cohort["extubation_sha256"].unique().to_list()
    if len(hashes) != 1 or hashes[0] != content_manifest(cohort, columns=HASH_COLUMNS)["sha256"]:
        raise QualificationError("extubation cohort content hash mismatch")


def _read_tables(
    base: Path, config: Mapping[str, Any], availability: Mapping[str, Availability]
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Read only the rows and columns the cohort needs; record each file's hash."""
    risk = config["risk_factors"]
    flag = config["exclusions"]["tracheostomy"]["flag_col"]
    categories = [_norm(risk["bmi"]["weight_category"]), _norm(risk["bmi"]["height_category"])]
    narrow = {
        "respiratory_support": lambda scan, at: scan.select(
            "hospitalization_id", at, "device_category", flag
        ),
        "labs": lambda scan, at: scan.filter(
            _norm_expr("lab_category") == _norm(risk["hypercapnia"]["lab_category"])
        ).select("hospitalization_id", at, "lab_category", "lab_value_numeric", "reference_unit"),
        "vitals": lambda scan, at: scan.filter(
            _norm_expr("vital_category").is_in(categories)
        ).select("hospitalization_id", at, "vital_category", "vital_value"),
    }
    tables: dict[str, pl.DataFrame] = {}
    provenance: dict[str, str] = {}
    for name, source in availability.items():
        path = base / f"{source.file}.parquet"
        if not path.exists():
            if name in ("hospitalization", "respiratory_support"):
                raise QualificationError(f"required CLIF table is missing: {path.name}")
            continue
        scan = pl.scan_parquet(path)
        try:
            tables[name] = (narrow[name](scan, source.column) if name in narrow else scan).collect()
        except pl.exceptions.ColumnNotFoundError as exc:
            raise QualificationError(f"{name} is missing required columns") from exc
        provenance[path.stem] = _file_sha256(path)
    return tables, provenance


def build_extubation_artifact(
    data_dir: str | Path,
    output: str | Path,
    *,
    site: str,
    extubation_config: str | Path,
    data_config: str | Path,
    train_config: str | Path,
    artifact_policy: str | Path,
    episode_artifact: str | Path | None = None,
    allow_missing_episode_artifact: bool = False,
    grace_hours: float | None = None,
) -> dict[str, int]:
    """Build, validate and persist a site's extubation cohort; return only the waterfall."""
    config = load_extubation_config(extubation_config)
    policy = yaml.safe_load(Path(artifact_policy).read_text())
    destination = Path(output)
    device_destination = destination.with_name(f"{destination.stem}_device_rows.parquet")
    for path in (destination, device_destination):
        validate_artifact_destination(path, "patient_level_phi", policy)
    availability = table_availability(config, yaml.safe_load(Path(data_config).read_text()))
    split = yaml.safe_load(Path(train_config).read_text())["data_contract"]
    period_source = _period_source(config, site)
    base = Path(data_dir)
    tables, provenance = _read_tables(base, config, availability)

    episode_path = Path(episode_artifact or config["episode_artifact"])
    episodes = None
    if episode_path.exists():
        episodes = pl.read_parquet(episode_path)
        validate_episode_artifact(episodes)
    elif not allow_missing_episode_artifact:
        raise QualificationError(
            f"episode artifact is missing: {episode_path.name}; build it with src.data.cohort "
            "so partitions are inherited, or allow a build without it explicitly"
        )
    periods = None
    if period_source is not None:
        period_path = base / f"{period_source['file']}.parquet"
        if period_path.exists():
            periods = pl.read_parquet(period_path)
            provenance[period_path.stem] = _file_sha256(period_path)

    result = build_extubation_cohort(
        tables,
        config,
        availability=availability,
        split=split,
        site=site,
        episodes=episodes,
        calendar_periods=periods,
        grace_hours=grace_hours,
    )
    cohort = result.cohort.with_columns(
        pl.lit(json.dumps(provenance, sort_keys=True)).alias("source_provenance_json"),
        pl.lit(json.dumps(result.waterfall, sort_keys=True)).alias("waterfall_json"),
    )
    validate_extubation_artifact(cohort)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cohort.write_parquet(destination)
    result.device_rows.write_parquet(device_destination)
    return result.waterfall


def _equations(waterfall: Mapping[str, int]) -> list[list[str]]:
    """Published totals and the cells summing to them, `[total, *cells]`."""
    equations = [
        ["extubation_events", "extubation_excluded_not_first", "patient_first_extubation"],
        ["patient_first_extubation", *(key for key in waterfall if key.startswith("partition_"))],
        ["patient_eligible", *(key for key in waterfall if key.startswith("eligible_partition_"))],
        ["patient_eligible", *(key for key in waterfall if key.startswith("highest_support_arm_"))],
    ]
    for key in waterfall:
        if not key.endswith("patient_eligible"):
            continue
        prefix = key.removesuffix("patient_eligible")
        equations.append(
            [
                "patient_first_extubation",
                key,
                *(k for k in waterfall if k.startswith(f"{prefix}patient_excluded_")),
            ]
        )
        equations.append([key, *(k for k in waterfall if k.startswith(f"{prefix}arm_"))])
    equations = [[key for key in equation if key in waterfall] for equation in equations]
    return [equation for equation in equations if len(equation) > 2]


def suppress_waterfall(waterfall: Mapping[str, int], min_cell: int = 10) -> dict[str, int | str]:
    """Replace every count below `min_cell` with ``"<{min_cell}"`` and withhold a
    complementary cell (``"suppressed"``) wherever one hidden non-zero count could be
    recovered from a published total. On-node display only: anything that leaves the
    node goes through the release ledger (KTD10)."""
    hidden = {key for key, value in waterfall.items() if value < min_cell}
    complementary: set[str] = set()
    changed = True
    while changed:
        changed = False
        for equation in _equations(waterfall):
            exposed = [key for key in equation if key in hidden and waterfall[key] != 0]
            candidates = sorted((waterfall[key], key) for key in equation if key not in hidden)
            if len(exposed) == 1 and candidates:
                hidden.add(candidates[0][1])
                complementary.add(candidates[0][1])
                changed = True
    return {
        key: SUPPRESSED if key in complementary else f"<{min_cell}" if key in hidden else value
        for key, value in waterfall.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a site's extubation cohort artifact")
    parser.add_argument("--data", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--config", default="configs/extubation.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--artifact-policy", default=None)
    parser.add_argument("--episodes", default=None)
    parser.add_argument("--allow-missing-episode-artifact", action="store_true")
    parser.add_argument("--grace-hours", type=float, default=None)
    args = parser.parse_args()
    config = load_extubation_config(args.config)
    artifact_policy = args.artifact_policy or config["artifact_policy"]
    waterfall = build_extubation_artifact(
        args.data,
        args.out or config["outputs"]["cohort"],
        site=args.site,
        extubation_config=args.config,
        data_config=args.data_config or config["data_config"],
        train_config=args.train_config or config["train_config"],
        artifact_policy=artifact_policy,
        episode_artifact=args.episodes,
        allow_missing_episode_artifact=args.allow_missing_episode_artifact,
        grace_hours=args.grace_hours,
    )
    policy = yaml.safe_load(Path(artifact_policy).read_text())
    min_cell = policy["classes"]["aggregate_no_phi"]["minimum_cell_size"]
    # Aggregate counts only: never a row, an identifier or a timestamp.
    print(json.dumps(suppress_waterfall(waterfall, min_cell), indent=2))


if __name__ == "__main__":
    main()
