"""Extubation study outcome labels: reintubation, death and their composite (plan U10).

Hard rule #1, scoped amendment (2026-10-03): the trunk never trains on treatment
targets. Reintubation is defined by a treatment event, so it is a label for study heads
and estimators downstream of the frozen trunk and nothing else. `clif_auto_labeler`
still refuses every outcome read from an input-only table; this module is the scoped
replacement of that refusal for endpoints DECLARED under `study_endpoints` in
configs/cohort.yaml with `use: label_only`. It never reads or writes the trunk's
`outcomes` block, and it refuses to read respiratory support without the declaration.

Definitions (every time is in hours from the cohort's time zero):

- Reintubation: the first invasive-ventilation device row of the index hospitalization
  charted strictly after time zero, on the availability clock the cohort used. NIV or
  HFNC after extubation is never a failure. The cohort only calls a row an extubation
  when the return to invasive ventilation is at least `detection.stitch_gap_hours`
  later, so a shorter gap cannot follow a valid time zero; one that does is refused.
- Tracheostomy: invasive ventilation through a tracheostomy is a return to invasive
  ventilation and counts. A tracheostomy with no invasive row (trach collar, or the
  flag on a non-invasive row) is not a reintubation; its time is kept in
  `tracheostomy_hours` for a sensitivity analysis.
- Death: `patient.death_dttm`, read with the index stay's discharge category. A stay
  discharged `expired` died no later than its discharge (the earlier of the two times,
  or the discharge time when there is no timestamp). A patient discharged alive cannot
  have died before the discharge, so an earlier timestamp (day-resolution timestamps
  are floored to midnight) is moved to the discharge time.
- Hospice: a discharge to hospice is a competing event at the discharge time. A death
  after it does not turn it into a composite event.
- Follow-up for reintubation ends at death or at the index discharge. Readmissions are
  not searched.
- A window is right-closed: an event at exactly the horizon is inside it. Events at
  the same instant are ordered reintubation, hospice discharge, death.
- A patient discharged alive before the horizon with no event inside the window is
  `censored` at discharge under `discharge_alive_rule="censor"` (default) and
  `negative` at the horizon under `"event_free"`, which assumes no event because an
  in-hospital reintubation cannot be observed after discharge. An unknown discharge
  disposition is censored under both rules: it is never read as survival.

Row-level output is `patient_level_phi` (configs/artifact_policy.yaml). The path-based
builder and the CLI return only aggregate counts, never by device arm:

    uv run python -m src.eval.extubation_labeler --data <clif dir>
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
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
)
from src.data.extubation_cohort import (
    INVASIVE,
    SUPPRESSED,
    TRACHEOSTOMY,
    Availability,
    _device_rows,
    _hours,
    _norm,
    _norm_expr,
    _table,
    death_record_flags,
    load_extubation_config,
    table_availability,
    validate_extubation_artifact,
)
from src.data.targets import OUTCOME_STATUSES

STUDY = "extubation"
COMPOSITE = "reintubation_or_death"
EVENTS = (COMPOSITE, "reintubation", "death")
DISCHARGE_ALIVE_RULES = ("censor", "event_free")
# `event_type` under coding="cause": which event came first inside the window.
CAUSE_CODES = {"none": 0, "reintubation": 1, "death": 2, "hospice": 3}
# `event_type` under coding="status": the endpoint's own event against everything that
# competes with it, so `event_of_interest=1` is always the endpoint.
STATUS_CODES = {"none": 0, "event": 1, "competing": 2}
# Deaths this soon after extubation with no reintubation are flagged (not removed) for
# the terminal-extubation sensitivity analysis.
TERMINAL_EXTUBATION_HOURS = 24.0
# Source of the death timestamp; the discharge category comes from hospitalization.
DEATH_SOURCE = "patient"
# Per endpoint event: the candidate events in tie order, and the status each one gives.
_ROLES = {
    COMPOSITE: (
        ("reintubation", "positive"),
        ("hospice", "competing_event"),
        ("death", "positive"),
    ),
    "reintubation": (
        ("reintubation", "positive"),
        ("hospice", "competing_event"),
        ("death", "competing_event"),
    ),
    # Reintubation does not prevent death, so it is not a candidate here.
    "death": (("hospice", "competing_event"), ("death", "positive")),
}
# The cells that partition the eligible cohort at one horizon. The two discharge rules
# differ only in where `discharged_alive_before_horizon` goes: censored under `censor`,
# event-free under `event_free`.
WINDOW_CELLS = (
    "event_reintubation",
    "event_death",
    "competing_hospice",
    "event_free_in_hospital",
    "discharged_alive_before_horizon",
    "discharge_unknown_before_horizon",
)
LABEL_COLUMNS = {
    "hospitalization_id",
    "patient_id",
    "reintubation_hours",
    "death_hours",
    "hospice_hours",
    "followup_end_hours",
    "followup_end_reason",
}

assert {status for roles in _ROLES.values() for _, status in roles} <= OUTCOME_STATUSES


class EndpointDeclarationError(QualificationError):
    """An endpoint is not a declared label-only study endpoint (hard rule #1, amended)."""


class StudyEndpoint(NamedTuple):
    name: str
    event: str  # one of EVENTS
    horizon_hours: float
    sources: frozenset[str]


class EstimatorArrays(NamedTuple):
    """One entry per label row, in label (= cohort) order, for one horizon and rule."""

    event_time: np.ndarray  # hours: the first event, the horizon, or the censoring time
    event_type: np.ndarray  # integer codes, see `codes`; 0 = no event by `event_time`
    resolved: np.ndarray  # False where follow-up ended before the horizon with no event
    codes: dict[str, int]


# --------------------------------------------------------------------- declarations


def _endpoint_sources(name: str, endpoints: Mapping[str, Any], seen: tuple = ()) -> set[str]:
    """Every source table a study endpoint is read from, through its composites."""
    if name in seen:
        raise EndpointDeclarationError(f"study endpoint {name} is a circular composite")
    spec = endpoints.get(name)
    if not isinstance(spec, Mapping):
        raise EndpointDeclarationError(f"study endpoint {name} is not declared")
    if "composite_of" in spec:
        sources: set[str] = set()
        for part in spec["composite_of"]:
            sources |= _endpoint_sources(part, endpoints, (*seen, name))
        return sources
    if not spec.get("source"):
        raise EndpointDeclarationError(
            f"study endpoint {name} declares neither source nor composite_of"
        )
    return {spec["source"]}


def _study_endpoint(
    name: str, endpoints: Mapping[str, Any], sources: set[str]
) -> StudyEndpoint:
    spec = endpoints[name]
    if spec.get("time_zero") != "extubation":
        raise EndpointDeclarationError(f"study endpoint {name}: time_zero must be 'extubation'")
    horizon = spec.get("horizon_hours")
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, (int, float))
        or not math.isfinite(horizon)
        or horizon <= 0
    ):
        raise EndpointDeclarationError(
            f"study endpoint {name}: horizon_hours must be a positive number"
        )
    if "composite_of" not in spec:
        if spec.get("event") not in ("reintubation", "death"):
            raise EndpointDeclarationError(
                f"study endpoint {name} declares event {spec.get('event')!r}, "
                "which this labeler does not derive"
            )
        return StudyEndpoint(name, spec["event"], float(horizon), frozenset(sources))
    parts = [endpoints[part] for part in spec["composite_of"]]
    if any("composite_of" in part for part in parts) or {part.get("event") for part in parts} != {
        "reintubation",
        "death",
    }:
        raise EndpointDeclarationError(
            f"study endpoint {name}: a composite must be made of one reintubation and one "
            "death endpoint"
        )
    if any(part.get("horizon_hours") != horizon for part in parts):
        raise EndpointDeclarationError(
            f"study endpoint {name}: a composite and its components must share horizon_hours"
        )
    return StudyEndpoint(name, COMPOSITE, float(horizon), frozenset(sources))


def declared_study_endpoints(
    cohort_config: Mapping[str, Any], data_config: Mapping[str, Any], *, study: str = STUDY
) -> dict[str, StudyEndpoint]:
    """The `study`'s label-only endpoints, after checking the whole `study_endpoints` block.

    The same rule as tests/test_data_config.py (`check_treatment_target_rule`): every
    study endpoint is declared `use: label_only`, none is also a trunk outcome, and the
    treatment-target policy is `context_only`. The trunk's `outcomes` block is consulted
    only to refuse an overlap; no label is read from it.
    """
    if cohort_config.get("treatment_target_policy") != "context_only":
        raise EndpointDeclarationError("treatment_target_policy must stay context_only")
    input_only = {
        name for name, spec in data_config["tables"].items() if spec.get("input_only")
    }
    treatment = set(cohort_config.get("treatment_sources") or ()) | input_only
    endpoints = cohort_config.get("study_endpoints") or {}
    if not isinstance(endpoints, Mapping):
        raise EndpointDeclarationError(
            "study_endpoints must be a mapping of endpoint name to declaration"
        )
    overlap = sorted(set(endpoints) & set(cohort_config.get("outcomes") or {}))
    if overlap:
        raise EndpointDeclarationError(f"study endpoints are also trunk targets: {overlap}")
    declared: dict[str, StudyEndpoint] = {}
    for name, spec in endpoints.items():
        sources = _endpoint_sources(name, endpoints)
        if spec.get("use") != "label_only":
            derived = sorted(sources & treatment)
            raise EndpointDeclarationError(
                f"study endpoint {name} is not declared label_only"
                + (f"; it derives from treatment source(s) {derived}" if derived else "")
            )
        if spec.get("study") == study:
            declared[name] = _study_endpoint(name, endpoints, sources)
    return declared


def _label_endpoints(
    cohort_config: Mapping[str, Any], data_config: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, StudyEndpoint]:
    """Declared endpoints, refusing to read a label source no endpoint declares."""
    endpoints = declared_study_endpoints(cohort_config, data_config)
    expected = {
        "reintubation": config["source_tables"]["respiratory_support"].get("data_table"),
        "death": DEATH_SOURCE,
    }
    for event, source in expected.items():
        leaves = [endpoint for endpoint in endpoints.values() if endpoint.event == event]
        if not leaves:
            raise EndpointDeclarationError(
                f"no study endpoint declares {event} as label_only; {source} is not read "
                "as a label source without that declaration"
            )
        for endpoint in leaves:
            if endpoint.sources != {source}:
                raise EndpointDeclarationError(
                    f"study endpoint {endpoint.name}: {event} must be read from source "
                    f"{source!r}, not {sorted(endpoint.sources)}"
                )
    return endpoints


# --------------------------------------------------------------------- labels


def _index_stays(
    tables: Mapping[str, pl.DataFrame], data_config: Mapping[str, Any]
) -> pl.DataFrame:
    """Discharge time and disposition class (configs/data.yaml `gem.disposition_map`)."""
    frame = _table(tables, "hospitalization", required=True)
    _require_columns(
        frame, "hospitalization", {"hospitalization_id", "discharge_dttm", "discharge_category"}
    )
    _require_string(frame, "hospitalization", ["hospitalization_id", "discharge_category"])
    _reject_null_identifiers(frame, "hospitalization", ["hospitalization_id"])
    _require_utc(frame, "hospitalization", ["discharge_dttm"])
    if frame["hospitalization_id"].n_unique() != frame.height:
        raise QualificationError("hospitalization_id must be unique in hospitalization")
    dispositions = {
        _norm(key): value for key, value in data_config["gem"]["disposition_map"].items()
    }
    category = _norm_expr("discharge_category")
    # Unmapped, blank or null: end of observation, never read as survival.
    return frame.select(
        "hospitalization_id",
        "discharge_dttm",
        pl.when(category.is_not_null())
        .then(category.replace_strict(dispositions, default="unknown", return_dtype=pl.String))
        .otherwise(pl.lit("unknown"))
        .alias("discharge_disposition"),
    )


def _death_times(tables: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    frame = _table(tables, "patient", required=True)
    _require_columns(frame, "patient", {"patient_id", "death_dttm"})
    _require_string(frame, "patient", ["patient_id"])
    _reject_null_identifiers(frame, "patient", ["patient_id"])
    if frame.schema["death_dttm"] == pl.Null:  # a site with no death timestamp at all
        frame = frame.with_columns(pl.col("death_dttm").cast(pl.Datetime("us", "UTC")))
    _require_utc(frame, "patient", ["death_dttm"])
    if frame["patient_id"].n_unique() != frame.height:
        raise QualificationError("patient_id must be unique in patient")
    return frame.select("patient_id", "death_dttm", pl.lit(True).alias("_in_patient"))


def _event_rows(
    index: pl.DataFrame, devices: pl.DataFrame, config: Mapping[str, Any]
) -> pl.DataFrame:
    """First invasive row and first tracheostomy evidence strictly after time zero."""
    stitch = timedelta(hours=float(config["detection"]["stitch_gap_hours"]))
    # An invasive row charted at time zero itself precedes the extubation row (the
    # cohort orders it first), so only rows strictly after time zero are read.
    post = (
        index.select("hospitalization_id", "time_zero_dttm")
        .join(devices, on="hospitalization_id", how="inner")
        .filter(pl.col("device_dttm") > pl.col("time_zero_dttm"))
    )
    invasive = pl.col("device_class") == INVASIVE
    tracheostomy = pl.col("_trach") | (pl.col("device_class") == TRACHEOSTOMY)
    events = post.group_by("hospitalization_id").agg(
        pl.col("device_dttm").filter(invasive).min().alias("_reintubation_dttm"),
        pl.col("device_dttm").filter(tracheostomy).min().alias("_tracheostomy_dttm"),
        pl.col("time_zero_dttm").first(),
    )
    stitched = events.filter(pl.col("_reintubation_dttm") - pl.col("time_zero_dttm") < stitch)
    if stitched.height:
        raise QualificationError(
            f"{stitched.height} cohort rows return to invasive ventilation sooner than the "
            "stitch gap after time zero: the cohort was not built with this extubation contract"
        )
    return events.drop("time_zero_dttm")


def build_extubation_labels(
    cohort: pl.DataFrame,
    tables: Mapping[str, pl.DataFrame],
    config: Mapping[str, Any],
    *,
    availability: Mapping[str, Availability],
    data_config: Mapping[str, Any],
    cohort_config: Mapping[str, Any],
) -> pl.DataFrame:
    """One label row per cohort row, in cohort order, with times in hours from time zero.

    `cohort` is the extubation cohort (`build_extubation_cohort(...).cohort`); every row
    is labelled, eligible or not. `tables` needs `hospitalization`, `respiratory_support`
    and `patient`. Windows are resolved from the result with `resolve_window`.
    """
    _label_endpoints(cohort_config, data_config, config)
    _require_columns(
        cohort, "extubation cohort", {"hospitalization_id", "patient_id", "time_zero_dttm"}
    )
    _require_utc(cohort, "extubation cohort", ["time_zero_dttm"])
    if cohort["patient_id"].n_unique() != cohort.height:
        raise QualificationError("extubation cohort must hold one first extubation per patient")
    # Binds the labels to the cohort build they were derived from.
    passthrough = [name for name in ("extubation_sha256",) if name in cohort.columns]
    index = cohort.select("hospitalization_id", "patient_id", "time_zero_dttm", *passthrough)
    # Death-record consistency (exclusions.death_record): the cohort's flag (every stay of
    # the patient) OR this table's own check (the stays it was given). Flagged, not removed.
    stays = _table(tables, "hospitalization", required=True)
    own = (death_record_flags(stays.select("patient_id", "admission_dttm"),
                              _table(tables, "patient"), config)
           if {"patient_id", "admission_dttm"} <= set(stays.columns)
           else pl.DataFrame(schema={"patient_id": pl.String,
                                     "death_record_inconsistent": pl.Boolean}))
    cohort_flag = (cohort.select("patient_id", pl.col("death_record_inconsistent")
                                 .alias("_cohort_flag"))
                   if "death_record_inconsistent" in cohort.columns
                   else cohort.select("patient_id", pl.lit(False).alias("_cohort_flag")))
    death_flags = cohort_flag.join(own, on="patient_id", how="left").select(
        "patient_id",
        (pl.col("_cohort_flag").fill_null(False)
         | pl.col("death_record_inconsistent").fill_null(False))
        .alias("death_record_inconsistent"))
    frame = (
        index.join(
            _index_stays(tables, data_config),
            on="hospitalization_id",
            how="left",
            maintain_order="left",
        )
        .join(_death_times(tables), on="patient_id", how="left", maintain_order="left")
        .join(death_flags, on="patient_id", how="left", maintain_order="left")
        .join(
            _event_rows(
                index, _device_rows(tables, config, availability["respiratory_support"]), config
            ),
            on="hospitalization_id",
            how="left",
            maintain_order="left",
        )
    )
    if frame["discharge_disposition"].has_nulls():
        raise QualificationError(
            "cohort rows have no index hospitalization in the hospitalization table"
        )
    if frame["discharge_dttm"].has_nulls():
        raise QualificationError(
            "an index hospitalization has no discharge time; follow-up cannot be bounded"
        )
    if frame["_in_patient"].has_nulls():
        raise QualificationError("cohort patients are missing from the patient table")
    expired = pl.col("discharge_disposition") == "expired"
    hospice = pl.col("discharge_disposition") == "hospice"
    frame = frame.with_columns(
        pl.when(expired)
        .then(pl.min_horizontal("death_dttm", "discharge_dttm"))
        .when(pl.col("death_dttm").is_not_null())
        .then(pl.max_horizontal("death_dttm", "discharge_dttm"))
        .alias("_death_dttm"),
        pl.when(pl.col("death_dttm").is_not_null())
        .then(pl.lit("death_dttm"))
        .when(expired)
        .then(pl.lit("discharge_category"))
        .alias("death_time_source"),
    ).with_columns(
        _hours("_reintubation_dttm", "time_zero_dttm").alias("reintubation_hours"),
        _hours("_tracheostomy_dttm", "time_zero_dttm").alias("tracheostomy_hours"),
        _hours("_death_dttm", "time_zero_dttm").alias("_death_raw"),
        _hours("discharge_dttm", "time_zero_dttm").alias("_discharge_raw"),
    )
    # A discharge or death recorded before the time-zero row (late charting) is moved to
    # time zero and flagged; no time in the label artifact is negative.
    frame = frame.with_columns(
        pl.col("_death_raw").clip(lower_bound=0.0).alias("death_hours"),
        pl.col("_discharge_raw").clip(lower_bound=0.0).alias("discharge_hours"),
        ((pl.col("_death_raw") < 0).fill_null(False) | (pl.col("_discharge_raw") < 0)).alias(
            "times_clamped_to_time_zero"
        ),
    ).with_columns(
        pl.when(hospice).then(pl.col("discharge_hours")).alias("hospice_hours"),
        pl.when(expired)
        .then(pl.col("death_hours"))
        .otherwise(pl.col("discharge_hours"))
        .alias("followup_end_hours"),
        pl.when(expired)
        .then(pl.lit("death"))
        .when(hospice)
        .then(pl.lit("hospice_discharge"))
        .when(pl.col("discharge_disposition") == "unknown")
        .then(pl.lit("discharge_unknown"))
        .otherwise(pl.lit("discharge_alive"))
        .alias("followup_end_reason"),
        (
            (pl.col("death_hours") <= TERMINAL_EXTUBATION_HOURS)
            & (pl.col("reintubation_hours") > pl.col("death_hours")).fill_null(True)
        )
        .fill_null(False)
        .alias("death_within_24h_without_reintubation"),
    )
    return frame.select(
        "hospitalization_id",
        "patient_id",
        "reintubation_hours",
        "tracheostomy_hours",
        "death_hours",
        "death_time_source",
        "hospice_hours",
        "discharge_hours",
        "discharge_disposition",
        "followup_end_hours",
        "followup_end_reason",
        "death_within_24h_without_reintubation",
        "times_clamped_to_time_zero",
        "death_record_inconsistent",
        *passthrough,
    )


# --------------------------------------------------------------------- windows


def _horizon(horizon_hours: float) -> float:
    if (
        isinstance(horizon_hours, bool)
        or not isinstance(horizon_hours, (int, float))
        or not math.isfinite(horizon_hours)
        or horizon_hours <= 0
    ):
        raise QualificationError("horizon must be a finite, positive number of hours")
    return float(horizon_hours)


def resolve_window(
    labels: pl.DataFrame,
    horizon_hours: float,
    *,
    event: str = COMPOSITE,
    discharge_alive_rule: str = "censor",
) -> pl.DataFrame:
    """One state per label row for the window (0, horizon_hours] from time zero."""
    horizon = _horizon(horizon_hours)
    if event not in EVENTS:
        raise QualificationError(f"unknown event {event!r}; expected one of {EVENTS}")
    if discharge_alive_rule not in DISCHARGE_ALIVE_RULES:
        raise QualificationError(
            f"unknown discharge_alive_rule {discharge_alive_rule!r}; "
            f"expected one of {DISCHARGE_ALIVE_RULES}"
        )
    _require_columns(labels, "extubation labels", LABEL_COLUMNS)
    roles = _ROLES[event]
    # Right-closed window: an event at exactly the horizon is inside it.
    inside = {
        cause: pl.when(pl.col(f"{cause}_hours") <= horizon).then(pl.col(f"{cause}_hours"))
        for cause, _ in roles
    }
    first = pl.min_horizontal(*inside.values())
    # `roles` is in tie order: at equal times the earlier entry is the cause.
    cause = pl.coalesce(
        *[pl.when(inside[name] == first).then(pl.lit(name)) for name, _ in roles]
    )
    status = pl.coalesce(
        *[pl.when(inside[name] == first).then(pl.lit(state)) for name, state in roles]
    )
    # No event inside the window. In hospital through the horizon: event-free. Otherwise
    # follow-up ended first; only a discharge alive may be assumed event-free.
    event_free = pl.col("followup_end_hours") >= horizon
    if discharge_alive_rule == "event_free":
        event_free = event_free | (pl.col("followup_end_reason") == "discharge_alive")
    return labels.select(
        "hospitalization_id",
        "patient_id",
        pl.when(first.is_not_null())
        .then(status)
        .when(event_free)
        .then(pl.lit("negative"))
        .otherwise(pl.lit("censored"))
        .alias("status"),
        cause.alias("cause"),
        pl.when(first.is_not_null())
        .then(first)
        .when(event_free)
        .then(pl.lit(horizon))
        .otherwise(pl.col("followup_end_hours"))
        .alias("time_hours"),
    ).with_columns(
        pl.col("cause")
        .fill_null("none")
        .replace_strict(CAUSE_CODES, return_dtype=pl.Int64)
        .alias("event_type")
    )


def resolve_endpoint(
    labels: pl.DataFrame,
    name: str,
    endpoints: Mapping[str, StudyEndpoint],
    *,
    discharge_alive_rule: str = "censor",
) -> pl.DataFrame:
    """`resolve_window` for a declared endpoint; anything undeclared is refused."""
    endpoint = endpoints.get(name)
    if endpoint is None:
        raise EndpointDeclarationError(
            f"endpoint {name} is not declared under study_endpoints as label_only"
        )
    return resolve_window(
        labels,
        endpoint.horizon_hours,
        event=endpoint.event,
        discharge_alive_rule=discharge_alive_rule,
    )


def estimator_arrays(
    labels: pl.DataFrame,
    horizon_hours: float,
    *,
    event: str = COMPOSITE,
    discharge_alive_rule: str = "censor",
    coding: str = "cause",
) -> EstimatorArrays:
    """`event_time` and `event_type` for `src.eval.causal.estimators`, in label order."""
    if coding not in ("cause", "status"):
        raise QualificationError(f"unknown coding {coding!r}; expected 'cause' or 'status'")
    resolved = resolve_window(
        labels, horizon_hours, event=event, discharge_alive_rule=discharge_alive_rule
    )
    if coding == "status":
        codes = {"positive": STATUS_CODES["event"], "competing_event": STATUS_CODES["competing"]}
        event_type = resolved["status"].replace_strict(
            codes, default=STATUS_CODES["none"], return_dtype=pl.Int64
        )
    else:
        event_type = resolved["event_type"]
    return EstimatorArrays(
        event_time=resolved["time_hours"].to_numpy().astype(np.float64),
        event_type=event_type.to_numpy().astype(np.int64),
        resolved=(resolved["status"] != "censored").to_numpy(),
        codes=dict(CAUSE_CODES if coding == "cause" else STATUS_CODES),
    )


# --------------------------------------------------------------------- aggregates


def label_counts(
    cohort: pl.DataFrame, labels: pl.DataFrame, horizons: Sequence[float]
) -> dict[str, int]:
    """Aggregate counts over the primary-eligible cohort. Never by device arm.

    Per horizon the six `WINDOW_CELLS` partition the eligible patients for the composite
    endpoint. Under `censor` the censored share is the two `*_before_horizon` cells;
    under `event_free` it is `discharge_unknown_before_horizon` alone.
    """
    _require_columns(cohort, "extubation cohort", {"patient_id", "eligible"})
    eligible = labels.join(
        cohort.filter(pl.col("eligible")).select("patient_id"), on="patient_id", how="inner"
    )
    expired = pl.col("discharge_disposition") == "expired"
    timestamp = pl.col("death_time_source") == "death_dttm"
    flags = {
        "eligible_reintubation_in_index_stay": pl.col("reintubation_hours").is_not_null(),
        "eligible_death_time_known": pl.col("death_hours").is_not_null(),
        "eligible_death_within_24h_without_reintubation": pl.col(
            "death_within_24h_without_reintubation"
        ),
        "eligible_tracheostomy_without_reintubation": pl.col("tracheostomy_hours").is_not_null()
        & pl.col("reintubation_hours").is_null(),
        "eligible_times_clamped_to_time_zero": pl.col("times_clamped_to_time_zero"),
        "eligible_death_record_inconsistent": pl.col("death_record_inconsistent"),
        "eligible_discharged_expired": expired,
        "eligible_discharged_expired_with_death_dttm": expired & timestamp.fill_null(False),
        "eligible_discharged_expired_without_death_dttm": expired & ~timestamp.fill_null(False),
    }
    counts = {"patient_first_extubation": labels.height, "patient_eligible": eligible.height,
              "patient_death_record_inconsistent": int(
                  labels["death_record_inconsistent"].sum())
              if "death_record_inconsistent" in labels.columns else 0}
    counts.update({key: eligible.filter(flag).height for key, flag in flags.items()})
    for horizon in horizons:
        prefix = f"h{_horizon(horizon):g}"
        resolved = resolve_window(eligible, horizon).join(
            eligible.select("patient_id", "followup_end_reason"), on="patient_id"
        )
        positive = pl.col("status") == "positive"
        censored = pl.col("status") == "censored"
        cells = {
            "event_reintubation": positive & (pl.col("cause") == "reintubation"),
            "event_death": positive & (pl.col("cause") == "death"),
            "competing_hospice": pl.col("status") == "competing_event",
            "event_free_in_hospital": pl.col("status") == "negative",
            "discharged_alive_before_horizon": censored
            & (pl.col("followup_end_reason") == "discharge_alive"),
            "discharge_unknown_before_horizon": censored
            & (pl.col("followup_end_reason") == "discharge_unknown"),
        }
        for cell in WINDOW_CELLS:
            counts[f"{prefix}_{cell}"] = resolved.filter(cells[cell]).height
        # Death timestamps of patients who left the index stay alive (completeness of
        # out-of-hospital death ascertainment), whatever came first inside the window.
        counts[f"{prefix}_death_dttm_after_discharge"] = eligible.filter(
            timestamp & ~expired & (pl.col("death_hours") <= float(horizon))
        ).height
    return counts


def _equations(counts: Mapping[str, int]) -> list[list[str]]:
    """Published totals and the cells summing to them, `[total, *cells]`."""
    equations = [
        [
            "eligible_discharged_expired",
            "eligible_discharged_expired_with_death_dttm",
            "eligible_discharged_expired_without_death_dttm",
        ]
    ]
    suffix = f"_{WINDOW_CELLS[0]}"
    for key in counts:
        if key.endswith(suffix):
            prefix = key.removesuffix(suffix)
            equations.append(["patient_eligible", *(f"{prefix}_{cell}" for cell in WINDOW_CELLS)])
    equations = [[key for key in equation if key in counts] for equation in equations]
    return [equation for equation in equations if len(equation) > 2]


def suppress_counts(counts: Mapping[str, int], min_cell: int = 10) -> dict[str, int | str]:
    """Replace every count below `min_cell` with ``"<{min_cell}"`` and withhold a
    complementary cell (``"suppressed"``) wherever one hidden non-zero count could be
    recovered from a published total (the rule of `suppress_waterfall`). On-node display
    only: anything that leaves the node goes through the release ledger (KTD10)."""
    hidden = {key for key, value in counts.items() if value < min_cell}
    complementary: set[str] = set()
    changed = True
    while changed:
        changed = False
        for equation in _equations(counts):
            exposed = [key for key in equation if key in hidden and counts[key] != 0]
            candidates = sorted((counts[key], key) for key in equation if key not in hidden)
            if len(exposed) == 1 and candidates:
                hidden.add(candidates[0][1])
                complementary.add(candidates[0][1])
                changed = True
    return {
        key: SUPPRESSED if key in complementary else f"<{min_cell}" if key in hidden else value
        for key, value in counts.items()
    }


def _read_tables(
    base: Path,
    cohort: pl.DataFrame,
    config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    availability: Availability,
) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    """Read only the cohort's rows and the columns the labels need; hash each file."""
    flag = config["exclusions"]["tracheostomy"]["flag_col"]
    stays = cohort.select("hospitalization_id").lazy()
    patients = cohort.select("patient_id").lazy()
    sources = {
        "respiratory_support": (
            availability.file,
            lambda scan: scan.join(stays, on="hospitalization_id", how="semi").select(
                "hospitalization_id", availability.column, "device_category", flag
            ),
        ),
        "hospitalization": (
            config["source_tables"]["hospitalization"]["file"],
            lambda scan: scan.join(stays, on="hospitalization_id", how="semi").select(
                "hospitalization_id", "patient_id", "admission_dttm", "discharge_dttm",
                "discharge_category"
            ),
        ),
        "patient": (
            data_config["static_source"]["patient_file"],
            lambda scan: scan.join(patients, on="patient_id", how="semi").select(
                "patient_id", "death_dttm"
            ),
        ),
    }
    tables: dict[str, pl.DataFrame] = {}
    provenance: dict[str, str] = {}
    for name, (file, narrow) in sources.items():
        path = base / f"{file}.parquet"
        if not path.exists():
            raise QualificationError(f"required CLIF table is missing: {path.name}")
        try:
            tables[name] = narrow(pl.scan_parquet(path)).collect()
        except (pl.exceptions.ColumnNotFoundError, pl.exceptions.SchemaError) as exc:
            raise QualificationError(f"{name} is missing required columns or key types") from exc
        provenance[path.stem] = _file_sha256(path)
    return tables, provenance


def build_label_artifact(
    data_dir: str | Path,
    output: str | Path,
    *,
    cohort_artifact: str | Path,
    extubation_config: str | Path,
    data_config: str | Path,
    cohort_config: str | Path,
    artifact_policy: str | Path,
) -> dict[str, int]:
    """Label a site's extubation cohort and persist the rows; return only the counts."""
    config = load_extubation_config(extubation_config)
    policy = yaml.safe_load(Path(artifact_policy).read_text())
    destination = Path(output)
    validate_artifact_destination(destination, "patient_level_phi", policy)
    data = yaml.safe_load(Path(data_config).read_text())
    contract = yaml.safe_load(Path(cohort_config).read_text())
    # Refuse before any source table is opened.
    endpoints = _label_endpoints(contract, data, config)
    cohort_path = Path(cohort_artifact)
    if not cohort_path.exists():
        raise QualificationError(
            f"extubation cohort artifact is missing: {cohort_path.name}; build it with "
            "src.data.extubation_cohort"
        )
    cohort = pl.read_parquet(cohort_path)
    validate_extubation_artifact(cohort)
    availability = table_availability(config, data)
    tables, provenance = _read_tables(
        Path(data_dir), cohort, config, data, availability["respiratory_support"]
    )
    labels = build_extubation_labels(
        cohort,
        tables,
        config,
        availability=availability,
        data_config=data,
        cohort_config=contract,
    )
    horizons = sorted({endpoint.horizon_hours for endpoint in endpoints.values()})
    counts = label_counts(cohort, labels, horizons)
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels.with_columns(
        pl.lit(json.dumps(provenance, sort_keys=True)).alias("label_provenance_json")
    ).write_parquet(destination)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Label a site's extubation cohort artifact")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/extubation.yaml")
    parser.add_argument("--cohort", default=None, help="extubation cohort artifact")
    parser.add_argument("--out", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--cohort-config", default="configs/cohort.yaml")
    parser.add_argument("--artifact-policy", default=None)
    args = parser.parse_args()
    config = load_extubation_config(args.config)
    artifact_policy = args.artifact_policy or config["artifact_policy"]
    cohort = Path(args.cohort or config["outputs"]["cohort"])
    counts = build_label_artifact(
        args.data,
        # Default: the cohort artifact's sibling `<stem>_labels.parquet`.
        args.out or cohort.with_name(f"{cohort.stem}_labels.parquet"),
        cohort_artifact=cohort,
        extubation_config=args.config,
        data_config=args.data_config or config["data_config"],
        cohort_config=args.cohort_config,
        artifact_policy=artifact_policy,
    )
    policy = yaml.safe_load(Path(artifact_policy).read_text())
    min_cell = policy["classes"]["aggregate_no_phi"]["minimum_cell_size"]
    # Aggregate counts only: never a row, an identifier, a timestamp or an arm.
    print(json.dumps(suppress_counts(counts, min_cell), indent=2))


if __name__ == "__main__":
    main()
