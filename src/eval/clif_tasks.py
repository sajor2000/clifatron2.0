"""The standard post-24-hour CLIF prediction tasks (plan U7; R24, R38).

`configs/tasks.yaml` records the published CLIF GEM task table (Burkhart et al. 2026,
arXiv:2608.02939, Table 2) task by task: the published definition, ours, and whether ours
is `matched`, `approximated` or `excluded`. This module loads and validates that record,
derives the labels, cuts the token features, and holds the one fit/select/calibrate/
evaluate routine every row of the task table goes through (`partitioned_cell`), so the
baselines (`src/eval/baselines.py`) and the frozen probe (`src/eval/probe.py`) run on
identical rows.

Two windows, both on AVAILABILITY time (the table's `availability_col` in the data
config plus its declared lag — the same clock the tokenizer orders events by):

    features   availability time <= anchor                 (`observation_sequences`)
    labels     anchor < availability time <= discharge     (`derive_task_labels`)

A stay whose outcome was already available at or before the anchor is `prevalent` and is
left out of that task, which is the published rule ("given that it did not occur within
the first 24 hours"). No event can be both a feature and a label.

Hard rule #1 (treatments are inputs, never targets) is enforced by the loader: a
treatment-initiation task, or any task whose source table is input-only in the data
config, cannot be active.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src.data.cohort import QualificationError
from src.eval import metrics as M
from src.eval import schema as _schema
from src.eval.method3 import PartitionError

ROOT = Path(__file__).parents[2]
DEFAULT_TASKS = ROOT / "configs/tasks.yaml"
DEFAULT_DATA_CONFIG = ROOT / "configs/data.yaml"

STATUSES = ("matched", "approximated", "excluded")
SOURCE_STATUSES = ("retrieved", "unverified")
KINDS = ("threshold", "death", "treatment_initiation")
ROLES = ("fit", "selection", "calibration", "evaluation")
OPS: dict[str, Callable[[pl.Expr, float], pl.Expr]] = {
    "lt": lambda value, threshold: value < threshold,
    "le": lambda value, threshold: value <= threshold,
    "gt": lambda value, threshold: value > threshold,
    "ge": lambda value, threshold: value >= threshold,
}
# Discharge dispositions that are the end of observation, never read as survival
# (the same convention as the `gem.disposition_map` unknowns in configs/data.yaml).
UNKNOWN_DISPOSITIONS = ("", "missing", "unknown")
NOT_EVALUABLE = "not evaluable"
SUPPRESSED = "suppressed (small cell)"


class TaskConfigError(QualificationError):
    """The task suite does not describe a set of tasks this repo may run."""


class TreatmentTargetError(TaskConfigError):
    """An active task predicts a treatment (hard rule #1)."""


@dataclass(frozen=True)
class Condition:
    source: str
    concept: str
    op: str
    threshold: float
    units: tuple[str, ...] = ()


@dataclass(frozen=True)
class Task:
    name: str
    status: str
    active: bool
    kind: str
    published: str
    ours: str = ""
    how: str = ""
    reason: str = ""
    any_of: tuple[Condition, ...] = ()


@dataclass(frozen=True)
class TaskSuite:
    version: str
    source: dict[str, Any]
    anchor: dict[str, Any]
    partitions: dict[str, str]
    tasks: dict[str, Task]
    baselines: dict[str, Any] = field(default_factory=dict)
    comparators: dict[str, Any] = field(default_factory=dict)

    def active(self) -> list[Task]:
        return [task for task in self.tasks.values() if task.active]

    def comparison(self) -> dict[str, Any]:
        """The published-versus-ours record the report carries (R24): which tasks were
        matched, which approximated and how, which left out and why."""
        tasks = {}
        for task in self.tasks.values():
            row = {"status": task.status, "active": task.active,
                   "published": task.published}
            for key in ("ours", "how", "reason"):
                if getattr(task, key):
                    row[key] = getattr(task, key)
            tasks[task.name] = row
        return {
            "source": dict(self.source),
            "anchor": dict(self.anchor),
            "counts": {status: sum(t.status == status for t in self.tasks.values())
                       for status in STATUSES},
            "tasks": tasks,
        }


# ------------------------------------------------------------------------- loader
def _text(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _condition(task: str, raw: object, tables: dict) -> Condition:
    if not isinstance(raw, dict):
        raise TaskConfigError(f"task {task!r}: each any_of entry must be a mapping")
    missing = sorted({"source", "concept", "op", "threshold"} - set(raw))
    if missing:
        raise TaskConfigError(f"task {task!r}: condition is missing {', '.join(missing)}")
    if raw["op"] not in OPS:
        raise TaskConfigError(
            f"task {task!r}: unknown comparator {raw['op']!r}; expected one of {sorted(OPS)}")
    table = tables.get(raw["source"])
    if table is None:
        raise TaskConfigError(
            f"task {task!r}: source {raw['source']!r} is not a table in the data config")
    if table.get("input_only"):
        raise TreatmentTargetError(
            f"task {task!r} reads its label from {raw['source']!r}, an input-only "
            "(treatment or context) table. Hard rule #1: treatments are model inputs, "
            "never prediction targets.")
    if not table.get("concept_col") or not table.get("value_col"):
        raise TaskConfigError(
            f"task {task!r}: source {raw['source']!r} is not a long measurement table")
    return Condition(str(raw["source"]), str(raw["concept"]), str(raw["op"]),
                     float(raw["threshold"]), tuple(str(u) for u in raw.get("units") or ()))


def _task(name: str, raw: object, tables: dict) -> Task:
    if not isinstance(raw, dict):
        raise TaskConfigError(f"task {name!r} must be a mapping")
    status, kind = raw.get("status"), raw.get("kind")
    if status not in STATUSES:
        raise TaskConfigError(f"task {name!r}: status must be one of {STATUSES}, got {status!r}")
    if kind not in KINDS:
        raise TaskConfigError(f"task {name!r}: kind must be one of {KINDS}, got {kind!r}")
    if not _text(raw.get("published")):
        raise TaskConfigError(f"task {name!r} must record its published definition")
    active = raw.get("active")
    if not isinstance(active, bool):
        raise TaskConfigError(f"task {name!r}: active must be true or false")
    if active and kind == "treatment_initiation":
        raise TreatmentTargetError(
            f"task {name!r} predicts treatment initiation and is marked active. Hard "
            "rule #1: treatments are model inputs, never prediction targets. Record "
            "it as `status: excluded`, `active: false`.")
    if kind == "treatment_initiation" and status != "excluded":
        raise TreatmentTargetError(
            f"task {name!r} predicts treatment initiation; its status must be excluded "
            "(hard rule #1)")
    if status == "excluded":
        if active:
            raise TaskConfigError(f"task {name!r} is excluded and cannot be active")
        if not _text(raw.get("reason")):
            raise TaskConfigError(f"excluded task {name!r} must say why (reason)")
    else:
        if not _text(raw.get("ours")):
            raise TaskConfigError(f"task {name!r} must record our definition (ours)")
        if status == "approximated" and not _text(raw.get("how")):
            raise TaskConfigError(f"approximated task {name!r} must say how it differs")
    conditions: tuple[Condition, ...] = ()
    if kind == "threshold":
        entries = raw.get("any_of")
        if not isinstance(entries, list) or not entries:
            raise TaskConfigError(f"threshold task {name!r} needs a non-empty any_of")
        conditions = tuple(_condition(name, entry, tables) for entry in entries)
    return Task(name, status, active, kind, _text(raw["published"]), _text(raw.get("ours")),
                _text(raw.get("how")), _text(raw.get("reason")), conditions)


def load_task_suite(path: str | Path = DEFAULT_TASKS, *,
                    data_cfg: dict | None = None) -> TaskSuite:
    """Load and validate the task suite against a data config (default: the repo's).

    The data config decides which tables may carry a label: an input-only table cannot.
    """
    blob = yaml.safe_load(Path(path).read_text())
    if data_cfg is None:
        data_cfg = yaml.safe_load(DEFAULT_DATA_CONFIG.read_text())
    tables = data_cfg.get("tables") or {}
    source = blob.get("source") or {}
    if source.get("status") not in SOURCE_STATUSES:
        raise TaskConfigError(
            f"source.status must be one of {SOURCE_STATUSES}: say whether the published "
            "task table was actually retrieved")
    raw_tasks = blob.get("tasks")
    if not isinstance(raw_tasks, dict) or not raw_tasks:
        raise TaskConfigError("the task suite declares no tasks")
    tasks = {name: _task(name, raw, tables) for name, raw in raw_tasks.items()}
    published = source.get("n_published_tasks")
    if source["status"] == "retrieved" and published != len(tasks):
        raise TaskConfigError(
            f"the published table has {published} tasks but {len(tasks)} are recorded; "
            "every published task must appear, including the ones left out")
    partitions = blob.get("partitions") or {}
    if set(partitions) != set(ROLES):
        raise TaskConfigError(f"partitions must name exactly the roles {ROLES}")
    validate_roles(partitions)
    anchor = {key: _text(value) for key, value in (blob.get("anchor") or {}).items()}
    if anchor.get("alignment") not in ("matched", "approximated"):
        raise TaskConfigError("anchor.alignment must be matched or approximated")
    return TaskSuite(
        version=str(blob.get("suite_version", "")),
        source={key: _text(value) if isinstance(value, str) else value
                for key, value in source.items()},
        anchor=anchor,
        partitions=dict(partitions),
        tasks=tasks,
        baselines=blob.get("baselines") or {},
        comparators=blob.get("comparators") or {},
    )


def validate_roles(roles: dict[str, str]) -> None:
    """Rows that fit or calibrate a predictor are never the rows that score it."""
    if len({roles["fit"], roles["selection"], roles["calibration"]}) != 3:
        raise PartitionError("fit, selection and calibration must be three partitions")
    if roles["evaluation"] in (roles["fit"], roles["calibration"]):
        raise PartitionError(
            f"evaluation partition {roles['evaluation']!r} is also a fit or calibration "
            "partition; a predictor cannot be scored on rows that shaped it")


# ------------------------------------------------------------------------- labels
def load_task_events(data_dir: str | Path, suite: TaskSuite,
                     data_cfg: dict) -> dict[str, pl.DataFrame | None]:
    """Per source table the active tasks read: its observations on the availability
    clock (declared lag applied, as the tokenizer does before windowing), or None when
    the site does not have the table."""
    from src.data.tokenize import validate_table_availability
    from src.eval.clif_auto_labeler import _outcome_events

    sources = sorted({c.source for task in suite.active() for c in task.any_of})
    specs = {name: data_cfg["tables"][name] for name in sources}
    lags = validate_table_availability(specs) if specs else {}
    events: dict[str, pl.DataFrame | None] = {}
    for name, spec in specs.items():
        frame = _outcome_events(Path(data_dir), spec)
        lag = lags[name]["lag_minutes"]
        if frame is not None and lag:
            frame = frame.with_columns(pl.col("dttm") + pl.duration(minutes=lag))
        events[name] = frame
    return events


def _require_utc(frame: pl.DataFrame, name: str, columns: list[str]) -> None:
    for column in columns:
        dtype = frame.schema.get(column)
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise QualificationError(f"{name}.{column} must be timezone-aware UTC")


def _labelled(stays: pl.DataFrame, task: str, status: pl.Expr) -> pl.DataFrame:
    return stays.select(
        "hospitalization_id",
        pl.when(status == "positive").then(True)
        .when(status == "negative").then(False)
        .otherwise(None).cast(pl.Boolean).alias(task),
        status.alias(f"{task}_status"),
    )


def _death_states(stays: pl.DataFrame, task: Task) -> pl.DataFrame:
    disposition = (pl.col("discharge_category").cast(pl.String)
                   .str.strip_chars().str.to_lowercase())
    status = (
        pl.when(disposition.is_null() | disposition.is_in(UNKNOWN_DISPOSITIONS))
        .then(pl.lit("not_ascertainable"))
        .when(disposition == "expired").then(pl.lit("positive"))
        .otherwise(pl.lit("negative"))
    )
    return _labelled(stays, task.name, status)


def _check_units(rows: pl.DataFrame, condition: Condition, task: Task) -> None:
    if not condition.units:
        return
    allowed = [unit.strip().lower() for unit in condition.units]
    charted = rows["unit"].drop_nulls().str.strip_chars().str.to_lowercase().unique()
    wrong = sorted(set(charted.to_list()) - set(allowed))
    if wrong:
        raise QualificationError(
            f"non-canonical unit for task {task.name} ({condition.concept}): "
            f"{', '.join(wrong)}; expected {', '.join(condition.units)}")


def _threshold_states(stays: pl.DataFrame, task: Task,
                      events: dict[str, pl.DataFrame | None]) -> pl.DataFrame:
    crossings = []
    for condition in task.any_of:
        observations = events.get(condition.source)
        if observations is None:
            # One arm of a composite cannot stand in for the whole outcome.
            return _labelled(stays, task.name, pl.lit("unsupported_at_site"))
        rows = observations.filter(pl.col("concept") == condition.concept)
        _check_units(rows, condition, task)
        # NaN compares greater than every number in polars, so a NaN would cross every
        # `>=` cut-off; only finite, timestamped values can be an outcome.
        crossings.append(
            rows.filter(pl.col("value").is_finite() & pl.col("dttm").is_not_null()
                        & OPS[condition.op](pl.col("value"), condition.threshold))
            .select("hospitalization_id", "dttm"))
    after_anchor = pl.col("dttm") > pl.col("anchor_dttm")
    flags = (
        pl.concat(crossings)
        .join(stays.select("hospitalization_id", "anchor_dttm", "discharge_dttm"),
              on="hospitalization_id", how="inner")
        .group_by("hospitalization_id")
        .agg((~after_anchor).any().alias("_prior"),
             (after_anchor & (pl.col("dttm") <= pl.col("discharge_dttm"))).any()
             .alias("_incident"))
    )
    status = (
        pl.when(pl.col("_prior").fill_null(False)).then(pl.lit("prevalent"))
        .when(pl.col("_incident").fill_null(False)).then(pl.lit("positive"))
        .otherwise(pl.lit("negative"))
    )
    return _labelled(stays.join(flags, on="hospitalization_id", how="left"),
                     task.name, status)


def derive_task_labels(episodes: pl.DataFrame, events: dict[str, pl.DataFrame | None],
                       suite: TaskSuite) -> pl.DataFrame:
    """One row per ELIGIBLE stay: `partition`, and per active task a Boolean `<task>`
    (null when the stay is not at risk or not ascertainable) and `<task>_status`.

    `events` maps each source table to its observations (`hospitalization_id`, `dttm` =
    availability time, `concept`, `value`, `unit`) or None when the site lacks it.
    Patient-level: the result stays in restricted storage.
    """
    required = {"hospitalization_id", "partition", "eligible", "anchor_dttm",
                "discharge_dttm", "discharge_category"}
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise QualificationError(
            f"episode artifact is missing required columns: {', '.join(missing)}")
    _require_utc(episodes, "episodes", ["anchor_dttm", "discharge_dttm"])
    for name, frame in events.items():
        if frame is not None:
            _require_utc(frame, name, ["dttm"])
    stays = episodes.filter(pl.col("eligible")).select(
        "hospitalization_id", "partition", "anchor_dttm", "discharge_dttm",
        "discharge_category")
    labels = stays.select("hospitalization_id", "partition")
    for task in suite.active():
        states = (_death_states(stays, task) if task.kind == "death"
                  else _threshold_states(stays, task, events))
        labels = labels.join(states, on="hospitalization_id", how="left")
    return labels


def label_site(data_dir: str | Path, episodes: pl.DataFrame, suite: TaskSuite,
               data_cfg: dict) -> pl.DataFrame:
    """`derive_task_labels` for a site directory of CLIF tables."""
    return derive_task_labels(episodes, load_task_events(data_dir, suite, data_cfg), suite)


# ----------------------------------------------------------------------- features
def observation_sequences(shards: pl.DataFrame, episodes: pl.DataFrame) -> pl.DataFrame:
    """Per stay, the tokens available at or before its anchor: `hosp_id`, `partition`,
    `token`, `pos_min`.

    24 h shards (`events.parquet`; positions = minutes since ICU admission) are cut at
    each stay's own anchor minute from the episode artifact, so a longer stream handed
    in by mistake still yields hour-24 features. Full-hospitalization GEM windows
    (`gem_events.parquet`) are stitched in window order and cut at the stay's
    `anchor_idx`, the index of its last token at or before the anchor.
    """
    if "trajectory" in shards.columns and (shards["trajectory"] == "hospitalization").any():
        if not (shards["trajectory"] == "hospitalization").all():
            raise QualificationError("shards mix 24 h rows and full-hospitalization windows")
        stays = (
            shards.sort("hosp_id", "continuation_index")
            .group_by("hosp_id", maintain_order=True)
            .agg(pl.col("partition").first(),
                 pl.col("token").explode(empty_as_null=False, keep_nulls=False),
                 pl.col("pos_min").explode(empty_as_null=False, keep_nulls=False),
                 pl.col("anchor_idx").first())
        )
        keep = pl.col("anchor_idx") + 1
        return stays.select("hosp_id", "partition", pl.col("token").list.head(keep),
                            pl.col("pos_min").list.head(keep))
    anchors = episodes.select(
        pl.col("hospitalization_id").alias("hosp_id"),
        (pl.col("anchor_dttm") - pl.col("icu_admit_dttm")).dt.total_minutes()
        .cast(pl.Int64).alias("_anchor_min"))
    joined = shards.join(anchors, on="hosp_id", how="left")
    if joined["_anchor_min"].has_nulls():
        raise QualificationError(
            "token shards contain stays with no anchor in the episode artifact")
    available = (pl.col("pos_min") - pl.col("_anchor_min")).list.eval(
        (pl.element() <= 0).arg_true())
    return joined.select("hosp_id", "partition",
                         pl.col("token").list.gather(available),
                         pl.col("pos_min").list.gather(available))


def token_count_matrix(sequences: list[list[int]], vocab_size: int):
    """Sparse [stays, vocab] matrix of how often each token occurs in each sequence."""
    from scipy.sparse import csr_matrix

    lengths = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=len(sequences))
    indices = np.fromiter((t for s in sequences for t in s), dtype=np.int64,
                          count=int(lengths.sum()))
    if len(indices) and (indices.min() < 0 or indices.max() >= vocab_size):
        raise ValueError(f"token id outside the vocabulary of {vocab_size}")
    indptr = np.concatenate([[0], np.cumsum(lengths)])
    counts = csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr),
                        shape=(len(sequences), vocab_size))
    counts.sum_duplicates()
    return counts


# -------------------------------------------------------------- one cell, one split
Fitter = Callable[[Any, np.ndarray, Any, np.ndarray | None], tuple[Callable, dict]]


def grid_candidates(cfg: dict) -> list[dict]:
    """Every combination of a predictor's `grid` (sorted keys, stable order)."""
    grid = cfg.get("grid") or {}
    keys = sorted(grid)
    return [dict(zip(keys, values))
            for values in itertools.product(*(grid[key] for key in keys))]


def select_candidate(cfg: dict, fit_one: Callable[[dict], tuple[Callable, dict]],
                     X_sel, y_sel: np.ndarray | None) -> tuple[Callable, dict]:
    """Choose among a predictor's candidates by AUROC on the selection rows.

    `fit_one(params)` fits on the fit rows (its closure) and returns `(logits_fn,
    extra_info)`. With no usable selection rows the configured `default` is fitted and
    nothing is compared. Ties keep the earlier candidate.
    """
    if X_sel is None:
        params = dict(cfg["default"])
        logits_fn, extra = fit_one(params)
        return logits_fn, {**params, **extra}
    best = None
    for params in grid_candidates(cfg) or [dict(cfg["default"])]:
        logits_fn, extra = fit_one(params)
        score = roc_auc_score(y_sel, logits_fn(X_sel))
        if best is None or score > best[0]:
            best = (score, logits_fn, {**params, **extra})
    return best[1], best[2]


def not_evaluable(status: str, reason: str | None, n: int | None = None) -> dict:
    """A cell that carries a status instead of a number (and a size band, never n).
    A single-class cell reads "not evaluable"; one under the minimum cell size reads
    "suppressed"."""
    small = status in (_schema.INSUFFICIENT_N, _schema.SMALL_CELL_SUPPRESSED)
    cell = {"status": status, "reason": reason,
            "display": SUPPRESSED if small else NOT_EVALUABLE}
    if n is not None:
        cell["n_band"] = _schema.n_band(n)
    return cell


def partitioned_cell(fit: Fitter, X, y: np.ndarray, partitions: np.ndarray,
                     roles: dict[str, str], *, site: str) -> dict:
    """Fit, select, calibrate and score one task for one predictor on fixed roles.

    `fit(X_fit, y_fit, X_sel, y_sel)` returns `(logits_fn, info)`; it is handed the
    fit-partition rows, and the selection-partition rows only for choosing among
    candidates (`None` when that partition has a single class). The calibration and
    evaluation rows never reach it: the temperature is fitted here on the calibration
    partition and the panel is computed here on the evaluation partition.

    `y` is float with NaN for stays outside the task (prevalent, not ascertainable).
    A cell that cannot be released or scored returns a status, never a number.
    """
    validate_roles(roles)
    y = np.asarray(y, dtype=float)
    partitions = np.asarray(partitions)
    at_risk = ~np.isnan(y)

    def rows(role: str) -> np.ndarray:
        return np.flatnonzero(at_risk & (partitions == roles[role]))

    evaluate = rows("evaluation")
    y_eval = y[evaluate].astype(int)
    # Suppress BEFORE fitting or scoring: a single-class or small cell is a status.
    status, reason = _schema.suppress_cell(len(y_eval), int(y_eval.sum()))
    if status != _schema.EVALUABLE:
        return not_evaluable(status, f"{roles['evaluation']} partition: {reason}",
                             len(y_eval))
    fit_rows = rows("fit")
    y_fit = y[fit_rows].astype(int)
    if len(np.unique(y_fit)) < 2:
        return not_evaluable(
            _schema.SINGLE_CLASS,
            f"{roles['fit']} partition: outcome has a single class, nothing to fit")
    select = rows("selection")
    y_sel = y[select].astype(int)
    selectable = len(np.unique(y_sel)) > 1
    logits_fn, info = fit(X[fit_rows], y_fit,
                          X[select] if selectable else None,
                          y_sel if selectable else None)

    calibrate = rows("calibration")
    y_cal = y[calibrate].astype(int)
    temperature = None
    if len(np.unique(y_cal)) > 1:
        temperature = M.fit_temperature(logits_fn(X[calibrate]), y_cal,
                                        partition=f"{site}:{roles['calibration']}")
    logits = np.asarray(logits_fn(X[evaluate]), dtype=float)
    cell = M.full_panel(expit(logits), y_eval, logits=logits, temperature=temperature,
                        partition=f"{site}:{roles['evaluation']}")
    cell["status"] = _schema.EVALUABLE
    cell["prevalence"] = _schema.round_prevalence(cell.get("prevalence"))
    cell["calibrated"] = temperature is not None
    cell["selection"] = info if selectable else {
        **info, "note": f"defaults: {roles['selection']} partition has a single class"}
    return cell
