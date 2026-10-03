"""Count and token baselines, and the aggregate CLIF task table (plan U7; R24, R38).

    # one command, data-free: synthetic site -> baselines + probe -> aggregate table
    python -m src.eval.baselines --synthetic

    # a site (development view: scored on the validation partition)
    python -m src.eval.baselines --site mimic --data $MIMIC_DIR \
        --episodes output/intermediate_phi/episodes.parquet \
        --shards output/intermediate_phi/mimic/events.parquet \
        --vocab output/intermediate_phi/mimic/vocab.json --eval-partition validation

Rows of the table, every one on the same stays and the same partition roles
(`clif_tasks.partitioned_cell`):

    lightgbm_counts        gradient-boosted trees on token counts
    logistic_tokens        L2 logistic regression on token indicators
    decile_ntp_probe       the published CLIF GEM replica (decile next-token + probe)
    clifatron_0p5b_probe   the released CLIFATRON 0.5B checkpoint, frozen (larger comparator)

The two comparator rows are filled when their checkpoint and token sequences are staged
(`configs/tasks.yaml` `comparators`, or the CLI flags); otherwise they read "not
available" and the table still builds.

Baselines are fitted on the fit partition only. Their hyperparameters (and LightGBM's
stopping round) are chosen on the selection partition only. The calibration and
evaluation partitions never reach a fitter. `internal_test` is sealed: it is scored only
when `--final-evaluation` is passed.

The output is AGGREGATE ONLY: one cell per (row, task) with the TRIPOD+AI panel, or a
status when the cell is single-class or under the minimum cell size. No identifiers, no
rows, no paths. LightGBM is a repo-side dependency (KTD7); this module is not vendored
into `clif-validate`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from src.data.cohort import validate_artifact_destination, validate_episode_artifact
from src.data.tokenization_report import assert_aggregate_only
from src.eval import schema as _schema
from src.eval.clif_tasks import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_TASKS,
    ROOT,
    Fitter,
    TaskSuite,
    label_site,
    load_task_suite,
    observation_sequences,
    partitioned_cell,
    select_candidate,
    token_count_matrix,
    validate_roles,
)
from src.eval.method3 import PartitionError
from src.eval.probe import frozen_probe, trunk_states

DEFAULT_TRAIN_CONFIG = ROOT / "configs/train.yaml"
DEFAULT_POLICY = ROOT / "configs/artifact_policy.yaml"
DEFAULT_OUT = "output/final_no_phi/clif_task_baselines.json"

ROW_LIGHTGBM = "lightgbm_counts"
ROW_LOGISTIC = "logistic_tokens"
ROW_SYNTHETIC_PROBE = "synthetic_trunk_probe"
NOT_AVAILABLE = "not available"
SYNTHETIC_TASK_SITE = "SYNTH-TASKS"


# ----------------------------------------------------------------------- fitters
def _lightgbm_threads(requested: int) -> int:
    """LightGBM threads to use. One on macOS: torch and LightGBM each bundle an OpenMP
    runtime there, and a multi-threaded LightGBM fit in a process that loaded torch
    first hangs or segfaults (seen on the dev Macs with lightgbm 4.7 / torch 2.13).
    Linux (the L40 node) uses the configured count."""
    return 1 if sys.platform == "darwin" else requested


def lightgbm_fitter(cfg: dict, *, seed: int = 0) -> Fitter:
    """Gradient-boosted trees on token counts. Trained on the fit rows; the selection
    rows stop the boosting (AUROC) and choose among `grid` candidates."""

    def fit(X_fit, y_fit, X_sel, y_sel):
        import lightgbm as lgb

        base = {
            "learning_rate": float(cfg["learning_rate"]),
            "min_child_samples": int(cfg["min_child_samples"]),
            "colsample_bytree": float(cfg["colsample_bytree"]),
            "n_jobs": _lightgbm_threads(int(cfg.get("n_jobs", 1))),
            "random_state": seed,
            "verbose": -1,
        }

        def fit_one(params):
            if X_sel is None:
                model = lgb.LGBMClassifier(
                    n_estimators=int(cfg.get("default_n_estimators", 200)), **base, **params)
                model.fit(X_fit, y_fit)
                trees = model.n_estimators
            else:
                model = lgb.LGBMClassifier(n_estimators=int(cfg["n_estimators"]),
                                           **base, **params)
                stop = {"eval_metric": "auc", "callbacks": [lgb.early_stopping(
                    int(cfg["early_stopping_rounds"]), first_metric_only=True,
                    verbose=False)]}
                try:
                    # LightGBM 4.7 deprecates `eval_set` for `eval_X` / `eval_y`.
                    model.fit(X_fit, y_fit, eval_X=X_sel, eval_y=y_sel, **stop)
                except TypeError as exc:
                    if "eval_X" not in str(exc):
                        raise
                    model.fit(X_fit, y_fit, eval_set=[(X_sel, y_sel)], **stop)
                trees = model.best_iteration_ or model.n_estimators
            return (lambda X: model.predict_proba(X, raw_score=True)), {"n_trees": int(trees)}

        return select_candidate(cfg, fit_one, X_sel, y_sel)

    return fit


def logistic_fitter(cfg: dict, *, seed: int = 0) -> Fitter:
    """L2 logistic regression on token indicators (the published form) or counts.
    Trained on the fit rows; `C` is chosen on the selection rows. `seed` is accepted
    for symmetry with the other fitters; lbfgs is deterministic."""
    from scipy.sparse import issparse

    indicators = cfg.get("features", "indicators") == "indicators"

    def features(X):
        if not indicators:
            return X
        if issparse(X):
            X = X.copy()
            X.data = (X.data > 0).astype(np.float32)
            return X
        return (np.asarray(X) > 0).astype(np.float32)

    def fit(X_fit, y_fit, X_sel, y_sel):
        from sklearn.linear_model import LogisticRegression

        Z_fit = features(X_fit)

        def fit_one(params):
            # No `penalty` argument: it is deprecated from scikit-learn 1.8, and the
            # default (l1_ratio=0) is the L2 penalty in every supported version.
            model = LogisticRegression(C=float(params["C"]), solver="lbfgs",
                                       max_iter=int(cfg["max_iter"]))
            model.fit(Z_fit, y_fit)
            return (lambda X: model.decision_function(features(X))), {}

        return select_candidate(cfg, fit_one, X_sel, y_sel)

    return fit


# ------------------------------------------------------------------------- rows
@dataclass
class TaskData:
    """Stays with both a label row and at least one hour-24 token, aligned. Patient-level:
    never exported; the report is built from it."""

    partitions: np.ndarray
    sequences: list[list[int]]
    positions: list[list[int]] | None
    labels: dict[str, np.ndarray]          # task -> float, NaN outside the task
    n_without_tokens: int


def align(sequences: pl.DataFrame, labels: pl.DataFrame, suite: TaskSuite) -> TaskData:
    """Join token sequences to labels. The partition role comes from the labels (the
    episode artifact), never from the sequence file."""
    columns = ["hosp_id", "token"] + (["pos_min"] if "pos_min" in sequences.columns else [])
    joined = labels.join(
        sequences.filter(pl.col("token").list.len() > 0).select(columns),
        left_on="hospitalization_id", right_on="hosp_id", how="inner",
    ).sort("hospitalization_id")
    return TaskData(
        partitions=np.asarray(joined["partition"].to_list()),
        sequences=joined["token"].to_list(),
        positions=joined["pos_min"].to_list() if "pos_min" in joined.columns else None,
        labels={task.name: joined[task.name].cast(pl.Float64).to_numpy()
                for task in suite.active()},
        n_without_tokens=labels.height - joined.height,
    )


def _clean(value: Any) -> Any:
    """Strict-JSON form of a cell: numpy scalars unwrapped, non-finite floats -> None."""
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def available_row(row_id: str, label: str, cells: dict[str, dict], data: TaskData,
                  **extra: Any) -> dict:
    scored = [task for task, cell in cells.items() if cell["status"] == _schema.EVALUABLE]
    mean = None
    if scored:
        mean = {"auroc": float(np.mean([cells[t]["auroc"] for t in scored])),
                "auprc": float(np.mean([cells[t]["auprc"] for t in scored])),
                "n_tasks": len(scored), "tasks": scored}
    return _clean({"row": row_id, "label": label, "status": "available", "cells": cells,
                   "mean": mean,
                   "stays_without_tokens": _schema.band_dropped_count(data.n_without_tokens),
                   **extra})


def unavailable_row(row_id: str, label: str, reason: str) -> dict:
    return {"row": row_id, "label": label, "status": "not_available",
            "display": NOT_AVAILABLE, "reason": reason}


def predictor_row(row_id: str, label: str, fitter: Fitter, X, data: TaskData,
                  suite: TaskSuite, roles: dict[str, str], *, site: str,
                  **extra: Any) -> dict:
    cells = {task.name: partitioned_cell(fitter, X, data.labels[task.name],
                                         data.partitions, roles, site=site)
             for task in suite.active()}
    return available_row(row_id, label, cells, data, **extra)


def probe_row(row_id: str, label: str, states: np.ndarray, data: TaskData,
              suite: TaskSuite, roles: dict[str, str], *, site: str, cfg: dict,
              **extra: Any) -> dict:
    cells = {task.name: frozen_probe(data.labels[task.name], data.partitions, roles,
                                     site=site, states=states, cfg=cfg["probe"],
                                     seed=int(cfg.get("seed", 0)))
             for task in suite.active()}
    return available_row(row_id, label, cells, data, features="frozen hour-24 state",
                         **extra)


def comparator_row(row_id: str, spec: dict, labels: pl.DataFrame,
                   episodes: pl.DataFrame | None, suite: TaskSuite, roles: dict[str, str],
                   *, site: str, cfg: dict, device: str = "cpu",
                   batch_size: int = 16) -> dict:
    """A frozen-probe row for a staged checkpoint, or a "not available" row.

    Staged means both paths exist: `checkpoint` (a transformers-format trunk directory)
    and `shards` (that model's own hour-24 token sequences: `hosp_id` + `token`). A file
    with `pos_min` is cut at each stay's anchor here; one without is taken as tokenized,
    and the row says so. The reason never carries a local path.
    """
    label = str(spec.get("label") or row_id)
    checkpoint, shards = spec.get("checkpoint"), spec.get("shards")
    if not checkpoint or not Path(checkpoint).is_dir():
        return unavailable_row(row_id, label, "no checkpoint staged")
    if not shards or not Path(shards).is_file():
        return unavailable_row(
            row_id, label, "checkpoint staged, but its hour-24 token sequences are not")
    from src.model.head_adapter import load_backbone

    frame = pl.read_parquet(shards)
    if "pos_min" in frame.columns and episodes is not None:
        if "partition" not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("partition"))
        frame = observation_sequences(frame, episodes).drop("pos_min")
        window = "cut at each stay's anchor"
    else:
        frame = frame.select("hosp_id", "token")
        window = "as tokenized by the comparator (not re-checked)"
    data = align(frame, labels, suite)
    states = trunk_states(load_backbone(str(checkpoint)), data.sequences, device=device,
                          batch_size=batch_size)
    return probe_row(row_id, label, states, data, suite, roles, site=site, cfg=cfg,
                     feature_window=window)


# ----------------------------------------------------------------------- report
def check_evaluation_partition(roles: dict[str, str], data_contract: dict, *,
                               final_evaluation: bool) -> None:
    """Refuse to score a sealed partition unless the caller names the final evaluation."""
    validate_roles(roles)
    sealed = set(data_contract.get("sealed_partitions") or ())
    if roles["evaluation"] in sealed and not final_evaluation:
        raise PartitionError(
            f"partition {roles['evaluation']!r} is sealed (final evaluation once the "
            "model is frozen). Pass --eval-partition validation for a development "
            "view, or --final-evaluation to score it.")


def build_report(*, suite: TaskSuite, site: str, roles: dict[str, str], rows: list[dict],
                 synthetic: bool) -> dict:
    comparison = suite.comparison()
    provenance = (
        f"Published task table: {suite.source['status']} (arXiv:{suite.source.get('arxiv')}). "
        "'matched' compares the outcome definition; time zero is "
        f"{suite.anchor['alignment']}: {suite.anchor.get('how', '')}").strip()
    means = (
        "Mean AUROC/AUPRC are over the evaluable tasks listed in each row's mean; the "
        f"published means are over all {len(suite.tasks)} tasks, "
        f"{comparison['counts']['excluded']} of which are left out here.")
    notes = [provenance, means]
    if roles["evaluation"] == roles["selection"]:
        notes.append(
            "Development view: the evaluation partition is also the selection "
            "partition, so tuned rows are optimistic.")
    if synthetic:
        notes.append("Synthetic site: this run proves the machinery, not a result.")
    return {
        "report": "clif_task_suite",
        "suite_version": suite.version,
        "site": site,
        "synthetic": synthetic,
        "roles": dict(roles),
        "min_cell_size": _schema.min_cell_size(),
        "comparison": comparison,
        "rows": rows,
        "notes": notes,
    }


def _cell_text(cell: dict | None) -> str:
    if cell is None:
        return "—"
    if cell["status"] != _schema.EVALUABLE:
        return cell["display"]
    return f"{cell['auroc']:.3f} / {cell['auprc']:.3f} (n={cell['n']})"


def render_table(report: dict) -> str:
    """The table as text: one line per published task, one column per available row
    (AUROC / AUPRC, evaluation n), then the rows that are not available."""
    available = [row for row in report["rows"] if row["status"] == "available"]
    header = ["task", "vs published"] + [row["label"] for row in available]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for name, task in report["comparison"]["tasks"].items():
        cells = [_cell_text(row["cells"].get(name)) for row in available]
        lines.append("| " + " | ".join([name, task["status"], *cells]) + " |")
    means = [f"{row['mean']['auroc']:.3f} / {row['mean']['auprc']:.3f} "
             f"({row['mean']['n_tasks']} tasks)" if row.get("mean") else "—"
             for row in available]
    lines.append("| " + " | ".join(["mean", "", *means]) + " |")
    lines.append("")
    for row in report["rows"]:
        if row["status"] != "available":
            lines.append(f"- {row['label']}: {row['display']} ({row['reason']})")
    lines.extend(f"- {note}" for note in report.get("notes", []))
    return "\n".join(lines)


def write_report(report: dict, path: str | Path, identifiers: Iterable[str],
                 policy: dict | None = None) -> Path:
    """Write the aggregate table: destination class checked, identifiers refused,
    strict JSON."""
    path = Path(path)
    if policy is None:
        policy = yaml.safe_load(DEFAULT_POLICY.read_text())
    validate_artifact_destination(path, "aggregate_no_phi", policy)
    assert_aggregate_only(report, identifiers)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    return path


# -------------------------------------------------------------------------- run
_SHARD_COLUMNS = ("hosp_id", "partition", "token", "pos_min", "trajectory",
                  "continuation_index", "anchor_idx")


def read_shards(path: str | Path) -> pl.DataFrame:
    """Only the columns the features need (a shard also carries soft tokens and values)."""
    present = pl.read_parquet_schema(path)
    return pl.read_parquet(path, columns=[c for c in _SHARD_COLUMNS if c in present])


def _site_report(*, site: str, data_dir: str | Path, episodes: pl.DataFrame,
                 shards: pl.DataFrame, vocab_size: int, suite: TaskSuite, data_cfg: dict,
                 roles: dict[str, str], trunks: Iterable[dict] = (),
                 comparators: dict | None = None, synthetic: bool = False,
                 device: str = "cpu") -> tuple[dict, list[str]]:
    """Labels, features and every row for one site. Returns the report and the
    identifiers it must not contain."""
    validate_episode_artifact(episodes)
    labels = label_site(data_dir, episodes, suite, data_cfg)
    data = align(observation_sequences(shards, episodes), labels, suite)
    cfg = suite.baselines
    seed = int(cfg.get("seed", 0))
    counts = token_count_matrix(data.sequences, vocab_size)
    logistic_features = cfg["logistic"].get("features", "indicators")
    rows = [
        predictor_row(ROW_LIGHTGBM, "LightGBM on token counts",
                      lightgbm_fitter(cfg["lightgbm"], seed=seed), counts, data, suite,
                      roles, site=site, features="token counts"),
        predictor_row(ROW_LOGISTIC, f"Logistic regression on token {logistic_features}",
                      logistic_fitter(cfg["logistic"], seed=seed), counts, data, suite,
                      roles, site=site, features=f"token {logistic_features}"),
    ]
    for trunk in trunks:
        states = trunk_states(trunk["trunk"], data.sequences,
                              data.positions if trunk.get("positions") else None,
                              device=device)
        rows.append(probe_row(trunk["row"], trunk["label"], states, data, suite, roles,
                              site=site, cfg=cfg))
    for row_id, spec in (comparators if comparators is not None
                         else suite.comparators).items():
        rows.append(comparator_row(row_id, spec, labels, episodes, suite, roles,
                                   site=site, cfg=cfg, device=device))
    report = build_report(suite=suite, site=site, roles=roles, rows=rows,
                          synthetic=synthetic)
    identifiers = [str(value) for column in ("hospitalization_id", "patient_id")
                   for value in episodes[column].drop_nulls().to_list()]
    assert_aggregate_only(report, identifiers)
    return report, identifiers


def run_site(*, site: str, data_dir: str | Path, episodes_path: str | Path,
             shards_path: str | Path, vocab_path: str | Path,
             tasks_path: str | Path = DEFAULT_TASKS,
             data_config: str | Path = DEFAULT_DATA_CONFIG,
             train_config: str | Path = DEFAULT_TRAIN_CONFIG,
             eval_partition: str | None = None, final_evaluation: bool = False,
             comparators: dict | None = None, device: str = "cpu",
             out: str | Path | None = None) -> dict:
    """The task table for one site's CLIF tables, episode artifact and 24 h shards."""
    data_cfg = yaml.safe_load(Path(data_config).read_text())
    suite = load_task_suite(tasks_path, data_cfg=data_cfg)
    roles = dict(suite.partitions)
    if eval_partition:
        roles["evaluation"] = eval_partition
    contract = yaml.safe_load(Path(train_config).read_text())["data_contract"]
    check_evaluation_partition(roles, contract, final_evaluation=final_evaluation)
    vocab = json.loads(Path(vocab_path).read_text())["vocab"]
    report, identifiers = _site_report(
        site=site, data_dir=data_dir, episodes=pl.read_parquet(episodes_path),
        shards=read_shards(shards_path), vocab_size=max(vocab.values()) + 1,
        suite=suite, data_cfg=data_cfg, roles=roles, comparators=comparators,
        device=device)
    if out is not None:
        write_report(report, out, identifiers)
    return report


# -------------------------------------------------------------------- synthetic
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_synthetic_task_site(site_dir: str | Path, *, n_stays: int = 600,
                              seed: int = 7) -> Path:
    """Write a synthetic single-hospital CLIF site for the task suite and return its
    episode artifact path. Arithmetic patients, no real data.

    One latent severity per stay shifts its first-24 h vitals and labs (the features)
    and raises its chance of each post-anchor outcome (the labels), so a count model
    has something to find. By construction: death, tachycardia, hypotension, anemia and
    hyperkalemia are common; hypertension is rare (a small cell); sodium never leaves
    its range and potassium never falls, so three tasks have no positives; a tenth of
    stays are hypotensive before the anchor (prevalent, left out of that task).
    Partitions come from the real grouped split with the repo's ratios and seed.
    """
    from src.data.cohort import build_cohort
    from src.data.splits import (
        assign_grouped_splits,
        content_manifest,
        validate_required_partitions,
    )
    from src.eval.synthetic_bundle import FIXTURE_COHORT

    rng = np.random.default_rng(seed)
    base = Path(site_dir)
    base.mkdir(parents=True, exist_ok=True)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    ids = [f"synth-{i:04d}" for i in range(n_stays)]
    admits = [t0 + timedelta(hours=6 * i) for i in range(n_stays)]
    severity = rng.normal(size=n_stays)

    def happens(slope: float, offset: float) -> np.ndarray:
        return rng.random(n_stays) < _sigmoid(slope * severity + offset)

    died = happens(1.5, -1.2)
    tachycardia, hypotension = happens(1.2, -0.8), happens(1.2, -0.5)
    anemia, hyperkalemia = happens(1.2, -0.7), happens(1.0, -1.0)
    hypertension = rng.random(n_stays) < 0.03
    early_hypotension = rng.random(n_stays) < 0.10

    pl.DataFrame({
        "hospitalization_id": ids,
        "patient_id": [f"synth-p-{i:04d}" for i in range(n_stays)],
        "hospitalization_joined_id": ids,
        "admission_dttm": admits,
        "discharge_dttm": [a + timedelta(days=5) for a in admits],
        "age_at_admission": [40 + (i % 40) for i in range(n_stays)],
        "discharge_category": ["Expired" if d else "Home" for d in died],
        "hospital_id": [SYNTHETIC_TASK_SITE] * n_stays,
    }).write_parquet(base / "clif_hospitalization.parquet")
    pl.DataFrame({
        "hospitalization_id": ids,
        "in_dttm": admits,
        "out_dttm": [a + timedelta(days=4) for a in admits],
        "location_category": ["icu"] * n_stays,
        "hospital_id": [SYNTHETIC_TASK_SITE] * n_stays,
    }).write_parquet(base / "clif_adt.parquet")

    vitals: dict[str, list] = {k: [] for k in ("id", "dttm", "category", "value", "unit")}
    labs: dict[str, list] = {k: [] for k in ("id", "collect", "result", "category",
                                             "value", "unit")}

    def vital(i, hour, category, value, unit="mmHg"):
        for key, item in zip(vitals, (ids[i], admits[i] + timedelta(hours=hour), category,
                                      round(float(value), 1), unit)):
            vitals[key].append(item)

    def lab(i, hour, category, value, unit):
        when = admits[i] + timedelta(hours=hour)
        for key, item in zip(labs, (ids[i], when - timedelta(hours=1), when, category,
                                    round(float(value), 1), unit)):
            labs[key].append(item)

    for i in range(n_stays):
        z = severity[i]
        for hour in range(1, 70, 4):
            after = hour > 24
            heart_rate = min(85 + 10 * z + rng.normal(0, 5), 127)
            mean_pressure = max(80 - 5 * z + rng.normal(0, 3), 66)
            systolic = float(np.clip(120 - 8 * z + rng.normal(0, 6), 92, 175))
            if after and hour == 29:
                if tachycardia[i]:
                    heart_rate = 138
                if hypotension[i]:
                    mean_pressure = 58
                if hypertension[i]:
                    systolic = 188
            if not after and hour == 9 and early_hypotension[i]:
                mean_pressure = 60
            vital(i, hour, "heart_rate", heart_rate, "beats per minute")
            vital(i, hour, "map", mean_pressure)
            vital(i, hour, "sbp", systolic)
            vital(i, hour, "dbp", min(70 + rng.normal(0, 6), 115))
        for hour in range(2, 70, 12):
            after = hour > 24
            hemoglobin = max(10.5 - 1.0 * z + rng.normal(0, 0.5), 7.2)
            potassium = float(np.clip(4.2 + 0.3 * z + rng.normal(0, 0.2), 3.0, 6.0))
            if after and hour == 38:
                if anemia[i]:
                    hemoglobin = 6.5
                if hyperkalemia[i]:
                    potassium = 6.8
            lab(i, hour, "hemoglobin", hemoglobin, "g/dL")
            lab(i, hour, "potassium", potassium, "mmol/L")
            lab(i, hour, "sodium", float(np.clip(139 + rng.normal(0, 3), 125, 155)), "mmol/L")
    pl.DataFrame({
        "hospitalization_id": vitals["id"], "recorded_dttm": vitals["dttm"],
        "vital_category": vitals["category"], "vital_value": vitals["value"],
        "vital_unit": vitals["unit"],
    }).write_parquet(base / "clif_vitals.parquet")
    pl.DataFrame({
        "hospitalization_id": labs["id"], "lab_collect_dttm": labs["collect"],
        "lab_result_dttm": labs["result"], "lab_category": labs["category"],
        "lab_value_numeric": labs["value"], "reference_unit": labs["unit"],
    }).write_parquet(base / "clif_labs.parquet")

    contract = yaml.safe_load(DEFAULT_TRAIN_CONFIG.read_text())["data_contract"]
    episodes = build_cohort(
        pl.read_parquet(base / "clif_hospitalization.parquet"),
        pl.read_parquet(base / "clif_adt.parquet"),
        {"anchor_hours": 24, "prediction_horizon_hours": 48, "minimum_age": 18,
         "icu_location_category": "icu"})
    eligible = assign_grouped_splits(episodes.filter(pl.col("eligible")),
                                     contract["partitions"], seed=contract["split_seed"])
    validate_required_partitions(eligible, contract["required_partitions"])
    episodes = episodes.join(eligible.select("hospitalization_id", "partition"),
                             on="hospitalization_id", how="left")
    episodes = episodes.with_columns(
        pl.lit(FIXTURE_COHORT["contract_version"]).alias("cohort_contract_version"),
        pl.lit(content_manifest(
            eligible, columns=["hospitalization_id", "patient_id", "partition"])["sha256"]
        ).alias("split_sha256"),
        pl.lit(content_manifest(
            episodes, columns=["hospitalization_id", "patient_id", "eligible", "partition"]
        )["sha256"]).alias("episode_sha256"),
        pl.lit("{}").alias("source_provenance_json"),
    )
    validate_episode_artifact(episodes)
    episode_path = base / "episodes.parquet"
    episodes.write_parquet(episode_path)
    return episode_path


def synthetic_data_config(workdir: Path) -> dict:
    """The synthetic-bundle fixture config plus a labs table, bound to `workdir`."""
    from src.eval.synthetic_bundle import (
        FIXTURE_COHORT,
        FIXTURE_DATA_CONFIG,
        FIXTURE_POLICY,
    )

    (workdir / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT, sort_keys=True))
    (workdir / "artifact_policy.yaml").write_text(
        yaml.safe_dump(FIXTURE_POLICY, sort_keys=True))
    cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
    cfg["cohort_contract"] = str((workdir / "cohort.yaml").resolve())
    cfg["artifact_policy"] = str((workdir / "artifact_policy.yaml").resolve())
    cfg["tables"]["labs"] = {
        "file": "clif_labs",
        "availability_col": "lab_result_dttm",   # when the result was knowable
        "availability": "result",
        "concept_col": "lab_category",
        "value_col": "lab_value_numeric",
        "unit_col": "reference_unit",
    }
    cfg["value_binning"]["build_from_site"] = SYNTHETIC_TASK_SITE
    return cfg


def run_synthetic(*, n_stays: int = 600, seed: int = 7,
                  out: str | Path | None = None) -> dict:
    """Baselines + frozen probe end to end on a synthetic site; returns the aggregate
    report (and writes it to `out`, resolved against the caller's directory).

    The site, its real tokenization (`tokenize_site`) and the shards live in a
    throwaway directory; the tokenizer's CWD contract (shards under
    `output/intermediate_phi`) is met by working there, and the caller's directory is
    restored on every exit path. The probe row uses a randomly initialised encoder:
    contract realism, not a model result. The sealed test partition is scored — it is
    synthetic.
    """
    import torch

    from src.data.tokenize import tokenize_site
    from src.eval.synthetic_bundle import FIXTURE_POLICY
    from src.model.encoder import CLIFEncoder

    saved_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as name:
        work = Path(name)
        try:
            os.chdir(work)
            site_dir = work / "site"
            episodes = pl.read_parquet(
                build_synthetic_task_site(site_dir, n_stays=n_stays, seed=seed))
            data_cfg = synthetic_data_config(work)
            shard_dir = Path("output/intermediate_phi/synthetic_tasks")
            tokenize_site(data_cfg, SYNTHETIC_TASK_SITE, site_dir, shard_dir, None,
                          episodes=episodes, artifact_policy=FIXTURE_POLICY, report=False)
            vocab = json.loads((shard_dir / "vocab.json").read_text())["vocab"]
            vocab_size = max(vocab.values()) + 1
            suite = load_task_suite(DEFAULT_TASKS, data_cfg=data_cfg)
            torch.manual_seed(seed)
            trunk = CLIFEncoder(vocab_size, {"trunk": {
                "d_model": 32, "n_heads": 2, "n_layers": 1, "ffn_mult": 2, "dropout": 0.0}})
            report, identifiers = _site_report(
                site=SYNTHETIC_TASK_SITE, data_dir=site_dir, episodes=episodes,
                shards=read_shards(shard_dir / "events.parquet"),
                vocab_size=vocab_size, suite=suite, data_cfg=data_cfg,
                roles=dict(suite.partitions), synthetic=True,
                trunks=[{"row": ROW_SYNTHETIC_PROBE, "trunk": trunk, "positions": True,
                         "label": "Frozen probe on a random-init trunk (machinery check)"}],
                comparators={row_id: {**spec, "checkpoint": None, "shards": None}
                             for row_id, spec in suite.comparators.items()})
        finally:
            os.chdir(saved_cwd)
    if out is not None:
        write_report(report, out, identifiers)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CLIF post-24-hour task table: baselines, frozen probes, comparators")
    parser.add_argument("--synthetic", action="store_true",
                        help="build a synthetic site and run everything on it (data-free)")
    parser.add_argument("--site", help="site name (labels the calibration partitions)")
    parser.add_argument("--data", help="directory of the site's CLIF 2.1 parquet tables")
    parser.add_argument("--episodes", help="canonical episode/split artifact")
    parser.add_argument("--shards", help="the site's 24 h events.parquet")
    parser.add_argument("--vocab", help="the vocab.json those shards were encoded with")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS))
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument("--eval-partition",
                        help="partition to score (default: tasks.yaml partitions.evaluation)")
    parser.add_argument("--final-evaluation", action="store_true",
                        help="allow scoring a sealed partition (internal_test)")
    parser.add_argument("--clifatron-checkpoint", help="CLIFATRON 0.5B checkpoint directory")
    parser.add_argument("--clifatron-shards",
                        help="CLIFATRON-tokenized hour-24 sequences (hosp_id, token)")
    parser.add_argument("--replica-checkpoint",
                        help="decile next-token replica trunk (transformers format)")
    parser.add_argument("--replica-shards", help="the replica arm's hour-24 sequences")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="aggregate JSON (must be under output/final_no_phi)")
    args = parser.parse_args(argv)

    if args.synthetic:
        report = run_synthetic(out=args.out)
    else:
        required = ("site", "data", "episodes", "shards", "vocab")
        missing = [f"--{name}" for name in required if not getattr(args, name)]
        if missing:
            parser.error(f"pass --synthetic, or all of: {', '.join(missing)}")
        suite = load_task_suite(
            args.tasks, data_cfg=yaml.safe_load(Path(args.data_config).read_text()))
        comparators = copy.deepcopy(suite.comparators)
        for row_id, checkpoint, shards in (
                ("clifatron_0p5b_probe", args.clifatron_checkpoint, args.clifatron_shards),
                ("decile_ntp_probe", args.replica_checkpoint, args.replica_shards)):
            spec = comparators.setdefault(row_id, {})
            spec["checkpoint"] = checkpoint or spec.get("checkpoint")
            spec["shards"] = shards or spec.get("shards")
        report = run_site(
            site=args.site, data_dir=args.data, episodes_path=args.episodes,
            shards_path=args.shards, vocab_path=args.vocab, tasks_path=args.tasks,
            data_config=args.data_config, train_config=args.train_config,
            eval_partition=args.eval_partition, final_evaluation=args.final_evaluation,
            comparators=comparators, device=args.device, out=args.out)
    # Aggregate only: the table, never a row or an identifier.
    print(render_table(report))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
