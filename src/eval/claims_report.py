"""Claims report: finished runs -> the claim 1 and claim 2 tables (plan U8; R31, R35, R39).

One command reads the `threshold_eval` outputs of the claim-bearing runs, aggregates over
seeds with paired bootstrap intervals, applies the registered Benjamini-Hochberg
correction and the decision rules of `configs/claims.yaml`, and marks each claim
`supported`, `not supported` or `incomplete`. The output is aggregate only: no pair,
stay or patient identifier; small cells suppressed (`schema.suppress_cell`); label-state
counts banded with complementary suppression.

RUN RECORD (`run_spec.json`, written by the experiment matrix BEFORE launch, KTD12). One
file in each run's directory, exactly these keys:

    run_id            str   unique, opaque (no patient data)
    tokenization_arm  str   a configs/tokenization_ablation.yaml arm (e.g. clinical_soft)
    objective_arm     str   a configs/objective_arms.yaml arm (e.g. full, next_token_only)
    seed              int   the run's seed (model init, batch order, anchor sampling)
    budget            str   "screening" | "full"
    claim_bearing     bool  listed by arm and seed as claim evidence before launch
    vocab_hash        str   64-hex `segments.artifact_binding(vocab)["vocabulary"]`
    checkpoint        str   the run's checkpoint path (provenance only, never reported)

Only `budget: full` and `claim_bearing: true` runs are read; any other run offered as
claim evidence is refused (not skipped), as is a second run of the same (tokenization
arm, objective arm, seed). Expected layout of a run directory:

    <run_dir>/run_spec.json
    <run_dir>/threshold_eval/scores.parquet   row-level, governed storage only
    <run_dir>/threshold_eval/summary.json     (`src.eval.threshold_eval`)

AGGREGATION. An arm is a (tokenization arm, objective arm); with fewer than
`min_seeds_per_arm` runs it is `incomplete`: its cells carry no metric and it enters no
comparison. A cell is (threshold, horizon); its labels are shared by every run (runs that
disagree on a pair or a label were not evaluated on the same set and are refused). A
cell whose label counts fail `suppress_cell` is suppressed everywhere and excluded from
every comparison. An arm's point metric is the mean over its seeds.

BOOTSTRAP (`claims.yaml bootstrap`). Each replicate resamples, with replacement, (1) the
stays - every (anchor, threshold, horizon) pair of a hospitalization moves together - once
per replicate and shared by every arm, scorer, threshold and horizon, so every comparison
is paired on the same pairs; and (2) each arm's seeds, independently per arm. A replicate
whose difference is undefined (single-class resample) is dropped and counted.

COMPARISONS. A difference is oriented so positive favours the arm the claim says should
win (``a - b`` for AUROC / AUPRC, ``b - a`` for ICI / ECE), pooled as the mean of the
per-cell differences over the comparison's cells; an edge-specific difference subtracts
the same mean over the paired off-edge controls. Its p value is two-sided from the
bootstrap distribution, ``min(1, 2 (min(#d <= 0, #d >= 0) + 1) / (B + 1))``.
Benjamini-Hochberg runs over the evaluable decision comparisons of ONE claim; a
comparison is significant only when BH rejects it at `alpha` AND its FCR-adjusted
percentile interval (level ``1 - R alpha / m``; ``1 - alpha / m`` when R = 0) lies above
zero. Descriptive rows are outside the family and carry intervals at `confidence`.

DECISION RULES: `configs/claims.yaml` (`claim_1`, `claim_2`), implemented by
`decide_claim_1` and `decide_claim_2`.

CLI:
    uv run python -m src.eval.claims_report --runs <run_dir> [<run_dir> ...] \
        --out output/final_no_phi/claims_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml

from src.data.cohort import validate_artifact_destination
from src.data.threshold_grid import load_thresholds
from src.data.tokenization_report import assert_aggregate_only
from src.eval import schema as _schema
from src.eval import threshold_eval as te

ROOT = Path(__file__).parents[2]
DEFAULT_POLICY = ROOT / "configs/artifact_policy.yaml"
DEFAULT_OUT = "output/final_no_phi/claims_report.json"
REPORT_VERSION = 1
RUN_SPEC_FILE = "run_spec.json"
RUN_SPEC_FIELDS = {"run_id": str, "tokenization_arm": str, "objective_arm": str, "seed": int,
                   "budget": str, "claim_bearing": bool, "vocab_hash": str,
                   "checkpoint": str}
BUDGETS = ("screening", "full")
SUPPORTED, NOT_SUPPORTED, INCOMPLETE = "supported", "not supported", "incomplete"
COMPLETE = "complete"
# Keys of the row-level run outputs that must never reach the report.
ROW_IDENTIFIER_KEYS = frozenset({"pair_id", "cluster_id"})
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")

Cell = tuple[str, float]                     # (threshold key, horizon hours)
ArmKey = tuple[str, str]                     # (tokenization arm, objective arm)


class ClaimsError(ValueError):
    """Run outputs cannot be read as claim evidence."""


# ------------------------------------------------------------------------ run records

def validate_run_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Check one `run_spec.json` against the schema (module docstring)."""
    if not isinstance(spec, Mapping):
        raise ClaimsError("run_spec.json must be an object")
    unknown = set(spec) - set(RUN_SPEC_FIELDS)
    if unknown:
        raise ClaimsError(f"run_spec.json has unknown field(s) {sorted(unknown)}")
    for name, kind in RUN_SPEC_FIELDS.items():
        value = spec.get(name)
        if value is None or not isinstance(value, kind) or (
                kind is int and isinstance(value, bool)):
            raise ClaimsError(f"run_spec.json field {name!r} must be a {kind.__name__}")
        if kind is str and not value:
            raise ClaimsError(f"run_spec.json field {name!r} is empty")
    if spec["budget"] not in BUDGETS:
        raise ClaimsError(f"run_spec.json budget must be one of {BUDGETS}")
    if not _HEX64.match(spec["vocab_hash"]):
        raise ClaimsError("run_spec.json vocab_hash must be the 64-hex vocabulary hash")
    return dict(spec)


def load_run_spec(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / RUN_SPEC_FILE
    if not path.is_file():
        raise ClaimsError(f"run directory has no {RUN_SPEC_FILE}")
    return validate_run_spec(json.loads(path.read_text()))


def admit(spec: Mapping[str, Any]) -> None:
    """KTD12: only full-budget runs listed as claim-bearing before launch are evidence."""
    if spec["budget"] != "full":
        raise ClaimsError(
            f"screening-budget run {spec['run_id']!r} is offered as claim evidence: claims "
            "are read only from full-budget runs (KTD12)")
    if not spec["claim_bearing"]:
        raise ClaimsError(
            f"run {spec['run_id']!r} was not listed as claim-bearing before launch (KTD12)")


@dataclass
class Run:
    spec: dict[str, Any]
    summary: dict[str, Any]
    scores: Any                          # polars frame (row-level)


def load_runs(run_dirs: Iterable[str | Path]) -> list[Run]:
    """Read and admit every run: schema, KTD12, one run per (arm, seed), and the run's
    threshold evaluation bound to the vocabulary its spec names."""
    runs, seen = [], {}
    for run_dir in run_dirs:
        spec = load_run_spec(run_dir)
        admit(spec)
        key = (spec["tokenization_arm"], spec["objective_arm"], spec["seed"])
        if key in seen or spec["run_id"] in seen.values():
            raise ClaimsError(f"run {spec['run_id']!r} repeats an arm and seed, or a run_id: "
                              "a run is offered twice")
        seen[key] = spec["run_id"]
        outputs = te.read_run_outputs(run_dir)
        if outputs["summary"]["vocabulary"] != spec["vocab_hash"]:
            raise ClaimsError(f"run {spec['run_id']!r}: its threshold evaluation is bound to "
                              "a different vocabulary than its run_spec vocab_hash")
        if outputs["summary"]["objective_arm"] != spec["objective_arm"]:
            raise ClaimsError(f"run {spec['run_id']!r}: evaluated as another objective arm")
        runs.append(Run(spec, outputs["summary"], outputs["scores"]))
    if not runs:
        raise ClaimsError("no run to report")
    return runs


def run_identifiers(runs: Sequence[Run]) -> set[str]:
    """Every pair and stay identifier in the row-level outputs (refused in the report)."""
    ids: set[str] = set()
    for run in runs:
        for column in ROW_IDENTIFIER_KEYS:
            ids.update(run.scores[column].unique().to_list())
    return ids


# ------------------------------------------------------------------- multiplicity / CIs

def benjamini_hochberg(p_values: Sequence[float], alpha: float
                       ) -> tuple[np.ndarray, np.ndarray, int]:
    """Benjamini-Hochberg (1995) step-up: `(adjusted p values, rejected, R)`.

    Sort the m p values, find the largest rank k with ``p_(k) <= k alpha / m`` and reject
    the k smallest. Adjusted ``p_(i) = min_{j >= i} m p_(j) / j``, capped at 1 (equal to
    `scipy.stats.false_discovery_control(method="bh")`)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0), np.zeros(0, dtype=bool), 0
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1)
    ranked = p[order] * m / ranks
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    passing = np.flatnonzero(p[order] <= ranks * alpha / m)
    n_rejected = int(passing.max()) + 1 if passing.size else 0
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:n_rejected]] = True
    return adjusted, rejected, n_rejected


def fcr_level(alpha: float, n_rejected: int, m: int) -> float:
    """Benjamini & Yekutieli (2005) FCR-adjusted interval level, ``1 - R alpha / m``;
    ``1 - alpha / m`` when nothing is rejected."""
    return 1.0 - max(1, int(n_rejected)) * float(alpha) / int(m)


def bootstrap_p_value(draws: np.ndarray) -> float:
    """Two-sided bootstrap p value of H0: difference = 0 (module docstring)."""
    draws = np.asarray(draws, dtype=float)
    tail = min(int(np.sum(draws <= 0)), int(np.sum(draws >= 0)))
    return float(min(1.0, 2 * (tail + 1) / (len(draws) + 1)))


def _interval(draws: np.ndarray, level: float) -> list[float] | None:
    if not len(draws):
        return None
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]


# ------------------------------------------------------------------------------ panel

@dataclass
class CellData:
    pairs: np.ndarray                    # canonical pair order
    y: np.ndarray
    members: list[np.ndarray]            # per global stay index: this cell's rows
    status: str
    reason: str | None


@dataclass
class ArmData:
    key: ArmKey
    runs: list[Run]
    status: str
    reason: str | None
    # scorer -> cell -> [seeds, pairs] predictions (cells evaluable in every seed only)
    probs: dict[str, dict[Cell, np.ndarray]] = field(default_factory=dict)
    # scorer -> cell -> reason it is not evaluable for this arm
    not_evaluable: dict[str, dict[Cell, str]] = field(default_factory=dict)


@dataclass
class Comparison:
    """One registered or descriptive difference between two (arm, scorer) series."""

    claim: str
    test: str
    a: tuple[ArmKey, str]
    b: tuple[ArmKey, str]
    metric: str
    plus: list[Cell]
    minus: list[Cell] = field(default_factory=list)
    decision: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
    point: float = float("nan")
    draws: np.ndarray = field(default_factory=lambda: np.zeros(0))


class Panel:
    """Every run aligned on the shared evaluation set."""

    def __init__(self, runs: Sequence[Run], claims: Mapping[str, Any]):
        self.claims = claims
        first = runs[0].summary
        for run in runs:
            for field_name in ("evaluation_partition", "horizons_hours", "label_horizon_hours"):
                if run.summary[field_name] != first[field_name]:
                    raise ClaimsError(f"runs disagree on {field_name}: they were not "
                                      "evaluated on the same evaluation set")
        self.evaluation_partition = first["evaluation_partition"]
        self.horizons = [float(h) for h in first["horizons_hours"]]
        self.label_horizon = float(first["label_horizon_hours"])
        self.metric_names = tuple(claims["metric_names"])
        self._align(runs)
        self._edges(runs)

    def _align(self, runs: Sequence[Run]) -> None:
        labels: dict[Cell, dict[str, tuple[int, str]]] = {}
        per_run: list[dict[tuple[str, Cell], dict[str, float]]] = []
        for run in runs:
            series: dict[tuple[str, Cell], dict[str, float]] = {}
            for row in run.scores.iter_rows(named=True):
                cell = (row["threshold"], float(row["horizon_hours"]))
                known = labels.setdefault(cell, {})
                pair = row["pair_id"]
                entry = (int(row["label"]), row["cluster_id"])
                if known.setdefault(pair, entry) != entry:
                    raise ClaimsError(
                        "runs disagree on the label of an evaluation pair: they were not "
                        "evaluated on the same evaluation set")
                series.setdefault((row["scorer"], cell), {})[pair] = float(row["prob"])
            per_run.append(series)
        clusters = sorted({c for cell in labels.values() for _, c in cell.values()})
        cluster_index = {c: i for i, c in enumerate(clusters)}
        self.n_clusters = len(clusters)
        self.cells: dict[Cell, CellData] = {}
        for cell, pairs in labels.items():
            order = np.array(sorted(pairs))
            y = np.array([pairs[p][0] for p in order], dtype=int)
            codes = np.array([cluster_index[pairs[p][1]] for p in order])
            members = [np.zeros(0, dtype=int)] * self.n_clusters
            for code in np.unique(codes):
                members[code] = np.flatnonzero(codes == code)
            status, reason = _schema.suppress_cell(len(y), int(y.sum()))
            self.cells[cell] = CellData(order, y, members, status, reason)

        self.arms: dict[ArmKey, ArmData] = {}
        grouped: dict[ArmKey, list[tuple[Run, dict]]] = {}
        for run, series in zip(runs, per_run):
            key = (run.spec["tokenization_arm"], run.spec["objective_arm"])
            grouped.setdefault(key, []).append((run, series))
        minimum = int(self.claims["min_seeds_per_arm"])
        for key, members in grouped.items():
            members.sort(key=lambda item: item[0].spec["seed"])
            n = len(members)
            status = COMPLETE if n >= minimum else INCOMPLETE
            arm = ArmData(key, [run for run, _ in members], status,
                          None if status == COMPLETE else f"{n} of {minimum} seeds")
            hashes = {run.spec["vocab_hash"] for run, _ in members}
            if len(hashes) > 1:
                raise ClaimsError(f"the seeds of arm {key[0]!r} / {key[1]!r} were trained on "
                                  "different vocabularies")
            reasons = {}
            for run, _ in members:
                for row in run.summary["rows"]:
                    if row["status"] != te.EVALUABLE:
                        reasons[(row["scorer"], (row["threshold"],
                                                 float(row["horizon_hours"])))] = row["reason"]
            scorers = {scorer for _, series in members for scorer, _ in series}
            scorers |= {scorer for scorer, _ in reasons}
            for scorer in scorers:
                arm.probs[scorer] = {}
                arm.not_evaluable[scorer] = {}
                for cell, data in self.cells.items():
                    rows = [series.get((scorer, cell)) for _, series in members]
                    if all(r is not None for r in rows):
                        if any(set(r) != set(data.pairs) for r in rows):
                            raise ClaimsError(
                                "a run scored a different set of evaluation pairs: runs were "
                                "not evaluated on the same evaluation set")
                        arm.probs[scorer][cell] = np.array(
                            [[r[p] for p in data.pairs] for r in rows])
                    else:
                        arm.not_evaluable[scorer][cell] = reasons.get(
                            (scorer, cell), "not scored in every seed")
                for (s, cell), reason in reasons.items():
                    if s == scorer and cell not in self.cells:
                        arm.not_evaluable[scorer][cell] = reason
            self.arms[key] = arm

    def _edges(self, runs: Sequence[Run]) -> None:
        self.edge_table: dict[str, dict[str, dict]] = {}
        self.bin_counts: dict[str, dict[str, int]] = {}
        for run in runs:
            arm = run.spec["tokenization_arm"]
            table = {te.threshold_key(r["kind"], r["concept"], r["value"], r["direction"]): r
                     for r in run.summary["edge_table"]}
            for key, row in table.items():
                if row["kind"] == "control" and row["on_edge"]:
                    raise ClaimsError(
                        f"control {row['concept']} {row['value']:g} is on a bin edge in arm "
                        f"{arm!r}: a control threshold must be off-edge in every arm (KTD3)")
            if self.edge_table.setdefault(arm, table) != table \
                    or self.bin_counts.setdefault(arm, run.summary["bin_counts"]) \
                    != run.summary["bin_counts"]:
                raise ClaimsError(f"the runs of tokenization arm {arm!r} disagree on its "
                                  "edges or bin counts")

    # -------------------------------------------------------------------- availability

    def usable(self, series: tuple[ArmKey, str], cell: Cell) -> bool:
        arm = self.arms.get(series[0])
        return (arm is not None and arm.status == COMPLETE
                and cell in arm.probs.get(series[1], {})
                and self.cells[cell].status == _schema.EVALUABLE)

    def arm_problem(self, key: ArmKey, scorer: str) -> str | None:
        """Why an (arm, scorer) cannot enter a comparison, or None."""
        arm = self.arms.get(key)
        if arm is None:
            return f"arm {key[0]} / {key[1]} has no claim-bearing run"
        if arm.status != COMPLETE:
            return f"arm {key[0]} / {key[1]} is incomplete ({arm.reason})"
        if not arm.probs.get(scorer):
            return f"arm {key[0]} / {key[1]} has no evaluable {scorer} row"
        return None

    # ---------------------------------------------------------------------- bootstrap

    def _metric(self, cache: dict, series: tuple[ArmKey, str], cell: Cell, metric: str,
                rows: np.ndarray | None, seeds: np.ndarray) -> float:
        """An arm's metric on one cell: the mean over `seeds` (a resample may repeat a
        seed). Every primary metric of a (series, cell) is computed in one pass, once per
        distinct seed."""
        key = (series, cell)
        if key not in cache:
            probs = self.arms[series[0]].probs[series[1]][cell]
            y = self.cells[cell].y if rows is None else self.cells[cell].y[rows]
            per_seed = {}
            for seed in np.unique(seeds):
                p = probs[seed] if rows is None else probs[seed][rows]
                per_seed[int(seed)] = te.metric_values(self.metric_names, p, y)
            cache[key] = {name: float(np.mean([per_seed[int(s)][name] for s in seeds]))
                          for name in self.metric_names}
        return cache[key][metric]

    def _value(self, comparison: Comparison, metric_of: Callable) -> float:
        sign = te.METRIC_ORIENTATION[comparison.metric]

        def mean_diff(cells: list[Cell]) -> float:
            diffs = [sign * (metric_of(comparison.a, cell, comparison.metric)
                             - metric_of(comparison.b, cell, comparison.metric))
                     for cell in cells]
            diffs = [d for d in diffs if math.isfinite(d)]
            return float(np.mean(diffs)) if diffs else float("nan")

        value = mean_diff(comparison.plus)
        if comparison.minus:
            value -= mean_diff(comparison.minus)
        return value

    def run_bootstrap(self, comparisons: Sequence[Comparison]) -> None:
        """Point estimates and paired bootstrap draws for every comparison."""
        boot = self.claims["bootstrap"]
        all_seeds = {key: np.arange(len(arm.runs)) for key, arm in self.arms.items()}
        cache: dict = {}
        for comparison in comparisons:
            comparison.point = self._value(
                comparison, lambda s, c, m: self._metric(cache, s, c, m, None,
                                                         all_seeds[s[0]]))
        rng = np.random.default_rng(int(boot["seed"]))
        draws = [[] for _ in comparisons]
        for _ in range(int(boot["n_resamples"])):
            stays = rng.integers(0, self.n_clusters, size=self.n_clusters)
            seeds = {key: rng.integers(0, len(arm.runs), size=len(arm.runs))
                     for key, arm in sorted(self.arms.items())}
            rows: dict[Cell, np.ndarray] = {}
            cache = {}

            def metric_of(series, cell, metric):
                if cell not in rows:
                    members = self.cells[cell].members
                    rows[cell] = np.concatenate([members[s] for s in stays])
                return self._metric(cache, series, cell, metric, rows[cell],
                                    seeds[series[0]])

            for index, comparison in enumerate(comparisons):
                value = self._value(comparison, metric_of)
                if math.isfinite(value):
                    draws[index].append(value)
        for comparison, values in zip(comparisons, draws):
            comparison.draws = np.asarray(values)


def _arm_label(key: ArmKey) -> str:
    return f"{key[0]} / {key[1]}"


def finalize(comparisons: Sequence[Comparison], claims: Mapping[str, Any]) -> list[dict]:
    """Apply the registered correction to the decision family and render every row."""
    alpha = float(claims["multiplicity"]["alpha"])
    confidence = float(claims["bootstrap"]["confidence"])
    decision = [c for c in comparisons if c.decision and len(c.draws)]
    p_values = [bootstrap_p_value(c.draws) for c in decision]
    adjusted, rejected, n_rejected = benjamini_hochberg(p_values, alpha)
    level = fcr_level(alpha, n_rejected, len(decision)) if decision else None
    corrected = {id(c): (p, a, r) for c, p, a, r in zip(decision, p_values, adjusted, rejected)}
    out = []
    for c in comparisons:
        row = {"test": c.test, "metric": c.metric, "arm": _arm_label(c.a[0]),
               "scorer": c.a[1], "comparator": c.b[0][0] if c.b[0][0] != c.a[0][0]
               else c.b[0][1], "comparator_arm": _arm_label(c.b[0]),
               "comparator_scorer": c.b[1], "n_cells": len(c.plus),
               "n_control_cells": len(c.minus), "estimate": _finite(c.point),
               "n_resamples_used": int(len(c.draws)), **c.extra}
        if c.decision and id(c) in corrected:
            p, a, r = corrected[id(c)]
            interval = _interval(c.draws, level)
            row.update({"family": "decision", "p_value": p, "p_adjusted": float(a),
                        "bh_rejected": bool(r), "interval": interval,
                        "interval_level": level,
                        "significant": bool(r) and interval is not None and interval[0] > 0})
        else:
            interval = _interval(c.draws, confidence)
            row.update({"family": "decision" if c.decision else "descriptive",
                        "interval": interval, "interval_level": confidence,
                        "significant": False})
        out.append(row)
    return out


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


# ------------------------------------------------------------------------- claim 1

def _metric_series(claims) -> tuple[str, ...]:
    return tuple(claims["metric_names"])


def claim_1_comparisons(panel: Panel, thresholds: Mapping[str, Any]
                        ) -> tuple[list[Comparison], dict[str, Any]]:
    """The registered claim-1 comparisons (and descriptive rows) plus the facts the
    decision needs: per comparator, the matched concepts and usable thresholds."""
    cfg = panel.claims["claim_1"]
    objective, scorer = cfg["objective_arm"], cfg["scorer"]
    primary = (cfg["primary_arm"], objective)
    rule_arms = list(cfg["comparator_arms"])
    if cfg["attribution_in_rule"]:
        rule_arms.append(cfg["attribution_arm"])
    context: dict[str, Any] = {"primary": primary, "rule_arms": rule_arms, "problems": [],
                               "per_arm": {}}
    problem = panel.arm_problem(primary, scorer)
    if problem:
        context["problems"].append(problem)
    descriptive_arms = sorted({key[0] for key in panel.arms if key[1] == objective}
                              - {primary[0], *rule_arms})
    comparisons: list[Comparison] = []
    horizon = panel.label_horizon
    controls: dict[tuple[str, float], list] = {}
    for control in thresholds["control"]:
        controls.setdefault((control.concept, control.paired_decision), []).append(control)
    for comparator in rule_arms + descriptive_arms:
        key = (comparator, objective)
        decision = comparator in rule_arms
        info = {"matched": [], "unmatched": [], "usable": [], "problem": None}
        context["per_arm"][comparator] = info
        problem = panel.arm_problem(key, scorer)
        if problem and decision:
            context["problems"].append(problem)
        info["problem"] = problem or panel.arm_problem(primary, scorer)
        if info["problem"]:
            continue
        primary_bins = panel.bin_counts[primary[0]]
        other_bins = panel.bin_counts[comparator]
        for concept in sorted(primary_bins):
            (info["matched"] if other_bins.get(concept) == primary_bins[concept]
             else info["unmatched"]).append(concept)
        plus, minus = [], []
        for d in thresholds["decision"]:
            d_key = te.threshold_key("decision", d.concept, d.value, d.direction)
            edge = panel.edge_table[primary[0]].get(d_key)
            if not edge or not edge["on_edge"] or d.concept not in info["matched"]:
                continue
            cell = (d_key, horizon)
            paired = [(te.threshold_key("control", c.concept, c.value, c.direction), horizon)
                      for c in controls.get((d.concept, d.value), ())]
            paired = [c for c in paired
                      if panel.usable((primary, scorer), c) and panel.usable((key, scorer), c)]
            if paired and panel.usable((primary, scorer), cell) \
                    and panel.usable((key, scorer), cell):
                plus.append(cell)
                minus.extend(c for c in paired if c not in minus)
                info["usable"].append(d_key)
        if not plus:
            continue
        for metric in _metric_series(panel.claims):
            extra = {"evaluation": "threshold", "horizon_hours": horizon}
            comparisons.append(Comparison("claim_1", "on_edge_gain", (primary, scorer),
                                          (key, scorer), metric, plus, decision=decision,
                                          extra=extra))
            comparisons.append(Comparison("claim_1", "edge_specific_gain", (primary, scorer),
                                          (key, scorer), metric, plus, minus,
                                          decision=decision, extra=extra))
    return comparisons, context


def decide_claim_1(rows: Sequence[Mapping], context: Mapping[str, Any]
                   ) -> tuple[str, list[str]]:
    """Claim 1 (configs/claims.yaml `claim_1`): supported only if, against every
    comparator in the rule, some metric has a significant on-edge gain AND a significant
    edge-specific gain."""
    if context["problems"]:
        return INCOMPLETE, list(context["problems"])
    reasons, verdicts = [], []
    for comparator in context["rule_arms"]:
        info = context["per_arm"][comparator]
        if not info["matched"]:
            verdicts.append(NOT_SUPPORTED)
            reasons.append(f"{comparator}: no concept is at matched bin count, so a gain "
                           "cannot be shown at matched granularity")
            continue
        if not info["usable"]:
            verdicts.append(INCOMPLETE)
            reasons.append(f"{comparator}: no evaluable on-edge decision threshold at "
                           "matched bin count with an evaluable paired control")
            continue
        mine = [r for r in rows if r["comparator"] == comparator and r["family"] == "decision"]
        gain = {r["metric"] for r in mine if r["test"] == "on_edge_gain" and r["significant"]}
        specific = {r["metric"] for r in mine
                    if r["test"] == "edge_specific_gain" and r["significant"]}
        if not gain:
            verdicts.append(NOT_SUPPORTED)
            reasons.append(f"{comparator} matches the primary arm on-edge within the "
                           "corrected interval on every primary metric")
        elif not gain & specific:
            verdicts.append(NOT_SUPPORTED)
            reasons.append(f"against {comparator} the on-edge gain ({', '.join(sorted(gain))}) "
                           "is not specific to the edges: it does not shrink on the paired "
                           "off-edge controls")
        else:
            verdicts.append(SUPPORTED)
            reasons.append(f"against {comparator}: significant on-edge and edge-specific "
                           f"gain on {', '.join(sorted(gain & specific))}")
        if info["unmatched"]:
            reasons.append(f"{comparator}: concepts not at matched bin count, left out: "
                           f"{', '.join(info['unmatched'])}")
    if INCOMPLETE in verdicts:
        return INCOMPLETE, reasons
    if NOT_SUPPORTED in verdicts:
        return NOT_SUPPORTED, reasons
    return SUPPORTED, reasons


# ------------------------------------------------------------------------- claim 2

def claim_2_comparisons(panel: Panel, thresholds: Mapping[str, Any]
                        ) -> tuple[list[Comparison], dict[str, Any]]:
    claims = panel.claims
    tok = claims["claim_2"]["tokenization_arm"]
    arms = claims["objective_arms"]
    combined, next_token = (tok, arms["combined"]), (tok, arms["next_token"])
    keys = [te.threshold_key(kind, t.concept, t.value, t.direction)
            for kind in te.EVALUATED_KINDS for t in thresholds[kind]]
    evaluations = {"threshold": [(k, panel.label_horizon) for k in keys],
                   "time_to_event": [(k, h) for k in keys for h in panel.horizons]}
    context: dict[str, Any] = {"problems": [], "not_evaluable": [], "rollout": {}}
    for key, scorer in ((combined, "head"), (next_token, "probe")):
        problem = panel.arm_problem(key, scorer)
        if problem:
            context["problems"].append(problem)
    comparisons: list[Comparison] = []

    def add(a, b, decision, test):
        for evaluation, cells in evaluations.items():
            usable = [c for c in cells if panel.usable(a, c) and panel.usable(b, c)]
            if not usable:
                context["not_evaluable"].append({
                    "test": test, "evaluation": evaluation, "comparator": b[0][1],
                    "comparator_scorer": b[1], "status": te.NOT_EVALUABLE,
                    "reason": _not_evaluable_reason(panel, b, cells)})
                if b[1] == "rollout":
                    context["rollout"][evaluation] = False
                continue
            if b[1] == "rollout":
                context["rollout"][evaluation] = True
            for metric in _metric_series(claims):
                comparisons.append(Comparison(
                    "claim_2", test, a, b, metric, usable, decision=decision,
                    extra={"evaluation": evaluation}))

    if not context["problems"]:
        add((combined, "head"), (next_token, "probe"), True, "combined_vs_next_token_probe")
        add((combined, "head"), (next_token, "rollout"), True, "combined_vs_rollout")
    if panel.arm_problem(combined, "probe") is None:
        if panel.arm_problem(next_token, "probe") is None:
            add((combined, "probe"), (next_token, "probe"), False, "probe_vs_probe")
        for ablation in arms["ablations"]:
            if panel.arm_problem((tok, ablation), "probe") is None:
                add((combined, "probe"), ((tok, ablation), "probe"), False, "ablation")
    return comparisons, context


def _not_evaluable_reason(panel: Panel, series: tuple[ArmKey, str],
                          cells: Sequence[Cell]) -> str:
    problem = panel.arm_problem(*series)
    if problem:
        return problem
    arm = panel.arms[series[0]]
    reasons = sorted({arm.not_evaluable.get(series[1], {}).get(c) or "" for c in cells} - {""})
    return "; ".join(reasons) or "every cell is suppressed or not evaluable"


def decide_claim_2(rows: Sequence[Mapping], context: Mapping[str, Any]
                   ) -> tuple[str, list[str]]:
    """Claim 2 (configs/claims.yaml `claim_2`): on both evaluations the combined head
    must beat the next-token probe, and an evaluable rollout, on some primary metric."""
    if context["problems"]:
        return INCOMPLETE, list(context["problems"])
    decision = [r for r in rows if r["family"] == "decision"]
    reasons, verdicts = [], []
    for evaluation in ("threshold", "time_to_event"):
        probe = [r for r in decision if r["test"] == "combined_vs_next_token_probe"
                 and r["evaluation"] == evaluation]
        if not probe:
            verdicts.append(INCOMPLETE)
            reasons.append(f"{evaluation}: no evaluable cell for the probe comparison")
            continue
        wins = sorted(r["metric"] for r in probe if r["significant"])
        if not wins:
            verdicts.append(NOT_SUPPORTED)
            reasons.append(f"{evaluation}: the next-token arm with a linear probe matches the "
                           "combined objective on every primary metric")
        else:
            verdicts.append(SUPPORTED)
            reasons.append(f"{evaluation}: the combined head beats the next-token probe on "
                           f"{', '.join(wins)}")
        if context["rollout"].get(evaluation):
            rollout = [r for r in decision if r["test"] == "combined_vs_rollout"
                       and r["evaluation"] == evaluation]
            if not any(r["significant"] for r in rollout):
                verdicts.append(NOT_SUPPORTED)
                reasons.append(f"{evaluation}: generated rollouts match the combined "
                               "objective on every primary metric (rollouts match)")
        else:
            reasons.append(f"{evaluation}: the rollout comparison is not evaluable, so the "
                           "rollout route of the rejection rule was not tested")
    if INCOMPLETE in verdicts:
        return INCOMPLETE, reasons
    if NOT_SUPPORTED in verdicts:
        return NOT_SUPPORTED, reasons
    return SUPPORTED, reasons


# --------------------------------------------------------------------------- report

def _banded_counts(counts: Mapping[str, int]) -> dict[str, Any]:
    """Label-state counts with every count in (0, floor) banded, plus complementary
    suppression: a lone banded count is a subtraction away from the others, so the next
    smallest count is banded with it."""
    floor = _schema.min_cell_size()
    out: dict[str, Any] = dict(counts)
    small = [k for k, v in counts.items() if 0 < v < floor]
    for key in small:
        out[key] = f"<{floor}"
    if len(small) == 1:
        rest = [k for k, v in counts.items() if k not in small and v > 0]
        if rest:
            victim = min(rest, key=lambda k: (counts[k], k))
            out[victim] = _schema.n_band(counts[victim])
    return out


def _cell_rows(panel: Panel) -> list[dict]:
    rows = []
    for key, arm in sorted(panel.arms.items()):
        for scorer in sorted(set(arm.probs) | set(arm.not_evaluable)):
            for cell in sorted(set(arm.probs[scorer]) | set(arm.not_evaluable[scorer])):
                data = panel.cells.get(cell)
                row = {"tokenization_arm": key[0], "objective_arm": key[1], "scorer": scorer,
                       "threshold": cell[0], "horizon_hours": cell[1]}
                if arm.status != COMPLETE:          # never averaged
                    row.update(status=INCOMPLETE, reason=arm.reason)
                elif cell in arm.not_evaluable[scorer]:
                    row.update(status=te.NOT_EVALUABLE, reason=arm.not_evaluable[scorer][cell])
                elif data.status != _schema.EVALUABLE:
                    row.update(status=data.status, reason=data.reason,
                               n_band=_schema.n_band(len(data.y)))
                else:
                    probs = arm.probs[scorer][cell]
                    row.update(status=_schema.EVALUABLE, n=int(len(data.y)),
                               prevalence=_schema.round_prevalence(float(data.y.mean())),
                               n_seeds=int(probs.shape[0]))
                    per_seed = [te.metric_values(panel.metric_names, p, data.y)
                                for p in probs]
                    for metric in panel.metric_names:
                        row[metric] = _finite(float(np.mean([v[metric] for v in per_seed])))
                rows.append(row)
    return rows


def build_report(runs: Sequence[Run], claims: Mapping[str, Any],
                 thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """The claims report (aggregate only) from admitted runs (`load_runs`)."""
    panel = Panel(runs, claims)
    c1, context_1 = claim_1_comparisons(panel, thresholds)
    c2, context_2 = claim_2_comparisons(panel, thresholds)
    panel.run_bootstrap(c1 + c2)
    rows_1, rows_2 = finalize(c1, claims), finalize(c2, claims)
    status_1, reasons_1 = decide_claim_1(rows_1, context_1)
    status_2, reasons_2 = decide_claim_2(rows_2, context_2)
    first = runs[0].summary
    boot = claims["bootstrap"]
    report = {
        "report": "claims_panel",
        "version": REPORT_VERSION,
        "evaluation_partition": panel.evaluation_partition,
        "horizons_hours": panel.horizons,
        "label_horizon_hours": panel.label_horizon,
        "min_cell_size": _schema.min_cell_size(),
        "config": {
            "primary_metrics": dict(claims["primary_metrics"]),
            "multiplicity": dict(claims["multiplicity"]),
            "bootstrap": {**boot, "resampled": (
                "stays (all pairs of a hospitalization together), once per replicate and "
                "shared by every arm, scorer, threshold and horizon; and each arm's seeds, "
                "independently per arm")},
            "min_seeds_per_arm": claims["min_seeds_per_arm"],
            "claim_1": dict(claims["claim_1"]),
            "claim_2": dict(claims["claim_2"]),
        },
        "arms": [{"tokenization_arm": key[0], "objective_arm": key[1],
                  "n_seeds": len(arm.runs), "status": arm.status, "reason": arm.reason,
                  "run_ids": [run.spec["run_id"] for run in arm.runs]}
                 for key, arm in sorted(panel.arms.items())],
        "edge_table": {arm: sorted(
            ({k: row[k] for k in ("kind", "concept", "value", "direction", "on_edge",
                                  "distance")} for row in table.values()),
            key=lambda r: (r["kind"], r["concept"], r["value"]))
            for arm, table in sorted(panel.edge_table.items())},
        "bin_counts": dict(sorted(panel.bin_counts.items())),
        "label_status": {key: {h: _banded_counts(counts) for h, counts in by_h.items()}
                         for key, by_h in sorted(first["status_counts"].items())},
        "cells": _cell_rows(panel),
        "claims": {
            "claim_1": {"status": status_1, "reasons": reasons_1,
                        "comparisons": [r for r in rows_1 if r["family"] == "decision"],
                        "descriptive": [r for r in rows_1 if r["family"] != "decision"]},
            "claim_2": {"status": status_2, "reasons": reasons_2,
                        "comparisons": [r for r in rows_2 if r["family"] == "decision"],
                        "descriptive": [r for r in rows_2 if r["family"] != "decision"],
                        "not_evaluable": context_2["not_evaluable"]},
        },
        "notes": [
            "Claim 1 is scored by the threshold head, zero-shot; claim 2 compares the "
            "combined head (zero-shot, uncalibrated) with a probe whose temperature is "
            "fitted on the calibration partition.",
            "Label states other than positive and negative are counted, never scored.",
        ],
    }
    return json.loads(json.dumps(report, allow_nan=False))


def write_report(report: Mapping[str, Any], path: str | Path, *, identifiers: Iterable[str],
                 policy: Mapping | None = None) -> Path:
    """Write the aggregate report: destination class checked (`aggregate_no_phi`),
    identifiers refused, strict JSON."""
    path = Path(path)
    if policy is None:
        policy = yaml.safe_load(DEFAULT_POLICY.read_text())
    validate_artifact_destination(path, "aggregate_no_phi", policy)
    ids = set(identifiers)
    assert_aggregate_only(report, ids)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if any(f'"{key}"' in text for key in ROW_IDENTIFIER_KEYS):
        raise ClaimsError("the claims report carries a pair or stay identifier field")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    return path


def render_text(report: Mapping[str, Any]) -> str:
    """Each claim's status, reasons and decision comparisons as text."""
    lines = []
    for name, claim in report["claims"].items():
        lines.append(f"{name}: {claim['status'].upper()}")
        lines.extend(f"  - {reason}" for reason in claim["reasons"])
        for row in claim["comparisons"]:
            interval = row["interval"]
            span = "—" if interval is None else f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"
            estimate = "—" if row["estimate"] is None else f"{row['estimate']:+.3f}"
            lines.append(
                f"    {row['test']} {row.get('evaluation', '')} {row['metric']} vs "
                f"{row['comparator_arm']} ({row['comparator_scorer']}): {estimate} {span} "
                f"p_adj={row.get('p_adjusted', float('nan')):.3g}"
                f"{' *' if row['significant'] else ''}")
    for arm in report["arms"]:
        if arm["status"] != COMPLETE:
            lines.append(f"- arm {arm['tokenization_arm']} / {arm['objective_arm']}: "
                         f"incomplete ({arm['reason']})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Claims report: claim 1 and claim 2 tables from finished runs")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories (run_spec.json + threshold_eval/)")
    ap.add_argument("--claims", default=str(te.CLAIMS_PATH))
    ap.add_argument("--thresholds", default=str(ROOT / "configs/thresholds.yaml"))
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="aggregate JSON (must be under output/final_no_phi)")
    args = ap.parse_args(argv)
    runs = load_runs(args.runs)
    report = build_report(runs, te.load_claims_config(args.claims),
                          load_thresholds(args.thresholds))
    write_report(report, args.out, identifiers=run_identifiers(runs))
    print(render_text(report))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
