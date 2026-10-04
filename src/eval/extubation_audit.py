"""Go/no-go design audit for the extubation application (plan U13; R32, R6, R7, R25, R33, R40).

Three separate commands (KTD9), each writing one aggregate JSON and one table (CSV) under
`output/final_no_phi` through the audit export gate (`schema.validate_audit_export`) and
the cumulative release ledger (KTD10):

    blind      Outcome-blind. Per site, pooled across sites, and per data partition (and
               the held-out union outside the pretraining partition, KTD6): arm sizes,
               effective sample size, minimal detectable effect, covariate coverage,
               overlap and balance from a device-choice model, the R25 feasibility screen
               per registered trial, and the share of first devices by calendar period
               and by unit. It never imports the labeler and never opens a labels file:
               the command has no labels argument.
    simulate   The planted-effect simulation (U12, R29) per evaluable trial on the frozen
               cohort, with the baseline risk supplied as a registered number.
    unblinded  The one outcome-by-arm comparison allowed before registration: the Casey
               2021 all-comer contrast on an exploratory site (MIMIC), through
               `emulate.authorize_outcome_by_arm`. Any other trial needs the recorded
               protocol hash, and a confirmatory site (Rush) also needs the R28 freeze
               hashes. The gate runs before any label is read.

Stop rules (R32) and the adoption-era trigger (R40) use the thresholds in
configs/extubation_audit.yaml; every report states whether each rule was evaluated and
whether it fired. Before registration every threshold is `proposed`.

Nothing here computes or prints unadjusted outcome rates by arm. Reading the unblinded
result: agreement with the trial's pattern is not proof of cause.

    uv run python -m src.eval.extubation_audit blind --cohort mimic=output/intermediate_phi/extubation_cohort.parquet
"""

from __future__ import annotations

import os

# The causal estimators are numpy (BLAS) nuisance fits on small matrices: a BLAS / OpenMP
# pool of every core on the L40 node only oversubscribes it (and makes the last bits of a
# reduction depend on the core count). Capped to one thread unless the caller set the
# variables; read by the BLAS library when numpy is first imported, so this must precede
# that import (`python -m src.eval.extubation_audit`). The audit's output is unchanged.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv
import hashlib
import io
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from src.data.extubation_cohort import validate_extubation_artifact
from src.eval import attestation as attest
from src.eval import schema
from src.eval.causal import benchmark as bm
from src.eval.causal import diagnostics as dx
from src.eval.causal import emulate as em
from src.eval.causal import estimators as est

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/extubation_benchmarks.yaml"
DEFAULT_COHORT_CONFIG = ROOT / "configs/extubation.yaml"
DEFAULT_AUDIT_CONFIG = ROOT / "configs/extubation_audit.yaml"
DEFAULT_POLICY = ROOT / "configs/artifact_policy.yaml"
OUT_DIR = Path("output/final_no_phi")
DEFAULT_LEDGER = Path("output/intermediate_phi/extubation_audit_ledger.jsonl")
STAGES = ("blind", "simulate", "unblinded")
POOLED = "pooled"
ALL, HELD_OUT = "all", "held_out"
OTHER_ARM = "other_arm"
UNKNOWN = "unknown"
# Risk-factor flags reported for coverage beside the registry's adjustment covariates.
RISK_FACTOR_FLAGS = ("age_over_65", "bmi_over_30", "hypercapnia", "prolonged_ventilation")
TRIAL_PUBLISHED = "trial_published_control_risk"
SUPPLIED = "supplied_registered_number"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_cohort(path: str | Path) -> pl.DataFrame:
    """A site's extubation cohort artifact (patient_level_phi), validated."""
    cohort = pl.read_parquet(path)
    validate_extubation_artifact(cohort)
    return cohort


def _site_pairs(values: Sequence[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        site, sep, path = value.partition("=")
        if not sep or not site or not path:
            raise SystemExit(f"--{what} takes SITE=PATH, got {value!r}")
        if site in out or site == POOLED:
            raise SystemExit(f"--{what}: site {site!r} given twice or reserved")
        out[site] = path
    return out


def _period_source(cohort_config: Mapping[str, Any], site: str) -> Mapping[str, Any] | None:
    return ((cohort_config.get("sites") or {}).get(site) or {}).get("calendar_period_source")


def _parameter(node: Mapping[str, Any], *keys: str) -> tuple[Any, str]:
    for key in keys:
        node = node[key]
    return node["value"], node["status"]


def _label(value: Any) -> str:
    """A category as a cell-id fragment: no path separators, no empty string."""
    text = UNKNOWN if value is None else str(value)
    return text.replace("/", "-").replace("\\", "-").replace("|", "-") or UNKNOWN


# ---------------------------------------------------------------------------
# the cell book: raw counts and their declared relations, suppressed at release
# ---------------------------------------------------------------------------

class Book:
    """Raw cells, partitions and results of one stage, before suppression."""

    def __init__(self) -> None:
        self.cells: dict[str, dict] = {}
        self.partitions: list[dict] = []
        self.results: dict[str, dict] = {}
        self.tables: dict[str, dict] = {}

    def cell(self, key: str, n: int, *, table: str, site: str, dims: Mapping[str, str],
             parents: Sequence[str] = (), share_of: str | None = None) -> str:
        entry = {"table": table, "site": site, "dimensions": dict(dims),
                 "parents": [p for p in parents if p != key], "n": int(n)}
        if share_of is not None:
            entry["share_of"] = share_of
        if key in self.cells:
            if self.cells[key]["n"] != entry["n"]:
                raise ValueError(f"cell {key!r} declared twice with different counts")
            merged = self.cells[key]["parents"] + [p for p in entry["parents"] if p not in self.cells[key]["parents"]]
            self.cells[key]["parents"] = merged
            return key
        self.cells[key] = entry
        return key

    def partition(self, total: str, parts: Sequence[str]) -> None:
        parts = [p for p in parts if p in self.cells]
        if parts and total in self.cells:
            declared = {"total": total, "parts": sorted(parts)}
            if declared not in self.partitions:
                self.partitions.append(declared)

    def result(self, key: str, **fields: Any) -> None:
        self.results[key] = {"status": schema.EVALUABLE, **fields}


def _scopes(cohort: pl.DataFrame, pretraining: str) -> dict[str, np.ndarray]:
    """Row masks per scope: all, held_out (outside the pretraining partition), each partition."""
    partition = np.asarray(cohort["partition"].to_list(), dtype=object)
    scopes = {ALL: np.ones(cohort.height, dtype=bool), HELD_OUT: partition != pretraining}
    for value in sorted({str(p) for p in partition if p is not None}):
        scopes[value] = partition == value
    return scopes


def _cell_id(site: str, scope: str, *parts: str) -> str:
    return "|".join([site, scope, *parts])


def _count_cells(book: Book, registry: bm.Registry, sites: Mapping[str, pl.DataFrame],
                 pretraining: str, *, scopes: Sequence[str] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Eligible, arm and trial-arm cells per site, pooled and scope, with every relation.

    Returns site -> scope -> eligible-row mask, for the stages that compute on the cells.
    """
    masks: dict[str, dict[str, np.ndarray]] = {}
    trial_masks = {}
    for site, cohort in sites.items():
        eligible = cohort["eligible"].fill_null(False).to_numpy()
        site_scopes = _scopes(cohort, pretraining)
        if scopes is not None:
            site_scopes = {k: v for k, v in site_scopes.items() if k in scopes}
        masks[site] = {scope: mask & eligible for scope, mask in site_scopes.items()}
        trial_masks[site] = {t: em.eligibility_mask(cohort, registry, t).to_numpy() for t in registry.trials}

    names = list(sites)
    all_scopes = sorted({s for m in masks.values() for s in m}, key=lambda s: (s not in (ALL, HELD_OUT), s))
    for scope in all_scopes:
        keys = [*names, POOLED] if len(names) > 1 else names
        for site in keys:
            members = names if site == POOLED else [site]
            members = [m for m in members if scope in masks[m]]
            if not members:
                continue
            parents_of = (lambda *parts: [_cell_id(POOLED, scope, *parts)]) if site != POOLED and len(names) > 1 \
                else (lambda *parts: [])

            def scope_parents(*parts: str) -> list[str]:
                out = list(parents_of(*parts))
                if scope != ALL:
                    out.append(_cell_id(site, ALL, *parts))
                return out

            def total(fn) -> int:
                return int(sum(fn(m) for m in members))

            elig = book.cell(_cell_id(site, scope, "eligible"),
                             total(lambda m: masks[m][scope].sum()), table="arm_sizes", site=site,
                             dims={"scope": scope}, parents=scope_parents("eligible"))
            arm_keys = []
            for arm in registry.arms:
                arm_keys.append(book.cell(
                    _cell_id(site, scope, "eligible", f"arm={arm}"),
                    total(lambda m: (masks[m][scope] & (sites[m]["arm"].to_numpy() == arm)).sum()),
                    table="arm_sizes", site=site, dims={"scope": scope, "arm": arm},
                    parents=[elig, *scope_parents("eligible", f"arm={arm}")], share_of=elig))
            book.partition(elig, arm_keys)
            for trial_id, trial in registry.trials.items():
                def in_trial(m: str) -> np.ndarray:
                    return masks[m][scope] & trial_masks[m][trial_id]

                # A trial applied on the cohort's own eligibility is nested in the all-comer cell.
                nested = [elig] if trial.base_column == "eligible" else []
                t_key = book.cell(_cell_id(site, scope, f"trial={trial_id}"), total(lambda m: in_trial(m).sum()),
                                  table="trial_arm_sizes", site=site, dims={"scope": scope, "trial": trial_id},
                                  parents=[*nested, *scope_parents(f"trial={trial_id}")])
                parts = []
                for label in (trial.treated, trial.control, OTHER_ARM):
                    def in_arm(m: str, label: str = label) -> np.ndarray:
                        arm = np.asarray(sites[m][trial.assignment_column].to_list(), dtype=object)
                        if label == OTHER_ARM:
                            return in_trial(m) & (arm != trial.treated) & (arm != trial.control)
                        return in_trial(m) & (arm == label)

                    parents = [t_key, *scope_parents(f"trial={trial_id}", f"arm={label}")]
                    if label != OTHER_ARM and trial.assignment_column == "arm":
                        parents.append(_cell_id(site, scope, "eligible", f"arm={label}"))
                    parts.append(book.cell(
                        _cell_id(site, scope, f"trial={trial_id}", f"arm={label}"),
                        total(lambda m: in_arm(m).sum()), table="trial_arm_sizes", site=site,
                        dims={"scope": scope, "trial": trial_id, "arm": label}, parents=parents))
                book.partition(t_key, parts)

    # Exact decompositions across scopes and sites.
    for site in [*names, POOLED] if len(names) > 1 else names:
        members = names if site == POOLED else [site]
        partitions = sorted({s for m in members for s in masks[m] if s not in (ALL, HELD_OUT)})
        held = [p for p in partitions if p != pretraining]
        suffixes = [("eligible",), *(("eligible", f"arm={a}") for a in registry.arms)]
        for trial_id, trial in registry.trials.items():
            suffixes.append((f"trial={trial_id}",))
            suffixes += [(f"trial={trial_id}", f"arm={label}") for label in (trial.treated, trial.control, OTHER_ARM)]
        for suffix in suffixes:
            book.partition(_cell_id(site, ALL, *suffix), [_cell_id(site, p, *suffix) for p in partitions])
            book.partition(_cell_id(site, HELD_OUT, *suffix), [_cell_id(site, p, *suffix) for p in held])
    if len(names) > 1:
        for key in [k for k in book.cells if k.startswith(f"{POOLED}|")]:
            rest = key[len(POOLED):]
            book.partition(key, [f"{site}{rest}" for site in names])
    return masks


# ---------------------------------------------------------------------------
# blind stage
# ---------------------------------------------------------------------------

def _coverage(frame: pl.DataFrame, columns: Sequence[str], floor: int) -> dict[str, float | str]:
    """Share of rows with each column observed. A share whose observed or missing count is
    a small non-zero cell would give that count back with the denominator, so it is banded."""
    n = frame.height
    out: dict[str, float | str] = {}
    for column in columns:
        if column not in frame.columns:
            out[column] = "not_built"
            continue
        observed = int(frame[column].is_not_null().sum())
        missing = n - observed
        if (0 < observed < floor) or (0 < missing < floor):
            out[column] = f"missing<{floor}" if missing < floor else f"observed<{floor}"
        else:
            out[column] = round(observed / n, 4) if n else None
    return out


def _quantiles(values: Sequence[float]) -> list[float]:
    return [round(float(v), 4) for v in values]


def _device_choice(frame: pl.DataFrame, registry: bm.Registry, config: est.NuisanceConfig) -> dict[str, dict]:
    """Per arm, one-vs-rest cross-fitted device-choice model: overlap and covariate balance.

    Outcome-blind: covariates and the device received only. The support region is the
    registered floor of the feasibility screen, [floor, 1 - floor].
    """
    X = em._covariate_matrix(frame, registry.covariates)
    arm = np.asarray(frame["arm"].to_list(), dtype=object)
    groups = frame["patient_id"].to_numpy()
    floor = registry.screen["min_probability"]
    out: dict[str, dict] = {}
    for label in registry.arms:
        in_arm = arm == label
        entry: dict[str, Any] = {}
        try:
            p = est.cross_fit_probability(X, in_arm.astype(np.float64), groups=groups, config=config)
            overlap = dx.overlap_summary(p, in_arm, support=(floor, 1.0 - floor))
            entry["overlap"] = {
                "quantiles": list(overlap.quantiles),
                "quantiles_in_arm": _quantiles(overlap.quantiles_in_arm),
                "quantiles_out_of_arm": _quantiles(overlap.quantiles_out_of_arm),
                "support": list(overlap.support),
                "share_outside": round(overlap.share_outside, 4),
                "share_outside_in_arm": round(overlap.share_outside_in_arm, 4),
                "share_outside_out_of_arm": round(overlap.share_outside_out_of_arm, 4),
            }
            weights = est.inverse_probability_weights(p, in_arm, clip=config.clip)
            entry["effective_sample_size"] = round(dx.effective_sample_size(weights[in_arm]), 2)
            try:
                balance = dx.covariate_balance(X, in_arm, weights, names=registry.covariates)
                entry["balance"] = {
                    "max_abs_smd_unweighted": _finite(balance.max_abs_unweighted),
                    "max_abs_smd_weighted": _finite(balance.max_abs_weighted),
                    "smd_weighted": {n: _finite(v) for n, v in zip(balance.names, balance.smd_weighted)},
                }
            except ValueError as exc:
                entry["balance"] = {"error": str(exc)}
        except (est.EstimationError, ValueError) as exc:
            entry["error"] = str(exc)
        out[label] = entry
    return out


def _rounded(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, float):
        return round(value, 4)
    return value


def _finite(value: float) -> float | None:
    value = float(value)
    return round(value, 4) if math.isfinite(value) else None


def _feasibility(site_data: em.SiteData, registry: bm.Registry, trial_id: str, config: est.NuisanceConfig,
                 baseline_risk: float | None) -> tuple[bm.FeasibilityResult, dict[str, Any]]:
    """The R25 screen for one trial, plus the statistics the report carries."""
    trial = registry.trials[trial_id]
    try:
        inputs = em.measure_feasibility(site_data, registry, trial_id, config=config)
    except (est.EstimationError, ValueError) as exc:
        inputs = bm.FeasibilityInputs(0, 0, 0.0, 0.0, 1.0, {"clone_censor_weight": False})
        note = str(exc)
    else:
        note = None
    screen = bm.feasibility_screen(registry, trial_id, inputs)
    baseline = trial.effect.risk_control if baseline_risk is None else baseline_risk
    mde = None
    if inputs.ess_treated > 0 and inputs.ess_control > 0:
        mde = round(dx.minimal_detectable_effect(
            inputs.ess_treated, inputs.ess_control, baseline,
            alpha=registry.screen["alpha"], power=registry.screen["power"]), 4)
    checks = {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in screen.checks.items() if k not in ("min_arm_n",)}
    stats = {
        "effective_sample_size": {"treated": round(inputs.ess_treated, 2), "control": round(inputs.ess_control, 2)},
        "minimal_detectable_effect": mde,
        "baseline_risk": round(baseline, 4),
        "baseline_risk_source": TRIAL_PUBLISHED if baseline_risk is None else SUPPLIED,
        "checks": checks,
        "thresholds_status": registry.screen["status"],
    }
    if note:
        stats["checks"]["measurement_error"] = note
    return screen, stats


def _first_device_table(book: Book, site: str, cohort: pl.DataFrame, eligible: np.ndarray, *,
                        column: str, table: str, arms: Sequence[str]) -> None:
    eligible_key = _cell_id(site, ALL, "eligible")
    values = np.asarray([_label(v) for v in cohort[column].to_list()], dtype=object)
    arm = np.asarray(cohort["arm"].to_list(), dtype=object)
    levels = sorted({v for v in values[eligible]})
    level_keys = []
    for level in levels:
        in_level = eligible & (values == level)
        level_key = book.cell(_cell_id(site, ALL, f"{table}={level}"), int(in_level.sum()), table=table,
                              site=site, dims={"scope": ALL, "level": level}, parents=[eligible_key],
                              share_of=eligible_key)
        level_keys.append(level_key)
        parts = [book.cell(_cell_id(site, ALL, f"{table}={level}", f"arm={a}"), int((in_level & (arm == a)).sum()),
                           table=table, site=site, dims={"scope": ALL, "level": level, "arm": a},
                           parents=[level_key, _cell_id(site, ALL, "eligible", f"arm={a}")], share_of=level_key)
                 for a in arms]
        book.partition(level_key, parts)
    book.partition(eligible_key, level_keys)
    for a in arms:
        book.partition(_cell_id(site, ALL, "eligible", f"arm={a}"),
                       [_cell_id(site, ALL, f"{table}={level}", f"arm={a}") for level in levels])


def blind_book(sites: Mapping[str, pl.DataFrame], registry: bm.Registry, *, cohort_config: Mapping[str, Any],
               audit_config: Mapping[str, Any], config: est.NuisanceConfig,
               baseline_risk: float | None = None) -> tuple[Book, dict[str, dict]]:
    """Every outcome-blind count, result and table, raw. Takes cohorts only: no labels."""
    for site, cohort in sites.items():
        bm.check_registry_against_cohort(registry, cohort.columns, registry.arms)
    pretraining = audit_config["pretraining_partition"]
    book = Book()
    masks = _count_cells(book, registry, sites, pretraining)
    floor = schema.min_cell_size()
    screens: dict[str, dict[str, bm.FeasibilityResult]] = {}
    for site, cohort in sites.items():
        screens[site] = {}
        for scope in (ALL, HELD_OUT):
            mask = masks[site].get(scope)
            if mask is None:
                continue
            frame = cohort.filter(pl.Series(mask))
            elig_key = _cell_id(site, scope, "eligible")
            coverage_columns = [*registry.covariates, *(c for c in RISK_FACTOR_FLAGS if c not in registry.covariates)]
            book.result(f"{site}|{scope}|coverage", kind="covariate_coverage", site=site, scope=scope,
                        basis=[elig_key], covariate_coverage=_coverage(frame, coverage_columns, floor))
            if scope == ALL:
                book.result(f"{site}|{scope}|device_choice", kind="device_choice", site=site, scope=scope,
                            basis=[elig_key, *(_cell_id(site, scope, "eligible", f"arm={a}") for a in registry.arms)],
                            **_device_choice_fields(_device_choice(frame, registry, config)))
            data = em.SiteData(cohort=frame)
            for trial_id, trial in registry.trials.items():
                screen, stats = _feasibility(data, registry, trial_id, config, baseline_risk)
                if scope == ALL:
                    screens[site][trial_id] = screen
                basis = [_cell_id(site, scope, f"trial={trial_id}", f"arm={a}") for a in (trial.treated, trial.control)]
                book.result(f"{site}|{scope}|feasibility|{trial_id}", kind="feasibility", site=site, scope=scope,
                            trial_id=trial_id, basis=basis, evaluable=screen.evaluable,
                            reasons=list(screen.reasons), **stats)

        eligible = masks[site][ALL]
        source = _period_source(cohort_config, site)
        period = cohort["calendar_period"]
        if source is None:
            book.tables[f"{site}|period"] = {
                "status": "not_evaluable", "site": site,
                "reason": "no calendar period source is declared for this site; a period is never "
                          "derived from event dates, which may be shifted per patient"}
        elif period.filter(pl.Series(eligible)).is_null().all():
            book.tables[f"{site}|period"] = {
                "status": "not_evaluable", "site": site,
                "reason": "the declared calendar period source is not staged on this node; a period "
                          "is never derived from event dates"}
        else:
            _first_device_table(book, site, cohort, eligible, column="calendar_period",
                                table="first_device_by_period", arms=registry.arms)
            book.tables[f"{site}|period"] = {"status": "evaluable", "site": site}
        _first_device_table(book, site, cohort, eligible, column="unit_at_time_zero",
                            table="first_device_by_unit", arms=registry.arms)
        book.tables[f"{site}|unit"] = {"status": "evaluable", "site": site}
    return book, screens


def _device_choice_fields(per_arm: Mapping[str, dict]) -> dict[str, Any]:
    return {
        "overlap": {a: v.get("overlap", {"error": v.get("error")}) for a, v in per_arm.items()},
        "balance": {a: v.get("balance", {"error": v.get("error")}) for a, v in per_arm.items()},
        "effective_sample_size": {a: v.get("effective_sample_size") for a, v in per_arm.items()},
    }


# ---------------------------------------------------------------------------
# stop rules and the adoption-era trigger
# ---------------------------------------------------------------------------

def precision_rule(screens: Mapping[str, bm.FeasibilityResult], audit_config: Mapping[str, Any],
                   site: str) -> dict[str, Any]:
    threshold, status = _parameter(audit_config, "stop_rules", "precision", "max_share_not_evaluable")
    failing = sum(not s.evaluable for s in screens.values())
    share = failing / len(screens) if screens else None
    return {
        "status": "evaluated" if screens else "not_evaluable", "site": site,
        "fired": bool(share is not None and share > threshold),
        "observed": None if share is None else round(share, 4),
        "threshold": threshold, "threshold_status": status,
        "requirement": "share of registered trials failing the R25 feasibility screen",
    }


def harmful_side_rule(ratio: Mapping[str, float] | None, audit_config: Mapping[str, Any]) -> dict[str, Any]:
    """R32: the unblinded all-comer risk ratio interval lies wholly on the harmful side."""
    threshold, status = _parameter(audit_config, "stop_rules", "harmful_side", "lower_bound_ratio_above")
    rule = {"threshold": threshold, "threshold_status": status,
            "requirement": "lower bound of the primary risk ratio (HFNC over conventional oxygen)"}
    if ratio is None:
        return {**rule, "status": "not_evaluated", "fired": False,
                "reason": "needs the unblinded all-comer comparison (Casey 2021), run only in the unblinded stage"}
    lower = ratio.get("lower")
    if lower is None or not math.isfinite(lower):
        return {**rule, "status": "not_evaluable", "fired": False, "reason": "the estimate is suppressed or undefined"}
    return {**rule, "status": "evaluated", "fired": bool(lower > threshold), "observed": round(float(lower), 4)}


NCO_MIN, NCO_MAX = 5, 15


def negative_control_outcomes(audit_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The registered negative-control outcomes (item 45): 5 to 15 entries, each with a
    definition, a one-line rationale (shared confounding by severity, no causal path from
    the post-extubation device) and a status. Refuses a malformed register."""
    entries = audit_config.get("negative_control_outcomes")
    if not isinstance(entries, Mapping) or not NCO_MIN <= len(entries) <= NCO_MAX:
        raise ValueError(f"negative_control_outcomes must register {NCO_MIN} to {NCO_MAX} outcomes")
    out = {}
    for name, entry in entries.items():
        for key in ("definition", "rationale", "status"):
            if not isinstance(entry, Mapping) or not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"negative_control_outcomes.{name} needs a `{key}`")
        if entry["status"] not in bm.STATUSES:
            raise ValueError(f"negative_control_outcomes.{name}: status must be one of {bm.STATUSES}")
        out[str(name)] = dict(entry)
    return out


def negative_control_failed(interval: Mapping[str, float]) -> bool:
    """A negative control FAILS when its risk-ratio interval excludes no effect (1)."""
    lower, upper = float(interval["lower"]), float(interval["upper"])
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
        raise ValueError("a negative-control interval needs finite lower <= upper")
    return bool(lower > 1.0 or upper < 1.0)


def negative_control_rule(audit_config: Mapping[str, Any],
                          results: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, Any]:
    """R32 (3), item 45: fires when more than `max_failed` registered negative-control
    outcomes have a risk-ratio interval (device contrast) that excludes 1. `results` maps
    a registered outcome name to its interval ({"lower", "upper"}); None before
    registration, when the comparisons may not run (KTD9)."""
    threshold, status = _parameter(audit_config, "stop_rules", "negative_controls", "max_failed")
    registered = negative_control_outcomes(audit_config)
    rule = {"threshold": threshold, "threshold_status": status,
            "requirement": f"registered negative-control outcomes ({len(registered)}) whose risk-ratio "
                           "interval excludes 1"}
    if results is None:
        return {**rule, "status": "not_evaluated", "fired": False,
                "reason": "negative-control comparisons by arm need the registered protocol (KTD9); "
                          "they are pre-registered in the audit thresholds file and run after registration"}
    unknown = sorted(set(results) - set(registered))
    if unknown:
        raise ValueError(f"negative-control results for unregistered outcome(s) {unknown}")
    if not results:
        return {**rule, "status": "not_evaluable", "fired": False, "reason": "no negative-control result"}
    failed = sorted(name for name, interval in results.items() if negative_control_failed(interval))
    reason = (f"{len(results)} of {len(registered)} evaluated; failed: {', '.join(failed)}" if failed
              else f"{len(results)} of {len(registered)} evaluated; none failed")
    return {**rule, "status": "evaluated", "fired": bool(len(failed) > threshold),
            "observed": len(failed), "reason": reason}


def positive_control_check(rate: float | None, audit_config: Mapping[str, Any]) -> dict[str, Any]:
    """Item 45: the pipeline must detect a planted true effect at least `min_detection_rate`
    of the time. A planted-effect statistic; no outcome is read by arm."""
    spec = audit_config.get("positive_control") or {}
    floor, status = _parameter(audit_config, "positive_control", "min_detection_rate")
    planted = spec.get("planted_risk_ratio", {}).get("value")
    if rate is None:
        return {"status": "not_evaluable", "detected": None, "min_detection_rate": floor,
                "planted_risk_ratio": planted, "threshold_status": status}
    return {"status": "evaluated", "detection_rate": round(float(rate), 4), "min_detection_rate": floor,
            "passed": bool(rate >= floor), "planted_risk_ratio": planted, "threshold_status": status}


def adoption_era_trigger(cells: Mapping[str, dict], table: str, site: str, audit_config: Mapping[str, Any],
                         tables: Mapping[str, dict]) -> dict[str, Any]:
    """R40, read from RELEASED cells only, so the flag discloses nothing the table hides."""
    spread_max, status = _parameter(audit_config, "adoption_era", "share_spread_above")
    min_level, _ = _parameter(audit_config, "adoption_era", "min_level_n")
    base = {"site": site, "table": table, "threshold": spread_max, "threshold_status": status}
    kind = "period" if table == "first_device_by_period" else "unit"
    if tables.get(f"{site}|{kind}", {}).get("status") != "evaluable":
        return {**base, "status": "not_evaluable", "triggered": False,
                "reason": tables.get(f"{site}|{kind}", {}).get("reason", "table not built")}
    levels = {key: c for key, c in cells.items()
              if c["table"] == table and c["site"] == site and "arm" not in c["dimensions"]
              and c["dimensions"].get("level") != UNKNOWN and c["status"] == schema.EVALUABLE
              and c["n"] >= min_level}
    shares: dict[str, list[float]] = {}
    for key, c in cells.items():
        if c["table"] == table and c.get("share_of") in levels and "share" in c:
            shares.setdefault(c["dimensions"]["arm"], []).append(c["share"])
    spreads = {arm: max(v) - min(v) for arm, v in shares.items() if len(v) >= 2}
    if not spreads:
        return {**base, "status": "not_evaluable", "triggered": False,
                "reason": "fewer than two released levels to compare"}
    observed = max(spreads.values())
    triggered = observed > spread_max
    out = {**base, "status": "evaluated", "triggered": triggered, "observed": round(observed, 4)}
    if triggered:
        out["contrast_status"] = "triggered_pending"
        out["reason"] = ("adoption-era contrast against Casey 2021 is triggered as a supporting analysis "
                         "(reported as a bound); its estimator is not built in this unit")
    return out


# ---------------------------------------------------------------------------
# release: suppression against the ledger, gate, ledger, publish
# ---------------------------------------------------------------------------

def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(book: Book, *, stage: str, sites: Sequence[str], registry: bm.Registry, release_id: str,
             prior_released: Mapping[str, int], prior_suppressed: set[str], envelope: Mapping[str, Any],
             stop_rules: Mapping[str, dict], audit_config: Mapping[str, Any]) -> dict[str, Any]:
    cells = schema.suppress_audit_cells(book.cells, book.partitions, prior_released=dict(prior_released),
                                        prior_suppressed=set(prior_suppressed))
    results = {key: schema.finalize_audit_result(r, cells) for key, r in book.results.items()}
    triggers = {}
    if stage == "blind":
        for site in sites:
            triggers[f"adoption_era_period|{site}" if len(sites) > 1 else "adoption_era_period"] = \
                adoption_era_trigger(cells, "first_device_by_period", site, audit_config, book.tables)
            triggers[f"adoption_era_unit|{site}" if len(sites) > 1 else "adoption_era_unit"] = \
                adoption_era_trigger(cells, "first_device_by_unit", site, audit_config, book.tables)
    return {
        "export_type": schema.AUDIT_EXPORT_TYPE, "schema_version": schema.AUDIT_SCHEMA_VERSION,
        "stage": stage, "sites": list(sites), "site_roles": {s: registry.site_role(s) for s in sites},
        "release_id": release_id, "generated_by": "extubation_audit",
        "registry_version": registry.version, "registry_status": registry.status,
        "margins_status": registry.margins_status(), "thresholds_status": audit_config["status"],
        **envelope, "cells": cells, "partitions": book.partitions, "results": results,
        "tables": book.tables, "stop_rules": dict(stop_rules), "triggers": triggers,
    }


def table_rows(payload: Mapping[str, Any]) -> list[list[str]]:
    """Human-readable rows of the released report: counts, shares, results, rules."""
    rows = [["section", "id", "status", "n", "share", "detail"]]
    for key, c in payload["cells"].items():
        rows.append(["cell", key, c["status"], str(c.get("n", c.get("n_band", ""))),
                     "" if "share" not in c else f"{c['share']:.4f}", c["table"]])
    for key, t in payload.get("tables", {}).items():
        rows.append(["table", key, t["status"], "", "", t.get("reason", "")])
    for key, r in payload["results"].items():
        detail = {k: v for k, v in r.items() if k not in ("basis", "kind", "status", "site", "scope", "trial_id")}
        rows.append(["result", key, r["status"], "", "", json.dumps(detail, sort_keys=True)])
    for section in ("stop_rules", "triggers"):
        for key, rule in payload.get(section, {}).items():
            flag = rule.get("fired", rule.get("triggered"))
            rows.append([section, key, rule["status"], "", "",
                         ("FIRED" if flag else "not fired") + f"; observed={rule.get('observed')}; "
                         f"threshold={rule.get('threshold')} ({rule.get('threshold_status')})"
                         + (f"; {rule['reason']}" if rule.get("reason") else "")])
    return rows


def publish(book: Book, *, stage: str, sites: Sequence[str], registry: bm.Registry, audit_config: Mapping[str, Any],
            envelope: Mapping[str, Any], stop_rules: Mapping[str, dict], release_id: str | None,
            out_dir: str | Path = OUT_DIR, ledger: str | Path = DEFAULT_LEDGER,
            published_release_ids: set | None = None, approved: bool = False) -> dict[str, Any]:
    """Suppress (seeded by the ledger), validate, record, publish, confirm.

    The sequence and its crash windows are those of `clif_validate.write_export`: the
    artifact is written to a side path, the ledger intent is appended, the rename
    publishes, the confirmation follows. All of it runs under the ledger lock.

    A report is written as `pending_review` unless the disclosure decision is recorded
    (`approved`). It is recorded in the ledger either way: a count that reached a planning
    note or the protocol draft is a release (R6), and over-recording only ever blocks.
    """
    from src.data.cohort import validate_artifact_destination

    policy = yaml.safe_load(DEFAULT_POLICY.read_text())
    tag = "_".join(sites)
    out_json = Path(out_dir) / f"extubation_audit_{stage}_{tag}.json"
    out_csv = out_json.with_suffix(".csv")
    for path in (out_json, out_csv):
        validate_artifact_destination(path, "aggregate_no_phi", policy)
    release_id = release_id or f"extubation-audit-{stage}-{tag}-{uuid.uuid4().hex[:12]}"
    ledger = Path(ledger)
    with attest.ledger_lock(ledger):
        residue = attest.unconfirmed_releases(ledger)
        if residue:
            if published_release_ids is None:
                raise schema.DisclosureError(
                    f"the audit ledger holds unconfirmed releases {sorted(residue)}; pass the "
                    "release ids visible on the export volume so they can be classified")
            unresolved = attest.reconcile_ledger(ledger, published_release_ids)
            if unresolved:
                raise schema.DisclosureError(f"releases {unresolved} are published but unconfirmed")
        prior_released, prior_suppressed = attest.prior_audit_cells(ledger)
        payload = assemble(book, stage=stage, sites=sites, registry=registry, release_id=release_id,
                           prior_released=prior_released, prior_suppressed=prior_suppressed,
                           envelope={**envelope, "disclosure_status":
                                     "reviewed_approved" if approved else schema.DRAFT_DISCLOSURE_STATUS},
                           stop_rules=stop_rules, audit_config=audit_config)
        schema.validate_audit_export(payload)
        attest.check_cross_release_differencing(payload, ledger)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_json.with_suffix(".json.partial")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        try:
            attest.append_to_ledger(payload, ledger)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        try:
            tmp.replace(out_json)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        buffer = io.StringIO()
        csv.writer(buffer).writerows(table_rows(payload))
        out_csv.write_text(buffer.getvalue())
        attest.confirm_publication(payload, ledger)
    return payload


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def _config_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {"benchmark_registry": _sha256(args.registry), "cohort_definition": _sha256(args.cohort_config),
            "audit_thresholds": _sha256(args.audit_config)}


def _nuisance(args: argparse.Namespace) -> est.NuisanceConfig:
    return est.NuisanceConfig(n_folds=args.n_folds, seed=args.seed, max_iter=args.max_iter,
                              max_depth=args.max_depth)


def _load_sites(args: argparse.Namespace) -> dict[str, pl.DataFrame]:
    sites = {}
    for site, path in _site_pairs(args.cohort, "cohort").items():
        cohort = load_cohort(path)
        recorded = set(cohort["site"].drop_nulls().unique().to_list()) if "site" in cohort.columns else set()
        if recorded and recorded != {site}:
            raise SystemExit(f"--cohort {site}: the artifact was built for site(s) {sorted(recorded)}")
        sites[site] = cohort
    return sites


def run_blind(args: argparse.Namespace) -> dict[str, Any]:
    registry = bm.load_benchmark_registry(args.registry)
    cohort_config = yaml.safe_load(Path(args.cohort_config).read_text())
    audit_config = yaml.safe_load(Path(args.audit_config).read_text())
    sites = _load_sites(args)
    book, screens = blind_book(sites, registry, cohort_config=cohort_config, audit_config=audit_config,
                               config=_nuisance(args), baseline_risk=args.baseline_risk)
    stop_rules: dict[str, dict] = {}
    for site in sites:
        key = "precision" if len(sites) == 1 else f"precision|{site}"
        stop_rules[key] = precision_rule(screens[site], audit_config, site)
    stop_rules["harmful_side"] = harmful_side_rule(None, audit_config)
    stop_rules["negative_controls"] = negative_control_rule(audit_config)
    envelope = {"config_hashes": _config_hashes(args),
                "cohort_hashes": {s: c["extubation_sha256"][0] for s, c in sites.items() if c.height},
                "notes": ["outcome-blind: no label or outcome was read",
                          "minimal detectable effect uses the baseline risk stated in each result"]}
    return publish(book, stage="blind", sites=list(sites), registry=registry, audit_config=audit_config,
                   envelope=envelope, stop_rules=stop_rules, release_id=args.release_id, ledger=args.ledger,
                   approved=args.disclosure_approved)


def run_simulate(args: argparse.Namespace) -> dict[str, Any]:
    from src.eval.causal import simulation as sim

    registry = bm.load_benchmark_registry(args.registry)
    audit_config = yaml.safe_load(Path(args.audit_config).read_text())
    sites = _load_sites(args)
    config = _nuisance(args)
    book = Book()
    masks = _count_cells(book, registry, sites, audit_config["pretraining_partition"], scopes=[ALL])
    stop_rules: dict[str, dict] = {}
    for site, cohort in sites.items():
        data = em.SiteData(cohort=cohort)
        screens = {}
        for trial_id, trial in registry.trials.items():
            screen, _ = _feasibility(data, registry, trial_id, config, args.baseline_risk)
            screens[trial_id] = screen
            basis = [_cell_id(site, ALL, f"trial={trial_id}", f"arm={a}") for a in (trial.treated, trial.control)]
            baseline = trial.effect.risk_control if args.baseline_risk is None else args.baseline_risk
            fields = {"kind": "simulation", "site": site, "scope": ALL, "trial_id": trial_id, "basis": basis,
                      "evaluable": screen.evaluable, "reasons": list(screen.reasons)}
            forced = args.trial and trial_id in args.trial
            if not (screen.evaluable or forced):
                book.results[f"{site}|simulation|{trial_id}"] = {
                    **fields, "status": schema.INSUFFICIENT_N,
                    "reason": "not evaluable under the R25 screen; not simulated"}
                continue
            covariates = em.trial_covariates(data, registry, trial_id)
            keep = np.array([a is not None for a in covariates.arm])
            try:
                report = sim.simulate_operating_characteristics(
                    covariates.X[keep], covariates.arm[keep], registry, trial_id, baseline=baseline,
                    n_reps=args.n_reps, n_sim=args.n_sim, config=config, seed=args.seed)
            except (est.EstimationError, ValueError) as exc:
                book.results[f"{site}|simulation|{trial_id}"] = {
                    **fields, "status": schema.RUNTIME_FAILURE, "reason": str(exc)}
                continue
            planted_rr, _ = _parameter(audit_config, "positive_control", "planted_risk_ratio")
            positive = sim.positive_control_detection(
                covariates.X[keep], covariates.arm[keep], registry, trial_id, baseline=baseline,
                planted_rr=float(planted_rr), n_reps=args.n_reps, n_sim=args.n_sim, config=config,
                seed=args.seed)
            checks = {"positive_control": positive_control_check(positive.detection_rate, audit_config),
                      "margin": _rounded(trial.margin.as_dict()),
                      "second_hurdle": registry.second_hurdle.value}
            book.result(
                f"{site}|simulation|{trial_id}", **fields, pass_rates=report.summary(), n_reps=args.n_reps,
                checks=checks,
                n_sim=report.n_sim, baseline_risk=round(baseline, 4),
                baseline_risk_source=TRIAL_PUBLISHED if args.baseline_risk is None else SUPPLIED,
                n_completed={f"{e}|{'confounded' if c else 'measured'}": r.n_completed
                             for (e, c), r in report.scenarios.items()},
                bias={f"{e}|{'confounded' if c else 'measured'}": None if r.bias is None else round(r.bias, 4)
                      for (e, c), r in report.scenarios.items()},
                margins_status=report.margins_status)
        stop_rules["precision" if len(sites) == 1 else f"precision|{site}"] = precision_rule(screens, audit_config, site)
    del masks
    stop_rules["harmful_side"] = harmful_side_rule(None, audit_config)
    stop_rules["negative_controls"] = negative_control_rule(audit_config)
    envelope = {"config_hashes": _config_hashes(args),
                "notes": ["planted effects on the frozen cohort; no outcome was read by arm"]}
    return publish(book, stage="simulate", sites=list(sites), registry=registry, audit_config=audit_config,
                   envelope=envelope, stop_rules=stop_rules, release_id=args.release_id, ledger=args.ledger,
                   approved=args.disclosure_approved)


def run_unblinded(args: argparse.Namespace) -> dict[str, Any]:
    registry = bm.load_benchmark_registry(args.registry)
    audit_config = yaml.safe_load(Path(args.audit_config).read_text())
    cohort_paths = _site_pairs(args.cohort, "cohort")
    trial_id = args.trial[0] if args.trial else registry.audit_trial
    if args.trial and len(args.trial) != 1:
        raise SystemExit("the unblinded stage runs one trial at a time")
    freeze = json.loads(Path(args.freeze_manifest).read_text()) if args.freeze_manifest else None
    local = em.local_freeze_hashes(cohort_config=args.cohort_config, registry_path=args.registry) if freeze else None
    # The gate runs first: nothing (not even the cohort) is read for a refused comparison.
    authorization = em.authorize_outcome_by_arm(
        registry, trial_id=trial_id, sites=list(cohort_paths), protocol_hash=args.protocol_hash,
        freeze_manifest=freeze, local_hashes=local)
    label_paths = _site_pairs(args.labels or [], "labels")
    missing = sorted(set(cohort_paths) - set(label_paths))
    if missing:
        raise SystemExit(f"--labels is required for site(s) {missing}")
    sites = _load_sites(args)
    data = {}
    for site, cohort in sites.items():
        labels = pl.read_parquet(label_paths[site])
        labels = cohort.select("patient_id").join(labels, on="patient_id", how="left", maintain_order="left")
        devices_path = Path(cohort_paths[site])
        devices_path = devices_path.with_name(f"{devices_path.stem}_device_rows.parquet")
        devices = pl.read_parquet(devices_path) if devices_path.exists() else None
        data[site] = em.SiteData(cohort=cohort, labels=labels, device_rows=devices)
    results = em.emulate_trial(data, registry, trial_id, authorization=authorization, config=_nuisance(args),
                               n_boot=args.n_boot, seed=args.seed)

    book = Book()
    _count_cells(book, registry, sites, audit_config["pretraining_partition"], scopes=[ALL])
    trial = registry.trials[trial_id]
    stop_rules: dict[str, dict] = {}
    primary_ratio = None
    for name, result in results.items():
        site = name
        trial_key = _cell_id(site, ALL, f"trial={trial_id}")
        analysed = book.cell(f"{trial_key}|analysed", result.n_analysed, table="trial_analysis", site=site,
                             dims={"scope": ALL, "trial": trial_id, "set": "analysed"}, parents=[trial_key])
        unresolved = book.cell(f"{trial_key}|unresolved", result.n_unresolved, table="trial_analysis", site=site,
                               dims={"scope": ALL, "trial": trial_id, "set": "unresolved"}, parents=[trial_key])
        book.partition(trial_key, [analysed, unresolved])
        basis = [_cell_id(site, ALL, f"trial={trial_id}", f"arm={a}") for a in (trial.treated, trial.control)]
        aggregate = result.to_aggregate(min_cell=schema.min_cell_size())
        fields: dict[str, Any] = {"kind": "effect_estimate", "site": site, "scope": ALL, "trial_id": trial_id,
                                  "basis": basis}
        if aggregate["estimates"] == "suppressed":
            book.results[f"{site}|effect|{trial_id}"] = {
                **fields, "status": schema.SMALL_CELL_SUPPRESSED, "reason": "an analysed arm is below the floor"}
            ratio = None
        else:
            diagnostics = {k: v for k, v in aggregate["diagnostics"].items() if k != "n_followers"}
            estimate = result.trial_estimate()
            verdict = bm.trial_agreement(trial, estimate, second_hurdle=registry.uses_second_hurdle)
            book.result(f"{site}|effect|{trial_id}", **fields, estimates=aggregate["estimates"],
                        diagnostics=diagnostics, primary=aggregate["primary"], event=aggregate["event"],
                        horizon_hours=aggregate["horizon_hours"],
                        discharge_alive_rule=aggregate["discharge_alive_rule"],
                        authorization_basis=aggregate["authorization_basis"],
                        unresolved_handling=aggregate["unresolved_handling"],
                        agreement={**_rounded(verdict.as_dict()),
                                   "benchmark_risk_ratio": round(trial.effect.estimate, 4)},
                        margins_status=aggregate["margins_status"], notes=aggregate["notes"])
            ratio = aggregate["estimates"][aggregate["primary"]]["risk_ratio"]
        if name != POOLED and trial_id == registry.audit_trial:
            key = "harmful_side" if len(results) == 1 else f"harmful_side|{site}"
            stop_rules[key] = {**harmful_side_rule(ratio, audit_config), "site": site}
            primary_ratio = ratio if primary_ratio is None else primary_ratio
    if trial_id != registry.audit_trial:
        stop_rules["harmful_side"] = harmful_side_rule(None, audit_config)
    stop_rules["negative_controls"] = negative_control_rule(audit_config)
    envelope = {
        "config_hashes": _config_hashes(args),
        "authorization": {"basis": authorization.basis, "protocol_hash": authorization.protocol_hash,
                          "freeze_verified": authorization.freeze_verified, "trial_id": trial_id},
        "notes": ["agreement with the trial's pattern is not proof of cause",
                  "margins and thresholds are proposed until the protocol is registered"],
    }
    return publish(book, stage="unblinded", sites=list(sites), registry=registry, audit_config=audit_config,
                   envelope=envelope, stop_rules=stop_rules, release_id=args.release_id, ledger=args.ledger,
                   approved=args.disclosure_approved)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extubation go/no-go design audit (aggregate output only)")
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in STAGES:
        p = sub.add_parser(stage)
        p.add_argument("--cohort", action="append", required=True,
                       help="SITE=PATH of the site's extubation cohort artifact (repeat per site)")
        p.add_argument("--registry", default=str(DEFAULT_REGISTRY))
        p.add_argument("--cohort-config", default=str(DEFAULT_COHORT_CONFIG))
        p.add_argument("--audit-config", default=str(DEFAULT_AUDIT_CONFIG))
        p.add_argument("--ledger", default=str(DEFAULT_LEDGER))
        p.add_argument("--release-id", default=None)
        p.add_argument("--disclosure-approved", action="store_true",
                       help="stamp reviewed_approved (a recorded disclosure decision); default pending_review")
        p.add_argument("--n-folds", type=int, default=5)
        p.add_argument("--max-iter", type=int, default=100)
        p.add_argument("--max-depth", type=int, default=3)
        p.add_argument("--seed", type=int, default=0)
        if stage != "unblinded":
            p.add_argument("--baseline-risk", type=float, default=None,
                           help="registered baseline risk for the detectable effect (default: each "
                                "trial's published control-arm risk)")
        if stage == "simulate":
            p.add_argument("--trial", action="append", help="simulate this trial even if not evaluable")
            p.add_argument("--n-reps", type=int, default=200)
            p.add_argument("--n-sim", type=int, default=None)
        if stage == "unblinded":
            p.add_argument("--labels", action="append", help="SITE=PATH of the site's label artifact")
            p.add_argument("--trial", action="append", help="default: the registry's pre-registration audit trial")
            p.add_argument("--protocol-hash", default=None)
            p.add_argument("--freeze-manifest", default=None, help="JSON file of the R28 freeze hashes")
            p.add_argument("--n-boot", type=int, default=200)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    if args.stage == "blind":
        payload = run_blind(args)
    elif args.stage == "simulate":
        payload = run_simulate(args)
    else:
        payload = run_unblinded(args)
    # Released aggregates only: suppressed cells print their band, never a count.
    for row in table_rows(payload):
        print("\t".join(row))
    return payload


if __name__ == "__main__":
    main()
