"""Per-trial emulation on a site's extubation cohort (plan U12; R8, R14, R25, R28, KTD8, KTD9).

One emulation applies one benchmark trial's eligibility, exposure definition and outcome
window (configs/extubation_benchmarks.yaml) to a site's cohort and labels, runs the U11
primary estimator (one-step clone-censor-weight, doubly robust) and the point-treatment
sensitivity analyses, and returns aggregate results only: estimates with intervals, arm
sizes, diagnostics and counts. It runs per site and pooled across sites (R14).

Gates (KTD9, R28). `authorize_outcome_by_arm` is the single gate every outcome-by-arm
run passes through, here and in the audit runner (U13):
  - Without a recorded protocol hash, the only run allowed is the pre-registration audit
    comparison named in the registry (Casey 2021 all-comers, HFNC vs conventional
    oxygen), exactly as registered, on one exploratory site (MIMIC).
  - A site that is not exploratory (Rush; any site the registry does not list) also needs
    the R28 freeze manifest: a sha256 for every required item, and the items that can be
    recomputed on the node (cohort definition, estimator code, benchmark registry) must
    match what was frozen.
`emulate_trial` refuses to run without an authorization that covers the trial and the
sites; `run_trial_emulation` authorizes and then emulates.

Unresolved follow-up (`discharge_alive_rule`, a registered parameter):
  - `event_free`: a patient discharged alive before the horizon with no event is
    event-free at the horizon. Rows still unresolved (unknown discharge disposition) are
    excluded from the doubly robust step, counted and reported; above the registered
    maximum share the emulation refuses (`UnresolvedFollowUp`).
  - `censor`: such a patient is censored at discharge. The doubly robust step cannot use
    a censored binary outcome, so every row is kept and the primary estimate is the
    clone-weighted Aalen-Johansen cumulative incidence at the horizon, with a patient
    bootstrap interval that holds the clone weights fixed. The count of censored rows is
    reported.
No row is ever dropped silently.

Layering. The polars handling of the cohort, label and device-row frames is confined to
this module's frame layer (`SiteData`, `eligibility_mask`, `trial_covariates`,
`trial_arrays`, `eligibility_counts`); everything after `TrialArrays` is numpy and calls
`estimators`, `diagnostics` and `benchmark`. polars is declared by `clif-validate`, so
the module can be vendored as is.

Reading the output: an emulation that agrees with a trial is consistent with the trial's
pattern; it is not proof of cause. Margins are `proposed` until the protocol is registered.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import polars as pl

from . import benchmark as bm
from . import diagnostics as dx
from . import estimators as est

OUTCOMES = ("trial", "study_primary")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PACKAGE_DIR = Path(__file__).resolve().parent


class EmulationRefused(PermissionError):
    """An outcome-by-arm run is not authorized (no protocol hash, no freeze, wrong scope)."""


class UnresolvedFollowUp(ValueError):
    """Too many rows have unresolved follow-up to estimate on the rest."""


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Authorization:
    trial_id: str
    sites: tuple[str, ...]
    basis: str                       # "registered_protocol" or "pre_registration_audit"
    protocol_hash: str | None
    roles: dict[str, str]
    freeze_verified: bool


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_freeze_hashes(
    *, cohort_config: str | Path, registry_path: str | Path, package_dir: str | Path | None = None
) -> dict[str, str]:
    """The R28 hashes that can be recomputed on the node.

    cohort_definition   sha256 of configs/extubation.yaml
    benchmark_registry  sha256 of configs/extubation_benchmarks.yaml
    estimator_code      sha256 over every `*.py` file of the causal package, sorted by
                        name, each contributing its name and its bytes
    """
    package = Path(package_dir) if package_dir is not None else PACKAGE_DIR
    digest = hashlib.sha256()
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "cohort_definition": _sha256_file(Path(cohort_config)),
        "estimator_code": digest.hexdigest(),
        "benchmark_registry": _sha256_file(Path(registry_path)),
    }


def authorize_outcome_by_arm(
    registry: bm.Registry,
    *,
    trial_id: str,
    sites: Sequence[str],
    protocol_hash: str | None = None,
    freeze_manifest: Mapping[str, str] | None = None,
    local_hashes: Mapping[str, str] | None = None,
    modified: bool = False,
) -> Authorization:
    """Decide whether an outcome-by-arm comparison may run; raise `EmulationRefused` if not.

    `modified` is True when the caller departs from the registered trial definition
    (another outcome, window, discharge rule, eligibility or exposure). Only the
    registered protocol can authorize that; the pre-registration audit exception cannot.
    """
    if trial_id not in registry.trials:
        raise EmulationRefused(f"unknown trial {trial_id!r}")
    site_list = tuple(str(site) for site in sites)
    if not site_list or len(set(site_list)) != len(site_list):
        raise EmulationRefused("name at least one site, each once")
    roles = {site: registry.site_role(site) for site in site_list}

    if protocol_hash is None:
        if trial_id != registry.audit_trial:
            raise EmulationRefused(
                f"trial {trial_id}: no recorded protocol hash; outcome-by-arm comparisons other than "
                f"the pre-registration audit ({registry.audit_trial}) need the registered protocol"
            )
        if modified:
            raise EmulationRefused(
                "the pre-registration audit comparison may only run as registered; "
                "a modified comparison needs the registered protocol"
            )
        if len(site_list) != 1 or roles[site_list[0]] != registry.audit_site_role:
            raise EmulationRefused(
                "no recorded protocol hash: the pre-registration audit comparison runs on one "
                f"{registry.audit_site_role} site only, not {site_list}"
            )
        return Authorization(trial_id, site_list, "pre_registration_audit", None, roles, False)

    if not isinstance(protocol_hash, str) or not _SHA256.match(protocol_hash):
        raise EmulationRefused("the protocol hash must be a lowercase hex sha256 digest")

    freeze_verified = False
    gated = sorted(site for site, role in roles.items() if role != "exploratory")
    if gated:
        if not freeze_manifest:
            raise EmulationRefused(
                f"site(s) {gated} are not exploratory and need the R28 freeze manifest "
                f"({', '.join(registry.freeze_required)})"
            )
        missing = [key for key in registry.freeze_required if not freeze_manifest.get(key)]
        if missing:
            raise EmulationRefused(f"the freeze manifest is missing {', '.join(missing)}")
        malformed = [
            key for key in registry.freeze_required
            if not isinstance(freeze_manifest[key], str) or not _SHA256.match(freeze_manifest[key])
        ]
        if malformed:
            raise EmulationRefused(f"freeze hash for {', '.join(malformed)} is not a sha256 digest")
        if local_hashes is None or any(key not in local_hashes for key in registry.freeze_locally_verified):
            raise EmulationRefused(
                f"the freeze hashes for {', '.join(registry.freeze_locally_verified)} must be "
                "recomputed on the node (local_freeze_hashes) before a confirmatory run"
            )
        stale = [key for key in registry.freeze_locally_verified if local_hashes[key] != freeze_manifest[key]]
        if stale:
            raise EmulationRefused(
                f"the frozen hash of {', '.join(stale)} does not match this node; "
                "something changed after the freeze"
            )
        freeze_verified = True
    return Authorization(trial_id, site_list, "registered_protocol", protocol_hash, roles, freeze_verified)


# ---------------------------------------------------------------------------
# frame layer (polars)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteData:
    """One site's cohort (one row per patient), labels in cohort order, device rows.

    `labels` may be None for outcome-blind work (eligibility counts, feasibility).
    `device_rows` may be None, in which case the first device row is taken to be at
    time zero for every patient with an arm.
    """

    cohort: pl.DataFrame
    labels: pl.DataFrame | None = None
    device_rows: pl.DataFrame | None = None


def _rule_expr(rule: bm.EligibilityRule) -> pl.Expr:
    flags = [pl.col(column).cast(pl.Boolean).fill_null(False) for column in rule.columns]
    count = pl.sum_horizontal([flag.cast(pl.Int32) for flag in flags])
    if rule.kind == "all_true":
        return pl.all_horizontal(flags)
    if rule.kind == "any_true":
        return pl.any_horizontal(flags)
    if rule.kind == "none_true":
        return ~pl.any_horizontal(flags)
    if rule.kind == "count_at_least":
        return count >= rule.value
    if rule.kind == "count_at_most":
        return count <= rule.value
    raise bm.RegistryError(f"unknown rule kind {rule.kind!r}")


def eligibility_mask(cohort: pl.DataFrame, registry: bm.Registry, trial_id: str) -> pl.Series:
    """Trial eligibility per cohort row: the base eligibility column AND every rule.

    A NULL risk factor counts as absent (`missing_risk_factor: absent`).
    """
    trial = registry.trials[trial_id]
    absent = sorted(trial.columns() - set(cohort.columns))
    if absent:
        raise bm.RegistryError(f"trial {trial_id}: cohort has no column(s) {absent}")
    expression = pl.col(trial.base_column).fill_null(False)
    for rule in trial.rules:
        expression = expression & _rule_expr(rule)
    return cohort.select(expression.alias(trial_id)).to_series()


def _covariate_matrix(cohort: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    frame = cohort.select([pl.col(c).cast(pl.Float64) for c in columns])
    return frame.to_numpy().astype(np.float64) if columns else np.empty((cohort.height, 0))


def _device_times(site: SiteData, registry: bm.Registry) -> np.ndarray:
    """Hours from time zero to the first arm-device row inside the grace window."""
    cohort = site.cohort
    if site.device_rows is None:
        return np.where(cohort["arm"].is_null().to_numpy(), np.nan, 0.0)
    first = (
        site.device_rows.filter(
            pl.col("in_grace_window") & pl.col("device_class").is_in(list(registry.arms))
            & ~pl.col("after_invasive_row")
        )
        .group_by("patient_id")
        .agg(pl.col("hours_from_time_zero").min().alias("_device_hours"))
    )
    joined = cohort.select("patient_id").join(first, on="patient_id", how="left", maintain_order="left")
    return joined["_device_hours"].cast(pl.Float64).fill_null(float("nan")).to_numpy()


@dataclass(frozen=True)
class TrialCovariates:
    """Outcome-blind view of one trial's eligible patients."""

    trial_id: str
    mask: np.ndarray              # over cohort rows
    patient_id: np.ndarray
    arm: np.ndarray               # assignment column value, object array
    X: np.ndarray
    covariate_names: tuple[str, ...]
    device_time: np.ndarray
    grace: float


def trial_covariates(site: SiteData, registry: bm.Registry, trial_id: str) -> TrialCovariates:
    """Eligible patients, their assigned arm and adjustment covariates. Reads no outcome."""
    trial = registry.trials[trial_id]
    cohort = site.cohort
    absent = sorted(set(registry.covariates) - set(cohort.columns))
    if absent:
        raise bm.RegistryError(f"cohort has no adjustment column(s) {absent}")
    mask = eligibility_mask(cohort, registry, trial_id).to_numpy()
    if "grace_hours" in cohort.columns and cohort.height:
        graces = cohort["grace_hours"].drop_nulls().unique().to_list()
        if len(graces) != 1:
            raise ValueError("the cohort must be built with a single grace window")
        grace = float(graces[0])
    else:
        grace = 0.0
    eligible = cohort.filter(pl.Series(mask))
    return TrialCovariates(
        trial_id=trial_id, mask=mask, patient_id=eligible["patient_id"].to_numpy(),
        arm=np.asarray(eligible[trial.assignment_column].to_list(), dtype=object),
        X=_covariate_matrix(eligible, registry.covariates), covariate_names=tuple(registry.covariates),
        device_time=_device_times(site, registry)[mask], grace=grace,
    )


@dataclass(frozen=True)
class TrialArrays:
    """Everything the numeric core needs for one trial on one site (eligible rows only)."""

    trial_id: str
    treated: str
    control: str
    event: str
    horizon: float
    discharge_alive_rule: str
    patient_id: np.ndarray
    arm: np.ndarray
    X: np.ndarray
    covariate_names: tuple[str, ...]
    device_time: np.ndarray
    grace: float
    event_time: np.ndarray
    event_type: np.ndarray        # status coding: 0 none, 1 the endpoint, 2 competing
    resolved: np.ndarray

    @property
    def n_eligible(self) -> int:
        return int(self.patient_id.size)

    @property
    def n_unresolved(self) -> int:
        return int((~self.resolved).sum())


def _outcome_spec(registry: bm.Registry, trial: bm.Trial, outcome: str) -> tuple[str, float]:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    if outcome == "trial":
        return trial.event, trial.window_hours
    return registry.study_primary_event, registry.study_primary_window_hours


def _rule(registry: bm.Registry, discharge_alive_rule: str | None) -> str:
    rule = registry.discharge_alive_rule.value if discharge_alive_rule is None else discharge_alive_rule
    if rule not in bm.DISCHARGE_ALIVE_RULES:
        raise ValueError(f"discharge_alive_rule must be one of {bm.DISCHARGE_ALIVE_RULES}, got {rule!r}")
    return rule


def trial_arrays(
    site: SiteData,
    registry: bm.Registry,
    trial_id: str,
    *,
    outcome: str = "trial",
    discharge_alive_rule: str | None = None,
) -> TrialArrays:
    """Outcome arrays for one trial's own event and window (or the study primary outcome)."""
    from src.eval.extubation_labeler import estimator_arrays  # repo-side labeler (polars)

    trial = registry.trials[trial_id]
    event, horizon = _outcome_spec(registry, trial, outcome)
    rule = _rule(registry, discharge_alive_rule)
    if site.labels is None:
        raise ValueError("outcome arrays need the site's labels")
    if site.labels.height != site.cohort.height or not site.labels["patient_id"].equals(site.cohort["patient_id"]):
        raise ValueError("cohort and labels must describe the same rows in the same order")
    covariates = trial_covariates(site, registry, trial_id)
    arrays = estimator_arrays(site.labels, horizon, event=event, discharge_alive_rule=rule, coding="status")
    mask = covariates.mask
    return TrialArrays(
        trial_id=trial_id, treated=trial.treated, control=trial.control, event=event, horizon=float(horizon),
        discharge_alive_rule=rule, patient_id=covariates.patient_id, arm=covariates.arm, X=covariates.X,
        covariate_names=covariates.covariate_names, device_time=covariates.device_time, grace=covariates.grace,
        event_time=arrays.event_time[mask], event_type=arrays.event_type[mask], resolved=arrays.resolved[mask],
    )


def _suppress(value: int, min_cell: int) -> int | str:
    return value if value >= min_cell else f"<{min_cell}"


def eligibility_counts(
    cohort: pl.DataFrame, registry: bm.Registry, *, min_cell: int = 10
) -> dict[str, dict[str, int | str]]:
    """Outcome-blind feasibility counts per trial: eligible, treated, control, other arm.

    Takes the cohort alone (no labels), so no outcome can be read. Cells under `min_cell`
    are printed as "<min_cell". These are counts for planning; any count that leaves the
    node still passes the audit's suppression and differencing checks (U13, R6).
    """
    bm.check_registry_against_cohort(registry, cohort.columns, registry.arms)
    counts = {}
    for trial_id, trial in registry.trials.items():
        mask = eligibility_mask(cohort, registry, trial_id)
        arm = cohort.filter(mask)[trial.assignment_column]
        eligible, treated, control = int(mask.sum()), int((arm == trial.treated).sum()), int((arm == trial.control).sum())
        counts[trial_id] = {
            "eligible": _suppress(eligible, min_cell), "treated": _suppress(treated, min_cell),
            "control": _suppress(control, min_cell), "other_arm": _suppress(eligible - treated - control, min_cell),
        }
    return counts


# ---------------------------------------------------------------------------
# numeric core
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmulationResult:
    """Aggregate result of one trial's emulation on one site (or pooled)."""

    trial_id: str
    sites: tuple[str, ...]
    outcome: str
    event: str
    horizon: float
    discharge_alive_rule: str
    authorization_basis: str
    primary: str
    estimates: dict[str, est.ContrastEstimate]
    influence_log_rr: dict[str, np.ndarray]
    n_eligible: int
    n_analysed: int
    n_unresolved: int
    unresolved_share: float
    unresolved_handling: str
    arm_sizes: dict[str, int]
    diagnostics: dict[str, Any]
    covariate_names: tuple[str, ...]
    covariate_coverage: dict[str, float]
    margins_status: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def trial_estimate(self, name: str | None = None) -> bm.TrialEstimate:
        """The estimate as the agreement rule reads it (log risk ratio and its se)."""
        estimate = self.estimates[name or self.primary]
        return bm.TrialEstimate.from_ratio(estimate.risk_ratio, alpha=estimate.alpha)

    def to_aggregate(self, *, min_cell: int = 10) -> dict[str, Any]:
        """Plain numbers and strings only; counts under `min_cell` suppressed, and with
        them every estimate and diagnostic computed from the small arm."""
        small = any(self.arm_sizes[key] < min_cell for key in ("treated", "control"))

        def interval(value: est.IntervalEstimate) -> dict[str, float]:
            return {"estimate": value.estimate, "se": value.se, "lower": value.lower, "upper": value.upper}

        return {
            "trial_id": self.trial_id, "sites": list(self.sites), "outcome": self.outcome, "event": self.event,
            "horizon_hours": self.horizon, "discharge_alive_rule": self.discharge_alive_rule,
            "authorization_basis": self.authorization_basis, "primary": self.primary,
            "n_eligible": _suppress(self.n_eligible, min_cell), "n_analysed": _suppress(self.n_analysed, min_cell),
            "n_unresolved": _suppress(self.n_unresolved, min_cell) if self.n_unresolved else 0,
            "unresolved_handling": self.unresolved_handling,
            "arm_sizes": {key: _suppress(value, min_cell) for key, value in self.arm_sizes.items()},
            "estimates": "suppressed" if small else {
                name: {"risk_difference": interval(value.risk_difference), "risk_ratio": interval(value.risk_ratio),
                       "estimand": value.estimand, "alpha": value.alpha}
                for name, value in self.estimates.items()
            },
            "diagnostics": "suppressed" if small else _plain(self.diagnostics),
            "covariate_coverage": dict(self.covariate_coverage),
            "margins_status": self.margins_status, "notes": list(self.notes),
        }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _log_rr_influence(risks: est.ArmRisks, treated: str, control: str) -> np.ndarray:
    t, c = risks.arms.index(treated), risks.arms.index(control)
    return risks.influence[:, t] / risks.risk[t] - risks.influence[:, c] / risks.risk[c]


def _clone_diagnostics(
    result: est.CloneCensorResult, X: np.ndarray, names: Sequence[str], registry: bm.Registry
) -> dict[str, Any]:
    design, risks = result.design, result.risks
    at_risk = design.at_risk
    floor = registry.screen["min_probability"]
    out: dict[str, Any] = {"effective_sample_size": {}, "overlap_share_below_min_probability": {},
                           "balance_max_abs_smd": {}, "n_followers": {}}
    for key, k in (("treated", 0), ("control", 1)):
        followers = design.uncensored[:, k] & at_risk
        out["n_followers"][key] = int(followers.sum())
        out["effective_sample_size"][key] = dx.effective_sample_size(risks.weights[followers, k])
        out["overlap_share_below_min_probability"][key] = float(np.mean(risks.probability[at_risk, k] < floor))
        stacked = np.concatenate([X[followers], X[at_risk]])
        is_clone = np.concatenate([np.ones(followers.sum(), bool), np.zeros(at_risk.sum(), bool)])
        weights = np.concatenate([risks.weights[followers, k], np.ones(at_risk.sum())])
        try:
            balance = dx.covariate_balance(stacked, is_clone, weights, names=names)
            out["balance_max_abs_smd"][key] = {
                "unweighted": balance.max_abs_unweighted, "weighted": balance.max_abs_weighted}
        except ValueError as exc:  # a covariate unobserved in one group
            out["balance_max_abs_smd"][key] = {"unweighted": None, "weighted": None, "error": str(exc)}
    return out


def _bootstrap_aalen_johansen(
    arrays: TrialArrays, weights: np.ndarray, groups: np.ndarray, *, n_boot: int, seed: int, alpha: float
) -> tuple[est.ContrastEstimate, dict[str, float]]:
    """Clone-weighted Aalen-Johansen incidence of cause 1 at the horizon, treated vs control.

    The interval is a patient bootstrap with the clone weights held fixed (the censoring
    model is not refitted), so it understates uncertainty from that model; this is stated
    in the result's notes.
    """
    def risks(index: np.ndarray) -> tuple[float, float]:
        values = []
        for k in (0, 1):
            curve = est.cumulative_incidence(
                arrays.event_time[index], arrays.event_type[index], weights=weights[index, k],
                horizon=arrays.horizon, causes=(1, 2),
            )
            values.append(float(curve.incidence[1][-1]))
        return values[0], values[1]

    n = arrays.n_eligible
    risk_t, risk_c = risks(np.arange(n))
    if not (risk_t > 0 and risk_c > 0):
        raise est.EstimationError("the risk ratio is undefined: an arm's estimated incidence is zero")
    patients, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == p) for p in range(patients.size)]
    rng = np.random.default_rng(seed)
    differences, log_ratios = [], []
    for _ in range(n_boot):
        draw = rng.integers(0, patients.size, size=patients.size)
        index = np.concatenate([members[p] for p in draw])
        try:
            boot_t, boot_c = risks(index)
        except ValueError:
            continue
        if boot_t > 0 and boot_c > 0:
            differences.append(boot_t - boot_c)
            log_ratios.append(math.log(boot_t / boot_c))
    if len(log_ratios) < 2:
        raise est.EstimationError("the bootstrap produced too few usable resamples")
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    se_rd, se_log = float(np.std(differences, ddof=1)), float(np.std(log_ratios, ddof=1))
    rd, log_rr = risk_t - risk_c, math.log(risk_t / risk_c)
    contrast = est.ContrastEstimate(
        treated=arrays.treated, control=arrays.control, risk_treated=risk_t, risk_control=risk_c,
        risk_difference=est.IntervalEstimate(rd, se_rd, rd - z * se_rd, rd + z * se_rd),
        risk_ratio=est.IntervalEstimate(
            math.exp(log_rr), se_log, math.exp(log_rr - z * se_log), math.exp(log_rr + z * se_log)),
        alpha=alpha, n=n, method="clone_censor_weight_aalen_johansen", estimand="ATE",
    )
    return contrast, {"n_boot": n_boot, "n_boot_used": len(log_ratios)}


def _estimate(
    arrays: TrialArrays,
    groups: np.ndarray,
    registry: bm.Registry,
    *,
    config: est.NuisanceConfig,
    alpha: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Run the estimators on one set of trial arrays. Returns the pieces of a result."""
    n = arrays.n_eligible
    n_unresolved = arrays.n_unresolved
    share = n_unresolved / n
    arms = (arrays.treated, arrays.control)
    notes: list[str] = []
    if arrays.discharge_alive_rule == "event_free":
        if share > registry.max_unresolved_share.value:
            raise UnresolvedFollowUp(
                f"{n_unresolved} of {n} eligible patients ({share:.1%}) have unresolved follow-up at "
                f"{arrays.horizon:g} h, above the registered maximum of {registry.max_unresolved_share.value:.1%}; "
                "refusing to estimate on the remainder"
            )
        keep = arrays.resolved
        handling = "excluded_and_reported"
    else:
        keep = np.ones(n, dtype=bool)
        handling = "censored_at_discharge"
    X, arm, groups_k = arrays.X[keep], arrays.arm[keep], groups[keep]
    device_time, event_time, event_type = arrays.device_time[keep], arrays.event_time[keep], arrays.event_type[keep]
    device_arm = np.where(_is_null(arm), "__none__", arm).astype(object)

    estimates: dict[str, est.ContrastEstimate] = {}
    influence: dict[str, np.ndarray] = {}
    if handling == "excluded_and_reported":
        result = est.clone_censor_weight(
            X, device_arm, device_time, event_time, event_type, arms=arms, grace=arrays.grace,
            horizon=arrays.horizon, event_of_interest=1, groups=groups_k, config=config,
        )
        primary = "clone_censor_weight"
        estimates[primary] = est.contrast(result.risks, *arms, alpha=alpha)
        influence[primary] = _log_rr_influence(result.risks, *arms)
        y = ((event_type == 1) & (event_time <= arrays.horizon)).astype(np.float64)
        point = est.point_treatment_arm_risks(X, device_arm, y, arms=arms, groups=groups_k, config=config)
        estimates["point_treatment_aipw"] = est.contrast(point, *arms, alpha=alpha)
        influence["point_treatment_aipw"] = _log_rr_influence(point, *arms)
        estimates["overlap_weights"] = est.overlap_weighted_contrast(
            X, device_arm, y, treated=arms[0], control=arms[1], groups=groups_k, config=config, alpha=alpha)
    else:
        # Weights depend only on the device-choice model; the outcome passed here only
        # lets the shared routine run and is not used for the reported estimate.
        happened = ((event_type == 1) & (event_time <= arrays.horizon)).astype(np.float64)
        result = est.clone_censor_weight(
            X, device_arm, device_time, event_time, event_type, arms=arms, grace=arrays.grace,
            horizon=arrays.horizon, outcome=happened, groups=groups_k, config=config,
        )
        primary = "clone_censor_weight_aalen_johansen"
        sub = TrialArrays(**{**arrays.__dict__, "event_time": event_time, "event_type": event_type,
                             "resolved": arrays.resolved[keep]})
        estimates[primary], boot = _bootstrap_aalen_johansen(
            sub, result.risks.weights, groups_k, n_boot=n_boot, seed=seed, alpha=alpha)
        notes.append(
            f"censor rule: weighted Aalen-Johansen at the horizon; bootstrap interval over {boot['n_boot_used']} "
            "patient resamples with clone weights held fixed"
        )
    diagnostics = _clone_diagnostics(result, X, arrays.covariate_names, registry)
    ratio = estimates[primary].risk_ratio
    e_value = est.e_value(ratio.estimate, lower=ratio.lower, upper=ratio.upper)
    diagnostics["e_value"] = {"point": e_value.point, "confidence_limit": e_value.confidence_limit}
    diagnostics["expected_bias_direction"] = registry.trials[arrays.trial_id].expected_bias_direction
    arm_sizes = {
        "treated": int(np.sum(arm == arrays.treated)), "control": int(np.sum(arm == arrays.control)),
    }
    arm_sizes["other_arm"] = int(keep.sum()) - arm_sizes["treated"] - arm_sizes["control"]
    coverage = {
        name: float(np.mean(~np.isnan(arrays.X[:, j]))) for j, name in enumerate(arrays.covariate_names)
    }
    return dict(
        primary=primary, estimates=estimates, influence_log_rr=influence, n_eligible=n,
        n_analysed=int(keep.sum()), n_unresolved=n_unresolved, unresolved_share=share,
        unresolved_handling=handling, arm_sizes=arm_sizes, diagnostics=diagnostics,
        covariate_names=arrays.covariate_names, covariate_coverage=coverage, notes=tuple(notes),
    )


def _is_null(values: np.ndarray) -> np.ndarray:
    return np.array([v is None for v in values], dtype=bool)


def _pooled_arrays(per_site: Mapping[str, TrialArrays]) -> tuple[TrialArrays, np.ndarray]:
    names = list(per_site)
    first = per_site[names[0]]
    graces = {a.grace for a in per_site.values()}
    if len(graces) != 1:
        raise ValueError("sites were built with different grace windows; pooling is refused")
    indicators = [f"site_{name}" for name in names[1:]]
    blocks, groups = [], []
    for name, arrays in per_site.items():
        site_columns = np.zeros((arrays.n_eligible, len(indicators)))
        if name in names[1:]:
            site_columns[:, names.index(name) - 1] = 1.0
        blocks.append(np.hstack([arrays.X, site_columns]))
        groups.append(np.array([f"{name}::{p}" for p in arrays.patient_id], dtype=object))

    def cat(attr: str) -> np.ndarray:
        return np.concatenate([getattr(a, attr) for a in per_site.values()])

    pooled = TrialArrays(
        trial_id=first.trial_id, treated=first.treated, control=first.control, event=first.event,
        horizon=first.horizon, discharge_alive_rule=first.discharge_alive_rule,
        patient_id=np.concatenate(groups), arm=cat("arm"), X=np.vstack(blocks),
        covariate_names=first.covariate_names + tuple(indicators), device_time=cat("device_time"),
        grace=first.grace, event_time=cat("event_time"), event_type=cat("event_type"), resolved=cat("resolved"),
    )
    return pooled, pooled.patient_id


def emulate_trial(
    sites: Mapping[str, SiteData],
    registry: bm.Registry,
    trial_id: str,
    *,
    authorization: Authorization | None,
    outcome: str = "trial",
    discharge_alive_rule: str | None = None,
    pooled: bool = True,
    config: est.NuisanceConfig | None = None,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, EmulationResult]:
    """Emulate one trial on each site and, with more than one site, pooled (R14).

    Needs an `Authorization` from `authorize_outcome_by_arm` covering this trial and
    exactly these sites. Returns site name (and "pooled") -> `EmulationResult`.
    """
    if not sites:
        raise ValueError("no sites to emulate on")
    if authorization is None:
        raise EmulationRefused("no authorization: call authorize_outcome_by_arm first")
    if authorization.trial_id != trial_id or set(authorization.sites) != set(sites):
        raise EmulationRefused("the authorization does not cover this trial and these sites")
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    rule = _rule(registry, discharge_alive_rule)
    if authorization.basis == "pre_registration_audit" and (
        outcome != "trial" or rule != registry.discharge_alive_rule.value
    ):
        raise EmulationRefused("the pre-registration audit comparison may only run as registered")
    config = config or est.NuisanceConfig()
    alpha = registry.alpha
    per_site = {
        name: trial_arrays(data, registry, trial_id, outcome=outcome, discharge_alive_rule=rule)
        for name, data in sites.items()
    }
    for name, arrays in per_site.items():
        if arrays.n_eligible == 0:
            raise ValueError(f"site {name}: no eligible patients for trial {trial_id}")
    runs: dict[str, tuple[tuple[str, ...], TrialArrays, np.ndarray]] = {
        name: ((name,), arrays, arrays.patient_id) for name, arrays in per_site.items()
    }
    if pooled and len(per_site) > 1:
        pooled_arrays, groups = _pooled_arrays(per_site)
        runs["pooled"] = (tuple(per_site), pooled_arrays, groups)
    results = {}
    for name, (site_names, arrays, groups) in runs.items():
        pieces = _estimate(arrays, groups, registry, config=config, alpha=alpha, n_boot=n_boot, seed=seed)
        results[name] = EmulationResult(
            trial_id=trial_id, sites=site_names, outcome=outcome, event=arrays.event, horizon=arrays.horizon,
            discharge_alive_rule=rule, authorization_basis=authorization.basis,
            margins_status=registry.margins_status(), **pieces,
        )
    return results


def run_trial_emulation(
    sites: Mapping[str, SiteData],
    registry: bm.Registry,
    trial_id: str,
    *,
    protocol_hash: str | None = None,
    freeze_manifest: Mapping[str, str] | None = None,
    local_hashes: Mapping[str, str] | None = None,
    outcome: str = "trial",
    discharge_alive_rule: str | None = None,
    **kwargs: Any,
) -> dict[str, EmulationResult]:
    """The emulation command: authorize, then emulate. Refuses before reading outcomes."""
    if not sites:
        raise ValueError("no sites to emulate on")
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
    rule = _rule(registry, discharge_alive_rule)
    modified = outcome != "trial" or rule != registry.discharge_alive_rule.value
    authorization = authorize_outcome_by_arm(
        registry, trial_id=trial_id, sites=list(sites), protocol_hash=protocol_hash,
        freeze_manifest=freeze_manifest, local_hashes=local_hashes, modified=modified,
    )
    return emulate_trial(
        sites, registry, trial_id, authorization=authorization, outcome=outcome,
        discharge_alive_rule=rule, **kwargs,
    )


# ---------------------------------------------------------------------------
# feasibility (outcome-blind)
# ---------------------------------------------------------------------------

def measure_feasibility(
    site: SiteData, registry: bm.Registry, trial_id: str, *, config: est.NuisanceConfig | None = None
) -> bm.FeasibilityInputs:
    """Arm sizes, overlap and effective sample size for one trial, without any outcome.

    The probability of following each compared arm is cross-fitted on the trial's
    eligible patients (everyone treated as reaching the decision, since that depends on
    outcomes). Feed the result to `benchmark.feasibility_screen`.
    """
    config = config or est.NuisanceConfig()
    trial = registry.trials[trial_id]
    data = trial_covariates(site, registry, trial_id)
    n_treated = int(np.sum(data.arm == trial.treated))
    n_control = int(np.sum(data.arm == trial.control))
    ess, below, runnable = {}, [], True
    for key, label in (("treated", trial.treated), ("control", trial.control)):
        follows = (data.arm == label).astype(np.float64)
        try:
            probability = est.cross_fit_probability(data.X, follows, groups=data.patient_id, config=config)
        except (est.EstimationError, ValueError):
            runnable = False
            ess[key] = 0.0
            below.append(1.0)
            continue
        clipped = np.clip(probability, config.clip, 1.0)
        ess[key] = dx.effective_sample_size(1.0 / clipped[follows == 1]) if follows.any() else 0.0
        below.append(float(np.mean(probability < registry.screen["min_probability"])))
    return bm.FeasibilityInputs(
        n_treated=n_treated, n_control=n_control, ess_treated=ess["treated"], ess_control=ess["control"],
        share_below_min_probability=max(below), estimators_runnable={"clone_censor_weight": runnable},
    )
