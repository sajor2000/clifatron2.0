"""Benchmark registry and agreement rule for the extubation known-answer test (plan U12).

Three things live here:

1. The registry loader and validator for `configs/extubation_benchmarks.yaml`: one entry
   per benchmark trial with its identifier, eligibility over the cohort's risk-factor
   columns, the arms compared, the outcome and window, and the published effect (R8).
2. The feasibility screen (R25): arm sizes, overlap, effective sample size and projected
   precision, judged against registered thresholds without any outcome by arm. A trial
   that fails is "not evaluable" and is left out of every estimator's results.
3. The agreement rule (R10, R27): the gap between an emulation's estimate and the
   benchmark effect on the log risk-ratio scale, pooled per estimator over the evaluable
   trials that found an effect and judged against each trial's own margin; null trials
   are scored separately and count as reproduced only when the estimate's whole interval
   lies inside the equivalence margin; and the paired difference in absolute gap between
   two estimators on identical patients, with a paired bootstrap interval.

Margins (product authority, 2026-10-03, item 42; docs/decisions/2026-10-03-clinical-
decisions.md). Each trial that found an effect gets its own margin by the FDA fixed-margin
approach: M1 is the interval bound of the benchmark effect nearest no effect, and the
margin keeps `preserved_fraction` (0.5) of it, so an emulation may lose at most half of
the trial's conservatively estimated effect. It is stated on the relative scale (log risk
ratio, the scale the rule decides on) and on the absolute scale (risk difference, Wald
bound), and for a mortality-type event (`mortality_type_events`) the relative margin is
capped at `mortality_composite_cap_ratio` (1.1-1.2). A null trial has no effect to
preserve, so it keeps the registered equivalence margin (capped the same way).

Second hurdle (Roehmel & Kieser 2013, doi:10.1002/sim.5563). A trial counts as reproduced
only when, besides the interval rule, the emulation's point estimate is on the same side
of no effect as the benchmark's (`direction_consistent`): a reversed effect cannot pass by
landing inside a wide margin. The pooled verdict also requires every scored effect trial
to clear it.

Reading the result: an estimator that passes agrees with the pattern the trials
established, to within the margin. That is not proof that its estimates are causal.
Margins and thresholds carry a `status`; until the protocol is registered they are
`proposed` and every report says so.

Scale. Every benchmark effect is a risk ratio, treated over control, computed from the
arm counts the trial published (Katz log interval). No published odds ratio is converted
by formula; the effect as published is kept and checked against the counts.

Standard library and numpy only, plus PyYAML inside `load_benchmark_registry`: this
module is vendored into `clif-validate`. Mappings and arrays in, plain dataclasses out.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

FINDINGS = ("effect", "null")
REPRESENTABILITY = ("representable", "proxied", "not_representable")
RULE_KINDS = ("all_true", "any_true", "none_true", "count_at_least", "count_at_most")
EVENTS = ("reintubation_or_death", "reintubation", "death")
# `competing` (item 39 sensitivity): a discharge alive before the horizon is a competing
# event (cumulative incidence), not event-free and not censored.
DISCHARGE_ALIVE_RULES = ("censor", "event_free", "competing")
# Item 41: hospice discharge then death competes with the endpoint (primary) or joins it
# (composite sensitivity).
HOSPICE_RULES = ("competing", "composite")
SECOND_HURDLES = ("direction_consistent", "none")
MARGIN_METHODS = ("fda_fixed_margin",)
# Item 42: the cap on the relative margin of a mortality-type endpoint must lie here.
MORTALITY_CAP_RANGE = (1.1, 1.2)
STATUSES = ("proposed", "registered")
POOLINGS = ("mean_absolute_gap", "inverse_variance_signed")
PUBLISHED_MEASURES = ("absolute_risk_difference", "odds_ratio", "risk_ratio")
ORIENTATIONS = {
    "absolute_risk_difference": ("treated_minus_control", "control_minus_treated"),
    "odds_ratio": ("treated_over_control", "control_over_treated"),
    "risk_ratio": ("treated_over_control", "control_over_treated"),
}
BIAS_DIRECTIONS = {"upward": 1, "downward": -1}
SITE_ROLES = ("exploratory", "confirmatory", "external")
# Tolerances for checking that the published numbers follow from the published counts.
PERCENT_TOLERANCE = 0.15            # percentage points, for a percent published to 0.1
WHOLE_PERCENT_TOLERANCE = 0.6       # for a percent published as a whole number
DIFFERENCE_TOLERANCE = 0.002        # absolute risk difference
RATIO_TOLERANCE = 0.02              # relative, for odds and risk ratios
_PMID = re.compile(r"^\d{1,9}$")


class RegistryError(ValueError):
    """The benchmark registry is incomplete, inconsistent or does not fit the cohort."""


class AgreementError(ValueError):
    """The agreement rule was asked to score something it cannot score."""


# ---------------------------------------------------------------------------
# registry objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Parameter:
    """A registered value and whether it is still `proposed` or already `registered`."""

    value: Any
    status: str


@dataclass(frozen=True)
class BenchmarkEffect:
    """A trial's effect on the comparison scale: risk ratio, treated over control."""

    measure: str
    estimate: float
    lower: float
    upper: float
    log_estimate: float
    se_log: float
    risk_treated: float
    risk_control: float
    derivation: str


@dataclass(frozen=True)
class TrialMargin:
    """One trial's agreement margin on the relative and the absolute scale.

    kind            `fda_fixed_margin` (effect trials) or `equivalence` (null trials)
    ratio           relative margin as a ratio of risk ratios (> 1); `log` is its log
    absolute        the same margin as a risk difference
    m1_log          effect trials: |log| of the benchmark interval bound nearest 1
    m1_absolute     effect trials: |risk difference| bound nearest 0 (Wald)
    capped          the mortality-type cap bound the relative margin
    """

    kind: str
    ratio: float
    log: float
    absolute: float
    m1_log: float | None
    m1_absolute: float | None
    preserved_fraction: float | None
    capped: bool
    status: str
    derivation: str

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ratio": self.ratio, "absolute": self.absolute,
                "m1_log": self.m1_log, "m1_absolute": self.m1_absolute,
                "preserved_fraction": self.preserved_fraction, "capped": self.capped,
                "status": self.status, "derivation": self.derivation}


@dataclass(frozen=True)
class EligibilityRule:
    """One rule over boolean cohort columns; a NULL column value counts as absent."""

    kind: str
    columns: tuple[str, ...]
    value: int | None = None


@dataclass(frozen=True)
class Trial:
    trial_id: str
    label: str
    pmid: str
    identifier_verified: bool
    finding: str                      # "effect" or "null"; the two are scored separately
    approximate: bool
    base_column: str                  # cohort eligibility column the rules are applied on
    rules: tuple[EligibilityRule, ...]
    assignment_column: str
    treated: str
    control: str
    event: str
    window_hours: float
    representability: dict[str, str]  # eligibility / exposure / outcome
    effect: BenchmarkEffect
    published: dict[str, Any]         # the effect as the paper states it, on its own scale
    expected_bias_direction: str
    expected_bias_sign: int           # +1: confounding pushes the risk ratio up; -1: down
    unverified: tuple[str, ...]       # fields that could not be confirmed against the source
    overrides: dict[str, Any] = field(default_factory=dict)
    margin: TrialMargin | None = None

    def columns(self) -> set[str]:
        used = {self.base_column, self.assignment_column}
        for rule in self.rules:
            used.update(rule.columns)
        return used


@dataclass(frozen=True)
class Registry:
    version: str
    status: str
    trials: dict[str, Trial]
    arms: tuple[str, ...]
    assignment_columns: tuple[str, ...]
    eligibility_columns: tuple[str, ...]
    patient_id_column: str
    sites: dict[str, str]
    audit_trial: str
    audit_site_role: str
    freeze_required: tuple[str, ...]
    freeze_locally_verified: tuple[str, ...]
    study_primary_event: str
    study_primary_window_hours: float
    discharge_alive_rule: Parameter
    discharge_alive_rule_sensitivity: tuple[str, ...]
    hospice_rule: Parameter
    hospice_rule_sensitivity: tuple[str, ...]
    max_unresolved_share: Parameter
    missing_risk_factor: Parameter
    covariates: tuple[str, ...]
    forbidden_covariates: tuple[str, ...]
    alpha: float
    preserved_fraction: Parameter
    mortality_cap_ratio: Parameter
    mortality_type_events: tuple[str, ...]
    equivalence_margin_ratio: Parameter
    second_hurdle: Parameter
    pooling: Parameter
    screen: dict[str, Any]

    def columns_used(self) -> set[str]:
        """Every cohort column the registry reads (eligibility, exposure, adjustment)."""
        used = {self.patient_id_column, *self.covariates}
        for trial in self.trials.values():
            used |= trial.columns()
        return used

    def site_role(self, site: str) -> str:
        """A site the registry does not list is treated as confirmatory."""
        return self.sites.get(site, "confirmatory")

    def margins_status(self) -> str:
        """`registered` only when every margin parameter, the second hurdle and the
        pooling rule are registered."""
        parts = (self.preserved_fraction, self.mortality_cap_ratio, self.equivalence_margin_ratio,
                 self.second_hurdle, self.pooling)
        return "registered" if all(p.status == "registered" for p in parts) else "proposed"

    @property
    def uses_second_hurdle(self) -> bool:
        return self.second_hurdle.value == "direction_consistent"


# ---------------------------------------------------------------------------
# effects from published counts
# ---------------------------------------------------------------------------

def _z(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def relative_effect_from_counts(
    events_treated: int, n_treated: int, events_control: int, n_control: int, *, alpha: float = 0.05
) -> BenchmarkEffect:
    """Risk ratio (treated over control) with a Katz log interval from two arms' counts.

    se(log RR) = sqrt(1/a - 1/n1 + 1/c - 1/n0). Both arms need at least one event and at
    least one patient without the event.
    """
    for events, n in ((events_treated, n_treated), (events_control, n_control)):
        if isinstance(events, bool) or isinstance(n, bool) or not isinstance(events, int) or not isinstance(n, int):
            raise ValueError("events and n must be integers")
        if not 0 < events < n:
            raise ValueError("each arm needs 0 < events < n for a risk ratio with an interval")
    risk_t, risk_c = events_treated / n_treated, events_control / n_control
    log_ratio = math.log(risk_t / risk_c)
    se = math.sqrt(1.0 / events_treated - 1.0 / n_treated + 1.0 / events_control - 1.0 / n_control)
    z = _z(alpha)
    return BenchmarkEffect(
        measure="risk_ratio", estimate=math.exp(log_ratio), lower=math.exp(log_ratio - z * se),
        upper=math.exp(log_ratio + z * se), log_estimate=log_ratio, se_log=se,
        risk_treated=risk_t, risk_control=risk_c, derivation="published_arm_counts",
    )


def derive_margin(
    effect: BenchmarkEffect,
    *,
    finding: str,
    event: str,
    n_treated: int,
    n_control: int,
    alpha: float,
    preserved_fraction: float,
    equivalence_ratio: float,
    cap_ratio: float,
    mortality_type_events: Sequence[str],
    status: str,
) -> TrialMargin:
    """A trial's agreement margin (module docstring, "Margins").

    Effect trial: M1 = |log| of the Katz interval bound nearest 1; the relative margin
    is exp((1 - preserved_fraction) * M1). On the absolute scale M1 is the Wald interval
    bound of the risk difference nearest 0 (0 if that interval crosses 0) and the margin
    is (1 - preserved_fraction) * M1. Null trial: the registered equivalence ratio; its
    absolute form is control risk * (ratio - 1). A mortality-type event caps the relative
    margin at `cap_ratio` (the absolute margin is scaled down with it).
    """
    if not 0.0 < preserved_fraction < 1.0:
        raise ValueError("preserved_fraction must lie strictly between 0 and 1")
    mortality = event in mortality_type_events
    if finding == "effect":
        m1_log = min(abs(math.log(effect.lower)), abs(math.log(effect.upper)))
        rd = effect.risk_treated - effect.risk_control
        se_rd = math.sqrt(effect.risk_treated * (1.0 - effect.risk_treated) / n_treated
                          + effect.risk_control * (1.0 - effect.risk_control) / n_control)
        m1_abs = max(0.0, abs(rd) - _z(alpha) * se_rd)
        keep = 1.0 - preserved_fraction
        log_margin, absolute = keep * m1_log, keep * m1_abs
        derivation = (f"FDA fixed margin: M1 = |log {('upper' if effect.upper < 1 else 'lower')} "
                      f"bound| = {m1_log:.4f}; margin keeps {preserved_fraction:g} of M1")
        kind = "fda_fixed_margin"
    else:
        m1_log = m1_abs = None
        log_margin = math.log(equivalence_ratio)
        absolute = effect.risk_control * (equivalence_ratio - 1.0)
        derivation = ("null benchmark: no effect to preserve, so the FDA fixed-margin approach "
                      "does not apply; registered equivalence margin")
        kind = "equivalence"
    capped = False
    if mortality and log_margin > math.log(cap_ratio):
        absolute *= math.log(cap_ratio) / log_margin
        log_margin, capped = math.log(cap_ratio), True
        derivation += f"; capped at {cap_ratio:g} (mortality-type endpoint)"
    if not log_margin > 0.0:
        raise RegistryError("a derived margin must be above no effect")
    return TrialMargin(kind=kind, ratio=math.exp(log_margin), log=log_margin, absolute=absolute,
                       m1_log=m1_log, m1_absolute=m1_abs,
                       preserved_fraction=preserved_fraction if finding == "effect" else None,
                       capped=capped, status=status, derivation=derivation)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _need(node: Any, key: str, where: str) -> Any:
    if not isinstance(node, Mapping) or key not in node or node[key] is None:
        raise RegistryError(f"{where}: missing `{key}`")
    return node[key]


def _parameter(node: Any, key: str, where: str) -> Parameter:
    raw = _need(node, key, where)
    value, status = _need(raw, "value", f"{where}.{key}"), _need(raw, "status", f"{where}.{key}")
    if status not in STATUSES:
        raise RegistryError(f"{where}.{key}: status must be one of {STATUSES}, got {status!r}")
    return Parameter(value, status)


def _number(value: Any, where: str, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RegistryError(f"{where}: expected a finite number, got {value!r}")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise RegistryError(f"{where}: {value!r} is below the allowed range")
    return float(value)


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str) or not all(isinstance(v, str) for v in value):
        raise RegistryError(f"{where}: expected a list of names")
    return tuple(value)


def _rules(raw: Any, sets: Mapping[str, tuple[str, ...]], where: str) -> tuple[EligibilityRule, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise RegistryError(f"{where}: `rules` must be a list")
    rules = []
    for index, rule in enumerate(raw):
        here = f"{where}.rules[{index}]"
        kind = _need(rule, "kind", here)
        if kind not in RULE_KINDS:
            raise RegistryError(f"{here}: unknown rule kind {kind!r}; expected one of {RULE_KINDS}")
        if ("set" in rule) == ("columns" in rule):
            raise RegistryError(f"{here}: give exactly one of `set` or `columns`")
        if "set" in rule:
            if rule["set"] not in sets:
                raise RegistryError(f"{here}: unknown risk-factor set {rule['set']!r}")
            columns = sets[rule["set"]]
        else:
            columns = _strings(rule["columns"], here)
        if not columns:
            raise RegistryError(f"{here}: no columns")
        value = None
        if kind.startswith("count_"):
            value = _need(rule, "value", here)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= len(columns):
                raise RegistryError(f"{here}: `value` must be an integer between 0 and {len(columns)}")
        rules.append(EligibilityRule(kind, tuple(columns), value))
    return tuple(rules)


def _arm_counts(raw: Any, where: str) -> tuple[int, int, list[str]]:
    events, n = _need(raw, "events", where), _need(raw, "n", where)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (events, n)) or not 0 < events < n:
        raise RegistryError(f"{where}: needs integer counts with 0 < events < n")
    unverified: list[str] = []
    if raw.get("unverified"):
        _need(raw, "unverified_reason", where)
        unverified.append(where.split(": ", 1)[-1])
    elif "percent" in raw and abs(100.0 * events / n - float(raw["percent"])) > (
        WHOLE_PERCENT_TOLERANCE if isinstance(raw["percent"], int) else PERCENT_TOLERANCE
    ):
        raise RegistryError(
            f"{where}: published percent {raw['percent']} does not match events / n "
            f"({100.0 * events / n:.2f}); correct it or mark the arm `unverified` with a reason"
        )
    return events, n, unverified


def _check_published(published: Mapping[str, Any], counts: tuple[int, int, int, int], where: str) -> None:
    """The effect as published must follow from the published counts."""
    measure = _need(published, "measure", where)
    if measure not in PUBLISHED_MEASURES:
        raise RegistryError(f"{where}: measure must be one of {PUBLISHED_MEASURES}")
    orientation = _need(published, "orientation", where)
    if orientation not in ORIENTATIONS[measure]:
        raise RegistryError(f"{where}: orientation must be one of {ORIENTATIONS[measure]}")
    estimate = _number(_need(published, "estimate", where), f"{where}.estimate")
    if published.get("unverified"):
        return
    a, n1, c, n0 = counts
    risk_t, risk_c = a / n1, c / n0
    if measure == "absolute_risk_difference":
        expected = risk_t - risk_c
        expected = expected if orientation == "treated_minus_control" else -expected
        matches = abs(expected - estimate) <= DIFFERENCE_TOLERANCE
    else:
        if measure == "odds_ratio":
            expected = (a / (n1 - a)) / (c / (n0 - c))
        else:
            expected = risk_t / risk_c
        expected = expected if orientation == "treated_over_control" else 1.0 / expected
        matches = estimate > 0 and abs(math.log(expected / estimate)) <= RATIO_TOLERANCE
    if not matches:
        raise RegistryError(
            f"{where}: the published effect ({estimate}) does not follow from the published "
            f"arm counts ({expected:.4f}); correct it or mark it `unverified`"
        )


def _trial(
    trial_id: str,
    raw: Any,
    *,
    arms: tuple[str, ...],
    assignment_columns: tuple[str, ...],
    eligibility_columns: tuple[str, ...],
    sets: Mapping[str, tuple[str, ...]],
    alpha: float,
    margin_rule: Mapping[str, Any],
) -> Trial:
    where = f"trial {trial_id}"
    if not isinstance(raw, Mapping):
        raise RegistryError(f"{where}: must be a mapping")
    identifier = _need(raw, "identifier", where)
    pmid = str(_need(identifier, "pmid", f"{where}: identifier"))
    if not _PMID.match(pmid):
        raise RegistryError(f"{where}: identifier.pmid {pmid!r} is not a PubMed identifier")
    finding = _need(raw, "finding", where)
    if finding not in FINDINGS:
        raise RegistryError(f"{where}: finding must be one of {FINDINGS}, got {finding!r}")

    eligibility = _need(raw, "eligibility", where)
    base = _need(eligibility, "base", f"{where}: eligibility")
    if base not in eligibility_columns:
        raise RegistryError(f"{where}: eligibility.base {base!r} is not a cohort eligibility column")
    rules = _rules(eligibility.get("rules") or [], sets, f"{where}: eligibility")

    exposure = _need(raw, "exposure", where)
    column = _need(exposure, "assignment_column", f"{where}: exposure")
    if column not in assignment_columns:
        raise RegistryError(f"{where}: exposure.assignment_column {column!r} is not a cohort arm column")
    treated = _need(exposure, "treated", f"{where}: exposure")
    control = _need(exposure, "control", f"{where}: exposure")
    for arm in (treated, control):
        if arm not in arms:
            raise RegistryError(f"{where}: arm {arm!r} is not a cohort arm {arms}")
    if treated == control:
        raise RegistryError(f"{where}: treated and control arms must differ")

    outcome = _need(raw, "outcome", where)
    event = _need(outcome, "event", f"{where}: outcome")
    if event not in EVENTS:
        raise RegistryError(f"{where}: outcome.event must be one of {EVENTS}, got {event!r}")
    window = _number(
        _need(outcome, "window_hours", f"{where}: outcome"), f"{where}: outcome.window_hours",
        minimum=0.0, strict=True,
    )

    representability = {
        "eligibility": _need(eligibility, "representability", f"{where}: eligibility"),
        "exposure": _need(exposure, "representability", f"{where}: exposure"),
        "outcome": _need(outcome, "representability", f"{where}: outcome"),
    }
    for part, value in representability.items():
        if value not in REPRESENTABILITY:
            raise RegistryError(f"{where}: {part}.representability must be one of {REPRESENTABILITY}")

    published_arms = _need(raw, "arms_published", where)
    a, n1, flags_t = _arm_counts(
        _need(published_arms, "treated", f"{where}: arms_published"), f"{where}: arms_published.treated")
    c, n0, flags_c = _arm_counts(
        _need(published_arms, "control", f"{where}: arms_published"), f"{where}: arms_published.control")
    published = _need(raw, "published_effect", where)
    unverified = [*flags_t, *flags_c]
    if unverified:
        # An arm whose counts do not reproduce its own published percentage cannot
        # reproduce the published effect either; the effect is carried as unverified.
        published = {**published, "unverified": True}
    _check_published(published, (a, n1, c, n0), f"{where}: published effect")
    if published.get("unverified"):
        unverified.append("published_effect")
    if not identifier.get("verified", False):
        unverified.append("identifier")

    bias = _need(raw, "expected_bias", where)
    direction = _need(bias, "direction", f"{where}: expected_bias")
    if direction not in BIAS_DIRECTIONS:
        raise RegistryError(f"{where}: expected_bias.direction must be one of {tuple(BIAS_DIRECTIONS)}")
    _need(bias, "reason", f"{where}: expected_bias")

    effect = relative_effect_from_counts(a, n1, c, n0, alpha=alpha)
    covers_one = effect.lower <= 1.0 <= effect.upper
    if covers_one != (finding == "null"):
        raise RegistryError(
            f"{where}: finding {finding!r} does not match the benchmark interval "
            f"({effect.lower:.2f} to {effect.upper:.2f})"
        )
    margin = derive_margin(effect, finding=finding, event=event, n_treated=n1, n_control=n0,
                           alpha=alpha, **margin_rule)
    return Trial(
        trial_id=trial_id, label=str(raw.get("label", trial_id)), pmid=pmid,
        identifier_verified=bool(identifier.get("verified", False)), finding=finding,
        approximate=bool(raw.get("approximate", False)), base_column=base, rules=rules,
        assignment_column=column, treated=treated, control=control, event=event, window_hours=window,
        representability=representability, effect=effect, published=dict(published),
        expected_bias_direction=direction, expected_bias_sign=BIAS_DIRECTIONS[direction],
        unverified=tuple(unverified), overrides=dict(raw.get("feasibility_screen") or {}),
        margin=margin,
    )


def _screen(raw: Any) -> dict[str, Any]:
    where = "feasibility_screen"
    if _need(raw, "status", where) not in STATUSES:
        raise RegistryError(f"{where}: status must be one of {STATUSES}")
    overlap, precision = _need(raw, "overlap", where), _need(raw, "precision", where)
    return {
        "status": raw["status"],
        "min_arm_n": _number(_need(raw, "min_arm_n", where), f"{where}.min_arm_n", minimum=1),
        "min_probability": _number(
            _need(overlap, "min_probability", where), f"{where}.overlap.min_probability", minimum=0.0),
        "max_share_below": _number(
            _need(overlap, "max_share_below", where), f"{where}.overlap.max_share_below", minimum=0.0),
        "min_effective_sample_size": _number(
            _need(raw, "min_effective_sample_size", where), f"{where}.min_effective_sample_size", minimum=1),
        "alpha": _number(_need(precision, "alpha", where), f"{where}.precision.alpha", minimum=0.0, strict=True),
        "power": _number(_need(precision, "power", where), f"{where}.precision.power", minimum=0.0, strict=True),
        "max_multiple_of_trial_difference": _number(
            _need(precision, "max_multiple_of_trial_difference", where),
            f"{where}.precision.max_multiple_of_trial_difference", minimum=0.0, strict=True),
        "max_fraction_of_equivalence_margin": _number(
            _need(precision, "max_fraction_of_equivalence_margin", where),
            f"{where}.precision.max_fraction_of_equivalence_margin", minimum=0.0, strict=True),
    }


def validate_registry(raw: Mapping[str, Any]) -> Registry:
    """Check a parsed registry and return it as typed objects; raise `RegistryError`."""
    if not isinstance(raw, Mapping):
        raise RegistryError("registry must be a mapping")
    status = _need(raw, "status", "registry")
    if status not in STATUSES:
        raise RegistryError(f"registry: status must be one of {STATUSES}")
    cohort = _need(raw, "cohort", "registry")
    arms = _strings(_need(cohort, "arms", "cohort"), "cohort.arms")
    grouping = raw.get("arm_device_categories")
    if grouping is not None:
        if not isinstance(grouping, Mapping) or set(grouping) != set(arms) or not all(
                _strings(v, f"arm_device_categories.{k}") for k, v in grouping.items()):
            raise RegistryError("arm_device_categories must give a non-empty list of CLIF device "
                                "categories for exactly the cohort arms")
    assignment_columns = _strings(_need(cohort, "assignment_columns", "cohort"), "cohort.assignment_columns")
    eligibility_columns = _strings(_need(cohort, "eligibility_columns", "cohort"), "cohort.eligibility_columns")
    sets = {
        name: _strings(columns, f"risk_factor_sets.{name}")
        for name, columns in _need(raw, "risk_factor_sets", "registry").items()
    }

    agreement = _need(raw, "agreement", "registry")
    if _need(agreement, "scale", "agreement") != "log_risk_ratio":
        raise RegistryError("agreement.scale must be log_risk_ratio")
    alpha = _number(_need(agreement, "alpha", "agreement"), "agreement.alpha", minimum=0.0, strict=True)
    if alpha >= 1.0:
        raise RegistryError("agreement.alpha must lie strictly between 0 and 1")
    if "agreement_margin_ratio" in agreement:
        raise RegistryError("agreement.agreement_margin_ratio is retired: each trial's margin is "
                            "derived under agreement.margin_derivation (item 42)")
    derivation = _need(agreement, "margin_derivation", "agreement")
    if _need(derivation, "method", "agreement.margin_derivation") not in MARGIN_METHODS:
        raise RegistryError(f"agreement.margin_derivation.method must be one of {MARGIN_METHODS}")
    preserved = _parameter(derivation, "preserved_fraction", "agreement.margin_derivation")
    if not 0.0 < _number(preserved.value, "agreement.margin_derivation.preserved_fraction") < 1.0:
        raise RegistryError("agreement.margin_derivation.preserved_fraction must lie in (0, 1)")
    cap = _parameter(derivation, "mortality_composite_cap_ratio", "agreement.margin_derivation")
    lo, hi = MORTALITY_CAP_RANGE
    if not lo <= _number(cap.value, "agreement.margin_derivation.mortality_composite_cap_ratio") <= hi:
        raise RegistryError(f"agreement.margin_derivation.mortality_composite_cap_ratio must lie in "
                            f"[{lo}, {hi}] (item 42)")
    mortality_events = _strings(_need(derivation, "mortality_type_events", "agreement.margin_derivation"),
                                "agreement.margin_derivation.mortality_type_events")
    if not set(mortality_events) <= set(EVENTS):
        raise RegistryError(f"agreement.margin_derivation.mortality_type_events must be within {EVENTS}")
    equivalence = _parameter(agreement, "equivalence_margin_ratio", "agreement")
    _number(equivalence.value, "agreement.equivalence_margin_ratio", minimum=1.0, strict=True)
    hurdle = _parameter(agreement, "second_hurdle", "agreement")
    if hurdle.value not in SECOND_HURDLES:
        raise RegistryError(f"agreement.second_hurdle must be one of {SECOND_HURDLES}")
    status_of = {preserved.status, cap.status, equivalence.status}
    margin_rule = {
        "preserved_fraction": float(preserved.value), "equivalence_ratio": float(equivalence.value),
        "cap_ratio": float(cap.value), "mortality_type_events": mortality_events,
        "status": "registered" if status_of == {"registered"} else "proposed",
    }
    pooling = _parameter(agreement, "pooling", "agreement")
    if pooling.value not in POOLINGS:
        raise RegistryError(f"agreement.pooling must be one of {POOLINGS}, got {pooling.value!r}")

    estimand = _need(raw, "estimand", "registry")
    rule = _parameter(estimand, "discharge_alive_rule", "estimand")
    if rule.value not in DISCHARGE_ALIVE_RULES:
        raise RegistryError(f"estimand.discharge_alive_rule must be one of {DISCHARGE_ALIVE_RULES}")
    sensitivity = _strings(estimand.get("discharge_alive_rule_sensitivity", []), "estimand")
    if not set(sensitivity) <= set(DISCHARGE_ALIVE_RULES):
        raise RegistryError(f"estimand.discharge_alive_rule_sensitivity must be within {DISCHARGE_ALIVE_RULES}")
    hospice = _parameter(estimand, "hospice_rule", "estimand")
    if hospice.value not in HOSPICE_RULES:
        raise RegistryError(f"estimand.hospice_rule must be one of {HOSPICE_RULES}")
    hospice_sensitivity = _strings(estimand.get("hospice_rule_sensitivity", []), "estimand")
    if not set(hospice_sensitivity) <= set(HOSPICE_RULES):
        raise RegistryError(f"estimand.hospice_rule_sensitivity must be within {HOSPICE_RULES}")
    unresolved = _parameter(estimand, "max_unresolved_share", "estimand")
    if not 0.0 <= _number(unresolved.value, "estimand.max_unresolved_share") < 1.0:
        raise RegistryError("estimand.max_unresolved_share must lie in [0, 1)")
    primary = _need(estimand, "study_primary_outcome", "estimand")
    if _need(primary, "event", "estimand.study_primary_outcome") not in EVENTS:
        raise RegistryError(f"estimand.study_primary_outcome.event must be one of {EVENTS}")
    missing = _parameter(raw, "missing_risk_factor", "registry")
    if missing.value != "absent":
        raise RegistryError("missing_risk_factor: only `absent` is implemented")

    adjustment = _need(raw, "adjustment", "registry")
    covariates = _strings(_need(adjustment, "covariates", "adjustment"), "adjustment.covariates")
    forbidden = _strings(adjustment.get("forbidden", []), "adjustment.forbidden")
    if not covariates or len(set(covariates)) != len(covariates):
        raise RegistryError("adjustment.covariates must be a non-empty list of distinct columns")
    leaked = sorted(set(covariates) & (set(forbidden) | set(assignment_columns)))
    if leaked:
        raise RegistryError(f"adjustment.covariates holds exposure or instrument-like columns: {leaked}")

    sites = {}
    for site, declaration in _need(raw, "sites", "registry").items():
        role = _need(declaration, "role", f"sites.{site}")
        if role not in SITE_ROLES:
            raise RegistryError(f"sites.{site}: role must be one of {SITE_ROLES}")
        sites[str(site)] = role
    freeze = _need(raw, "freeze_manifest", "registry")
    required = _strings(_need(freeze, "required", "freeze_manifest"), "freeze_manifest.required")
    local = _strings(_need(freeze, "locally_verified", "freeze_manifest"), "freeze_manifest.locally_verified")
    if not set(local) <= set(required):
        raise RegistryError("freeze_manifest.locally_verified must be a subset of `required`")

    raw_trials = _need(raw, "trials", "registry")
    if not isinstance(raw_trials, Mapping) or not raw_trials:
        raise RegistryError("registry: `trials` must hold at least one trial")
    trials = {
        str(trial_id): _trial(
            str(trial_id), entry, arms=arms, assignment_columns=assignment_columns,
            eligibility_columns=eligibility_columns, sets=sets, alpha=alpha, margin_rule=margin_rule,
        )
        for trial_id, entry in raw_trials.items()
    }
    audit = _need(raw, "pre_registration_audit", "registry")
    audit_trial = _need(audit, "trial", "pre_registration_audit")
    if audit_trial not in trials:
        raise RegistryError(f"pre_registration_audit: unknown trial {audit_trial!r}")
    if _need(audit, "site_role", "pre_registration_audit") != "exploratory":
        raise RegistryError("pre_registration_audit: the audit comparison is for exploratory sites only")

    return Registry(
        version=str(_need(raw, "registry_version", "registry")), status=status, trials=trials, arms=arms,
        assignment_columns=assignment_columns, eligibility_columns=eligibility_columns,
        patient_id_column=str(_need(cohort, "patient_id_column", "cohort")), sites=sites,
        audit_trial=audit_trial, audit_site_role="exploratory", freeze_required=required,
        freeze_locally_verified=local, study_primary_event=primary["event"],
        study_primary_window_hours=_number(
            _need(primary, "window_hours", "estimand.study_primary_outcome"),
            "estimand.study_primary_outcome.window_hours", minimum=0.0, strict=True),
        discharge_alive_rule=rule, discharge_alive_rule_sensitivity=sensitivity,
        hospice_rule=hospice, hospice_rule_sensitivity=hospice_sensitivity,
        max_unresolved_share=unresolved, missing_risk_factor=missing, covariates=covariates,
        forbidden_covariates=forbidden, alpha=alpha, preserved_fraction=preserved,
        mortality_cap_ratio=cap, mortality_type_events=mortality_events,
        equivalence_margin_ratio=equivalence, second_hurdle=hurdle, pooling=pooling,
        screen=_screen(_need(raw, "feasibility_screen", "registry")),
    )


def load_benchmark_registry(path: Any) -> Registry:
    """Read and validate `configs/extubation_benchmarks.yaml`."""
    import yaml  # PyYAML is declared by the site package; kept out of the numeric paths

    with open(path, encoding="utf-8") as handle:
        return validate_registry(yaml.safe_load(handle))


def check_registry_against_cohort(registry: Registry, columns: Sequence[str], arms: Sequence[str]) -> None:
    """Every trial must map to arms and columns that exist in the cohort."""
    present, cohort_arms = set(columns), set(arms)
    missing_arms = sorted(set(registry.arms) - cohort_arms)
    if missing_arms:
        raise RegistryError(f"registry arms missing from the cohort: {missing_arms}")
    for trial in registry.trials.values():
        absent = sorted(trial.columns() - present)
        if absent:
            raise RegistryError(f"trial {trial.trial_id}: cohort has no column(s) {absent}")
        for arm in (trial.treated, trial.control):
            if arm not in cohort_arms:
                raise RegistryError(f"trial {trial.trial_id}: cohort has no arm {arm!r}")
    absent = sorted(registry.columns_used() - present)
    if absent:
        raise RegistryError(f"cohort has no column(s) {absent} named by the registry")


# ---------------------------------------------------------------------------
# estimates, gaps and equivalence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialEstimate:
    """One emulation's relative effect for one trial: log risk ratio and its standard error."""

    log_effect: float
    se: float
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_effect):
            raise ValueError("log_effect must be finite")
        if not math.isfinite(self.se) or self.se < 0:
            raise ValueError("se must be finite and non-negative")
        _z(self.alpha)

    @classmethod
    def from_ratio(cls, interval: Any, *, alpha: float = 0.05) -> TrialEstimate:
        """From a risk-ratio `IntervalEstimate` (`estimate` a ratio, `se` on the log scale)."""
        return cls(log_effect=math.log(interval.estimate), se=float(interval.se), alpha=alpha)

    @property
    def ratio(self) -> float:
        return math.exp(self.log_effect)

    @property
    def lower(self) -> float:
        return math.exp(self.log_effect - _z(self.alpha) * self.se)

    @property
    def upper(self) -> float:
        return math.exp(self.log_effect + _z(self.alpha) * self.se)


@dataclass(frozen=True)
class TrialGap:
    """Emulation minus benchmark on the log risk-ratio scale."""

    trial_id: str
    finding: str
    log_gap: float
    abs_log_gap: float
    ratio_of_ratios: float
    se: float                 # sqrt(se_emulation^2 + se_benchmark^2)
    estimate_ratio: float
    estimate_lower: float
    estimate_upper: float
    benchmark_ratio: float
    approximate: bool


@dataclass(frozen=True)
class NullTrialResult:
    trial_id: str
    gap: TrialGap
    margin_ratio: float
    reproduced: bool
    interval_rule: bool = True
    direction_consistent: bool = True


@dataclass(frozen=True)
class TrialAgreement:
    """One trial's verdict: the interval rule, the second hurdle and their conjunction."""

    trial_id: str
    finding: str
    interval_rule: bool
    direction_consistent: bool
    second_hurdle_applied: bool
    reproduced: bool
    margin: TrialMargin

    def as_dict(self) -> dict[str, Any]:
        return {"finding": self.finding, "interval_rule": self.interval_rule,
                "direction_consistent": self.direction_consistent,
                "second_hurdle_applied": self.second_hurdle_applied,
                "agrees_with_trial": self.reproduced, "margin": self.margin.as_dict()}


def trial_gap(trial: Trial, estimate: TrialEstimate) -> TrialGap:
    log_gap = estimate.log_effect - trial.effect.log_estimate
    return TrialGap(
        trial_id=trial.trial_id, finding=trial.finding, log_gap=log_gap, abs_log_gap=abs(log_gap),
        ratio_of_ratios=math.exp(log_gap), se=math.hypot(estimate.se, trial.effect.se_log),
        estimate_ratio=estimate.ratio, estimate_lower=estimate.lower, estimate_upper=estimate.upper,
        benchmark_ratio=trial.effect.estimate, approximate=trial.approximate,
    )


def reproduces_null(estimate: TrialEstimate, margin_ratio: float) -> bool:
    """A null trial is reproduced only when the whole interval lies inside [1/m, m].

    A point estimate near one is not enough: a wide interval fails, so an underpowered
    emulation cannot "reproduce" a null result by being imprecise.
    """
    if not margin_ratio > 1.0:
        raise ValueError("the equivalence margin must be a ratio above 1")
    return bool(1.0 / margin_ratio <= estimate.lower and estimate.upper <= margin_ratio)


def direction_consistent(trial: Trial, estimate: TrialEstimate) -> bool:
    """The second hurdle: the emulation's point estimate is not on the opposite side of
    no effect from the benchmark's point estimate (an estimate of exactly 1 is allowed)."""
    return bool(estimate.log_effect * trial.effect.log_estimate >= 0.0)


def trial_agreement(trial: Trial, estimate: TrialEstimate, *, second_hurdle: bool = True) -> TrialAgreement:
    """One trial's verdict under its own margin: effect trial, |gap| within the margin;
    null trial, the whole interval inside [1/m, m]; and, with `second_hurdle`, the point
    estimate direction-consistent with the benchmark."""
    if trial.margin is None:
        raise AgreementError(f"trial {trial.trial_id} has no derived margin")
    if trial.finding == "null":
        interval = reproduces_null(estimate, trial.margin.ratio)
    else:
        interval = bool(trial_gap(trial, estimate).abs_log_gap <= trial.margin.log)
    consistent = direction_consistent(trial, estimate)
    return TrialAgreement(
        trial_id=trial.trial_id, finding=trial.finding, interval_rule=interval,
        direction_consistent=consistent, second_hurdle_applied=second_hurdle,
        reproduced=bool(interval and (consistent or not second_hurdle)), margin=trial.margin)


def single_trial_agreement(trial: Trial, estimate: TrialEstimate, *, second_hurdle: bool = True) -> bool:
    """The rule applied to one trial alone (used by the planted-effect simulation)."""
    return trial_agreement(trial, estimate, second_hurdle=second_hurdle).reproduced


def pooled_absolute_gap(log_gaps: Sequence[float]) -> float:
    """Unweighted mean of |gap|: gaps of opposite sign cannot cancel."""
    if len(log_gaps) == 0:
        raise ValueError("no gaps to pool")
    return float(np.mean(np.abs(np.asarray(log_gaps, dtype=np.float64))))


def pooled_signed_gap(log_gaps: Sequence[float], ses: Sequence[float]) -> tuple[float, float]:
    """Inverse-variance weighted mean of signed gaps and its standard error."""
    gaps, se = np.asarray(log_gaps, dtype=np.float64), np.asarray(ses, dtype=np.float64)
    if gaps.size == 0 or gaps.shape != se.shape:
        raise ValueError("gaps and standard errors must be non-empty and the same length")
    if np.any(se <= 0):
        return float(gaps.mean()), 0.0
    weights = 1.0 / se ** 2
    return float(np.sum(weights * gaps) / weights.sum()), float(math.sqrt(1.0 / weights.sum()))


# ---------------------------------------------------------------------------
# feasibility screen (R25)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeasibilityInputs:
    """Outcome-blind measurements for one trial on one cohort.

    n_*                          patients observed in each compared arm.
    ess_*                        Kish effective sample size of each arm's weights.
    share_below_min_probability  share of the patients in the TWO compared arms whose
                                 probability of the treated arm (given one of the two)
                                 lies outside [floor, 1 - floor] (item 44).
    estimators_runnable          estimator name -> whether it can run on this trial.
    """

    n_treated: int
    n_control: int
    ess_treated: float
    ess_control: float
    share_below_min_probability: float
    estimators_runnable: Mapping[str, bool] = field(default_factory=dict)
    # Item 44: overlap is measured on the two compared arms only. `n_compared` patients
    # follow one of them; `share_trimmed_of_compared` / `share_trimmed_of_eligible` are the
    # shares a trim to [min_probability, 1 - min_probability] would exclude, over the two
    # arms and over all trial-eligible patients. None when not measured.
    n_eligible: int | None = None
    n_compared: int | None = None
    share_trimmed_of_compared: float | None = None
    share_trimmed_of_eligible: float | None = None


@dataclass(frozen=True)
class FeasibilityResult:
    trial_id: str
    evaluable: bool
    reasons: tuple[str, ...]
    checks: dict[str, Any]


def _detectable_difference(n_treated: float, n_control: float, baseline_risk: float, alpha: float, power: float) -> float:
    """Same closed form as `diagnostics.minimal_detectable_effect` (kept local: numpy-free)."""
    normal = NormalDist()
    z = normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(power)
    return z * math.sqrt(baseline_risk * (1.0 - baseline_risk) * (1.0 / n_treated + 1.0 / n_control))


def feasibility_screen(registry: Registry, trial_id: str, inputs: FeasibilityInputs) -> FeasibilityResult:
    """Apply the registered R25 thresholds to one trial. No outcome by arm is read.

    Precision is projected from the effective sample sizes and the TRIAL's published arm
    risks, never from the cohort's own outcomes. For a trial that found an effect the
    smallest detectable risk difference must be at most a registered multiple of the
    trial's difference; for a null trial the projected interval half-width of the log
    risk ratio must fit inside the equivalence margin.
    """
    if trial_id not in registry.trials:
        raise AgreementError(f"unknown trial {trial_id!r}")
    trial = registry.trials[trial_id]
    screen = {**registry.screen, **trial.overrides}
    reasons: list[str] = []
    for part, value in trial.representability.items():
        if value == "not_representable":
            reasons.append(f"{part} is not representable in the cohort")
    for name, runnable in sorted(inputs.estimators_runnable.items()):
        if not runnable:
            reasons.append(f"estimator {name} cannot run on this trial")
    smallest = min(inputs.n_treated, inputs.n_control)
    if smallest < screen["min_arm_n"]:
        reasons.append(f"arm size below the minimum of {screen['min_arm_n']:g}")
    if inputs.share_below_min_probability > screen["max_share_below"]:
        reasons.append(
            f"overlap: more than {screen['max_share_below']:.0%} of the population has a probability "
            f"below {screen['min_probability']:g} of following a compared arm"
        )
    ess = min(inputs.ess_treated, inputs.ess_control)
    if not ess >= screen["min_effective_sample_size"]:
        reasons.append(f"effective sample size below the minimum of {screen['min_effective_sample_size']:g}")

    checks: dict[str, Any] = {
        "min_arm_n": smallest, "effective_sample_size": ess,
        "share_below_min_probability": inputs.share_below_min_probability,
        "thresholds_status": screen["status"],
    }
    for key in ("share_trimmed_of_compared", "share_trimmed_of_eligible"):
        if getattr(inputs, key) is not None:
            checks[key] = getattr(inputs, key)
    effect = trial.effect
    if inputs.ess_treated > 0 and inputs.ess_control > 0:
        if trial.finding == "effect":
            detectable = _detectable_difference(
                inputs.ess_treated, inputs.ess_control, effect.risk_control, screen["alpha"], screen["power"])
            allowed = screen["max_multiple_of_trial_difference"] * abs(effect.risk_treated - effect.risk_control)
            checks.update(detectable_difference=detectable, allowed_detectable_difference=allowed)
            if detectable > allowed:
                reasons.append("precision: the smallest detectable difference exceeds the registered multiple of the trial's")
        else:
            projected_se = math.sqrt(
                (1.0 - effect.risk_treated) / (inputs.ess_treated * effect.risk_treated)
                + (1.0 - effect.risk_control) / (inputs.ess_control * effect.risk_control)
            )
            half_width = NormalDist().inv_cdf(1.0 - screen["alpha"] / 2.0) * projected_se
            allowed = screen["max_fraction_of_equivalence_margin"] * trial.margin.log
            checks.update(projected_half_width=half_width, allowed_half_width=allowed)
            if half_width > allowed:
                reasons.append("precision: the projected interval cannot fit inside the equivalence margin")
    return FeasibilityResult(trial_id=trial_id, evaluable=not reasons, reasons=tuple(reasons), checks=checks)


# ---------------------------------------------------------------------------
# agreement across trials (R10)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EstimatorAgreement:
    """One estimator's agreement with the evaluable trials.

    `passes` judges the trials that found an effect. Each gap is divided by its trial's
    own log margin, so the pooled statistic is on the margin scale and passes at most
    `threshold` (1.0); with the second hurdle every scored effect trial must also be
    direction-consistent. It is None when no such trial is evaluable (no verdict, not a
    pass). Null trials are listed separately with their equivalence result.
    """

    estimator: str
    effect_gaps: tuple[TrialGap, ...]
    pooled_abs_gap: float | None
    pooled_signed_gap: float | None
    pooled_signed_se: float | None
    pooled_statistic: float | None
    pooling: str
    threshold: float
    margins_log: dict[str, float]
    direction_consistent: dict[str, bool]
    margin_status: str
    passes: bool | None
    null_results: tuple[NullTrialResult, ...]
    n_null_reproduced: int


@dataclass(frozen=True)
class AgreementReport:
    evaluable: tuple[str, ...]
    not_evaluable: dict[str, tuple[str, ...]]
    estimators: dict[str, EstimatorAgreement]
    margins_status: str


def score_agreement(
    registry: Registry,
    estimates: Mapping[str, Mapping[str, TrialEstimate]],
    feasibility: Mapping[str, FeasibilityResult],
) -> AgreementReport:
    """Score every estimator on the same set of evaluable trials.

    `estimates` maps estimator name -> trial id -> `TrialEstimate`. `feasibility` must
    hold a screen result for every registered trial (fail closed: a trial is never scored
    without one). A trial is evaluable only if it passed the screen AND every estimator
    produced an estimate for it, so all estimators are always compared on identical
    trials; anything else is listed under `not_evaluable` with the reasons and is left
    out of every estimator's results, whatever estimates were supplied for it.
    """
    if not estimates:
        raise AgreementError("no estimator results to score")
    missing = sorted(set(registry.trials) - set(feasibility))
    if missing:
        raise AgreementError(f"no feasibility screen result for trial(s) {missing}")
    for name, by_trial in estimates.items():
        unknown = sorted(set(by_trial) - set(registry.trials))
        if unknown:
            raise AgreementError(f"estimator {name}: unknown trial(s) {unknown}")

    not_evaluable: dict[str, tuple[str, ...]] = {}
    evaluable: list[str] = []
    for trial_id in registry.trials:
        screen = feasibility[trial_id]
        if screen.trial_id != trial_id:
            raise AgreementError(f"feasibility result for {trial_id} belongs to {screen.trial_id}")
        reasons = list(screen.reasons) if not screen.evaluable else []
        if not screen.evaluable and not reasons:
            reasons.append("failed the feasibility screen")
        if screen.evaluable:
            reasons.extend(
                f"estimator {name} produced no estimate"
                for name, by_trial in sorted(estimates.items()) if trial_id not in by_trial
            )
        if reasons:
            not_evaluable[trial_id] = tuple(reasons)
        else:
            evaluable.append(trial_id)

    pooling = registry.pooling.value
    hurdle = registry.uses_second_hurdle
    margins_log = {trial_id: registry.trials[trial_id].margin.log for trial_id in evaluable}
    results = {}
    for name, by_trial in estimates.items():
        gaps = {trial_id: trial_gap(registry.trials[trial_id], by_trial[trial_id]) for trial_id in evaluable}
        verdicts = {trial_id: trial_agreement(registry.trials[trial_id], by_trial[trial_id], second_hurdle=hurdle)
                    for trial_id in evaluable}
        effect_gaps = tuple(g for g in gaps.values() if g.finding == "effect")
        null_results = tuple(
            NullTrialResult(trial_id, gap, verdicts[trial_id].margin.ratio, verdicts[trial_id].reproduced,
                            verdicts[trial_id].interval_rule, verdicts[trial_id].direction_consistent)
            for trial_id, gap in gaps.items() if gap.finding == "null"
        )
        pooled_abs = signed = signed_se = statistic = passes = None
        if effect_gaps:
            pooled_abs = pooled_absolute_gap([g.log_gap for g in effect_gaps])
            signed, signed_se = pooled_signed_gap([g.log_gap for g in effect_gaps], [g.se for g in effect_gaps])
            scaled = [g.log_gap / margins_log[g.trial_id] for g in effect_gaps]
            if pooling == "mean_absolute_gap":
                statistic = pooled_absolute_gap(scaled)
            else:
                statistic = abs(pooled_signed_gap(
                    scaled, [g.se / margins_log[g.trial_id] for g in effect_gaps])[0])
            consistent = all(verdicts[g.trial_id].direction_consistent for g in effect_gaps)
            passes = bool(statistic <= 1.0 and (consistent or not hurdle))
        results[name] = EstimatorAgreement(
            estimator=name, effect_gaps=effect_gaps, pooled_abs_gap=pooled_abs, pooled_signed_gap=signed,
            pooled_signed_se=signed_se, pooled_statistic=statistic, pooling=pooling, threshold=1.0,
            margins_log=dict(margins_log),
            direction_consistent={t: v.direction_consistent for t, v in verdicts.items()},
            margin_status=registry.margins_status(), passes=passes, null_results=null_results,
            n_null_reproduced=sum(r.reproduced for r in null_results),
        )
    return AgreementReport(
        evaluable=tuple(evaluable), not_evaluable=not_evaluable, estimators=results,
        margins_status=registry.margins_status(),
    )


# ---------------------------------------------------------------------------
# paired comparison between two estimators (R27)
# ---------------------------------------------------------------------------

Estimator = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class PairedTrial:
    """Two estimators of one trial's log risk ratio on the same patients.

    `estimate_a` and `estimate_b` take an integer index array into the trial's own
    patients (positions 0..len(members)-1, repeated for patients drawn more than once)
    and return the log risk ratio on that resample. `members` gives each of the trial's
    patients a position in the union cohort, so a patient who is in several trials is
    drawn the same number of times in each.
    """

    trial_id: str
    estimate_a: Estimator
    estimate_b: Estimator
    benchmark_log_effect: float
    members: np.ndarray


@dataclass(frozen=True)
class GapDifference:
    difference: float       # |gap of a| - |gap of b|; negative when a is nearer the benchmark
    lower: float
    upper: float


@dataclass(frozen=True)
class PairedGapDifference:
    difference: float       # mean over trials of the per-trial difference
    lower: float
    upper: float
    per_trial: dict[str, GapDifference]
    n_boot: int
    n_failed: int           # resamples on which an estimator could not be computed
    alpha: float


def linearized_estimator(log_effect: float, influence: np.ndarray) -> Estimator:
    """Bootstrap an estimate from its per-patient influence values instead of refitting.

    `influence` holds one centred influence value of the LOG risk ratio per patient (for
    `ArmRisks`: influence_treated / risk_treated - influence_control / risk_control). On a
    resample the estimate is `log_effect + mean(influence[index])`. The full sample
    returns `log_effect` exactly.
    """
    values = np.asarray(influence, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("influence must be a non-empty 1-D array of finite values")
    centred = values - values.mean()

    def estimate(index: np.ndarray) -> float:
        return float(log_effect + centred[index].mean())

    return estimate


def paired_gap_difference(
    pairs: Sequence[PairedTrial], *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> PairedGapDifference:
    """Difference in absolute gap to the benchmark, estimator a minus estimator b.

    Patients of the union cohort are resampled with replacement and both estimators are
    evaluated on the identical resample, trial by trial, so the comparison is paired at
    the patient level. The interval is the percentile interval over `n_boot` resamples.
    A negative difference means estimator a is nearer the benchmark. Two identical
    estimators give exactly zero on every resample.
    """
    if not pairs:
        raise ValueError("no trials to compare")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    members = []
    for pair in pairs:
        positions = np.asarray(pair.members)
        if positions.ndim != 1 or positions.size == 0 or not np.issubdtype(positions.dtype, np.integer):
            raise ValueError(f"trial {pair.trial_id}: members must be a non-empty 1-D integer array")
        if np.any(positions < 0) or np.unique(positions).size != positions.size:
            raise ValueError(f"trial {pair.trial_id}: members must be distinct non-negative positions")
        members.append(positions)
    n_union = int(max(positions.max() for positions in members)) + 1

    def differences(counts: np.ndarray | None) -> np.ndarray:
        out = np.empty(len(pairs))
        for i, (pair, positions) in enumerate(zip(pairs, members, strict=True)):
            local = np.arange(positions.size)
            index = local if counts is None else np.repeat(local, counts[positions])
            if index.size == 0:
                raise ArithmeticError("a trial has no patients in this resample")
            gap_a = abs(pair.estimate_a(index) - pair.benchmark_log_effect)
            gap_b = abs(pair.estimate_b(index) - pair.benchmark_log_effect)
            out[i] = gap_a - gap_b
        if not np.all(np.isfinite(out)):
            raise ArithmeticError("an estimator returned a non-finite value")
        return out

    point = differences(None)
    rng = np.random.default_rng(seed)
    draws, failed = [], 0
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n_union, size=n_union), minlength=n_union)
        try:
            draws.append(differences(counts))
        except (ArithmeticError, ValueError):
            failed += 1
    if not draws:
        raise AgreementError("every bootstrap resample failed")
    sample = np.stack(draws)
    per_trial = {
        pair.trial_id: GapDifference(
            float(point[i]), float(np.quantile(sample[:, i], lo)), float(np.quantile(sample[:, i], hi)))
        for i, pair in enumerate(pairs)
    }
    pooled = sample.mean(axis=1)
    return PairedGapDifference(
        difference=float(point.mean()), lower=float(np.quantile(pooled, lo)), upper=float(np.quantile(pooled, hi)),
        per_trial=per_trial, n_boot=n_boot, n_failed=failed, alpha=alpha,
    )
