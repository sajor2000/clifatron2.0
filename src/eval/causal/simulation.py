"""Planted-effect (plasmode) simulation of the agreement rule (plan U12; R29).

How often does the registered agreement rule pass when the true effect equals the
trial's, is zero, or has the opposite sign, with and without a confounder the estimator
cannot see? The answer is published with the results so a reader can judge what a pass
means.

Recipe (Shaw 2026, PMID 42487285; Desai 2026, PMID 42093129):
1. Resample covariate rows from the frozen cohort.
2. Draw each row's device from a device-choice model fitted to the real covariates and
   arms. Treatment is never assigned at random, which would create positivity violations
   the real data do not have; the simulated propensities are the fitted ones.
3. Give each row a baseline risk: from an outcome model fitted WITHOUT the arm (overall
   risk given covariates) or from a supplied number. No outcome is ever read by arm.
4. Plant the effect on the risk-ratio scale: risk under the treated arm is the baseline
   times the planted ratio; the control and any other arm keep the baseline.
5. Optionally withhold a confounder U ~ N(0, 1): it shifts the device choice toward the
   arm the registry expects sicker patients to receive (its `expected_bias` direction)
   and multiplies every arm's risk by exp(delta U - delta^2 / 2). U is not given to the
   estimator.
6. Run the estimator and apply the agreement rule to the trial alone.

The default estimator is the point-treatment AIPW estimator. With the decision at time
zero it is the same computation as the one-step clone-censor-weight estimator (U11), and
the simulated data have no grace window or censoring. Pass `estimator` to use another.

Standard library, numpy and scikit-learn only: vendored into `clif-validate`.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import benchmark as bm
from . import estimators as est

EFFECTS = ("trial", "zero", "reversed")
RISK_CEILING = 0.99
RISK_FLOOR = 1e-4


def _learner(config: est.NuisanceConfig):
    if config.learner == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=config.logistic_c, max_iter=1000))
    return HistGradientBoostingClassifier(
        max_iter=config.max_iter, learning_rate=config.learning_rate, max_depth=config.max_depth,
        early_stopping=False, random_state=config.seed,
    )


@dataclass(frozen=True)
class DeviceChoiceModel:
    """P(arm | covariates), fitted on the real cohort. Outcome-blind."""

    classes: tuple
    model: object

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        """(n, K) probabilities in the order of `classes`."""
        return np.asarray(self.model.predict_proba(np.asarray(X, dtype=np.float64)))


def fit_device_choice_model(
    X: np.ndarray, arm: np.ndarray, *, config: est.NuisanceConfig | None = None
) -> DeviceChoiceModel:
    config = config or est.NuisanceConfig()
    X = np.asarray(X, dtype=np.float64)
    arm = np.asarray(arm, dtype=object)
    if X.ndim != 2 or arm.shape != (X.shape[0],):
        raise ValueError("X must be 2-D with one arm label per row")
    labels = np.unique(arm.astype(str))
    if labels.size < 2:
        raise ValueError("the device-choice model needs at least two arms")
    model = _learner(config).fit(X, arm.astype(str))
    classes = tuple(str(c) for c in model.classes_)
    return DeviceChoiceModel(classes=classes, model=model)


@dataclass(frozen=True)
class BaselineRiskModel:
    """P(outcome | covariates), fitted with no arm: the overall risk, never risk by arm."""

    model: object

    def risk(self, X: np.ndarray) -> np.ndarray:
        probability = self.model.predict_proba(np.asarray(X, dtype=np.float64))[:, 1]
        return np.clip(probability, RISK_FLOOR, RISK_CEILING)


def fit_baseline_risk_model(
    X: np.ndarray, outcome: np.ndarray, *, config: est.NuisanceConfig | None = None
) -> BaselineRiskModel:
    """Fit the overall outcome risk given covariates. Takes no arm, by design."""
    config = config or est.NuisanceConfig()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    if y.shape != (X.shape[0],) or not np.all((y == 0) | (y == 1)):
        raise ValueError("outcome must be binary with one value per row of X")
    if np.unique(y).size < 2:
        raise ValueError("the baseline model needs both outcome values")
    return BaselineRiskModel(_learner(config).fit(X, y.astype(np.int64)))


@dataclass(frozen=True)
class SimulatedData:
    X: np.ndarray
    arm: np.ndarray
    y: np.ndarray
    propensity: np.ndarray       # (n, K) probabilities the arm was drawn from
    confounder: np.ndarray | None
    share_risk_clipped: float


def _baseline(baseline: float | BaselineRiskModel, X: np.ndarray) -> np.ndarray:
    if isinstance(baseline, BaselineRiskModel):
        return baseline.risk(X)
    value = float(baseline)
    if not 0.0 < value < 1.0:
        raise ValueError("a supplied baseline risk must lie strictly between 0 and 1")
    return np.full(X.shape[0], value)


def simulate_once(
    model: DeviceChoiceModel,
    X: np.ndarray,
    *,
    treated: str,
    control: str,
    baseline: float | BaselineRiskModel,
    planted_rr: float,
    rng: np.random.Generator,
    n_sim: int | None = None,
    confounder: tuple[float, float] | None = None,
    bias_sign: int = 1,
) -> SimulatedData:
    """One plasmode data set. `confounder` = (gamma, delta) withholds U ~ N(0, 1).

    gamma shifts the log-odds of the treated arm by `bias_sign * gamma * U`; delta scales
    every arm's risk by exp(delta U - delta^2 / 2) (mean one over U).
    """
    if treated == control:
        raise ValueError("treated and control arms must differ")
    for label in (treated, control):
        if label not in model.classes:
            raise ValueError(f"arm {label!r} is not in the device-choice model {model.classes}")
    if not math.isfinite(planted_rr) or planted_rr <= 0:
        raise ValueError("planted_rr must be a positive risk ratio")
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0] if n_sim is None else int(n_sim)
    if n < 2:
        raise ValueError("n_sim must be at least 2")
    rows = rng.integers(0, X.shape[0], size=n)
    X_sim = X[rows]
    propensity = model.probabilities(X_sim)
    base = _baseline(baseline, X_sim)
    t = model.classes.index(treated)
    u = None
    if confounder is not None:
        gamma, delta = confounder
        u = rng.normal(size=n)
        logits = np.log(np.clip(propensity, 1e-12, 1.0))
        logits[:, t] += bias_sign * gamma * u
        logits -= logits.max(axis=1, keepdims=True)
        propensity = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        base = base * np.exp(delta * u - delta ** 2 / 2.0)
    cumulative = np.cumsum(propensity, axis=1)
    draw = (rng.random(n)[:, None] > cumulative).sum(axis=1)
    draw = np.minimum(draw, len(model.classes) - 1)
    arm = np.asarray(model.classes, dtype=object)[draw]
    risk = np.where(arm == treated, base * planted_rr, base)
    clipped = (risk > RISK_CEILING) | (risk < RISK_FLOOR)
    risk = np.clip(risk, RISK_FLOOR, RISK_CEILING)
    y = (rng.random(n) < risk).astype(np.float64)
    return SimulatedData(X=X_sim, arm=arm, y=y, propensity=propensity, confounder=u,
                         share_risk_clipped=float(clipped.mean()))


Estimator = Callable[[np.ndarray, np.ndarray, np.ndarray, str, str], bm.TrialEstimate]


def aipw_estimator(config: est.NuisanceConfig, alpha: float) -> Estimator:
    def run(X: np.ndarray, arm: np.ndarray, y: np.ndarray, treated: str, control: str) -> bm.TrialEstimate:
        risks = est.point_treatment_arm_risks(X, arm, y, arms=(treated, control), config=config)
        return bm.TrialEstimate.from_ratio(est.contrast(risks, treated, control, alpha=alpha).risk_ratio, alpha=alpha)

    return run


@dataclass(frozen=True)
class ScenarioResult:
    effect: str
    confounded: bool
    planted_log_rr: float
    n_reps: int
    n_completed: int
    n_failed: int
    n_pass: int
    pass_rate: float | None          # n_pass / n_completed
    passes: tuple[bool, ...]
    estimates: tuple[bm.TrialEstimate, ...]
    mean_log_estimate: float | None
    bias: float | None               # mean log estimate minus planted log ratio
    expected_bias_sign: int
    bias_matches_expected_direction: bool | None


@dataclass(frozen=True)
class SimulationReport:
    trial_id: str
    treated: str
    control: str
    benchmark_log_rr: float
    scenarios: dict[tuple[str, bool], ScenarioResult]
    fitted_propensity_range: dict[str, tuple[float, float]]
    margins_status: str
    n_sim: int

    def summary(self) -> dict[str, dict[str, float | None]]:
        """Pass rate per planted effect, without and with the withheld confounder."""
        out: dict[str, dict[str, float | None]] = {}
        for (effect, confounded), result in self.scenarios.items():
            out.setdefault(effect, {})["confounded" if confounded else "measured"] = result.pass_rate
        return out


def simulate_operating_characteristics(
    X: np.ndarray,
    arm: np.ndarray,
    registry: bm.Registry,
    trial_id: str,
    *,
    baseline: float | BaselineRiskModel,
    n_reps: int = 200,
    n_sim: int | None = None,
    effects: Sequence[str] = EFFECTS,
    confounded: Sequence[bool] = (False, True),
    confounder_strength: tuple[float, float] = (1.0, 0.7),
    config: est.NuisanceConfig | None = None,
    estimator: Estimator | None = None,
    seed: int = 0,
) -> SimulationReport:
    """Pass rates of the registered agreement rule for one trial, per planted scenario.

    `X` and `arm` are the trial-eligible patients of the frozen cohort (covariates and the
    device they received); no outcome is passed. `baseline` is a supplied risk or a model
    from `fit_baseline_risk_model`.
    """
    if n_reps < 1:
        raise ValueError("n_reps must be positive")
    unknown = [e for e in effects if e not in EFFECTS]
    if unknown or not effects:
        raise ValueError(f"effects must be drawn from {EFFECTS}, got {list(effects)}")
    trial = registry.trials[trial_id]
    config = config or est.NuisanceConfig()
    estimator = estimator or aipw_estimator(config, registry.alpha)
    model = fit_device_choice_model(X, arm, config=config)
    fitted = model.probabilities(np.asarray(X, dtype=np.float64))
    margin, equivalence = registry.agreement_margin_ratio.value, registry.equivalence_margin_ratio.value
    planted = {"trial": trial.effect.log_estimate, "zero": 0.0, "reversed": -trial.effect.log_estimate}
    rng = np.random.default_rng(seed)
    scenarios = {}
    for effect in effects:
        for hidden in confounded:
            passes, estimates, failed = [], [], 0
            for _ in range(n_reps):
                data = simulate_once(
                    model, X, treated=trial.treated, control=trial.control, baseline=baseline,
                    planted_rr=math.exp(planted[effect]), rng=rng, n_sim=n_sim,
                    confounder=confounder_strength if hidden else None, bias_sign=trial.expected_bias_sign,
                )
                try:
                    estimate = estimator(data.X, data.arm, data.y, trial.treated, trial.control)
                except (est.EstimationError, ValueError, ArithmeticError):
                    failed += 1
                    continue
                estimates.append(estimate)
                passes.append(bm.single_trial_agreement(
                    trial, estimate, agreement_margin_ratio=margin, equivalence_margin_ratio=equivalence))
            completed = len(passes)
            mean = float(np.mean([e.log_effect for e in estimates])) if estimates else None
            bias = None if mean is None else mean - planted[effect]
            scenarios[(effect, bool(hidden))] = ScenarioResult(
                effect=effect, confounded=bool(hidden), planted_log_rr=planted[effect], n_reps=n_reps,
                n_completed=completed, n_failed=failed, n_pass=int(sum(passes)),
                pass_rate=(sum(passes) / completed) if completed else None, passes=tuple(passes),
                estimates=tuple(estimates), mean_log_estimate=mean, bias=bias,
                expected_bias_sign=trial.expected_bias_sign,
                bias_matches_expected_direction=(
                    None if bias is None or not hidden else bool(np.sign(bias) == trial.expected_bias_sign)),
            )
    return SimulationReport(
        trial_id=trial_id, treated=trial.treated, control=trial.control,
        benchmark_log_rr=trial.effect.log_estimate, scenarios=scenarios,
        fitted_propensity_range={
            label: (float(fitted[:, k].min()), float(fitted[:, k].max())) for k, label in enumerate(model.classes)},
        margins_status=registry.margins_status(), n_sim=int(np.asarray(X).shape[0] if n_sim is None else n_sim),
    )
