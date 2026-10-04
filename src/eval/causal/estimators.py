"""Classical estimators for the extubation target-trial emulation (plan U11; KTD7, KTD8).

Primary analysis: a one-step clone-censor-weight emulation (`clone_censor_weight`). Every
patient is cloned into each compared device arm at extubation, clones that deviate are
artificially censored, the censoring is undone with inverse-probability-of-censoring weights
from a cross-fitted device-choice model, and a cross-fitted outcome model makes the estimate
doubly robust. Sensitivity analysis: the first device as a fixed exposure
(`point_treatment_arm_risks`, augmented inverse-probability weighting) and overlap weights
(`overlap_weighted_contrast`). With the decision at time zero the two analyses are the same
computation.

The target population is always every row supplied: a two-arm contrast in a three-arm cohort
keeps the third arm's patients in the population and estimates what each compared device
would have done for all of them. `overlap_weighted_contrast` is the exception and says so.

Intervals come from influence functions summed within patient, with the nuisance models
treated as known (the usual cross-fitting argument). There is no bootstrap helper here.

The device-choice model is a nuisance fitted inside the estimator. It is never a training
target of the foundation model (hard rule 1 stays intact).

Standard library, numpy and scikit-learn only: this module is vendored into `clif-validate`,
which does not declare LightGBM, polars or scipy. Arrays in, plain dataclasses out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LEARNERS: tuple[str, ...] = ("hist_gradient_boosting", "logistic")


class EstimationError(ValueError):
    """The data cannot support the requested estimate (empty arm, single-class outcome, ...)."""


@dataclass(frozen=True)
class NuisanceConfig:
    """How the device-choice and outcome models are fitted.

    learner        "hist_gradient_boosting" (default; accepts missing covariates) or
                   "logistic" (ridge-penalized, standardized inputs; refuses missing values).
    n_folds        cross-fitting folds, assigned by patient.
    seed           fixes the fold assignment and the learner.
    clip           fitted probabilities of following an arm are truncated below at this
                   value before they are inverted, so no weight exceeds 1 / clip.
    max_iter, learning_rate, max_depth   gradient-boosting size (early stopping is off, so
                   the fit does not depend on a random validation split).
    logistic_c     inverse ridge strength for the logistic learner.
    """

    learner: str = "hist_gradient_boosting"
    n_folds: int = 5
    seed: int = 0
    clip: float = 0.01
    max_iter: int = 100
    learning_rate: float = 0.05
    max_depth: int = 3
    logistic_c: float = 1.0

    def __post_init__(self) -> None:
        if self.learner not in LEARNERS:
            raise ValueError(f"learner must be one of {LEARNERS}, got {self.learner!r}")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if not 0.0 <= self.clip < 0.5:
            raise ValueError("clip must lie in [0, 0.5)")
        if self.max_iter < 1 or self.max_depth < 1:
            raise ValueError("max_iter and max_depth must be positive")
        if self.learning_rate <= 0 or self.logistic_c <= 0:
            raise ValueError("learning_rate and logistic_c must be positive")


@dataclass(frozen=True)
class ArmRisks:
    """Estimated risk under each arm for the whole supplied population.

    risk         (K,) risk under each arm.
    influence    (n, K) centred influence values; var(risk[k]) ~ var(influence[:, k]) / n.
    cluster      (n,) patient index used to sum influence values before taking the variance.
    n_followers  (K,) rows observed to follow each arm (for clones: uncensored at-risk rows).
    probability  (n, K) clipped cross-fitted probability of following each arm; NaN for rows
                 that were never at risk of censoring. None for estimators without one.
    weights      (n, K) weight of each row in each arm's pseudo-population: 1 / probability
                 for followers, 1 for rows whose event preceded the decision, 0 otherwise.
    """

    arms: tuple
    risk: np.ndarray
    influence: np.ndarray
    cluster: np.ndarray
    n: int
    method: str
    n_followers: np.ndarray
    probability: np.ndarray | None
    weights: np.ndarray | None


@dataclass(frozen=True)
class IntervalEstimate:
    """A point estimate with its interval. For a ratio, `se` is on the log scale."""

    estimate: float
    se: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ContrastEstimate:
    """Treated-versus-control risk difference and risk ratio with (1 - alpha) intervals."""

    treated: object
    control: object
    risk_treated: float
    risk_control: float
    risk_difference: IntervalEstimate
    risk_ratio: IntervalEstimate
    alpha: float
    n: int
    method: str
    estimand: str


@dataclass(frozen=True)
class CloneDesign:
    """Clone bookkeeping for the one-step emulation (see `build_clones`)."""

    arms: tuple
    at_risk: np.ndarray       # (n,) follow-up reached the device decision
    uncensored: np.ndarray    # (n, K) the clone in arm k is never artificially censored
    censor_time: np.ndarray   # (n, K) time of artificial censoring; inf when uncensored


@dataclass(frozen=True)
class CumulativeIncidence:
    """Cause-specific cumulative incidence and event-free survival on a common time grid."""

    times: np.ndarray
    causes: tuple[int, ...]
    incidence: dict[int, np.ndarray]
    survival: np.ndarray


@dataclass(frozen=True)
class CloneCensorResult:
    """`incidence` maps each arm to its weighted curve; None when `outcome` was overridden."""

    risks: ArmRisks
    design: CloneDesign
    incidence: dict[object, CumulativeIncidence] | None


@dataclass(frozen=True)
class NegativeControlResult:
    treated: object
    control: object
    risk_difference: IntervalEstimate
    covers_null: bool


@dataclass(frozen=True)
class EValue:
    """E-value for the point estimate and for the confidence limit nearer the null."""

    point: float
    confidence_limit: float | None


# ---------------------------------------------------------------------------
# input checks
# ---------------------------------------------------------------------------

def _design_matrix(X: np.ndarray, config: NuisanceConfig) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of covariates")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values")
    if config.learner == "logistic" and np.any(np.isnan(X)):
        raise EstimationError(
            "X has missing values, which the logistic learner cannot use; "
            "impute them or use the hist_gradient_boosting learner"
        )
    return X


def _vector(values: np.ndarray, n: int, name: str) -> np.ndarray:
    out = np.asarray(values)
    if out.shape != (n,):
        raise ValueError(f"{name} must be a 1-D array of length {n}, got shape {out.shape}")
    return out


def _binary(values: np.ndarray, n: int, name: str) -> np.ndarray:
    raw = _vector(values, n, name)
    try:
        out = raw.astype(np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be binary (0 or 1)") from None
    if not np.all((out == 0) | (out == 1)):
        raise ValueError(f"{name} must be binary (0 or 1) with no missing values")
    return out


def _arm_labels(arm: np.ndarray, arms: Sequence | None) -> tuple:
    if arms is None:
        return tuple(np.unique(arm).tolist())
    labels = tuple(arms)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("arms must be a non-empty sequence of distinct labels")
    return labels


def _clusters(groups: np.ndarray | None, n: int) -> np.ndarray:
    if groups is None:
        return np.arange(n)
    return np.unique(_vector(groups, n, "groups"), return_inverse=True)[1].reshape(n)


def _weight_vector(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n)
    w = _vector(weights, n, "weights").astype(np.float64)
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    return w


def _z(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


# ---------------------------------------------------------------------------
# cross-fitting
# ---------------------------------------------------------------------------

def patient_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign each row a fold in [0, n_folds) so that a patient's rows share one fold.

    Patients are shuffled with `numpy.random.default_rng(seed)` and dealt round-robin, so
    fold sizes differ by at most one patient. Implemented here rather than with
    `GroupKFold(shuffle=True)`, which needs scikit-learn 1.6 or later.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be a 1-D array")
    patients, inverse = np.unique(groups, return_inverse=True)
    if patients.size < n_folds:
        raise EstimationError(
            f"cross-fitting needs at least as many patients as folds "
            f"({patients.size} patients, {n_folds} folds)"
        )
    order = np.random.default_rng(seed).permutation(patients.size)
    fold_of_patient = np.empty(patients.size, dtype=np.int64)
    fold_of_patient[order] = np.arange(patients.size) % n_folds
    return fold_of_patient[inverse.reshape(groups.shape)]


def _learner(config: NuisanceConfig):
    if config.learner == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=config.logistic_c, max_iter=1000, random_state=config.seed),
        )
    return HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        early_stopping=False,
        random_state=config.seed,
    )


def _cross_fit(
    X: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    config: NuisanceConfig,
    *,
    fit_mask: np.ndarray,
    predict_mask: np.ndarray,
    what: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold class probabilities.

    Each fold's model is fitted on `fit_mask` rows of the other folds and predicts the
    `predict_mask` rows of its own. Returns (classes, probabilities of shape (n, n_classes)),
    with NaN on rows outside `predict_mask`.
    """
    classes = np.unique(labels[fit_mask])
    if classes.size < 2:
        raise EstimationError(f"{what}: single class among the rows available for fitting")
    out = np.full((X.shape[0], classes.size), np.nan)
    for fold in range(config.n_folds):
        train = fit_mask & (folds != fold)
        test = predict_mask & (folds == fold)
        if not test.any():
            continue
        if np.unique(labels[train]).size < classes.size:
            raise EstimationError(
                f"{what}: a class is absent from a training fold; "
                "use fewer folds or a larger cohort"
            )
        model = _learner(config).fit(X[train], labels[train])
        out[test] = model.predict_proba(X[test])
    return classes, out


def cross_fit_probability(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    config: NuisanceConfig | None = None,
) -> np.ndarray:
    """Cross-fitted P(y = 1 | X) for every row, unclipped. `y` is binary."""
    config = config or NuisanceConfig()
    X = _design_matrix(X, config)
    n = X.shape[0]
    y = _binary(y, n, "y").astype(np.int64)
    folds = patient_folds(_clusters(groups, n), config.n_folds, config.seed)
    everyone = np.ones(n, dtype=bool)
    _, proba = _cross_fit(X, y, folds, config, fit_mask=everyone, predict_mask=everyone, what="y")
    return proba[:, 1]


# ---------------------------------------------------------------------------
# doubly robust arm risks
# ---------------------------------------------------------------------------

def doubly_robust_scores(
    outcome: np.ndarray,
    follows: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    at_risk: np.ndarray | None = None,
) -> np.ndarray:
    """Per-row doubly robust score for one arm's risk; its mean is the estimate.

    For a row at risk of censoring (R = 1):  m + F (Y - m) / pi
    For a row whose follow-up ended before the decision (R = 0):  Y
    where F marks rows that follow the arm, pi = P(F = 1 | X, R = 1) and
    m = E[Y | X, F = 1, R = 1]. The mean is consistent when either pi or m is right.
    """
    y = np.asarray(outcome, dtype=np.float64)
    f = np.asarray(follows, dtype=bool)
    risk_set = np.ones(y.shape, dtype=bool) if at_risk is None else np.asarray(at_risk, dtype=bool)
    pi = np.where(risk_set, np.asarray(probability, dtype=np.float64), 1.0)
    m = np.where(risk_set, np.asarray(prediction, dtype=np.float64), 0.0)
    if np.any(pi[risk_set & f] <= 0):
        raise EstimationError("a row follows the arm with a fitted probability of zero")
    augmentation = np.zeros_like(y)
    used = risk_set & f
    augmentation[used] = (y[used] - m[used]) / pi[used]
    return np.where(risk_set, m + augmentation, y)


def _dr_arm_risks(
    X: np.ndarray,
    arms: tuple,
    uncensored: np.ndarray,
    outcome: np.ndarray,
    at_risk: np.ndarray,
    cluster: np.ndarray,
    config: NuisanceConfig,
    method: str,
) -> ArmRisks:
    """Shared core of the point-treatment and clone-censor-weight estimators."""
    n, k_arms = uncensored.shape
    folds = patient_folds(cluster, config.n_folds, config.seed)
    if np.unique(outcome).size < 2:
        raise EstimationError("single-class outcome: every row has the same outcome value")
    follows = uncensored & at_risk[:, None]
    for k, label in enumerate(arms):
        if not follows[:, k].any():
            raise EstimationError(f"arm {label!r} has no rows that follow it")
        if np.unique(outcome[follows[:, k]]).size < 2:
            raise EstimationError(f"single-class outcome among the rows that follow arm {label!r}")

    # One device-choice model for all compared arms: class k follows arm k, class K follows none.
    choice = np.where(follows.any(axis=1), follows.argmax(axis=1), k_arms)
    classes, choice_proba = _cross_fit(
        X, choice, folds, config, fit_mask=at_risk, predict_mask=at_risk, what="device-choice model"
    )
    probability = np.full((n, k_arms), np.nan)
    weights = np.where(at_risk[:, None], 0.0, 1.0) * np.ones((1, k_arms))
    scores = np.empty((n, k_arms))
    for k, label in enumerate(arms):
        pi = np.clip(choice_proba[:, int(np.flatnonzero(classes == k)[0])], config.clip, 1.0)
        _, outcome_proba = _cross_fit(
            X, outcome.astype(np.int64), folds, config,
            fit_mask=follows[:, k], predict_mask=at_risk, what=f"outcome model for arm {label!r}",
        )
        scores[:, k] = doubly_robust_scores(outcome, follows[:, k], pi, outcome_proba[:, 1], at_risk)
        probability[at_risk, k] = pi[at_risk]
        weights[follows[:, k], k] = 1.0 / pi[follows[:, k]]
    risk = scores.mean(axis=0)
    return ArmRisks(
        arms=arms, risk=risk, influence=scores - risk, cluster=cluster, n=n, method=method,
        n_followers=follows.sum(axis=0), probability=probability, weights=weights,
    )


def point_treatment_arm_risks(
    X: np.ndarray,
    arm: np.ndarray,
    outcome: np.ndarray,
    *,
    arms: Sequence | None = None,
    groups: np.ndarray | None = None,
    config: NuisanceConfig | None = None,
) -> ArmRisks:
    """Cross-fitted AIPW risk under each arm, with the first device as a fixed exposure.

    `arm` holds one label per row. `arms` names the compared arms (default: every label
    present, sorted); rows with any other label stay in the target population and are
    modelled as following none of the compared arms. `outcome` is binary and fully observed.
    `groups` are patient identifiers for fold assignment and clustered standard errors
    (default: one patient per row).
    """
    config = config or NuisanceConfig()
    X = _design_matrix(X, config)
    n = X.shape[0]
    arm = _vector(arm, n, "arm")
    y = _binary(outcome, n, "outcome")
    labels = _arm_labels(arm, arms)
    uncensored = np.stack([arm == label for label in labels], axis=1)
    return _dr_arm_risks(
        X, labels, uncensored, y, np.ones(n, dtype=bool), _clusters(groups, n), config, method="aipw"
    )


# ---------------------------------------------------------------------------
# clone-censor-weight
# ---------------------------------------------------------------------------

def build_clones(
    device_arm: np.ndarray,
    device_time: np.ndarray,
    event_time: np.ndarray,
    event_type: np.ndarray,
    *,
    arms: Sequence,
    grace: float,
) -> CloneDesign:
    """Clone every patient into each arm in `arms` and apply the one-step censoring rule.

    Inputs, one entry per patient, times measured from extubation (time zero):
      device_time  time of the first post-extubation device row; NaN or inf when there is none.
      device_arm   the arm that row shows (ignored when there is no row inside the window).
      event_time   time of the outcome event, or the end of follow-up when there was none.
      event_type   0 for no event, a positive integer cause otherwise (e.g. 1 reintubation,
                   2 death).

    The strategy for arm a is "start device a within `grace` of extubation". A patient has
    started when device_time <= grace. The decision time is device_time for a patient who
    started and `grace` for one who did not.

    Rule, applied in this order:
      1. Follow-up ended before the decision: event_time < device_time for a patient who
         started, event_time <= grace for one who did not. The patient had not yet deviated
         from any strategy, so the clone in EVERY arm is uncensored and carries the event.
         A device row and an event at the same instant count as the device row first.
      2. The patient started and the first device row shows arm a: the clone in arm a is
         uncensored for all of follow-up; the clone in every other arm is censored at
         device_time. A first device outside `arms` censors every clone at device_time.
      3. The patient had not started by the end of the window (no device row, or a first row
         after `grace`, which is rescue rather than assignment): every clone is censored at
         `grace`.

    Only the first device row is read: there is one decision step, and later switches do
    not censor.
    """
    labels = _arm_labels(np.asarray(device_arm), arms)
    device_arm = np.asarray(device_arm)
    if device_arm.ndim != 1:
        raise ValueError("device_arm must be a 1-D array")
    n = device_arm.shape[0]
    if not math.isfinite(grace) or grace < 0:
        raise ValueError("grace must be a finite, non-negative window length")
    device_time = _vector(device_time, n, "device_time").astype(np.float64)
    event_time = _vector(event_time, n, "event_time").astype(np.float64)
    event_type = _vector(event_type, n, "event_type")
    if np.any(device_time < 0):
        raise ValueError("device_time must not precede time zero")
    if np.any(np.isnan(event_time)) or np.any(event_time < 0):
        raise ValueError("event_time must be present and must not precede time zero")
    if not np.issubdtype(event_type.dtype, np.integer) or np.any(event_type < 0):
        raise ValueError("event_type must hold non-negative integers (0 = no event)")

    with np.errstate(invalid="ignore"):
        started = np.isfinite(device_time) & (device_time <= grace)
        ended_first = np.where(started, event_time < device_time, event_time <= grace)
    at_risk = ~ended_first
    decision_time = np.where(started, device_time, grace)
    uncensored = np.stack(
        [ended_first | (started & (device_arm == label)) for label in labels], axis=1
    )
    censor_time = np.where(uncensored, np.inf, decision_time[:, None])
    return CloneDesign(arms=labels, at_risk=at_risk, uncensored=uncensored, censor_time=censor_time)


def clone_censor_weight(
    X: np.ndarray,
    device_arm: np.ndarray,
    device_time: np.ndarray,
    event_time: np.ndarray,
    event_type: np.ndarray,
    *,
    arms: Sequence,
    grace: float,
    horizon: float,
    event_of_interest: int | None = None,
    outcome: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    config: NuisanceConfig | None = None,
) -> CloneCensorResult:
    """One-step clone-censor-weight risk under each arm by `horizon`, doubly robust.

    Clones follow `build_clones`. The censoring (device-choice) model is cross-fitted on the
    patients who reached the decision; an uncensored clone of such a patient gets weight
    1 / P(first device in the window is this arm | X), a clone whose follow-up ended before
    the decision gets weight 1, and a censored clone gets weight 0. The outcome model for an
    arm is cross-fitted on that arm's uncensored clones among patients who reached the
    decision, and the two are combined as in `doubly_robust_scores`.

    The decision is treated as a single step: its timing inside the window is not modelled,
    so a patient whose event preceded the first device row is never up-weighted.

    Outcome: any event by `horizon` (the composite), or only cause `event_of_interest`, in
    which case other causes compete and the estimate is that cause's cumulative incidence.
    Follow-up must be complete: a patient with no event and event_time < horizon is refused,
    because a binary outcome at the horizon is then undefined. `cumulative_incidence` does
    accept such rows.

    `outcome` replaces the event-derived outcome with another binary variable (a
    negative-control outcome); clones and weights are unchanged and no curve is returned.

    With grace = 0 and every first device row at time zero this equals
    `point_treatment_arm_risks` on the same rows.
    """
    config = config or NuisanceConfig()
    X = _design_matrix(X, config)
    n = X.shape[0]
    if not math.isfinite(horizon) or horizon <= 0 or horizon < grace:
        raise ValueError("horizon must be finite, positive and no shorter than the grace window")
    design = build_clones(
        _vector(device_arm, n, "device_arm"), device_time, event_time, event_type, arms=arms, grace=grace
    )
    event_time = np.asarray(event_time, dtype=np.float64)
    event_type = np.asarray(event_type)
    if outcome is None:
        if np.any((event_type == 0) & (event_time < horizon)):
            raise EstimationError(
                "incomplete follow-up: some patients have no event and leave before the horizon; "
                "resolve their status at the horizon before the doubly robust step"
            )
        if event_of_interest is None:
            happened = event_type != 0
        else:
            happened = event_type == event_of_interest
        y = (happened & (event_time <= horizon)).astype(np.float64)
    else:
        y = _binary(outcome, n, "outcome")
    risks = _dr_arm_risks(
        X, design.arms, design.uncensored, y, design.at_risk, _clusters(groups, n), config,
        method="clone_censor_weight",
    )
    incidence = None
    if outcome is None:
        incidence = {
            label: cumulative_incidence(event_time, event_type, weights=risks.weights[:, k], horizon=horizon)
            for k, label in enumerate(design.arms)
        }
    return CloneCensorResult(risks=risks, design=design, incidence=incidence)


# ---------------------------------------------------------------------------
# contrasts
# ---------------------------------------------------------------------------

def _clustered_se(values: np.ndarray, cluster: np.ndarray, n: int) -> float:
    """Standard error of a mean of n centred influence values, summed within patient."""
    sums = np.bincount(cluster, weights=values)
    g = sums.size
    if g < 2:
        raise EstimationError("a standard error needs at least two patients")
    return float(math.sqrt(g / (g - 1.0) * np.sum(sums ** 2)) / n)


def _difference(
    risk_t: float, risk_c: float, if_t: np.ndarray, if_c: np.ndarray, cluster: np.ndarray, alpha: float
) -> IntervalEstimate:
    z = _z(alpha)
    estimate = risk_t - risk_c
    se = _clustered_se(if_t - if_c, cluster, if_t.size)
    return IntervalEstimate(float(estimate), se, float(estimate - z * se), float(estimate + z * se))


def _ratio(
    risk_t: float, risk_c: float, if_t: np.ndarray, if_c: np.ndarray, cluster: np.ndarray, alpha: float
) -> IntervalEstimate:
    if not (risk_t > 0 and risk_c > 0):
        raise EstimationError("the risk ratio is undefined: an arm's estimated risk is not positive")
    z = _z(alpha)
    log_ratio = math.log(risk_t / risk_c)
    se = _clustered_se(if_t / risk_t - if_c / risk_c, cluster, if_t.size)
    return IntervalEstimate(
        float(math.exp(log_ratio)), se, float(math.exp(log_ratio - z * se)), float(math.exp(log_ratio + z * se))
    )


def _arm_index(risks: ArmRisks, treated: object, control: object) -> tuple[int, int]:
    if treated == control:
        raise ValueError("treated and control arms must differ")
    for label in (treated, control):
        if label not in risks.arms:
            raise KeyError(f"arm {label!r} was not estimated; available arms: {risks.arms}")
    return risks.arms.index(treated), risks.arms.index(control)


def contrast(risks: ArmRisks, treated: object, control: object, *, alpha: float = 0.05) -> ContrastEstimate:
    """Risk difference and risk ratio, treated minus (over) control, from `ArmRisks`.

    Intervals are Wald intervals from the per-patient influence values; the ratio's interval
    is built on the log scale by the delta method. Contrasts from one `ArmRisks` are
    consistent with each other: RD(a, c) = RD(a, b) + RD(b, c).
    """
    t, c = _arm_index(risks, treated, control)
    args = (float(risks.risk[t]), float(risks.risk[c]), risks.influence[:, t], risks.influence[:, c],
            risks.cluster, alpha)
    return ContrastEstimate(
        treated=treated, control=control, risk_treated=args[0], risk_control=args[1],
        risk_difference=_difference(*args), risk_ratio=_ratio(*args),
        alpha=alpha, n=risks.n, method=risks.method, estimand="ATE",
    )


def negative_control_check(
    risks: ArmRisks, treated: object, control: object, *, alpha: float = 0.05
) -> NegativeControlResult:
    """Read a negative-control run: does the risk-difference interval cover zero?

    Run the same estimator on an outcome the device cannot affect, either
    `point_treatment_arm_risks(X, arm, negative_control_outcome, ...)` or
    `clone_censor_weight(..., outcome=negative_control_outcome)`, and pass the result here.
    Only the difference is read, so a rare control outcome cannot fail on the ratio.
    """
    t, c = _arm_index(risks, treated, control)
    difference = _difference(
        float(risks.risk[t]), float(risks.risk[c]), risks.influence[:, t], risks.influence[:, c],
        risks.cluster, alpha,
    )
    return NegativeControlResult(
        treated=treated, control=control, risk_difference=difference,
        covers_null=bool(difference.lower <= 0.0 <= difference.upper),
    )


# ---------------------------------------------------------------------------
# weighting estimators for the point-treatment sensitivity analysis
# ---------------------------------------------------------------------------

def _propensity(propensity: np.ndarray, treated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(propensity, dtype=np.float64)
    t = np.asarray(treated, dtype=bool)
    if e.ndim != 1 or e.shape != t.shape:
        raise ValueError("propensity and treated must be 1-D arrays of the same length")
    if not np.all(np.isfinite(e)) or np.any(e < 0) or np.any(e > 1):
        raise ValueError("propensity must be finite and inside [0, 1]")
    return e, t


def inverse_probability_weights(propensity: np.ndarray, treated: np.ndarray, *, clip: float = 0.0) -> np.ndarray:
    """1 / e for treated rows and 1 / (1 - e) for the rest, with e truncated to [clip, 1 - clip]."""
    e, t = _propensity(propensity, treated)
    if not 0.0 <= clip < 0.5:
        raise ValueError("clip must lie in [0, 0.5)")
    e = np.clip(e, clip, 1.0 - clip)
    denominator = np.where(t, e, 1.0 - e)
    if np.any(denominator <= 0):
        raise EstimationError("a row has probability zero of the arm it is in; set clip above zero")
    return 1.0 / denominator


def overlap_weights(propensity: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """Overlap weights (Li, Morgan & Zaslavsky 2018): 1 - e for treated rows, e for the rest.

    Bounded by one, so no truncation is needed; they target the population in which the two
    arms overlap (the ATO).
    """
    e, t = _propensity(propensity, treated)
    return np.where(t, 1.0 - e, e)


def weighted_contrast(
    outcome: np.ndarray,
    treated: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    groups: np.ndarray | None = None,
    alpha: float = 0.05,
    method: str | None = None,
    estimand: str = "",
) -> ContrastEstimate:
    """Hajek (normalized) weighted two-arm contrast; unweighted it is the crude contrast.

    The interval uses the linearized influence values of the two weighted means and treats
    the weights as fixed.
    """
    t = np.asarray(treated, dtype=bool)
    if t.ndim != 1:
        raise ValueError("treated must be a 1-D array")
    n = t.shape[0]
    y = _binary(outcome, n, "outcome")
    w = _weight_vector(weights, n)
    cluster = _clusters(groups, n)
    risk, influence = [], []
    for mask, name in ((t, "treated"), (~t, "control")):
        total = w[mask].sum()
        if total <= 0:
            raise EstimationError(f"the {name} arm has no rows with positive weight")
        mean = float((w[mask] * y[mask]).sum() / total)
        risk.append(mean)
        influence.append(n * np.where(mask, w * (y - mean), 0.0) / total)
    args = (risk[0], risk[1], influence[0], influence[1], cluster, alpha)
    return ContrastEstimate(
        treated=True, control=False, risk_treated=risk[0], risk_control=risk[1],
        risk_difference=_difference(*args), risk_ratio=_ratio(*args), alpha=alpha, n=n,
        method=method or ("unweighted" if weights is None else "weighted"), estimand=estimand,
    )


def overlap_weighted_contrast(
    X: np.ndarray,
    arm: np.ndarray,
    outcome: np.ndarray,
    *,
    treated: object,
    control: object,
    groups: np.ndarray | None = None,
    config: NuisanceConfig | None = None,
    alpha: float = 0.05,
) -> ContrastEstimate:
    """Overlap-weighted contrast between two arms, restricted to rows in either arm.

    The propensity of `treated` versus `control` is cross-fitted on those rows only, so the
    estimand is the effect in their overlap population (ATO), not the whole cohort. `n` in
    the result is the number of rows in the two arms.
    """
    config = config or NuisanceConfig()
    X = _design_matrix(X, config)
    n = X.shape[0]
    arm = _vector(arm, n, "arm")
    y = _binary(outcome, n, "outcome")
    if treated == control:
        raise ValueError("treated and control arms must differ")
    for label in (treated, control):
        if not np.any(arm == label):
            raise EstimationError(f"arm {label!r} has no rows")
    pair = (arm == treated) | (arm == control)
    is_treated = arm[pair] == treated
    pair_groups = None if groups is None else _vector(groups, n, "groups")[pair]
    propensity = cross_fit_probability(X[pair], is_treated, groups=pair_groups, config=config)
    result = weighted_contrast(
        y[pair], is_treated, overlap_weights(propensity, is_treated),
        groups=pair_groups, alpha=alpha, method="overlap_weights", estimand="ATO",
    )
    return ContrastEstimate(
        treated=treated, control=control, risk_treated=result.risk_treated,
        risk_control=result.risk_control, risk_difference=result.risk_difference,
        risk_ratio=result.risk_ratio, alpha=alpha, n=result.n, method=result.method, estimand="ATO",
    )


# ---------------------------------------------------------------------------
# cumulative incidence with competing events
# ---------------------------------------------------------------------------

def cumulative_incidence(
    time: np.ndarray,
    event_type: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    horizon: float | None = None,
    causes: Sequence[int] | None = None,
) -> CumulativeIncidence:
    """Weighted Aalen-Johansen cumulative incidence on the grid of observed event times.

    `event_type` is 0 for a censored row and a positive integer cause otherwise. At each
    distinct event time t, with weighted counts d_k(t) of cause-k events and n(t) of rows
    still under observation (time >= t; a row censored at t is still at risk at t):
        incidence_k(t) = incidence_k(t-) + S(t-) d_k(t) / n(t)
        S(t)           = S(t-) (1 - sum_k d_k(t) / n(t))
    so the cause-specific incidences and event-free survival S sum to one at every grid
    point. Rows with weight zero are ignored. With `horizon`, events after it are dropped
    and the grid ends at the horizon. `causes` defaults to the causes present.
    """
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("time must be a non-empty 1-D array")
    n = time.shape[0]
    event_type = _vector(event_type, n, "event_type")
    if np.any(np.isnan(time)) or np.any(time < 0):
        raise ValueError("time must be present and non-negative")
    if not np.issubdtype(event_type.dtype, np.integer) or np.any(event_type < 0):
        raise ValueError("event_type must hold non-negative integers (0 = censored)")
    w = _weight_vector(weights, n)
    if w.sum() <= 0:
        raise ValueError("weights sum to zero")
    if horizon is not None and (not math.isfinite(horizon) or horizon <= 0):
        raise ValueError("horizon must be finite and positive")
    if causes is None:
        cause_labels = tuple(int(c) for c in np.unique(event_type[event_type > 0]))
    else:
        cause_labels = tuple(int(c) for c in causes)
        if any(c <= 0 for c in cause_labels) or len(set(cause_labels)) != len(cause_labels):
            raise ValueError("causes must be distinct positive integers")
        if np.any((event_type > 0) & ~np.isin(event_type, cause_labels)):
            raise ValueError("event_type holds a cause that is not listed in causes")

    grid, position = np.unique(time, return_inverse=True)
    position = position.reshape(n)
    at_risk = np.cumsum(np.bincount(position, weights=w, minlength=grid.size)[::-1])[::-1]
    safe = np.where(at_risk > 0, at_risk, 1.0)
    hazards = np.stack(
        [np.bincount(position, weights=w * (event_type == c), minlength=grid.size) / safe for c in cause_labels]
    ) if cause_labels else np.zeros((0, grid.size))
    survival = np.cumprod(1.0 - hazards.sum(axis=0))
    survival_before = np.concatenate([[1.0], survival[:-1]])
    incidence = np.cumsum(survival_before * hazards, axis=1)

    keep = hazards.sum(axis=0) > 0
    if horizon is not None:
        keep &= grid <= horizon
    times, survival, incidence = grid[keep], survival[keep], incidence[:, keep]
    if horizon is not None and (times.size == 0 or times[-1] < horizon):
        times = np.append(times, horizon)
        survival = np.append(survival, survival[-1] if survival.size else 1.0)
        last = incidence[:, -1:] if incidence.shape[1] else np.zeros((len(cause_labels), 1))
        incidence = np.concatenate([incidence, last], axis=1)
    return CumulativeIncidence(
        times=times, causes=cause_labels,
        incidence={c: incidence[i] for i, c in enumerate(cause_labels)}, survival=survival,
    )


# ---------------------------------------------------------------------------
# quantitative bias analysis
# ---------------------------------------------------------------------------

def _e_value(ratio: float) -> float:
    ratio = 1.0 / ratio if ratio < 1.0 else ratio
    return float(ratio + math.sqrt(ratio * (ratio - 1.0)))


def e_value(risk_ratio: float, *, lower: float | None = None, upper: float | None = None) -> EValue:
    """E-value for a risk ratio (VanderWeele & Ding 2017).

    The minimum risk ratio an unmeasured confounder would need with both the device and the
    outcome to explain the estimate away: RR + sqrt(RR (RR - 1)), after inverting a ratio
    below one. For the interval, the same formula is applied to the confidence limit nearer
    the null, and the value is 1 when the interval already includes the null.
    """
    if not math.isfinite(risk_ratio) or risk_ratio <= 0:
        raise ValueError("risk_ratio must be positive and finite")
    if (lower is None) != (upper is None):
        raise ValueError("pass both confidence limits or neither")
    limit = None
    if lower is not None and upper is not None:
        if not (0 < lower <= risk_ratio <= upper and math.isfinite(upper)):
            raise ValueError("confidence limits must be positive and bracket the risk ratio")
        if lower <= 1.0 <= upper:
            limit = 1.0
        else:
            limit = _e_value(lower if risk_ratio > 1.0 else upper)
    return EValue(point=_e_value(risk_ratio), confidence_limit=limit)
