"""Design diagnostics for the extubation emulation (plan U11; requirement R9).

Overlap/positivity, covariate balance, effective sample size and a minimal detectable effect.
None of these functions takes an outcome by arm, so the outcome-blind audit stage can call
all of them before any effect estimate is read. Arrays in, plain dataclasses and floats out.

Standard library, numpy and scikit-learn only: this module is vendored into `clif-validate`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np

DEFAULT_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


@dataclass(frozen=True)
class OverlapSummary:
    """Distribution of one arm's propensity among rows in and out of that arm.

    `support` is the common-support region used for the shares: the caller's bounds, or the
    empirical region [max of the two group minima, min of the two group maxima].
    """

    quantiles: tuple[float, ...]
    quantiles_in_arm: tuple[float, ...]
    quantiles_out_of_arm: tuple[float, ...]
    support: tuple[float, float]
    share_outside: float
    share_outside_in_arm: float
    share_outside_out_of_arm: float
    n_in_arm: int
    n_out_of_arm: int


@dataclass(frozen=True)
class BalanceTable:
    """Standardized mean differences per covariate, before and after weighting."""

    names: tuple[str, ...]
    smd_unweighted: np.ndarray
    smd_weighted: np.ndarray
    max_abs_unweighted: float
    max_abs_weighted: float


def _weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty 1-D array")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    if w.sum() <= 0:
        raise ValueError("weights sum to zero")
    return w


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size, (sum w)^2 / sum w^2. Invariant to rescaling the weights."""
    w = _weights(weights)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def overlap_summary(
    propensity: np.ndarray,
    in_arm: np.ndarray,
    *,
    support: tuple[float, float] | None = None,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> OverlapSummary:
    """Positivity summary for one arm.

    `propensity` is the probability of being in the arm and `in_arm` marks the rows that are.
    Rows whose propensity lies outside `support` (bounds inclusive) are counted as outside
    common support. With `support=None` the region is read from the data, so a fixed region
    such as (0.05, 0.95) is the one to pre-register.
    """
    p = np.asarray(propensity, dtype=np.float64)
    mask = np.asarray(in_arm, dtype=bool)
    if p.ndim != 1 or p.shape != mask.shape:
        raise ValueError("propensity and in_arm must be 1-D arrays of the same length")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("propensity must be finite and inside [0, 1]")
    if not mask.any() or mask.all():
        raise ValueError("overlap needs rows both in and out of the arm")
    q = tuple(float(v) for v in quantiles)
    if any(not 0.0 <= v <= 1.0 for v in q):
        raise ValueError("quantiles must lie in [0, 1]")
    inside_arm, outside_arm = p[mask], p[~mask]
    if support is None:
        lo = float(max(inside_arm.min(), outside_arm.min()))
        hi = float(min(inside_arm.max(), outside_arm.max()))
    else:
        lo, hi = float(support[0]), float(support[1])
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("support must satisfy 0 <= lower < upper <= 1")
    outside = (p < lo) | (p > hi)
    return OverlapSummary(
        quantiles=q,
        quantiles_in_arm=tuple(float(v) for v in np.quantile(inside_arm, q)),
        quantiles_out_of_arm=tuple(float(v) for v in np.quantile(outside_arm, q)),
        support=(lo, hi),
        share_outside=float(outside.mean()),
        share_outside_in_arm=float(outside[mask].mean()),
        share_outside_out_of_arm=float(outside[~mask].mean()),
        n_in_arm=int(mask.sum()),
        n_out_of_arm=int((~mask).sum()),
    )


def _weighted_column_means(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-column weighted mean over the rows where the column is observed."""
    observed = ~np.isnan(X)
    total = (observed * w[:, None]).sum(axis=0)
    if np.any(total <= 0):
        raise ValueError("a covariate has no observed, positively weighted row in one group")
    return (np.where(observed, X, 0.0) * w[:, None]).sum(axis=0) / total


def _column_variances(X: np.ndarray) -> np.ndarray:
    """Per-column sample variance (ddof=1) over observed rows; 0 with fewer than two."""
    observed = ~np.isnan(X)
    count = observed.sum(axis=0)
    filled = np.where(observed, X, 0.0)
    mean = filled.sum(axis=0) / np.maximum(count, 1)
    squares = (np.where(observed, X - mean, 0.0) ** 2).sum(axis=0)
    return np.where(count > 1, squares / np.maximum(count - 1, 1), 0.0)


def covariate_balance(
    X: np.ndarray,
    treated: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    names: Sequence[str] | None = None,
) -> BalanceTable:
    """Standardized mean differences between `treated` rows and the rest.

    SMD = (mean_treated - mean_other) / sqrt((var_treated + var_other) / 2). Both the weighted
    and the unweighted difference are divided by the same unweighted pooled standard
    deviation, so weighting cannot shrink an SMD by inflating the variance. Missing covariate
    values are skipped per column. A column that is constant in both groups gets 0 when the
    group means agree and a signed infinity when they do not (perfect separation).

    To compare weighted clones with the cohort they should represent, stack the two sets of
    rows and pass weights of one for the cohort copy.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    mask = np.asarray(treated, dtype=bool)
    if mask.shape != (X.shape[0],):
        raise ValueError("treated must have one entry per row of X")
    if not mask.any() or mask.all():
        raise ValueError("balance needs rows in both groups")
    w = np.ones(X.shape[0]) if weights is None else _weights(weights)
    if w.shape != (X.shape[0],):
        raise ValueError("weights must have one entry per row of X")
    if names is None:
        names = tuple(f"x{j}" for j in range(X.shape[1]))
    elif len(names) != X.shape[1]:
        raise ValueError("names must have one entry per column of X")

    pooled_sd = np.sqrt((_column_variances(X[mask]) + _column_variances(X[~mask])) / 2.0)

    def smd(row_weights: np.ndarray) -> np.ndarray:
        diff = _weighted_column_means(X[mask], row_weights[mask]) - _weighted_column_means(
            X[~mask], row_weights[~mask]
        )
        out = np.zeros_like(diff)
        scaled = pooled_sd > 0
        out[scaled] = diff[scaled] / pooled_sd[scaled]
        separated = ~scaled & (diff != 0)
        out[separated] = np.sign(diff[separated]) * np.inf
        return out

    unweighted, weighted = smd(np.ones(X.shape[0])), smd(w)
    return BalanceTable(
        names=tuple(str(v) for v in names),
        smd_unweighted=unweighted,
        smd_weighted=weighted,
        max_abs_unweighted=float(np.abs(unweighted).max()) if unweighted.size else 0.0,
        max_abs_weighted=float(np.abs(weighted).max()) if weighted.size else 0.0,
    )


def minimal_detectable_effect(
    n_treated: float,
    n_control: float,
    baseline_risk: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest absolute two-arm risk difference detectable at two-sided `alpha` with `power`.

    Normal approximation with the baseline risk p used for the variance in both arms:
    (z_{1-alpha/2} + z_{power}) * sqrt(p (1 - p) (1/n_treated + 1/n_control)).
    The arm sizes may be effective sample sizes. The baseline risk is a number supplied by the
    caller (a published control-arm risk, for instance), so no outcome by arm is read.
    """
    if not (math.isfinite(n_treated) and math.isfinite(n_control)) or n_treated <= 0 or n_control <= 0:
        raise ValueError("arm sizes must be positive")
    if not 0.0 < baseline_risk < 1.0:
        raise ValueError("baseline_risk must lie strictly between 0 and 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie strictly between 0 and 1")
    normal = NormalDist()
    z = normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(power)
    se = math.sqrt(baseline_risk * (1.0 - baseline_risk) * (1.0 / n_treated + 1.0 / n_control))
    return float(z * se)
