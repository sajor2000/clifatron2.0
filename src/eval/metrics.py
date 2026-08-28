"""State-of-the-art evaluation panel for CLIF-federated ICU prediction (TRIPOD+AI).

One place for every metric reviewers now expect at a Nature-Medicine-caliber venue:

  discrimination   auroc, auprc (report BOTH — ICU outcomes are imbalanced; HiRID/YAIB convention)
  calibration      ece, brier, calibration slope + intercept-in-the-large, ICI
  recalibration    temperature_scale (Cadence single-scalar T)
  clinical utility net_benefit / decision-curve analysis (Vickers) — run AFTER recalibration
  label efficiency local_patient_equivalence (ICareFM LPE)
  fairness         subgroup_panel over sex / race / age / site (aggregate only — ICareFM precedent)

Pure numpy + sklearn; torch only inside temperature_scale. Everything is probability-in,
scalar-out so it composes across the 3x3 internal matrix and the external CLIF federation.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic sigmoid."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _irls_logistic(X: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None,
                   iters: int = 50) -> np.ndarray:
    """Newton/IRLS MLE for logistic regression (unregularized). Returns coefficients.
    `offset` fixes a known linear term (used for calibration-in-the-large)."""
    n, k = X.shape
    beta = np.zeros(k)
    off = np.zeros(n) if offset is None else offset
    for _ in range(iters):
        eta = X @ beta + off
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        w = np.clip(mu * (1 - mu), _EPS, None)
        z = eta - off + (y - mu) / w
        XtW = X.T * w
        try:
            beta_new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(k), XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


def score(p: np.ndarray, y: np.ndarray) -> dict:
    """Discrimination. Returns nan for a degenerate (single-class) label vector."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "n": int(len(y)),
                "prevalence": float(y.mean()) if len(y) else float("nan")}
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "n": int(len(y)), "prevalence": float(y.mean())}


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum():
            ece += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(brier_score_loss(y, p))


def calibration_slope_intercept(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Cox calibration: fit y ~ a + b*logit(p). Slope b (1=ideal; <1 = overfit/too extreme).
    Intercept-in-the-large = intercept of y ~ 1 + offset(logit(p)) (0 = ideal, calibration-in-large)."""
    lp = _logit(p)
    X2 = np.column_stack([np.ones_like(lp), lp])
    a, b = _irls_logistic(X2, y.astype(float))
    citl = _irls_logistic(np.ones((len(lp), 1)), y.astype(float), offset=lp)[0]
    return float(b), float(citl)


def integrated_calibration_index(p: np.ndarray, y: np.ndarray) -> float:
    """ICI (Austin & Steyerberg): mean |p - calibrated(p)|. Isotonic stand-in for loess —
    monotone, dependency-light, and robust for a fixed calibration curve."""
    iso = IsotonicRegression(out_of_bounds="clip")
    cal = iso.fit_transform(p, y.astype(float))
    return float(np.mean(np.abs(p - cal)))


def net_benefit(p: np.ndarray, y: np.ndarray, thresholds=None) -> dict:
    """Decision-curve analysis (Vickers). NB(pt) = TP/N - FP/N * (pt/(1-pt)).
    Report model vs treat-all vs treat-none. Assumes p is CALIBRATED (run after temperature)."""
    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.01)
    y = np.asarray(y).astype(int)
    N = len(y)
    prev = y.mean()
    model, treat_all = [], []
    for pt in thresholds:
        pred = p >= pt
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        w = pt / (1 - pt)
        model.append(tp / N - fp / N * w)
        treat_all.append(prev - (1 - prev) * w)
    return {"thresholds": np.asarray(thresholds), "model": np.asarray(model),
            "treat_all": np.asarray(treat_all), "treat_none": np.zeros_like(thresholds)}


def _sanitize_logits(logits: np.ndarray) -> np.ndarray:
    """Replace non-finite logits with a large finite magnitude so downstream
    torch/sklearn ops never see inf/NaN. A real model can emit saturated logits
    (p==0 or p==1); clamp them rather than crash the whole panel."""
    z = np.asarray(logits, dtype=np.float64)
    # |logit| of the clipped-prob bound _EPS is our natural finite ceiling.
    bound = float(np.log((1 - _EPS) / _EPS))
    return np.nan_to_num(z, nan=0.0, posinf=bound, neginf=-bound)


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """Cadence single-scalar temperature T* minimizing NLL (LBFGS). Divide logits by T*."""
    import torch
    z = torch.tensor(_sanitize_logits(logits), dtype=torch.float64)
    y = torch.tensor(np.asarray(labels), dtype=torch.float64)
    T = torch.ones(1, requires_grad=True, dtype=torch.float64)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / T.clamp_min(1e-3), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp_min(1e-3))


def local_patient_equivalence(fm_auroc: float, X, y,
                              sizes=(250, 500, 1000, 2000, 4000, 8000, 16000)) -> int:
    """ICareFM LPE: smallest local-LightGBM training size whose AUROC matches the FM.
    Returns crossing size, or -1 if never reached within `sizes`."""
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return -1
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    for n in sizes:
        if n > len(Xtr):
            break
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=64, verbose=-1)
        clf.fit(Xtr[:n], ytr[:n])
        if roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]) >= fm_auroc:
            return int(n)
    return -1


def full_panel(p: np.ndarray, y: np.ndarray, logits: np.ndarray | None = None,
               recalibrate: bool = True) -> dict:
    """Every scalar metric for one (task, site) cell. If `logits` given and recalibrate,
    temperature-scale first so calibration + DCA reflect the deployable model."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    T = 1.0
    if recalibrate and logits is not None and len(np.unique(y)) > 1:
        T = temperature_scale(logits, y)
        p = _sigmoid(_sanitize_logits(logits) / T)
    p = np.clip(np.nan_to_num(p, nan=0.5), 0.0, 1.0)
    out = score(p, y)
    if not np.isnan(out["auroc"]):
        slope, citl = calibration_slope_intercept(p, y)
        out.update({
            "ece": expected_calibration_error(p, y),
            "brier": brier(p, y),
            "calib_slope": slope,
            "calib_intercept": citl,
            "ici": integrated_calibration_index(p, y),
            "temperature": float(T),
        })
    return out


def subgroup_panel(p: np.ndarray, y: np.ndarray, groups: dict[str, np.ndarray]) -> dict:
    """Per-subgroup metrics for TRIPOD+AI fairness reporting (aggregate only).
    `groups` maps attribute name -> per-example category array (e.g. sex, race, age_band)."""
    p, y = np.asarray(p, float), np.asarray(y).astype(int)
    result = {}
    for attr, vals in groups.items():
        vals = np.asarray(vals)
        result[attr] = {}
        for cat in np.unique(vals):
            m = vals == cat
            if m.sum() >= 30:
                cell = score(p[m], y[m])
                if not np.isnan(cell["auroc"]):
                    cell["ece"] = expected_calibration_error(p[m], y[m])
                result[attr][str(cat)] = cell
    return result


# -------------------------------------------------------------------- competing-risk calib
def cr_d_calibration(cif: np.ndarray, events: np.ndarray,
                     event_times: np.ndarray, n_bins: int = 10) -> dict:
    """D-calibration for competing-risks models (arXiv:2602.00194).

    cif:    [N, K, H]  cause-specific CIF evaluated at observed event times
    events: [N]        event type (0..K-1)
    event_times: [N]   observed event time bin indices (0..H-1)
    n_bins: number of bins for the uniformity test in [0,1]

    Returns D-calib p-value (chi-squared) and per-bin histogram."""
    K = cif.shape[1]
    probs = cif[np.arange(len(events)), events, np.clip(event_times, 0, cif.shape[2] - 1)]
    probs = probs[np.isfinite(probs)]
    if len(probs) < n_bins:
        return {"d_calib_p": float("nan"), "d_calib_bins": [], "n": 0}

    edges = np.linspace(0, 1, n_bins + 1)
    hist, _ = np.histogram(probs, bins=edges)
    expected = np.full(n_bins, len(probs) / n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.sum((hist - expected) ** 2 / expected)
    from scipy.stats import chi2 as chi2_dist
    p = float(chi2_dist.sf(chi2, n_bins - 1))
    return {
        "d_calib_p": p,
        "d_calib_bins": hist.tolist(),
        "d_calib_chi2": float(chi2),
        "n": int(len(probs)),
    }


def aj_k_calibration(cif: np.ndarray, events: np.ndarray,
                     event_times: np.ndarray, time_horizon: int) -> dict:
    """Aalen-Johansen K-calibration: for each cause k, group predicted CIF into
    deciles and compare mean predicted vs observed. Returns per-cause slope."""
    K = cif.shape[1]
    results = {}
    for k in range(K):
        mask = events == k
        if mask.sum() < 20:
            continue
        p = cif[mask, k, event_times[mask]]
        y = np.ones_like(p)
        finite = np.isfinite(p)
        p, y = p[finite], y[finite]
        if len(np.unique(p)) < 3:
            continue
        dec = np.percentile(p, np.linspace(10, 100, 10))
        preds, obss = [], []
        for i in range(10):
            lo = dec[i - 1] if i > 0 else 0
            hi = dec[i]
            in_bucket = (p > lo) & (p <= hi)
            if in_bucket.sum() >= 5:
                preds.append(p[in_bucket].mean())
                obss.append(y[in_bucket].mean())
        if len(preds) < 3:
            continue
        slope, _ = np.polyfit(preds, obss, 1) if len(preds) >= 2 else (float("nan"), float("nan"))
        results[k] = {
            "aj_k_slope": float(slope),
            "n_deciles": len(preds),
            "n": int(mask.sum()),
        }
    return results
