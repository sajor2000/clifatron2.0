"""Classical extubation estimators and diagnostics (plan U11, KTD7/KTD8).

Every array here is simulated. Correctness is carried by planted-effect tests: the data
generator fixes an additive risk difference per arm under measured confounding, so the
crude contrast is biased by construction and a correct estimator must recover the planted
value.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from src.eval.causal import diagnostics as dx
from src.eval.causal import estimators as est

ARMS = ("cot", "hfnc", "niv")
EFFECT = {"cot": 0.0, "hfnc": -0.04, "niv": -0.10}
LOG_ODDS_SHIFT = {"cot": 0.0, "hfnc": -0.25, "niv": -0.6}
LOGISTIC = est.NuisanceConfig(learner="logistic", n_folds=3, seed=0)
PACKAGE_DIR = Path(est.__file__).resolve().parent


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def simulate(
    n: int, seed: int, *, confounding: float = 1.2, outcome: str = "smooth", choice: str = "linear"
) -> dict[str, np.ndarray]:
    """Three arms, sicker patients steered to HFNC and NIV, a planted effect per arm.

    outcome  "smooth": additive effects on a smooth baseline risk (the default);
             "logistic": each arm's risk is exactly logistic in the covariates;
             "bump": additive effects on a baseline that a linear-logit model cannot fit.
    choice   "linear": device choice is multinomial-logit in the covariates;
             "bump": device choice follows the same bump, which a linear-logit model cannot fit.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    severity = X[:, 0] + 0.5 * X[:, 1]
    bump = (np.abs(severity - 1.0) < 0.8).astype(float)
    drive = severity if choice == "linear" else 1.25 * (2.0 * bump - 1.0)
    logits = np.stack([np.zeros(n), 0.6 * confounding * drive, confounding * drive], axis=1)
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    arm_idx = (rng.random(n)[:, None] > prob.cumsum(axis=1)).sum(axis=1).clip(0, 2)
    if outcome == "logistic":
        risk = np.stack([_sigmoid(-1.0 + 1.2 * severity + LOG_ODDS_SHIFT[a]) for a in ARMS], axis=1)
    else:
        base = 0.15 + (0.5 * _sigmoid(1.5 * severity) if outcome == "smooth" else 0.6 * bump)
        risk = np.stack([base + EFFECT[a] for a in ARMS], axis=1)
    draw = rng.random(n)
    potential = draw[:, None] < risk
    return {
        "X": X,
        "arm": np.asarray(ARMS)[arm_idx],
        "outcome": potential[np.arange(n), arm_idx].astype(int),
        "potential": potential,
        "risk": risk,
        "base": risk[:, 0],
        "propensity": prob,
        "negative_control": (rng.random(n) < risk[:, 0]).astype(int),
    }


def simulate_ccw(
    n: int, seed: int, *, grace: float = 6.0, horizon: float = 168.0, late_rate: float = 0.3, **scenario
) -> dict:
    """Device decision inside a grace window, early events before it, late (rescue) starts."""
    sim = simulate(n, seed, **scenario)
    rng = np.random.default_rng(seed + 10_000)
    severity = sim["X"][:, 0] + 0.5 * sim["X"][:, 1]
    late = rng.random(n) < late_rate * _sigmoid(-severity)
    device_time = np.where(late, rng.uniform(grace + 1.0, grace + 20.0, n), rng.uniform(0.5, grace - 1.0, n))
    early = rng.random(n) < 0.03 + 0.05 * _sigmoid(severity)
    event = np.where(early, True, sim["outcome"].astype(bool))
    event_time = np.where(
        early,
        rng.uniform(0.0, 0.99, n) * np.minimum(device_time, grace),
        np.where(event, np.maximum(device_time, grace) + rng.uniform(0.5, horizon - 30.0, n), horizon),
    )
    event_type = np.where(event, np.where(rng.random(n) < 0.3, 2, 1), 0)
    truth = {a: float(np.where(early, 1.0, sim["risk"][:, k]).mean()) for k, a in enumerate(ARMS)}
    return {**sim, "device_arm": sim["arm"], "device_time": device_time, "event_time": event_time,
            "event_type": event_type, "early": early, "late": late, "grace": grace,
            "horizon": horizon, "truth": truth}


def _crude(sim: dict, treated: str, control: str) -> est.ContrastEstimate:
    pair = np.isin(sim["arm"], [treated, control])
    return est.weighted_contrast(sim["outcome"][pair], sim["arm"][pair] == treated)


# ---------------------------------------------------------------------------
# planted effect: point treatment
# ---------------------------------------------------------------------------

def test_crude_contrast_is_biased_and_aipw_recovers_planted_difference_across_seeds():
    planted = EFFECT["niv"] - EFFECT["cot"]
    crude, adjusted = [], []
    for seed in range(5):
        sim = simulate(6000, seed)
        risks = est.point_treatment_arm_risks(sim["X"], sim["arm"], sim["outcome"], config=LOGISTIC)
        result = est.contrast(risks, "niv", "cot")
        crude.append(_crude(sim, "niv", "cot").risk_difference.estimate)
        adjusted.append(result.risk_difference.estimate)
        assert abs(result.risk_difference.estimate - planted) < 0.05
        assert result.risk_difference.lower < result.risk_difference.estimate < result.risk_difference.upper
    assert min(crude) - planted > 0.08, "confounding must make the crude contrast wrong"
    assert abs(np.mean(adjusted) - planted) < 0.02


def test_gradient_boosting_nuisances_recover_planted_difference():
    sim = simulate(3000, 11)
    risks = est.point_treatment_arm_risks(
        sim["X"], sim["arm"], sim["outcome"], config=est.NuisanceConfig(n_folds=3, seed=0, max_iter=60)
    )
    result = est.contrast(risks, "niv", "cot")
    assert risks.method == "aipw"
    assert abs(result.risk_difference.estimate - (EFFECT["niv"] - EFFECT["cot"])) < 0.05
    truth_ratio = sim["potential"][:, 2].mean() / sim["potential"][:, 0].mean()
    assert result.risk_ratio.lower < truth_ratio < result.risk_ratio.upper


def test_gradient_boosting_accepts_missing_covariates_and_logistic_refuses_them():
    sim = simulate(900, 3)
    X = sim["X"].copy()
    X[::7, 2] = np.nan
    risks = est.point_treatment_arm_risks(
        X, sim["arm"], sim["outcome"], config=est.NuisanceConfig(n_folds=2, max_iter=20)
    )
    assert np.all(np.isfinite(risks.risk))
    with pytest.raises(est.EstimationError, match="missing"):
        est.point_treatment_arm_risks(X, sim["arm"], sim["outcome"], config=LOGISTIC)


def test_doubly_robust_scores_survive_one_wrong_nuisance_but_not_two():
    sim = simulate(60_000, 5)
    follows = sim["arm"] == "niv"
    truth = sim["potential"][:, 2].mean()
    true_pi = sim["propensity"][:, 2]
    true_m = sim["base"] + EFFECT["niv"]
    flat_pi = np.full(follows.size, follows.mean())
    flat_m = np.full(follows.size, sim["outcome"].mean())

    def risk(pi: np.ndarray, m: np.ndarray) -> float:
        return float(est.doubly_robust_scores(sim["outcome"], follows, pi, m).mean())

    assert abs(risk(true_pi, flat_m) - truth) < 0.012   # weights carry it
    assert abs(risk(flat_pi, true_m) - truth) < 0.012   # outcome model carries it
    assert abs(risk(flat_pi, flat_m) - truth) > 0.04    # nothing carries it


@pytest.mark.parametrize(
    "scenario",
    [
        {"outcome": "bump", "choice": "linear"},      # outcome model cannot fit: the weights carry it
        {"outcome": "logistic", "choice": "bump"},    # choice model cannot fit: the outcome model does
    ],
    ids=["outcome-model-wrong", "choice-model-wrong"],
)
def test_fitted_estimator_recovers_when_only_one_nuisance_model_can_be_right(scenario):
    errors, crude_errors = [], []
    for seed in range(5):
        sim = simulate(6000, seed, **scenario)
        truth = float((sim["risk"][:, 2] - sim["risk"][:, 0]).mean())
        risks = est.point_treatment_arm_risks(sim["X"], sim["arm"], sim["outcome"], config=LOGISTIC)
        errors.append(est.contrast(risks, "niv", "cot").risk_difference.estimate - truth)
        crude_errors.append(_crude(sim, "niv", "cot").risk_difference.estimate - truth)
        assert abs(errors[-1]) < 0.05
    assert abs(np.mean(errors)) < 0.02
    assert min(crude_errors) > 0.08


def test_both_nuisance_models_wrong_leaves_the_bias_in():
    sim = simulate(6000, 0, outcome="bump", choice="bump")
    risks = est.point_treatment_arm_risks(sim["X"], sim["arm"], sim["outcome"], config=LOGISTIC)
    error = est.contrast(risks, "niv", "cot").risk_difference.estimate - (EFFECT["niv"] - EFFECT["cot"])
    assert error > 0.1, "double robustness is not robustness to both models failing"


def test_three_arm_pairwise_contrasts_are_consistent():
    sim = simulate(2400, 2)
    risks = est.point_treatment_arm_risks(sim["X"], sim["arm"], sim["outcome"], config=LOGISTIC)
    assert risks.arms == ARMS
    niv_cot = est.contrast(risks, "niv", "cot").risk_difference.estimate
    niv_hfnc = est.contrast(risks, "niv", "hfnc").risk_difference.estimate
    hfnc_cot = est.contrast(risks, "hfnc", "cot").risk_difference.estimate
    assert niv_cot == pytest.approx(niv_hfnc + hfnc_cot, abs=1e-12)
    reverse = est.contrast(risks, "cot", "niv")
    assert reverse.risk_difference.estimate == pytest.approx(-niv_cot, abs=1e-12)
    assert reverse.risk_ratio.estimate == pytest.approx(
        1.0 / est.contrast(risks, "niv", "cot").risk_ratio.estimate
    )
    assert niv_cot < hfnc_cot < 0, "planted ordering niv < hfnc < cot"


def test_a_two_arm_contrast_keeps_third_arm_patients_in_the_target_population():
    sim = simulate(2400, 2)
    full = est.point_treatment_arm_risks(sim["X"], sim["arm"], sim["outcome"], config=LOGISTIC)
    pair = est.point_treatment_arm_risks(
        sim["X"], sim["arm"], sim["outcome"], arms=("hfnc", "cot"), config=LOGISTIC
    )
    assert pair.n == full.n == 2400
    assert pair.arms == ("hfnc", "cot")
    assert abs(pair.risk[0] - full.risk[full.arms.index("hfnc")]) < 0.03


def test_output_is_deterministic_for_a_fixed_seed_and_moves_with_the_seed():
    sim = simulate(1200, 4)
    args = (sim["X"], sim["arm"], sim["outcome"])
    first = est.point_treatment_arm_risks(*args, config=LOGISTIC)
    second = est.point_treatment_arm_risks(*args, config=LOGISTIC)
    other = est.point_treatment_arm_risks(*args, config=est.NuisanceConfig(learner="logistic", n_folds=3, seed=1))
    np.testing.assert_array_equal(first.risk, second.risk)
    np.testing.assert_array_equal(first.influence, second.influence)
    assert not np.array_equal(first.risk, other.risk)


def test_folds_are_assigned_by_patient_and_reproducible():
    groups = np.repeat(np.arange(40), 3)
    folds = est.patient_folds(groups, n_folds=4, seed=7)
    assert folds.shape == groups.shape
    for g in range(40):
        assert np.unique(folds[groups == g]).size == 1
    assert set(np.unique(folds)) == {0, 1, 2, 3}
    np.testing.assert_array_equal(folds, est.patient_folds(groups, n_folds=4, seed=7))
    assert not np.array_equal(folds, est.patient_folds(groups, n_folds=4, seed=8))


def test_repeated_rows_per_patient_widen_the_interval():
    sim = simulate(900, 6)
    X, arm, y = (np.concatenate([sim[k], sim[k]]) for k in ("X", "arm", "outcome"))
    ids = np.concatenate([np.arange(900), np.arange(900)])
    clustered = est.contrast(
        est.point_treatment_arm_risks(X, arm, y, groups=ids, config=LOGISTIC), "niv", "cot"
    )
    independent = est.contrast(est.point_treatment_arm_risks(X, arm, y, config=LOGISTIC), "niv", "cot")
    assert clustered.risk_difference.se > 1.2 * independent.risk_difference.se


# ---------------------------------------------------------------------------
# overlap weights and limited overlap
# ---------------------------------------------------------------------------

def _limited_overlap(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    propensity = _sigmoid(3.5 * x)
    treated = rng.random(n) < propensity
    outcome = (rng.random(n) < 0.3 + 0.1 * _sigmoid(x) - 0.08 * treated).astype(int)
    return {"X": x[:, None], "treated": treated, "outcome": outcome, "propensity": propensity}


def test_overlap_weights_stay_stable_where_inverse_probability_weights_blow_up():
    ipw_estimates, ow_estimates, ipw_se, ow_se = [], [], [], []
    for seed in range(40):
        sim = _limited_overlap(2000, seed)
        ipw = est.inverse_probability_weights(sim["propensity"], sim["treated"])
        ow = est.overlap_weights(sim["propensity"], sim["treated"])
        assert ow.max() <= 1.0 and ipw.max() > 30.0
        for mask in (sim["treated"], ~sim["treated"]):
            assert ow[mask].max() / ow[mask].sum() < 0.01      # no single row carries an arm
        assert max(ipw[m].max() / ipw[m].sum() for m in (sim["treated"], ~sim["treated"])) > 0.02
        by_ipw = est.weighted_contrast(sim["outcome"], sim["treated"], ipw).risk_difference
        by_ow = est.weighted_contrast(sim["outcome"], sim["treated"], ow).risk_difference
        ipw_estimates.append(by_ipw.estimate)
        ow_estimates.append(by_ow.estimate)
        ipw_se.append(by_ipw.se)
        ow_se.append(by_ow.se)
    assert np.std(ow_estimates) < 0.5 * np.std(ipw_estimates)
    assert np.mean(ow_se) < 0.75 * np.mean(ipw_se)
    assert abs(np.mean(ow_estimates) + 0.08) < 0.02


def test_inverse_probability_weight_clip_is_a_named_parameter():
    propensity = np.array([0.001, 0.5, 0.999])
    treated = np.array([True, True, False])
    np.testing.assert_allclose(est.inverse_probability_weights(propensity, treated), [1000.0, 2.0, 1000.0])
    np.testing.assert_allclose(
        est.inverse_probability_weights(propensity, treated, clip=0.05), [20.0, 2.0, 20.0]
    )
    np.testing.assert_allclose(est.overlap_weights(propensity, treated), [0.999, 0.5, 0.999])


def test_overlap_weighted_contrast_recovers_planted_difference_with_fitted_propensity():
    sim = simulate(3000, 8)
    result = est.overlap_weighted_contrast(
        sim["X"], sim["arm"], sim["outcome"], treated="niv", control="cot", config=LOGISTIC
    )
    assert result.estimand == "ATO"
    assert result.n == int(np.isin(sim["arm"], ["niv", "cot"]).sum())
    assert abs(result.risk_difference.estimate - (EFFECT["niv"] - EFFECT["cot"])) < 0.05
    assert result.risk_difference.lower < EFFECT["niv"] - EFFECT["cot"] < result.risk_difference.upper


def test_weighted_contrast_without_weights_is_the_crude_contrast():
    sim = simulate(1500, 9)
    pair = np.isin(sim["arm"], ["niv", "cot"])
    y, t = sim["outcome"][pair], sim["arm"][pair] == "niv"
    crude = est.weighted_contrast(y, t)
    assert crude.risk_treated == pytest.approx(y[t].mean())
    assert crude.risk_control == pytest.approx(y[~t].mean())
    p1, p0 = y[t].mean(), y[~t].mean()
    wald = math.sqrt(p1 * (1 - p1) / t.sum() + p0 * (1 - p0) / (~t).sum())
    assert crude.risk_difference.se == pytest.approx(wald, rel=5e-3)
    assert crude.risk_ratio.estimate == pytest.approx(p1 / p0)


# ---------------------------------------------------------------------------
# clone-censor-weight
# ---------------------------------------------------------------------------

def test_clone_rules_on_hand_built_rows():
    device_arm = np.array(["hfnc", "niv", "none", "niv", "none", "cot", "niv"])
    device_time = np.array([2.0, 3.0, np.inf, 10.0, np.nan, 0.0, 9.0])
    event_time = np.array([50.0, 1.0, 168.0, 168.0, 4.0, 0.0, 7.0])
    event_type = np.array([1, 2, 0, 0, 1, 1, 2])
    design = est.build_clones(device_arm, device_time, event_time, event_type, arms=ARMS, grace=6.0)

    assert design.arms == ARMS
    np.testing.assert_array_equal(design.at_risk, [True, False, True, True, False, True, True])
    expected = np.array([
        [False, True, False],    # HFNC row at 2 h: only the HFNC clone survives
        [True, True, True],      # event at 1 h, before the 3 h device row: every clone
        [False, False, False],   # no device row by 6 h: every clone censored
        [False, False, False],   # NIV at 10 h is rescue: every clone censored
        [True, True, True],      # event at 4 h with no device row yet: every clone
        [True, False, False],    # device and event tied at 0 h: the device row is first
        [False, False, False],   # event at 7 h, after the window closed with no device
    ])
    np.testing.assert_array_equal(design.uncensored, expected)
    assert design.censor_time[0, 0] == 2.0 and math.isinf(design.censor_time[0, 1])
    np.testing.assert_array_equal(design.censor_time[2], [6.0, 6.0, 6.0])
    np.testing.assert_array_equal(design.censor_time[3], [6.0, 6.0, 6.0])
    np.testing.assert_array_equal(design.censor_time[6], [6.0, 6.0, 6.0])


def test_event_before_the_first_device_row_counts_in_every_clone():
    sim = simulate_ccw(2400, 1)
    result = est.clone_censor_weight(
        sim["X"], sim["device_arm"], sim["device_time"], sim["event_time"], sim["event_type"],
        arms=ARMS, grace=sim["grace"], horizon=sim["horizon"], config=LOGISTIC,
    )
    early = sim["early"]
    assert early.sum() > 50
    np.testing.assert_array_equal(result.design.uncensored[early], True)
    np.testing.assert_array_equal(result.risks.weights[early], 1.0)
    for k in range(3):
        centred = result.risks.influence[early, k] + result.risks.risk[k]
        np.testing.assert_allclose(centred, 1.0)   # each contributes one whole event to each arm
    # dropping those rows lowers every arm's risk by the same early-event mass
    keep = ~early
    without = est.clone_censor_weight(
        sim["X"][keep], sim["device_arm"][keep], sim["device_time"][keep], sim["event_time"][keep],
        sim["event_type"][keep], arms=ARMS, grace=sim["grace"], horizon=sim["horizon"], config=LOGISTIC,
    )
    assert np.all(result.risks.risk > without.risks.risk)


def test_zero_grace_window_equals_the_point_treatment_estimate():
    sim = simulate(1800, 12)
    n = sim["arm"].size
    event = sim["outcome"].astype(bool)
    event_time = np.where(event, 24.0, 168.0)
    result = est.clone_censor_weight(
        sim["X"], sim["arm"], np.zeros(n), event_time, event.astype(int),
        arms=ARMS, grace=0.0, horizon=168.0, config=LOGISTIC,
    )
    point = est.point_treatment_arm_risks(sim["X"], sim["arm"], sim["outcome"], config=LOGISTIC)
    np.testing.assert_allclose(result.risks.risk, point.risk, rtol=0, atol=1e-12)
    ccw, fixed = est.contrast(result.risks, "niv", "cot"), est.contrast(point, "niv", "cot")
    assert ccw.risk_difference.estimate == pytest.approx(fixed.risk_difference.estimate, abs=1e-12)
    assert ccw.risk_difference.se == pytest.approx(fixed.risk_difference.se, abs=1e-12)
    assert ccw.risk_ratio.estimate == pytest.approx(fixed.risk_ratio.estimate, abs=1e-12)


def test_zero_grace_window_treats_later_device_rows_as_no_assignment():
    sim = simulate(1800, 13)
    n = sim["arm"].size
    rng = np.random.default_rng(0)
    device_time = np.where(rng.random(n) < 0.2, 3.0, 0.0)
    event = sim["outcome"].astype(bool)
    result = est.clone_censor_weight(
        sim["X"], sim["arm"], device_time, np.where(event, 24.0, 168.0), event.astype(int),
        arms=ARMS, grace=0.0, horizon=168.0, config=LOGISTIC,
    )
    exposure = np.where(device_time == 0.0, sim["arm"], "unassigned")
    point = est.point_treatment_arm_risks(sim["X"], exposure, sim["outcome"], arms=ARMS, config=LOGISTIC)
    np.testing.assert_allclose(result.risks.risk, point.risk, rtol=0, atol=1e-12)
    assert not np.allclose(
        point.risk, est.point_treatment_arm_risks(sim["X"], sim["arm"], sim["outcome"], config=LOGISTIC).risk
    )


def test_clone_censor_weight_recovers_planted_difference_with_a_grace_window():
    estimates, truths = [], []
    for seed in range(5):
        sim = simulate_ccw(6000, seed)
        result = est.clone_censor_weight(
            sim["X"], sim["device_arm"], sim["device_time"], sim["event_time"], sim["event_type"],
            arms=ARMS, grace=sim["grace"], horizon=sim["horizon"], config=LOGISTIC,
        )
        truth = sim["truth"]["niv"] - sim["truth"]["cot"]
        estimate = est.contrast(result.risks, "niv", "cot").risk_difference.estimate
        assert abs(estimate - truth) < 0.05
        estimates.append(estimate)
        truths.append(truth)
        assert result.risks.method == "clone_censor_weight"
    assert abs(np.mean(estimates) - np.mean(truths)) < 0.02


def test_censoring_weights_carry_the_clone_estimate_when_the_outcome_model_cannot():
    errors = []
    for seed in range(5):
        sim = simulate_ccw(6000, seed, late_rate=0.0, outcome="bump")
        result = est.clone_censor_weight(
            sim["X"], sim["device_arm"], sim["device_time"], sim["event_time"], sim["event_type"],
            arms=ARMS, grace=sim["grace"], horizon=sim["horizon"], config=LOGISTIC,
        )
        truth = sim["truth"]["niv"] - sim["truth"]["cot"]
        errors.append(est.contrast(result.risks, "niv", "cot").risk_difference.estimate - truth)
        assert abs(errors[-1]) < 0.05
    assert abs(np.mean(errors)) < 0.02


def test_clone_censor_weight_reports_cause_specific_incidence_per_arm():
    sim = simulate_ccw(2400, 2)
    args = (sim["X"], sim["device_arm"], sim["device_time"], sim["event_time"], sim["event_type"])
    kwargs = dict(arms=ARMS, grace=sim["grace"], horizon=sim["horizon"], config=LOGISTIC)
    composite = est.clone_censor_weight(*args, **kwargs)
    assert set(composite.incidence) == set(ARMS)
    for k, arm in enumerate(ARMS):
        curve = composite.incidence[arm]
        total = curve.incidence[1] + curve.incidence[2] + curve.survival
        np.testing.assert_allclose(total, 1.0, atol=1e-12)
        assert curve.times[-1] == sim["horizon"]
        # the weighted curve and the doubly robust risk estimate the same quantity
        assert abs((1.0 - curve.survival[-1]) - composite.risks.risk[k]) < 0.04
    reintubation = est.clone_censor_weight(*args, event_of_interest=1, **kwargs)
    death = est.clone_censor_weight(*args, event_of_interest=2, **kwargs)
    np.testing.assert_allclose(reintubation.risks.risk + death.risks.risk, composite.risks.risk, atol=0.02)
    assert np.all(reintubation.risks.risk > death.risks.risk)


# ---------------------------------------------------------------------------
# cumulative incidence with a competing event
# ---------------------------------------------------------------------------

def test_cause_specific_incidences_and_survival_sum_to_one():
    rng = np.random.default_rng(0)
    n = 500
    time = rng.integers(1, 15, n).astype(float)
    event_type = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    weights = rng.uniform(0.2, 5.0, n)
    for w in (None, weights):
        curve = est.cumulative_incidence(time, event_type, weights=w, horizon=10.0)
        assert curve.causes == (1, 2)
        np.testing.assert_allclose(curve.incidence[1] + curve.incidence[2] + curve.survival, 1.0, atol=1e-12)
        assert np.all(np.diff(curve.incidence[1]) >= 0) and np.all(np.diff(curve.survival) <= 0)
        assert curve.times[-1] == 10.0 and np.all(curve.times <= 10.0)


def test_cumulative_incidence_matches_a_hand_computed_example():
    # five patients: reintubated at 1, censored at 2, died at 2, reintubated at 3, event-free to 4
    time = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    event_type = np.array([1, 0, 2, 1, 0])
    curve = est.cumulative_incidence(time, event_type)
    np.testing.assert_array_equal(curve.times, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(curve.incidence[1], [0.2, 0.2, 0.2 + 0.6 * 0.5])
    np.testing.assert_allclose(curve.incidence[2], [0.0, 0.8 * 0.25, 0.2])
    np.testing.assert_allclose(curve.survival, [0.8, 0.6, 0.3])


def test_without_censoring_incidence_is_the_weighted_event_share():
    time = np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0])
    event_type = np.array([1, 2, 1, 0, 0, 0])
    weights = np.array([2.0, 1.0, 1.0, 3.0, 2.0, 1.0])
    curve = est.cumulative_incidence(time, event_type, weights=weights, horizon=5.0)
    assert curve.incidence[1][-1] == pytest.approx(3.0 / 10.0)
    assert curve.incidence[2][-1] == pytest.approx(1.0 / 10.0)
    zeroed = est.cumulative_incidence(
        np.append(time, 0.5), np.append(event_type, 2), weights=np.append(weights, 0.0), horizon=5.0
    )
    assert zeroed.incidence[2][-1] == pytest.approx(1.0 / 10.0)


# ---------------------------------------------------------------------------
# negative control and bias analysis
# ---------------------------------------------------------------------------

def test_negative_control_outcome_interval_covers_zero_while_crude_does_not():
    covered = 0
    for seed in range(5):
        sim = simulate(3000, 20 + seed)
        risks = est.point_treatment_arm_risks(sim["X"], sim["arm"], sim["negative_control"], config=LOGISTIC)
        check = est.negative_control_check(risks, "niv", "cot")
        covered += check.covers_null
        assert check.covers_null == (check.risk_difference.lower <= 0.0 <= check.risk_difference.upper)
        pair = np.isin(sim["arm"], ["niv", "cot"])
        crude = est.weighted_contrast(sim["negative_control"][pair], sim["arm"][pair] == "niv")
        assert crude.risk_difference.lower > 0.0, "the control outcome is confounded in the crude contrast"
    assert covered >= 4


def test_negative_control_outcome_runs_through_the_clone_estimator():
    sim = simulate_ccw(3000, 21)
    result = est.clone_censor_weight(
        sim["X"], sim["device_arm"], sim["device_time"], sim["event_time"], sim["event_type"],
        arms=ARMS, grace=sim["grace"], horizon=sim["horizon"], config=LOGISTIC,
        outcome=sim["negative_control"],
    )
    assert result.incidence is None
    assert est.negative_control_check(result.risks, "niv", "cot").covers_null


def test_e_value_matches_published_examples():
    # VanderWeele & Ding 2017: RR 3.9 (95% CI 1.8 to 8.7) gives 7.26 and 3.0.
    value = est.e_value(3.9, lower=1.8, upper=8.7)
    assert value.point == pytest.approx(7.26, abs=0.005)
    assert value.confidence_limit == pytest.approx(3.0, abs=0.005)
    # Mathur et al. 2018: RR 1.33 gives 2.0.
    assert est.e_value(1.33).point == pytest.approx(2.0, abs=0.01)
    assert est.e_value(1.33).confidence_limit is None


def test_e_value_inverts_protective_ratios_and_returns_one_when_the_interval_covers_null():
    protective = est.e_value(0.5, lower=0.4, upper=0.8)
    assert protective.point == pytest.approx(2.0 + math.sqrt(2.0))
    assert protective.confidence_limit == pytest.approx(1.25 + math.sqrt(1.25 * 0.25))
    assert est.e_value(0.8, lower=0.6, upper=1.1).confidence_limit == 1.0
    assert est.e_value(1.4, lower=0.9, upper=2.0).confidence_limit == 1.0
    assert est.e_value(1.0).point == 1.0
    with pytest.raises(ValueError):
        est.e_value(0.0)
    with pytest.raises(ValueError):
        est.e_value(1.5, lower=1.2)
    with pytest.raises(ValueError):
        est.e_value(1.5, lower=1.6, upper=2.0)


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def test_effective_sample_size_matches_the_closed_form():
    assert dx.effective_sample_size(np.ones(50)) == pytest.approx(50.0)
    assert dx.effective_sample_size(np.array([1.0, 1.0, 2.0])) == pytest.approx(16.0 / 6.0)
    assert dx.effective_sample_size(np.array([5.0, 0.0, 0.0])) == pytest.approx(1.0)
    w = np.random.default_rng(0).uniform(0.1, 9.0, 400)
    assert dx.effective_sample_size(w) == pytest.approx(w.sum() ** 2 / (w ** 2).sum())
    assert dx.effective_sample_size(3.0 * w) == pytest.approx(dx.effective_sample_size(w))
    for bad in (np.array([]), np.zeros(4), np.array([1.0, -1.0]), np.array([1.0, np.nan])):
        with pytest.raises(ValueError):
            dx.effective_sample_size(bad)


def test_overlap_summary_reports_distribution_by_arm_and_share_outside_support():
    propensity = np.array([0.02, 0.2, 0.4, 0.6, 0.5, 0.7, 0.9, 0.98])
    in_arm = np.array([False, False, False, False, True, True, True, True])
    summary = dx.overlap_summary(propensity, in_arm)
    assert summary.support == (0.5, 0.6)          # max of the minima, min of the maxima
    assert summary.share_outside == pytest.approx(6 / 8)
    assert summary.n_in_arm == 4 and summary.n_out_of_arm == 4
    fixed = dx.overlap_summary(propensity, in_arm, support=(0.05, 0.95))
    assert fixed.share_outside == pytest.approx(2 / 8)
    assert fixed.share_outside_in_arm == pytest.approx(1 / 4)
    assert fixed.share_outside_out_of_arm == pytest.approx(1 / 4)
    assert fixed.quantiles_in_arm[fixed.quantiles.index(0.5)] == pytest.approx(0.8)
    assert fixed.quantiles_out_of_arm[fixed.quantiles.index(0.5)] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        dx.overlap_summary(propensity, np.ones(8, dtype=bool))
    with pytest.raises(ValueError):
        dx.overlap_summary(np.array([0.2, np.nan]), np.array([True, False]))


def test_limited_overlap_shows_up_in_the_overlap_summary():
    good = simulate(2000, 0, confounding=0.3)
    poor = _limited_overlap(2000, 0)
    share_good = dx.overlap_summary(good["propensity"][:, 2], good["arm"] == "niv", support=(0.05, 0.95)).share_outside
    share_poor = dx.overlap_summary(poor["propensity"], poor["treated"], support=(0.05, 0.95)).share_outside
    assert share_good < 0.05 < 0.3 < share_poor


def test_covariate_balance_before_and_after_weighting():
    sim = simulate(6000, 1)
    pair = np.isin(sim["arm"], ["niv", "cot"])
    X, treated = sim["X"][pair], sim["arm"][pair] == "niv"
    propensity = sim["propensity"][pair, 2] / (sim["propensity"][pair, 2] + sim["propensity"][pair, 0])
    weights = est.inverse_probability_weights(propensity, treated)
    table = dx.covariate_balance(X, treated, weights, names=("x0", "x1", "x2", "x3"))
    assert table.names == ("x0", "x1", "x2", "x3")
    assert table.smd_unweighted[0] > 0.5
    assert table.max_abs_unweighted == pytest.approx(np.abs(table.smd_unweighted).max())
    assert table.max_abs_weighted < 0.1
    pooled = math.sqrt((X[treated, 0].var(ddof=1) + X[~treated, 0].var(ddof=1)) / 2.0)
    assert table.smd_unweighted[0] == pytest.approx((X[treated, 0].mean() - X[~treated, 0].mean()) / pooled)
    unweighted_only = dx.covariate_balance(X, treated)
    np.testing.assert_allclose(unweighted_only.smd_weighted, unweighted_only.smd_unweighted)


def test_covariate_balance_handles_missing_and_constant_columns():
    X = np.array([[1.0, 5.0], [2.0, 5.0], [np.nan, 5.0], [4.0, 5.0], [6.0, 5.0], [8.0, 5.0]])
    treated = np.array([True, True, True, False, False, False])
    table = dx.covariate_balance(X, treated)
    assert table.smd_unweighted[1] == 0.0
    assert table.smd_unweighted[0] == pytest.approx((1.5 - 6.0) / math.sqrt((0.5 + 4.0) / 2.0))
    with pytest.raises(ValueError):
        dx.covariate_balance(X, np.zeros(6, dtype=bool))


def test_minimal_detectable_effect_matches_the_closed_form_without_outcomes_by_arm():
    mde = dx.minimal_detectable_effect(1000, 1000, 0.2)
    assert mde == pytest.approx((1.959964 + 0.841621) * math.sqrt(0.16 * 0.002), rel=1e-5)
    assert dx.minimal_detectable_effect(250.5, 1000, 0.2) > mde               # ESS in, not only counts
    assert dx.minimal_detectable_effect(4000, 4000, 0.2) == pytest.approx(mde / 2.0)
    assert dx.minimal_detectable_effect(1000, 1000, 0.2, power=0.9) > mde
    assert dx.minimal_detectable_effect(1000, 1000, 0.2, alpha=0.01) > mde
    for bad in ((0, 100, 0.2), (100, 100, 0.0), (100, 100, 1.0), (100, -5, 0.2)):
        with pytest.raises(ValueError):
            dx.minimal_detectable_effect(*bad)
    with pytest.raises(ValueError):
        dx.minimal_detectable_effect(100, 100, 0.2, alpha=1.5)
    with pytest.raises(ValueError):
        dx.minimal_detectable_effect(100, 100, 0.2, power=0.0)


# ---------------------------------------------------------------------------
# errors: fail closed rather than return NaN
# ---------------------------------------------------------------------------

def test_empty_arm_raises():
    sim = simulate(600, 0)
    with pytest.raises(est.EstimationError, match="no rows"):
        est.point_treatment_arm_risks(
            sim["X"], sim["arm"], sim["outcome"], arms=("niv", "cot", "helmet"), config=LOGISTIC
        )
    with pytest.raises(est.EstimationError, match="no rows"):
        est.overlap_weighted_contrast(
            sim["X"], sim["arm"], sim["outcome"], treated="helmet", control="cot", config=LOGISTIC
        )
    with pytest.raises(est.EstimationError, match="no rows"):
        est.weighted_contrast(sim["outcome"], np.zeros(600, dtype=bool))


def test_single_class_outcome_raises():
    sim = simulate(600, 0)
    with pytest.raises(est.EstimationError, match="single"):
        est.point_treatment_arm_risks(sim["X"], sim["arm"], np.zeros(600, dtype=int), config=LOGISTIC)
    outcome = sim["outcome"].copy()
    outcome[sim["arm"] == "hfnc"] = 0
    with pytest.raises(est.EstimationError, match="single"):
        est.point_treatment_arm_risks(sim["X"], sim["arm"], outcome, config=LOGISTIC)


def test_fewer_patients_than_folds_raises():
    sim = simulate(600, 0)
    with pytest.raises(est.EstimationError, match="folds"):
        est.patient_folds(np.arange(3), n_folds=5, seed=0)
    with pytest.raises(est.EstimationError, match="folds"):
        est.point_treatment_arm_risks(
            sim["X"][:4], sim["arm"][:4], sim["outcome"][:4], config=est.NuisanceConfig(learner="logistic")
        )
    with pytest.raises(est.EstimationError, match="folds"):     # 600 rows but only 2 patients
        est.point_treatment_arm_risks(
            sim["X"], sim["arm"], sim["outcome"], groups=np.arange(600) % 2, config=LOGISTIC
        )


def test_a_class_missing_from_a_training_fold_raises_instead_of_guessing():
    sim = simulate(60, 0)
    arm = np.array(["cot"] * 58 + ["niv"] * 2)
    outcome = np.tile([0, 1], 30)
    with pytest.raises(est.EstimationError, match="training fold"):
        est.point_treatment_arm_risks(sim["X"], arm, outcome, config=est.NuisanceConfig(learner="logistic", n_folds=2))


def test_malformed_inputs_raise():
    sim = simulate(300, 0)
    X, arm, y = sim["X"], sim["arm"], sim["outcome"]
    with pytest.raises(ValueError, match="length"):
        est.point_treatment_arm_risks(X, arm[:-1], y, config=LOGISTIC)
    with pytest.raises(ValueError, match="binary"):
        est.point_treatment_arm_risks(X, arm, y + 1, config=LOGISTIC)
    with pytest.raises(ValueError, match="binary"):
        est.point_treatment_arm_risks(X, arm, np.where(y == 1, np.nan, 0.0), config=LOGISTIC)
    with pytest.raises(ValueError, match="2-D"):
        est.point_treatment_arm_risks(X[:, 0], arm, y, config=LOGISTIC)
    with pytest.raises(ValueError, match="distinct"):
        est.point_treatment_arm_risks(X, arm, y, arms=("niv", "niv"), config=LOGISTIC)
    risks = est.point_treatment_arm_risks(X, arm, y, config=LOGISTIC)
    with pytest.raises(ValueError, match="differ"):
        est.contrast(risks, "niv", "niv")
    with pytest.raises(KeyError):
        est.contrast(risks, "niv", "helmet")
    with pytest.raises(ValueError):
        est.contrast(risks, "niv", "cot", alpha=0.0)
    for bad in (dict(learner="lightgbm"), dict(n_folds=1), dict(clip=0.6), dict(clip=-0.1), dict(max_iter=0)):
        with pytest.raises(ValueError):
            est.NuisanceConfig(**bad)


def test_clone_censor_weight_refuses_incomplete_follow_up_and_bad_windows():
    sim = simulate_ccw(600, 0)
    args = [sim["X"], sim["device_arm"], sim["device_time"], sim["event_time"].copy(), sim["event_type"]]
    kwargs = dict(arms=ARMS, grace=sim["grace"], horizon=sim["horizon"], config=LOGISTIC)
    lost = np.flatnonzero(sim["event_type"] == 0)[:5]
    args[3][lost] = 48.0   # censored at discharge before the horizon
    with pytest.raises(est.EstimationError, match="follow-up"):
        est.clone_censor_weight(*args, **kwargs)
    args[3] = sim["event_time"]
    with pytest.raises(ValueError, match="grace"):
        est.clone_censor_weight(*args, **{**kwargs, "grace": -1.0})
    with pytest.raises(ValueError, match="horizon"):
        est.clone_censor_weight(*args, **{**kwargs, "horizon": 2.0})
    with pytest.raises(ValueError, match="device_time"):
        est.build_clones(sim["device_arm"], -np.ones(600), sim["event_time"], sim["event_type"], arms=ARMS, grace=6.0)
    with pytest.raises(ValueError, match="event_type"):
        est.build_clones(sim["device_arm"], sim["device_time"], sim["event_time"], -sim["event_type"] - 1,
                         arms=ARMS, grace=6.0)


def test_risk_ratio_is_refused_when_an_arm_risk_is_not_positive():
    risks = est.ArmRisks(
        arms=("a", "b"), risk=np.array([0.2, 0.0]), influence=np.zeros((4, 2)) + [[0.1], [-0.1], [0.1], [-0.1]],
        cluster=np.arange(4), n=4, method="aipw", n_followers=np.array([2, 2]), probability=None, weights=None,
    )
    with pytest.raises(est.EstimationError, match="risk ratio"):
        est.contrast(risks, "a", "b")
    assert est.negative_control_check(risks, "a", "b").risk_difference.estimate == pytest.approx(0.2)


def test_cumulative_incidence_rejects_bad_input():
    with pytest.raises(ValueError):
        est.cumulative_incidence(np.array([]), np.array([], dtype=int))
    with pytest.raises(ValueError):
        est.cumulative_incidence(np.array([1.0, 2.0]), np.array([1, 0]), weights=np.zeros(2))
    with pytest.raises(ValueError):
        est.cumulative_incidence(np.array([1.0, 2.0]), np.array([1, -1]))
    with pytest.raises(ValueError):
        est.cumulative_incidence(np.array([1.0, np.nan]), np.array([1, 0]))


# ---------------------------------------------------------------------------
# integration and packaging
# ---------------------------------------------------------------------------

def test_emulation_pipeline_from_clones_to_bias_analysis():
    sim = simulate_ccw(3000, 3)
    config = est.NuisanceConfig(n_folds=3, seed=0, max_iter=60)
    result = est.clone_censor_weight(
        sim["X"], sim["device_arm"], sim["device_time"], sim["event_time"], sim["event_type"],
        arms=("niv", "cot"), grace=sim["grace"], horizon=sim["horizon"], groups=np.arange(3000), config=config,
    )
    effect = est.contrast(result.risks, "niv", "cot")
    truth = sim["truth"]["niv"] - sim["truth"]["cot"]
    assert abs(effect.risk_difference.estimate - truth) < 0.06
    assert effect.risk_ratio.estimate < 1.0

    at_risk = result.design.at_risk
    k = result.risks.arms.index("niv")
    followers = result.design.uncensored[:, k] & at_risk
    probability = result.risks.probability[:, k]
    assert np.all(np.isnan(probability[~at_risk])) and np.all(probability[at_risk] >= config.clip)
    overlap = dx.overlap_summary(probability[at_risk], followers[at_risk], support=(0.05, 0.95))
    assert 0.0 <= overlap.share_outside < 0.5

    # weighted NIV clones should look like the whole at-risk cohort on the confounder
    stacked = np.concatenate([sim["X"][followers], sim["X"][at_risk]])
    is_clone = np.concatenate([np.ones(followers.sum(), bool), np.zeros(at_risk.sum(), bool)])
    weights = np.concatenate([result.risks.weights[followers, k], np.ones(at_risk.sum())])
    balance = dx.covariate_balance(stacked, is_clone, weights)
    assert abs(balance.smd_unweighted[0]) > 0.3
    assert balance.max_abs_weighted < 0.15

    ess = dx.effective_sample_size(result.risks.weights[followers, k])
    assert 1.0 < ess < followers.sum()
    assert dx.minimal_detectable_effect(ess, ess, baseline_risk=0.3) > 0.0

    bias = est.e_value(effect.risk_ratio.estimate, lower=effect.risk_ratio.lower, upper=effect.risk_ratio.upper)
    assert bias.point > 1.0 and 1.0 <= bias.confidence_limit <= bias.point


def test_modules_import_nothing_beyond_stdlib_numpy_and_sklearn():
    sources = sorted(PACKAGE_DIR.glob("*.py"))
    assert {p.name for p in sources} >= {"__init__.py", "estimators.py", "diagnostics.py"}
    allowed = set(sys.stdlib_module_names) | {"numpy", "sklearn"}
    for path in (PACKAGE_DIR / name for name in ("__init__.py", "estimators.py", "diagnostics.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    assert node.level == 1, f"{path.name}: relative import leaves the package"
                    continue
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                assert root in allowed, f"{path.name} imports {root!r}"
        text = path.read_text()
        assert "__import__(" not in text and "importlib" not in text, f"{path.name}: dynamic import"
