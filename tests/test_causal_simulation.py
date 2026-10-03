"""Planted-effect (plasmode) simulation of the agreement rule (plan U12; R29).

Synthetic covariates and arms only. The real registry supplies the trials and margins.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from src.eval.causal import benchmark as bm
from src.eval.causal import simulation as sim
from src.eval.causal.estimators import NuisanceConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG = NuisanceConfig(learner="logistic", n_folds=2, seed=0)
ARMS = np.array(["conventional_oxygen", "hfnc", "niv"], dtype=object)


@pytest.fixture(scope="module")
def registry() -> bm.Registry:
    return bm.load_benchmark_registry(ROOT / "configs/extubation_benchmarks.yaml")


@pytest.fixture(scope="module")
def frozen_cohort():
    """Stand-in for a frozen cohort: covariates and the device each patient received."""
    rng = np.random.default_rng(42)
    n = 1500
    X = rng.normal(size=(n, 3))
    logits = np.column_stack([np.zeros(n), -0.3 + 0.8 * X[:, 0], -0.6 + 1.0 * X[:, 1] - 0.5 * X[:, 2]])
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    arm = ARMS[[rng.choice(3, p=row) for row in p]]
    return X, arm


def report(registry, frozen_cohort, trial_id, **kwargs):
    X, arm = frozen_cohort
    kwargs.setdefault("baseline", 0.25)
    kwargs.setdefault("n_reps", 8)
    kwargs.setdefault("config", CONFIG)
    kwargs.setdefault("seed", 3)
    return sim.simulate_operating_characteristics(X, arm, registry, trial_id, **kwargs)


def test_planted_zero_effect_false_pass_rate_matches_the_simulation_count(registry, frozen_cohort):
    result = report(registry, frozen_cohort, "hernandez_2022_very_high_risk",
                    effects=("zero",), confounded=(False,))
    zero = result.scenarios[("zero", False)]
    assert zero.planted_log_rr == 0.0
    assert zero.n_reps == 8 and zero.n_completed + zero.n_failed == 8
    assert zero.n_pass == sum(zero.passes)
    assert zero.pass_rate == pytest.approx(zero.n_pass / zero.n_completed)
    # Each recorded pass is the rule applied to that replicate's estimate.
    trial = registry.trials["hernandez_2022_very_high_risk"]
    for estimate, passed in zip(zero.estimates, zero.passes, strict=True):
        assert passed == bm.single_trial_agreement(trial, estimate, second_hurdle=registry.uses_second_hurdle)
    # Deterministic for a fixed seed.
    again = report(registry, frozen_cohort, "hernandez_2022_very_high_risk",
                   effects=("zero",), confounded=(False,))
    assert again.scenarios[("zero", False)].passes == zero.passes


def test_rule_passes_more_often_for_the_trial_effect_than_for_zero_or_reversed(registry, frozen_cohort):
    result = report(registry, frozen_cohort, "hernandez_2022_very_high_risk", confounded=(False,), n_sim=3000)
    planted = result.scenarios[("trial", False)]
    trial = registry.trials["hernandez_2022_very_high_risk"]
    assert planted.planted_log_rr == pytest.approx(trial.effect.log_estimate)
    assert abs(planted.mean_log_estimate - planted.planted_log_rr) < 0.15
    assert planted.pass_rate > result.scenarios[("reversed", False)].pass_rate
    assert planted.pass_rate > result.scenarios[("zero", False)].pass_rate
    assert result.scenarios[("reversed", False)].planted_log_rr == pytest.approx(-trial.effect.log_estimate)
    assert set(result.summary()) == {"trial", "zero", "reversed"}
    assert result.margins_status == "proposed"


def test_simulated_treatment_keeps_overlap_in_the_range_of_the_fitted_model(frozen_cohort):
    X, arm = frozen_cohort
    model = sim.fit_device_choice_model(X, arm, config=CONFIG)
    fitted = model.probabilities(X)
    data = sim.simulate_once(model, X, treated="niv", control="hfnc", baseline=0.2, planted_rr=1.0,
                             rng=np.random.default_rng(0), n_sim=4000)
    lo, hi = fitted.min(axis=0), fitted.max(axis=0)
    assert np.all(data.propensity >= lo - 1e-12) and np.all(data.propensity <= hi + 1e-12)
    # Every arm is drawn, at the rate the fitted model implies.
    for k, label in enumerate(model.classes):
        assert abs(np.mean(data.arm == label) - data.propensity[:, k].mean()) < 0.03
    assert np.allclose(data.propensity.sum(axis=1), 1.0)


@pytest.mark.parametrize("trial_id", ["hernandez_2022_very_high_risk", "hernandez_2016_high_risk"])
def test_withheld_confounder_shows_the_expected_bias_direction(registry, frozen_cohort, trial_id):
    result = report(registry, frozen_cohort, trial_id, effects=("zero",), confounded=(False, True),
                    n_reps=6, n_sim=2500, confounder_strength=(1.2, 0.8))
    clean, confounded = result.scenarios[("zero", False)], result.scenarios[("zero", True)]
    sign = registry.trials[trial_id].expected_bias_sign
    assert sign * (confounded.mean_log_estimate - clean.mean_log_estimate) > 0.1
    assert sign * confounded.bias > 0.1
    assert abs(clean.bias) < 0.12
    assert confounded.bias_matches_expected_direction is True


def test_baseline_risk_never_reads_outcomes_by_arm(frozen_cohort):
    X, arm = frozen_cohort
    parameters = inspect.signature(sim.fit_baseline_risk_model).parameters
    assert "arm" not in parameters and list(parameters)[:2] == ["X", "outcome"]
    outcome = (np.random.default_rng(1).random(X.shape[0]) < 0.2).astype(float)
    baseline = sim.fit_baseline_risk_model(X, outcome, config=CONFIG)
    risk = baseline.risk(X)
    assert risk.shape == (X.shape[0],) and np.all((risk > 0) & (risk < 1))
    model = sim.fit_device_choice_model(X, arm, config=CONFIG)
    data = sim.simulate_once(model, X, treated="niv", control="hfnc", baseline=baseline, planted_rr=0.6,
                             rng=np.random.default_rng(2))
    assert data.y.shape == (X.shape[0],) and set(np.unique(data.y)) <= {0.0, 1.0}


def test_simulation_rejects_bad_input(registry, frozen_cohort):
    X, arm = frozen_cohort
    model = sim.fit_device_choice_model(X, arm, config=CONFIG)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sim.simulate_once(model, X, treated="niv", control="hfnc", baseline=1.5, planted_rr=1.0, rng=rng)
    with pytest.raises(ValueError):
        sim.simulate_once(model, X, treated="niv", control="niv", baseline=0.2, planted_rr=1.0, rng=rng)
    with pytest.raises(ValueError):
        sim.simulate_once(model, X, treated="ecmo", control="hfnc", baseline=0.2, planted_rr=1.0, rng=rng)
    with pytest.raises(ValueError):
        sim.simulate_once(model, X, treated="niv", control="hfnc", baseline=0.2, planted_rr=-1.0, rng=rng)
    with pytest.raises(ValueError):
        report(registry, frozen_cohort, "hernandez_2022_very_high_risk", effects=("double",))
    with pytest.raises(ValueError):
        report(registry, frozen_cohort, "hernandez_2022_very_high_risk", n_reps=0)
    with pytest.raises(ValueError):
        sim.fit_device_choice_model(X, np.array(["niv"] * X.shape[0], dtype=object), config=CONFIG)


def test_failed_replicates_are_counted_and_left_out_of_the_pass_rate(registry, frozen_cohort):
    trial = registry.trials["hernandez_2022_very_high_risk"]
    calls = {"n": 0}

    def flaky(X, arm, y, treated, control):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise ValueError("synthetic estimation failure")
        return bm.TrialEstimate(log_effect=trial.effect.log_estimate, se=0.05)

    result = report(registry, frozen_cohort, "hernandez_2022_very_high_risk", effects=("zero",),
                    confounded=(False,), n_reps=9, estimator=flaky)
    zero = result.scenarios[("zero", False)]
    assert (zero.n_completed, zero.n_failed, zero.n_pass) == (6, 3, 6)
    assert zero.pass_rate == 1.0                     # 6 of 6 completed, not 6 of 9
    assert len(zero.passes) == len(zero.estimates) == 6


def test_second_hurdle_cuts_the_reversed_effect_pass_rate_of_a_null_trial(registry, frozen_cohort):
    """Casey (a null benchmark, RR about 1.2) passed most replicates under a REVERSED planted
    effect with the old rule (interval inside a 1.5 equivalence margin, no hurdle). With the
    Roehmel-Kieser second hurdle the reversed effect lands on the wrong side of 1 and fails."""
    from dataclasses import replace

    result = report(registry, frozen_cohort, "casey_2021_all_comers", effects=("reversed",),
                    confounded=(False,), n_reps=12, n_sim=3000)
    reversed_ = result.scenarios[("reversed", False)]
    trial = registry.trials["casey_2021_all_comers"]
    without = sum(bm.single_trial_agreement(trial, e, second_hurdle=False) for e in reversed_.estimates)
    assert reversed_.n_pass < without
    assert reversed_.pass_rate <= 0.25
    assert replace(registry.second_hurdle).value == "direction_consistent"


def test_positive_control_detects_a_planted_true_effect(registry, frozen_cohort):
    X, arm = frozen_cohort
    result = sim.positive_control_detection(X, arm, registry, "hernandez_2022_very_high_risk",
                                            baseline=0.25, planted_rr=0.5, n_reps=8, n_sim=3000,
                                            config=CONFIG, seed=1)
    assert result.n_completed == 8
    assert result.detection_rate >= 0.8
    assert result.mean_log_estimate == pytest.approx(math.log(0.5), abs=0.2)
    # A planted null is not a positive control.
    with pytest.raises(ValueError):
        sim.positive_control_detection(X, arm, registry, "hernandez_2022_very_high_risk",
                                       baseline=0.25, planted_rr=1.0, n_reps=2, config=CONFIG)
