"""Benchmark registry and agreement rule (plan U12; R8, R10, R25, R27).

Everything here is synthetic: hand-built estimates and a registry read from
configs/extubation_benchmarks.yaml. No cohort outcome is read.
"""

from __future__ import annotations

import ast
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.eval.causal import benchmark as bm

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/extubation_benchmarks.yaml"
R8_TRIALS = {
    "hernandez_2016_low_risk": ("26975498", "hfnc", "conventional_oxygen", "reintubation", 72, "effect"),
    "hernandez_2016_high_risk": ("27706464", "hfnc", "niv", "reintubation", 72, "null"),
    "high_wean_2019": ("31577036", "niv", "hfnc", "reintubation", 168, "effect"),
    "hernandez_2022_very_high_risk": ("36400984", "niv", "hfnc", "reintubation", 168, "effect"),
    "ferrer_2009_hypercapnic": ("19682735", "niv", "conventional_oxygen", "reintubation", 72, "effect"),
    "casey_2021_all_comers": ("33794131", "hfnc", "conventional_oxygen", "reintubation", 96, "null"),
}


@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text())


@pytest.fixture(scope="module")
def registry() -> bm.Registry:
    return bm.load_benchmark_registry(REGISTRY_PATH)


def estimate(ratio: float, se: float) -> bm.TrialEstimate:
    return bm.TrialEstimate(log_effect=math.log(ratio), se=se)


def feasible(registry: bm.Registry, trial_id: str, **changes) -> bm.FeasibilityResult:
    inputs = dict(
        n_treated=2000, n_control=2000, ess_treated=1500.0, ess_control=1500.0,
        share_below_min_probability=0.01,
        estimators_runnable={"classical": True, "model": True},
    )
    inputs.update(changes)
    return bm.feasibility_screen(registry, trial_id, bm.FeasibilityInputs(**inputs))


# ---------------------------------------------------------------------------
# registry: content
# ---------------------------------------------------------------------------

def test_registry_holds_one_entry_per_candidate_trial_with_its_own_arms_and_window(registry):
    assert set(registry.trials) == set(R8_TRIALS)
    for trial_id, (pmid, treated, control, event, window, finding) in R8_TRIALS.items():
        trial = registry.trials[trial_id]
        assert (trial.pmid, trial.treated, trial.control) == (pmid, treated, control)
        assert (trial.event, trial.window_hours, trial.finding) == (event, window, finding)
    assert registry.trials["high_wean_2019"].approximate
    assert registry.trials["high_wean_2019"].assignment_column == "arm_highest_support"
    assert registry.audit_trial == "casey_2021_all_comers"


def test_benchmark_risk_ratio_is_computed_from_the_published_arm_counts(registry):
    # Hernandez 2016 low risk: 13/264 vs 32/263.
    effect = registry.trials["hernandez_2016_low_risk"].effect
    ratio = (13 / 264) / (32 / 263)
    se = math.sqrt(1 / 13 - 1 / 264 + 1 / 32 - 1 / 263)
    assert effect.measure == "risk_ratio" and effect.derivation == "published_arm_counts"
    assert effect.estimate == pytest.approx(ratio)
    assert effect.se_log == pytest.approx(se)
    assert effect.lower == pytest.approx(math.exp(math.log(ratio) - 1.959964 * se), rel=1e-5)
    assert effect.upper == pytest.approx(math.exp(math.log(ratio) + 1.959964 * se), rel=1e-5)
    assert effect.risk_control == pytest.approx(32 / 263)
    # Trials that found an effect have an interval that excludes 1; null trials include it.
    for trial in registry.trials.values():
        covers_one = trial.effect.lower <= 1.0 <= trial.effect.upper
        assert covers_one == (trial.finding == "null"), trial.trial_id


def test_margins_and_thresholds_are_proposed_not_final(registry, raw):
    assert registry.status == "proposed"
    for parameter in (registry.agreement_margin_ratio, registry.equivalence_margin_ratio,
                      registry.pooling, registry.discharge_alive_rule, registry.max_unresolved_share):
        assert parameter.status == "proposed"
    assert raw["feasibility_screen"]["status"] == "proposed"
    assert registry.agreement_margin_ratio.value > 1.0


def test_every_trial_states_representability_expected_bias_and_verification(registry):
    for trial in registry.trials.values():
        assert set(trial.representability) == {"eligibility", "exposure", "outcome"}
        assert set(trial.representability.values()) <= set(bm.REPRESENTABILITY)
        assert trial.expected_bias_sign in (-1, 1)
        assert trial.identifier_verified
    # The one published percentage that the counts do not reproduce is flagged, not hidden.
    assert registry.trials["hernandez_2022_very_high_risk"].unverified
    assert registry.trials["hernandez_2016_low_risk"].unverified == ()
    # HFNC against NIV: confounding by indication makes HFNC look better (downward).
    assert registry.trials["hernandez_2016_high_risk"].expected_bias_sign == -1


# ---------------------------------------------------------------------------
# registry: validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    [
        ("identifier",),
        ("identifier", "pmid"),
        ("outcome", "window_hours"),
        ("outcome", "event"),
        ("exposure", "treated"),
        ("exposure", "control"),
        ("arms_published",),
        ("arms_published", "control"),
        ("published_effect",),
        ("finding",),
        ("expected_bias",),
    ],
)
def test_entry_missing_identifier_window_arms_or_effect_is_rejected(raw, path):
    broken = copy.deepcopy(raw)
    node = broken["trials"]["high_wean_2019"]
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    with pytest.raises(bm.RegistryError, match="high_wean_2019"):
        bm.validate_registry(broken)


def test_malformed_values_are_rejected(raw):
    def broken(mutate):
        copy_ = copy.deepcopy(raw)
        mutate(copy_)
        return copy_

    cases = [
        lambda r: r["trials"]["high_wean_2019"]["identifier"].update(pmid="not-a-pmid"),
        lambda r: r["trials"]["high_wean_2019"]["outcome"].update(window_hours=0),
        lambda r: r["trials"]["high_wean_2019"]["outcome"].update(event="respiratory_failure"),
        lambda r: r["trials"]["high_wean_2019"]["exposure"].update(treated="hfnc"),      # same as control
        lambda r: r["trials"]["high_wean_2019"]["exposure"].update(treated="ecmo"),
        lambda r: r["trials"]["high_wean_2019"]["arms_published"]["treated"].update(events=400),
        lambda r: r["trials"]["high_wean_2019"].update(finding="positive"),
        lambda r: r["trials"]["high_wean_2019"]["eligibility"].update(rules=[{"kind": "sometimes"}]),
        lambda r: r["trials"]["high_wean_2019"]["eligibility"].update(
            rules=[{"kind": "any_true", "set": "no_such_set"}]),
        lambda r: r["agreement"]["agreement_margin_ratio"].update(value=0.9),
        lambda r: r["agreement"]["agreement_margin_ratio"].update(status="final"),
        lambda r: r["agreement"]["pooling"].update(value="vote_count"),
        lambda r: r["estimand"]["discharge_alive_rule"].update(value="ignore"),
        lambda r: r["adjustment"].update(covariates=["age_at_admission", "rescue_support"]),
        lambda r: r["pre_registration_audit"].update(trial="no_such_trial"),
        lambda r: r.update(trials={}),
    ]
    for mutate in cases:
        with pytest.raises(bm.RegistryError):
            bm.validate_registry(broken(mutate))


def test_published_percentage_or_effect_that_the_counts_do_not_reproduce_is_rejected(raw):
    wrong_percent = copy.deepcopy(raw)
    wrong_percent["trials"]["high_wean_2019"]["arms_published"]["treated"]["percent"] = 14.0
    with pytest.raises(bm.RegistryError, match="percent"):
        bm.validate_registry(wrong_percent)
    # The same mismatch is accepted only when it is declared unverified with a reason.
    declared = copy.deepcopy(wrong_percent)
    declared["trials"]["high_wean_2019"]["arms_published"]["treated"].update(
        unverified=True, unverified_reason="test")
    assert bm.validate_registry(declared).trials["high_wean_2019"].unverified

    wrong_effect = copy.deepcopy(raw)
    wrong_effect["trials"]["casey_2021_all_comers"]["published_effect"]["estimate"] = 2.0
    with pytest.raises(bm.RegistryError, match="published effect"):
        bm.validate_registry(wrong_effect)


def test_every_registered_trial_maps_to_cohort_arms_and_columns(registry, raw):
    from src.data.extubation_cohort import (
        build_extubation_cohort, load_extubation_config, table_availability,
    )
    from tests.fixtures_extubation import PARTITIONS, SPLIT_SEED, build_extubation_fixture

    config = load_extubation_config(ROOT / "configs/extubation.yaml")
    data_config = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    fixture = build_extubation_fixture(n_background=30)
    cohort = build_extubation_cohort(
        fixture.tables, config, availability=table_availability(config, data_config),
        split={"partitions": PARTITIONS, "split_seed": SPLIT_SEED}, site="mimic",
        episodes=fixture.episodes, calendar_periods=fixture.calendar_periods,
    ).cohort
    bm.check_registry_against_cohort(registry, cohort.columns, config["arms"]["order"])
    assert registry.columns_used() <= set(cohort.columns)

    with pytest.raises(bm.RegistryError, match="bmi_over_30"):
        bm.check_registry_against_cohort(
            registry, [c for c in cohort.columns if c != "bmi_over_30"], config["arms"]["order"])
    with pytest.raises(bm.RegistryError, match="niv"):
        bm.check_registry_against_cohort(registry, cohort.columns, ["conventional_oxygen", "hfnc"])


# ---------------------------------------------------------------------------
# agreement rule
# ---------------------------------------------------------------------------

def test_gap_is_the_log_ratio_of_emulation_to_benchmark(registry):
    trial = registry.trials["high_wean_2019"]
    gap = bm.trial_gap(trial, estimate(trial.effect.estimate * 1.2, 0.1))
    assert gap.log_gap == pytest.approx(math.log(1.2))
    assert gap.ratio_of_ratios == pytest.approx(1.2)
    assert gap.se == pytest.approx(math.hypot(0.1, trial.effect.se_log))


def test_gap_inside_the_margin_passes_and_outside_fails(registry):
    effect_trials = [t for t in registry.trials.values() if t.finding == "effect"]
    screens = {t.trial_id: feasible(registry, t.trial_id) for t in registry.trials.values()}
    evaluable = [t for t in effect_trials if screens[t.trial_id].evaluable]
    assert len(evaluable) >= 3

    def score(factor: float) -> bm.EstimatorAgreement:
        estimates = {"classical": {
            t.trial_id: estimate(t.effect.estimate * factor, 0.1) for t in registry.trials.values()
            if screens[t.trial_id].evaluable}}
        return bm.score_agreement(registry, estimates, screens).estimators["classical"]

    inside = score(1.2)
    assert inside.pooled_abs_gap == pytest.approx(math.log(1.2))
    assert inside.margin_log == pytest.approx(math.log(registry.agreement_margin_ratio.value))
    assert inside.passes is True
    assert {g.trial_id for g in inside.effect_gaps} == {t.trial_id for t in evaluable}
    outside = score(1.8)
    assert outside.pooled_abs_gap == pytest.approx(math.log(1.8))
    assert outside.passes is False
    assert inside.margin_status == "proposed"


def test_gaps_of_opposite_sign_do_not_cancel_under_the_absolute_pooling(registry):
    screens = {t: feasible(registry, t) for t in registry.trials}
    ids = [t.trial_id for t in registry.trials.values() if t.finding == "effect" and screens[t.trial_id].evaluable]
    factors = {trial_id: (2.0 if i % 2 == 0 else 0.5) for i, trial_id in enumerate(ids)}
    estimates = {"classical": {
        t.trial_id: estimate(t.effect.estimate * factors.get(t.trial_id, 1.0), 0.1)
        for t in registry.trials.values() if screens[t.trial_id].evaluable}}
    result = bm.score_agreement(registry, estimates, screens).estimators["classical"]
    assert result.pooled_abs_gap == pytest.approx(math.log(2.0))
    assert result.passes is False
    # The signed pooled gap is reported as a description and is much nearer zero.
    assert abs(result.pooled_signed_gap) < math.log(2.0)


def test_null_trial_is_reproduced_only_when_the_interval_lies_inside_the_equivalence_margin(registry):
    margin = registry.equivalence_margin_ratio.value
    narrow, wide = estimate(1.05, 0.05), estimate(1.05, 0.6)
    assert bm.reproduces_null(narrow, margin) is True
    assert bm.reproduces_null(wide, margin) is False          # same point estimate, wide interval
    assert bm.reproduces_null(estimate(1.45, 0.05), margin) is False   # upper limit crosses the margin

    screens = {t: feasible(registry, t) for t in registry.trials}
    base = {t.trial_id: estimate(t.effect.estimate, 0.1) for t in registry.trials.values()
            if screens[t.trial_id].evaluable}
    for name, null_estimate, reproduced in (("narrow", narrow, True), ("wide", wide, False)):
        estimates = {name: {**base, "casey_2021_all_comers": null_estimate,
                            "hernandez_2016_high_risk": narrow}}
        result = bm.score_agreement(registry, estimates, screens).estimators[name]
        by_trial = {r.trial_id: r for r in result.null_results}
        assert by_trial["casey_2021_all_comers"].reproduced is reproduced
        assert by_trial["hernandez_2016_high_risk"].reproduced is True
        assert result.n_null_reproduced == (2 if reproduced else 1)
        # Null trials are scored separately: they never enter the pooled gap.
        assert "casey_2021_all_comers" not in {g.trial_id for g in result.effect_gaps}
        assert result.passes is True


def test_trial_failing_the_feasibility_screen_is_excluded_from_every_estimator(registry):
    screens = {t: feasible(registry, t) for t in registry.trials}
    screens["high_wean_2019"] = feasible(registry, "high_wean_2019", n_treated=12, ess_treated=9.0)
    assert not screens["high_wean_2019"].evaluable
    assert any("arm size" in reason for reason in screens["high_wean_2019"].reasons)
    # Both estimators supply an estimate for the failing trial; neither may be scored on it.
    estimates = {
        name: {t.trial_id: estimate(t.effect.estimate * factor, 0.1) for t in registry.trials.values()}
        for name, factor in (("classical", 1.1), ("model", 1.3))
    }
    report = bm.score_agreement(registry, estimates, screens)
    assert "high_wean_2019" in report.not_evaluable
    assert "high_wean_2019" not in report.evaluable
    for result in report.estimators.values():
        scored = {g.trial_id for g in result.effect_gaps} | {r.trial_id for r in result.null_results}
        assert "high_wean_2019" not in scored
        assert scored == set(report.evaluable)
    # The excluded trial would have moved the pooled gap had it been scored.
    estimates["model"]["high_wean_2019"] = estimate(50.0, 0.1)
    again = bm.score_agreement(registry, estimates, screens)
    assert again.estimators["model"].pooled_abs_gap == pytest.approx(report.estimators["model"].pooled_abs_gap)


def test_feasibility_screen_checks_each_registered_threshold(registry):
    trial_id = "hernandez_2016_low_risk"
    assert feasible(registry, trial_id).evaluable
    for changes, word in (
        (dict(n_control=20), "arm size"),
        (dict(share_below_min_probability=0.6), "overlap"),
        (dict(ess_treated=10.0), "effective sample size"),
        (dict(ess_treated=60.0, ess_control=60.0), "precision"),
        (dict(estimators_runnable={"classical": True, "model": False}), "model"),
    ):
        result = feasible(registry, trial_id, **changes)
        assert not result.evaluable, changes
        assert any(word in reason for reason in result.reasons), (changes, result.reasons)
    # A null trial needs enough precision for its interval to fit inside the margin.
    null_screen = feasible(registry, "casey_2021_all_comers", ess_treated=80.0, ess_control=80.0)
    assert not null_screen.evaluable and any("precision" in r for r in null_screen.reasons)
    # An outcome that cannot be represented fails the screen whatever the sample size.
    ferrer = feasible(registry, "ferrer_2009_hypercapnic")
    assert not ferrer.evaluable and any("outcome" in r for r in ferrer.reasons)


def test_a_trial_one_estimator_could_not_estimate_is_not_evaluable_for_any(registry):
    screens = {t: feasible(registry, t) for t in registry.trials}
    full = {t.trial_id: estimate(t.effect.estimate, 0.1) for t in registry.trials.values()
            if screens[t.trial_id].evaluable}
    partial = {k: v for k, v in full.items() if k != "high_wean_2019"}
    report = bm.score_agreement(registry, {"classical": full, "model": partial}, screens)
    assert "high_wean_2019" in report.not_evaluable
    for result in report.estimators.values():
        assert "high_wean_2019" not in {g.trial_id for g in result.effect_gaps}


def test_scoring_fails_closed_on_missing_screens_and_unknown_trials(registry):
    screens = {t: feasible(registry, t) for t in registry.trials}
    estimates = {"classical": {"high_wean_2019": estimate(0.7, 0.1)}}
    with pytest.raises(bm.AgreementError, match="feasibility"):
        bm.score_agreement(registry, estimates, {})
    with pytest.raises(bm.AgreementError, match="unknown trial"):
        bm.score_agreement(registry, {"classical": {"made_up": estimate(0.7, 0.1)}}, screens)
    with pytest.raises(bm.AgreementError):
        bm.score_agreement(registry, {}, screens)
    with pytest.raises(ValueError):
        bm.TrialEstimate(log_effect=0.0, se=-1.0)
    with pytest.raises(ValueError):
        bm.TrialEstimate(log_effect=float("nan"), se=0.1)


def test_no_evaluable_effect_trial_gives_no_verdict_instead_of_a_pass(registry):
    screens = {t: feasible(registry, t, n_treated=5) for t in registry.trials}
    report = bm.score_agreement(registry, {"classical": {}}, screens)
    assert report.evaluable == ()
    assert report.estimators["classical"].passes is None
    assert report.estimators["classical"].pooled_abs_gap is None


# ---------------------------------------------------------------------------
# paired comparison between estimators (R27)
# ---------------------------------------------------------------------------

def _influence(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    values = rng.normal(0.0, scale, n)
    return values - values.mean()


def test_paired_comparison_of_two_identical_estimators_is_centred_on_zero():
    rng = np.random.default_rng(0)
    n = 400
    influence = _influence(rng, n, 3.0)
    estimator = bm.linearized_estimator(math.log(0.8), influence)
    pair = bm.PairedTrial("t", estimator, estimator, benchmark_log_effect=math.log(0.65), members=np.arange(n))
    result = bm.paired_gap_difference([pair], n_boot=300, seed=1)
    assert result.difference == 0.0
    assert result.lower == 0.0 and result.upper == 0.0        # identical on every resample
    assert result.n_boot == 300 and result.n_failed == 0

    # Same accuracy, independent noise: the interval is wide but still covers zero.
    other = bm.linearized_estimator(math.log(0.8), _influence(rng, n, 3.0))
    noisy = bm.paired_gap_difference(
        [bm.PairedTrial("t", estimator, other, math.log(0.65), np.arange(n))], n_boot=400, seed=2)
    assert noisy.difference == pytest.approx(0.0)
    assert noisy.lower < 0.0 < noisy.upper


def test_paired_comparison_detects_the_estimator_that_is_nearer_the_benchmark():
    rng = np.random.default_rng(3)
    n = 500
    shared = _influence(rng, n, 2.0)
    near = bm.linearized_estimator(math.log(0.70), shared)
    far = bm.linearized_estimator(math.log(1.10), shared + _influence(rng, n, 0.5))
    benchmark = math.log(0.65)
    result = bm.paired_gap_difference(
        [bm.PairedTrial("t", far, near, benchmark, np.arange(n))], n_boot=400, seed=4)
    expected = abs(math.log(1.10) - benchmark) - abs(math.log(0.70) - benchmark)
    assert result.difference == pytest.approx(expected)
    assert result.lower > 0.0                                  # `far` has the larger gap
    assert result.per_trial["t"].difference == pytest.approx(expected)


def test_paired_bootstrap_uses_identical_patients_for_both_estimators_and_across_trials():
    seen: dict[str, list[np.ndarray]] = {"a": [], "b": [], "c": []}

    def recorder(name: str):
        def run(index: np.ndarray) -> float:
            seen[name].append(np.array(index))
            return 0.1 * float(np.mean(index))
        return run

    members_1, members_2 = np.arange(0, 30), np.arange(20, 50)     # trials share patients 20-29
    pairs = [
        bm.PairedTrial("t1", recorder("a"), recorder("b"), 0.0, members_1),
        bm.PairedTrial("t2", recorder("c"), recorder("c"), 0.0, members_2),
    ]
    bm.paired_gap_difference(pairs, n_boot=5, seed=0)
    assert len(seen["a"]) == len(seen["b"]) == 6                # the full sample, then 5 resamples
    for index_a, index_b in zip(seen["a"], seen["b"], strict=True):
        assert np.array_equal(index_a, index_b)
    assert np.array_equal(seen["a"][0], np.arange(30))
    # A shared patient is drawn the same number of times in both trials.
    for index_1, index_2 in zip(seen["a"][1:], seen["c"][2::2], strict=True):
        counts_1 = np.bincount(index_1, minlength=30)[20:30]     # union patients 20-29
        counts_2 = np.bincount(index_2, minlength=30)[0:10]
        assert np.array_equal(counts_1, counts_2)


def test_paired_comparison_rejects_bad_input():
    estimator = bm.linearized_estimator(0.0, np.zeros(10))
    with pytest.raises(ValueError):
        bm.paired_gap_difference([], n_boot=10)
    with pytest.raises(ValueError):
        bm.paired_gap_difference([bm.PairedTrial("t", estimator, estimator, 0.0, np.arange(10))], n_boot=0)
    with pytest.raises(ValueError):
        bm.linearized_estimator(0.0, np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# packaging
# ---------------------------------------------------------------------------

def test_benchmark_and_simulation_modules_stay_inside_the_site_package_dependencies():
    package = ROOT / "src/eval/causal"
    allowed = set(sys.stdlib_module_names) | {"numpy", "sklearn", "yaml"}
    for name in ("benchmark.py", "simulation.py"):
        for node in ast.walk(ast.parse((package / name).read_text())):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    assert node.level == 1, f"{name}: relative import leaves the package"
                    continue
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                assert root in allowed, f"{name} imports {root!r}"
