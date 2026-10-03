"""Per-trial emulation and its gates (plan U12; R8, R14, R25, R28, KTD9).

Synthetic data only: the cohort and labels come from `tests.fixtures_extubation` through
the real cohort builder and labeler. Nothing here reads a real site.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from src.data.extubation_cohort import (
    build_extubation_cohort,
    load_extubation_config,
    table_availability,
)
from src.eval.causal import benchmark as bm
from src.eval.causal import emulate as em
from src.eval.causal.estimators import NuisanceConfig
from src.eval.extubation_labeler import build_extubation_labels
from tests.fixtures_extubation import PARTITIONS, SPLIT_SEED, build_extubation_fixture

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/extubation_benchmarks.yaml"
COHORT_CONFIG_PATH = ROOT / "configs/extubation.yaml"
PROTOCOL_HASH = hashlib.sha256(b"synthetic registered protocol").hexdigest()
NUISANCE = NuisanceConfig(n_folds=3, seed=0, max_iter=40, max_depth=2)
# The fixture's planted risk factors (it has no prolonged-ventilation effect).
PLANTED_FACTORS = ["age_over_65", "bmi_over_30", "hypercapnia", "comorbidity_copd",
                   "comorbidity_heart_failure"]
PUBLISHED = {
    "arms_published": {"treated": {"events": 60, "n": 200, "percent": 30.0},
                       "control": {"events": 100, "n": 200, "percent": 50.0}},
    "published_effect": {"measure": "absolute_risk_difference", "orientation": "treated_minus_control",
                         "estimate": -0.20, "ci_lower": -0.29, "ci_upper": -0.11, "verified": True},
}


def synthetic_trial(label, treated, control, rules, window=168, event="reintubation_or_death"):
    return {
        "label": label,
        "identifier": {"pmid": "00000000", "citation": "synthetic", "verified": True},
        "finding": "effect",
        "approximate": False,
        "eligibility": {"base": "eligible", "representability": "representable", "rules": rules,
                        "representable": [], "proxied": [], "not_representable": []},
        "exposure": {"assignment_column": "arm", "treated": treated, "control": control,
                     "representability": "representable"},
        "outcome": {"trial_primary": "synthetic", "event": event, "window_hours": window,
                    "representability": "representable"},
        **copy.deepcopy(PUBLISHED),
        "expected_bias": {"direction": "upward", "reason": "synthetic"},
    }


@pytest.fixture(scope="module")
def registry() -> bm.Registry:
    """The real registry plus synthetic trials that match the fixture's planted structure."""
    raw = yaml.safe_load(REGISTRY_PATH.read_text())
    raw["risk_factor_sets"]["planted"] = PLANTED_FACTORS
    at_risk = [{"kind": "count_at_least", "set": "planted", "value": 1}]
    low_risk = [{"kind": "count_at_most", "set": "planted", "value": 0}]
    raw["trials"].update(
        syn_high_risk_niv_vs_hfnc=synthetic_trial("NIV vs HFNC, any risk factor", "niv", "hfnc", at_risk),
        syn_high_risk_niv_vs_oxygen=synthetic_trial(
            "NIV vs oxygen, any risk factor", "niv", "conventional_oxygen", at_risk),
        syn_low_risk_niv_vs_oxygen=synthetic_trial(
            "NIV vs oxygen, no risk factor", "niv", "conventional_oxygen", low_risk),
        syn_low_risk_72h=synthetic_trial(
            "HFNC vs oxygen, no risk factor, 72 h", "hfnc", "conventional_oxygen", low_risk,
            window=72, event="reintubation"),
    )
    return bm.validate_registry(raw)


class Site:
    def __init__(self, name: str, n_background: int, seed: int) -> None:
        self.config = load_extubation_config(COHORT_CONFIG_PATH)
        data_config = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
        cohort_config = yaml.safe_load((ROOT / "configs/cohort.yaml").read_text())
        availability = table_availability(self.config, data_config)
        self.fixture = build_extubation_fixture(n_background=n_background, seed=seed, outcome_scenarios=True)
        built = build_extubation_cohort(
            self.fixture.tables, self.config, availability=availability,
            split={"partitions": PARTITIONS, "split_seed": SPLIT_SEED}, site=name,
            episodes=self.fixture.episodes, calendar_periods=self.fixture.calendar_periods,
        )
        labels = build_extubation_labels(
            built.cohort, self.fixture.tables, self.config, availability=availability,
            data_config=data_config, cohort_config=cohort_config,
        )
        self.data = em.SiteData(cohort=built.cohort, labels=labels, device_rows=built.device_rows)


@pytest.fixture(scope="module")
def mimic() -> Site:
    return Site("mimic", 5000, 7)


@pytest.fixture(scope="module")
def rush() -> Site:
    return Site("rush", 1500, 11)


def manifest(local: dict[str, str], **overrides) -> dict[str, str]:
    full = {key: hashlib.sha256(key.encode()).hexdigest() for key in
            ("model_checkpoint", "vocabulary", "hyperparameters", "thresholds")}
    full.update(local)
    full.update(overrides)
    return full


@pytest.fixture(scope="module")
def local_hashes() -> dict[str, str]:
    return em.local_freeze_hashes(cohort_config=COHORT_CONFIG_PATH, registry_path=REGISTRY_PATH)


def run(registry, sites, trial_id, **kwargs):
    kwargs.setdefault("protocol_hash", PROTOCOL_HASH)
    kwargs.setdefault("config", NUISANCE)
    return em.run_trial_emulation(sites, registry, trial_id, **kwargs)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def test_outcome_by_arm_run_without_a_protocol_hash_is_refused(registry, mimic):
    with pytest.raises(em.EmulationRefused, match="protocol"):
        em.authorize_outcome_by_arm(registry, trial_id="high_wean_2019", sites=["mimic"])
    with pytest.raises(em.EmulationRefused, match="protocol"):
        run(registry, {"mimic": mimic.data}, "syn_high_risk_niv_vs_hfnc", protocol_hash=None)
    for bad in ("", "abc", "z" * 64, PROTOCOL_HASH.upper() + "0"):
        with pytest.raises(em.EmulationRefused, match="protocol"):
            em.authorize_outcome_by_arm(registry, trial_id="high_wean_2019", sites=["mimic"], protocol_hash=bad)
    granted = em.authorize_outcome_by_arm(
        registry, trial_id="high_wean_2019", sites=["mimic"], protocol_hash=PROTOCOL_HASH)
    assert granted.basis == "registered_protocol" and granted.protocol_hash == PROTOCOL_HASH


def test_casey_all_comer_comparison_is_the_only_run_allowed_before_registration(registry, local_hashes):
    granted = em.authorize_outcome_by_arm(registry, trial_id="casey_2021_all_comers", sites=["mimic"])
    assert granted.basis == "pre_registration_audit" and granted.protocol_hash is None
    # Only that trial, only as registered, only on an exploratory site, only one site.
    for trial_id in set(registry.trials) - {"casey_2021_all_comers"}:
        with pytest.raises(em.EmulationRefused):
            em.authorize_outcome_by_arm(registry, trial_id=trial_id, sites=["mimic"])
    with pytest.raises(em.EmulationRefused, match="as registered"):
        em.authorize_outcome_by_arm(
            registry, trial_id="casey_2021_all_comers", sites=["mimic"], modified=True)
    for sites in (["rush"], ["mimic", "rush"], ["uchicago"], ["unlisted_site"]):
        with pytest.raises(em.EmulationRefused):
            em.authorize_outcome_by_arm(registry, trial_id="casey_2021_all_comers", sites=sites)
    # Even with a full freeze, a confirmatory site needs the registered protocol.
    with pytest.raises(em.EmulationRefused, match="protocol"):
        em.authorize_outcome_by_arm(
            registry, trial_id="casey_2021_all_comers", sites=["rush"],
            freeze_manifest=manifest(local_hashes), local_hashes=local_hashes)


def test_confirmatory_site_is_refused_without_the_freeze_hashes(registry, rush, local_hashes):
    with pytest.raises(em.EmulationRefused, match="freeze"):
        run(registry, {"rush": rush.data}, "syn_high_risk_niv_vs_oxygen")
    complete = manifest(local_hashes)
    for missing in registry.freeze_required:
        partial = {k: v for k, v in complete.items() if k != missing}
        with pytest.raises(em.EmulationRefused, match=missing):
            em.authorize_outcome_by_arm(
                registry, trial_id="high_wean_2019", sites=["rush"], protocol_hash=PROTOCOL_HASH,
                freeze_manifest=partial, local_hashes=local_hashes)
    with pytest.raises(em.EmulationRefused, match="vocabulary"):
        em.authorize_outcome_by_arm(
            registry, trial_id="high_wean_2019", sites=["rush"], protocol_hash=PROTOCOL_HASH,
            freeze_manifest=manifest(local_hashes, vocabulary="not-a-hash"), local_hashes=local_hashes)
    # A hash that can be recomputed on the node must match what was frozen.
    for key in registry.freeze_locally_verified:
        stale = manifest(local_hashes, **{key: "0" * 64})
        with pytest.raises(em.EmulationRefused, match=key):
            em.authorize_outcome_by_arm(
                registry, trial_id="high_wean_2019", sites=["rush"], protocol_hash=PROTOCOL_HASH,
                freeze_manifest=stale, local_hashes=local_hashes)
    # Without the local recomputation the gate cannot verify, so it refuses.
    with pytest.raises(em.EmulationRefused, match="recomputed"):
        em.authorize_outcome_by_arm(
            registry, trial_id="high_wean_2019", sites=["rush"], protocol_hash=PROTOCOL_HASH,
            freeze_manifest=complete)
    granted = em.authorize_outcome_by_arm(
        registry, trial_id="high_wean_2019", sites=["rush"], protocol_hash=PROTOCOL_HASH,
        freeze_manifest=complete, local_hashes=local_hashes)
    assert granted.freeze_verified and granted.roles == {"rush": "confirmatory"}
    # A site the registry does not list is treated as confirmatory.
    with pytest.raises(em.EmulationRefused, match="freeze"):
        em.authorize_outcome_by_arm(
            registry, trial_id="high_wean_2019", sites=["somewhere"], protocol_hash=PROTOCOL_HASH)


def test_local_freeze_hashes_follow_the_files(tmp_path, local_hashes):
    assert set(local_hashes) == {"cohort_definition", "estimator_code", "benchmark_registry"}
    assert local_hashes["cohort_definition"] == hashlib.sha256(COHORT_CONFIG_PATH.read_bytes()).hexdigest()
    assert local_hashes["benchmark_registry"] == hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    package = tmp_path / "causal"
    package.mkdir()
    (package / "estimators.py").write_text("A = 1\n")
    before = em.local_freeze_hashes(
        cohort_config=COHORT_CONFIG_PATH, registry_path=REGISTRY_PATH, package_dir=package)
    (package / "estimators.py").write_text("A = 2\n")
    after = em.local_freeze_hashes(
        cohort_config=COHORT_CONFIG_PATH, registry_path=REGISTRY_PATH, package_dir=package)
    assert before["estimator_code"] != after["estimator_code"]
    assert before["cohort_definition"] == after["cohort_definition"]


def test_emulation_needs_an_authorization_that_covers_the_trial_and_site(registry, mimic):
    granted = em.authorize_outcome_by_arm(
        registry, trial_id="syn_high_risk_niv_vs_oxygen", sites=["mimic"], protocol_hash=PROTOCOL_HASH)
    with pytest.raises(em.EmulationRefused):
        em.emulate_trial({"mimic": mimic.data}, registry, "syn_high_risk_niv_vs_hfnc",
                         authorization=granted, config=NUISANCE)
    with pytest.raises(em.EmulationRefused):
        em.emulate_trial({"rush": mimic.data}, registry, "syn_high_risk_niv_vs_oxygen",
                         authorization=granted, config=NUISANCE)
    with pytest.raises(em.EmulationRefused):
        em.emulate_trial({"mimic": mimic.data}, registry, "syn_high_risk_niv_vs_oxygen",
                         authorization=None, config=NUISANCE)
    # The audit exception covers the trial's own outcome and the registered rule only.
    audit = em.authorize_outcome_by_arm(registry, trial_id="casey_2021_all_comers", sites=["mimic"])
    with pytest.raises(em.EmulationRefused, match="as registered"):
        em.emulate_trial({"mimic": mimic.data}, registry, "casey_2021_all_comers",
                         authorization=audit, config=NUISANCE, outcome="study_primary")
    with pytest.raises(em.EmulationRefused, match="as registered"):
        em.emulate_trial({"mimic": mimic.data}, registry, "casey_2021_all_comers",
                         authorization=audit, config=NUISANCE, discharge_alive_rule="censor")


# ---------------------------------------------------------------------------
# eligibility, exposure and window per trial
# ---------------------------------------------------------------------------

def test_two_trials_select_different_patients_each_with_its_own_window(registry, mimic):
    cohort = mimic.data.cohort
    low = em.eligibility_mask(cohort, registry, "syn_low_risk_72h").to_numpy()
    high = em.eligibility_mask(cohort, registry, "syn_high_risk_niv_vs_hfnc").to_numpy()
    eligible = cohort["eligible"].to_numpy()
    assert low.sum() > 50 and high.sum() > 50
    assert not np.any(low & high)
    assert np.array_equal(low | high, eligible)                 # at most 0 / at least 1 partition it
    assert not np.any(low & ~eligible)

    low_arrays = em.trial_arrays(mimic.data, registry, "syn_low_risk_72h")
    high_arrays = em.trial_arrays(mimic.data, registry, "syn_high_risk_niv_vs_hfnc")
    assert (low_arrays.horizon, low_arrays.event) == (72.0, "reintubation")
    assert (high_arrays.horizon, high_arrays.event) == (168.0, "reintubation_or_death")
    assert low_arrays.n_eligible == low.sum() and high_arrays.n_eligible == high.sum()
    assert low_arrays.event_time.max() <= 72.0 < high_arrays.event_time.max() <= 168.0
    # A reintubation after 72 h is an event in the 7-day window and not in the 72-hour one.
    name = mimic.fixture.scenarios["out_reintubated_80h"]
    position = cohort["patient_id"].to_list().index(name)
    assert eligible[position]
    seven_day = em.trial_arrays(mimic.data, registry, "syn_low_risk_72h", outcome="study_primary")
    assert seven_day.horizon == 168.0
    index_72 = list(np.flatnonzero(low)).index(position)
    assert low_arrays.event_type[index_72] == 0 and seven_day.event_type[index_72] == 1


def test_registered_trials_apply_their_own_rules_and_assignment_column(registry, mimic):
    cohort = mimic.data.cohort
    masks = {t: em.eligibility_mask(cohort, registry, t).to_numpy() for t in registry.trials}
    eligible = cohort["eligible"].to_numpy()
    assert np.array_equal(masks["casey_2021_all_comers"], eligible)
    assert not np.any(masks["hernandez_2016_low_risk"] & masks["hernandez_2016_high_risk"])
    assert masks["hernandez_2022_very_high_risk"].sum() < masks["hernandez_2016_high_risk"].sum()
    assert not np.any(masks["hernandez_2022_very_high_risk"] & ~masks["hernandez_2016_high_risk"])
    hypercapnic = cohort["hypercapnia"].fill_null(False).to_numpy()
    assert not np.any(masks["ferrer_2009_hypercapnic"] & ~hypercapnic)
    # A null risk factor counts as absent, so a patient with no PaCO2 can be low risk.
    unknown = cohort["hypercapnia"].is_null().to_numpy()
    assert np.any(masks["hernandez_2016_low_risk"] & unknown)
    # HIGH-WEAN reads "any NIV in the grace window"; the others read the first device.
    high_wean = em.trial_arrays(mimic.data, registry, "high_wean_2019")
    expected = cohort.filter(pl.Series(masks["high_wean_2019"]))["arm_highest_support"].to_numpy()
    assert np.array_equal(high_wean.arm, expected)
    assert set(high_wean.covariate_names) >= set(registry.covariates)


def test_eligibility_counts_are_outcome_blind_and_suppressed(registry, mimic, rush):
    counts = em.eligibility_counts(mimic.data.cohort, registry)
    assert set(counts) == set(registry.trials)
    casey = counts["casey_2021_all_comers"]
    assert casey["eligible"] == int(mimic.data.cohort["eligible"].sum())
    assert casey["treated"] + casey["control"] + casey["other_arm"] == casey["eligible"]
    small = em.eligibility_counts(rush.data.cohort, registry, min_cell=10_000)
    assert all(value == "<10000" for cells in small.values() for value in cells.values())
    # The cohort alone is enough: no label frame is passed, so no outcome can be read.
    with pytest.raises(bm.RegistryError):
        em.eligibility_counts(mimic.data.cohort.drop("hypercapnia"), registry)


def test_cohort_and_labels_must_describe_the_same_rows(registry, mimic):
    shuffled = em.SiteData(mimic.data.cohort, mimic.data.labels.reverse(), mimic.data.device_rows)
    with pytest.raises(ValueError, match="same rows"):
        em.trial_arrays(shuffled, registry, "syn_high_risk_niv_vs_hfnc")
    with pytest.raises(KeyError):
        em.trial_arrays(mimic.data, registry, "no_such_trial")


# ---------------------------------------------------------------------------
# integration on the synthetic fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def high_risk(registry, mimic):
    return run(registry, {"mimic": mimic.data}, "syn_high_risk_niv_vs_oxygen")["mimic"]


def test_high_risk_emulation_recovers_the_planted_benefit_direction(registry, mimic, high_risk):
    truth = mimic.fixture.truth.filter(pl.col("n_risk_factors") >= 1)
    planted = truth["p_event_niv"].mean() / truth["p_event_conventional_oxygen"].mean()
    assert planted < 0.75
    primary = high_risk.estimates[high_risk.primary]
    assert high_risk.primary == "clone_censor_weight"
    assert primary.risk_ratio.upper < 1.0                       # NIV benefit, interval excludes 1
    assert abs(math.log(primary.risk_ratio.estimate) - math.log(planted)) < 0.35
    assert primary.risk_difference.estimate < 0.0
    # Sensitivity analyses run on the same patients and agree on direction.
    assert set(high_risk.estimates) == {"clone_censor_weight", "point_treatment_aipw", "overlap_weights"}
    for result in high_risk.estimates.values():
        assert result.risk_ratio.estimate < 1.0
    # NIV against HFNC in the same patients: NIV still better, by less.
    versus_hfnc = run(registry, {"mimic": mimic.data}, "syn_high_risk_niv_vs_hfnc")["mimic"]
    assert versus_hfnc.estimates["clone_censor_weight"].risk_ratio.estimate < 1.0
    assert (versus_hfnc.estimates["clone_censor_weight"].risk_ratio.estimate
            > primary.risk_ratio.estimate)


def test_low_risk_contrast_shows_no_benefit(registry, mimic, high_risk):
    low = run(registry, {"mimic": mimic.data}, "syn_low_risk_niv_vs_oxygen")["mimic"]
    ratio = low.estimates[low.primary].risk_ratio
    assert ratio.lower <= 1.0 <= ratio.upper                    # planted: no NIV effect without a risk factor
    assert low.arm_sizes["treated"] < high_risk.arm_sizes["treated"]
    assert ratio.estimate > high_risk.estimates["clone_censor_weight"].risk_ratio.estimate


def test_result_reports_arm_sizes_diagnostics_and_only_aggregates(high_risk):
    sizes = high_risk.arm_sizes
    assert sizes["treated"] + sizes["control"] + sizes["other_arm"] == high_risk.n_analysed
    assert high_risk.n_eligible == high_risk.n_analysed + high_risk.n_unresolved
    diagnostics = high_risk.diagnostics
    assert 1.0 < diagnostics["effective_sample_size"]["treated"] <= sizes["treated"]
    assert 0.0 <= diagnostics["overlap_share_below_min_probability"]["treated"] <= 1.0
    assert diagnostics["balance_max_abs_smd"]["treated"]["weighted"] < diagnostics[
        "balance_max_abs_smd"]["treated"]["unweighted"]
    assert diagnostics["e_value"]["point"] > 1.0
    assert 0.0 < high_risk.covariate_coverage["paco2_mmhg"] < 1.0
    aggregate = high_risk.to_aggregate()
    json.dumps(aggregate)                                        # plain numbers and strings only
    text = json.dumps(aggregate)
    assert "SYN-PT" not in text and "SYN-HOSP" not in text
    assert aggregate["estimates"]["clone_censor_weight"]["risk_ratio"]["estimate"] < 1.0
    assert aggregate["margins_status"] == "proposed"
    assert high_risk.influence_log_rr["clone_censor_weight"].shape == (high_risk.n_analysed,)
    suppressed = high_risk.to_aggregate(min_cell=10**6)
    assert suppressed["arm_sizes"]["treated"] == "<1000000"
    assert suppressed["estimates"] == "suppressed"


def test_per_site_and_pooled_runs_both_work(registry, mimic, rush, local_hashes):
    sites = {"mimic": mimic.data, "rush": rush.data}
    results = run(registry, sites, "syn_high_risk_niv_vs_oxygen",
                  freeze_manifest=manifest(local_hashes), local_hashes=local_hashes)
    assert set(results) == {"mimic", "rush", "pooled"}
    pooled = results["pooled"]
    assert pooled.sites == ("mimic", "rush")
    assert pooled.n_eligible == results["mimic"].n_eligible + results["rush"].n_eligible
    for key in ("treated", "control", "other_arm"):
        assert pooled.arm_sizes[key] == results["mimic"].arm_sizes[key] + results["rush"].arm_sizes[key]
    assert "site_rush" in pooled.covariate_names and "site_rush" not in results["mimic"].covariate_names
    ratios = {name: r.estimates["clone_censor_weight"].risk_ratio for name, r in results.items()}
    assert all(ratio.estimate < 1.0 for ratio in ratios.values())
    assert ratios["pooled"].se < ratios["rush"].se               # more patients, tighter interval
    # A single-site call returns that site alone; pooling can be switched off.
    alone = run(registry, sites, "syn_high_risk_niv_vs_oxygen", pooled=False,
                freeze_manifest=manifest(local_hashes), local_hashes=local_hashes)
    assert set(alone) == {"mimic", "rush"}


def test_unresolved_rows_are_counted_and_handled_by_the_registered_rule(registry, mimic):
    """Under `event_free`, discharge alive is event-free and only an unknown disposition is
    unresolved; under `censor`, a discharge alive before the horizon is censored."""
    sites = {"mimic": mimic.data}
    arrays = em.trial_arrays(mimic.data, registry, "casey_2021_all_comers", outcome="study_primary")
    assert arrays.discharge_alive_rule == "event_free"
    assert arrays.n_unresolved == int((~arrays.resolved).sum()) >= 1   # out_discharge_unknown
    censored = em.trial_arrays(
        mimic.data, registry, "casey_2021_all_comers", outcome="study_primary", discharge_alive_rule="censor")
    assert censored.n_unresolved > arrays.n_unresolved            # out_discharged_day2_alive, ...

    result = run(registry, sites, "casey_2021_all_comers", outcome="study_primary")["mimic"]
    assert result.n_unresolved == arrays.n_unresolved
    assert result.n_analysed == result.n_eligible - result.n_unresolved
    assert result.unresolved_handling == "excluded_and_reported"
    assert 0.0 < result.unresolved_share < registry.max_unresolved_share.value

    # Under `censor` the doubly robust step cannot use censored rows: every row is kept
    # and the weighted Aalen-Johansen estimate is reported instead.
    kept = run(registry, sites, "casey_2021_all_comers", outcome="study_primary",
               discharge_alive_rule="censor", n_boot=40)["mimic"]
    assert kept.n_analysed == kept.n_eligible and kept.n_unresolved == censored.n_unresolved
    assert kept.unresolved_handling == "censored_at_discharge"
    assert kept.primary == "clone_censor_weight_aalen_johansen"
    assert set(kept.estimates) == {"clone_censor_weight_aalen_johansen"}
    ratio = kept.estimates[kept.primary].risk_ratio
    assert 0.0 < ratio.lower < ratio.estimate < ratio.upper
    # With almost no censoring the two analyses agree closely.
    assert abs(math.log(ratio.estimate) - math.log(result.estimates["clone_censor_weight"].risk_ratio.estimate)) < 0.15


def test_too_many_unresolved_rows_refuse_instead_of_estimating_on_the_rest(registry, mimic):
    labels = mimic.data.labels.with_columns(
        pl.when(pl.col("followup_end_reason") == "discharge_alive")
        .then(pl.lit("discharge_unknown")).otherwise(pl.col("followup_end_reason"))
        .alias("followup_end_reason"),
        (pl.col("followup_end_hours") * 0.2).alias("followup_end_hours"),
        (pl.col("discharge_hours") * 0.2).alias("discharge_hours"),
    )
    site = em.SiteData(mimic.data.cohort, labels, mimic.data.device_rows)
    with pytest.raises(em.UnresolvedFollowUp, match="unresolved"):
        run(registry, {"mimic": site}, "syn_high_risk_niv_vs_oxygen")


def test_unknown_rule_outcome_and_empty_population_are_rejected(registry, mimic):
    sites = {"mimic": mimic.data}
    with pytest.raises(ValueError, match="discharge_alive_rule"):
        run(registry, sites, "syn_high_risk_niv_vs_oxygen", discharge_alive_rule="drop")
    with pytest.raises(ValueError, match="outcome"):
        run(registry, sites, "syn_high_risk_niv_vs_oxygen", outcome="mortality")
    nobody = em.SiteData(
        mimic.data.cohort.with_columns(pl.lit(False).alias("eligible")), mimic.data.labels,
        mimic.data.device_rows)
    with pytest.raises(ValueError, match="no eligible"):
        run(registry, {"mimic": nobody}, "syn_high_risk_niv_vs_oxygen")
    with pytest.raises(ValueError):
        run(registry, {}, "syn_high_risk_niv_vs_oxygen")


# ---------------------------------------------------------------------------
# feasibility measurement feeds the screen without outcomes
# ---------------------------------------------------------------------------

def test_feasibility_is_measured_without_outcomes_and_feeds_the_screen(registry, mimic):
    cohort_only = em.SiteData(mimic.data.cohort, None, mimic.data.device_rows)
    big = em.measure_feasibility(cohort_only, registry, "syn_high_risk_niv_vs_oxygen", config=NUISANCE)
    assert big.n_treated > 300 and 1.0 < big.ess_treated <= big.n_treated
    assert bm.feasibility_screen(registry, "syn_high_risk_niv_vs_oxygen", big).evaluable
    small = em.measure_feasibility(cohort_only, registry, "syn_low_risk_niv_vs_oxygen", config=NUISANCE)
    assert small.n_treated < 50
    screen = bm.feasibility_screen(registry, "syn_low_risk_niv_vs_oxygen", small)
    assert not screen.evaluable and any("arm size" in reason for reason in screen.reasons)
