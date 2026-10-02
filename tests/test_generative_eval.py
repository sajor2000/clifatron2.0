"""Tests for the G3 generative evals (src/eval/generative.py)."""

from __future__ import annotations

import json
from collections import Counter

import polars as pl
import pytest

from src.eval.generative import (
    concept_groups,
    evaluate_rollout_mortality,
    observed_gem_outcome,
    rollout_mortality,
    wilson_interval,
    distinct_n,
    distance_to_observed,
    evaluate,
    event_rate_calibration,
    js_divergence,
    key_event_recall,
    rollout_hygiene,
    token_concept,
    top_k_overlap,
)

GROUPS = concept_groups({
    "target_concepts": [
        {"name": "spo2", "source": "vitals"},
        {"name": "lactate", "source": "labs"},
    ]
})


# ------------------------------------------------------------------ primitives


def test_token_concept_fused_and_bare():
    assert token_concept("spo2=10") == "spo2"
    assert token_concept("norepinephrine") == "norepinephrine"


def test_js_divergence_identical_zero_disjoint_one():
    a = Counter(["spo2=10", "spo2=10", "norepinephrine"])
    assert js_divergence(a, Counter(["spo2=10", "spo2=10", "norepinephrine"])) == 0.0
    assert js_divergence(a, Counter(["lactate=3", "lactate=4"])) == 1.0


def test_top_k_overlap():
    real = Counter({"dbp": 10, "spo2=10": 8, "map=9": 6})
    gen = Counter({"dbp": 9, "spo2=10": 7, "norepinephrine": 5})
    assert top_k_overlap(real, gen, 2) == 1.0  # same top-2
    assert top_k_overlap(real, gen, 3) == pytest.approx(2 / 3)


# ------------------------------------------------------------------- metrics


def test_event_rate_calibration_groups_and_closed_world():
    real = ["spo2=10", "spo2=10", "dbp", "norepinephrine", "icu"]
    gen = ["spo2=10", "lactate=3", "lactate=4", "ph_arterial"]
    cal = event_rate_calibration(real, gen, GROUPS)
    assert cal["n_real_tokens"] == 5 and cal["n_gen_tokens"] == 4
    assert 0.0 < cal["js_divergence"] <= 1.0
    # lactate=3, lactate=4, ph_arterial are all absent from the real corpus
    assert cal["gen_only_tokens"] == 3
    assert cal["gen_only_mass"] > 0.0
    assert cal["group_rates"]["labs"]["gen"] > cal["group_rates"]["labs"]["real"] == 0.0
    assert cal["group_rates"]["vitals"]["real"] > 0.0


def test_event_rate_calibration_identical_is_zero():
    real = ["spo2=10", "dbp", "norepinephrine"] * 10
    cal = event_rate_calibration(real, list(real), GROUPS)
    assert cal["js_divergence"] == 0.0
    assert cal["gen_only_tokens"] == 0
    assert cal["gen_only_mass"] == 0.0


def test_key_event_recall_and_categorical_focus():
    real_continuation = ["spo2=10", "norepinephrine", "IMV", "lactate=4"]
    gen = ["spo2=9", "norepinephrine", "icu", "dbp"]
    ker = key_event_recall(real_continuation, gen, GROUPS)
    # concepts: real {spo2, norepinephrine, IMV, lactate} vs gen {spo2, norepinephrine, icu, dbp}
    assert ker["unigram_recall"] == pytest.approx(2 / 4)
    # categoricals real {norepinephrine, IMV} vs gen {norepinephrine, icu, dbp}
    assert ker["categorical_recall"] == pytest.approx(1 / 2)


def test_distance_to_observed():
    real_sets = [("h1", frozenset({"spo2", "dbp"})), ("h2", frozenset({"lactate", "norepinephrine"}))]
    near = distance_to_observed(["spo2=10", "dbp"], real_sets)
    assert near["nearest_hospitalization_id"] == "h1"
    assert near["nearest_jaccard_distance"] == 0.0  # identical concept set
    far = distance_to_observed(["icu", "weight_kg"], real_sets)
    assert far["nearest_jaccard_distance"] == 1.0


def test_distinct_n_and_hygiene():
    assert distinct_n(["a", "b", "a", "b"], 2) == pytest.approx(2 / 3)  # 2 unique grams of 3
    hyg = rollout_hygiene([
        ["spo2=10", "spo2=10", "spo2=10", "spo2=10", "<eos>"],
        ["dbp", "map=9", "dbp", "norepinephrine", "<eos>"],
    ])
    assert hyg["n_rollouts"] == 2
    assert hyg["eos_rate"] == 1.0
    assert hyg["unk_rate"] == 0.0
    assert 0.0 < hyg["mean_distinct_2"] < 1.0


# ------------------------------------------------------------------ end-to-end


def test_evaluate_end_to_end(tmp_path, monkeypatch):
    sims = [
        {"hospitalization_id": "prompt-1", "simulation_number": 1, "generated_sequence": "spo2=10 dbp norepinephrine"},
        {"hospitalization_id": "prompt-1", "simulation_number": 2, "generated_sequence": "dbp icu"},
    ]
    pl.DataFrame(sims).write_parquet(tmp_path / "sims.parquet")
    events = [
        {"hosp_id": "h1", "token": [1, 10, 11, 12]},
        {"hosp_id": "h2", "token": [1, 13]},
    ]
    pl.DataFrame(events).write_parquet(tmp_path / "events.parquet")
    (tmp_path / "vocab.json").write_text('{"vocab": {"<bos>": 1, "spo2=10": 10, "dbp": 11, "norepinephrine": 12, "icu": 13}}')

    from src.eval import generative as genmod

    sims_rows = genmod._load_sims(tmp_path / "sims.parquet")
    real_rows = genmod._load_real(tmp_path / "events.parquet", {1: "<bos>", 10: "spo2=10", 11: "dbp", 12: "norepinephrine", 13: "icu"})
    report = evaluate(sims_rows, real_rows, GROUPS, prompt_source=(1, 2))
    assert report["event_rate_calibration"]["n_gen_tokens"] == 5
    # continuation = real[0].tokens[2:] = [dbp, norepinephrine]; gen rollout 1
    # contains both -> unigram recall 1.0
    assert report["key_event_recall"]["mean_unigram_recall"] == 1.0
    assert "distance_to_observed" in report
    assert report["rollout_hygiene"]["n_rollouts"] == 2


def test_evaluate_uses_prompt_column_pairing(tmp_path):
    """The sims `prompt` provenance column (emitted by the generate CLI) pairs
    rollouts with their real continuations by exact prefix match — no head:N
    convention needed."""
    import polars as pl

    from src.eval import generative as genmod

    sims = [
        {"hospitalization_id": "prompt-1", "simulation_number": 1,
         "generated_sequence": "dbp icu", "prompt": "<bos> spo2=10"},
        {"hospitalization_id": "prompt-1", "simulation_number": 2,
         "generated_sequence": "norepinephrine dbp", "prompt": "<bos> spo2=10"},
        {"hospitalization_id": "prompt-2", "simulation_number": 1,
         "generated_sequence": "icu", "prompt": "<bos> dbp"},
    ]
    pl.DataFrame(sims).write_parquet(tmp_path / "sims.parquet")
    events = [
        {"hosp_id": "h1", "token": [1, 10, 11, 12]},  # <bos> spo2=10 dbp norepinephrine
        {"hosp_id": "h2", "token": [1, 11, 13]},      # <bos> dbp icu
    ]
    pl.DataFrame(events).write_parquet(tmp_path / "events.parquet")

    id2tok = {1: "<bos>", 10: "spo2=10", 11: "dbp", 12: "norepinephrine", 13: "icu"}
    sims_rows = genmod._load_sims(tmp_path / "sims.parquet")
    real_rows = genmod._load_real(tmp_path / "events.parquet", id2tok)

    # prompt_source is IGNORED when the prompt column is present
    report = evaluate(sims_rows, real_rows, GROUPS, prompt_source=(9, 9))
    ker = report["key_event_recall"]
    assert ker["prompt_convention"] == "sims `prompt` column (exact prefix match)"
    per = {r["hospitalization_id"]: r for r in ker["per_prompt"]}
    # prompt-1 -> h1 continuation [dbp, norepinephrine]; rollout 1 [dbp, icu] hits dbp
    assert per["h1"]["unigram_recall"] == pytest.approx(0.5)
    # prompt-2 -> h2 continuation [icu]; rollout [icu] hits it
    assert per["h2"]["unigram_recall"] == 1.0
    assert ker["mean_unigram_recall"] == pytest.approx(0.75)


# ------------------------------------------------- U5: groups from the vocab artifact

VOCAB_ARTIFACT = {
    "segments": {"map": [], "lactate": [], "norepinephrine_mcg_kg_min": [], "sodium": []},
    "binning_sources": {"map": "csv", "lactate": "csv",
                        "norepinephrine_mcg_kg_min": "csv", "sodium": "quantile"},
    "concept_sources": {
        "tables": {
            "map": ["vitals"], "lactate": ["labs"], "sodium": ["labs"],
            "norepinephrine_mcg_kg_min": ["meds"], "device_category": ["resp_support"],
            "cam_total": ["assessments"], "sex": ["static"],
        },
        "treatment_sources": ["meds", "resp_support", "static"],
    },
}
ARTIFACT_GROUPS = concept_groups({"target_concepts": [
    {"name": "map", "source": "vitals"}, {"name": "lactate", "source": "labs"},
]}, VOCAB_ARTIFACT)


def test_treatment_and_device_fused_tokens_land_in_the_treatment_group():
    """`device_category=imv` and a med dose token contain `=`, but they are treatments
    and devices, not measurements — grouping follows the token's source table."""
    from src.eval.generative import _group_of

    assert _group_of("device_category=imv", ARTIFACT_GROUPS) == "treatments"
    assert _group_of("norepinephrine_mcg_kg_min=3", ARTIFACT_GROUPS) == "treatments"
    assert _group_of("sex=female", ARTIFACT_GROUPS) == "treatments"
    assert _group_of("map=4", ARTIFACT_GROUPS) == "vitals"
    assert _group_of("lactate=2", ARTIFACT_GROUPS) == "labs"
    assert _group_of("sodium=2", ARTIFACT_GROUPS) == "measurements"
    assert _group_of("cam_total=positive", ARTIFACT_GROUPS) == "categoricals"
    assert _group_of("<eos>", ARTIFACT_GROUPS) == "specials"


def test_calibration_reports_the_treatment_group_and_recall_counts_it_as_key():
    real = ["map=4", "device_category=imv", "norepinephrine_mcg_kg_min=3", "sodium=2"]
    gen = ["map=4", "norepinephrine_mcg_kg_min=1", "sodium=2", "sodium=2"]
    cal = event_rate_calibration(real, gen, ARTIFACT_GROUPS)
    assert cal["group_rates"]["treatments"] == {"real": 0.5, "gen": 0.25}
    assert cal["group_rates"]["measurements"] == {"real": 0.25, "gen": 0.5}
    ker = key_event_recall(real, gen, ARTIFACT_GROUPS)
    # key events: device_category + norepinephrine (treatments); gen hits the infusion
    assert ker["n_real_categorical"] == 2
    assert ker["categorical_recall"] == pytest.approx(0.5)


# ------------------------------------------- U9: rollout mortality evaluation (R20)

DISPOSITIONS = ("home", "facility", "hospice", "expired", "ama", "other", "unknown")


def rec(reason, kind=None, step=5, elapsed=None):
    return {"stop_reason": reason, "terminal_type": kind, "step": step, "elapsed_min": elapsed}


def rollouts(**counts):
    """terminal rollouts by disposition, plus `censored=` / `eos=` non-terminal ones."""
    out = []
    for kind, n in counts.items():
        if kind in ("censored", "eos"):
            out += [rec(kind) for _ in range(n)]
        else:
            out += [rec("terminal", kind) for _ in range(n)]
    return out


def test_wilson_interval_matches_the_closed_form():
    lo, hi = wilson_interval(3, 8)
    assert lo == pytest.approx(0.13684, abs=1e-4)
    assert hi == pytest.approx(0.69426, abs=1e-4)
    assert wilson_interval(0, 0) == (None, None)
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0 and 0.0 < hi < 1.0


def test_rollout_mortality_divides_by_terminated_rollouts_only():
    result = rollout_mortality(rollouts(expired=3, home=5, censored=2))
    assert result["n_rollouts"] == 10
    assert result["n_expired"] == 3
    assert result["n_terminated"] == 8
    assert result["n_censored"] == 2
    assert result["mortality"] == pytest.approx(3 / 8)
    assert result["ci_low"] == pytest.approx(0.13684, abs=1e-4)
    assert result["ci_high"] == pytest.approx(0.69426, abs=1e-4)
    assert result["nontermination_rate"] == pytest.approx(0.2)
    assert result["terminal_distribution"] == {"expired": 3, "home": 5}
    assert result["modal_terminal"] == "home"


def test_censored_rollouts_are_never_counted_as_survival():
    result = rollout_mortality(rollouts(expired=1, censored=5))
    assert result["mortality"] == 1.0  # 1/1, not 1/6
    all_censored = rollout_mortality(rollouts(censored=4))
    assert all_censored["mortality"] is None
    assert all_censored["ci_low"] is None and all_censored["modal_terminal"] is None


def test_eos_and_unknown_disposition_are_excluded_from_the_denominator():
    """`<eos>` without a disposition and DISCHARGE//unknown are end of observation, not
    survival (R18): reported separately, excluded like censoring."""
    result = rollout_mortality(rollouts(expired=1, home=1, unknown=2, eos=3))
    assert result["n_terminated"] == 2
    assert result["n_unknown"] == 2 and result["n_eos"] == 3
    assert result["mortality"] == pytest.approx(0.5)
    assert result["terminal_distribution"] == {"expired": 1, "home": 1, "unknown": 2}


def four_stays():
    return [
        {"key": "s1", "observed": "expired", "rollouts": rollouts(expired=3, home=1)},
        {"key": "s2", "observed": "expired", "rollouts": rollouts(expired=2, home=1, facility=1)},
        {"key": "s3", "observed": "home", "rollouts": rollouts(expired=1, home=3, censored=2)},
        {"key": "s4", "observed": "facility", "rollouts": rollouts(expired=3, facility=2)},
    ]


def test_evaluation_on_four_stays_gives_expected_auroc_and_confusion():
    report = evaluate_rollout_mortality(four_stays(), dispositions=DISPOSITIONS)
    summary = report["summary"]
    # p = .75, .5 (expired) vs .25, .6 (survived): 3 of 4 pairs ordered
    assert summary["n_evaluable"] == 4
    assert summary["auroc"] == pytest.approx(0.75)
    assert summary["auprc"] == pytest.approx(5 / 6)
    assert summary["observed_mortality"] == pytest.approx(0.5)
    assert isinstance(summary["calibration_slope"], float)
    assert isinstance(summary["calibration_intercept"], float)
    confusion = summary["terminal_confusion"]["counts"]
    assert confusion["expired"]["expired"] == 2
    assert confusion["home"]["home"] == 1
    assert confusion["facility"]["expired"] == 1
    assert sum(sum(row.values()) for row in confusion.values()) == 4
    # 2 censored of 4 + 4 + 6 + 5 = 19 rollouts
    assert summary["n_rollouts"] == 19 and summary["n_censored"] == 2
    assert summary["nontermination_rate"] == pytest.approx(2 / 19)
    per = {row["key"]: row for row in report["per_stay"]}
    assert per["s3"]["mortality"] == pytest.approx(0.25) and per["s3"]["n_censored"] == 2


def test_evaluation_excludes_unknown_outcomes_and_stays_without_terminations():
    stays = [*four_stays(),
             {"key": "s5", "observed": "unknown", "rollouts": rollouts(expired=4)},
             {"key": "s6", "observed": "expired", "rollouts": rollouts(censored=3)}]
    summary = evaluate_rollout_mortality(stays, dispositions=DISPOSITIONS)["summary"]
    assert summary["n_stays"] == 6 and summary["n_evaluable"] == 4
    assert summary["n_excluded_unknown_observed"] == 1
    assert summary["n_excluded_no_termination"] == 1
    assert summary["auroc"] == pytest.approx(0.75)
    # the all-censored stay's modal terminal is "none", never a disposition
    assert summary["terminal_confusion"]["counts"]["expired"]["none"] == 1


def test_summary_is_aggregate_only():
    report = evaluate_rollout_mortality(four_stays(), dispositions=DISPOSITIONS)
    assert "s1" not in json.dumps(report["summary"])


def test_time_to_terminal_mae_is_marked_unavailable_without_elapsed_minutes():
    stays = [dict(stay, observed_minutes=600.0) for stay in four_stays()]
    ttt = evaluate_rollout_mortality(stays, dispositions=DISPOSITIONS)["summary"][
        "time_to_terminal"]
    assert ttt["available"] is False and ttt["mae_min"] is None
    assert "reason" in ttt


def test_time_to_terminal_mae_uses_the_median_terminal_elapsed_minutes():
    stays = [
        {"key": "a", "observed": "home", "observed_minutes": 100.0,
         "rollouts": [rec("terminal", "home", elapsed=e) for e in (80.0, 120.0, 400.0)]
         + [rec("censored", elapsed=999.0)]},
        {"key": "b", "observed": "expired", "observed_minutes": 50.0,
         "rollouts": [rec("terminal", "expired", elapsed=110.0)]},
    ]
    ttt = evaluate_rollout_mortality(stays, dispositions=DISPOSITIONS)["summary"][
        "time_to_terminal"]
    # |120 - 100| = 20 and |110 - 50| = 60; censored minutes never enter the median
    assert ttt["available"] is True and ttt["n"] == 2
    assert ttt["mae_min"] == pytest.approx(40.0)


def test_observed_gem_outcome_reads_the_terminal_after_the_anchor():
    vocab = {"<bos>": 1, "<eos>": 2, "hr=1": 4, "ADMISSION//ed": 6,
             **{f"DISCHARGE//{d}": 10 + i for i, d in enumerate(DISPOSITIONS)}}
    windows = [
        {"continuation_index": 0, "source_start": 0, "token": [1, 6, 4, 4],
         "pos_min": [0, 0, 30, 90], "anchor_idx": 3},
        {"continuation_index": 1, "source_start": 4, "token": [4, 13, 2],
         "pos_min": [200, 400, 400], "anchor_idx": 3},
    ]
    assert observed_gem_outcome(windows, vocab) == {"disposition": "expired",
                                                    "minutes_after_anchor": 310}
