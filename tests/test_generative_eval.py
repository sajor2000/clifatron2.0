"""Tests for the G3 generative evals (src/eval/generative.py)."""

from __future__ import annotations

from collections import Counter

import polars as pl
import pytest

from src.eval.generative import (
    concept_groups,
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
