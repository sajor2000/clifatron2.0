"""U5 (R13, KTD1/KTD7): outcome_join's threshold query bins with the tokenizer's
`bin_index` over the same frozen segments, so the queried threshold bin is exactly the
token bin of a value just on the event side of the threshold."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.data.outcome_join import join_outcomes
from src.data.segments import bin_index, load_csv_segments, threshold_bin

ROOT = Path(__file__).parents[1]
CSV = ROOT / "external/clifatron/tokenETL/config/critical_illness_tokenization_final_with_intervals.csv"


@pytest.fixture(scope="module")
def segments():
    if not CSV.exists():
        pytest.skip("physician segmentation CSV is not checked out")
    return load_csv_segments(
        CSV, ["map", "lactate"],
        forced_edges={"map": [65.0], "lactate": [2.0, 4.0]},
        directions={"map": "below", "lactate": "above"},
    )


def _blob(segments):
    return {"vocab": {}, "segments": segments, "manifest": {"tokenizer_version": 2}}


DATA_CONFIG = {"target_concepts": [
    {"name": "map", "direction": "below"},
    {"name": "lactate", "direction": "above"},
]}
COHORT_CONFIG = {"outcomes": {
    "map_below_65": {"concept": "map", "direction": "below", "threshold": 65.0},
    "lactate_above_4": {"concept": "lactate", "direction": "above", "threshold": 4.0},
}}


def _join(segments):
    labels = pl.DataFrame({
        "hospitalization_id": ["s1"],
        "map_below_65_status": ["positive"],
        "map_below_65_time_from_anchor_hours": [3.0],
        "lactate_above_4_status": ["negative"],
        "lactate_above_4_time_from_anchor_hours": [None],
    }, schema_overrides={"lactate_above_4_time_from_anchor_hours": pl.Float64})
    events = pl.DataFrame({"hosp_id": ["s1"], "token": [[1, 2]]})
    joined = join_outcomes(labels, events, _blob(segments), DATA_CONFIG, COHORT_CONFIG)
    return {o["target_idx"]: o for o in joined["outcomes"].to_list()[0]}


def test_map_below_65_threshold_bin_is_the_token_bin_of_64_9(segments):
    outcomes = _join(segments)
    assert outcomes[0]["threshold_bin"] == bin_index(64.9, segments["map"])
    # ...and not the bin 65 itself lands in (65 is the non-event side under `below`).
    assert outcomes[0]["threshold_bin"] != bin_index(65.0, segments["map"])


def test_lactate_above_4_threshold_bin_is_the_token_bin_of_4_01(segments):
    outcomes = _join(segments)
    assert outcomes[1]["threshold_bin"] == bin_index(4.01, segments["lactate"])
    assert outcomes[1]["threshold_bin"] != bin_index(4.0, segments["lactate"])


def test_threshold_bin_helper_matches_both_directions(segments):
    assert threshold_bin(65.0, segments["map"], "below") == bin_index(64.9, segments["map"])
    assert threshold_bin(4.0, segments["lactate"], "above") == bin_index(4.01, segments["lactate"])
    with pytest.raises(ValueError):
        threshold_bin(65.0, segments["map"], "sideways")


def test_unbinned_concept_has_no_threshold_bin(segments):
    outcomes = _join({"lactate": segments["lactate"]})
    assert outcomes[0]["threshold_bin"] == -1


def test_v1_vocab_without_segments_is_refused():
    with pytest.raises(ValueError, match="re-tokenize"):
        join_outcomes(
            pl.DataFrame({"hospitalization_id": ["s1"]}),
            pl.DataFrame({"hosp_id": ["s1"]}),
            {"vocab": {}, "edges": {"map": [65.0]}},
            DATA_CONFIG, COHORT_CONFIG,
        )
