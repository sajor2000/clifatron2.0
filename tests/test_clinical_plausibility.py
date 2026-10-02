from pathlib import Path

import pytest

from src.data.segments import load_csv_segments, segments_from_edges
from src.eval.clinical_plausibility import assess_sequence, split_sequence
from src.viewer.sequence_viewer import parse_token, summarize_sequence

ROOT = Path(__file__).parents[1]
CSV = ROOT / "external/clifatron/tokenETL/config/critical_illness_tokenization_final_with_intervals.csv"
# Two segments: (-inf, 65) and [65, +inf) — `map=0` and `map=1` are its only bins.
MAP_TWO = {"map": segments_from_edges([65.0])}


@pytest.fixture(scope="module")
def csv_map():
    if not CSV.exists():
        pytest.skip("physician segmentation CSV is not checked out")
    return load_csv_segments(CSV, ["map"], {"map": [65.0]}, {"map": "below"})["map"]


def test_assess_sequence_flags_structural_problems():
    result = assess_sequence(
        ["icu", "map=0", "map=0", "map=0", "map=0", "<pad>"],
        vocab={"icu", "map=0", "<pad>"},
        segments=MAP_TWO,
        prompt_tokens=["icu"],
    )

    assert result["status"] == "poor"
    assert result["error_count"] == 1
    codes = {warning["code"] for warning in result["warnings"]}
    assert "structural-special-token" in codes
    assert "repeated-token-run" in codes


def test_plausibility_flags_a_bin_beyond_the_frozen_segments(csv_map):
    """U5: the bin count is len(segments) — `len(edges) + 1` over segment dicts let one
    out-of-range index through."""
    n = len(csv_map)
    last_ok = assess_sequence([f"map={n - 1}"], segments={"map": csv_map}, prompt_tokens=[])
    assert last_ok["stats"]["invalid_bin_count"] == 0
    beyond = assess_sequence([f"map={n}"], segments={"map": csv_map}, prompt_tokens=[])
    assert beyond["stats"]["invalid_bin_count"] == 1
    assert "invalid-bin" in {w["code"] for w in beyond["warnings"]}


def test_fused_bin_parser_decodes_segments():
    token = parse_token("map=1", segments=MAP_TWO)
    assert token["kind"] == "bin"
    assert token["concept"] == "map"
    assert token["bin"] == 1
    assert token["low"] == 65.0
    assert token["high"] is None
    assert token["interval"] == "[65, +inf)"
    assert parse_token("map=0", segments=MAP_TWO)["interval"] == "(-inf, 65)"


def test_viewer_decodes_map_6_with_brackets_from_closure(csv_map):
    # Forced 65 edge under `below`: (61, 65) | [65, 67] — the threshold value lands on
    # the non-event side.
    six = parse_token("map=6", segments={"map": csv_map})
    assert (six["low"], six["high"]) == (65.0, 67.0)
    assert six["interval"] == "[65, 67]"
    assert parse_token("map=5", segments={"map": csv_map})["interval"] == "(61, 65)"
    # A bin index beyond the segments decodes to no interval rather than a neighbour's.
    beyond = parse_token(f"map={len(csv_map)}", segments={"map": csv_map})
    assert beyond["interval"] is None and beyond["low"] is None


def test_summary_exposes_explainable_review():
    summary = summarize_sequence(
        ["icu", "map=1"],
        vocab={"icu", "map=1"},
        segments=MAP_TWO,
        prompt_tokens=["icu"],
    )
    assert summary["plausibility"]["status"] == "good"
    assert summary["parsed_tokens"][1]["kind"] == "bin"


def test_split_sequence_preserves_space_containing_vocab_tokens():
    assert split_sequence(
        "icu sodium chloride map=1",
        vocab={"icu", "sodium chloride", "map=1"},
    ) == ["icu", "sodium chloride", "map=1"]


def test_viewer_defines_load_vocab_segments_once():
    import ast

    source = (ROOT / "src/viewer/sequence_viewer.py").read_text()
    names = [n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    assert len(names) == len(set(names)), "a viewer function is defined twice"
    assert "load_vocab_edges" not in names
