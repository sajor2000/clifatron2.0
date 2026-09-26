from src.eval.clinical_plausibility import assess_sequence, split_sequence
from src.viewer.sequence_viewer import parse_token, summarize_sequence


def test_assess_sequence_flags_structural_problems():
    result = assess_sequence(
        ["icu", "map=0", "map=0", "map=0", "map=0", "<pad>"],
        vocab={"icu", "map=0", "<pad>"},
        edges={"map": [65.0]},
        prompt_tokens=["icu"],
    )

    assert result["status"] == "poor"
    assert result["error_count"] == 1
    codes = {warning["code"] for warning in result["warnings"]}
    assert "structural-special-token" in codes
    assert "repeated-token-run" in codes


def test_fused_bin_parser_uses_numeric_edges():
    token = parse_token("map=1", edges={"map": [65.0]})
    assert token["kind"] == "bin"
    assert token["concept"] == "map"
    assert token["bin"] == 1
    assert token["low"] == 65.0


def test_summary_exposes_explainable_review():
    summary = summarize_sequence(
        ["icu", "map=1"],
        vocab={"icu", "map=1"},
        edges={"map": [65.0]},
        prompt_tokens=["icu"],
    )
    assert summary["plausibility"]["status"] == "good"
    assert summary["parsed_tokens"][1]["kind"] == "bin"


def test_split_sequence_preserves_space_containing_vocab_tokens():
    assert split_sequence(
        "icu sodium chloride map=1",
        vocab={"icu", "sodium chloride", "map=1"},
    ) == ["icu", "sodium chloride", "map=1"]
