from __future__ import annotations

from src.viewer.sequence_viewer import parse_token, summarize_sequence


def test_parse_token_special():
    tok = parse_token("[BOS]")
    assert tok["kind"] == "special"
    assert tok["group"] == "special"
    assert tok["concept"] == "[BOS]"


def test_parse_token_range():
    tok = parse_token("vitals_hr_80_100")
    assert tok["kind"] == "range"
    assert tok["group"] == "vitals"
    assert tok["concept"] == "vitals_hr"
    assert tok["low"] == 80.0
    assert tok["high"] == 100.0


def test_parse_token_value():
    tok = parse_token("age_65")
    assert tok["kind"] == "value"
    assert tok["concept"] == "age"
    assert tok["value"] == 65.0


def test_parse_token_categorical():
    tok = parse_token("sex_male")
    assert tok["kind"] == "categorical"
    assert tok["group"] == "sex"
    assert tok["concept"] == "sex"
    assert tok["category"] == "male"


def test_summarize_sequence():
    tokens = ["sex_male", "vitals_hr_80_100", "vitals_hr_100_120"]
    summary = summarize_sequence(tokens)
    assert summary["n_tokens"] == 3
    groups = dict(summary["groups"])
    assert groups["vitals"] == 2
    assert groups["sex"] == 1
