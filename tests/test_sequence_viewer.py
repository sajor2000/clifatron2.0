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


def test_parquet_source_ingests_list_typed_token_column(tmp_path):
    """Tokenizer events.parquet stores `token` as an Int64 LIST; the viewer joins it
    into a string at read time so real ETL output loads without a txt export. Integer
    ids are decoded to vocab token strings when an id_to_token map is provided."""
    import polars as pl

    from src.viewer.sequence_viewer import ParquetSource

    id_to_token = {1: "<bos>", 2: "<eos>", 4: "hr=80_90", 5: "spo2=88_90", 6: "map=60_65"}
    df = pl.DataFrame({
        "hosp_id": ["h1", "h2"],
        "token": [[1, 4, 2], [1, 5, 6, 999]],
        "partition": ["train", "validation"],
    })
    path = tmp_path / "events.parquet"
    df.write_parquet(path)

    src = ParquetSource(path=path, name="events", id_to_token=id_to_token)
    assert src.seq_col == "token"
    assert src.id_col == "hosp_id"
    assert src._seq_is_list and src._seq_is_int
    rows = src.list_rows(offset=0, limit=10, search=None)
    assert [r["n_tokens"] for r in rows] == [3, 4]
    assert rows[0]["preview"] == "<bos> hr=80_90 <eos>"
    # unknown id 999 stays numeric (never silent)
    assert rows[1]["preview"] == "<bos> spo2=88_90 map=60_65 999"
    hits = src.list_rows(offset=0, limit=10, search="spo2=88_90")
    assert len(hits) == 1
    row = src.get_row(row_id="h2")
    assert "spo2=88_90" in row["token"]
