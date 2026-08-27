"""CLIF 2.1 parquet -> per-site event-token shards + vocab.json.

REVISED per Lee et al. 2026 (arXiv:2604.16775): each event is ONE FUSED token
`concept=bin` (numeric) or `concept` (categorical) — the fused code+value token
was the single biggest win on MIMIC-IV-Ext-CLIF (mortality 0.891->0.915), so the
old split concept/value tokens are retired. Value bins are per-concept deciles,
frozen from one site (config: value_binning.build_from_site) and applied to BOTH
sites so Rush and MIMIC share a vocabulary without pooling raw data.

Position is admission-relative minutes (`pos_min`), consumed by RoPE downstream
(replaces Δt). Events are ordered by `storetime` (availability), never charttime.

Treatments (meds) are emitted as INPUT events only; they are never targets
(ICareFM rule 1). See notes/METHODS.md and notes/RESEARCH.md §2.

Usage:
    python -m src.data.tokenize --site mimic --in $MIMIC_DIR --out data/mimic --build-vocab
    python -m src.data.tokenize --site rush  --in $RUSH_DIR  --out data/rush  --vocab data/mimic/vocab.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import yaml

SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2}


def _read_table(con, base: Path, spec: dict) -> pl.DataFrame:
    """Melt one CLIF table to long events ordered by its availability timestamp."""
    fp = base / f"{spec['file']}.parquet"
    if not fp.exists():
        print(f"  [skip] {fp.name} not found")
        return pl.DataFrame(
            schema={
                "hosp_id": pl.Utf8,
                "dttm": pl.Datetime,
                "concept": pl.Utf8,
                "value": pl.Float64,
                "unit": pl.Utf8,
            }
        )
    val = spec.get("value_col")
    val_sql = f"CAST({val} AS DOUBLE)" if val else "NULL"
    unit = spec.get("unit_col")
    unit_sql = f"CAST({unit} AS VARCHAR)" if unit else "NULL"
    time_col = spec["availability_col"]
    q = f"""
        SELECT hospitalization_id       AS hosp_id,
               {time_col}                AS dttm,
               {spec['concept_col']}     AS concept,
               {val_sql}                 AS value,
               {unit_sql}                AS unit
        FROM read_parquet('{fp}')
        WHERE {spec['concept_col']} IS NOT NULL
          AND {time_col} IS NOT NULL
    """
    return con.execute(q).pl()


def validate_units(events: pl.DataFrame, cfg: dict) -> None:
    def normalized(unit: str) -> str:
        return unit.strip().lower().replace("¬µ", "u").replace("µ", "u").replace("μ", "u")

    expected = cfg.get("unit_normalization", {}).get("concepts", {})
    observed = events.filter(pl.col("unit").is_not_null()).select("concept", "unit").unique()
    mismatches = [
        f"{concept}: expected {expected[concept]!r}, found {unit!r}"
        for concept, unit in observed.iter_rows()
        if concept in expected and normalized(unit) != normalized(expected[concept])
    ]
    if mismatches and cfg["unit_normalization"].get("on_mismatch") == "error":
        raise ValueError("Non-canonical CLIF units: " + "; ".join(sorted(mismatches)))


def build_value_bins(events: pl.DataFrame, n_bins: int) -> dict[str, list[float]]:
    """Per-concept quantile edges (deciles by default). Numeric concepts only."""
    edges: dict[str, list[float]] = {}
    numeric = events.filter(pl.col("value").is_not_null())
    for concept in numeric["concept"].unique().to_list():
        vals = numeric.filter(pl.col("concept") == concept)["value"].drop_nulls()
        if len(vals) < n_bins:
            continue
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges[concept] = [float(vals.quantile(q)) for q in qs]
    return edges


def fused_token(concept: str, b: int | None) -> str:
    """One token per event: `concept=bin` if numeric, else bare `concept`."""
    return f"{concept}={b}" if b is not None else concept


def build_vocab(events: pl.DataFrame, edges: dict[str, list[float]]) -> dict:
    """One id per FUSED token: `concept` (categorical) or `concept=bin` (numeric)."""
    vocab = dict(SPECIAL)
    nxt = len(vocab)
    for concept in sorted(events["concept"].unique().to_list()):
        if concept in edges:
            for b in range(len(edges[concept]) + 1):        # n_bins fused tokens
                vocab[fused_token(concept, b)] = nxt
                nxt += 1
        else:
            vocab[fused_token(concept, None)] = nxt
            nxt += 1
    return vocab


def _bin_of(value: float | None, concept: str, edges: dict[str, list[float]]) -> int | None:
    if value is None or concept not in edges:
        return None
    return int(np.searchsorted(edges[concept], value, side="right"))


def tokenize_site(cfg: dict, site: str, base: Path, out: Path,
                  vocab: dict | None, edges: dict | None):
    con = duckdb.connect()
    frames = []
    for name, spec in cfg["tables"].items():
        df = _read_table(con, base, spec)
        if len(df):
            df = df.with_columns(source=pl.lit(name))
            frames.append(df)
    events = pl.concat(frames, how="vertical_relaxed").sort(["hosp_id", "dttm"])
    validate_units(events, cfg)
    print(f"  {site}: {len(events):,} raw events, {events['hosp_id'].n_unique():,} stays")

    if vocab is None:  # --build-vocab path
        edges = build_value_bins(events, cfg["value_binning"]["n_bins"])
        vocab = build_vocab(events, edges)
        print(f"  built vocab: {len(vocab):,} tokens, {len(edges):,} numeric concepts")

    # map each event to (fused token, admission-relative minutes, raw value)
    def encode(group: pl.DataFrame) -> pl.DataFrame:
        dt = group["dttm"].to_numpy()
        pos_min = (dt - dt[0]).astype("timedelta64[s]").astype(np.int64) // 60  # since admission
        token, valnum = [], []
        for c, v in zip(group["concept"], group["value"]):
            b = _bin_of(v, c, edges)
            token.append(vocab.get(fused_token(c, b), SPECIAL["<pad>"]))
            valnum.append(float(v) if v is not None else float("nan"))  # ORA value-regression target
        return pl.DataFrame({
            "hosp_id": group["hosp_id"][0],
            "token": [token],
            "pos_min": [pos_min.tolist()],
            "value": [valnum],
            "n_events": len(token),
        })

    shards = events.group_by("hosp_id", maintain_order=True).map_groups(encode)
    out.mkdir(parents=True, exist_ok=True)
    shards.write_parquet(out / "events.parquet")
    if edges is not None:
        (out / "vocab.json").write_text(json.dumps({"vocab": vocab, "edges": edges}))
    print(f"  wrote {out/'events.parquet'} ({len(shards):,} stays)")
    return vocab, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--build-vocab", action="store_true")
    ap.add_argument("--vocab", help="path to an existing vocab.json to reuse")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    vocab, edges = None, None
    if args.vocab:
        blob = json.loads(Path(args.vocab).read_text())
        vocab, edges = blob["vocab"], blob["edges"]
    elif not args.build_vocab:
        raise SystemExit("pass --build-vocab (first site) or --vocab PATH (later sites)")

    if args.dry_run:
        con = duckdb.connect()
        for name, spec in cfg["tables"].items():
            df = _read_table(con, Path(args.indir), spec)
            print(f"{name}: {len(df):,} events, concepts={df['concept'].n_unique() if len(df) else 0}")
        return

    tokenize_site(cfg, args.site, Path(args.indir), Path(args.out), vocab, edges)


if __name__ == "__main__":
    main()
