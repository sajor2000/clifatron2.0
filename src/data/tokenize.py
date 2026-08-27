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
    unit_sql = f"CAST({unit} AS VARCHAR)" if unit else "CAST('' AS VARCHAR)"
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
        u = unit.strip().lower().replace("¬µ", "u").replace("µ", "u").replace("μ", "u")
        return u.replace("k/ul", "10^3/ul").replace("10^3/ul", "10^3/ul")

    expected = cfg.get("unit_normalization", {}).get("concepts", {})
    observed = events.filter(pl.col("unit").is_not_null()).select("concept", "unit").unique()
    mismatches = [
        f"{concept}: expected {expected[concept]!r}, found {unit!r}"
        for concept, unit in observed.iter_rows()
        if concept in expected and normalized(unit) != normalized(expected[concept])
        and unit not in (None, "")
    ]
    if mismatches and cfg["unit_normalization"].get("on_mismatch") == "error":
        raise ValueError("Non-canonical CLIF units: " + "; ".join(sorted(mismatches)))
    if not expected:
        return


def build_value_bins(events: pl.DataFrame, n_bins: int,
                     forced_edges: dict[str, list[float]] | None = None) -> dict[str, list[float]]:
    """Per-concept quantile edges with clinical cutpoints replacing nearest edges.

    Finite forced edges are retained regardless of the reference data range
    so frozen vocab remains valid across sites with wider distributions."""
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    edges: dict[str, list[float]] = {}
    forced_edges = forced_edges or {}
    numeric = events.filter(pl.col("value").is_not_null())
    for concept in numeric["concept"].unique().to_list():
        vals = numeric.filter(pl.col("concept") == concept)["value"].drop_nulls()
        if len(vals) < n_bins:
            continue
        if not np.isfinite(vals.to_numpy()).all():
            raise ValueError(f"{concept} contains non-finite values; filter before binning")

        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        concept_edges = sorted(set(float(vals.quantile(q)) for q in qs))
        lo, hi = float(vals.min()), float(vals.max())

        pinned = sorted(set(
            float(edge) for edge in forced_edges.get(concept, [])
            if np.isfinite(float(edge))
        ))
        if len(pinned) > n_bins - 1:
            raise ValueError(
                f"{concept} has {len(pinned)} forced edges but only {n_bins - 1} boundaries"
            )

        for edge in pinned:
            existing = next(
                (i for i, e in enumerate(concept_edges) if np.isclose(e, edge)),
                None,
            )
            if existing is not None:
                concept_edges[existing] = edge
            else:
                removable = [e for e in concept_edges if not any(np.isclose(e, p) for p in pinned)]
                if len(concept_edges) >= n_bins - 1 and removable:
                    concept_edges.remove(min(removable, key=lambda e: abs(e - edge)))
                concept_edges.append(edge)
                concept_edges.sort()
        edges[concept] = concept_edges
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
    return min(max(int(np.searchsorted(edges[concept], value, side="right")),
                   0), len(edges[concept]))


def _soft_bins(value: float | None, concept: str, edges: dict[str, list[float]],
               kernel_bins: int) -> list[tuple[int | None, float]]:
    """Return fixed-width (2*kernel_bins+1) assignments so every event produces
    a uniform-length list for the [B,T,K] dense tensor the encoder expects."""
    hard_bin = _bin_of(value, concept, edges)
    if hard_bin is None or kernel_bins <= 0:
        return [(hard_bin, 1.0)]

    boundaries = edges[concept]
    lower = boundaries[hard_bin - 1] if hard_bin else None
    upper = boundaries[hard_bin] if hard_bin < len(boundaries) else None
    if lower is None and upper is not None:
        width = boundaries[1] - upper if len(boundaries) > 1 else 1.0
        lower = upper - max(width, 1e-12)
    if upper is None and lower is not None:
        width = lower - boundaries[-2] if len(boundaries) > 1 else 1.0
        upper = lower + max(width, 1e-12)
    center = hard_bin
    if lower is not None and upper is not None and upper > lower:
        center += float(np.clip((value - lower) / (upper - lower), 0, 1)) - 0.5

    half = kernel_bins
    candidates = np.arange(max(0, hard_bin - half), min(len(boundaries), hard_bin + half) + 1)
    sigma = max(half / 2, 0.5)
    weights = np.exp(-0.5 * ((candidates - center) / sigma) ** 2)
    weights /= weights.sum()

    width = 2 * half + 1
    padded_bins = [hard_bin] * width
    padded_weights = [1.0] + [0.0] * (width - 1)
    for idx, weight in zip(candidates, weights):
        slot = int(idx) - (hard_bin - half)
        if 0 <= slot < width:
            padded_bins[slot] = int(idx)
            padded_weights[slot] = float(weight)
    return list(zip(padded_bins, padded_weights))


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
        build_from = cfg["value_binning"].get("build_from_site")
        if build_from is not None and site != build_from:
            raise ValueError(
                f"value_binning.build_from_site is {build_from!r} but "
                f"--build-vocab was called with --site {site!r}. "
                f"Only {build_from!r} may build the frozen vocabulary."
            )
        bin_cfg = cfg["value_binning"]
        edges = build_value_bins(events, bin_cfg["n_bins"], bin_cfg.get("forced_edges"))
        vocab = build_vocab(events, edges)
        print(f"  built vocab: {len(vocab):,} tokens, {len(edges):,} numeric concepts")

    # map each event to (fused token, admission-relative minutes, raw value)
    def encode(group: pl.DataFrame) -> pl.DataFrame:
        dt = group["dttm"].to_numpy()
        pos_min = (dt - dt[0]).astype("timedelta64[s]").astype(np.int64) // 60  # since admission
        token, soft_token, soft_weight, valnum = [], [], [], []
        bin_cfg = cfg["value_binning"]
        for c, v in zip(group["concept"], group["value"]):
            v_for_bin = float(v) if v is not None and np.isfinite(v) else None
            b = _bin_of(v_for_bin, c, edges)
            if b is None:
                key = fused_token(c, None)
                hard_token = vocab.get(key)
                if hard_token is None and c in edges:
                    b = 0
                    key = fused_token(c, b)
                    hard_token = vocab.get(key, SPECIAL["<pad>"])
            else:
                key = fused_token(c, b)
                hard_token = vocab.get(key, SPECIAL["<pad>"])
            if hard_token is None:
                raise KeyError(
                    f"unknown token {key!r} — frozen vocab does not cover this concept+bin"
                )
            token.append(hard_token)
            assignments = (
                _soft_bins(v, c, edges, bin_cfg["soft_kernel_bins"])
                if bin_cfg.get("soft_discretization")
                else [(b, 1.0)]
            )
            soft_tokens, weights = [], []
            for soft_bin, weight in assignments:
                st_key = fused_token(c, soft_bin) if soft_bin is not None else fused_token(c, None)
                soft_tokens.append(vocab.get(st_key, SPECIAL["<pad>"]))
                weights.append(weight)
            soft_token.append(soft_tokens)
            soft_weight.append(weights)
            valnum.append(float(v) if v is not None else float("nan"))  # ORA value-regression target
        return pl.DataFrame({
            "hosp_id": group["hosp_id"][0],
            "token": [token],
            "soft_token": [soft_token],
            "soft_weight": [soft_weight],
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
