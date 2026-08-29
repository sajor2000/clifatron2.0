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
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import yaml

from src.data.cohort import (
    QualificationError,
    validate_artifact_destination,
    validate_episode_artifact,
)
from src.data.splits import fit_partition

SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
ROOT = Path(__file__).parents[2]


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_vocabulary_artifact(
    blob: dict,
    cfg: dict,
    policy: dict,
    *,
    expected_family: str = "experimental_representation",
) -> tuple[dict, dict, dict]:
    """Validate imported vocabulary content and its original compatibility record."""
    if not isinstance(blob.get("vocab"), dict) or not isinstance(blob.get("edges"), dict):
        raise QualificationError("vocabulary artifact must contain vocab and edges mappings")
    manifest = blob.get("manifest")
    if not isinstance(manifest, dict):
        raise QualificationError("vocabulary artifact is missing its manifest")
    if manifest.get("artifact_family") != expected_family:
        raise QualificationError("incompatible vocabulary artifact family")
    if manifest.get("clif_version") != cfg["schema_version"]:
        raise QualificationError("vocabulary CLIF version is incompatible")
    if manifest.get("mcide_version") != cfg["mcide_version"]:
        raise QualificationError("vocabulary mCIDE version is incompatible")
    family = policy.get("compatibility_manifests", {}).get(expected_family)
    if family is None:
        raise QualificationError("artifact policy does not define the vocabulary family")
    hashes = manifest.get("hashes")
    if not isinstance(hashes, dict):
        raise QualificationError("vocabulary manifest is missing compatibility hashes")
    missing = sorted(set(family["required_hashes"]) - set(hashes))
    if missing:
        raise QualificationError(f"vocabulary manifest is missing hashes: {', '.join(missing)}")
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise QualificationError("vocabulary manifest contains an invalid SHA-256 hash")
    if hashes["vocabulary"] != _json_sha256(blob["vocab"]):
        raise QualificationError("vocabulary hash mismatch")
    if hashes["numeric_edges"] != _json_sha256(blob["edges"]):
        raise QualificationError("numeric-edge hash mismatch")
    if hashes["clif_version"] != _json_sha256(cfg["schema_version"]):
        raise QualificationError("CLIF-version compatibility hash mismatch")
    if hashes.get("target_map") != _json_sha256(cfg["target_concepts"]):
        raise QualificationError("target-map compatibility hash mismatch")
    cohort_cfg = yaml.safe_load((ROOT / cfg["cohort_contract"]).read_text())
    if hashes.get("outcome_spec") != _json_sha256(cohort_cfg["outcomes"]):
        raise QualificationError("outcome-spec compatibility hash mismatch")
    return blob["vocab"], blob["edges"], manifest


def _read_table(con, base: Path, spec: dict,
                keep_ids: list | None = None) -> pl.DataFrame:
    """Melt one CLIF table to long events ordered by its availability timestamp.

    If `keep_ids` is given, only rows for those hospitalization_ids are read
    (pushed into the SQL WHERE so the 45M+ row tables are filtered on scan, not
    after loading). Used to tokenize a small sample fast for smoke tests / dev."""
    fp = base / f"{spec['file']}.parquet"
    if not fp.exists():
        print(f"  [skip] {fp.name} not found")
        return pl.DataFrame(
            schema={
                "hosp_id": pl.String,
                "dttm": pl.Datetime("us", "UTC"),
                "concept": pl.String,
                "value": pl.Float64,
                "unit": pl.String,
            }
        )
    val = spec.get("value_col")
    val_sql = f"CAST({val} AS DOUBLE)" if val else "NULL"
    unit = spec.get("unit_col")
    unit_sql = f"CAST({unit} AS VARCHAR)" if unit else "CAST('' AS VARCHAR)"
    time_col = spec["availability_col"]
    id_filter = ""
    if keep_ids is not None:
        id_list = ", ".join(f"'{i}'" for i in keep_ids)
        id_filter = f"AND CAST(hospitalization_id AS VARCHAR) IN ({id_list})"
    q = f"""
        SELECT hospitalization_id       AS hosp_id,
               {time_col}                AS dttm,
               {spec['concept_col']}     AS concept,
               {val_sql}                 AS value,
               {unit_sql}                AS unit
        FROM read_parquet('{fp}')
        WHERE {spec['concept_col']} IS NOT NULL
          AND {time_col} IS NOT NULL
          {id_filter}
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


def restrict_to_observation_window(
    events: pl.DataFrame,
    episodes: pl.DataFrame,
    treatment_sources: set[str] | None = None,
) -> pl.DataFrame:
    """Join canonical episodes and retain only anchor-available ICU events."""
    validate_episode_artifact(episodes)
    if events.schema.get("hosp_id") != pl.String:
        raise QualificationError("events.hosp_id must be a string identifier")
    if events["hosp_id"].has_nulls():
        raise QualificationError("events.hosp_id contains null identifiers")
    for frame, name, column in [
        (events, "events", "dttm"),
        (episodes, "episodes", "icu_admit_dttm"),
        (episodes, "episodes", "anchor_dttm"),
    ]:
        dtype = frame.schema[column]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise QualificationError(f"{name}.{column} must be timezone-aware UTC")

    observed = (
        events.join(
            episodes.select(
                "hospitalization_id",
                "icu_admit_dttm",
                "anchor_dttm",
                "eligible",
                "partition",
            ),
            left_on="hosp_id",
            right_on="hospitalization_id",
            how="inner",
        )
        .filter(
            pl.col("eligible")
            & (pl.col("dttm") >= pl.col("icu_admit_dttm"))
            & (pl.col("dttm") <= pl.col("anchor_dttm"))
        )
        .with_columns(
            (
                (pl.col("dttm") - pl.col("icu_admit_dttm")).dt.total_minutes()
            ).cast(pl.Int64).alias("pos_min"),
            (~pl.col("source").is_in(sorted(treatment_sources or set())))
            .alias("target_eligible"),
        )
    )
    return observed


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
        concept_edges = sorted({float(vals.quantile(q)) for q in qs})

        pinned = sorted({
            float(edge) for edge in forced_edges.get(concept, [])
            if np.isfinite(float(edge))
        })
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
    fixed_width = 2 * max(kernel_bins, 0) + 1

    def _uniform(bin_idx, w=1.0):
        """Pad a single assignment to `fixed_width` so every event yields a
        uniform-length list for the [B,T,K] dense tensor the encoder expects.
        Non-numeric events (hard_bin is None) and edgeless concepts land here."""
        bins = [bin_idx] * fixed_width
        weights = [w] + [0.0] * (fixed_width - 1)
        return list(zip(bins, weights))

    hard_bin = _bin_of(value, concept, edges)
    if hard_bin is None or kernel_bins <= 0:
        return _uniform(hard_bin)

    boundaries = edges[concept]
    if len(boundaries) < 2:
        return _uniform(hard_bin)

    lower = boundaries[hard_bin - 1] if hard_bin else None
    upper = boundaries[hard_bin] if hard_bin < len(boundaries) else None
    if lower is None and upper is not None:
        width = boundaries[1] - upper
        lower = upper - max(width, 1e-12)
    if upper is None and lower is not None:
        width = lower - boundaries[-2]
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
    padded_weights = [0.0] * width
    for idx, weight in zip(candidates, weights):
        slot = int(idx) - (hard_bin - half)
        if 0 <= slot < width:
            padded_bins[slot] = int(idx)
            padded_weights[slot] = float(weight)
    return list(zip(padded_bins, padded_weights))


def _check_single_hospital(con, base: Path) -> None:
    adt_path = base / "clif_adt.parquet"
    if not adt_path.exists():
        return
    n_hospitals = con.execute(
        f"SELECT COUNT(DISTINCT hospital_id) FROM read_parquet('{adt_path}')"
    ).fetchone()[0]
    if n_hospitals > 1:
        raise ValueError(
            f"site has {n_hospitals} distinct hospital_id values — "
            f"cross-hospital pooling violates the frozen-vocab contract. "
            f"Each hospital must be a separate site."
        )


def tokenize_site(cfg: dict, site: str, base: Path, out: Path,
                   vocab: dict | None, edges: dict | None,
                   limit_stays: int | None = None,
                   episodes: pl.DataFrame | None = None,
                   vocab_manifest: dict | None = None,
                   artifact_policy: dict | None = None):
    policy = artifact_policy or yaml.safe_load((ROOT / cfg["artifact_policy"]).read_text())
    events_path = out / "events.parquet"
    validate_artifact_destination(events_path, "patient_level_phi", policy)
    if vocab is not None:
        vocab, edges, vocab_manifest = validate_vocabulary_artifact(
            {"vocab": vocab, "edges": edges, "manifest": vocab_manifest}, cfg, policy
        )
    con = duckdb.connect()
    # DuckDB renders TIMESTAMPTZ columns in the SESSION timezone, so on a non-UTC
    # host every tz-aware parquet came back as e.g. America/Chicago and
    # restrict_to_observation_window refused it (fail closed, but host-dependent).
    # Pin the session so tokenization is byte-identical wherever it runs (U9).
    con.execute("SET TimeZone = 'UTC'")
    keep_ids = None
    if limit_stays is not None:
        hosp_spec = cfg["tables"].get("adt") or next(iter(cfg["tables"].values()))
        hosp_fp = base / f"{hosp_spec['file']}.parquet"
        keep_ids = [
            str(r[0]) for r in con.execute(
                "SELECT DISTINCT hospitalization_id "
                f"FROM read_parquet('{hosp_fp}') "
                f"WHERE hospitalization_id IS NOT NULL LIMIT {int(limit_stays)}"
            ).fetchall()
        ]
        print(f"  limiting to {len(keep_ids):,} stays (sample mode)")
    frames = []
    for name, spec in cfg["tables"].items():
        df = _read_table(con, base, spec, keep_ids=keep_ids)
        if len(df):
            df = df.with_columns(source=pl.lit(name))
            frames.append(df)
    if not frames:
        raise QualificationError("no configured CLIF event tables were found")
    events = pl.concat(frames, how="vertical_relaxed").sort(["hosp_id", "dttm"])
    validate_units(events, cfg)

    if episodes is None:
        raise QualificationError("a canonical episode/split artifact is required")
    treatment_sources = {
        name for name, spec in cfg["tables"].items() if spec.get("input_only")
    }
    events = restrict_to_observation_window(events, episodes, treatment_sources)

    # Guard: verify single-hospital consistency for reference-site vocab.
    # hospital_id is a CLIF 2.1 column that distinguishes hospitals within
    # a health system. Multi-hospital pooling under one vocab silently merges
    # different clinical workflows and populations.
    _check_single_hospital(con, base)
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
        fit_events = fit_partition(events, bin_cfg.get("fit_partition", "train"))
        edges = build_value_bins(fit_events, bin_cfg["n_bins"], bin_cfg.get("forced_edges"))
        vocab = build_vocab(fit_events, edges)
        cohort_cfg = yaml.safe_load((ROOT / cfg["cohort_contract"]).read_text())
        split_hashes = episodes["split_sha256"].drop_nulls().unique().to_list()
        if len(split_hashes) != 1:
            raise QualificationError("episode artifact must contain one split hash")
        hashes = {
            "training_split": split_hashes[0],
            "vocabulary": _json_sha256(vocab),
            "numeric_edges": _json_sha256(edges),
            "target_map": _json_sha256(cfg["target_concepts"]),
            "outcome_spec": _json_sha256(cohort_cfg["outcomes"]),
            "clif_version": _json_sha256(cfg["schema_version"]),
        }
        vocab_manifest = {
            "artifact_family": "experimental_representation",
            "clif_version": cfg["schema_version"],
            "mcide_version": cfg["mcide_version"],
            "hashes": hashes,
            "provenance": {
                "source_site": site,
                "fit_partition": cfg["value_binning"].get("fit_partition", "train"),
                "cohort_contract_version": episodes["cohort_contract_version"].item(0),
            },
        }
        print(f"  built vocab: {len(vocab):,} tokens, {len(edges):,} numeric concepts")
    elif vocab_manifest is None:
        raise QualificationError("an imported vocabulary requires a validated manifest")

    # Map each event using positions derived from canonical ICU admission.
    def encode(group: pl.DataFrame) -> pl.DataFrame:
        token, soft_token, soft_weight, valnum = [], [], [], []
        bin_cfg = cfg["value_binning"]
        for c, v in zip(group["concept"], group["value"]):
            v_for_bin = float(v) if v is not None and np.isfinite(v) else None
            if v_for_bin is None and c in edges:
                # Missing numeric measurements are not physiologic low-bin events.
                # They are skipped rather than converted to bin 0.
                continue
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
                _soft_bins(v_for_bin, c, edges, bin_cfg["soft_kernel_bins"])
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
            "pos_min": [group["pos_min"].to_list()],
            "value": [valnum],
            "target_eligible": [group["target_eligible"].to_list()],
            "partition": group["partition"][0],
            "n_events": len(token),
        })

    shards = events.group_by("hosp_id", maintain_order=True).map_groups(encode)
    out.mkdir(parents=True, exist_ok=True)
    shards.write_parquet(events_path)
    # DATA-CLASSIFICATION: PHI — contains hosp_id + per-stay token sequences + timing.
    # Do not export off-node. For external validation, use clif_validate.py which
    # returns only aggregate metrics. See NEXT_STEPS.md §6 rule 4.
    if edges is not None:
        (out / "vocab.json").write_text(
            json.dumps({"vocab": vocab, "edges": edges, "manifest": vocab_manifest})
        )
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
    ap.add_argument("--episodes", required=True,
                    help="canonical local episode/split parquet from configs/cohort.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    policy = yaml.safe_load((ROOT / cfg["artifact_policy"]).read_text())
    vocab, edges, vocab_manifest = None, None, None
    if args.vocab:
        blob = json.loads(Path(args.vocab).read_text())
        vocab, edges, vocab_manifest = validate_vocabulary_artifact(blob, cfg, policy)
    elif not args.build_vocab:
        raise SystemExit("pass --build-vocab (first site) or --vocab PATH (later sites)")

    if args.dry_run:
        con = duckdb.connect()
        con.execute("SET TimeZone = 'UTC'")  # same session-tz pin as tokenize_site
        for name, spec in cfg["tables"].items():
            df = _read_table(con, Path(args.indir), spec)
            print(f"{name}: {len(df):,} events, concepts={df['concept'].n_unique() if len(df) else 0}")
        return

    episodes = pl.read_parquet(args.episodes)
    tokenize_site(
        cfg,
        args.site,
        Path(args.indir),
        Path(args.out),
        vocab,
        edges,
        episodes=episodes,
        vocab_manifest=vocab_manifest,
        artifact_policy=policy,
    )


if __name__ == "__main__":
    main()
