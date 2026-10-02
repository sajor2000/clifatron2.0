"""CLIF 2.1 parquet -> per-site event-token shards + vocab.json.

Value bins are per-concept **clinician-designed segments** (the CLIF consortium's
`critical_illness_tokenization_final_with_intervals.csv`, 1268 segments across
labs/vitals) — NOT data-driven deciles. The scheme is physician-defined: normal
range subdivided by measurement density, above/below with progressively wider
intervals, extreme-value quintiles at the tails. Population deciles are retained
as an ablation arm (`scheme: decile_ablation` in data.yaml).

Under `value_binning.coverage: all` every numeric concept in the reference-site train
partition is binned (CSV segments, else ordinal point bins, else frozen quantiles; dose
concepts always get a [0,0] stop bin) and the per-concept choice is recorded in vocab.json
`binning_sources` (see `build_segments`).

Each event is ONE FUSED token: `concept=bin` (numeric), `concept=<value>` (a categorical
result on a row with no numeric value, from a table's `categorical_value_col`), or bare
`concept` (presence).
Position is minutes since ICU admission (`pos_min`). Events are ordered by their
availability timestamp (storetime semantics), then the stable tiebreak (source, concept,
value), sorted AFTER the observation-window join so identical input always yields an
identical stream (KTD6). Every table declares its availability semantics
(`availability: result | recorded | missing_storetime`) and an optional conservative
`availability_lag_minutes` applied before windowing (R12). Full spec, including known
issues: website/docs/data-tokenization.md.

Usage:
    python -m src.data.tokenize --site mimic --in $MIMIC_DIR --out output/intermediate_phi/mimic --build-vocab --episodes output/intermediate_phi/episodes.parquet
    python -m src.data.tokenize --site rush  --in $RUSH_DIR  --out output/intermediate_phi/rush --vocab output/intermediate_phi/mimic/vocab.json --episodes output/intermediate_phi/rush_episodes.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import yaml

from clif_validate._vendor.data.cohort import (
    QualificationError,
    validate_artifact_destination,
    validate_episode_artifact,
)
from clif_validate._vendor.data.segments import (
    POLICY_VERSION,
    as_segments,
    bin_index,
    dose_segments_from_edges,
    is_ordinal,
    load_csv_segments,
    n_bins as segment_count,
    ordinal_segments,
    segments_from_edges,
    soft_bins,
    with_zero_point,
)
from clif_validate._vendor.data.splits import fit_partition

SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
ROOT = Path(__file__).parents[2]
# value_binning.coverage: "all" bins every numeric concept in the reference-site train
# partition (KTD3); "targets_only" is the pre-v2 behaviour (CSV segments for the target
# concepts only, or the legacy decile arm), kept for back-compat comparisons.
COVERAGES = ("all", "targets_only")
BINNING_SOURCES = ("csv", "ordinal", "quantile", "single")
DEFAULT_DOSE_SOURCES = ("meds", "meds_intermittent")
# R12: what a table's `availability_col` means. `result` = the time a result became
# available (e.g. lab_result_dttm); `recorded` = the time the event was charted
# (e.g. admin_dttm, in_dttm); `missing_storetime` = CLIF carries no store time for the
# table, so `recorded_dttm` stands in and may precede true availability.
AVAILABILITY_SEMANTICS = ("result", "recorded", "missing_storetime")
# KTD6: the full event order within a stay. Applied after the observation-window join.
# `cat_value` is a trailing tiebreak so value-only categorical rows are ordered too.
EVENT_ORDER = ("hosp_id", "dttm", "source", "concept", "value", "cat_value")


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
    # Per-concept binning sources (U2) are bound when both the map and its hash travel;
    # the full v2 artifact contract (required hash, tokenizer version) is U5.
    if blob.get("binning_sources") is not None and "binning_sources" in hashes:
        if hashes["binning_sources"] != _json_sha256(blob["binning_sources"]):
            raise QualificationError("binning-sources hash mismatch")
    if hashes["clif_version"] != _json_sha256(cfg["schema_version"]):
        raise QualificationError("CLIF-version compatibility hash mismatch")
    if hashes.get("target_map") != _json_sha256(cfg["target_concepts"]):
        raise QualificationError("target-map compatibility hash mismatch")
    expected_fit_partition = cfg["value_binning"].get("fit_partition", "train")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("fit_partition") != expected_fit_partition:
        raise QualificationError(
            "vocabulary artifact was not fitted on the configured training partition"
        )
    cohort_cfg = yaml.safe_load((ROOT / cfg["cohort_contract"]).read_text())
    if hashes.get("outcome_spec") != _json_sha256(cohort_cfg["outcomes"]):
        raise QualificationError("outcome-spec compatibility hash mismatch")
    return blob["vocab"], blob["edges"], manifest


def validate_table_availability(tables: dict) -> dict[str, dict]:
    """Require an availability declaration on every configured table (R12).

    Returns ``{table: {"availability": semantics, "lag_minutes": lag}}``. A table with no
    (or an unknown) `availability`, or a lag that is not a non-negative integer number of
    minutes, fails closed: an undeclared table would silently be read as if its
    timestamp were the time the value was knowable."""
    if not isinstance(tables, dict) or not tables:
        raise QualificationError("data config declares no tables")
    declared: dict[str, dict] = {}
    for name, spec in tables.items():
        semantics = spec.get("availability") if isinstance(spec, dict) else None
        if semantics not in AVAILABILITY_SEMANTICS:
            raise QualificationError(
                f"table {name!r} must declare availability as one of "
                f"{', '.join(AVAILABILITY_SEMANTICS)} (got {semantics!r})"
            )
        lag = spec.get("availability_lag_minutes", 0)
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
            raise QualificationError(
                f"table {name!r} availability_lag_minutes must be a non-negative "
                f"integer (got {lag!r})"
            )
        declared[name] = {"availability": semantics, "lag_minutes": lag}
    return declared


def _read_table(con, base: Path, spec: dict,
                keep_ids: list | None = None) -> pl.DataFrame:
    """Melt one CLIF table to long events keyed by its availability timestamp.

    Row order is NOT meaningful here; `tokenize_site` imposes the full KTD6 order after
    the observation-window join.

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
                "cat_value": pl.String,
            }
        )
    val = spec.get("value_col")
    val_sql = f"CAST({val} AS DOUBLE)" if val else "NULL"
    unit = spec.get("unit_col")
    unit_sql = f"CAST({unit} AS VARCHAR)" if unit else "CAST('' AS VARCHAR)"
    # KTD4: a table may declare the column holding a categorical result (CLIF
    # `*_category` / assessment `categorical_value`). It becomes a fused token only on
    # rows without a finite numeric value; every other table carries a typed NULL so
    # the per-table frames concatenate.
    cat = spec.get("categorical_value_col")
    cat_sql = f"CAST({cat} AS VARCHAR)" if cat else "CAST(NULL AS VARCHAR)"
    time_col = spec["availability_col"]
    id_filter = ""
    params: list = []
    if keep_ids is not None:
        # An empty keep_ids would generate `IN ()`, which DuckDB rejects as a syntax
        # error (CodeRabbit). An empty allow-list means "keep nothing", so match no
        # rows explicitly rather than emitting invalid SQL.
        if not keep_ids:
            id_filter = "AND 1 = 0"
        else:
            # Parameterize the id list rather than inlining quoted literals: a
            # hospitalization_id containing an apostrophe would otherwise terminate the
            # literal and corrupt the query (CodeRabbit). Column/table identifiers
            # cannot be bound as parameters, so those stay interpolated — the bundle
            # path validates them as identifiers before they reach here.
            placeholders = ", ".join("?" for _ in keep_ids)
            id_filter = f"AND CAST(hospitalization_id AS VARCHAR) IN ({placeholders})"
            params = [str(i) for i in keep_ids]
    q = f"""
        SELECT hospitalization_id       AS hosp_id,
               {time_col}                AS dttm,
               {spec['concept_col']}     AS concept,
               {val_sql}                 AS value,
               {unit_sql}                AS unit,
               {cat_sql}                 AS cat_value
        FROM read_parquet('{fp}')
        WHERE {spec['concept_col']} IS NOT NULL
          AND {time_col} IS NOT NULL
          {id_filter}
    """
    return con.execute(q, params).pl()


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
    """Per-concept quantile edges (decile ablation arm only).

    The default scheme is clinical_segment (build_clinical_segment_bins); this function
    exists for the decile ablation arm. Finite forced edges are retained regardless
    of the reference data range so frozen vocab remains valid across sites."""
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
        edges[concept] = _quantile_edges(vals, n_bins, forced_edges.get(concept, []), concept)
    return edges


def _quantile_edges(vals: pl.Series, n_bins: int, forced: list[float],
                    concept: str) -> list[float]:
    """Deduplicated interior quantile edges with forced edges pinned (each replaces the
    nearest non-forced quantile edge, so `n_bins` is an upper bound)."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    concept_edges = sorted({float(vals.quantile(q)) for q in qs})

    pinned = sorted({float(edge) for edge in forced if np.isfinite(float(edge))})
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
    return concept_edges


def build_clinical_segment_bins(
    csv_path: str | Path,
    target_concepts: list[str],
    forced_edges: dict[str, list[float]] | None = None,
    directions: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """Per-concept closure-aware segments from the CLIF consortium's physician-designed
    segmentation CSV (`critical_illness_tokenization_final_with_intervals.csv`).

    These 1268 clinician-designed segments encode measurement-density granularity
    (tighter intervals in decision zones, extreme-value quintiles at the tails) that
    data-driven deciles cannot recover — the primary v2 scheme
    (`value_binning.scheme: clinical_segment`). The CSV interval flags are honoured
    and gaps/overlaps/forced edges resolve under precedence policy v1
    (`src/data/segments.py`); `directions` (target concept -> "above"/"below") sets
    the closure at forced edges so the threshold value lands on the non-event side."""
    return load_csv_segments(csv_path, target_concepts, forced_edges, directions)


def build_edges(bin_cfg: dict, fit_events: pl.DataFrame,
                target_concepts: list[str],
                directions: dict[str, str] | None = None,
                *, tables: dict | None = None) -> dict[str, list[dict]]:
    """Per-concept segments only; see `build_segments` for the binning sources."""
    return build_segments(bin_cfg, fit_events, target_concepts, directions, tables=tables)[0]


def _build_edges_targets_only(bin_cfg: dict, fit_events: pl.DataFrame,
                              target_concepts: list[str],
                              directions: dict[str, str] | None = None) -> dict[str, list[dict]]:
    """`value_binning.coverage: targets_only` (pre-v2 behaviour), dispatched on scheme.

    - clinical_segment: physician-designed CSV segments for the target concepts only.
    - decile / decile_ablation: population quantile edges (the Lee-2026 comparison arm)
      for every concept with at least `n_bins` values, as `[a, b)` segments with
      unbounded ends.

    Forced edges take outcome-direction closure from `directions` in both schemes.
    """
    scheme = bin_cfg.get("scheme", "clinical_segment")
    forced = bin_cfg.get("forced_edges") or {}
    directions = directions or {}
    if scheme in ("clinical_segment",):
        source = bin_cfg.get("segment_source")
        if not source:
            raise ValueError("value_binning.scheme=clinical_segment requires segment_source")
        return build_clinical_segment_bins(ROOT / source, target_concepts, forced, directions)
    if scheme in ("decile", "decile_ablation"):
        n_quantile_bins = bin_cfg.get("n_bins")
        if not n_quantile_bins:
            raise ValueError(f"value_binning.scheme={scheme} requires n_bins")
        return {
            concept: segments_from_edges(
                edges, forced.get(concept, ()), directions.get(concept)
            )
            for concept, edges in build_value_bins(fit_events, n_quantile_bins, forced).items()
        }
    raise ValueError(f"unknown value_binning.scheme: {scheme!r}")


def dose_concepts(fit_events: pl.DataFrame, bin_cfg: dict,
                  tables: dict | None = None) -> set[str]:
    """Concepts charted by a medication (dose) table: `value_binning.dose_sources` plus
    any table spec flagged `dose: true`. Dose concepts get a `[0, 0]` stop bin."""
    sources = set(bin_cfg.get("dose_sources", DEFAULT_DOSE_SOURCES) or ())
    sources |= {name for name, spec in (tables or {}).items()
                if isinstance(spec, dict) and spec.get("dose")}
    if not sources or "source" not in fit_events.columns:
        return set()
    return set(
        fit_events.filter(pl.col("source").is_in(sorted(sources)))["concept"]
        .unique().drop_nulls().to_list()
    )


def build_segments(bin_cfg: dict, fit_events: pl.DataFrame,
                   target_concepts: list[str],
                   directions: dict[str, str] | None = None,
                   *, tables: dict | None = None) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Segments for every numeric concept + its binning source (R4; KTD3).

    `fit_events` must already be the reference site's fit (train) partition. Under
    `coverage: all`, every concept with at least one finite training value is binned,
    choosing its source in priority order:

    1. ``csv``: the physician CSV defines the concept (clinical_segment scheme only).
    2. ``single``: fewer than `min_count` fitting values -> one segment (reported).
    3. ``ordinal``: integer-valued with <= `ordinal_max_distinct` distinct values -> one
       point segment per value (clinical_segment scheme only).
    4. ``quantile``: frozen quantile segments (`quantile_n_bins`; the decile arm uses
       `n_bins`), deduplicated, ``[a, b)`` closure, forced edges pinned.

    The decile arm (scheme decile/decile_ablation) forces every concept to quantile.
    Every dose concept (`dose_concepts`) gets a ``[0, 0]`` point segment and its
    quantiles are fit on strictly positive values only. Target concepts the CSV defines
    keep their segments even when absent from the fit partition (threshold queries).

    Returns ``({concept: segments}, {concept: source})``.
    """
    coverage = bin_cfg.get("coverage", "all")
    if coverage not in COVERAGES:
        raise ValueError(f"unknown value_binning.coverage {coverage!r}; expected one of {COVERAGES}")
    scheme = bin_cfg.get("scheme", "clinical_segment")
    if coverage == "targets_only":
        segments = _build_edges_targets_only(bin_cfg, fit_events, target_concepts, directions)
        source = "csv" if scheme == "clinical_segment" else "quantile"
        return segments, {concept: source for concept in segments}

    forced = bin_cfg.get("forced_edges") or {}
    directions = directions or {}
    if scheme == "clinical_segment":
        csv_source = bin_cfg.get("segment_source")
        if not csv_source:
            raise ValueError("value_binning.scheme=clinical_segment requires segment_source")
        n_quantile_bins = int(bin_cfg.get("quantile_n_bins", 10))
    elif scheme in ("decile", "decile_ablation"):
        csv_source = None
        if not bin_cfg.get("n_bins"):
            raise ValueError(f"value_binning.scheme={scheme} requires n_bins")
        n_quantile_bins = int(bin_cfg["n_bins"])
    else:
        raise ValueError(f"unknown value_binning.scheme: {scheme!r}")
    if n_quantile_bins < 2:
        raise ValueError(f"quantile bin target must be >= 2, got {n_quantile_bins}")
    min_count = int(bin_cfg.get("min_count", 20))
    max_distinct = int(bin_cfg.get("ordinal_max_distinct", 25))

    numeric = fit_events.filter(
        pl.col("value").is_not_null() & pl.col("value").is_finite()
        & pl.col("concept").is_not_null()
    )
    values = {
        concept: np.asarray(vals, dtype=float)
        for concept, vals in numeric.group_by("concept").agg(pl.col("value"))
        .select("concept", "value").iter_rows()
    }
    doses = dose_concepts(fit_events, bin_cfg, tables)
    csv_segments = (
        build_clinical_segment_bins(ROOT / csv_source, sorted(set(values) | set(target_concepts)),
                                    forced, directions)
        if csv_source else {}
    )

    segments: dict[str, list[dict]] = {}
    sources: dict[str, str] = {}
    for concept in sorted(set(values) | set(csv_segments)):
        is_dose = concept in doses
        concept_forced = forced.get(concept, ())
        direction = directions.get(concept)
        if concept in csv_segments:
            segs = csv_segments[concept]
            segments[concept] = with_zero_point(segs) if is_dose else segs
            sources[concept] = "csv"
            continue
        vals = np.sort(values[concept])
        fit_vals = vals[vals > 0] if is_dose else vals
        to_segments = dose_segments_from_edges if is_dose else segments_from_edges
        if len(fit_vals) < min_count:
            segments[concept] = to_segments([], concept_forced, direction)
            sources[concept] = "single"
        elif csv_source and is_ordinal(vals, max_distinct):
            # Point bins already separate every integer, so forced edges add nothing.
            points = [*vals.tolist(), 0.0] if is_dose else vals.tolist()
            segments[concept] = ordinal_segments(points)
            sources[concept] = "ordinal"
        else:
            edges = _quantile_edges(pl.Series(fit_vals), n_quantile_bins,
                                    list(concept_forced), concept)
            segments[concept] = to_segments(edges, concept_forced, direction)
            sources[concept] = "quantile"
    return segments, sources


def fused_token(concept: str, b: int | None) -> str:
    """One token per event: `concept=bin` if numeric, else bare `concept`."""
    return f"{concept}={b}" if b is not None else concept


_INTEGER_RE = re.compile(r"^[+-]?\d+$")


def categorical_token(concept: str, value: object) -> str | None:
    """Fused `concept=<value>` for a value-only categorical finding (KTD4).

    The value is normalized (strip, lowercase, whitespace runs -> ``_``). A bare-integer
    value would read as a bin index (`concept=2`), so it is emitted as `concept=cat_2`.
    Returns None for a missing/blank value."""
    if value is None:
        return None
    norm = re.sub(r"\s+", "_", str(value).strip().lower())
    if not norm:
        return None
    if _INTEGER_RE.match(norm):
        norm = f"cat_{norm}"
    return f"{concept}={norm}"


def _no_finite_value() -> pl.Expr:
    return ~pl.col("value").is_finite().fill_null(False)


def build_vocab(events: pl.DataFrame, edges: dict[str, list]) -> dict:
    """One id per FUSED token, from the fit (train) partition:

    - `concept=bin` for every segment of a binned concept (including binned concepts
      the fit partition lacks, e.g. a CSV target concept);
    - bare `concept` for a concept with no bins;
    - `concept=<value>` for every categorical value charted on a row without a finite
      numeric value (KTD4). Values never seen here map to `<unk>` at encode time."""
    vocab = dict(SPECIAL)
    nxt = len(vocab)
    categorical: dict[str, set[str]] = {}
    if "cat_value" in events.columns:
        rows = (
            events.filter(pl.col("cat_value").is_not_null() & _no_finite_value())
            .select("concept", "cat_value").unique()
        )
        for concept, value in rows.iter_rows():
            token = categorical_token(concept, value)
            if token is not None:
                categorical.setdefault(concept, set()).add(token)
    concepts = set(events["concept"].drop_nulls().unique().to_list()) | set(edges)
    for concept in sorted(concepts):
        if concept in edges:
            for b in range(segment_count(edges[concept])):          # one fused token per segment
                vocab[fused_token(concept, b)] = nxt
                nxt += 1
        else:
            vocab[fused_token(concept, None)] = nxt
            nxt += 1
        for token in sorted(categorical.get(concept, ())):
            if token not in vocab:
                vocab[token] = nxt
                nxt += 1
    return vocab


def _bin_of(value: float | None, concept: str, edges: dict[str, list]) -> int | None:
    """Delegates to `segments.bin_index` (the single binning function). `edges[concept]`
    is a segment list, or a legacy interior-edge list read with the `[a, b)` rule."""
    if value is None or concept not in edges:
        return None
    return bin_index(value, as_segments(edges[concept]))


def _soft_bins(value: float | None, concept: str, edges: dict[str, list],
               kernel_bins: int) -> list[tuple[int | None, float]]:
    """Return fixed-width (2*kernel_bins+1) assignments so every event produces
    a uniform-length list for the [B,T,K] dense tensor the encoder expects.
    Delegates to `segments.soft_bins`; unbinned concepts pad `(None, 1.0)`."""
    segments = as_segments(edges[concept]) if concept in edges else []
    return soft_bins(value, segments, kernel_bins)


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
                   artifact_policy: dict | None = None,
                   binning_sources: dict | None = None):
    policy = artifact_policy or yaml.safe_load((ROOT / cfg["artifact_policy"]).read_text())
    events_path = out / "events.parquet"
    validate_artifact_destination(events_path, "patient_level_phi", policy)
    if vocab is not None:
        vocab, edges, vocab_manifest = validate_vocabulary_artifact(
            {"vocab": vocab, "edges": edges, "manifest": vocab_manifest,
             "binning_sources": binning_sources},
            cfg, policy,
        )
    availability = validate_table_availability(cfg.get("tables"))
    con = duckdb.connect()
    # DuckDB renders TIMESTAMPTZ columns in the SESSION timezone, so on a non-UTC
    # host every tz-aware parquet came back as e.g. America/Chicago and
    # restrict_to_observation_window refused it (fail closed, but host-dependent).
    # Pin the session so tokenization is byte-identical wherever it runs (U9).
    con.execute("SET TimeZone = 'UTC'")
    keep_ids = None
    if limit_stays is not None:
        # A zero or negative sample size is a usage error, not "tokenize nothing":
        # fail fast rather than silently producing an empty shard (CodeRabbit).
        if limit_stays < 1:
            raise ValueError(f"limit_stays must be a positive integer, got {limit_stays}")
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
            lag = availability[name]["lag_minutes"]
            if lag:
                # Conservative availability (R12): the value becomes knowable `lag`
                # minutes after its timestamp, so shift BEFORE windowing — an event
                # whose shifted time passes the anchor is excluded, and a kept event
                # is positioned at its shifted time.
                df = df.with_columns(pl.col("dttm") + pl.duration(minutes=lag))
            df = df.with_columns(source=pl.lit(name))
            frames.append(df)
    if not frames:
        raise QualificationError("no configured CLIF event tables were found")
    # No sort here: join order is not guaranteed, so the order is imposed after it.
    events = pl.concat(frames, how="vertical_relaxed")
    validate_units(events, cfg)

    if episodes is None:
        raise QualificationError("a canonical episode/split artifact is required")
    treatment_sources = {
        name for name, spec in cfg["tables"].items() if spec.get("input_only")
    }
    events = restrict_to_observation_window(events, episodes, treatment_sources)
    # KTD6: full-key sort AFTER the join (polars does not guarantee row order for equal
    # keys through a join or an unmaintained sort). Nulls sort last so a missing value
    # or categorical result has one fixed place; `group_by(maintain_order=True)` below
    # hands `encode` each stay's rows in exactly this order.
    order = [key for key in EVENT_ORDER if key in events.columns]
    events = events.sort(order, nulls_last=True, maintain_order=True)

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
        target_concepts = [t["name"] for t in cfg.get("target_concepts", [])]
        # Fail closed on a missing/empty target_concepts under clinical_segment: an empty
        # list would silently build no edges, so every numeric event would collapse to a
        # bare categorical token — the configured clinical representation silently disabled.
        if bin_cfg.get("scheme", "clinical_segment") == "clinical_segment" and not target_concepts:
            raise QualificationError(
                "value_binning.scheme=clinical_segment requires a non-empty cfg.target_concepts; "
                "without it no clinical-segment edges are built and numeric events lose their value bins"
            )
        directions = {
            t["name"]: t["direction"] for t in cfg.get("target_concepts", []) if "direction" in t
        }
        edges, binning_sources = build_segments(
            bin_cfg, fit_events, target_concepts, directions, tables=cfg["tables"]
        )
        vocab = build_vocab(fit_events, edges)
        cohort_cfg = yaml.safe_load((ROOT / cfg["cohort_contract"]).read_text())
        split_hashes = episodes["split_sha256"].drop_nulls().unique().to_list()
        if len(split_hashes) != 1:
            raise QualificationError("episode artifact must contain one split hash")
        hashes = {
            "training_split": split_hashes[0],
            "vocabulary": _json_sha256(vocab),
            "numeric_edges": _json_sha256(edges),
            "binning_sources": _json_sha256(binning_sources),
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
                # Segment precedence policy (src/data/segments.py) the edges were built under.
                "precedence_policy": POLICY_VERSION,
                # R12: per-table availability semantics and lag, for the tokenization
                # report. Provenance only — deliberately not a compatibility hash.
                "availability": {
                    name: spec["availability"] for name, spec in availability.items()
                },
                "availability_lag_minutes": {
                    name: spec["lag_minutes"] for name, spec in availability.items()
                },
            },
        }
        by_source = {
            src: sum(1 for v in binning_sources.values() if v == src) for src in BINNING_SOURCES
        }
        print(f"  built vocab: {len(vocab):,} tokens, {len(edges):,} numeric concepts "
              f"(binning sources: {by_source})")
    elif vocab_manifest is None:
        raise QualificationError("an imported vocabulary requires a validated manifest")

    # The unknown-concept fallback below emits SPECIAL["<unk>"] for a concept+bin the
    # frozen vocab does not cover (cross-site coverage). That is only safe if the vocab
    # actually reserves that id for <unk>. A freshly built vocab always does (build_vocab
    # starts from dict(SPECIAL)); a legacy IMPORTED vocab predating the <unk> token would
    # map id 3 to a real token, so an unknown concept would silently become a valid token.
    # Fail closed rather than emit an incorrect or out-of-range id.
    if vocab.get("<unk>") != SPECIAL["<unk>"]:
        raise QualificationError(
            f"vocabulary must reserve id {SPECIAL['<unk>']} for '<unk>' "
            f"(found {vocab.get('<unk>')!r}); the unknown-concept fallback is unsafe otherwise"
        )

    # Map each event using positions derived from canonical ICU admission.
    def encode(group: pl.DataFrame) -> pl.DataFrame:
        token, soft_token, soft_weight, valnum = [], [], [], []
        # Positions and eligibility flags are collected INSIDE the loop, aligned with
        # `token`. A skipped event (missing numeric) drops its token, so emitting the
        # full `group["pos_min"]` / `group["target_eligible"]` instead would shift every
        # position and eligibility flag after the first skip and make n_events disagree
        # with the sequence length (CodeRabbit: critical desync).
        pos_min, target_eligible = [], []
        bin_cfg = cfg["value_binning"]
        soft_width = (
            2 * max(int(bin_cfg["soft_kernel_bins"]), 0) + 1
            if bin_cfg.get("soft_discretization") else 1
        )
        cat_values = (
            group["cat_value"] if "cat_value" in group.columns else [None] * len(group)
        )
        for c, v, cat, pos, eligible in zip(
            group["concept"], group["value"], cat_values,
            group["pos_min"], group["target_eligible"],
        ):
            v_for_bin = float(v) if v is not None and np.isfinite(v) else None
            # KTD4: a categorical value is fused only when the row has no finite numeric
            # value; a row WITH a number emits only its binned numeric token.
            cat_key = categorical_token(c, cat) if v_for_bin is None else None
            if cat_key is not None:
                hard_token = vocab.get(cat_key, SPECIAL["<unk>"])
                token.append(hard_token)
                soft_token.append([hard_token] * soft_width)
                soft_weight.append([1.0] + [0.0] * (soft_width - 1))
                pos_min.append(pos)
                target_eligible.append(eligible)
                valnum.append(float("nan"))
                continue
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
                    hard_token = vocab.get(key, SPECIAL["<unk>"])
            else:
                key = fused_token(c, b)
                hard_token = vocab.get(key, SPECIAL["<unk>"])
            if hard_token is None:
                hard_token = SPECIAL["<unk>"]
            token.append(hard_token)
            assignments = (
                _soft_bins(v_for_bin, c, edges, bin_cfg["soft_kernel_bins"])
                if bin_cfg.get("soft_discretization")
                else [(b, 1.0)]
            )
            soft_tokens, weights = [], []
            for soft_bin, weight in assignments:
                st_key = fused_token(c, soft_bin) if soft_bin is not None else fused_token(c, None)
                soft_tokens.append(vocab.get(st_key, SPECIAL["<unk>"]))
                weights.append(weight)
            soft_token.append(soft_tokens)
            soft_weight.append(weights)
            pos_min.append(pos)
            target_eligible.append(eligible)
            valnum.append(float(v) if v is not None else float("nan"))  # ORA value-regression target
        return pl.DataFrame({
            "hosp_id": group["hosp_id"][0],
            "token": [token],
            "soft_token": [soft_token],
            "soft_weight": [soft_weight],
            "pos_min": [pos_min],
            "value": [valnum],
            "target_eligible": [target_eligible],
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
        blob = {"vocab": vocab, "edges": edges, "manifest": vocab_manifest}
        if binning_sources is not None:
            blob["binning_sources"] = binning_sources
        (out / "vocab.json").write_text(json.dumps(blob))
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
    validate_table_availability(cfg.get("tables"))
    policy = yaml.safe_load((ROOT / cfg["artifact_policy"]).read_text())
    vocab, edges, vocab_manifest, binning_sources = None, None, None, None
    if args.vocab:
        blob = json.loads(Path(args.vocab).read_text())
        vocab, edges, vocab_manifest = validate_vocabulary_artifact(blob, cfg, policy)
        binning_sources = blob.get("binning_sources")
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
        binning_sources=binning_sources,
    )


if __name__ == "__main__":
    main()
