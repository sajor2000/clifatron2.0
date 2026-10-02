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

vocab.json is the tokenizer-v2 artifact (KTD7): `vocab`, `segments` (closure-aware, per
binned concept), `binning_sources`, `reference_units` (per binned concept, plus the dose
target units), `concept_sources` (charting tables + the input-only tables),
`precedence_policy`, and a `manifest` with `tokenizer_version: 2` and a SHA-256 for each
field (`numeric_edges` is the segments hash). Every events.parquet row carries the
binding (`artifact_hashes`); anything built by the previous tokenizer is refused with a
re-tokenize message.

Each event is ONE FUSED token: `concept=bin` (numeric), `concept=<value>` (a categorical
result on a row with no numeric value, from a table's `categorical_value_col`), or bare
`concept` (presence).

Event sources are config-declared (KTD5, U4): long tables (`concept_col`), wide tables
melted per column (`value_cols` / `categorical_value_cols`, optionally qualified by
`concept_qualifier_col`), medication doses converted to one unit per concept (`dose`,
`src/data/units.py`; per-kg via an availability-safe ASOF weight join), patient-keyed
state (`key: patient`, code status), state changes only (`emit: transitions`), state
carried to the window start (`carry_forward`), and static admission tokens
(`static_tokens`) at the stay start. Treatments and context are `input_only`.
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
    # GEM (U8): full-hospitalization windows beside events.parquet, same frozen vocab
    python -m src.data.tokenize --site mimic --in $MIMIC_DIR --out output/intermediate_phi/mimic --vocab output/intermediate_phi/mimic/vocab.json --episodes output/intermediate_phi/episodes.parquet --trajectory hospitalization
    # U7 verification sample (KTD9): N eligible ICU episodes, deterministic; the vocab is
    # marked provenance.sample: true and training refuses it (smoke/dry-run only)
    python -m src.data.tokenize --site mimic --in $MIMIC_DIR --out output/intermediate_phi/mimic_v2_sample --build-vocab --episodes output/intermediate_phi/episodes.parquet --sample-episodes 5000

Every run also writes an aggregate-only report beside its events
(`tokenization_report.json`; `gem_tokenization_report.json` for the GEM trajectory),
small cells suppressed, no identifiers (src/data/tokenization_report.py; gate:
`python -m src.data.tokenization_report --report <dir>/tokenization_report.json`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
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
    ADMISSION_PREFIX,
    DEVICE_METRIC_PREFIX,
    DISCHARGE_PREFIX,
    POLICY_VERSION,
    RETOKENIZE,
    SPECIAL,
    TOKENIZER_VERSION,
    UNK_ID,
    Binner,
    artifact_binding,
    as_segments,
    bin_index,
    binding_expr,
    dose_segments_from_edges,
    is_ordinal,
    json_sha256,
    load_csv_segments,
    n_bins as segment_count,
    ordinal_segments,
    segments_from_edges,
    soft_bins,
    soft_bins_at,
    validate_partition,
    with_zero_point,
)
from clif_validate._vendor.data.splits import fit_partition
from clif_validate._vendor.data.tokenization_report import (
    DEFAULT_CONTEXT,
    GEM_REPORT_FILE,
    REPORT_FILE,
    build_report,
    policy_min_cell,
    write_report,
)
from clif_validate._vendor.data.units import (
    STATUSES as DOSE_STATUSES,
    canonical_unit,
    dose_plan,
    normalize_name,
    normalize_unit,
    preferred_units_from_csv,
)

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
# `static_rank` (null for every non-static event, nulls last) puts the static admission
# tokens first at the stay start in their fixed order; `cat_value` is a trailing
# tiebreak so value-only categorical rows are ordered too.
EVENT_ORDER = ("hosp_id", "dttm", "static_rank", "source", "concept", "value", "cat_value")
EVENT_SCHEMA = {
    "hosp_id": pl.String,
    "dttm": pl.Datetime("us", "UTC"),
    "concept": pl.String,
    "value": pl.Float64,
    "unit": pl.String,
    "cat_value": pl.String,
}
# U4 (KTD5): table-spec enums. `dose.kind` selects the unit-conversion rule; `emit:
# transitions` keeps only a stay's first state and its changes; `key: patient` joins a
# patient-keyed table (code status) to stays through the hospitalization table.
DOSE_KINDS = ("continuous", "intermittent")
EMIT_MODES = ("all", "transitions")
TABLE_KEYS = ("hospitalization", "patient")
# R21 / KTD11: static admission tokens, in their fixed emission order:
# token -> (entity table, CLIF 2.1 column, numeric?). Numeric tokens are binned with
# frozen quantile (decile) segments; the rest are fused `token=value` categoricals.
STATIC_TOKENS = {
    "age_decile": ("hospitalization", "age_at_admission", True),
    "sex": ("patient", "sex_category", False),
    "race": ("patient", "race_category", False),
    "ethnicity": ("patient", "ethnicity_category", False),
    "admission_type": ("hospitalization", "admission_type_category", False),
}
STATIC_SOURCE = "static"
# U8 (R17, R18; KTD10): `icu_24h` = the 24 h prediction artifact (events.parquet, window
# [ICU admit, anchor], positions from ICU admission). `hospitalization` = the GEM
# artifact (gem_events.parquet, window [hospital admission, discharge], positions from
# hospital admission, framed by <bos> ADMISSION//x ... DISCHARGE//y <eos>).
TRAJECTORIES = ("icu_24h", "hospitalization")
GEM_EVENTS_FILE = "gem_events.parquet"
_GEM_LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# A table spec's literal concept / an MCS qualifier fallback, inlined into SQL as a string
# literal, so it must be a plain identifier (the bundle validator enforces the same).
_LITERAL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# KTD7: hashes every tokenizer-v2 vocabulary manifest carries, whatever the policy lists.
# `numeric_edges` is the segments hash.
V2_HASHES = ("vocabulary", "numeric_edges", "binning_sources", "reference_units",
             "concept_sources")
# R14: a binned concept's reference "unit" when the wide device table it comes from
# carries no unit column is `DEVICE_METRIC_PREFIX` + the device metric (column) it
# measures (src/data/segments.py, with the SPECIAL ids and the GEM prefixes).


def _validate_v2_fields(blob: dict) -> None:
    """Shape of the tokenizer-v2 fields (KTD7); their hashes are checked by the caller."""
    segments = blob["segments"]
    for concept, segs in segments.items():
        try:
            validate_partition(segs)
        except (TypeError, ValueError, KeyError) as exc:
            raise QualificationError(
                f"vocabulary segments for {concept!r} are not a valid partition: {exc}"
            ) from exc
    sources = blob.get("binning_sources")
    if (not isinstance(sources, dict) or set(sources) != set(segments)
            or any(v not in BINNING_SOURCES for v in sources.values())):
        raise QualificationError(
            "vocabulary binning sources must name exactly the binned concepts, each one of "
            f"{', '.join(BINNING_SOURCES)}"
        )
    units = blob.get("reference_units")
    if (not isinstance(units, dict) or not isinstance(units.get("concepts"), dict)
            or set(units["concepts"]) != set(segments)
            or not isinstance(units.get("dose_targets"), dict)):
        raise QualificationError(
            "vocabulary reference units must map every binned concept (concepts) and "
            "carry the dose target units (dose_targets)"
        )
    concept_sources = blob.get("concept_sources")
    if (not isinstance(concept_sources, dict)
            or not isinstance(concept_sources.get("tables"), dict)
            or not isinstance(concept_sources.get("treatment_sources"), list)):
        raise QualificationError(
            "vocabulary concept sources must carry tables and treatment_sources"
        )
    policy = blob.get("precedence_policy")
    recorded = ((blob["manifest"].get("provenance") or {}).get("precedence_policy", policy))
    if policy != POLICY_VERSION or recorded != POLICY_VERSION:
        raise QualificationError(
            f"vocabulary segments were built under precedence policy {policy!r}, not "
            f"{POLICY_VERSION}; {RETOKENIZE}"
        )


def validate_vocabulary_artifact(
    blob: dict,
    cfg: dict,
    policy: dict,
    *,
    expected_family: str = "experimental_representation",
) -> tuple[dict, dict, dict]:
    """Validate an imported tokenizer-v2 vocabulary artifact and its compatibility record.

    Returns ``(vocab, segments, manifest)``. An artifact built by the previous tokenizer
    (no ``tokenizer_version: 2``, ``edges`` instead of ``segments``) is refused with a
    re-tokenize message, never reused."""
    manifest = blob.get("manifest") if isinstance(blob, dict) else None
    if not isinstance(manifest, dict):
        raise QualificationError("vocabulary artifact is missing its manifest")
    version = manifest.get("tokenizer_version")
    if version != TOKENIZER_VERSION:
        raise QualificationError(
            f"vocabulary artifact was built by tokenizer version {version or 1}; this "
            f"build requires version {TOKENIZER_VERSION}: {RETOKENIZE}"
        )
    if not isinstance(blob.get("vocab"), dict) or not isinstance(blob.get("segments"), dict):
        raise QualificationError("vocabulary artifact must contain vocab and segments mappings")
    _validate_v2_fields(blob)
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
    missing = sorted((set(family["required_hashes"]) | set(V2_HASHES)) - set(hashes))
    if missing:
        raise QualificationError(f"vocabulary manifest is missing hashes: {', '.join(missing)}")
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise QualificationError("vocabulary manifest contains an invalid SHA-256 hash")
    if hashes["vocabulary"] != json_sha256(blob["vocab"]):
        raise QualificationError("vocabulary hash mismatch")
    if hashes["numeric_edges"] != json_sha256(blob["segments"]):
        raise QualificationError("numeric-edge hash mismatch")
    for field, label in (("binning_sources", "binning-sources"),
                         ("reference_units", "reference-units"),
                         ("concept_sources", "concept-sources")):
        if hashes[field] != json_sha256(blob[field]):
            raise QualificationError(f"{label} hash mismatch")
    if hashes["clif_version"] != json_sha256(cfg["schema_version"]):
        raise QualificationError("CLIF-version compatibility hash mismatch")
    if hashes.get("target_map") != json_sha256(cfg["target_concepts"]):
        raise QualificationError("target-map compatibility hash mismatch")
    expected_fit_partition = cfg["value_binning"].get("fit_partition", "train")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("fit_partition") != expected_fit_partition:
        raise QualificationError(
            "vocabulary artifact was not fitted on the configured training partition"
        )
    cohort_cfg = yaml.safe_load((ROOT / cfg["cohort_contract"]).read_text())
    if hashes.get("outcome_spec") != json_sha256(cohort_cfg["outcomes"]):
        raise QualificationError("outcome-spec compatibility hash mismatch")
    return blob["vocab"], blob["segments"], manifest


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


def validate_gem_config(gem: object) -> dict:
    """The `gem` block (U8): two label allowlists (`dispositions`, `admission_types`),
    each containing `unknown`, maps whose every value is in its allowlist, and a window
    size `max_tokens` >= 2. Fails closed: a malformed block would silently mint terminal
    tokens outside the frozen allowlist."""
    if not isinstance(gem, dict):
        raise QualificationError("data config has no `gem` block (terminal allowlists)")
    for labels_key, map_key in (("dispositions", "disposition_map"),
                                ("admission_types", "admission_map")):
        labels = gem.get(labels_key)
        if (not isinstance(labels, list) or len(set(labels)) != len(labels)
                or not all(isinstance(x, str) and _GEM_LABEL_RE.match(x) for x in labels)
                or "unknown" not in labels):
            raise QualificationError(
                f"gem.{labels_key} must be distinct lowercase labels including 'unknown'"
            )
        mapping = gem.get(map_key)
        if (not isinstance(mapping, dict)
                or any(v not in labels for v in mapping.values())
                or any(not isinstance(k, str) or k != k.strip().lower() for k in mapping)):
            raise QualificationError(
                f"gem.{map_key} keys must be lowercase raw categories and its values "
                f"members of gem.{labels_key}"
            )
    size = gem.get("max_tokens")
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise QualificationError("gem.max_tokens must be an integer >= 2")
    return gem


def _gem_label(raw: object, mapping: dict) -> str:
    key = "" if raw is None else str(raw).strip().lower()
    return mapping.get(key, "unknown") if key else "unknown"


def disposition_label(raw: object, gem: dict) -> str:
    """CLIF `discharge_category` -> one of `gem.dispositions`. Null, blank, `Missing` and
    any category the map does not name -> `unknown` (end of observation, never read as
    survival)."""
    return _gem_label(raw, gem["disposition_map"])


def admission_label(raw: object, gem: dict) -> str:
    """CLIF `admission_type_category` -> one of `gem.admission_types` (unmapped or
    missing -> `unknown`)."""
    return _gem_label(raw, gem["admission_map"])


def gem_tokens(gem: dict) -> list[str]:
    """The fixed GEM allowlist in vocabulary order: ADMISSION//* then DISCHARGE//*."""
    return ([f"{ADMISSION_PREFIX}{a}" for a in gem["admission_types"]]
            + [f"{DISCHARGE_PREFIX}{d}" for d in gem["dispositions"]])


def gem_window_bounds(n: int, max_tokens: int) -> list[tuple[int, int]]:
    """Consecutive `[start, end)` windows of at most `max_tokens` covering `n` tokens.
    The final `<eos>` is never left alone in a window, so the last window always ends
    `DISCHARGE//… <eos>` (the penultimate window gives up one token instead)."""
    if max_tokens < 2:
        raise ValueError(f"max_tokens must be >= 2, got {max_tokens}")
    bounds = [*range(0, n, max_tokens), n]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] == 1:
        bounds[-2] -= 1
    return list(zip(bounds[:-1], bounds[1:]))


def _empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema=EVENT_SCHEMA)


def _id_filter(keep_ids: list | None, column: str = "hospitalization_id") -> tuple[str, list]:
    """SQL fragment + bound parameters restricting `column` to `keep_ids`.

    An empty keep_ids would generate `IN ()`, which DuckDB rejects as a syntax error
    (CodeRabbit); an empty allow-list means "keep nothing", so match no rows explicitly.
    The ids are bound as parameters, never inlined: an id containing an apostrophe would
    otherwise terminate the literal and corrupt the query (CodeRabbit). Column/table
    identifiers cannot be bound, so those stay interpolated — the bundle path validates
    them as identifiers before they reach here."""
    if keep_ids is None:
        return "", []
    if not keep_ids:
        return "AND 1 = 0", []
    placeholders = ", ".join("?" for _ in keep_ids)
    return (f"AND CAST({column} AS VARCHAR) IN ({placeholders})",
            [str(i) for i in keep_ids])


SAMPLE_SALT = "clifatron2-verification-sample-v1"


def sample_episode_ids(episodes: pl.DataFrame, n: int) -> list[str]:
    """KTD9: a deterministic verification sample of `n` ELIGIBLE ICU episodes.

    Eligible hospitalization ids are ranked by SHA-256 of a fixed salt + the id (ties,
    impossible in practice, by the id) and the first `n` are taken. This spreads the
    sample across partitions and admission dates, never "the first n ids", and does not
    depend on the artifact's row order. Fewer than `n` eligible episodes -> all of them."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"sample_episodes must be a positive integer, got {n!r}")
    eligible = (episodes.filter(pl.col("eligible"))["hospitalization_id"]
                .cast(pl.String).drop_nulls().unique().to_list())

    def rank(identifier: str) -> tuple[str, str]:
        return hashlib.sha256(f"{SAMPLE_SALT}:{identifier}".encode()).hexdigest(), identifier

    return sorted(eligible, key=rank)[:n]


def _name_sql(expr: str) -> str:
    """SQL twin of `units.normalize_name`: lowercase, non-alphanumeric runs -> `_`."""
    return (f"regexp_replace(regexp_replace(lower(trim(CAST({expr} AS VARCHAR))), "
            f"'[^a-z0-9]+', '_', 'g'), '^_+|_+$', '', 'g')")


def _literal(value: object, what: str) -> str:
    if not isinstance(value, str) or not _LITERAL_RE.match(value):
        raise QualificationError(f"{what} must be a plain identifier, got {value!r}")
    return f"'{value}'"


def _long_sql(fp: Path, spec: dict, keep_ids: list | None) -> tuple[str, list]:
    """One event per row: concept from `concept_col` (or the literal `concept`), with an
    optional numeric `value_col`, `unit_col` and categorical `categorical_value_col`."""
    val = spec.get("value_col")
    val_sql = f"CAST({val} AS DOUBLE)" if val else "CAST(NULL AS DOUBLE)"
    unit = spec.get("unit_col")
    unit_sql = f"CAST({unit} AS VARCHAR)" if unit else "CAST('' AS VARCHAR)"
    # KTD4: a table may declare the column holding a categorical result (CLIF
    # `*_category` / assessment `categorical_value`). It becomes a fused token only on
    # rows without a finite numeric value; every other table carries a typed NULL so
    # the per-table frames concatenate.
    cat = spec.get("categorical_value_col")
    cat_sql = f"CAST({cat} AS VARCHAR)" if cat else "CAST(NULL AS VARCHAR)"
    time_col = spec["availability_col"]
    concept_col = spec.get("concept_col")
    if concept_col:
        concept_sql = _name_sql(concept_col) if spec.get("normalize_concept") else concept_col
        present = f"{concept_col} IS NOT NULL"
    else:
        # A single-concept table (position): every row is that concept, so a row
        # without a categorical (or numeric) value carries nothing.
        concept_sql = f"CAST({_literal(spec.get('concept'), 'concept')} AS VARCHAR)"
        present = " OR ".join(f"{c} IS NOT NULL" for c in (val, cat) if c) or "FALSE"
        present = f"({present})"
    id_sql, params = _id_filter(keep_ids)
    return f"""
        SELECT CAST(hospitalization_id AS VARCHAR) AS hosp_id,
               {time_col}                AS dttm,
               {concept_sql}             AS concept,
               {val_sql}                 AS value,
               {unit_sql}                AS unit,
               {cat_sql}                 AS cat_value
        FROM read_parquet('{fp}')
        WHERE {present}
          AND {time_col} IS NOT NULL
          {id_sql}
    """, params


def _melt_sql(fp: Path, spec: dict, keep_ids: list | None, cols: list, *,
              numeric: bool, qualifier_absent: bool = False) -> tuple[str, list]:
    """Wide -> long (R8, R10): one event per non-null cell of `cols`, the concept being
    the lowercased column name, qualified as `{qualifier}_{column}` when the table
    declares `concept_qualifier_col` (ECMO/MCS device group; a missing group, or a
    qualifier column the site's parquet lacks (`qualifier_absent`) -> `unknown`). One
    parquet scan (UNPIVOT drops NULL cells)."""
    time_col = spec["availability_col"]
    qual = None if qualifier_absent else spec.get("concept_qualifier_col")
    qual_sel = f"{qual} AS _qual," if qual else ""
    name = "'unknown_' || lower(_col)" if qualifier_absent else "lower(_col)"
    if qual:
        name = f"coalesce(nullif({_name_sql('_qual')}, ''), 'unknown') || '_' || lower(_col)"
    cast = "DOUBLE" if numeric else "VARCHAR"
    casts = ", ".join(f"CAST({c} AS {cast}) AS {c}" for c in cols)
    id_sql, params = _id_filter(keep_ids)
    value_sql, cat_sql = ("_val", "CAST(NULL AS VARCHAR)") if numeric else (
        "CAST(NULL AS DOUBLE)", "_val")
    return f"""
        SELECT hosp_id, dttm, {name} AS concept, {value_sql} AS value,
               CAST(NULL AS VARCHAR) AS unit, {cat_sql} AS cat_value
        FROM (
            UNPIVOT (
                SELECT CAST(hospitalization_id AS VARCHAR) AS hosp_id, {time_col} AS dttm,
                       {qual_sel} {casts}
                FROM read_parquet('{fp}')
                WHERE {time_col} IS NOT NULL {id_sql}
            ) ON {", ".join(cols)} INTO NAME _col VALUE _val
        )
    """, params


def _patient_keyed_sql(base: Path, fp: Path, spec: dict,
                       keep_ids: list | None) -> tuple[str, list] | None:
    """R21: a patient-keyed state table (code status) joined to stays.

    For each hospitalization, the latest state with `start <= admission` is emitted AT
    admission (all rows tied at that start, so the result is deterministic), and every
    state starting after admission (and, when `discharge_col` is set, no later than
    discharge) is emitted at its own start."""
    hosp_fp = base / f"{spec.get('hospitalization_file', 'clif_hospitalization')}.parquet"
    if not hosp_fp.exists():
        print(f"  [skip] {fp.name}: {hosp_fp.name} (patient -> stay join) not found")
        return None
    pid = spec.get("patient_id_col", "patient_id")
    adm = spec.get("admission_col", "admission_dttm")
    dis = spec.get("discharge_col")
    dis_sql = dis if dis else "CAST(NULL AS TIMESTAMPTZ)"
    time_col = spec["availability_col"]
    cat = spec["categorical_value_col"]
    concept = _literal(spec.get("concept"), "concept")
    id_sql, params = _id_filter(keep_ids)
    return f"""
        WITH s AS (
            SELECT CAST({pid} AS VARCHAR) AS pid, {time_col} AS t,
                   CAST({cat} AS VARCHAR) AS v
            FROM read_parquet('{fp}')
            WHERE {pid} IS NOT NULL AND {time_col} IS NOT NULL AND {cat} IS NOT NULL
        ), h AS (
            SELECT CAST(hospitalization_id AS VARCHAR) AS hosp_id,
                   CAST({pid} AS VARCHAR) AS pid, {adm} AS adm, {dis_sql} AS dis
            FROM read_parquet('{hosp_fp}')
            WHERE {pid} IS NOT NULL AND {adm} IS NOT NULL {id_sql}
        ), at_admission AS (
            SELECT h.hosp_id, h.adm AS dttm, s.v
            FROM h JOIN s ON h.pid = s.pid AND s.t <= h.adm
            QUALIFY s.t = max(s.t) OVER (PARTITION BY h.hosp_id)
        ), changes AS (
            SELECT h.hosp_id, s.t AS dttm, s.v
            FROM h JOIN s ON h.pid = s.pid AND s.t > h.adm
                          AND (h.dis IS NULL OR s.t <= h.dis)
        )
        SELECT hosp_id, dttm, CAST({concept} AS VARCHAR) AS concept,
               CAST(NULL AS DOUBLE) AS value, CAST(NULL AS VARCHAR) AS unit,
               v AS cat_value
        FROM (SELECT * FROM at_admission UNION ALL SELECT * FROM changes)
    """, params


def _transitions_sql(inner: str) -> str:
    """R21: keep a stay's first state and each row whose state differs from the previous
    row's. Rows are ordered by (dttm, normalized state); the state is the categorical
    value normalized like `categorical_token` (lowercase, whitespace runs -> `_`)."""
    return f"""
        SELECT hosp_id, dttm, concept, value, unit, cat_value FROM (
            SELECT *, lag(_state) OVER (
                PARTITION BY hosp_id, concept ORDER BY dttm, _state) AS _prev
            FROM (
                SELECT *, regexp_replace(lower(trim(cat_value)), '\\s+', '_', 'g') AS _state
                FROM ({inner}) WHERE cat_value IS NOT NULL
            )
        ) WHERE _prev IS NULL OR _prev <> _state
    """


def _read_dose_table(con, base: Path, fp: Path, spec: dict, keep_ids: list | None,
                     tables: dict | None, target_units: dict[str, str] | None,
                     fit_shadows: bool) -> tuple[pl.DataFrame, pl.DataFrame | None, dict]:
    """R6/R7 (KTD5): medication doses with unit conversion.

    SQL reads the doses (a stop action is a dose of 0) and, for continuous doses, ASOF
    joins the most recent weight whose availability time (charted time + the weight
    table's lag) is at or before the dose's (admin time + this table's lag). Each
    distinct (medication, unit) pair is then resolved ONCE by `units.dose_plan`, joined
    back and applied vectorized — never per row in Python.

    Returns (events, fit-only shadow events, {status: count}). Shadows (reference-site
    build only) are the native-unit fallback rows of every weight-dependent conversion,
    so fallback concepts get frozen bins even when this site converted every row."""
    dose = spec["dose"]
    kind = dose.get("kind")
    if kind not in DOSE_KINDS:
        raise QualificationError(f"dose.kind must be one of {DOSE_KINDS}, got {kind!r}")
    time_col, concept_col = spec["availability_col"], spec["concept_col"]
    val, unit = spec["value_col"], spec["unit_col"]
    dose_sql = f"CAST({val} AS DOUBLE)"
    params: list = []
    stops = [str(a).lower() for a in dose.get("stop_actions") or ()]
    action = dose.get("action_col")
    if action and stops:
        placeholders = ", ".join("?" for _ in stops)
        dose_sql = (f"CASE WHEN lower(CAST({action} AS VARCHAR)) IN ({placeholders}) "
                    f"THEN 0.0 ELSE {dose_sql} END")
        params += stops
    id_sql, id_params = _id_filter(keep_ids)
    params += id_params
    lag = int(spec.get("availability_lag_minutes", 0))
    doses = f"""
        SELECT CAST(hospitalization_id AS VARCHAR) AS hosp_id, {time_col} AS dttm,
               CAST({concept_col} AS VARCHAR) AS med, {dose_sql} AS dose,
               coalesce(CAST({unit} AS VARCHAR), '') AS unit_raw,
               {time_col} + INTERVAL {lag} MINUTE AS avail
        FROM read_parquet('{fp}')
        WHERE {concept_col} IS NOT NULL AND {time_col} IS NOT NULL {id_sql}
    """
    weight = dose.get("weight_source") if kind == "continuous" else None
    wspec = (tables or {}).get(weight["table"]) if weight else None
    wfp = base / f"{wspec['file']}.parquet" if wspec else None
    if weight and (wfp is None or not wfp.exists()):
        print(f"  [warn] {fp.name}: weight source {weight['table']!r} not found; "
              "per-kg conversions fall back to native units")
    if wfp is not None and wfp.exists():
        wtime, wval = wspec["availability_col"], wspec["value_col"]
        wlag = int(wspec.get("availability_lag_minutes", 0))
        w_id_sql, w_params = _id_filter(keep_ids)
        # One weight per (stay, availability time): ties are averaged so the ASOF match
        # is deterministic.
        sql = f"""
            WITH d AS ({doses}), w AS (
                SELECT hosp_id, avail, avg(weight_kg) AS weight_kg FROM (
                    SELECT CAST(hospitalization_id AS VARCHAR) AS hosp_id,
                           {wtime} + INTERVAL {wlag} MINUTE AS avail,
                           CAST({wval} AS DOUBLE) AS weight_kg
                    FROM read_parquet('{wfp}')
                    WHERE {wspec['concept_col']} = ? AND {wtime} IS NOT NULL
                      AND isfinite(CAST({wval} AS DOUBLE)) AND CAST({wval} AS DOUBLE) > 0
                      {w_id_sql}
                ) GROUP BY hosp_id, avail
            )
            SELECT d.hosp_id, d.dttm, d.med, d.dose, d.unit_raw, w.weight_kg
            FROM d ASOF LEFT JOIN w ON d.hosp_id = w.hosp_id AND d.avail >= w.avail
        """
        params += [weight["concept"], *w_params]
    else:
        sql = f"""SELECT hosp_id, dttm, med, dose, unit_raw,
                         CAST(NULL AS DOUBLE) AS weight_kg FROM ({doses})"""
    frame = con.execute(sql, params).pl()

    target_units = target_units or {}
    pairs = frame.select("med", "unit_raw").unique().iter_rows()
    plans = []
    for med, raw in pairs:
        target = (target_units.get(normalize_name(med)) if kind == "continuous"
                  else canonical_unit(raw))
        plan = dose_plan(med, raw, target)
        if kind == "intermittent" and target is None:
            plan["status"] = "unconvertible"   # `dose`, `mL`, unrecognised (R7)
        plans.append({"med": med, "unit_raw": raw, **plan,
                      "native_unit": plan["native_concept"][len(normalize_name(med)) + 1:]})
    mapping = pl.DataFrame(plans, schema={
        "med": pl.String, "unit_raw": pl.String, "native_concept": pl.String,
        "target_concept": pl.String, "factor": pl.Float64, "weight_power": pl.Int64,
        "status": pl.String, "target_unit": pl.String, "native_unit": pl.String,
    })
    needs_weight = pl.col("weight_power") != 0
    joined = frame.join(mapping, on=["med", "unit_raw"], how="left").with_columns(
        pl.when((pl.col("status") == "converted") & needs_weight
                & pl.col("weight_kg").is_null())
        .then(pl.lit("no_weight")).otherwise(pl.col("status")).alias("status")
    )
    converted = pl.col("status") == "converted"
    scale = (pl.when(needs_weight)
             .then(pl.col("factor") * pl.col("weight_kg").pow(pl.col("weight_power")))
             .otherwise(pl.col("factor")))
    native = pl.col("status").is_in(["no_weight", "unconvertible"])
    events = joined.select(
        "hosp_id", "dttm",
        pl.when(native).then(pl.col("native_concept"))
        .otherwise(pl.col("target_concept")).alias("concept"),
        pl.when(converted).then(pl.col("dose") * scale)
        .otherwise(pl.col("dose")).alias("value"),
        pl.when(converted).then(pl.col("target_unit"))
        .otherwise(pl.col("native_unit")).alias("unit"),
        pl.lit(None, dtype=pl.String).alias("cat_value"),
    )
    counts = dict.fromkeys(DOSE_STATUSES, 0)
    for status, n in joined.group_by("status").len().iter_rows():
        counts[status] = int(n)
    shadows = None
    if fit_shadows:
        shadows = joined.filter(converted & needs_weight).select(
            "hosp_id", "dttm", pl.col("native_concept").alias("concept"),
            pl.col("dose").alias("value"), pl.col("native_unit").alias("unit"),
            pl.lit(None, dtype=pl.String).alias("cat_value"),
        )
    return events, shadows, counts


def _parquet_columns(con, fp: Path) -> set[str]:
    return {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{fp}')").fetchall()}


# Optional per-site columns of a long/wide table spec: a site's parquet may lack any of
# them (synthetic CLIF releases omit 11 resp_support columns, the assessments
# categorical_value and ecmo fdO2), and the table is then read without them.
_OPTIONAL_LIST_COLS = ("value_cols", "categorical_value_cols")
_OPTIONAL_COLS = ("categorical_value_col", "concept_qualifier_col")


def _present_columns(con, fp: Path, spec: dict) -> tuple[dict, list[str]]:
    """`spec` restricted to the optional columns `fp` actually has (DuckDB identifiers
    are case-insensitive), plus the configured columns it lacks, in config order."""
    present = {c.lower() for c in _parquet_columns(con, fp)}
    spec, absent = dict(spec), []
    for key in _OPTIONAL_LIST_COLS:
        cols = list(spec.get(key) or ())
        absent += [c for c in cols if c.lower() not in present]
        spec[key] = [c for c in cols if c.lower() in present]
    for key in _OPTIONAL_COLS:
        col = spec.get(key)
        if col and col.lower() not in present:
            absent.append(col)
            spec[key] = None
    return spec, absent


def _read_source(con, base: Path, spec: dict, keep_ids: list | None = None, *,
                 tables: dict | None = None, target_units: dict[str, str] | None = None,
                 fit_shadows: bool = False, missing: list[str] | None = None,
                 ) -> tuple[pl.DataFrame, pl.DataFrame | None, dict | None]:
    """Read one configured table -> (events, fit-only shadows or None, dose status
    counts or None). See `_read_table`. Configured optional columns (`value_cols`,
    `categorical_value_cols`, `categorical_value_col`, `concept_qualifier_col`) the
    site's parquet lacks are skipped, logged, and appended to `missing`."""
    fp = base / f"{spec['file']}.parquet"
    if not fp.exists():
        print(f"  [skip] {fp.name} not found")
        return _empty_events(), None, None
    if spec.get("dose"):
        return _read_dose_table(con, base, fp, spec, keep_ids, tables, target_units,
                                fit_shadows)
    key = spec.get("key", "hospitalization")
    if key not in TABLE_KEYS:
        raise QualificationError(f"table key must be one of {TABLE_KEYS}, got {key!r}")
    emit = spec.get("emit", "all")
    if emit not in EMIT_MODES:
        raise QualificationError(f"table emit must be one of {EMIT_MODES}, got {emit!r}")
    parts: list[tuple[str, list]] = []
    absent: list[str] = []
    configured_qualifier = spec.get("concept_qualifier_col")
    if key == "patient":
        part = _patient_keyed_sql(base, fp, spec, keep_ids)
        if part is None:
            return _empty_events(), None, None
        parts.append(part)
    else:
        declared = any(spec.get(k) for k in ("concept_col", "concept", *_OPTIONAL_LIST_COLS))
        spec, absent = _present_columns(con, fp, spec)
        if absent:
            print(f"  [skip] {fp.name}: configured column(s) not found: {', '.join(absent)}")
            if missing is not None:
                missing.extend(absent)
        if spec.get("concept_col") or spec.get("concept"):
            parts.append(_long_sql(fp, spec, keep_ids))
    qualifier_absent = bool(configured_qualifier) and not spec.get("concept_qualifier_col")
    if spec.get("value_cols"):
        parts.append(_melt_sql(fp, spec, keep_ids, list(spec["value_cols"]), numeric=True,
                               qualifier_absent=qualifier_absent))
    if spec.get("categorical_value_cols"):
        parts.append(_melt_sql(fp, spec, keep_ids, list(spec["categorical_value_cols"]),
                               numeric=False, qualifier_absent=qualifier_absent))
    if not parts:
        if absent and declared:
            return _empty_events(), None, None   # every configured column is absent
        raise QualificationError(
            f"table {spec['file']!r} declares no concept source "
            "(concept_col, concept, value_cols or categorical_value_cols)"
        )
    sql = "\nUNION ALL\n".join(f"SELECT * FROM ({q})" for q, _ in parts)
    if emit == "transitions":
        sql = _transitions_sql(sql)
    params = [p for _, part_params in parts for p in part_params]
    return con.execute(sql, params).pl(), None, None


def _read_table(con, base: Path, spec: dict,
                keep_ids: list | None = None, *, tables: dict | None = None,
                target_units: dict[str, str] | None = None) -> pl.DataFrame:
    """Melt one CLIF table to long events keyed by its availability timestamp.

    A table spec declares its concepts one or more ways (KTD5): `concept_col` (one event
    per row; `normalize_concept` lowercases it), a literal `concept`, `value_cols` /
    `categorical_value_cols` (wide tables melted per column, optionally qualified by
    `concept_qualifier_col`), a `dose` block (unit conversion, R6/R7), `key: patient`
    (patient-keyed state, joined to stays) and `emit: transitions` (state changes only).

    Row order is NOT meaningful here; `tokenize_site` imposes the full KTD6 order after
    the observation-window join.

    If `keep_ids` is given, only rows for those hospitalization_ids are read
    (pushed into the SQL WHERE so the 45M+ row tables are filtered on scan, not
    after loading). Used to tokenize a small sample fast for smoke tests / dev."""
    return _read_source(con, base, spec, keep_ids, tables=tables,
                        target_units=target_units)[0]


def _read_static(con, base: Path, cfg: dict, keep_ids: list | None) -> pl.DataFrame:
    """R21: one row per (stay, static token) -> hosp_id, concept, value, cat_value,
    static_rank. Columns or entity files a site lacks are skipped (a missing value emits
    no token); the stay-start time is attached by `tokenize_site`."""
    tokens = list(cfg.get("static_tokens") or ())
    unknown = sorted(set(tokens) - set(STATIC_TOKENS))
    if unknown:
        raise QualificationError(f"unknown static_tokens: {', '.join(unknown)}")
    schema = {"hosp_id": pl.String, "concept": pl.String, "value": pl.Float64,
              "cat_value": pl.String, "static_rank": pl.UInt8}
    if not tokens:
        return pl.DataFrame(schema=schema)
    source = cfg.get("static_source") or {}
    hosp_fp = base / f"{source.get('hospitalization_file', 'clif_hospitalization')}.parquet"
    pat_fp = base / f"{source.get('patient_file', 'clif_patient')}.parquet"
    if not hosp_fp.exists():
        print(f"  [skip] static tokens: {hosp_fp.name} not found")
        return pl.DataFrame(schema=schema)

    hosp_cols = _parquet_columns(con, hosp_fp)
    pat_cols = (_parquet_columns(con, pat_fp) if pat_fp.exists() and "patient_id" in hosp_cols
                else set())
    select = []
    for token in tokens:
        entity, col, _ = STATIC_TOKENS[token]
        if col in (hosp_cols if entity == "hospitalization" else pat_cols):
            select.append(f"{'h' if entity == 'hospitalization' else 'p'}.{col} AS {token}")
        else:
            print(f"  [skip] static token {token}: {col} not found")
    if not select:
        return pl.DataFrame(schema=schema)
    pat_join = ""
    if pat_cols:
        pat_select = ", ".join(f"min({c}) AS {c}" for c in sorted(pat_cols - {"patient_id"})
                               if c in {v[1] for v in STATIC_TOKENS.values()})
        pat_join = f"""LEFT JOIN (
            SELECT CAST(patient_id AS VARCHAR) AS patient_id, {pat_select}
            FROM read_parquet('{pat_fp}') GROUP BY 1
        ) p ON CAST(h.patient_id AS VARCHAR) = p.patient_id"""
    id_sql, params = _id_filter(keep_ids, "h.hospitalization_id")
    wide = con.execute(f"""
        SELECT CAST(h.hospitalization_id AS VARCHAR) AS hosp_id, {", ".join(select)}
        FROM read_parquet('{hosp_fp}') h {pat_join}
        WHERE h.hospitalization_id IS NOT NULL {id_sql}
    """, params).pl()
    frames = []
    for token in tokens:
        if token not in wide.columns:
            continue
        rank = list(STATIC_TOKENS).index(token)
        numeric = STATIC_TOKENS[token][2]
        col = pl.col(token)
        frames.append(wide.filter(col.is_not_null()).select(
            "hosp_id", pl.lit(token).alias("concept"),
            (col.cast(pl.Float64) if numeric else pl.lit(None, pl.Float64)).alias("value"),
            (pl.lit(None, pl.String) if numeric else col.cast(pl.String)).alias("cat_value"),
            pl.lit(rank, pl.UInt8).alias("static_rank"),
        ))
    return pl.concat(frames) if frames else pl.DataFrame(schema=schema)


def _carry_forward(events: pl.DataFrame, episodes: pl.DataFrame,
                   sources: set[str], start_col: str = "icu_admit_dttm") -> pl.DataFrame:
    """State tables (`carry_forward: true`; code status, position): of each stay's rows
    charted BEFORE the observation window opens (`start_col`: ICU admission for the 24 h
    artifact, hospital admission for GEM), only the last state per concept is kept,
    moved to the window start. The state was already knowable then, so this never moves
    information earlier; without it a code status set at hospital admission would
    vanish from an ICU stay that starts later."""
    if not sources or events.is_empty():
        return events
    is_state = pl.col("source").is_in(sorted(sources))
    state = events.filter(is_state).join(
        episodes.select(pl.col("hospitalization_id").alias("hosp_id"),
                        pl.col(start_col).alias("_start")),
        on="hosp_id", how="left",
    )
    early = (pl.col("dttm") < pl.col("_start")).fill_null(False)
    carried = (
        state.filter(early)
        .sort(["hosp_id", "source", "concept", "dttm", "cat_value", "value"],
              nulls_last=True, maintain_order=True)
        .group_by(["hosp_id", "source", "concept"], maintain_order=True).last()
        .with_columns(pl.col("_start").alias("dttm"))
    )
    return pl.concat(
        [events.filter(~is_state), state.filter(~early).drop("_start"),
         carried.drop("_start").select(state.drop("_start").columns)],
        how="diagonal_relaxed",
    )


def _dose_target_units(cfg: dict, vocab_artifact: dict | None) -> dict[str, str]:
    """The continuous-dose target unit per med_category (R6). Building: the physician
    CSV's medication rows. Importing: the units the frozen vocabulary was built with
    (hashed `reference_units.dose_targets`), so every site converts to the same
    concepts."""
    if vocab_artifact is not None:
        return dict(vocab_artifact["reference_units"]["dose_targets"])
    source = cfg.get("value_binning", {}).get("segment_source")
    if not source or not (ROOT / source).exists():
        return {}
    return preferred_units_from_csv(ROOT / source)


def _normalized_unit(unit: str) -> str:
    """Spelling-insensitive unit key: `mm Hg` == `mmHg`, and a dose concept's suffix-form
    reference unit (`mg_hr`) == the charted `mg/hour`, so re-importing the reference
    site's own vocab never fails on spelling alone."""
    u = unit.strip().lower().replace("¬µ", "u").replace("µ", "u").replace("μ", "u")
    u = normalize_unit(u.replace("k/ul", "10^3/ul"))
    return re.sub(r"[\s_/]+", "", u)


def validate_units(events: pl.DataFrame, cfg: dict,
                   reference_units: dict | None = None) -> list[str]:
    """Fail closed (under `unit_normalization.on_mismatch: error`) on a charted unit that
    is not the expected one. Expected units: the config's canonical units, plus — when
    importing a frozen vocabulary — the reference site's unit for EVERY binned concept
    (`reference_units.concepts`), so a non-target concept charted in another unit is
    never silently binned against the reference site's segments. Device metrics and
    unit-less concepts are not unit-checked. Returns the (concept-level, aggregate)
    mismatch descriptions for the tokenization report when it does not raise."""
    mismatches = unit_mismatches(events, cfg, reference_units)
    if mismatches and cfg.get("unit_normalization", {}).get("on_mismatch") == "error":
        raise ValueError("Non-canonical CLIF units: " + "; ".join(mismatches))
    return mismatches


def unit_mismatches(events: pl.DataFrame, cfg: dict,
                    reference_units: dict | None = None) -> list[str]:
    """Sorted ``"concept: expected 'u', found 'v'"`` for every charted unit that differs
    from its expected unit (see `validate_units`)."""
    expected = {
        concept: unit
        for concept, unit in ((reference_units or {}).get("concepts") or {}).items()
        if isinstance(unit, str) and unit and not unit.startswith(DEVICE_METRIC_PREFIX)
    }
    expected.update(cfg.get("unit_normalization", {}).get("concepts", {}) or {})
    observed = events.filter(pl.col("unit").is_not_null()).select("concept", "unit").unique()
    mismatches = [
        f"{concept}: expected {expected[concept]!r}, found {unit!r}"
        for concept, unit in observed.iter_rows()
        if concept in expected and unit not in (None, "")
        and _normalized_unit(unit) != _normalized_unit(expected[concept])
    ]
    return sorted(mismatches)


def reference_units(fit_events: pl.DataFrame, segments: dict, cfg: dict,
                    dose_targets: dict[str, str]) -> dict:
    """R14: the reference unit of every binned concept, plus the dose target units.

    Per concept: the config's canonical unit; else the reference site's most frequent
    charted unit (ties -> lexicographically first); else, for a wide device table
    qualified by `concept_qualifier_col` (ECMO/MCS), ``device_metric:<column>``; else
    None. Hashed into the vocabulary manifest so every site checks units identically."""
    canonical = cfg.get("unit_normalization", {}).get("concepts", {}) or {}
    observed: dict[str, str] = {}
    if "unit" in fit_events.columns:
        counts = (
            fit_events.filter(pl.col("unit").is_not_null() & (pl.col("unit") != ""))
            .group_by("concept", "unit").len()
            .sort(["concept", "len", "unit"], descending=[False, True, False])
        )
        for concept, unit, _ in counts.iter_rows():
            observed.setdefault(concept, unit)
    metrics: dict[str, str] = {}
    sources = _concept_tables(fit_events)
    for name, spec in (cfg.get("tables") or {}).items():
        if not spec.get("concept_qualifier_col"):
            continue
        for col in spec.get("value_cols") or ():
            suffix = f"_{col.lower()}"
            for concept, tables in sources.items():
                if name in tables and concept.endswith(suffix):
                    metrics.setdefault(concept, f"{DEVICE_METRIC_PREFIX}{col}")
    units = {
        concept: canonical.get(concept) or observed.get(concept) or metrics.get(concept)
        for concept in sorted(segments)
    }
    return {"concepts": units, "dose_targets": dict(sorted(dose_targets.items()))}


def _concept_tables(fit_events: pl.DataFrame) -> dict[str, list[str]]:
    if fit_events.is_empty() or "source" not in fit_events.columns:
        return {}
    grouped = (
        fit_events.filter(pl.col("concept").is_not_null())
        .group_by("concept").agg(pl.col("source").unique().sort())
        .sort("concept")
    )
    return {concept: list(tables) for concept, tables in grouped.iter_rows()}


def concept_sources(fit_events: pl.DataFrame, treatment_sources: set[str]) -> dict:
    """Per concept, the table(s) charting it in the fit partition, plus the input-only
    (treatment/device/context) tables — what evaluation groups tokens by (KTD7)."""
    return {"tables": _concept_tables(fit_events),
            "treatment_sources": sorted(treatment_sources)}


def _check_window_inputs(events: pl.DataFrame, episodes: pl.DataFrame,
                         episode_times: list[str]) -> None:
    validate_episode_artifact(episodes)
    if events.schema.get("hosp_id") != pl.String:
        raise QualificationError("events.hosp_id must be a string identifier")
    if events["hosp_id"].has_nulls():
        raise QualificationError("events.hosp_id contains null identifiers")
    missing = sorted(set(episode_times) - set(episodes.columns))
    if missing:
        raise QualificationError(
            f"episode artifact is missing required columns: {', '.join(missing)}")
    for frame, name, column in [(events, "events", "dttm"),
                                *((episodes, "episodes", c) for c in episode_times)]:
        dtype = frame.schema[column]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
            raise QualificationError(f"{name}.{column} must be timezone-aware UTC")


def _windowed(events: pl.DataFrame, episodes: pl.DataFrame,
              treatment_sources: set[str] | None, times: tuple[str, ...], end: str,
              *extra: pl.Expr) -> pl.DataFrame:
    """Join the canonical episodes (their `times` columns) and keep each ELIGIBLE stay's
    events in the window ``[times[0], end]`` (inclusive); `pos_min` = minutes since
    ``times[0]``, then `target_eligible` and any `extra` columns. Ineligible episodes
    are dropped before the join (after the full artifact is validated) — the same rows
    the post-join eligibility filter keeps."""
    _check_window_inputs(events, episodes, list(times))
    start = times[0]
    return (
        events.join(
            episodes.filter(pl.col("eligible"))
            .select("hospitalization_id", *times, "eligible", "partition"),
            left_on="hosp_id", right_on="hospitalization_id", how="inner",
        )
        .filter(
            pl.col("eligible")
            & (pl.col("dttm") >= pl.col(start))
            & (pl.col("dttm") <= pl.col(end))
        )
        .with_columns(
            (pl.col("dttm") - pl.col(start)).dt.total_minutes()
            .cast(pl.Int64).alias("pos_min"),
            (~pl.col("source").is_in(sorted(treatment_sources or set())))
            .alias("target_eligible"),
            *extra,
        )
    )


def restrict_to_hospitalization_window(
    events: pl.DataFrame,
    episodes: pl.DataFrame,
    treatment_sources: set[str] | None = None,
) -> pl.DataFrame:
    """GEM (U8): join canonical episodes and retain each ELIGIBLE stay's events from
    hospital admission to discharge (inclusive) — pre-ICU (ED/ward), ICU and post-ICU.
    `pos_min` = minutes since hospital admission; `_le_anchor` marks events at or before
    the ICU-admit+24 h anchor (for the record's `anchor_idx`)."""
    return _windowed(events, episodes, treatment_sources,
                     ("admission_dttm", "discharge_dttm", "anchor_dttm"), "discharge_dttm",
                     (pl.col("dttm") <= pl.col("anchor_dttm")).alias("_le_anchor"))


def restrict_to_observation_window(
    events: pl.DataFrame,
    episodes: pl.DataFrame,
    treatment_sources: set[str] | None = None,
) -> pl.DataFrame:
    """Join canonical episodes and retain only anchor-available ICU events."""
    return _windowed(events, episodes, treatment_sources, ("icu_admit_dttm", "anchor_dttm"),
                     "anchor_dttm")


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
    by_concept = numeric.partition_by("concept", as_dict=True)
    for concept in numeric["concept"].unique().to_list():
        if concept is None:
            continue  # a null concept matches no `== concept` filter: never binned
        vals = by_concept[(concept,)]["value"].drop_nulls()
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
    and gaps/overlaps/forced edges resolve under the precedence policy
    (`src/data/segments.py`; dose concepts get step 8 in `build_segments`); `directions` (target concept -> "above"/"below") sets
    the closure at forced edges so the threshold value lands on the non-event side."""
    return load_csv_segments(csv_path, target_concepts, forced_edges, directions)


def build_edges(bin_cfg: dict, fit_events: pl.DataFrame,
                target_concepts: list[str],
                directions: dict[str, str] | None = None,
                *, tables: dict | None = None,
                quantile_concepts: set[str] | None = None) -> dict[str, list[dict]]:
    """Per-concept segments only; see `build_segments` for the binning sources."""
    return build_segments(bin_cfg, fit_events, target_concepts, directions, tables=tables,
                          quantile_concepts=quantile_concepts)[0]


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
                   *, tables: dict | None = None,
                   quantile_concepts: set[str] | None = None,
                   ) -> tuple[dict[str, list[dict]], dict[str, str]]:
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
    Every dose concept (`dose_concepts`) gets a ``[0, 0]`` point segment with no gap
    above it (policy step 8: a running dose never bins as stopped) and its quantiles
    are fit on strictly positive values only. Target concepts the CSV defines
    keep their segments even when absent from the fit partition (threshold queries).
    `quantile_concepts` skip the ordinal rule (the static `age_decile` token is deciles
    of age even when a small fit set happens to be integer-valued with few distinct ages).

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
        elif (csv_source and concept not in (quantile_concepts or ())
              and is_ordinal(vals, max_distinct)):
            # Point bins already separate every integer, so forced edges add nothing.
            points = ordinal_segments(vals.tolist())
            # Policy step 8: a dose's stop bin, with no gap to its first positive point.
            segments[concept] = with_zero_point(points) if is_dose else points
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


# Default categorical-value floor (distinct fit stays): the artifact policy's
# aggregate_no_phi.minimum_cell_size, which `tokenize_site` passes explicitly.
MIN_CATEGORY_STAYS = 10


def _categorical_tokens(events: pl.DataFrame, min_stays: int) -> dict[str, set[str]]:
    """``{concept: {concept=<value>}}`` for every categorical value (normalized by
    `categorical_token`) charted on a row without a finite numeric value in at least
    `min_stays` distinct stays of `events`."""
    if "cat_value" not in events.columns:
        return {}
    rows = events.filter(pl.col("cat_value").is_not_null() & _no_finite_value())
    pairs = rows.select("concept", "cat_value").unique()
    mapping = pl.DataFrame(
        [(c, v, categorical_token(c, v)) for c, v in pairs.iter_rows()],
        schema={"concept": pl.String, "cat_value": pl.String, "token": pl.String},
        orient="row",
    ).filter(pl.col("token").is_not_null())
    if "hosp_id" not in rows.columns:
        raise ValueError("categorical vocabulary tokens are counted by distinct stay; "
                         "the fit events need hosp_id")
    stays = (rows.select("concept", "cat_value", "hosp_id").unique()
             .join(mapping, on=["concept", "cat_value"], how="inner")
             .group_by("concept", "token").agg(pl.col("hosp_id").n_unique().alias("n")))
    categorical: dict[str, set[str]] = {}
    for concept, token, n in stays.iter_rows():
        if n >= min_stays:
            categorical.setdefault(concept, set()).add(token)
    return categorical


def build_vocab(events: pl.DataFrame, edges: dict[str, list], *,
                min_category_stays: int = MIN_CATEGORY_STAYS) -> dict:
    """One id per FUSED token, from the fit (train) partition:

    - `concept=bin` for every segment of a binned concept (including binned concepts
      the fit partition lacks, e.g. a CSV target concept);
    - bare `concept` for a concept with no bins;
    - `concept=<value>` for every categorical value charted on a row without a finite
      numeric value (KTD4) in at least `min_category_stays` distinct fit stays (the
      artifact policy's minimum cell size). vocab.json ships to every site in the
      bundle, so a value charted for fewer patients never leaves the node verbatim; it
      maps to `<unk>` at encode time, like a value never seen here. The floor applies
      to every categorical-value token, controlled CLIF `*_category` values and the
      static admission categories included; the fixed ADMISSION// / DISCHARGE//
      allowlist (`gem_tokens`) is configuration, not data, and is not subject to it."""
    vocab = dict(SPECIAL)
    nxt = len(vocab)
    categorical = _categorical_tokens(events, min_category_stays)
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


def _extend_for_gem(gem_cfg: dict, gem_fit: pl.DataFrame, fit_events: pl.DataFrame,
                    vocab: dict, edges: dict, binning_sources: dict, units: dict, *,
                    bin_cfg: dict, cfg: dict, directions: dict, target_units: dict,
                    treatment_sources: set[str], quantile_concepts: set[str],
                    min_category_stays: int = MIN_CATEGORY_STAYS,
                    ) -> tuple[dict, dict, dict, dict, dict]:
    """U8: extend the 24 h-fit vocabulary so the GEM artifact shares it.

    Every concept the 24 h train window charts keeps its segments, binning source,
    reference unit and token ids. Concepts charted ONLY in the full-hospitalization
    train events (`gem_fit`) get segments by the same source priority (`build_segments`)
    fit on those events; their tokens, and categorical values of existing concepts seen
    only outside the 24 h window (each in at least `min_category_stays` distinct
    full-hospitalization train stays), are appended after every 24 h token. The fixed
    ADMISSION// / DISCHARGE// allowlist is appended last, present even when no stay has
    that admission type or disposition. Returns ``(vocab, segments, binning_sources,
    reference_units, concept_sources)``."""
    known = set(fit_events["concept"].drop_nulls().unique().to_list()) | set(edges)
    extra_fit = gem_fit.filter(pl.col("concept").is_not_null()
                               & ~pl.col("concept").is_in(sorted(known)))
    extra_edges: dict = {}
    if not extra_fit.is_empty():
        built, built_sources = build_segments(
            bin_cfg, extra_fit, [], directions, tables=cfg["tables"],
            quantile_concepts=quantile_concepts)
        extra_edges = {c: segs for c, segs in built.items() if c not in edges}
        if extra_edges:
            extra_units = reference_units(extra_fit, extra_edges, cfg, target_units)
            edges = dict(sorted({**edges, **extra_edges}.items()))
            binning_sources = dict(sorted(
                {**binning_sources, **{c: built_sources[c] for c in extra_edges}}.items()))
            units = {"concepts": dict(sorted({**units["concepts"],
                                              **extra_units["concepts"]}.items())),
                     "dose_targets": units["dose_targets"]}
    vocab = dict(vocab)
    nxt = max(vocab.values()) + 1
    for token in [*build_vocab(gem_fit, edges, min_category_stays=min_category_stays),
                  *gem_tokens(gem_cfg)]:
        if token not in vocab:
            vocab[token] = nxt
            nxt += 1
    tables = _concept_tables(fit_events)
    for concept, names in _concept_tables(gem_fit).items():
        tables[concept] = sorted(set(tables.get(concept, ())) | set(names))
    sources = {"tables": dict(sorted(tables.items())),
               "treatment_sources": sorted(treatment_sources)}
    return vocab, edges, binning_sources, units, sources


GEM_SCHEMA = {
    "hosp_id": pl.String,
    "trajectory": pl.String,
    "token": pl.List(pl.Int64),
    "soft_token": pl.List(pl.List(pl.Int64)),
    "soft_weight": pl.List(pl.List(pl.Float64)),
    "pos_min": pl.List(pl.Int64),
    "value": pl.List(pl.Float64),
    "target_eligible": pl.List(pl.Boolean),
    "partition": pl.String,
    "n_events": pl.Int64,
    "source_start": pl.Int64,
    "source_end": pl.Int64,
    "continuation_index": pl.Int64,
    "n_windows": pl.Int64,
    "continues_from_previous": pl.Boolean,
    "continues_to_next": pl.Boolean,
    "anchor_idx": pl.Int64,
    "anchor_min": pl.Int64,
}


_GEM_STREAM_FIELDS = ("token", "soft_token", "soft_weight", "pos_min", "value",
                      "target_eligible")


def _gem_records(shards: pl.DataFrame, stays: pl.DataFrame, vocab: dict, gem_cfg: dict,
                 max_tokens: int, soft_width: int, binding: dict,
                 stats: dict | None) -> pl.DataFrame:
    """Frame each eligible stay `<bos> ADMISSION//a <events> DISCHARGE//d <eos>` and split
    it into windows (one row each) of at most `max_tokens`.

    The disposition appears ONLY as the single DISCHARGE// token, at the true
    discharge_dttm position (minutes since admission), target-eligible; `<bos>`,
    ADMISSION// and `<eos>` are inputs only. `anchor_idx` / `anchor_min` (stay-level
    index / minutes of the last token at or before ICU admit + 24 h) are retained for
    representation evaluation. Windows are contiguous slices of the stay stream:
    `source_start`/`source_end` index it, `continuation_index` counts windows, and the
    `continues_*` flags mark the cuts (the packed-segment continuation contract)."""
    if "discharge_category" not in stays.columns:
        raise QualificationError("episode artifact is missing discharge_category (GEM)")
    admission_type = (pl.col("admission_type_category").cast(pl.String)
                      if "admission_type_category" in stays.columns
                      else pl.lit(None, pl.String))
    frame = stays.select(
        pl.col("hospitalization_id").alias("hosp_id"), "partition",
        (pl.col("discharge_dttm") - pl.col("admission_dttm")).dt.total_minutes()
        .cast(pl.Int64).alias("discharge_min"),
        (pl.col("anchor_dttm") - pl.col("admission_dttm")).dt.total_minutes()
        .cast(pl.Int64).alias("anchor_min"),
        admission_type.alias("admission_type"),
        pl.col("discharge_category").cast(pl.String).alias("disposition"),
    ).sort("hosp_id")
    bos, eos = SPECIAL["<bos>"], SPECIAL["<eos>"]
    one_hot = [1.0] + [0.0] * (soft_width - 1)
    nan = float("nan")
    # Per stay, in Python (labels, counts, the framing tokens); the event bodies stay in
    # polars: framed = head + body + tail, then sliced into windows.
    counts: dict[str, dict[str, int]] = {"dispositions": {}, "admission_types": {}}
    heads: dict[str, list] = {key: [] for key in _GEM_STREAM_FIELDS}
    tails: dict[str, list] = {key: [] for key in _GEM_STREAM_FIELDS}
    for adm_raw, dis_raw, end_min in frame.select(
            "admission_type", "disposition", "discharge_min").iter_rows():
        adm = admission_label(adm_raw, gem_cfg)
        dis = disposition_label(dis_raw, gem_cfg)
        counts["admission_types"][adm] = counts["admission_types"].get(adm, 0) + 1
        counts["dispositions"][dis] = counts["dispositions"].get(dis, 0) + 1
        adm_id = vocab[f"{ADMISSION_PREFIX}{adm}"]
        dis_id = vocab[f"{DISCHARGE_PREFIX}{dis}"]
        for key, head, tail in (
            ("token", [bos, adm_id], [dis_id, eos]),
            ("soft_token", [[bos] * soft_width, [adm_id] * soft_width],
             [[dis_id] * soft_width, [eos] * soft_width]),
            ("soft_weight", [one_hot, one_hot], [one_hot, one_hot]),
            ("pos_min", [0, 0], [end_min, end_min]),
            ("value", [nan, nan], [nan, nan]),
            ("target_eligible", [False, False], [True, False]),
        ):
            heads[key].append(head)
            tails[key].append(tail)
    body = (shards.select("hosp_id", *_GEM_STREAM_FIELDS, "_n_le_anchor")
            if len(shards) else
            pl.DataFrame(schema={"hosp_id": pl.String,
                                 **{k: GEM_SCHEMA[k] for k in _GEM_STREAM_FIELDS},
                                 "_n_le_anchor": pl.Int64}))
    framed = frame.join(body.cast({k: GEM_SCHEMA[k] for k in _GEM_STREAM_FIELDS}),
                        on="hosp_id", how="left", maintain_order="left")
    framed = framed.select(
        "hosp_id", "partition", "anchor_min",
        *(pl.concat_list([
            pl.Series(f"_head_{k}", heads[k], dtype=GEM_SCHEMA[k]),
            pl.col(k).fill_null(pl.lit([], dtype=GEM_SCHEMA[k])),
            pl.Series(f"_tail_{k}", tails[k], dtype=GEM_SCHEMA[k]),
        ]).alias(k) for k in _GEM_STREAM_FIELDS),
        (1 + pl.col("_n_le_anchor").fill_null(0)).alias("anchor_idx"),
    )
    stay_rows, starts, ends, index, n_windows = [], [], [], [], []
    for row, n in enumerate(framed["token"].list.len().to_list()):
        bounds = gem_window_bounds(n, max_tokens)
        for i, (lo, hi) in enumerate(bounds):
            stay_rows.append(row)
            starts.append(lo)
            ends.append(hi)
            index.append(i)
            n_windows.append(len(bounds))
    windows = pl.DataFrame({"_row": stay_rows, "source_start": starts, "source_end": ends,
                            "continuation_index": index, "n_windows": n_windows},
                           schema={"_row": pl.UInt32, "source_start": pl.Int64,
                                   "source_end": pl.Int64, "continuation_index": pl.Int64,
                                   "n_windows": pl.Int64})
    lo, hi = pl.col("source_start"), pl.col("source_end")
    records = framed[windows["_row"]].hstack(windows.drop("_row").get_columns()).select(
        "hosp_id",
        pl.lit("hospitalization", pl.String).alias("trajectory"),
        *(pl.col(k).list.slice(lo, hi - lo) for k in _GEM_STREAM_FIELDS),
        "partition",
        (hi - lo).alias("n_events"),
        "source_start", "source_end", "continuation_index", "n_windows",
        (pl.col("continuation_index") > 0).alias("continues_from_previous"),
        (pl.col("continuation_index") < pl.col("n_windows") - 1).alias("continues_to_next"),
        "anchor_idx", "anchor_min",
    ).cast(GEM_SCHEMA).rechunk()
    if stats is not None:
        # Aggregate-only (no identifiers): stays, windows and label counts.
        stats["gem"] = {"stays": frame.height, "windows": len(records),
                        **{k: dict(sorted(v.items())) for k, v in counts.items()}}
    return records.with_columns(binding_expr(binding))


def _soft_width(bin_cfg: dict) -> int:
    """Soft assignments per event: ``2 * soft_kernel_bins + 1`` under soft
    discretization, else 1 (the hard bin)."""
    return (2 * max(int(bin_cfg["soft_kernel_bins"]), 0) + 1
            if bin_cfg.get("soft_discretization") else 1)


# Encode-loop memo bound: (concept, value) -> encoded event entries kept at once.
_MEMO_MAX = 1 << 20
_NOT_CATEGORICAL = object()   # `categorical_token` gave no token for (concept, value)


def _encode_events(events: pl.DataFrame, vocab: dict, edges: dict, bin_cfg: dict,
                   gem_mode: bool) -> tuple[pl.DataFrame, dict[str, int]]:
    """Encode ordered events (one contiguous run of rows per stay) into one record per
    stay, plus the aggregate `<unk>` count per concept. Module-level and picklable, so a
    process pool can run it on stay-contiguous chunks (`_parallel_encode`).

    One pass over the column lists, cutting a record at each stay change. Each binned
    concept's segments are compiled once (`segments.Binner`) with its `concept=bin` id
    table; an event's hard bin is computed once and reused for its soft assignment
    (`segments.soft_bins_at`), and (concept, value) results are memoized (bounded)."""
    unk_by_concept: dict[str, int] = {}   # aggregate-only, for the report
    if events.is_empty():
        # Unchanged contract: polars refuses group_by + apply on an empty frame
        # (ComputeError), so an empty window fails exactly as it always has.
        return events.group_by("hosp_id", maintain_order=True).map_groups(lambda g: g), \
            unk_by_concept
    ids = events["hosp_id"]
    starts = ids.ne_missing(ids.shift(1)).arg_true().to_list()
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    if len(starts) != ids.n_unique():
        # A stay in several runs: regroup by first appearance (rows keep their order),
        # which is what group_by(maintain_order=True) did.
        events = events.sort(pl.int_range(pl.len()).min().over("hosp_id"),
                             maintain_order=True)
        ids = events["hosp_id"]
        starts = ids.ne_missing(ids.shift(1)).arg_true().to_list()
        if not starts or starts[0] != 0:
            starts.insert(0, 0)

    unk_id = UNK_ID
    soft = bool(bin_cfg.get("soft_discretization"))
    kernel_bins = bin_cfg["soft_kernel_bins"] if soft else 0
    soft_width = _soft_width(bin_cfg)
    one_hot = [1.0] + [0.0] * (soft_width - 1)
    nan = float("nan")

    # Per concept, lazily: binned -> (Binner, hard ids, soft ids) by bin; unbinned ->
    # its bare (hard, soft ids, weights) entry. A hard id falls back to <unk> also when
    # the vocabulary maps the token to null; a soft id falls back only when absent.
    binned: dict = {}
    bare: dict = {}
    categorical: dict = {}
    memo: dict = {}

    def binned_info(c):
        info = binned.get(c)
        if info is None:
            binner = Binner(as_segments(edges[c]))
            keys = [fused_token(c, b) for b in range(len(binner.segments))]
            hard_ids = [vocab.get(key, unk_id) for key in keys]
            info = binned[c] = (binner,
                                [unk_id if h is None else h for h in hard_ids],
                                hard_ids)
        return info

    def encode_binned(c, v):
        binner, hard_ids, soft_ids = binned_info(c)
        b = binner.index(v)
        assignments = (soft_bins_at(v, binner.segments, kernel_bins, b) if soft
                       else [(b, 1.0)])
        entry = (hard_ids[b], [soft_ids[sb] for sb, _ in assignments],
                 [w for _, w in assignments])
        if len(memo) >= _MEMO_MAX:
            memo.clear()
        memo[(c, v)] = entry
        return entry

    def bare_entry(c):
        entry = bare.get(c)
        if entry is None:
            key = fused_token(c, None)
            hard = vocab.get(key)
            entry = bare[c] = (unk_id if hard is None else hard,
                               [vocab.get(key, unk_id)] * soft_width, one_hot)
        return entry

    def categorical_id(c, cat):
        key = (c, cat)
        if key not in categorical:
            cat_key = categorical_token(c, cat)
            if len(categorical) >= _MEMO_MAX:
                categorical.clear()
            categorical[key] = (_NOT_CATEGORICAL if cat_key is None
                                else vocab.get(cat_key, unk_id))
        return categorical[key]

    concepts = events["concept"].to_list()
    values = events["value"].to_list()
    cats = (events["cat_value"].to_list() if "cat_value" in events.columns
            else [None] * len(events))
    positions = events["pos_min"].to_list()
    eligibles = events["target_eligible"].to_list()
    partitions = events["partition"].to_list()
    le_anchors = events["_le_anchor"].to_list() if gem_mode else None
    hosp_ids = ids.to_list()

    columns: dict[str, list] = {name: [] for name in ENCODED_SCHEMA}
    anchors: list[int] = []
    for lo, hi in zip(starts, [*starts[1:], len(hosp_ids)]):
        token, soft_token, soft_weight, valnum = [], [], [], []
        # Positions and eligibility flags are collected INSIDE the loop, aligned with
        # `token`: a skipped event (missing numeric) drops its token, so the stay's full
        # `pos_min` / `target_eligible` would shift every later position and flag and
        # make n_events disagree with the sequence length (CodeRabbit: critical desync).
        pos_min, target_eligible = [], []
        # GEM: count of emitted events at or before the anchor (a prefix, since the
        # stream is time-ordered) -> the record's anchor_idx.
        n_le_anchor = 0
        for j in range(lo, hi):
            c, v = concepts[j], values[j]
            v_for_bin = float(v) if v is not None and math.isfinite(v) else None
            if v_for_bin is None:
                # KTD4: a categorical value is fused only when the row has no finite
                # numeric value; a row WITH a number emits only its binned token.
                hard = categorical_id(c, cats[j])
                if hard is not _NOT_CATEGORICAL:
                    if hard == unk_id:
                        unk_by_concept[c] = unk_by_concept.get(c, 0) + 1
                    token.append(hard)
                    soft_token.append([hard] * soft_width)
                    soft_weight.append(one_hot)
                    pos_min.append(positions[j])
                    target_eligible.append(eligibles[j])
                    valnum.append(nan)
                    if gem_mode:
                        n_le_anchor += bool(le_anchors[j])
                    continue
                if c in edges:
                    # Missing numeric measurements are not physiologic low-bin events.
                    # They are skipped rather than converted to bin 0.
                    continue
                hard, soft_ids, weights = bare_entry(c)
            elif c in edges:
                hard, soft_ids, weights = memo.get((c, v_for_bin)) or encode_binned(c, v_for_bin)
            else:
                hard, soft_ids, weights = bare_entry(c)
            if hard == unk_id:
                unk_by_concept[c] = unk_by_concept.get(c, 0) + 1
            token.append(hard)
            soft_token.append(soft_ids)
            soft_weight.append(weights)
            pos_min.append(positions[j])
            target_eligible.append(eligibles[j])
            valnum.append(float(v) if v is not None else nan)  # ORA value-regression target
            if gem_mode:
                n_le_anchor += bool(le_anchors[j])
        columns["hosp_id"].append(hosp_ids[lo])
        columns["token"].append(token)
        columns["soft_token"].append(soft_token)
        columns["soft_weight"].append(soft_weight)
        columns["pos_min"].append(pos_min)
        columns["value"].append(valnum)
        columns["target_eligible"].append(target_eligible)
        columns["partition"].append(partitions[lo])
        columns["n_events"].append(len(token))
        anchors.append(n_le_anchor)
    schema = dict(ENCODED_SCHEMA)
    if gem_mode:
        columns["_n_le_anchor"] = anchors
        schema["_n_le_anchor"] = pl.Int64
    return pl.DataFrame(columns, schema=schema), unk_by_concept


ENCODED_SCHEMA = {
    "hosp_id": pl.String,
    "token": pl.List(pl.Int64),
    "soft_token": pl.List(pl.List(pl.Int64)),
    "soft_weight": pl.List(pl.List(pl.Float64)),
    "pos_min": pl.List(pl.Int64),
    "value": pl.List(pl.Float64),
    "target_eligible": pl.List(pl.Boolean),
    "partition": pl.String,
    "n_events": pl.Int64,
}
_ENCODE_COLUMNS = ("hosp_id", "concept", "value", "cat_value", "pos_min",
                   "target_eligible", "partition", "_le_anchor")


# Events per encode chunk (`encode_chunk_events`, CLI --encode-chunk-events): encoding a
# chunk holds its columns as Python objects (a few hundred bytes per event), so ~2M
# events keep each in-process / per-worker batch to roughly 1 GB.
ENCODE_CHUNK_EVENTS = 2_000_000


def resolve_workers(workers: int | None) -> int:
    """`workers` 0 -> every CPU (`os.cpu_count()`); a positive count is used as is."""
    if workers is None:
        return 1
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError(f"workers must be a non-negative integer, got {workers!r}")
    return workers if workers > 0 else (os.cpu_count() or 1)


def resolve_chunk_events(chunk_events: int | None) -> int:
    """The per-chunk encode event budget (`None` -> `ENCODE_CHUNK_EVENTS`)."""
    if chunk_events is None:
        return ENCODE_CHUNK_EVENTS
    if isinstance(chunk_events, bool) or not isinstance(chunk_events, int) or chunk_events < 1:
        raise ValueError(f"encode_chunk_events must be a positive integer, got {chunk_events!r}")
    return chunk_events


def _stay_chunks(events: pl.DataFrame, max_events: int) -> list[tuple[int, int]]:
    """`[start, end)` row ranges of consecutive chunks of `events` (ordered, one
    contiguous run per stay), each holding whole stays and at most `max_events` events
    (a single stay larger than the budget is a chunk of its own). Cut ONLY at stay
    boundaries, so no stay is split across chunks."""
    total = len(events)
    if total == 0:
        return []
    ids = events["hosp_id"]
    starts = (ids != ids.shift(1)).fill_null(True).arg_true().to_list()
    cuts, size = [0], 0
    for lo, hi in zip(starts, [*starts[1:], total]):
        if size and size + (hi - lo) > max_events:
            cuts.append(lo)
            size = 0
        size += hi - lo
    cuts.append(total)
    return list(zip(cuts[:-1], cuts[1:]))


def _encode_chunk(args: tuple) -> tuple[pl.DataFrame, dict[str, int]]:
    return _encode_events(*args)


def _parallel_encode(events: pl.DataFrame, vocab: dict, edges: dict, bin_cfg: dict,
                     gem_mode: bool, workers: int,
                     chunk_events: int = ENCODE_CHUNK_EVENTS,
                     ) -> tuple[pl.DataFrame, dict[str, int]]:
    """Per-stay encoding over stay-contiguous chunks of at most `chunk_events` events
    (fewer when that gives each worker a share), in-process (`workers <= 1`) or in a
    spawn-context process pool of ``min(workers, chunks)`` processes. Encoding turns a
    chunk's columns into Python objects, so the budget, not the worker count, bounds
    peak memory: only the compact polars shard frames are kept, consumed in chunk order
    and concatenated. Byte-identical for any worker count and budget: every chunk is
    cast to `ENCODED_SCHEMA` and the result is rechunked."""
    columns = [c for c in _ENCODE_COLUMNS if c in events.columns]
    frame = events.select(columns)
    if frame.is_empty():
        shards, unk = _encode_events(frame, vocab, edges, bin_cfg, gem_mode)
        return shards, unk
    budget = min(chunk_events, -(-len(frame) // max(workers, 1)))
    bounds = _stay_chunks(frame, budget)
    jobs = ((frame.slice(lo, hi - lo), vocab, edges, bin_cfg, gem_mode) for lo, hi in bounds)
    parts: list[pl.DataFrame] = []
    unk: dict[str, int] = {}

    def collect(results) -> None:
        for shard, counts in results:
            parts.append(shard)
            for concept, n in counts.items():
                unk[concept] = unk.get(concept, 0) + n

    if workers <= 1 or len(bounds) <= 1:
        collect(_encode_events(*job) for job in jobs)
    else:
        context = multiprocessing.get_context("spawn")   # never fork a polars/duckdb process
        with ProcessPoolExecutor(max_workers=min(workers, len(bounds)),
                                 mp_context=context) as pool:
            collect(pool.map(_encode_chunk, jobs))
    shards = pl.concat(parts, how="vertical").rechunk()
    return shards, dict(sorted(unk.items()))


def tokenize_site(cfg: dict, site: str, base: Path, out: Path,
                  vocab_artifact: dict | None = None, *,
                  limit_stays: int | None = None,
                  episodes: pl.DataFrame | None = None,
                  artifact_policy: dict | None = None,
                  stats: dict | None = None,
                  trajectory: str = "icu_24h",
                  max_tokens: int | None = None,
                  sample_episodes: int | None = None,
                  report: bool = True,
                  workers: int = 1,
                  encode_chunk_events: int | None = None):
    """Tokenize one site.

    `vocab_artifact` None builds the frozen tokenizer-v2 vocabulary (reference site);
    otherwise it is a whole `vocab.json` blob, validated (`validate_vocabulary_artifact`)
    and applied unchanged. Every stay row of `events.parquet` carries the artifact
    binding (`artifact_hashes`: tokenizer version, vocabulary and segments hashes).
    `stats`, if given, is filled with aggregate-only run counts (`dose_conversion`:
    {dose table: {status: rows}}) for the tokenization report. Returns
    ``(vocab, segments)``.

    `trajectory` (U8, KTD10): ``icu_24h`` (default) writes events.parquet + vocab.json.
    ``hospitalization`` writes ONLY gem_events.parquet beside them, with the SAME frozen
    vocabulary, which must be imported (`vocab_artifact`; the reference site passes the
    vocab.json its icu_24h build just wrote) and must carry the `gem` allowlist. One row
    per window of at most `max_tokens` (default `gem.max_tokens`) tokens.

    One reference build, one vocabulary: when the config has a `gem` block, the build
    keeps the 24 h fit unchanged (same segments and ids for every concept in the 24 h
    train window) and appends (a) tokens for concepts / categorical values seen ONLY in
    the full-hospitalization train events (ED and ward ADT locations, ward-only labs) —
    their bins fit on those train events — and then (b) the fixed ADMISSION// and
    DISCHARGE// allowlist.

    `sample_episodes` (U7, KTD9) restricts the run to `sample_episode_ids(episodes, N)`:
    N eligible ICU episodes drawn deterministically from the episode artifact. A vocabulary
    built from a sample (or from `limit_stays`) records ``provenance.sample: true`` and
    ``sample_size``; training refuses it (`pretrain.build_loaders`, non-dry-run).

    `workers` (default 1; 0 = every CPU) encodes stays in a spawn-context process pool
    over stay-contiguous chunks of at most `encode_chunk_events` events (default
    `ENCODE_CHUNK_EVENTS`; `_parallel_encode`), which bounds encode memory; artifacts
    are byte-identical for any worker count and budget. Both are validated before any
    table is read.

    `report` (default on) writes the aggregate-only tokenization report beside the
    events (`tokenization_report.json`, or `gem_tokenization_report.json` for GEM; see
    src/data/tokenization_report.py): small cells suppressed, identifiers refused."""
    if trajectory not in TRAJECTORIES:
        raise ValueError(f"trajectory must be one of {TRAJECTORIES}, got {trajectory!r}")
    workers = resolve_workers(workers)
    encode_chunk_events = resolve_chunk_events(encode_chunk_events)
    gem_mode = trajectory == "hospitalization"
    gem_cfg = cfg.get("gem")
    if gem_mode:
        validate_gem_config(gem_cfg)
        if vocab_artifact is None:
            raise QualificationError(
                "the hospitalization (GEM) trajectory requires the frozen vocabulary built "
                "by the reference site's icu_24h run (pass its vocab.json)"
            )
        max_tokens = int(gem_cfg["max_tokens"] if max_tokens is None else max_tokens)
        if max_tokens < 2:
            raise ValueError(f"max_tokens must be >= 2, got {max_tokens}")
    elif gem_cfg is not None:
        validate_gem_config(gem_cfg)
    policy = artifact_policy or yaml.safe_load((ROOT / cfg["artifact_policy"]).read_text())
    # Disclosure floor (aggregate_no_phi.minimum_cell_size): the report's small-cell
    # threshold and the vocabulary's categorical-value floor. Fails closed if undeclared.
    min_cell = policy_min_cell(policy)
    events_path = out / (GEM_EVENTS_FILE if gem_mode else "events.parquet")
    validate_artifact_destination(events_path, "patient_level_phi", policy)
    vocab = edges = vocab_manifest = None
    if vocab_artifact is not None:
        vocab, edges, vocab_manifest = validate_vocabulary_artifact(vocab_artifact, cfg, policy)
        if gem_mode:
            absent = [t for t in gem_tokens(gem_cfg) if t not in vocab]
            if absent:
                raise QualificationError(
                    f"vocabulary lacks {len(absent)} gem allowlist token(s) (e.g. "
                    f"{absent[0]!r}); rebuild it with the configured `gem` block"
                )
    availability = validate_table_availability(cfg.get("tables"))
    con = duckdb.connect()
    # DuckDB renders TIMESTAMPTZ columns in the SESSION timezone, so on a non-UTC
    # host every tz-aware parquet came back as e.g. America/Chicago and
    # restrict_to_observation_window refused it (fail closed, but host-dependent).
    # Pin the session so tokenization is byte-identical wherever it runs (U9).
    con.execute("SET TimeZone = 'UTC'")
    keep_ids = None
    if sample_episodes is not None and limit_stays is not None:
        raise ValueError("pass either sample_episodes or limit_stays, not both")
    if sample_episodes is not None:
        if episodes is None:
            raise QualificationError("sample_episodes draws from the canonical episode "
                                     "artifact; pass episodes")
        keep_ids = sample_episode_ids(episodes, sample_episodes)
        print(f"  verification sample: {len(keep_ids):,} eligible episodes "
              "(smoke-only vocabulary)")
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
    building = vocab is None
    target_units = _dose_target_units(cfg, None if building else vocab_artifact)
    frames, shadow_frames, dose_stats = [], [], {}
    # Configured columns a site's parquet lacks, per table (report: `missing_columns`).
    missing_columns: dict[str, list[str]] = {}
    for name, spec in cfg["tables"].items():
        absent: list[str] = []
        df, shadows, counts = _read_source(
            con, base, spec, keep_ids, tables=cfg["tables"], target_units=target_units,
            fit_shadows=building, missing=absent,
        )
        if absent:
            missing_columns[name] = absent
        if counts is not None:
            dose_stats[name] = counts
        lag = availability[name]["lag_minutes"]
        for frame, sink in ((df, frames), (shadows, shadow_frames)):
            if frame is None or not len(frame):
                continue
            if lag:
                # Conservative availability (R12): the value becomes knowable `lag`
                # minutes after its timestamp, so shift BEFORE windowing — an event
                # whose shifted time passes the anchor is excluded, and a kept event
                # is positioned at its shifted time.
                frame = frame.with_columns(pl.col("dttm") + pl.duration(minutes=lag))
            sink.append(frame.with_columns(source=pl.lit(name)))
    if stats is not None:
        stats["dose_conversion"] = dose_stats
    if not frames:
        raise QualificationError("no configured CLIF event tables were found")
    static_tokens = list(cfg.get("static_tokens") or ())
    if STATIC_SOURCE in cfg["tables"]:
        raise QualificationError(f"table name {STATIC_SOURCE!r} is reserved for static tokens")
    # No sort here: join order is not guaranteed, so the order is imposed after it.
    events = pl.concat(frames, how="diagonal_relaxed")
    frames.clear()   # the per-table frames are now only a second copy of `events`
    mismatched_units = validate_units(
        events, cfg, None if building else vocab_artifact["reference_units"])

    if episodes is None:
        raise QualificationError("a canonical episode/split artifact is required")
    static = _read_static(con, base, cfg, keep_ids) if static_tokens else None
    carried = {name for name, spec in cfg["tables"].items() if spec.get("carry_forward")}
    raw_events = events

    def at_stay_start(start_col: str) -> pl.DataFrame:
        """R21: static admission tokens at the stay start (`start_col`: ICU admission
        for the 24 h artifact, hospital admission for GEM), ahead of every other event
        there (`static_rank`); then state carried forward to that start."""
        frame = raw_events
        if static is not None:
            placed = static.join(
                episodes.select(pl.col("hospitalization_id").alias("hosp_id"),
                                pl.col(start_col).alias("dttm")),
                on="hosp_id", how="inner",
            )
            if len(placed):
                frame = pl.concat(
                    [frame, placed.with_columns(unit=pl.lit(None, pl.String),
                                                source=pl.lit(STATIC_SOURCE))],
                    how="diagonal_relaxed",
                )
        return _carry_forward(frame, episodes, carried, start_col)

    treatment_sources = {
        name for name, spec in cfg["tables"].items() if spec.get("input_only")
    } | ({STATIC_SOURCE} if static_tokens else set())
    if gem_mode:
        events = restrict_to_hospitalization_window(
            at_stay_start("admission_dttm"), episodes, treatment_sources)
    else:
        events = restrict_to_observation_window(
            at_stay_start("icu_admit_dttm"), episodes, treatment_sources)
    if vocab is not None or gem_cfg is None:
        # The pre-window events are needed again only by the vocabulary build's GEM fit
        # (below); otherwise release them before the sort, the fit and the encode.
        raw_events = None
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
        fit_name = bin_cfg.get("fit_partition", "train")

        def fit_rows(windowed: pl.DataFrame, window) -> pl.DataFrame:
            """The fit partition of `windowed` events, plus — R6 / KTD5 — the
            native-unit fallback rows of weight-converted doses, which are fit
            (windowed by the same `window` and fit-partition-only, like every event)
            but never tokenized."""
            fit = fit_partition(windowed, fit_name)
            if shadow_frames:
                shadows = window(pl.concat(shadow_frames, how="diagonal_relaxed"),
                                 episodes, treatment_sources)
                fit = pl.concat([fit, shadows.filter(pl.col("partition") == fit_name)],
                                how="diagonal_relaxed")
            return fit

        fit_events = fit_rows(events, restrict_to_observation_window)
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
        quantile_concepts = {c for c, (_, _, numeric) in STATIC_TOKENS.items()
                             if numeric and c in static_tokens}
        edges, binning_sources = build_segments(
            bin_cfg, fit_events, target_concepts, directions, tables=cfg["tables"],
            quantile_concepts=quantile_concepts,
        )
        vocab = build_vocab(fit_events, edges, min_category_stays=min_cell)
        units = reference_units(fit_events, edges, cfg, target_units)
        sources = concept_sources(fit_events, treatment_sources)
        if gem_cfg is not None:
            # U8: one vocabulary for both artifacts. Extend it with what only the
            # full-hospitalization train events chart, then the fixed allowlist.
            gem_fit = fit_rows(restrict_to_hospitalization_window(
                at_stay_start("admission_dttm"), episodes, treatment_sources),
                restrict_to_hospitalization_window)
            vocab, edges, binning_sources, units, sources = _extend_for_gem(
                gem_cfg, gem_fit, fit_events, vocab, edges, binning_sources, units,
                bin_cfg=bin_cfg, cfg=cfg, directions=directions, target_units=target_units,
                treatment_sources=treatment_sources, quantile_concepts=quantile_concepts,
                min_category_stays=min_cell,
            )
            del gem_fit
        raw_events = None   # last pre-window use was the GEM fit above
        cohort_cfg = yaml.safe_load((ROOT / cfg["cohort_contract"]).read_text())
        split_hashes = episodes["split_sha256"].drop_nulls().unique().to_list()
        if len(split_hashes) != 1:
            raise QualificationError("episode artifact must contain one split hash")
        hashes = {
            "training_split": split_hashes[0],
            "vocabulary": json_sha256(vocab),
            "numeric_edges": json_sha256(edges),
            "binning_sources": json_sha256(binning_sources),
            "reference_units": json_sha256(units),
            "concept_sources": json_sha256(sources),
            "target_map": json_sha256(cfg["target_concepts"]),
            "outcome_spec": json_sha256(cohort_cfg["outcomes"]),
            "clif_version": json_sha256(cfg["schema_version"]),
        }
        vocab_manifest = {
            "artifact_family": "experimental_representation",
            "tokenizer_version": TOKENIZER_VERSION,
            "clif_version": cfg["schema_version"],
            "mcide_version": cfg["mcide_version"],
            "hashes": hashes,
            "provenance": {
                "source_site": site,
                "fit_partition": fit_name,
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
                # U4 (R6): aggregate conversion-status counts over every dose row read.
                # The dose target units themselves are hashed in `reference_units`.
                "dose_conversion": dose_stats,
                "static_tokens": static_tokens,
                # KTD9: a vocabulary fit on a verification sample is smoke-only;
                # training refuses it (segments.is_sample_vocab).
                "sample": keep_ids is not None,
                "sample_size": None if keep_ids is None else len(keep_ids),
            },
        }
        by_source = {
            src: sum(1 for v in binning_sources.values() if v == src) for src in BINNING_SOURCES
        }
        print(f"  built vocab: {len(vocab):,} tokens, {len(edges):,} numeric concepts "
              f"(binning sources: {by_source})")
        vocab_artifact = {
            "vocab": vocab,
            "segments": edges,
            "binning_sources": binning_sources,
            "reference_units": units,
            "concept_sources": sources,
            "precedence_policy": POLICY_VERSION,
            "manifest": vocab_manifest,
        }

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

    shards, unk_by_concept = _parallel_encode(
        events, vocab, edges, cfg["value_binning"], gem_mode, workers, encode_chunk_events)
    if gem_mode:
        soft_width = _soft_width(cfg["value_binning"])
        stays = episodes.filter(pl.col("eligible"))
        if keep_ids is not None:
            stays = stays.filter(pl.col("hospitalization_id").is_in(keep_ids))
        gem_stats: dict = {}
        gem = _gem_records(shards, stays, vocab, gem_cfg, max_tokens, soft_width,
                           artifact_binding(vocab_artifact),
                           stats if stats is not None else gem_stats)
        out.mkdir(parents=True, exist_ok=True)
        gem.write_parquet(events_path)
        # DATA-CLASSIFICATION: PHI (hosp_id + per-stay sequences + timing), like
        # events.parquet. vocab.json / events.parquet are not touched by this mode.
        print(f"  wrote {events_path} ({gem['hosp_id'].n_unique() if len(gem) else 0:,} "
              f"stays, {len(gem):,} windows)")
        if report:
            _write_run_report(out / GEM_REPORT_FILE, trajectory, events, gem,
                              vocab_artifact, site, availability, cfg, dose_stats,
                              mismatched_units, unk_by_concept, keep_ids, episodes,
                              max_tokens,
                              (stats if stats is not None else gem_stats).get("gem"),
                              min_cell=min_cell, missing_columns=missing_columns)
        return vocab, edges
    if len(shards):
        # KTD7: every shard row is bound to the tokenizer version, vocabulary and
        # segments it was encoded with; ModelDataset refuses a row without it.
        shards = shards.with_columns(binding_expr(artifact_binding(vocab_artifact)))
    out.mkdir(parents=True, exist_ok=True)
    shards.write_parquet(events_path)
    # DATA-CLASSIFICATION: PHI — contains hosp_id + per-stay token sequences + timing.
    # Do not export off-node. For external validation, use clif_validate.py which
    # returns only aggregate metrics. See NEXT_STEPS.md §6 rule 4.
    (out / "vocab.json").write_text(json.dumps(vocab_artifact))
    print(f"  wrote {out/'events.parquet'} ({len(shards):,} stays)")
    if report:
        context = int((gem_cfg or {}).get("max_tokens", DEFAULT_CONTEXT))
        _write_run_report(out / REPORT_FILE, trajectory, events, shards, vocab_artifact,
                          site, availability, cfg, dose_stats, mismatched_units,
                          unk_by_concept, keep_ids, episodes, context, None,
                          min_cell=min_cell, missing_columns=missing_columns)
    return vocab, edges


def _write_run_report(path: Path, trajectory: str, events: pl.DataFrame,
                      records: pl.DataFrame, vocab_artifact: dict, site: str,
                      availability: dict, cfg: dict, dose_stats: dict,
                      mismatched_units: list[str], unk_by_concept: dict,
                      keep_ids: list | None, episodes: pl.DataFrame, context: int,
                      gem: dict | None, *, min_cell: int,
                      missing_columns: dict[str, list[str]]) -> None:
    """Build, disclosure-control and write the aggregate-only tokenization report (R16).

    The fit partition is the vocabulary's own when this site built it (the reference
    site, including its GEM run); otherwise every partition is non-fit."""
    provenance = vocab_artifact["manifest"].get("provenance") or {}
    fit = provenance.get("fit_partition") if provenance.get("source_site") == site else None
    tables = list(cfg["tables"]) + ([STATIC_SOURCE] if cfg.get("static_tokens") else [])
    if records.is_empty():
        records = pl.DataFrame(schema={"hosp_id": pl.String, "token": pl.List(pl.Int64),
                                       "partition": pl.String, "n_events": pl.Int64})
    built = build_report(
        trajectory=trajectory, events=events, records=records,
        vocab=vocab_artifact["vocab"], segments=vocab_artifact["segments"],
        binning_sources=vocab_artifact["binning_sources"], availability=availability,
        tables=tables, dose_conversion=dose_stats, unit_mismatches=mismatched_units,
        unk_by_concept=unk_by_concept, fit_partition=fit,
        binding=artifact_binding(vocab_artifact),
        vocab_sample=bool(provenance.get("sample")),
        run_sample_size=None if keep_ids is None else len(keep_ids),
        context=context, unit_key=_normalized_unit, gem=gem, min_cell=min_cell,
        missing_columns=missing_columns,
    )
    identifiers = {
        str(v) for column in ("hospitalization_id", "patient_id", "hospitalization_joined_id")
        if column in episodes.columns
        for v in episodes[column].drop_nulls().to_list()
    }
    write_report(built, path, identifiers)
    print(f"  wrote {path} (aggregate-only)")


def build_arg_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--trajectory", choices=TRAJECTORIES, default="icu_24h",
                    help="icu_24h: events.parquet + vocab.json; hospitalization: "
                         "gem_events.parquet with the --vocab frozen vocabulary (U8)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="GEM window size (default: gem.max_tokens in the data config)")
    ap.add_argument("--sample-episodes", type=int, default=None, metavar="N",
                    help="verification sample (KTD9): N eligible ICU episodes drawn "
                         "deterministically from --episodes; the vocabulary is marked "
                         "provenance.sample: true and training refuses it")
    ap.add_argument("--workers", type=int, default=1, metavar="N",
                    help="processes for per-stay encoding (0 = every CPU); the output is "
                         "byte-identical for any value")
    ap.add_argument("--encode-chunk-events", type=int, default=None, metavar="N",
                    help=f"events per encode chunk (default {ENCODE_CHUNK_EVENTS:,}); bounds "
                         "encode memory, output byte-identical for any value")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the aggregate-only tokenization report")
    return ap


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    validate_table_availability(cfg.get("tables"))
    policy = yaml.safe_load((ROOT / cfg["artifact_policy"]).read_text())
    blob = None
    if args.vocab:
        blob = json.loads(Path(args.vocab).read_text())
        validate_vocabulary_artifact(blob, cfg, policy)
    elif not args.build_vocab:
        raise SystemExit("pass --build-vocab (first site) or --vocab PATH (later sites)")

    if args.dry_run:
        con = duckdb.connect()
        con.execute("SET TimeZone = 'UTC'")  # same session-tz pin as tokenize_site
        target_units = _dose_target_units(cfg, blob)
        for name, spec in cfg["tables"].items():
            df = _read_table(con, Path(args.indir), spec, tables=cfg["tables"],
                             target_units=target_units)
            print(f"{name}: {len(df):,} events, concepts={df['concept'].n_unique() if len(df) else 0}")
        return

    episodes = pl.read_parquet(args.episodes)
    tokenize_site(
        cfg,
        args.site,
        Path(args.indir),
        Path(args.out),
        blob,
        episodes=episodes,
        artifact_policy=policy,
        trajectory=args.trajectory,
        max_tokens=args.max_tokens,
        sample_episodes=args.sample_episodes,
        report=not args.no_report,
        workers=args.workers,
        encode_chunk_events=args.encode_chunk_events,
    )


if __name__ == "__main__":
    main()
