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

Concepts the CSV does not cover may be binned on literature-grounded edges
(`value_binning.literature_source`, configs/literature_segments/*.yaml; precedence csv ->
literature -> ordinal -> quantile -> single; recorded and hashed as vocab.json
`literature_segments`). Unit-less wide-table columns (CRRT, ECMO/MCS) have a declared
canonical unit and plausible range (`column_units`), medication doses a plausible range
per converted concept (`dose_plausibility`); a site on another scale declares its
conversion explicitly (`site_unit_conversions.<site>`), validated fail-closed per site
(`check_column_units`, `check_dose_plausibility`) and reported in the data-quality section.

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
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import yaml

from clif_validate._vendor.data.clif_conformance import (
    bp_config,
    bp_method_of,
    check_conformance,
    compliance_table,
    flag_sql,
    global_record,
    harmonization_record,
    mapped_sql,
    parquet_source,
    rule_columns,
    site_harmonization,
)
from clif_validate._vendor.data.cohort import (
    QualificationError,
    validate_artifact_destination,
    validate_episode_artifact,
)
from clif_validate._vendor.data.segments import _apply_forced_edge  # policy step 3, shared with quantile segments
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
    make_segment,
    n_bins as segment_count,
    ordinal_segments,
    segments_from_edges,
    soft_bins,
    soft_bins_at,
    validate_partition,
    with_zero_point,
)
from clif_validate._vendor.data.site_config import (
    SiteConfigError,
    censor_open_stays,
    profile_record,
    site_profile,
    strip_notes,
    to_utc,
    with_site_local,
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
    parse_unit,
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
# Per-concept binning source, in precedence order: the physician CSV, then the
# literature-grounded fragments (configs/literature_segments/*.yaml), then the data-driven
# rules. A fragment's keep_ordinal / keep_quantile decision is recorded as ordinal /
# quantile (its provenance is in vocab.json `literature_segments`).
BINNING_SOURCES = ("csv", "literature", "ordinal", "quantile", "single")
LITERATURE_DECISIONS = ("segments", "keep_ordinal", "keep_quantile")
LITERATURE_CLOSURES = ("left", "right")
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
    _check_unit_repair_binding(blob["reference_units"], cfg, provenance.get("source_site"))
    _check_literature_binding(blob, hashes)
    _check_harmonization_binding(blob, hashes, cfg)
    return blob["vocab"], blob["segments"], manifest


def _check_unit_repair_binding(units: dict, cfg: dict, source_site: str | None) -> None:
    """The frozen segments of a unit-less column are in its canonical unit, as repaired
    by the reference site's declared conversions. A config declaring other canonical
    units, or other conversions for the reference site, does not bind the vocabulary."""
    declared = _hashed_column_units(cfg)
    recorded_units = {k: {f: v for f, v in (spec or {}).items() if f != "flag"}
                      for k, spec in (units.get("column_units") or {}).items()}
    if recorded_units != declared:
        raise QualificationError(
            "vocabulary was built under different canonical column units (column_units) "
            f"than this config declares; {RETOKENIZE}")
    recorded = strip_notes(units.get("site_conversions") or {})
    expected = strip_notes(site_unit_conversions(cfg, source_site)) if source_site else {}
    if (declared or recorded) and recorded != expected:
        raise QualificationError(
            f"vocabulary was built with different unit conversions for its reference site "
            f"{source_site!r} (site_unit_conversions) than this config declares; "
            f"{RETOKENIZE}")


def _check_literature_binding(blob: dict, hashes: dict) -> None:
    """The literature record (fragment hashes, applied edges and sources) is hashed; a
    vocabulary with literature-binned concepts must carry it."""
    record = blob.get("literature_segments")
    uses = "literature" in (blob.get("binning_sources") or {}).values()
    if record is None and "literature_segments" not in hashes and not uses:
        return
    if not isinstance(record, dict) or "literature_segments" not in hashes:
        raise QualificationError(
            "vocabulary bins concepts from literature segments but lacks the hashed "
            f"literature record; {RETOKENIZE}")
    if hashes["literature_segments"] != json_sha256(record):
        raise QualificationError("literature-segments hash mismatch")


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


def with_availability_lag(cfg: dict, minutes: int) -> dict:
    """A copy of `cfg` with every table of the declared lag-sensitivity semantics
    (`availability_lag_sensitivities.semantics`, default `missing_storetime`) lagged
    `minutes` - the 15 / 60 min sensitivity configurations (hard rule #4)."""
    import copy

    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 0:
        raise QualificationError(f"availability lag must be a non-negative integer, got {minutes!r}")
    block = cfg.get("availability_lag_sensitivities") or {}
    semantics = block.get("semantics", "missing_storetime")
    out = copy.deepcopy(cfg)
    for spec in out["tables"].values():
        if spec.get("availability") == semantics:
            spec["availability_lag_minutes"] = minutes
    return out


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


def _from(fp: Path, spec: dict) -> str:
    """The FROM expression of a table: its parquet, with the site's declared column
    aliases applied (`_from_sql`, set by `_read_source`)."""
    return spec.get("_from_sql") or f"read_parquet('{fp}')"


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
    rules = spec.get("_category_rules") or {}
    cat_sql = mapped_sql(cat, rules.get(cat)) if cat else "CAST(NULL AS VARCHAR)"
    time_col = spec["availability_col"]
    concept_col = spec.get("concept_col")
    if concept_col:
        mapped = mapped_sql(concept_col, rules.get(concept_col))
        concept_sql = _name_sql(mapped) if spec.get("normalize_concept") else mapped
        present = f"{concept_col} IS NOT NULL"
    else:
        # A single-concept table (position): every row is that concept, so a row
        # without a categorical (or numeric) value carries nothing.
        concept_sql = f"CAST({_literal(spec.get('concept'), 'concept')} AS VARCHAR)"
        present = " OR ".join(f"{c} IS NOT NULL" for c in (val, cat) if c) or "FALSE"
        present = f"({present})"
    id_sql, params = _id_filter(keep_ids)
    # BP measurement method: the site's declared source column rides along (`_bp_src`).
    bp_src = spec.get("_bp_source_column")
    extra = f",\n               CAST({bp_src} AS VARCHAR) AS _bp_src" if bp_src else ""
    return f"""
        SELECT CAST(hospitalization_id AS VARCHAR) AS hosp_id,
               {time_col}                AS dttm,
               {concept_sql}             AS concept,
               {val_sql}                 AS value,
               {unit_sql}                AS unit,
               {cat_sql}                 AS cat_value{extra}
        FROM {_from(fp, spec)}
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
    rules = spec.get("_category_rules") or {}
    qual_sel = f"{mapped_sql(qual, rules.get(qual))} AS _qual," if qual else ""
    name = "'unknown_' || lower(_col)" if qualifier_absent else "lower(_col)"
    if qual:
        name = f"coalesce(nullif({_name_sql('_qual')}, ''), 'unknown') || '_' || lower(_col)"
    cast = "DOUBLE" if numeric else "VARCHAR"
    # Declared per-site unit conversions of unit-less numeric columns, applied before the
    # melt so every downstream step (fit, binning, reference units) sees canonical units.
    factors = (spec.get("_column_factors") or {}) if numeric else {}
    flags = set(spec.get("flag_cols") or ())

    def cast_sql(c: str) -> str:
        if numeric:
            return (_convert_sql(f"CAST({c} AS DOUBLE)", factors[c]) if c in factors
                    else f"CAST({c} AS DOUBLE)")
        # A 0/1 flag has one spelling at every site; a category takes the site's map.
        return flag_sql(c) if c in flags else mapped_sql(c, rules.get(c))

    casts = ", ".join(f"{cast_sql(c)} AS {c}" for c in cols)
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
                FROM {_from(fp, spec)}
                WHERE {time_col} IS NOT NULL {id_sql}
            ) ON {", ".join(cols)} INTO NAME _col VALUE _val
        )
    """, params


def _patient_keyed_sql(base: Path, fp: Path, spec: dict, keep_ids: list | None, *,
                       window: str | None = None) -> tuple[str, list] | None:
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
    if window is not None:
        # Only the windowed stays (window pushdown, `_register_window`).
        id_sql += (f" AND CAST(hospitalization_id AS VARCHAR) IN "
                   f"(SELECT hosp_id FROM {window})")
    return f"""
        WITH s AS (
            SELECT CAST({pid} AS VARCHAR) AS pid, {time_col} AS t,
                   {mapped_sql(cat, (spec.get("_category_rules") or {}).get(cat))} AS v
            FROM {_from(fp, spec)}
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
                     fit_shadows: bool, corrections: dict | None = None,
                     ) -> tuple[pl.DataFrame, pl.DataFrame | None, dict]:
    """R6/R7 (KTD5): medication doses with unit conversion.

    SQL reads the doses (a stop action is a dose of 0) and, for continuous doses, ASOF
    joins the most recent weight whose availability time (charted time + the weight
    table's lag) is at or before the dose's (admin time + this table's lag). Each
    distinct (medication, unit) pair is then resolved ONCE by `units.dose_plan`, joined
    back and applied vectorized — never per row in Python.

    Returns (events, fit-only shadow events, {status: count}). Shadows (reference-site
    build only) are the native-unit fallback rows of every weight-dependent conversion,
    so fallback concepts get frozen bins even when this site converted every row.

    `corrections` (`dose_corrections`: this site's declarations for this table) are
    applied to the matching (med_category, charted unit) rows BEFORE conversion: a
    correction rewrites the charted unit to its `to` and multiplies the dose by `factor`
    (counted under ``corrected``); a quarantine keeps the rows in their native-unit
    concept (status ``quarantined``)."""
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
    # Administrations whose action is not a dose (`exclude_actions`, e.g. intermittent
    # not_given / other): flagged here (after the site's category map), counted and
    # removed below, before any conversion.
    excluded = [str(a).strip().lower() for a in dose.get("exclude_actions") or ()]
    excl_sql, excl_params = "FALSE", []
    if excluded:
        if not action:
            raise QualificationError(
                f"{name_of(spec)}.dose.exclude_actions needs dose.action_col")
        mapped_action = mapped_sql(action, (spec.get("_category_rules") or {}).get(action))
        excl_sql = (f"coalesce(lower(trim({mapped_action})) IN "
                    f"({', '.join('?' for _ in excluded)}), FALSE)")
        excl_params = excluded
    id_sql, id_params = _id_filter(keep_ids)
    lag = int(spec.get("availability_lag_minutes", 0))
    med_sql = mapped_sql(concept_col, (spec.get("_category_rules") or {}).get(concept_col))
    source = _from(fp, spec)
    # Declared exact duplicates (site_harmonization.<site>.exact_duplicates): a `drop`
    # category row identical to a `keep` category row on every `match` column is the same
    # administration charted twice; it is flagged here and removed (counted) below.
    dup_sql, dup_join, dup_params = "FALSE", "", []
    rule = spec.get("_exact_duplicates")
    if rule:
        match = rule["match"]
        keys = ", ".join(f"{c} AS _k{i}" for i, c in enumerate(match))
        on = " AND ".join(f"s.{c} IS NOT DISTINCT FROM k._k{i}" for i, c in enumerate(match))
        drop = ", ".join("?" for _ in rule["drop"])
        dup_join = (f"LEFT JOIN (SELECT DISTINCT {keys}, TRUE AS _hit FROM {source} "
                    f"WHERE {_name_sql(concept_col)} = ?) k ON {_name_sql('s.' + concept_col)} "
                    f"IN ({drop}) AND {on}")
        dup_params = [rule["keep"], *rule["drop"]]
        dup_sql = "coalesce(k._hit, FALSE)"
    # Column references stay unqualified: the duplicate side only carries _k*/_hit.
    doses = f"""
        SELECT CAST(hospitalization_id AS VARCHAR) AS hosp_id, {time_col} AS dttm,
               {med_sql} AS med, {dose_sql} AS dose,
               coalesce(CAST({unit} AS VARCHAR), '') AS unit_raw,
               {time_col} + INTERVAL {lag} MINUTE AS avail,
               {dup_sql} AS _dup, {excl_sql} AS _excl
        FROM {source} s {dup_join}
        WHERE {concept_col} IS NOT NULL AND {time_col} IS NOT NULL {id_sql}
    """
    # Bound in textual order: the stop actions and excluded actions (SELECT), the
    # duplicate join, the ids.
    params = [*params, *excl_params, *dup_params, *id_params]
    # Continuous doses convert per-kg rates; an intermittent table declaring a weight
    # source converts per-kg single doses (e.g. ketamine mg/kg) to absolute mass.
    weight = dose.get("weight_source")
    wspec = (tables or {}).get(weight["table"]) if weight else None
    wfp = base / f"{wspec['file']}.parquet" if wspec else None
    if weight and (wfp is None or not wfp.exists()):
        print(f"  [warn] {fp.name}: weight source {weight['table']!r} not found; "
              "per-kg conversions fall back to native units")
    weight_counts: dict[str, int] = {}
    if wfp is not None and wfp.exists():
        wtime, wval = wspec["availability_col"], wspec["value_col"]
        wlag = int(wspec.get("availability_lag_minutes", 0))
        w_id_sql, w_params = _id_filter(keep_ids)
        # A weight outside the declared plausible range (`plausible_kg`) never converts a
        # per-kg dose (a 1 kg charting error would scale a dose x80); counted per run.
        plausible = weight.get("plausible_kg")
        w_value = f"CAST({wval} AS DOUBLE)"
        w_range = ""
        if plausible is not None:
            lo, hi = _range_pair(plausible, f"{name_of(spec)}.dose.weight_source.plausible_kg")
            w_range = f"AND {w_value} BETWEEN {lo!r} AND {hi!r}"
            total, kept = con.execute(
                f"SELECT count(*), count(*) FILTER (WHERE {w_value} BETWEEN {lo!r} AND {hi!r}) "
                f"FROM read_parquet('{wfp}') WHERE {wspec['concept_col']} = ? "
                f"AND {wtime} IS NOT NULL AND {w_value} IS NOT NULL {w_id_sql}",
                [weight["concept"], *w_params]).fetchone()
            weight_counts = {"weights": int(total), "weights_excluded": int(total - kept)}
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
                      {w_range} {w_id_sql}
                ) GROUP BY hosp_id, avail
            )
            SELECT d.hosp_id, d.dttm, d.med, d.dose, d.unit_raw, d._dup, d._excl, w.weight_kg
            FROM d ASOF LEFT JOIN w ON d.hosp_id = w.hosp_id AND d.avail >= w.avail
        """
        params += [weight["concept"], *w_params]
    else:
        sql = f"""SELECT hosp_id, dttm, med, dose, unit_raw, _dup, _excl,
                         CAST(NULL AS DOUBLE) AS weight_kg FROM ({doses})"""
    frame = con.execute(sql, params).pl()
    n_duplicates = int(frame["_dup"].sum()) if len(frame) else 0
    n_excluded = int((frame["_excl"] & ~frame["_dup"]).sum()) if len(frame) else 0
    frame = frame.filter(~pl.col("_dup") & ~pl.col("_excl")).drop("_dup", "_excl")
    frame, n_corrected = _apply_dose_corrections(frame, corrections or {})

    target_units = target_units or {}
    pairs = frame.select("med", "unit_raw").unique().iter_rows()
    plans = []
    for med, raw in pairs:
        target = (target_units.get(normalize_name(med)) if kind == "continuous"
                  else canonical_unit(raw))
        if (kind == "intermittent" and weight and target and "/kg" in target
                and parse_unit(target)["minutes"] is None):
            target = target.replace("/kg", "")   # mg/kg single dose -> mg (x weight)
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
        pl.when(pl.col("_quarantine"))
        .then(pl.lit("quarantined"))
        .when((pl.col("status") == "converted") & needs_weight
              & pl.col("weight_kg").is_null())
        .then(pl.lit("no_weight")).otherwise(pl.col("status")).alias("status")
    )
    converted = pl.col("status") == "converted"
    scale = (pl.when(needs_weight)
             .then(pl.col("factor") * pl.col("weight_kg").pow(pl.col("weight_power")))
             .otherwise(pl.col("factor")))
    native = pl.col("status").is_in(["no_weight", "unconvertible", "quarantined"])
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
    if corrections:
        counts.update(corrected=n_corrected, quarantined=0)
    if rule:
        counts["exact_duplicates_removed"] = n_duplicates
    if excluded:
        counts["excluded_action"] = n_excluded
    counts.update(weight_counts)
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


def name_of(spec: dict) -> str:
    return str(spec.get("_name") or spec.get("file"))


def _apply_dose_corrections(frame: pl.DataFrame, corrections: dict,
                            ) -> tuple[pl.DataFrame, int]:
    """Apply the declared (med_category, charted unit) corrections: ``(frame with a
    `_quarantine` flag, corrected row count)``. Pairs are resolved once, joined back."""
    rows = []
    keys: list[tuple[str, str]] = []
    for med, raw in frame.select("med", "unit_raw").unique().iter_rows():
        key = (normalize_name(med), normalize_unit(raw))
        decl = corrections.get(key)
        if decl is None:
            continue
        if key not in keys:
            keys.append(key)
        quarantine = decl.get("action") == "quarantine"
        rows.append({"med": med, "unit_raw": raw,
                     "_to": raw if quarantine else decl["to"],
                     "_factor": 1.0 if quarantine else float(decl["factor"]),
                     "_q": quarantine, "_key": keys.index(key)})
    plan = pl.DataFrame(rows, schema={"med": pl.String, "unit_raw": pl.String,
                                      "_to": pl.String, "_factor": pl.Float64,
                                      "_q": pl.Boolean, "_key": pl.Int64})
    joined = frame.join(plan, on=["med", "unit_raw"], how="left")
    # A declaration with `when` applies only to the charted values meeting it.
    applies = pl.lit(True)
    for index, key in enumerate(keys):
        when = corrections[key].get("when")
        if not when:
            continue
        cond = pl.lit(True)
        for op, value in when.items():
            cond = cond & {"gt": pl.col("dose") > value, "ge": pl.col("dose") >= value,
                           "lt": pl.col("dose") < value, "le": pl.col("dose") <= value}[op]
        applies = pl.when(pl.col("_key") == index).then(cond.fill_null(False)).otherwise(applies)
    joined = joined.with_columns(
        pl.when(applies).then(pl.col("_to")).otherwise(None).alias("_to"),
        pl.when(applies).then(pl.col("_q")).otherwise(None).alias("_q"),
    ).drop("_key")
    corrected = pl.col("_to").is_not_null() & ~pl.col("_q").fill_null(False)
    n_corrected = int(joined.select(corrected.sum()).item())
    out = joined.with_columns(
        pl.when(corrected).then(_scale_expr(pl.col("dose"), pl.col("_factor")))
        .otherwise(pl.col("dose")).alias("dose"),
        pl.when(corrected).then(pl.col("_to")).otherwise(pl.col("unit_raw")).alias("unit_raw"),
        pl.col("_q").fill_null(False).alias("_quarantine"),
    ).drop("_to", "_factor", "_q")
    return out, n_corrected


def _parquet_columns(con, fp: Path, source: str | None = None) -> set[str]:
    return {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM {source or f'read_parquet({chr(39)}{fp}{chr(39)})'}"
    ).fetchall()}


# Optional per-site columns of a long/wide table spec: a site's parquet may lack any of
# them (synthetic CLIF releases omit 11 resp_support columns, the assessments
# categorical_value and ecmo fdO2), and the table is then read without them.
_OPTIONAL_LIST_COLS = ("value_cols", "categorical_value_cols")
_OPTIONAL_COLS = ("categorical_value_col", "concept_qualifier_col")


def _present_columns(con, fp: Path, spec: dict) -> tuple[dict, list[str]]:
    """`spec` restricted to the optional columns `fp` actually has (DuckDB identifiers
    are case-insensitive; the site's column aliases applied), plus the configured columns
    it lacks, in config order."""
    present = {c.lower() for c in _parquet_columns(con, fp, spec.get("_from_sql"))}
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
                 column_factors: dict[str, dict] | None = None,
                 dose_corrections: dict | None = None,
                 harmonization: dict | None = None, name: str | None = None,
                 window: str | None = None,
                 ) -> tuple[pl.DataFrame, pl.DataFrame | None, dict | None]:
    """Read one configured table -> (events, fit-only shadows or None, dose status
    counts or None). See `_read_table`. Configured optional columns (`value_cols`,
    `categorical_value_cols`, `categorical_value_col`, `concept_qualifier_col`) the
    site's parquet lacks are skipped, logged, and appended to `missing`.
    `column_factors` / `dose_corrections`: the site's declared unit conversions for this
    table (`column_factors`, `dose_corrections`). `harmonization`: the site's declarations
    (`clif_conformance.site_harmonization`) - column aliases, category maps, exact
    duplicates and the BP-method source column - for table `name`. A table declaring
    `on_missing_column: error` refuses a configured column the site lacks.

    `window` (the name of a registered DuckDB relation ``hosp_id, lo, hi``;
    `_register_window`) pushes the eligible stays and their window bounds into the scan,
    so rows of other stays, and rows outside a stay's window, are never materialized
    (`_windowed_source`)."""
    fp = base / f"{spec['file']}.parquet"
    if not fp.exists():
        print(f"  [skip] {fp.name} not found")
        return _empty_events(), None, None
    spec = _harmonized_spec(con, fp, spec, harmonization, name)
    if window is not None and spec.get("key", "hospitalization") == "hospitalization":
        spec["_from_sql"] = _windowed_source(con, _from(fp, spec), spec, window)
    if spec.get("dose"):
        return _read_dose_table(con, base, fp, spec, keep_ids, tables, target_units,
                                fit_shadows, dose_corrections)
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
        part = _patient_keyed_sql(base, fp, spec, keep_ids, window=window)
        if part is None:
            return _empty_events(), None, None
        parts.append(part)
    else:
        declared = any(spec.get(k) for k in ("concept_col", "concept", *_OPTIONAL_LIST_COLS))
        spec, absent = _present_columns(con, fp, spec)
        if column_factors:
            spec["_column_factors"] = dict(column_factors)
        if absent and spec.get("on_missing_column") == "error":
            raise QualificationError(
                f"table {name or spec['file']!r} ({fp.name}): configured column(s) not found: "
                f"{', '.join(absent)}; the table declares on_missing_column: error (declare "
                "site_harmonization.<site>.column_aliases if the site names them otherwise)")
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


def _harmonized_spec(con, fp: Path, spec: dict, harmonization: dict | None,
                     name: str | None) -> dict:
    """`spec` with the site's declarations for table `name` attached (private `_*`
    keys read by the SQL builders)."""
    spec = {**spec, "_name": name}
    if not harmonization or name is None:
        return spec
    aliases = harmonization["column_aliases"].get(name) or {}
    if aliases:
        raw = {c.lower() for c in _parquet_columns(con, fp)}
        spec["_from_sql"] = parquet_source(fp, aliases, raw)
    rules = {key.split(".", 1)[1]: value
             for key, value in harmonization["category_map"].items()
             if key.split(".", 1)[0] == name}
    present = {c.lower() for c in _parquet_columns(con, fp, spec.get("_from_sql"))}
    if rules:
        needed = sorted(c for r in rules.values() for c in rule_columns(r)
                        if c.lower() not in present)
        if needed:
            raise QualificationError(
                f"table {name!r}: category_map conditions read column(s) the site lacks: "
                f"{', '.join(needed)}")
        spec["_category_rules"] = rules
    for rule in harmonization["exact_duplicates"]:
        if rule["table"] == name:
            spec["_exact_duplicates"] = rule
    bp = harmonization.get("_bp_table")
    if bp == name and harmonization.get("bp_method"):
        source = harmonization["bp_method"]["source_column"]
        if source.lower() not in present:
            raise QualificationError(
                f"table {name!r}: the declared BP-method source column {source!r} is missing")
        spec["_bp_source_column"] = source
    return spec


# Window pushdown (memory): the eligible stays and their window bounds are registered in
# the DuckDB session; every hospitalization-keyed table is scanned through a JOIN on
# them, so other stays' rows and rows outside a stay's window never reach polars. The
# polars window join (`_windowed`) still applies the exact bounds afterwards: the SQL
# bounds are a superset (a table's availability lag widens the lower bound; a naive
# local-time column is compared with a 26 h pad, since it is converted to UTC only after
# it is read). State tables (`carry_forward`, `emit: transitions`) keep every row before
# the window: their state at the window start is read from them.
WINDOW_RELATION = "_clif_window"
_NAIVE_PAD_HOURS = 26


def _register_window(con, stays: pl.DataFrame, start: str, end: str) -> str:
    """Register ``hosp_id, lo, hi`` (UTC) for `stays` (one row per stay) and return the
    relation name. A null `end` (open stay) leaves the stay unbounded above."""
    frame = stays.select(pl.col("hospitalization_id").cast(pl.String).alias("hosp_id"),
                         pl.col(start).alias("lo"), pl.col(end).alias("hi")).unique("hosp_id")
    con.register(WINDOW_RELATION, frame.to_arrow())
    return WINDOW_RELATION


def _column_type(con, source: str, column: str) -> str | None:
    for name, dtype, *_ in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall():
        if name.lower() == column.lower():
            return str(dtype).upper()
    return None


def _windowed_source(con, source: str, spec: dict, window: str) -> str:
    """`source` restricted to the registered stays and window bounds (see above)."""
    time_col = spec["availability_col"]
    lag = int(spec.get("availability_lag_minutes", 0))
    naive = (_column_type(con, source, time_col) or "").startswith("TIMESTAMP") and \
        "TIME ZONE" not in (_column_type(con, source, time_col) or "")
    pad = f" + INTERVAL {_NAIVE_PAD_HOURS} HOUR" if naive else ""
    upper = f"(w.hi IS NULL OR s.{time_col} <= w.hi{pad})"
    state = spec.get("carry_forward") or spec.get("emit") == "transitions"
    lower = "" if state else (
        f" AND s.{time_col} >= w.lo - INTERVAL {lag} MINUTE"
        + (f" - INTERVAL {_NAIVE_PAD_HOURS} HOUR" if naive else ""))
    # An equality join with the bounds in WHERE: DuckDB runs a SEMI JOIN carrying range
    # predicates in ON as a slow range join (measured 275 s against 0.1 s for this form on
    # 55M vitals rows). `hosp_id` is unique in the window relation, so no row is repeated.
    return (f"(SELECT s.* FROM {source} s JOIN {window} w "
            f"ON CAST(s.hospitalization_id AS VARCHAR) = w.hosp_id WHERE {upper}{lower})")


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
        raise ValueError("Non-canonical CLIF units: " + "; ".join(mismatches) + " "
                         + NODE_ONLY)
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
        and not units_equivalent(concept, unit, expected[concept], cfg)
    ]
    return sorted(mismatches)


def units_equivalent(concept: str, a: str, b: str, cfg: dict) -> bool:
    """Same spelling (`_normalized_unit`), or a declared concept-scoped equivalence
    (`unit_normalization.equivalent_units`: e.g. mEq/L == mmol/L for monovalent ions
    only, pH `units` == `(no units)`)."""
    if _normalized_unit(a) == _normalized_unit(b):
        return True
    for entry in (cfg.get("unit_normalization") or {}).get("equivalent_units") or ():
        units = {_normalized_unit(u) for u in entry.get("units") or ()}
        if (concept in (entry.get("concepts") or ())
                and {_normalized_unit(a), _normalized_unit(b)} <= units):
            return True
    return False


# ---- explicit unit repair: unit-less wide-table columns and medication doses ----------
#
# CRRT and ECMO/MCS rows carry no unit column, so their canonical unit and plausible range
# come from the config (`column_units`, keyed `table.column`, CLIF 2.1 data dictionary).
# A site storing another scale declares the conversion (`site_unit_conversions.<site>`);
# nothing is detected or converted silently. Declarations are per site: the reference
# site's are recorded (and hashed) in the vocabulary's `reference_units`, and every other
# site applies only its own.

# Failure messages may echo a site's free text (a BP source name, a charted unit) and
# aggregate counts or quantiles; counts under the minimum cell size are printed as "<10",
# quantiles of a small cell are withheld, and every such message says where it may go.
NODE_ONLY = ("[aggregate only; this message stays on the node: never paste it, or a "
             "traceback, off the node]")


def _cell(n: int, min_cell: int = 10) -> str:
    """A count for a failure message: ``<min_cell`` when 0 < n < min_cell."""
    return f"<{min_cell}" if 0 < int(n) < min_cell else f"{int(n):,}"


DEFAULT_MAX_OUT_OF_RANGE_SHARE = 0.01
DEFAULT_MAX_IMPLAUSIBLE_DOSE_SHARE = 0.02
_RANGE_TOL = 1e-6   # relative slack on plausible-range bounds (float32 storage)
# Scale hints offered when a column fails its plausible range: (factor, label).
SCALE_HINTS = ((1 / 60, "x1/60"), (60.0, "x60"), (0.01, "x1/100"), (100.0, "x100"),
               (0.001, "x1/1000"), (1000.0, "x1000"))
_COLUMN_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$")
_DOSE_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.([a-z0-9_]+)\[([^\]]+)\]$")
OUT_OF_RANGE_ACTIONS = ("error", "report")


def _positive_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0 else None


def _range_pair(value: object, what: str) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(float(v)) for v in value)
            or float(value[0]) >= float(value[1])):
        raise QualificationError(f"{what} range must be [low, high] with low < high, "
                                 f"got {value!r}")
    return [float(value[0]), float(value[1])]


def column_units(cfg: dict) -> dict[str, dict]:
    """The validated `column_units` block: ``{"table.column": {"unit", "range": [lo, hi],
    "on_out_of_range": "error" | "report", "flag": str | None}}``. The table must be
    configured and the column one of its `value_cols`."""
    raw = cfg.get("column_units") or {}
    if not isinstance(raw, dict):
        raise QualificationError("column_units must map `table.column` to {unit, range}")
    tables = cfg.get("tables") or {}
    out: dict[str, dict] = {}
    for key, spec in sorted(raw.items()):
        match = _COLUMN_KEY_RE.match(str(key))
        if not match or not isinstance(spec, dict):
            raise QualificationError(f"column_units key {key!r} must be `table.column`")
        table, column = match.groups()
        if column not in ((tables.get(table) or {}).get("value_cols") or ()):
            raise QualificationError(
                f"column_units {key!r}: {column!r} is not a value_col of table {table!r}")
        unit = spec.get("unit")
        if not isinstance(unit, str) or not unit.strip():
            raise QualificationError(f"column_units {key!r} needs a unit")
        action = spec.get("on_out_of_range", "error")
        if action not in OUT_OF_RANGE_ACTIONS:
            raise QualificationError(f"column_units {key!r} on_out_of_range must be one of "
                                     f"{OUT_OF_RANGE_ACTIONS}")
        flag = spec.get("flag")
        out[key] = {"unit": unit.strip(), "range": _range_pair(spec.get("range"), key),
                    "on_out_of_range": action,
                    "flag": str(flag).strip() if flag else None}
    return out


def site_unit_conversions(cfg: dict, site: str) -> dict[str, dict]:
    """The validated declarations of ONE site (`site_unit_conversions.<site>`), keyed as
    declared. Two kinds:

    - ``table.column`` (a `column_units` column): ``{from, to, factor}``; `to` must be
      the column's canonical unit; canonical value = charted value x factor.
    - ``table.med_category[charted unit]`` (a dose table): ``{from, to, factor}`` (the
      rows' true unit is `to` and their value x factor is in it; ``factor: 1`` relabels a
      mislabeled unit), or ``{from, action: quarantine}`` (the rows keep their native
      unit concept, ``{med}_{unit}``, and never reach the converted concept).

    Another site's declarations are never returned."""
    every = cfg.get("site_unit_conversions") or {}
    if not isinstance(every, dict):
        raise QualificationError("site_unit_conversions must map site -> declarations")
    raw = every.get(site) or {}
    if not isinstance(raw, dict):
        raise QualificationError(f"site_unit_conversions.{site} must be a mapping")
    columns = column_units(cfg)
    tables = cfg.get("tables") or {}
    out: dict[str, dict] = {}
    for key, decl in sorted(raw.items()):
        where = f"site_unit_conversions.{site}.{key}"
        if not isinstance(decl, dict) or not isinstance(decl.get("from"), str):
            raise QualificationError(f"{where} must declare `from` (the charted unit)")
        dose = _DOSE_KEY_RE.match(str(key))
        if dose:
            table, med, unit = dose.groups()
            if not (tables.get(table) or {}).get("dose"):
                raise QualificationError(f"{where}: {table!r} is not a dose table")
            if normalize_unit(decl["from"]) != normalize_unit(unit):
                raise QualificationError(f"{where}: `from` must be the charted unit {unit!r}")
            if decl.get("action") == "quarantine":
                out[key] = {"from": decl["from"], "action": "quarantine",
                            **({"note": str(decl["note"])} if decl.get("note") else {})}
                continue
            if decl.get("action") is not None:
                raise QualificationError(f"{where}: action must be `quarantine` if given")
        elif str(key) in columns:
            if not isinstance(decl.get("to"), str) or (
                    _normalized_unit(decl["to"]) != _normalized_unit(columns[key]["unit"])):
                raise QualificationError(
                    f"{where}: `to` must be the column's canonical unit "
                    f"{columns[key]['unit']!r}")
        else:
            raise QualificationError(
                f"{where}: not a column_units column nor a `dose_table.med_category[unit]`")
        factor = _positive_number(decl.get("factor"))
        if factor is None or not isinstance(decl.get("to"), str):
            raise QualificationError(f"{where} needs `to` and a positive finite `factor`")
        when = decl.get("when")
        if when is not None:
            # A column conversion takes one condition; a dose correction may bound the
            # charted value on both sides (e.g. ketamine "mg" 0.1-0.35 = mg/kg).
            if (not isinstance(when, dict) or not when or (not dose and len(when) != 1)
                    or any(op not in _WHEN_OPS or _finite_number(v) is None
                           for op, v in when.items())):
                raise QualificationError(
                    f"{where}: `when` must be {'conditions' if dose else 'one condition'} "
                    f"{{op: value}}, op one of {sorted(_WHEN_OPS)}")
            when = {op: float(v) for op, v in sorted(when.items())}
        out[key] = {"from": decl["from"], "to": decl["to"], "factor": factor,
                    **({"when": when} if when else {}),
                    **({"note": str(decl["note"])} if decl.get("note") else {})}
    return out


# A column conversion's optional condition: convert only charted values satisfying it
# (e.g. a site charting FdO2 mostly as percent but sometimes already as a fraction:
# `when: {gt: 1.0}`); every other value is taken as already canonical.
_WHEN_OPS = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}


def _convert_sql(expr: str, decl: dict) -> str:
    """SQL for one declared column conversion (factor, optional `when` condition)."""
    scaled = _scale_sql(expr, decl["factor"])
    when = decl.get("when")
    if not when:
        return scaled
    (op, value), = when.items()
    return f"(CASE WHEN {expr} {_WHEN_OPS[op]} {value!r} THEN {scaled} ELSE {expr} END)"


def _scale_sql(expr: str, factor: float) -> str:
    """`expr` x `factor`, dividing by the integer when `factor` is 1/n so a value on a
    published edge (12000 mL/h -> 200 mL/min) lands exactly on it."""
    inverse = 1.0 / factor
    if factor < 1 and abs(inverse - round(inverse)) < 1e-9 * inverse:
        return f"({expr} / CAST({float(round(inverse))!r} AS DOUBLE))"
    return f"({expr} * CAST({factor!r} AS DOUBLE))"


def _scale_expr(expr: pl.Expr, factor: pl.Expr) -> pl.Expr:
    inverse = 1.0 / factor
    exact = (factor < 1) & ((inverse - inverse.round(0)).abs() < 1e-9 * inverse)
    return pl.when(exact).then(expr / inverse.round(0)).otherwise(expr * factor)


def column_factors(cfg: dict, site: str, table: str) -> dict[str, dict]:
    """``{column: declaration}`` the site declared for `table`'s unit-less value columns."""
    return {key.split(".", 1)[1]: decl
            for key, decl in site_unit_conversions(cfg, site).items()
            if _COLUMN_KEY_RE.match(key) and key.split(".", 1)[0] == table}


def dose_corrections(cfg: dict, site: str, table: str) -> dict[tuple[str, str], dict]:
    """``{(normalized med_category, normalized charted unit): declaration}`` the site
    declared for dose table `table`."""
    out = {}
    for key, decl in site_unit_conversions(cfg, site).items():
        match = _DOSE_KEY_RE.match(key)
        if match and match.group(1) == table:
            out[(normalize_name(match.group(2)), normalize_unit(match.group(3)))] = decl
    return out


def _scale_hint(con, source_sql: str, params: list, lo: float, hi: float,
                current: float) -> tuple[str, float] | None:
    """The `SCALE_HINTS` factor putting the largest share of values in [lo, hi], when it
    beats the current share."""
    shares = ", ".join(
        f"avg(CASE WHEN {_scale_sql('v', f)} BETWEEN {lo!r} AND {hi!r} THEN 1.0 ELSE 0.0 END)"
        for f, _ in SCALE_HINTS)
    row = con.execute(f"SELECT {shares} FROM ({source_sql}) WHERE v IS NOT NULL",
                      params).fetchone()
    best = max(zip(row, SCALE_HINTS), key=lambda x: (x[0] or 0.0))
    if best[0] is None or best[0] <= max(current, 0.0) or best[0] == 0:
        return None
    return best[1][1], float(best[0])


def check_column_units(con, base: Path, cfg: dict, site: str,
                       keep_ids: list | None = None, *, min_cell: int = 10) -> dict:
    """Validate every `column_units` column a site charts, AFTER its declared conversion
    (aggregate-only). A column with more than `unit_normalization.max_out_of_range_share`
    of its non-null values outside the plausible range is refused (under
    ``on_mismatch: error``) with the site, column, observed quantiles and a likely scale
    factor; a column declared ``on_out_of_range: report`` is only reported. Returns the
    data-quality record: ``{site, conversions, columns, semantic_flags}``."""
    columns = column_units(cfg)
    conversions = site_unit_conversions(cfg, site)
    norm = cfg.get("unit_normalization") or {}
    max_share = float(norm.get("max_out_of_range_share", DEFAULT_MAX_OUT_OF_RANGE_SHARE))
    record: dict = {"site": site, "conversions": conversions, "columns": {},
                    "semantic_flags": {k: v["flag"] for k, v in columns.items() if v["flag"]}}
    failures = []
    aliases = site_harmonization(cfg, site)["column_aliases"]
    for key, spec in columns.items():
        table, column = key.split(".", 1)
        tspec = cfg["tables"][table]
        fp = base / f"{tspec['file']}.parquet"
        if not fp.exists():
            continue
        source = parquet_source(fp, aliases.get(table),
                                {c.lower() for c in _parquet_columns(con, fp)})
        if column.lower() not in {c.lower() for c in _parquet_columns(con, fp, source)}:
            continue
        decl = conversions.get(key)
        id_sql, params = _id_filter(keep_ids)
        raw = f"CAST({column} AS DOUBLE)"
        source = (f"SELECT {_convert_sql(raw, decl) if decl else raw} AS v "
                  f"FROM {source} WHERE {tspec['availability_col']} IS NOT NULL "
                  f"{id_sql}")
        lo, hi = spec["range"]
        # float32 storage (0.21 -> 0.2099999...) must not read as out of range.
        lo_t, hi_t = lo - _RANGE_TOL * max(1.0, abs(lo)), hi + _RANGE_TOL * max(1.0, abs(hi))
        n, out, q = con.execute(
            f"SELECT count(v), count(v) FILTER (WHERE v < {lo_t!r} OR v > {hi_t!r}), "
            f"quantile_cont(v, [0.01, 0.5, 0.99]) FROM ({source})", params).fetchone()
        share = out / n if n else 0.0
        entry = {"unit": spec["unit"], "range": [lo, hi], "n": int(n),
                 "out_of_range": int(out), "share": round(share, 6),
                 "p01": None if q is None else round(float(q[0]), 6),
                 "p50": None if q is None else round(float(q[1]), 6),
                 "p99": None if q is None else round(float(q[2]), 6),
                 "conversion": conversions.get(key), "status": "ok"}
        if n and share > max_share:
            entry["status"] = ("reported" if spec["on_out_of_range"] == "report"
                               else "refused")
            hint = _scale_hint(con, source, params, lo, hi, 1.0 - share)
            entry["hint"] = None if hint is None else hint[0]
            if entry["status"] == "refused":
                suggestion = (f"likely scale {hint[0]} ({hint[1]:.1%} in range after it)"
                              if hint else "no simple scale factor fits")
                quantiles = (f" (p1={entry['p01']}, p50={entry['p50']}, p99={entry['p99']})"
                             if n >= min_cell else " (quantiles withheld: small cell)")
                failures.append(
                    f"site {site!r} column {key}: {share:.1%} of {_cell(n, min_cell)} values "
                    f"outside the plausible range [{lo:g}, {hi:g}] {spec['unit']}{quantiles}; "
                    f"{suggestion}. Declare it "
                    f"explicitly: site_unit_conversions.{site}.{key}: {{from: <charted "
                    f"unit>, to: {spec['unit']}, factor: <factor>}}")
        record["columns"][key] = entry
    if failures and norm.get("on_mismatch", "error") == "error":
        raise QualificationError("Implausible unit-less column values: " + "; ".join(failures)
                                 + " " + NODE_ONLY)
    return record


def check_dose_plausibility(dose_events: dict[str, pl.DataFrame], cfg: dict,
                            site: str, *, min_cell: int = 10) -> dict:
    """Per declared dose concept (`dose_plausibility.ranges`, the unit after conversion),
    count running doses (value > 0; 0 is the stop bin) outside the plausible range.
    A share above `max_implausible_share` is refused (under ``on_mismatch: error``) with
    the site, concept and a scale hint, unless the range says ``on_implausible: report``;
    a smaller share is reported. Aggregate-only.
    Returns ``{"doses": {concept: {unit, range, n, implausible, share, status}}}``."""
    block = cfg.get("dose_plausibility") or {}
    ranges = block.get("ranges") or {}
    max_share = float(block.get("max_implausible_share", DEFAULT_MAX_IMPLAUSIBLE_DOSE_SHARE))
    frames = [f.select("concept", "value") for f in dose_events.values()
              if f is not None and len(f)]
    events = pl.concat(frames) if frames else pl.DataFrame(
        schema={"concept": pl.String, "value": pl.Float64})
    record, failures = {}, []
    for concept, spec in sorted(ranges.items()):
        if not isinstance(spec, dict):
            raise QualificationError(f"dose_plausibility.ranges.{concept} must be a mapping")
        lo, hi = _range_pair(spec.get("range"), f"dose_plausibility.ranges.{concept}")
        vals = events.filter((pl.col("concept") == concept) & pl.col("value").is_finite()
                             & (pl.col("value") > 0))["value"]
        n = len(vals)
        if not n:
            continue
        bad = int(((vals < lo - _RANGE_TOL * max(1.0, abs(lo)))
                   | (vals > hi + _RANGE_TOL * max(1.0, abs(hi)))).sum())
        share = bad / n
        entry = {"unit": spec.get("unit"), "range": [lo, hi], "n": n, "implausible": bad,
                 "share": round(share, 6), "status": "ok" if not bad else "reported"}
        action = spec.get("on_implausible", "error")
        if action not in OUT_OF_RANGE_ACTIONS:
            raise QualificationError(f"dose_plausibility.ranges.{concept} on_implausible "
                                     f"must be one of {OUT_OF_RANGE_ACTIONS}")
        if share > max_share and action == "error":
            entry["status"] = "refused"
            arr = vals.to_numpy()
            fits = max(((float(((arr * f >= lo) & (arr * f <= hi)).mean()), label)
                        for f, label in SCALE_HINTS))
            hint = (f"best scale {fits[1]} puts {fits[0]:.1%} in range" if fits[0] > 0
                    else "no simple scale factor fits")
            quantiles = (f"(p1={np.quantile(arr, 0.01):.4g}, p50={np.quantile(arr, 0.5):.4g}, "
                         f"p99={np.quantile(arr, 0.99):.4g})" if n >= min_cell
                         else "(quantiles withheld: small cell)")
            failures.append(
                f"site {site!r} dose {concept}: {share:.1%} of {_cell(n, min_cell)} running "
                f"doses outside [{lo:g}, {hi:g}] {spec.get('unit')} {quantiles}; "
                f"{hint}. Declare a correction "
                f"for the mislabeled charted unit: site_unit_conversions.{site}."
                "<dose_table>.<med_category>[<charted unit>]: {from, to, factor} or "
                "{from, action: quarantine}")
        record[concept] = entry
    if failures and (cfg.get("unit_normalization") or {}).get("on_mismatch",
                                                              "error") == "error":
        raise QualificationError("Implausible medication doses: " + "; ".join(failures)
                                 + " " + NODE_ONLY)
    return {"doses": record}


def _observed_units(fit_events: pl.DataFrame) -> dict[str, str]:
    """Per concept, the most frequent charted unit (ties -> lexicographically first)."""
    observed: dict[str, str] = {}
    if "unit" in fit_events.columns:
        counts = (
            fit_events.filter(pl.col("unit").is_not_null() & (pl.col("unit") != ""))
            .group_by("concept", "unit").len()
            .sort(["concept", "len", "unit"], descending=[False, True, False])
        )
        for concept, unit, _ in counts.iter_rows():
            observed.setdefault(concept, unit)
    return observed


def reference_units(fit_events: pl.DataFrame, segments: dict, cfg: dict,
                    dose_targets: dict[str, str], site: str | None = None) -> dict:
    """R14: the reference unit of every binned concept, plus the dose target units.

    Per concept: the config's canonical unit; else the reference site's most frequent
    charted unit (ties -> lexicographically first); else, for a unit-less wide-table
    column, its declared canonical unit (`column_units`, after the site's conversions);
    else, for a wide device table qualified by `concept_qualifier_col` (ECMO/MCS),
    ``device_metric:<column>``; else None. Hashed into the vocabulary manifest so every
    site checks units identically.

    When the config declares `column_units`, the record also carries them
    (``column_units``) and the reference `site`'s declared conversions
    (``site_conversions``), so a vocabulary built under different unit repairs has a
    different hash and is refused by `validate_vocabulary_artifact`."""
    canonical = cfg.get("unit_normalization", {}).get("concepts", {}) or {}
    observed = _observed_units(fit_events)
    declared = column_units(cfg)
    metrics: dict[str, str] = {}
    sources = _concept_tables(fit_events)
    for name, spec in (cfg.get("tables") or {}).items():
        qualified = bool(spec.get("concept_qualifier_col"))
        for col in spec.get("value_cols") or ():
            unit = (declared.get(f"{name}.{col}") or {}).get("unit")
            if not qualified and unit is None:
                continue
            suffix = f"_{col.lower()}"
            for concept, tables in sources.items():
                if name not in tables:
                    continue
                if (concept == col.lower() and not qualified) or (
                        qualified and concept.endswith(suffix)):
                    metrics.setdefault(concept, unit or f"{DEVICE_METRIC_PREFIX}{col}")
    units = {
        concept: canonical.get(concept) or observed.get(concept) or metrics.get(concept)
        for concept in sorted(segments)
    }
    # CLIF 2.1 units (labs: the mCIDE lab unit; vitals: the 2.1 DDL): an equivalent site
    # spelling (mEq/L for sodium, K/uL, "units" for pH) is recorded as the CLIF unit; a
    # non-equivalent one is kept and listed as a deviation.
    clif = clif_unit_map(cfg)
    deviations = {}
    for concept, unit in units.items():
        expected = clif.get(concept)
        if not expected:
            continue
        if unit is None or units_equivalent(concept, unit, expected, cfg):
            units[concept] = expected
        else:
            deviations[concept] = unit
    record = {"concepts": units, "dose_targets": dict(sorted(dose_targets.items()))}
    if deviations:
        record["clif_unit_deviations"] = dict(sorted(deviations.items()))
    if declared:
        # Hashed with the vocabulary: the rules only (free-text `flag` / `note` stripped,
        # so editing a note never invalidates a vocabulary).
        record["column_units"] = _hashed_column_units(cfg)
        record["site_conversions"] = (strip_notes(site_unit_conversions(cfg, site))
                                      if site is not None else {})
    return record


def _hashed_column_units(cfg: dict) -> dict:
    """`column_units` without its free-text semantic `flag` (reported, not hashed)."""
    return {key: {k: v for k, v in spec.items() if k != "flag"}
            for key, spec in column_units(cfg).items()}


def clif_unit_map(cfg: dict) -> dict[str, str]:
    """{concept: CLIF 2.1 unit} from the mCIDE snapshot (labs, vitals), or {} when the
    config declares no `clif_conformance`."""
    block = cfg.get("clif_conformance")
    if not block:
        return {}
    from clif_validate._vendor.data.clif_conformance import load_snapshot
    return dict(load_snapshot(block["snapshot"]).get("units") or {})


def _unit_plan(fit_events: pl.DataFrame, cfg: dict, dose_targets: dict[str, str],
               site: str | None) -> dict[str, str | None]:
    """Every fit concept's reference unit (`reference_units`) BEFORE segments exist: the
    unit a literature fragment must match."""
    names = fit_events["concept"].drop_nulls().unique().to_list() if len(fit_events) else []
    return reference_units(fit_events, dict.fromkeys(names), cfg, dose_targets,
                           site)["concepts"]


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
              *extra: pl.Expr, extra_stays: pl.DataFrame | None = None) -> pl.DataFrame:
    """Join the canonical episodes (their `times` columns) and keep each ELIGIBLE stay's
    events in the window ``[times[0], end]`` (inclusive); `pos_min` = minutes since
    ``times[0]``, then `target_eligible` and any `extra` columns. Ineligible episodes
    are dropped before the join (after the full artifact is validated) — the same rows
    the post-join eligibility filter keeps. `extra_stays` (validated by their builder,
    `extubation_index_stays`) are appended to the eligible episodes after validation."""
    _check_window_inputs(events, episodes, list(times))
    start = times[0]
    stays = episodes.filter(pl.col("eligible")).select(
        "hospitalization_id", *times, "eligible", "partition")
    if extra_stays is not None and len(extra_stays):
        stays = pl.concat([stays, extra_stays.select(stays.columns).cast(stays.schema)],
                          how="vertical")
    return (
        events.join(
            stays,
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


# Columns the window join adds that nothing after the window needs: the stay's own times
# (pos_min / _le_anchor were derived from them) and the eligibility flag. Dropped before
# the sort so each of the ~100M-row copies the sort and the report hold is ~70 bytes a row
# lighter (memory, 2026-10-03 full-MIMIC review).
_WINDOW_ONLY_COLUMNS = ("admission_dttm", "discharge_dttm", "anchor_dttm", "icu_admit_dttm",
                        "eligible")


def slim_windowed(events: pl.DataFrame) -> pl.DataFrame:
    return events.drop([c for c in _WINDOW_ONLY_COLUMNS if c in events.columns])


def restrict_to_hospitalization_window(
    events: pl.DataFrame,
    episodes: pl.DataFrame,
    treatment_sources: set[str] | None = None,
    *,
    extra_stays: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """GEM (U8): join canonical episodes and retain each ELIGIBLE stay's events from
    hospital admission to discharge (inclusive) — pre-ICU (ED/ward), ICU and post-ICU.
    `pos_min` = minutes since hospital admission; `_le_anchor` marks events at or before
    the ICU-admit+24 h anchor (for the record's `anchor_idx`; false for an
    `extra_stays` stay, which has no anchor)."""
    return _windowed(events, episodes, treatment_sources,
                     ("admission_dttm", "discharge_dttm", "anchor_dttm"), "discharge_dttm",
                     (pl.col("dttm") <= pl.col("anchor_dttm")).fill_null(False)
                     .alias("_le_anchor"), extra_stays=extra_stays)


def restrict_to_observation_window(
    events: pl.DataFrame,
    episodes: pl.DataFrame,
    treatment_sources: set[str] | None = None,
) -> pl.DataFrame:
    """Join canonical episodes and retain only anchor-available ICU events."""
    return _windowed(events, episodes, treatment_sources, ("icu_admit_dttm", "anchor_dttm"),
                     "anchor_dttm")


# ---- CLIF 2.1 harmonization: GCS not testable, BP measurement method ------------------

def gcs_rule(cfg: dict) -> dict | None:
    """The validated `gcs_not_testable` block (applied identically at every site)."""
    rule = cfg.get("gcs_not_testable")
    if rule is None:
        return None
    if not isinstance(rule, dict) or rule.get("table") not in (cfg.get("tables") or {}):
        raise QualificationError("gcs_not_testable needs a configured `table`")
    for key in ("verbal", "total", "token_value"):
        if not isinstance(rule.get(key), str) or not rule[key]:
            raise QualificationError(f"gcs_not_testable.{key} must be a name")
    codes = rule.get("numeric_codes") or []
    if any(_finite_number(c) is None for c in codes):
        raise QualificationError("gcs_not_testable.numeric_codes must be numbers")
    table = rule.get("imputation")
    if (not isinstance(table, list) or not table or any(
            not isinstance(r, dict) or len(r.get("eye_motor") or ()) != 2
            or _finite_number(r.get("verbal")) is None for r in table)):
        raise QualificationError(
            "gcs_not_testable.imputation must list {eye_motor: [lo, hi], verbal} rows")
    for key in ("eye", "motor", "imputed_marker_concept", "imputed_marker_value"):
        if not isinstance(rule.get(key), str) or not rule[key]:
            raise QualificationError(f"gcs_not_testable.{key} must be a name")
    return rule


def _cat_key(expr: pl.Expr) -> pl.Expr:
    return expr.str.strip_chars().str.to_lowercase().str.replace_all(r"[\s_]+", "_")


def gcs_imputed_verbal(eye_motor: float, table: list[dict]) -> float | None:
    """The imputed verbal score for an eye + motor sum (`gcs_not_testable.imputation`)."""
    for row in table:
        lo, hi = row["eye_motor"]
        if lo <= eye_motor <= hi:
            return float(row["verbal"])
    return None


def apply_gcs_not_testable(events: pl.DataFrame, rule: dict) -> tuple[pl.DataFrame, dict]:
    """A GCS verbal score that cannot be tested (a declared numeric or categorical code,
    e.g. MIMIC's 0 for an intubated patient) becomes the categorical value
    `rule.token_value` (token ``gcs_verbal=not_testable``). A total charted at the same
    (stay, time) is not a valid score (MIMIC forces it to 15): it is REPLACED by eye +
    motor + the imputed verbal score (`rule.imputation`, Brennan 2021) and marked by a
    ``gcs_total_source=imputed`` token at the same time; with no single same-time eye and
    motor value the total is dropped. Decided 2026-10-03 (product authority).
    Returns (events, aggregate counts)."""
    counts = {"verbal_not_testable": 0, "total_imputed": 0, "total_dropped": 0}
    if events.is_empty():
        return events, counts
    verbal = pl.col("concept") == rule["verbal"]
    numeric = [float(c) for c in rule.get("numeric_codes") or ()]
    categorical = [re.sub(r"[\s_]+", "_", str(c).strip().lower())
                   for c in rule.get("categorical_codes") or ()]
    nt = verbal & (pl.col("value").is_in(numeric).fill_null(False)
                   | _cat_key(pl.col("cat_value")).is_in(categorical).fill_null(False))
    events = events.with_columns(nt.alias("_nt"))
    counts["verbal_not_testable"] = int(events["_nt"].sum())
    if not counts["verbal_not_testable"]:
        return events.drop("_nt"), counts
    keys = events.filter(pl.col("_nt")).select("hosp_id", "dttm").unique()
    events = events.with_columns(
        pl.when(pl.col("_nt")).then(pl.lit(None, pl.Float64))
        .otherwise(pl.col("value")).alias("value"),
        pl.when(pl.col("_nt")).then(pl.lit(rule["token_value"]))
        .otherwise(pl.col("cat_value")).alias("cat_value"),
    ).drop("_nt")
    # Same-time eye + motor (exactly one distinct value each) -> imputed total.
    parts = (events.filter(pl.col("concept").is_in([rule["eye"], rule["motor"]])
                           & pl.col("value").is_finite().fill_null(False))
             .join(keys, on=["hosp_id", "dttm"], how="semi")
             .group_by("hosp_id", "dttm", "concept")
             .agg(pl.col("value").min().alias("v"), pl.col("value").n_unique().alias("n"))
             .filter(pl.col("n") == 1))
    eye = parts.filter(pl.col("concept") == rule["eye"]).select(
        "hosp_id", "dttm", pl.col("v").alias("_e"))
    motor = parts.filter(pl.col("concept") == rule["motor"]).select(
        "hosp_id", "dttm", pl.col("v").alias("_m"))
    table = rule["imputation"]
    em = eye.join(motor, on=["hosp_id", "dttm"], how="inner").with_columns(
        (pl.col("_e") + pl.col("_m")).alias("_em"))
    imputed_verbal = pl.lit(None, pl.Float64)
    for row in reversed(table):
        lo, hi = row["eye_motor"]
        imputed_verbal = (pl.when(pl.col("_em").is_between(lo, hi))
                          .then(pl.lit(float(row["verbal"]))).otherwise(imputed_verbal))
    em = em.with_columns((pl.col("_em") + imputed_verbal).alias("_imputed")).select(
        "hosp_id", "dttm", "_imputed")
    marked = (events.join(keys.with_columns(pl.lit(True).alias("_k")),
                          on=["hosp_id", "dttm"], how="left")
              .join(em, on=["hosp_id", "dttm"], how="left"))
    total = (pl.col("concept") == rule["total"]) & pl.col("_k").fill_null(False)
    counts["total_imputed"] = int(marked.select(
        (total & pl.col("_imputed").is_not_null()).sum()).item())
    counts["total_dropped"] = int(marked.select(
        (total & pl.col("_imputed").is_null()).sum()).item())
    replaced = total & pl.col("_imputed").is_not_null()
    markers = marked.filter(replaced).select("hosp_id", "dttm").unique().select(
        "hosp_id", "dttm", pl.lit(rule["imputed_marker_concept"]).alias("concept"),
        pl.lit(None, pl.Float64).alias("value"), pl.lit(None, pl.String).alias("unit"),
        pl.lit(rule["imputed_marker_value"]).alias("cat_value"))
    events = (marked.filter(~(total & pl.col("_imputed").is_null()))
              .with_columns(pl.when(replaced).then(pl.col("_imputed"))
                            .otherwise(pl.col("value")).alias("value"))
              .drop("_k", "_imputed"))
    return pl.concat([events, markers], how="diagonal_relaxed"), counts


def apply_bp_method(events: pl.DataFrame, glob: dict, decl: dict | None,
                    site: str, *, min_cell: int = 10) -> tuple[pl.DataFrame, dict]:
    """Emit one ``bp_method=<method>`` token per (stay, time, method) of the declared BP
    concepts, ordered immediately before that method's readings (`_okey` / `_osub` sort
    keys, `_event_sort`). The readings themselves are unchanged. The method comes from the
    site's declared source column (`_bp_src`); an unmapped name, or BP rows at a site with
    no declaration, fail. Returns (events, {method: readings, "tokens": n})."""
    concepts = glob["concepts"]
    is_bp = pl.col("concept").is_in(concepts)
    if events.is_empty() or not len(events.filter(is_bp)):
        return events.drop("_bp_src", strict=False), {}
    if decl is None or "_bp_src" not in events.columns:
        if not glob["require_site_declaration"]:
            return events.drop("_bp_src", strict=False), {}
        raise QualificationError(
            f"site {site!r} charts {', '.join(concepts)} but declares no BP measurement "
            f"method (site_harmonization.{site}.bp_method)")
    names = (events.filter(is_bp).group_by("_bp_src").len()
             .sort("_bp_src", nulls_last=True).iter_rows())
    mapping, unmapped, null_method = {}, [], None
    for name, n in names:
        method = bp_method_of(name, decl)
        if method is None:
            unmapped.append(f"{'<null>' if name is None else name!r} "
                            f"({_cell(n, min_cell)} rows)")
        elif name is None:
            null_method = method
        else:
            mapping[name] = method
    if unmapped:
        raise QualificationError(
            f"site {site!r}: BP source name(s) map to no measurement method: "
            f"{'; '.join(unmapped)}. Add a pattern (or map them to `unknown` explicitly) in "
            f"site_harmonization.{site}.bp_method (the site-local "
            f"configs/sites/{site}.local.yaml for a site's own source names) {NODE_ONLY}")
    method = (pl.when(pl.col("_bp_src").is_null()).then(pl.lit(null_method, pl.String))
              .otherwise(pl.col("_bp_src").replace_strict(
                  mapping, default=None, return_dtype=pl.String)))
    events = events.with_columns(
        pl.when(is_bp).then(method).otherwise(pl.lit(None, pl.String)).alias("_bp_method")
    ).drop("_bp_src")
    readings = events.filter(is_bp & pl.col("value").is_finite().fill_null(False))
    tokens = readings.select("hosp_id", "dttm", "_bp_method").unique().select(
        "hosp_id", "dttm", pl.lit(glob["token_concept"]).alias("concept"),
        pl.lit(None, pl.Float64).alias("value"), pl.lit(None, pl.String).alias("unit"),
        pl.col("_bp_method").alias("cat_value"),
        ("bp|" + pl.col("_bp_method")).alias("_okey"), pl.lit(0, pl.Int8).alias("_osub"))
    counts = {m: int(n) for m, n in readings.group_by("_bp_method").len().iter_rows()}
    counts["tokens"] = len(tokens)
    events = events.with_columns(
        pl.when(is_bp).then("bp|" + pl.col("_bp_method")).otherwise(None).alias("_okey"),
        pl.lit(1, pl.Int8).alias("_osub"),
    ).drop("_bp_method")
    return pl.concat([events, tokens], how="diagonal_relaxed"), dict(sorted(counts.items()))


def _event_sort(events: pl.DataFrame) -> pl.DataFrame:
    """KTD6 order. With BP-method tokens, a BP method token and its readings share the
    sort key ``bp|<method>`` (token first); every other row sorts exactly as before."""
    if "_okey" not in events.columns:
        order = [key for key in EVENT_ORDER if key in events.columns]
        return events.sort(order, nulls_last=True, maintain_order=True)
    keys: list = []
    for key in EVENT_ORDER:
        if key not in events.columns:
            continue
        if key == "concept":
            keys += [pl.coalesce("_okey", "concept"), pl.col("_osub").fill_null(1)]
        keys.append(pl.col(key))
    return events.sort(keys, nulls_last=True, maintain_order=True).drop("_okey", "_osub")


def harmonization_tokens(cfg: dict) -> list[str]:
    """Config-defined categorical tokens every vocabulary carries whether or not the
    reference site charts them (another site's are then never `<unk>`): every BP method,
    the GCS not-testable verbal, and both values of every 0/1 flag column."""
    tokens: list[str] = []
    bp = bp_config(cfg)
    if bp is not None:
        tokens += [categorical_token(bp["token_concept"], m) for m in bp["methods"]]
    gcs = gcs_rule(cfg)
    if gcs is not None:
        tokens.append(categorical_token(gcs["verbal"], gcs["token_value"]))
        tokens.append(categorical_token(gcs["imputed_marker_concept"],
                                        gcs["imputed_marker_value"]))
    for spec in (cfg.get("tables") or {}).values():
        for col in spec.get("flag_cols") or ():
            tokens += [categorical_token(col.lower(), v) for v in ("0", "1")]
    return [t for t in tokens if t]


ALLOWLIST_FORMS = ("categorical", "bare")
DEFAULT_MCIDE_SNAPSHOT = "configs/clif_mcide_2.1.1"


def vocabulary_allowlist(cfg: dict) -> list[dict]:
    """The validated `vocabulary_allowlist` block: ``[{permissible, concept, form}]``,
    each naming a list of the mCIDE snapshot (`clif_conformance.permissible`)."""
    raw = cfg.get("vocabulary_allowlist") or []
    if not isinstance(raw, list):
        raise QualificationError("vocabulary_allowlist must be a list of {permissible, concept}")
    out = []
    for i, entry in enumerate(raw):
        where = f"vocabulary_allowlist[{i}]"
        if not isinstance(entry, dict) or not isinstance(entry.get("permissible"), str):
            raise QualificationError(f"{where} needs `permissible` (a snapshot list name)")
        form = entry.get("form", "categorical")
        if form not in ALLOWLIST_FORMS:
            raise QualificationError(f"{where}.form must be one of {ALLOWLIST_FORMS}")
        concept = entry.get("concept")
        if form == "categorical" and (not isinstance(concept, str) or not concept):
            raise QualificationError(f"{where}: a categorical entry needs `concept`")
        out.append({"permissible": entry["permissible"], "form": form,
                    **({"concept": concept} if form == "categorical" else {})})
    return out


def allowlist_tokens(cfg: dict) -> list[str]:
    """Every permissible CLIF 2.1.1 value of the `vocabulary_allowlist` lists, in the
    token form the tokenizer emits (`categorical_token` for a categorical concept, the
    stripped value itself for a bare concept), in list then snapshot order. Decided
    2026-10-03 (product authority): the vocabulary is fit on MIMIC train only, so these
    keep another site's permissible values off `<unk>`."""
    entries = vocabulary_allowlist(cfg)
    if not entries:
        return []
    block = cfg.get("clif_conformance") or {}
    from clif_validate._vendor.data.clif_conformance import load_snapshot
    # The conformance gate's snapshot, else the repository's pinned CLIF 2.1.1 snapshot.
    permissible = load_snapshot(block.get("snapshot") or DEFAULT_MCIDE_SNAPSHOT)["permissible"]
    tokens: list[str] = []
    for entry in entries:
        values = permissible.get(entry["permissible"])
        if values is None:
            raise QualificationError(
                f"vocabulary_allowlist: the snapshot has no list {entry['permissible']!r}")
        for value in values:
            token = (categorical_token(entry["concept"], value) if entry["form"] == "categorical"
                     else str(value).strip() or None)
            if token and token not in tokens:
                tokens.append(token)
    return tokens


def apply_dose_floors(events: pl.DataFrame, cfg: dict) -> tuple[pl.DataFrame, dict]:
    """Remove running doses below their declared single-dose floor
    (`dose_plausibility.floors`, in the concept's converted unit): charting errors,
    counted per concept, never binned. A 0 (stop) dose is never below a floor."""
    floors = (cfg.get("dose_plausibility") or {}).get("floors") or {}
    if not floors or events.is_empty():
        return events, {}
    for concept, floor in floors.items():
        if _positive_number(floor) is None:
            raise QualificationError(f"dose_plausibility.floors.{concept} must be positive")
    floor = pl.col("concept").replace_strict(
        {c: float(v) for c, v in floors.items()}, default=None, return_dtype=pl.Float64)
    below = ((pl.col("value") > 0) & (pl.col("value") < floor)).fill_null(False)
    flagged = events.filter(below)
    counts = {c: int(n) for c, n in flagged.group_by("concept").len().iter_rows()}
    return events.filter(~below), dict(sorted(counts.items()))


def apply_concept_renames(events: pl.DataFrame, renames: dict[str, str]) -> pl.DataFrame:
    """A site's declared token-concept renames (`site_harmonization.<site>.concept_renames`,
    e.g. MIMIC's conventional cTnT -> `troponin_t_conventional`, so its assay never shares
    the high-sensitivity `troponin_t` bins). The CLIF category itself is unchanged."""
    if not renames or events.is_empty():
        return events
    return events.with_columns(pl.col("concept").replace(renames))


def derived_concept_specs(cfg: dict) -> dict[str, dict]:
    """The validated `derived_concepts` block (non-CLIF concepts computed from CLIF
    columns): ``{name: {table, sum_of, divide_by_weight, unit}}``."""
    out = {}
    for name, spec in sorted((cfg.get("derived_concepts") or {}).items()):
        if (not isinstance(spec, dict) or spec.get("table") not in (cfg.get("tables") or {})
                or not spec.get("sum_of") or not spec.get("unit")):
            raise QualificationError(f"derived_concepts.{name} needs table, sum_of and unit")
        if spec.get("clif_concept", False):
            raise QualificationError(f"derived_concepts.{name} must declare clif_concept: false")
        out[name] = spec
    return out


def derive_concepts(events: pl.DataFrame, table: str, cfg: dict, con, base: Path,
                    keep_ids: list | None, *, site_timezone: str | None = None,
                    ) -> tuple[pl.DataFrame, dict]:
    """Append each derived concept of `table`: the sum of its `sum_of` concepts charted on
    the same row (same stay and time; every component required), divided by the latest
    plausible weight available at or before that time (`divide_by_weight`; the weight
    table's lag applied, as for per-kg doses). Returns (events, {name: counts})."""
    counts: dict[str, dict] = {}
    for name, spec in derived_concept_specs(cfg).items():
        if spec["table"] != table or events.is_empty():
            continue
        parts = [c.lower() for c in spec["sum_of"]]
        rows = (events.filter(pl.col("concept").is_in(parts)
                              & pl.col("value").is_finite().fill_null(False))
                .group_by("hosp_id", "dttm")
                .agg(pl.col("value").sum().alias("_sum"),
                     pl.col("concept").n_unique().alias("_n"), pl.len().alias("_rows")))
        complete = rows.filter((pl.col("_n") == len(parts)) & (pl.col("_rows") == len(parts)))
        entry = {"rows_with_any_component": len(rows), "rows_complete": len(complete),
                 "emitted": 0, "no_weight": 0}
        weight = spec.get("divide_by_weight")
        derived = complete.select("hosp_id", "dttm", pl.col("_sum").alias("value"))
        if weight:
            wspec = cfg["tables"][weight["table"]]
            fp = base / f"{wspec['file']}.parquet"
            lo, hi = _range_pair(weight.get("plausible_kg", [25.0, 400.0]),
                                 f"derived_concepts.{name}.divide_by_weight.plausible_kg")
            lag = int(cfg["tables"][table].get("availability_lag_minutes", 0))
            wlag = int(wspec.get("availability_lag_minutes", 0))
            id_sql, params = _id_filter(keep_ids)
            weights = con.execute(f"""
                SELECT CAST(hospitalization_id AS VARCHAR) AS hosp_id,
                       {wspec['availability_col']} + INTERVAL {wlag} MINUTE AS _avail,
                       avg(CAST({wspec['value_col']} AS DOUBLE)) AS _w
                FROM read_parquet('{fp}')
                WHERE {wspec['concept_col']} = ? AND {wspec['availability_col']} IS NOT NULL
                  AND CAST({wspec['value_col']} AS DOUBLE) BETWEEN {lo!r} AND {hi!r} {id_sql}
                GROUP BY 1, 2""", [weight["concept"], *params]).pl() if fp.exists() else \
                pl.DataFrame(schema={"hosp_id": pl.String, "_avail": pl.Datetime("us", "UTC"),
                                     "_w": pl.Float64})
            # The weights' clock, like every event's, is UTC after ingest (site_config).
            weights, _ = to_utc(weights, ["_avail"], site_timezone)
            weights = weights.filter(pl.col("_avail").is_not_null())
            derived = (derived.with_columns(
                (pl.col("dttm") + pl.duration(minutes=lag)).alias("_avail"))
                .sort("hosp_id", "_avail")
                # Both sides are sorted by (stay, time) above, which `by` requires.
                .join_asof(weights.sort("hosp_id", "_avail"), on="_avail", by="hosp_id",
                           strategy="backward", check_sortedness=False))
            entry["no_weight"] = int(derived["_w"].is_null().sum())
            derived = derived.filter(pl.col("_w").is_not_null()).select(
                "hosp_id", "dttm", (pl.col("value") / pl.col("_w")).alias("value"))
        derived = derived.select("hosp_id", "dttm", pl.lit(name).alias("concept"), "value",
                                 pl.lit(spec["unit"]).alias("unit"),
                                 pl.lit(None, pl.String).alias("cat_value"))
        entry["emitted"] = len(derived)
        counts[name] = entry
        if len(derived):
            events = pl.concat([events, derived], how="diagonal_relaxed")
    return events, counts


def _check_harmonization_binding(blob: dict, hashes: dict, cfg: dict) -> None:
    """The harmonization record (mCIDE snapshot, GCS / BP / flag / weight / unit rules,
    the reference site's declarations) is hashed; a vocabulary built under other global
    rules, or before them, does not bind."""
    record = blob.get("harmonization")
    current = global_record(cfg)
    if record is None and "harmonization" not in hashes:
        if current:
            raise QualificationError(
                "vocabulary predates the CLIF 2.1 harmonization rules this config declares "
                f"(mCIDE gate, GCS, BP method, flags); {RETOKENIZE}")
        return
    if not isinstance(record, dict) or "harmonization" not in hashes:
        raise QualificationError(f"vocabulary lacks its hashed harmonization record; {RETOKENIZE}")
    if hashes["harmonization"] != json_sha256(record):
        raise QualificationError("harmonization hash mismatch")
    if record.get("global") != json.loads(json.dumps(current)):
        raise QualificationError(
            "vocabulary was built under different CLIF harmonization rules (mCIDE snapshot, "
            f"GCS, BP method, flags, weights or unit equivalences) than this config; {RETOKENIZE}")


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


def _matched_quantile_edges(vals: np.ndarray, n_edges: int, forced: list[float],
                            concept: str) -> list[float]:
    """KTD11: `n_edges` interior quantile edges, filling edges lost to ties.

    Starts from `_quantile_edges` at ``n_edges + 1`` bins (forced edges pinned, each
    replacing its nearest quantile edge). Where tied values leave fewer distinct edges,
    each lost (duplicate) quantile is replaced by the next unused distinct observed value
    above it (else the nearest below); a shortfall left after that is filled with the
    unused distinct value farthest from every edge. Candidates exclude the minimum, so
    every bin holds at least one fitting value. With too few distinct values the result
    keeps the smaller count (the caller reports it)."""
    pinned = sorted({float(e) for e in forced if np.isfinite(float(e))})
    series = pl.Series(vals)
    edges = _quantile_edges(series, max(n_edges, len(pinned)) + 1, pinned, concept)
    if len(edges) >= n_edges:
        return edges
    used = set(edges)
    candidates = [float(v) for v in np.unique(vals)[1:]]
    raw = sorted(float(series.quantile(q))
                 for q in np.linspace(0, 1, n_edges + 2)[1:-1])
    seen: set[float] = set()
    for r in raw:
        if len(used) >= n_edges:
            break
        if r not in seen:
            seen.add(r)
            continue
        pick = next((c for c in candidates if c > r and c not in used), None)
        if pick is None:
            pick = next((c for c in reversed(candidates) if c < r and c not in used), None)
        if pick is None:
            break
        used.add(pick)
    while len(used) < n_edges:
        free = [c for c in candidates if c not in used]
        if not free:
            break
        used.add(max(free, key=lambda c: (min((abs(c - e) for e in used), default=0.0),
                                          -c)))
    return sorted(used)


# ---- literature-grounded segments (configs/literature_segments/*.yaml) ----------------

def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _literature_path(source: str | Path) -> Path:
    path = Path(source)
    return path if path.is_absolute() else ROOT / path


def load_literature_segments(source: str | Path) -> dict:
    """Load and validate every literature fragment (`*.yaml` in a directory, or one file).

    Each concept declares `unit`, `decision` (segments | keep_ordinal | keep_quantile) and
    `sources` [{id, supports, verified}]; `segments` also needs `edges` (finite, strictly
    ascending) and `closure` (left = ``[a, b)``, right = ``(a, b]``), and every edge must be
    supported by at least one source (``verified: false`` sources count, and are reported).
    `keep_ordinal` may give `valid_range` [low, high] (integers) and `clinical_cutoffs`.
    A concept named by two fragments is refused. Returns ``{"source", "fragments":
    {file: sha256 of its bytes}, "concepts": {concept: spec + {"fragment": file}}}``."""
    path = _literature_path(source)
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    if not files or not all(f.is_file() for f in files):
        raise QualificationError(f"no literature fragments found at {source}")
    fragments: dict[str, str] = {}
    concepts: dict[str, dict] = {}
    for fp in files:
        data = fp.read_bytes()
        fragments[fp.name] = hashlib.sha256(data).hexdigest()
        doc = yaml.safe_load(data) or {}
        if not isinstance(doc, dict) or not isinstance(doc.get("concepts"), dict):
            raise QualificationError(f"literature fragment {fp.name} has no `concepts` mapping")
        for concept, spec in doc["concepts"].items():
            where = f"literature fragment {fp.name}, concept {concept!r}"
            if concept in concepts:
                raise QualificationError(
                    f"{where}: already defined in {concepts[concept]['fragment']}")
            concepts[concept] = {**_validate_literature_concept(spec, where),
                                 "fragment": fp.name}
    return {"source": str(source), "fragments": fragments,
            "concepts": dict(sorted(concepts.items()))}


def _validate_literature_concept(spec: object, where: str) -> dict:
    if not isinstance(spec, dict):
        raise QualificationError(f"{where} must be a mapping")
    decision = spec.get("decision")
    if decision not in LITERATURE_DECISIONS:
        raise QualificationError(f"{where}: decision must be one of {LITERATURE_DECISIONS}")
    unit = spec.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        raise QualificationError(f"{where} needs a unit")
    sources = spec.get("sources") or []
    if not isinstance(sources, list) or any(
            not isinstance(src, dict) or not isinstance(src.get("id"), str)
            or not isinstance(src.get("verified"), bool) for src in sources):
        raise QualificationError(f"{where}: every source needs an `id` and `verified`")
    out = {"decision": decision, "unit": unit.strip(),
           "rationale": str(spec.get("rationale") or "").strip(),
           "sources": [{"id": src["id"], "verified": src["verified"],
                        "supports": list(src.get("supports") or [])} for src in sources]}
    if decision == "segments":
        edges = spec.get("edges")
        values = [_finite_number(e) for e in edges] if isinstance(edges, list) else []
        if not values or any(v is None for v in values) or any(
                b <= a for a, b in zip(values, values[1:])):
            raise QualificationError(f"{where}: edges must be finite and strictly ascending")
        if spec.get("closure") not in LITERATURE_CLOSURES:
            raise QualificationError(f"{where}: closure must be one of {LITERATURE_CLOSURES}")
        by_edge = {e: [src for src in sources
                       if any(_finite_number(x) is not None and np.isclose(float(x), e)
                              for x in src.get("supports") or ())] for e in values}
        unsupported = [e for e, srcs in by_edge.items() if not srcs]
        if unsupported:
            raise QualificationError(
                f"{where}: edge(s) {unsupported} are not supported by any source")
        out.update(edges=values, closure=spec["closure"],
                   unverified_only_edges=[e for e, srcs in by_edge.items()
                                          if not any(s["verified"] for s in srcs)])
    elif decision == "keep_ordinal":
        rng = spec.get("valid_range")
        if rng is not None:
            if (not isinstance(rng, list) or len(rng) != 2
                    or any(_finite_number(v) is None or float(v) != round(float(v))
                           for v in rng) or float(rng[0]) >= float(rng[1])
                    or float(rng[1]) - float(rng[0]) > 100):
                raise QualificationError(
                    f"{where}: valid_range must be two integers [low, high], low < high")
            out["valid_range"] = [float(rng[0]), float(rng[1])]
        out["clinical_cutoffs"] = [str(c) for c in spec.get("clinical_cutoffs") or ()]
    return out


def _unit_text(unit: str) -> str:
    """A fragment unit without its trailing parenthetical qualifier (``fraction (0-1)``
    -> ``fraction``, ``points (3-15)`` -> ``points``)."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", unit).strip() or unit


def literature_segments(edges: list[float], closure: str, *, dose: bool = False,
                        forced: tuple | list = (), direction: str | None = None) -> list[dict]:
    """Interior `edges` -> segments with unbounded ends; ``closure: left`` puts a value on
    an edge in the bin above (``[a, b)``), ``right`` in the bin below (``(a, b]``), as the
    CSV interval flags are honoured. A dose concept keeps its ``[0, 0]`` stop bin with an
    open running-dose segment ``(0, e0)`` / ``(0, e0]`` below the first edge (policy
    step 8: a running dose never bins as stopped)."""
    left = closure == "left"
    bounds = [float(e) for e in edges]
    segs = [make_segment(None, bounds[0], False, not left)]
    segs += [make_segment(a, b, left, not left) for a, b in zip(bounds, bounds[1:])]
    segs.append(make_segment(bounds[-1], None, left, False))
    for t in sorted({float(e) for e in forced if math.isfinite(float(e))}):
        segs = _apply_forced_edge(segs, t, direction)
    if dose:
        if bounds[0] <= 0.0:
            raise QualificationError("a dose concept's literature edges must be positive")
        segs[0] = make_segment(0.0, segs[0]["hi"], False, segs[0]["hi_closed"])
        segs = [make_segment(0.0, 0.0, True, True), *segs]
    validate_partition(segs)
    return segs


def _plan_literature(loaded: dict, values: dict, csv_named: set[str],
                     concept_units: dict, include_absent: bool = False) -> tuple[dict, dict]:
    """Which fragment concepts apply, after the CSV-wins and unit rules: ``(plan
    {concept: spec}, record)``. A CSV concept is ignored with a warning; a concept with
    no fitting value is skipped with a warning; a unit that differs from the concept's
    reference unit (after declared unit conversions) is refused. A concept with no
    reference unit (scores, age) keeps the fragment's stated unit, recorded as assumed."""
    plan: dict[str, dict] = {}
    ignored, absent = [], []
    for concept, spec in loaded["concepts"].items():
        if concept in csv_named:
            warnings.warn(f"literature fragment {spec['fragment']}: {concept!r} is defined "
                          "by the physician CSV, which wins; the fragment entry is ignored")
            ignored.append(concept)
            continue
        if concept not in values:
            if include_absent and spec["decision"] == "segments":
                # `literature_coverage: all_segments`: published edges need no fit data, so
                # another site's events of a concept the reference site lacks are binned.
                plan[concept] = {**spec, "unit_check": "not_fit",
                                 "reference_unit": concept_units.get(concept)}
                continue
            absent.append(concept)
            continue
        expected = concept_units.get(concept)
        if not expected or str(expected).startswith(DEVICE_METRIC_PREFIX):
            unit_check = "assumed"
        elif _normalized_unit(_unit_text(spec["unit"])) == _normalized_unit(expected):
            unit_check = "matched"
        else:
            raise QualificationError(
                f"literature fragment {spec['fragment']}: {concept!r} unit "
                f"{spec['unit']!r} is not the concept's reference unit {expected!r} (after "
                "declared unit conversions); fix the fragment or the unit repair")
        plan[concept] = {**spec, "unit_check": unit_check, "reference_unit": expected}
    if absent:
        warnings.warn(f"{len(absent)} literature fragment concept(s) are not numeric "
                      f"concepts in the fit events and were skipped: {', '.join(absent)}")
    record = {
        "source": loaded["source"],
        "precedence": list(BINNING_SOURCES),
        "fragments": dict(loaded["fragments"]),
        "concepts": {c: _literature_entry(spec) for c, spec in sorted(plan.items())},
        "ignored_csv": sorted(ignored),
        "absent": sorted(absent),
    }
    return plan, record


def _literature_entry(spec: dict) -> dict:
    """Provenance of one applied fragment concept (source ids only, never quotes)."""
    entry = {"decision": spec["decision"], "unit": spec["unit"],
             "unit_check": spec["unit_check"], "reference_unit": spec["reference_unit"],
             "fragment": spec["fragment"],
             "sources": [s["id"] for s in spec["sources"]],
             "verified": sum(1 for s in spec["sources"] if s["verified"]),
             "unverified": sum(1 for s in spec["sources"] if not s["verified"])}
    if spec["decision"] == "segments":
        entry.update(edges=spec["edges"], closure=spec["closure"],
                     unverified_only_edges=spec["unverified_only_edges"])
    elif spec["decision"] == "keep_ordinal":
        entry.update(valid_range=spec.get("valid_range"),
                     clinical_cutoffs=spec["clinical_cutoffs"])
    else:
        entry["rationale"] = spec["rationale"]
    return entry


def merge_literature_records(main: dict, extra: dict) -> dict:
    """The 24 h build's literature record extended by the GEM extension's (concepts
    charted only outside the 24 h window)."""
    if not extra:
        return main
    if not main:
        return extra
    concepts = {**extra.get("concepts", {}), **main.get("concepts", {})}
    return {**main, "concepts": dict(sorted(concepts.items())),
            "absent": sorted(set(main.get("absent", ())) - set(concepts)),
            "ignored_csv": sorted(set(main.get("ignored_csv", ()))
                                  | set(extra.get("ignored_csv", ())))}


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
                   granularity: dict | None = None,
                   concept_units: dict | None = None,
                   literature_out: dict | None = None,
                   ) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Segments for every numeric concept + its binning source (R4; KTD3).

    `fit_events` must already be the reference site's fit (train) partition. Under
    `coverage: all`, every concept with at least one finite training value is binned,
    choosing its source in priority order:

    1. ``csv``: the physician CSV defines the concept (clinical_segment scheme only).
    1b. ``literature`` (clinical_segment scheme with `literature_source`): a fragment in
       configs/literature_segments/ gives published edges and a closure
       (`load_literature_segments`, `literature_segments`); never overrides a CSV concept
       (warned, ignored); its unit must equal the concept's reference unit
       (`concept_units`, after declared unit conversions; default: the most frequent
       charted unit in `fit_events`). A fragment's ``keep_ordinal`` gives one point bin per
       scale level of its `valid_range` (source ``ordinal``) and ``keep_quantile`` skips
       the ordinal rule (source ``quantile``). `literature_out`, when given, is filled with
       the hashed provenance record (fragment hashes, applied concepts with edges,
       closure, source ids and verified counts, ignored CSV and absent concepts).
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

    MATCHED GRANULARITY (KTD11; decile scheme with `matched_granularity: true`): each
    quantile concept requests the segment count the clinical_segment scheme gives it on
    the same fit events (`segment_source` required), and tied quantiles are filled from
    unused distinct observed values (`_matched_quantile_edges`).
    `decile_forced_edges` (default true) pins `forced_edges` in the decile scheme; the
    plain population-decile arm sets it false. `granularity`, when given, is filled with
    the aggregate record: ``reference_scheme``, ``forced_edges``, ``matched`` (concepts
    at the clinical count) and ``exceptions`` ({concept: requested, built,
    distinct_values}) for every concept that keeps a smaller count.

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
        if not bin_cfg.get("decile_forced_edges", True):
            forced = {}
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
    all_csv = bin_cfg.get("csv_coverage", "targets_only") == "all_measurements"
    if bin_cfg.get("csv_coverage", "targets_only") not in ("targets_only", "all_measurements"):
        raise ValueError("value_binning.csv_coverage must be targets_only or all_measurements")
    csv_segments = (
        build_clinical_segment_bins(
            ROOT / csv_source,
            None if all_csv else sorted(set(values) | set(target_concepts)),
            forced, directions)
        if csv_source else {}
    )
    if all_csv and csv_source:
        # A CSV medication the reference site never charts is still a dose concept: its
        # running doses never bin as stopped at another site (policy step 8).
        doses |= _csv_medications(ROOT / csv_source)
    literature: dict[str, dict] = {}
    if csv_source and bin_cfg.get("literature_source"):
        loaded = load_literature_segments(bin_cfg["literature_source"])
        csv_named = set(csv_segments) | set(build_clinical_segment_bins(
            ROOT / csv_source, sorted(loaded["concepts"])))
        literature, record = _plan_literature(
            loaded, values, csv_named,
            _observed_units(fit_events) if concept_units is None else concept_units,
            include_absent=bin_cfg.get("literature_coverage") == "all_segments")
        # A medication fragment concept absent from the fit is still a dose concept.
        doses |= {c for c, spec in literature.items()
                  if c not in values and spec["fragment"] == "medications.yaml"}
        if literature_out is not None:
            literature_out.update(record)

    reference = None
    if csv_source is None and bin_cfg.get("matched_granularity"):
        if not bin_cfg.get("segment_source"):
            raise ValueError("matched_granularity needs value_binning.segment_source: the "
                             "decile arm requests each concept's clinical-arm bin count")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the clinical arm's own build warns
            reference, _ = build_segments(
                {**bin_cfg, "scheme": "clinical_segment", "matched_granularity": False},
                fit_events, target_concepts, directions, tables=tables,
                quantile_concepts=quantile_concepts, concept_units=concept_units)
        if granularity is not None:
            granularity.update({"reference_scheme": "clinical_segment",
                                "forced_edges": bool(forced), "matched": 0,
                                "exceptions": {}})

    segments: dict[str, list[dict]] = {}
    sources: dict[str, str] = {}
    for concept in sorted(set(values) | set(csv_segments) | set(literature)):
        is_dose = concept in doses
        concept_forced = forced.get(concept, ())
        direction = directions.get(concept)
        if concept in csv_segments:
            segs = csv_segments[concept]
            segments[concept] = with_zero_point(segs) if is_dose else segs
            sources[concept] = "csv"
            continue
        lit = literature.get(concept)
        if lit is not None and lit["decision"] == "segments":
            segments[concept] = literature_segments(
                lit["edges"], lit["closure"], dose=is_dose, forced=concept_forced,
                direction=direction)
            sources[concept] = "literature"
            continue
        vals = np.sort(values[concept])
        if lit is not None and lit["decision"] == "keep_ordinal" and (
                lit.get("valid_range") or is_ordinal(vals, max_distinct)):
            lo, hi = lit.get("valid_range") or (None, None)
            points = (ordinal_segments(np.arange(lo, hi + 1).tolist()) if lo is not None
                      else ordinal_segments(vals.tolist()))
            segments[concept] = with_zero_point(points) if is_dose else points
            sources[concept] = "ordinal"
            continue
        force_quantile = (concept in (quantile_concepts or ())
                          or (lit is not None and lit["decision"] == "keep_quantile"))
        fit_vals = vals[vals > 0] if is_dose else vals
        to_segments = dose_segments_from_edges if is_dose else segments_from_edges
        if len(fit_vals) < min_count:
            segments[concept] = to_segments([], concept_forced, direction)
            sources[concept] = "single"
        elif (csv_source and not force_quantile
              and is_ordinal(vals, max_distinct)):
            # Point bins already separate every integer, so forced edges add nothing.
            points = ordinal_segments(vals.tolist())
            # Policy step 8: a dose's stop bin, with no gap to its first positive point.
            segments[concept] = with_zero_point(points) if is_dose else points
            sources[concept] = "ordinal"
        elif reference is not None and concept in reference:
            requested = len(reference[concept])
            edges = _matched_quantile_edges(fit_vals, requested - 1 - int(is_dose),
                                            list(concept_forced), concept)
            segments[concept] = to_segments(edges, concept_forced, direction)
            sources[concept] = "quantile"
        else:
            edges = _quantile_edges(pl.Series(fit_vals), n_quantile_bins,
                                    list(concept_forced), concept)
            segments[concept] = to_segments(edges, concept_forced, direction)
            sources[concept] = "quantile"
        if reference is not None and granularity is not None and concept in reference:
            requested, built = len(reference[concept]), len(segments[concept])
            if built == requested:
                granularity["matched"] += 1
            else:
                granularity["exceptions"][concept] = {
                    "requested": requested, "built": built,
                    "distinct_values": len(np.unique(fit_vals))}
    if reference is not None and granularity is not None:
        # Clinical-arm concepts with no fitting value (CSV targets absent from the fit).
        granularity["not_fit"] = sorted(set(reference) - set(segments))
    return segments, sources


def _csv_medications(csv_path: Path) -> set[str]:
    """The physician CSV's medication measurements (aliased, policy step 7)."""
    import csv as _csv

    from clif_validate._vendor.data.segments import alias_measurement
    with open(csv_path, newline="") as fh:
        return {alias_measurement((row.get("measurement") or "").strip())
                for row in _csv.DictReader(fh)
                if (row.get("category") or "").strip() == "medications"}


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
                    granularity: dict | None = None,
                    literature: dict | None = None, site: str | None = None,
                    ) -> tuple[dict, dict, dict, dict, dict]:
    """U8: extend the 24 h-fit vocabulary so the GEM artifact shares it.

    Every concept the 24 h train window charts keeps its segments, binning source,
    reference unit and token ids. Concepts charted ONLY in the full-hospitalization
    train events (`gem_fit`) get segments by the same source priority (`build_segments`)
    fit on those events; their tokens, and categorical values of existing concepts seen
    only outside the 24 h window (each in at least `min_category_stays` distinct
    full-hospitalization train stays), are appended after every 24 h token. The fixed
    ADMISSION// / DISCHARGE// allowlist is appended last, present even when no stay has
    that admission type or disposition. `literature` (the 24 h build's literature record)
    is extended in place with the fragment concepts the extension applies. Returns ``(vocab, segments, binning_sources,
    reference_units, concept_sources)``."""
    known = set(fit_events["concept"].drop_nulls().unique().to_list()) | set(edges)
    extra_fit = gem_fit.filter(pl.col("concept").is_not_null()
                               & ~pl.col("concept").is_in(sorted(known)))
    extra_edges: dict = {}
    if not extra_fit.is_empty():
        extra_granularity: dict = {}
        extra_literature: dict = {}
        with warnings.catch_warnings():
            # Fragment concepts absent here were already reported by the 24 h build.
            warnings.simplefilter("ignore")
            built, built_sources = build_segments(
                bin_cfg, extra_fit, [], directions, tables=cfg["tables"],
                quantile_concepts=quantile_concepts, granularity=extra_granularity,
                concept_units=_unit_plan(extra_fit, cfg, target_units, site),
                literature_out=extra_literature)
        if literature is not None and literature:
            merged = merge_literature_records(literature, {
                **extra_literature,
                "concepts": {c: e for c, e in extra_literature.get("concepts", {}).items()
                             if c not in edges}})
            literature.clear()
            literature.update(merged)
        if granularity and extra_granularity:
            granularity["matched"] += extra_granularity["matched"]
            granularity["exceptions"].update(extra_granularity["exceptions"])
        extra_edges = {c: segs for c, segs in built.items() if c not in edges}
        if extra_edges:
            extra_units = reference_units(extra_fit, extra_edges, cfg, target_units)
            edges = dict(sorted({**edges, **extra_edges}.items()))
            binning_sources = dict(sorted(
                {**binning_sources, **{c: built_sources[c] for c in extra_edges}}.items()))
            # Every other field (dose targets, column units, site conversions) is the
            # 24 h build's, unchanged.
            units = {**units, "concepts": dict(sorted({**units["concepts"],
                                                       **extra_units["concepts"]}.items()))}
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
                 stats: dict | None, continuation_header: bool = False) -> pl.DataFrame:
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
        # A stay with no ICU+24 h anchor (an added extubation index stay) has none.
        pl.when(pl.col("anchor_min").is_not_null())
        .then(1 + pl.col("_n_le_anchor").fill_null(0)).alias("anchor_idx"),
    )
    stay_rows, starts, ends, index, n_windows = [], [], [], [], []
    header_ids = None
    if continuation_header:
        # CONTINUATION HEADER (src/data/dataset.py): every continuation window of a stay
        # is cut max_tokens - header_len long, so the loader can re-insert the stay's
        # header (<bos>, ADMISSION//, static tokens) within max_tokens.
        from clif_validate._vendor.data.dataset import gem_window_bounds_with_header, header_token_ids, \
            stay_header_length
        header_ids = header_token_ids(vocab)
        heads64 = framed["token"].list.head(64).to_list()
    for row, n in enumerate(framed["token"].list.len().to_list()):
        if header_ids is None:
            bounds = gem_window_bounds(n, max_tokens)
        else:
            bounds = gem_window_bounds_with_header(
                n, max_tokens, stay_header_length(heads64[row], header_ids))
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


def model_continuation_header(path: Path | None = None) -> bool:
    """`trunk.continuation_header` of the model config (configs/model.yaml): when true, GEM
    continuation windows are cut with room for the stay's header."""
    fp = path or ROOT / "configs/model.yaml"
    if not fp.exists():
        return False
    return bool(((yaml.safe_load(fp.read_text()) or {}).get("trunk") or {})
                .get("continuation_header", False))


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
            # ORA value-regression target. An <unk> (a concept the fit partition never
            # saw) has no frozen stats to normalize against, so it carries no value.
            valnum.append(nan if v is None or hard == unk_id else float(v))
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


# `--workers 0` (every usable CPU) is capped: each worker holds a chunk's Python objects
# (~1 GB at the default chunk budget), and past ~8 the encode is bound by the parent's
# concatenation, not by the workers.
MAX_AUTO_WORKERS = 8


def usable_cpus() -> int:
    """CPUs this process may run on: the scheduler affinity mask where the platform has
    one (Linux: a cgroup / taskset limit), else `os.cpu_count()`."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


def resolve_workers(workers: int | None) -> int:
    """`workers` 0 -> every usable CPU (`usable_cpus`), capped at `MAX_AUTO_WORKERS`; a
    positive count is used as is."""
    if workers is None:
        return 1
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError(f"workers must be a non-negative integer, got {workers!r}")
    return workers if workers > 0 else min(usable_cpus(), MAX_AUTO_WORKERS)


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
        # Each spawned worker is single-threaded in polars: N workers x a full polars
        # thread pool each would oversubscribe the node. The children read the variable
        # at import; the parent's own pool (already started) is unaffected.
        previous = os.environ.get("POLARS_MAX_THREADS")
        os.environ["POLARS_MAX_THREADS"] = "1"
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(bounds)),
                                     mp_context=context) as pool:
                collect(pool.map(_encode_chunk, jobs))
        finally:
            if previous is None:
                os.environ.pop("POLARS_MAX_THREADS", None)
            else:
                os.environ["POLARS_MAX_THREADS"] = previous
    shards = pl.concat(parts, how="vertical").rechunk()
    return shards, dict(sorted(unk.items()))


# ---- memory gate, site binding, extubation index stays, CRRT coverage --------------------

# A full-site tokenization holds the windowed events, their sort and the encoded shard at
# once: measured at 33-36 GB peak on full MIMIC before the window pushdown (2026-10-03
# review). Below this much free RAM the run is likely to swap or be killed: warn first.
MEMORY_WARN_FREE_BYTES = 48 * 2**30


def available_memory_bytes() -> int | None:
    """Free RAM the kernel can hand out without swapping: Linux ``MemAvailable``
    (/proc/meminfo); macOS free + inactive + speculative pages (vm_stat); else None."""
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        fields = {}
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                fields[key] = int(parts[0]) * 1024
        return fields.get("MemAvailable", fields.get("MemFree"))
    import subprocess
    import sys

    if sys.platform == "darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                                 check=False).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            pages = {k.strip(): int(v.strip().rstrip(".")) for k, v in
                     (line.split(":", 1) for line in out.splitlines()[1:] if ":" in line)}
            return page * sum(pages.get(k, 0) for k in
                              ("Pages free", "Pages inactive", "Pages speculative"))
        except (OSError, AttributeError, ValueError):
            return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (OSError, ValueError, AttributeError):
        return None


def memory_gate(minimum: int = MEMORY_WARN_FREE_BYTES) -> int | None:
    """Warn (never fail) when less than `minimum` bytes of RAM are free before a run."""
    free = available_memory_bytes()
    if free is not None and free < minimum:
        print(f"  [warn] only {free / 2**30:.1f} GiB of RAM free (< {minimum / 2**30:.0f} GiB): "
              "a full-site tokenization may swap or be killed; close other jobs first "
              "(docs/plans/l40-runbook.md, before step 8)", flush=True)
    return free


def site_record(cfg: dict, site: str) -> dict:
    """What a site's shards are bound to beyond the vocabulary: the site's own
    harmonization declarations, unit conversions and profile (after the site-local merge,
    free-text notes stripped)."""
    decl = site_harmonization(cfg, site)
    return {"site": site,
            "harmonization": strip_notes({k: v for k, v in decl.items() if k != "declared"}),
            "unit_conversions": strip_notes(site_unit_conversions(cfg, site)),
            "profile": profile_record(cfg, site)}


def site_binding(cfg: dict, site: str) -> dict[str, str]:
    """The site fields every shard row's ``artifact_hashes`` carries next to the
    vocabulary binding: the site name and the SHA-256 of `site_record`, so a shard is
    reproducibly bound to the declarations (local ones included) it was built with."""
    return {"site": str(site), "site_declarations": json_sha256(site_record(cfg, site))}


def shard_binding(vocab_artifact: dict, cfg: dict, site: str) -> dict[str, str]:
    return {**artifact_binding(vocab_artifact), **site_binding(cfg, site)}


_GEM_STAY_COLUMNS = ("hospitalization_id", "admission_dttm", "discharge_dttm", "icu_admit_dttm",
                     "anchor_dttm", "discharge_category", "admission_type_category",
                     "partition", "eligible")


def extubation_index_stays(cohort: pl.DataFrame, episodes: pl.DataFrame, con, base: Path,
                           cfg: dict, site: str) -> tuple[pl.DataFrame, dict]:
    """Item 13 (product authority, 2026-10-03): the INDEX hospitalization of every
    eligible extubation that is not an eligible episode of the artifact (the extubation
    fell in another, or an ineligible, hospitalization of the patient), as GEM stays:
    ``(stays, counts)``. Each stay takes the extubation cohort's partition (inherited from
    the patient's episode-artifact partition, else the split rule); it has no ICU+24 h
    anchor (`anchor_dttm` null). Aggregate counts only: `added`, `not_found` (not in the
    hospitalization table, or no admission time)."""
    from clif_validate._vendor.data.site_config import episode_site, require_site_match

    need = {"hospitalization_id", "eligible", "partition"}
    missing = sorted(need - set(cohort.columns))
    if missing:
        raise QualificationError(f"extubation cohort is missing {', '.join(missing)}")
    recorded = cohort["site"].drop_nulls().unique().to_list() if "site" in cohort.columns else []
    require_site_match(recorded[0] if len(recorded) == 1 else None, site, "extubation cohort")
    require_site_match(episode_site(episodes), site, "episode artifact")
    cohort_eps = (cohort["episode_sha256"].drop_nulls().unique().to_list()
                  if "episode_sha256" in cohort.columns else [])
    artifact_eps = episodes["episode_sha256"].unique().to_list()
    if cohort_eps and cohort_eps != artifact_eps:
        raise QualificationError("the extubation cohort was built from another episode "
                                 "artifact (episode_sha256 differs): rebuild it first")
    eligible_ids = set(episodes.filter(pl.col("eligible"))["hospitalization_id"].to_list())
    index = (cohort.filter(pl.col("eligible").fill_null(False))
             .select(pl.col("hospitalization_id").cast(pl.String), "partition")
             .filter(~pl.col("hospitalization_id").is_in(sorted(eligible_ids)))
             .unique("hospitalization_id"))
    schema = {"hospitalization_id": pl.String, "admission_dttm": pl.Datetime("us", "UTC"),
              "discharge_dttm": pl.Datetime("us", "UTC"),
              "icu_admit_dttm": pl.Datetime("us", "UTC"),
              "anchor_dttm": pl.Datetime("us", "UTC"), "discharge_category": pl.String,
              "admission_type_category": pl.String, "partition": pl.String,
              "eligible": pl.Boolean}
    if index.is_empty():
        return pl.DataFrame(schema=schema), {"added": 0, "not_found": 0}
    source = cfg.get("static_source") or {}
    hosp_fp = base / f"{source.get('hospitalization_file', 'clif_hospitalization')}.parquet"
    if not hosp_fp.exists():
        raise QualificationError(f"{hosp_fp.name} is required to add extubation index stays")
    present = {c.lower() for c in _parquet_columns(con, hosp_fp)}
    admission_type = ("CAST(admission_type_category AS VARCHAR)"
                      if "admission_type_category" in present else "CAST(NULL AS VARCHAR)")
    con.register("_clif_index_ids", index.select("hospitalization_id").to_arrow())
    rows = con.execute(f"""
        SELECT CAST(hospitalization_id AS VARCHAR) AS hospitalization_id, admission_dttm,
               discharge_dttm, CAST(discharge_category AS VARCHAR) AS discharge_category,
               {admission_type} AS admission_type_category
        FROM read_parquet('{hosp_fp}')
        WHERE CAST(hospitalization_id AS VARCHAR) IN (SELECT hospitalization_id FROM _clif_index_ids)
    """).pl()
    con.unregister("_clif_index_ids")
    profile = site_profile(cfg, site)
    rows, _ = to_utc(rows, ["admission_dttm", "discharge_dttm"], profile["site_timezone"])
    rows, _ = censor_open_stays(rows, profile["extraction_dttm"])
    rows = rows.filter(pl.col("admission_dttm").is_not_null()
                       & pl.col("discharge_dttm").is_not_null())
    stays = (index.join(rows, on="hospitalization_id", how="inner")
             .with_columns(pl.lit(None, pl.Datetime("us", "UTC")).alias("icu_admit_dttm"),
                           pl.lit(None, pl.Datetime("us", "UTC")).alias("anchor_dttm"),
                           pl.lit(True).alias("eligible"))
             .select(list(schema)).cast(schema).sort("hospitalization_id"))
    return stays, {"added": stays.height, "not_found": index.height - stays.height}


def setting_coverage(con, source: str, columns: list[str], present: set[str]) -> dict:
    """Rows of a wide table and, per configured column, the rows where it is charted
    (aggregate counts; the report suppresses small cells and withholds their shares)."""
    cols = [c for c in columns if c.lower() in present]
    if not cols:
        return {"rows": 0, "with": {}}
    counts = con.execute(
        "SELECT count(*), " + ", ".join(f"count({c})" for c in cols) + f" FROM {source}"
    ).fetchone()
    return {"rows": int(counts[0]),
            "with": {c: int(n) for c, n in zip(cols, counts[1:])},
            "absent_columns": sorted(c for c in columns if c.lower() not in present)}


def crrt_coverage(con, base: Path, cfg: dict, harmonization: dict, name: str,
                  window: str | None) -> dict | None:
    """Item 6 (Rush onboarding): per site, the share of CRRT rows (of the windowed stays)
    with each setting present. A site charting mostly whether a patient was on dialysis
    has low shares, and the derived effluent dose (all four rates on one row) is then
    mostly a MIMIC concept. None when the table is absent."""
    spec = cfg["tables"].get(name)
    if not spec:
        return None
    fp = base / f"{spec['file']}.parquet"
    if not fp.exists():
        return None
    spec = _harmonized_spec(con, fp, spec, harmonization, name)
    source = _from(fp, spec)
    if window is not None:
        source = _windowed_source(con, source, spec, window)
    present = {c.lower() for c in _parquet_columns(con, fp, spec.get("_from_sql"))}
    columns = [*(spec.get("value_cols") or ()), *(spec.get("categorical_value_cols") or ())]
    return setting_coverage(con, source, columns, present)


def _write_shard(frame: pl.DataFrame, path: Path) -> None:
    """Write a shard grouped by partition (stable: the stay order within a partition is
    unchanged) in row groups of `SHARD_ROW_GROUP_ROWS` rows, so a partition filter
    (`pl.scan_parquet(...).filter(partition == ...)`, the GemCorpus cache build, the
    value-stats fit) skips the other partitions' row groups by their statistics."""
    if len(frame) and "partition" in frame.columns:
        frame = frame.sort("partition", maintain_order=True, nulls_last=True)
    frame.write_parquet(path, row_group_size=SHARD_ROW_GROUP_ROWS, statistics=True)


# Rows (stays or GEM windows) per parquet row group of a written shard.
SHARD_ROW_GROUP_ROWS = 4096


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
                  encode_chunk_events: int | None = None,
                  continuation_header: bool | None = None,
                  extubation_cohort: pl.DataFrame | None = None):
    """Tokenize one site.

    `vocab_artifact` None builds the frozen tokenizer-v2 vocabulary (reference site);
    otherwise it is a whole `vocab.json` blob, validated (`validate_vocabulary_artifact`)
    and applied unchanged. Every stay row of `events.parquet` carries the artifact
    binding (`artifact_hashes`: tokenizer version, vocabulary and segments hashes, and
    the site's binding `site_binding`: its name and the hash of its declarations).
    `stats`, if given, is filled with aggregate-only run counts (`dose_conversion`:
    {dose table: {status: rows}}) for the tokenization report. Returns
    ``(vocab, segments)``.

    `trajectory` (U8, KTD10): ``icu_24h`` (default) writes events.parquet + vocab.json.
    ``hospitalization`` writes ONLY gem_events.parquet beside them, with the SAME frozen
    vocabulary, which must be imported (`vocab_artifact`; the reference site passes the
    vocab.json its icu_24h build just wrote) and must carry the `gem` allowlist. One row
    per window of at most `max_tokens` (default `gem.max_tokens`) tokens.
    `extubation_cohort` (hospitalization only; item 13): the site's extubation cohort;
    the index stay of every eligible extubation that is not an eligible episode is added
    to the GEM stays (`extubation_index_stays`).

    One reference build, one vocabulary: when the config has a `gem` block, the build
    keeps the 24 h fit unchanged (same segments and ids for every concept in the 24 h
    train window) and appends (a) tokens for concepts / categorical values seen ONLY in
    the full-hospitalization train events (ED and ward ADT locations, ward-only labs) —
    their bins fit on those train events — and then (b) the fixed ADMISSION// and
    DISCHARGE// allowlist. The CLIF 2.1.1 mCIDE allowlist (`allowlist_tokens`) follows
    the data-derived 24 h tokens.

    Site profile (`site_config`): the site's git-ignored local declarations are merged
    first; naive or non-UTC timestamps are converted to UTC at ingest; a configured table
    the site lacks fails the build unless declared in `expected_absent_tables`.

    `sample_episodes` (U7, KTD9) restricts the run to `sample_episode_ids(episodes, N)`:
    N eligible ICU episodes drawn deterministically from the episode artifact. A vocabulary
    built from a sample (or from `limit_stays`) records ``provenance.sample: true`` and
    ``sample_size``; training refuses it (`pretrain.build_loaders`, non-dry-run).

    `workers` (default 1; 0 = every usable CPU, at most `MAX_AUTO_WORKERS`) encodes stays
    in a spawn-context process pool over stay-contiguous chunks of at most
    `encode_chunk_events` events (default `ENCODE_CHUNK_EVENTS`; `_parallel_encode`),
    which bounds encode memory; artifacts are byte-identical for any worker count and
    budget. Both are validated before any table is read.

    `report` (default on) writes the aggregate-only tokenization report beside the
    events (`tokenization_report.json`, or `gem_tokenization_report.json` for GEM; see
    src/data/tokenization_report.py): small cells suppressed, identifiers refused."""
    if trajectory not in TRAJECTORIES:
        raise ValueError(f"trajectory must be one of {TRAJECTORIES}, got {trajectory!r}")
    workers = resolve_workers(workers)
    encode_chunk_events = resolve_chunk_events(encode_chunk_events)
    try:
        cfg = with_site_local(cfg, site)
        profile = site_profile(cfg, site)
    except SiteConfigError as exc:
        raise QualificationError(str(exc)) from exc
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
    if extubation_cohort is not None and not gem_mode:
        raise QualificationError("extubation index stays are added to the hospitalization "
                                 "(GEM) trajectory only")
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
    if episodes is None:
        raise QualificationError("a canonical episode/split artifact is required")
    validate_episode_artifact(episodes)
    from clif_validate._vendor.data.site_config import episode_site, require_site_match
    try:
        require_site_match(episode_site(episodes), site, "episode artifact")
    except SiteConfigError as exc:
        raise QualificationError(str(exc)) from exc
    memory_gate()
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
    # The stays a run windows: the eligible episodes and, for GEM, the extubation index
    # stays (item 13). The 24 h import windows [ICU admit, anchor]; GEM and the reference
    # build (whose GEM extension fits the full hospitalization) window [admission,
    # discharge].
    extra_stays, extra_counts = None, None
    if extubation_cohort is not None:
        extra_stays, extra_counts = extubation_index_stays(
            extubation_cohort, episodes, con, base, cfg, site)
        if keep_ids is not None:
            extra_stays = extra_stays.filter(pl.col("hospitalization_id").is_in(keep_ids))
        print(f"  extubation index stays added to the GEM stays: {extra_counts['added']:,} "
              f"(not found: {extra_counts['not_found']:,})")
    eligible_episodes = episodes.filter(pl.col("eligible"))
    stays = eligible_episodes
    if gem_mode:
        stays = pl.concat(
            [eligible_episodes.select([c for c in _GEM_STAY_COLUMNS
                                       if c in eligible_episodes.columns]),
             *([extra_stays] if extra_stays is not None else [])], how="diagonal_relaxed")
    if keep_ids is not None:
        stays = stays.filter(pl.col("hospitalization_id").is_in(keep_ids))
    full_window = gem_mode or (building and gem_cfg is not None)
    window = _register_window(con, stays, *(("admission_dttm", "discharge_dttm")
                                            if full_window else
                                            ("icu_admit_dttm", "anchor_dttm")))
    target_units = _dose_target_units(cfg, None if building else vocab_artifact)
    frames, shadow_frames, dose_stats = [], [], {}
    # CLIF 2.1 / mCIDE conformance gate (per site, after the site's alias maps), before a
    # vocabulary is built or imported: fails closed on an undeclared non-permissible value
    # or a missing required column. Aggregate record -> the report's data-quality section.
    conformance = check_conformance(con, base, cfg, site, keep_ids, min_cell=min_cell)
    harmonization = site_harmonization(cfg, site)
    bp_glob = bp_config(cfg)
    gcs = gcs_rule(cfg)
    if bp_glob is not None:
        harmonization["_bp_table"] = bp_glob["table"]
    # Explicit unit repair (per site, fail closed): unit-less wide-table columns are
    # validated AFTER this site's declared conversions, before anything is read for
    # binning; dose plausibility is checked after conversion, below.
    data_quality = check_column_units(con, base, cfg, site, keep_ids, min_cell=min_cell)
    if conformance:
        data_quality["conformance"] = {
            "mcide_version": conformance["mcide_version"],
            "snapshot_sha256": conformance["snapshot_sha256"],
            "by_table": compliance_table(conformance, 1),
            "columns": conformance["columns"],
            "missing_optional_columns": conformance["missing_optional_columns"]}
    harmonized: dict[str, dict] = {}
    dose_frames: dict[str, pl.DataFrame] = {}
    # Configured columns a site's parquet lacks, per table (report: `missing_columns`).
    missing_columns: dict[str, list[str]] = {}
    expected_absent: list[str] = []
    dst_nulled: dict[str, int] = {}
    for name, spec in cfg["tables"].items():
        if not (base / f"{spec['file']}.parquet").exists():
            if name in profile["expected_absent_tables"]:
                # Declared (sites.<site>.expected_absent_tables): skipped quietly, reported.
                expected_absent.append(name)
                continue
            raise QualificationError(
                f"site {site!r}: configured table {name!r} ({spec['file']}.parquet) is "
                f"missing. If the site does not have it, declare it in "
                f"sites.{site}.expected_absent_tables (configs/data.yaml or the site-local "
                "configs/sites/<site>.local.yaml)")
        absent: list[str] = []
        df, shadows, counts = _read_source(
            con, base, spec, keep_ids, tables=cfg["tables"], target_units=target_units,
            fit_shadows=building, missing=absent,
            column_factors=column_factors(cfg, site, name),
            dose_corrections=dose_corrections(cfg, site, name) if spec.get("dose") else None,
            harmonization=harmonization, name=name, window=window,
        )
        # Timestamp ingest (site_config.to_utc): naive local / non-UTC -> UTC; a time
        # inside a DST gap becomes null and is counted, then dropped.
        nulled = 0
        try:
            df, nulled = to_utc(df, ["dttm"], profile["site_timezone"])
            if shadows is not None:
                shadows, _ = to_utc(shadows, ["dttm"], profile["site_timezone"])
        except SiteConfigError as exc:
            raise QualificationError(f"table {name!r}: {exc}") from exc
        if nulled:
            dst_nulled[name] = nulled
            df = df.filter(pl.col("dttm").is_not_null())
            if shadows is not None:
                shadows = shadows.filter(pl.col("dttm").is_not_null())
        if absent:
            missing_columns[name] = absent
        if counts is not None:
            extra = {k: counts.pop(k) for k in ("exact_duplicates_removed", "weights",
                                                "weights_excluded") if k in counts}
            if extra:
                harmonized.setdefault("dose_tables", {})[name] = extra
            df, below = apply_dose_floors(df, cfg)
            if below:
                harmonized.setdefault("dose_below_floor", {})[name] = below
            dose_stats[name] = counts
            dose_frames[name] = df.select("concept", "value")
        if gcs is not None and name == gcs["table"]:
            df, harmonized["gcs"] = apply_gcs_not_testable(df, gcs)
        if bp_glob is not None and name == bp_glob["table"]:
            df, harmonized["bp_method"] = apply_bp_method(
                df, bp_glob, harmonization["bp_method"], site, min_cell=min_cell)
        df = apply_concept_renames(df, harmonization["concept_renames"].get(name) or {})
        df, derived = derive_concepts(df, name, cfg, con, base, keep_ids,
                                      site_timezone=profile["site_timezone"])
        if derived:
            harmonized.setdefault("derived_concepts", {}).update(derived)
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
        del df, shadows
    data_quality.update(check_dose_plausibility(dose_frames, cfg, site, min_cell=min_cell))
    crrt = crrt_coverage(con, base, cfg, harmonization, "crrt", window)
    if crrt is not None:
        effluent = ((harmonized.get("derived_concepts") or {})
                    .get("crrt_effluent_dose_ml_kg_h") or {})
        crrt["derived_effluent_dose_rows"] = int(effluent.get("emitted", 0))
        data_quality["crrt_coverage"] = crrt
    data_quality["site_profile"] = {
        "site_timezone": profile["site_timezone"],
        "extraction_dttm_declared": profile["extraction_dttm"] is not None,
        "expected_absent_tables": sorted(expected_absent),
        "dst_nulled_rows": dict(sorted(dst_nulled.items())),
    }
    if harmonized:
        data_quality["harmonization"] = harmonized
    dose_frames.clear()
    if stats is not None:
        stats["dose_conversion"] = dose_stats
        stats["data_quality"] = data_quality
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

    static = _read_static(con, base, cfg, keep_ids) if static_tokens else None
    # Guard: verify single-hospital consistency (hospital_id is a CLIF 2.1 column that
    # distinguishes hospitals within a health system; pooling hospitals under one vocab
    # silently merges different clinical workflows and populations).
    _check_single_hospital(con, base)
    # Every DuckDB read is done: release its buffer pool before the polars joins, the sort
    # and the encode (it counted toward the process peak).
    con.close()
    carried = {name for name, spec in cfg["tables"].items() if spec.get("carry_forward")}
    # Stay starts for the static tokens and the carried state: every episode (eligible
    # or not; the window join keeps the eligible) plus, for GEM, the extubation stays.
    starts = (pl.concat([episodes.select("hospitalization_id", "admission_dttm",
                                         "icu_admit_dttm"),
                         extra_stays.select("hospitalization_id", "admission_dttm",
                                            "icu_admit_dttm")], how="vertical_relaxed")
              if extra_stays is not None and len(extra_stays) else episodes)

    def at_stay_start(raw: pl.DataFrame, start_col: str) -> pl.DataFrame:
        """R21: static admission tokens at the stay start (`start_col`: ICU admission
        for the 24 h artifact, hospital admission for GEM), ahead of every other event
        there (`static_rank`); then state carried forward to that start."""
        frame = raw
        if static is not None:
            placed = static.join(
                starts.select(pl.col("hospitalization_id").alias("hosp_id"),
                              pl.col(start_col).alias("dttm")),
                on="hosp_id", how="inner",
            ).filter(pl.col("dttm").is_not_null())
            if len(placed):
                frame = pl.concat(
                    [frame, placed.with_columns(unit=pl.lit(None, pl.String),
                                                source=pl.lit(STATIC_SOURCE))],
                    how="diagonal_relaxed",
                )
        return _carry_forward(frame, starts, carried, start_col)

    treatment_sources = {
        name for name, spec in cfg["tables"].items() if spec.get("input_only")
    } | ({STATIC_SOURCE} if static_tokens else set())
    gem_window = None
    if gem_mode:
        windowed = restrict_to_hospitalization_window(
            at_stay_start(events, "admission_dttm"), episodes, treatment_sources,
            extra_stays=extra_stays)
    else:
        windowed = restrict_to_observation_window(
            at_stay_start(events, "icu_admit_dttm"), episodes, treatment_sources)
        if building and gem_cfg is not None:
            # The reference build's GEM extension fits the full-hospitalization train
            # events: windowed here, from the same pre-window events, before those are
            # released.
            gem_window = restrict_to_hospitalization_window(
                at_stay_start(events, "admission_dttm"), episodes, treatment_sources)
    # The pre-window events are not needed again: release them (and the static frame)
    # before the sort, the fit and the encode.
    del events
    static = None
    events = slim_windowed(windowed)
    del windowed
    if gem_window is not None:
        gem_window = slim_windowed(gem_window)
    # KTD6: full-key sort AFTER the join (polars does not guarantee row order for equal
    # keys through a join or an unmaintained sort). Nulls sort last so a missing value
    # or categorical result has one fixed place; `group_by(maintain_order=True)` below
    # hands `encode` each stay's rows in exactly this order.
    events = _event_sort(events)

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

        def fit_rows(windowed: pl.DataFrame, window_fn) -> pl.DataFrame:
            """The fit partition of `windowed` events, plus — R6 / KTD5 — the
            native-unit fallback rows of weight-converted doses, which are fit
            (windowed by the same `window_fn` and fit-partition-only, like every event)
            but never tokenized."""
            fit = fit_partition(windowed, fit_name)
            if shadow_frames:
                shadows = window_fn(pl.concat(shadow_frames, how="diagonal_relaxed"),
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
        granularity: dict = {}
        literature: dict = {}
        edges, binning_sources = build_segments(
            bin_cfg, fit_events, target_concepts, directions, tables=cfg["tables"],
            quantile_concepts=quantile_concepts, granularity=granularity,
            concept_units=_unit_plan(fit_events, cfg, target_units, site),
            literature_out=literature,
        )
        vocab = build_vocab(fit_events, edges, min_category_stays=min_cell)
        for token in harmonization_tokens(cfg):   # config allowlist, after data tokens
            vocab.setdefault(token, max(vocab.values()) + 1)
        # CLIF 2.1.1 mCIDE allowlist (configuration, not data: no cell floor), after the
        # harmonization tokens; hashed with the vocabulary.
        for token in allowlist_tokens(cfg):
            vocab.setdefault(token, max(vocab.values()) + 1)
        units = reference_units(fit_events, edges, cfg, target_units, site)
        sources = concept_sources(fit_events, treatment_sources)
        if gem_cfg is not None:
            # U8: one vocabulary for both artifacts. Extend it with what only the
            # full-hospitalization train events chart, then the fixed allowlist.
            gem_fit = fit_rows(gem_window, restrict_to_hospitalization_window)
            gem_window = None
            vocab, edges, binning_sources, units, sources = _extend_for_gem(
                gem_cfg, gem_fit, fit_events, vocab, edges, binning_sources, units,
                bin_cfg=bin_cfg, cfg=cfg, directions=directions, target_units=target_units,
                treatment_sources=treatment_sources, quantile_concepts=quantile_concepts,
                min_category_stays=min_cell, granularity=granularity,
                literature=literature, site=site,
            )
            del gem_fit
        del fit_events
        shadow_frames.clear()
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
        if literature:
            # Fragment file hashes + applied edges/closures/source ids: a changed
            # fragment changes the vocabulary manifest.
            hashes["literature_segments"] = json_sha256(literature)
        harmonization_rec = harmonization_record(cfg, site)
        if harmonization_rec:
            # mCIDE snapshot version + hash, GCS / BP / flag / weight / unit rules,
            # the mCIDE allowlist and this reference site's alias maps, duplicates and
            # BP mapping (free-text notes stripped).
            harmonization_rec = json.loads(json.dumps(harmonization_rec))
            hashes["harmonization"] = json_sha256(harmonization_rec)
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
                # Per-concept binning source precedence (csv -> literature -> ...).
                "binning_source_order": list(BINNING_SOURCES),
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
        if granularity:
            # KTD11: decile arm at the clinical arm's bin counts; aggregate counts only.
            vocab_manifest["provenance"]["matched_granularity"] = granularity
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
        if literature:
            vocab_artifact["literature_segments"] = literature
        if harmonization_rec:
            vocab_artifact["harmonization"] = harmonization_rec
    shadow_frames.clear()

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

    binding = shard_binding(vocab_artifact, cfg, site)
    shards, unk_by_concept = _parallel_encode(
        events, vocab, edges, cfg["value_binning"], gem_mode, workers, encode_chunk_events)
    if gem_mode:
        soft_width = _soft_width(cfg["value_binning"])
        gem_stats: dict = {}
        if continuation_header is None:
            continuation_header = model_continuation_header()
        gem = _gem_records(shards, stays, vocab, gem_cfg, max_tokens, soft_width, binding,
                           stats if stats is not None else gem_stats,
                           continuation_header=bool(continuation_header))
        del shards
        out.mkdir(parents=True, exist_ok=True)
        _write_shard(gem, events_path)
        # DATA-CLASSIFICATION: PHI (hosp_id + per-stay sequences + timing), like
        # events.parquet. vocab.json / events.parquet are not touched by this mode.
        print(f"  wrote {events_path} ({gem['hosp_id'].n_unique() if len(gem) else 0:,} "
              f"stays, {len(gem):,} windows)")
        if report:
            gem_record = dict((stats if stats is not None else gem_stats).get("gem") or {})
            if extra_counts is not None:
                gem_record["extubation_index_stays"] = dict(extra_counts)
            _write_run_report(out / GEM_REPORT_FILE, trajectory, events, gem,
                              vocab_artifact, site, availability, cfg, dose_stats,
                              mismatched_units, unk_by_concept, keep_ids, episodes,
                              max_tokens, gem_record,
                              min_cell=min_cell, missing_columns=missing_columns,
                              data_quality=data_quality)
        return vocab, edges
    if len(shards):
        # KTD7: every shard row is bound to the tokenizer version, vocabulary and
        # segments it was encoded with, and to the site's declarations; ModelDataset
        # refuses a row without the vocabulary binding.
        shards = shards.with_columns(binding_expr(binding))
    out.mkdir(parents=True, exist_ok=True)
    _write_shard(shards, events_path)
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
                          min_cell=min_cell, missing_columns=missing_columns,
                          data_quality=data_quality)
    return vocab, edges


def _write_run_report(path: Path, trajectory: str, events: pl.DataFrame,
                      records: pl.DataFrame, vocab_artifact: dict, site: str,
                      availability: dict, cfg: dict, dose_stats: dict,
                      mismatched_units: list[str], unk_by_concept: dict,
                      keep_ids: list | None, episodes: pl.DataFrame, context: int,
                      gem: dict | None, *, min_cell: int,
                      missing_columns: dict[str, list[str]],
                      data_quality: dict | None = None) -> None:
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
        matched_granularity=provenance.get("matched_granularity"),
        data_quality={**(data_quality or {}),
                      "dose_corrections": {t: c.get("corrected", 0) for t, c in
                                           dose_stats.items() if "corrected" in c},
                      "dose_quarantined": {t: c.get("quarantined", 0) for t, c in
                                           dose_stats.items() if "quarantined" in c}},
        literature=vocab_artifact.get("literature_segments"),
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
                    help=f"processes for per-stay encoding (0 = every usable CPU, at most "
                         f"{MAX_AUTO_WORKERS}); the output is byte-identical for any value")
    ap.add_argument("--extubation-cohort", default=None, metavar="PATH",
                    help="hospitalization trajectory only: the site's extubation cohort; "
                         "the index stay of every eligible extubation that is not an "
                         "eligible episode is added to the GEM stays (partition inherited "
                         "from the patient)")
    ap.add_argument("--encode-chunk-events", type=int, default=None, metavar="N",
                    help=f"events per encode chunk (default {ENCODE_CHUNK_EVENTS:,}); bounds "
                         "encode memory, output byte-identical for any value")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the aggregate-only tokenization report")
    return ap


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)

    # The site's git-ignored local declarations (configs/sites/<site>.local.yaml).
    cfg = with_site_local(yaml.safe_load(Path(args.config).read_text()), args.site)
    validate_table_availability(cfg.get("tables"))
    policy = yaml.safe_load((ROOT / cfg["artifact_policy"]).read_text())
    blob = None
    if args.vocab:
        blob = json.loads(Path(args.vocab).read_text())
        validate_vocabulary_artifact(blob, cfg, policy)
    elif not args.build_vocab:
        raise SystemExit("pass --build-vocab (first site) or --vocab PATH (later sites)")

    if args.dry_run:
        return dry_run(cfg, args.site, Path(args.indir), blob)

    episodes = pl.read_parquet(args.episodes)
    cohort = None
    if args.extubation_cohort:
        if args.trajectory != "hospitalization":
            raise SystemExit("--extubation-cohort is read by --trajectory hospitalization only")
        cohort = pl.read_parquet(args.extubation_cohort)
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
        extubation_cohort=cohort,
    )


def dry_run(cfg: dict, site: str, base: Path, blob: dict | None) -> int:
    """`--dry-run`: read every configured table through the site's declarations (column
    aliases, category maps, unit conversions; no window) and print, per table, aggregate
    events and concepts, the timestamps inside a DST gap, and whether an absent table is
    declared absent. Returns 1 (and prints the fix) when a table is missing undeclared."""
    profile = site_profile(cfg, site)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")  # same session-tz pin as tokenize_site
    target_units = _dose_target_units(cfg, blob)
    harmonization = site_harmonization(cfg, site)
    bp_glob = bp_config(cfg)
    if bp_glob is not None:
        harmonization["_bp_table"] = bp_glob["table"]
    unexpected = []
    for name, spec in cfg["tables"].items():
        if not (base / f"{spec['file']}.parquet").exists():
            expected = name in profile["expected_absent_tables"]
            print(f"{name}: absent ({'declared in expected_absent_tables' if expected else 'NOT DECLARED: the build fails'})")
            if not expected:
                unexpected.append(name)
            continue
        df, _, _ = _read_source(
            con, base, spec, None, tables=cfg["tables"], target_units=target_units,
            column_factors=column_factors(cfg, site, name),
            dose_corrections=dose_corrections(cfg, site, name) if spec.get("dose") else None,
            harmonization=harmonization, name=name)
        df, nulled = to_utc(df, ["dttm"], profile["site_timezone"])
        print(f"{name}: {len(df):,} events, concepts={df['concept'].n_unique() if len(df) else 0}"
              + (f", {nulled:,} timestamps inside a DST gap (nulled)" if nulled else ""))
        del df
    if unexpected:
        print(f"MISSING undeclared table(s): {', '.join(unexpected)}; declare them in "
              f"sites.{site}.expected_absent_tables if the site does not have them")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
