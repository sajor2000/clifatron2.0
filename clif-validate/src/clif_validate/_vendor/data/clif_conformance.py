"""CLIF 2.1 / mCIDE conformance gate and per-site harmonization (hard rule #2).

One frozen vocabulary is applied identically to every site, so a category value spelled
differently at two sites, or a site-only value, silently becomes a different token or
`<unk>`. This module makes CLIF 2.1 conformance an enforced, tested gate:

- `load_snapshot`: the public CLIF v2.1.1 mCIDE permissible-value snapshot
  (configs/clif_mcide_2.1.1: category CSVs, the 2.1 DDL, clifpy's ECMO/MCS schema),
  hashed file by file.
- `site_harmonization`: a site's declarations in `configs/data.yaml site_harmonization`:
  `category_map` (raw value -> mCIDE value, optionally conditional on other columns of
  the same row), `exact_duplicates` (one administration charted under two categories),
  `column_aliases` (official 2.1 column name <-> the site's column), `declared_exceptions`
  (documented non-permissible values kept as their own category) and `bp_method`.
- `check_conformance`: per site, AFTER the alias maps, every table/column the pipeline
  reads exists under its 2.1 name (or the pinned variant / a declared alias), and every
  category value the pipeline turns into tokens is permissible, aliased to a permissible
  value, or a declared exception. Anything else fails closed with an aggregate message
  (table, column, value, count; counts under the minimum cell size shown as ``<10``).
- `harmonization_record`: what the vocabulary hashes (`hashes["harmonization"]`): the
  snapshot version and hash, the global rules, and the reference site's declarations.

Aggregate-only: values are category labels and counts, never rows or identifiers.

    uv run python -m src.data.clif_conformance --data ~/Data/clif-source --site mimic
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from clif_validate._vendor.data.cohort import QualificationError

ROOT = Path(__file__).parents[2]
STATUSES = ("compliant", "aliased", "declared_exception", "failing")
NULL_KEY = "<null>"
BP_METHODS = ("arterial", "noninvasive_auto", "noninvasive_manual", "unknown")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DDL_TABLE = re.compile(r"^\s*CREATE TABLE\s+(\w+)", re.I)
_DDL_COLUMN = re.compile(
    r"^\s+(\w+)\s+(VARCHAR|DATETIME|DATE|FLOAT|DOUBLE|INT|INTEGER|BIGINT|BOOLEAN|DECIMAL|"
    r"TEXT|TIMESTAMP)\b", re.I)
# Static (non-event) entities the checks may name besides the configured tables.
STATIC_ENTITIES = {"hospitalization": "hospitalization_file", "patient": "patient_file"}


# ------------------------------------------------------------------ helpers

def norm_value(value: object) -> str:
    """Spelling key of a category value: lowercase, whitespace/underscore runs -> ``_``.
    Two values with the same key yield the same token (`categorical_token`,
    `normalize_concept`, `units.normalize_name` for the medication names)."""
    return re.sub(r"[\s_]+", "_", str(value).strip().lower()).strip("_")


def norm_name(value: object) -> str:
    """`units.normalize_name`: lowercase, non-alphanumeric runs -> ``_``."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def name_sql(expr: str) -> str:
    """SQL twin of `norm_name`."""
    return (f"regexp_replace(regexp_replace(lower(trim(CAST({expr} AS VARCHAR))), "
            f"'[^a-z0-9]+', '_', 'g'), '^_+|_+$', '', 'g')")


def _quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _ident(value: object, what: str) -> str:
    if not isinstance(value, str) or not _IDENT.match(value):
        raise QualificationError(f"{what} must be a plain column identifier, got {value!r}")
    return value


def id_filter(keep_ids: list | None, column: str = "hospitalization_id") -> tuple[str, list]:
    if keep_ids is None:
        return "", []
    if not keep_ids:
        return "AND 1 = 0", []
    return (f"AND CAST({column} AS VARCHAR) IN ({', '.join('?' for _ in keep_ids)})",
            [str(i) for i in keep_ids])


def json_sha256(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def _cell(n: int, min_cell: int) -> int | str:
    return f"<{min_cell}" if 0 < n < min_cell else int(n)


# ------------------------------------------------------------------ the snapshot

def _read_csv_column(path: Path, column: str) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [line for line in text.splitlines() if line.strip()]
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    reader.fieldnames = [f.strip().lstrip("﻿") for f in reader.fieldnames or []]
    if column not in reader.fieldnames:
        raise QualificationError(f"mCIDE snapshot {path.name} has no column {column!r}")
    values = [(row.get(column) or "").strip() for row in reader]
    return sorted({v for v in values if v})


def _yaml_column(path: Path, column: str, key: str = "permissible_values") -> list[str]:
    doc = yaml.safe_load(path.read_text())
    for col in doc.get("columns") or ():
        if col.get("name") == column:
            return sorted(str(v) for v in col.get(key) or ())
    raise QualificationError(f"schema {path.name} has no column {column!r}")


def _ddl_tables(path: Path) -> dict[str, list[str]]:
    tables: dict[str, list[str]] = {}
    current = None
    for line in path.read_text().splitlines():
        match = _DDL_TABLE.match(line)
        if match:
            current = match.group(1)
            tables[current] = []
            continue
        match = _DDL_COLUMN.match(line)
        if match and current:
            tables[current].append(match.group(1))
    return tables


def snapshot_sha256(path: Path) -> str:
    """SHA-256 over every file of the snapshot (relative path + bytes, sorted)."""
    digest = hashlib.sha256()
    for fp in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(fp.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(fp.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_snapshot(source: str | Path) -> dict:
    """``{version, tag, commit, sha256, permissible: {clif_table.column: [values]},
    tables: {clif_table: [columns]}, variants: {clif_table: {name: [columns]}}}``."""
    path = Path(source)
    path = path if path.is_absolute() else ROOT / path
    manifest_fp = path / "manifest.yaml"
    if not manifest_fp.is_file():
        raise QualificationError(f"mCIDE snapshot manifest not found: {manifest_fp}")
    manifest = yaml.safe_load(manifest_fp.read_text())
    permissible: dict[str, list[str]] = {}
    for key, spec in (manifest.get("permissible") or {}).items():
        if "csv" in spec:
            permissible[key] = _read_csv_column(path / spec["csv"], spec["column"])
        elif "yaml" in spec:
            permissible[key] = _yaml_column(path / spec["yaml"], spec["column"])
        else:
            raise QualificationError(f"snapshot permissible entry {key!r} needs csv or yaml")
    variants: dict[str, dict[str, list[str]]] = {}
    for table, named in (manifest.get("variants") or {}).items():
        for name, spec in named.items():
            doc = yaml.safe_load((path / spec["schema"]).read_text())
            variants.setdefault(table, {})[name] = [c["name"] for c in doc.get("columns") or ()]
    units: dict[str, str] = {}
    for spec in (manifest.get("units") or {}).values():
        if isinstance(spec, dict) and "csv" in spec:
            text = (path / spec["csv"]).read_text(encoding="utf-8-sig")
            for row in csv.DictReader(io.StringIO(text)):
                key = (row.get(spec["key"]) or "").strip()
                if key:
                    units[key] = (row.get(spec["unit"]) or "").strip()
        else:
            units.update({str(k): str(v) for k, v in (spec or {}).items()})
    return {"version": str(manifest.get("version")), "tag": manifest.get("tag"),
            "commit": manifest.get("commit"), "sha256": snapshot_sha256(path),
            "permissible": permissible, "tables": _ddl_tables(path / manifest["ddl"]),
            "variants": variants, "units": units}


# ------------------------------------------------------------------ declarations

def _table_entity(cfg: Mapping, name: str) -> tuple[str, dict]:
    """(file stem, table spec) of a configured table or a static entity."""
    tables = cfg.get("tables") or {}
    if name in tables:
        return tables[name]["file"], tables[name]
    if name in STATIC_ENTITIES:
        static = cfg.get("static_source") or {}
        return static.get(STATIC_ENTITIES[name], f"clif_{name}"), {}
    raise QualificationError(f"{name!r} is neither a configured table nor a static entity")


def clif_table(cfg: Mapping, name: str) -> str:
    """The CLIF table name of a configured table (its file stem without ``clif_``)."""
    stem, _ = _table_entity(cfg, name)
    return stem[5:] if stem.startswith("clif_") else stem


def _column_key(key: object, cfg: Mapping, where: str) -> tuple[str, str]:
    parts = str(key).split(".")
    if len(parts) != 2:
        raise QualificationError(f"{where}: key {key!r} must be `table.column`")
    _table_entity(cfg, parts[0])
    return parts[0], _ident(parts[1], f"{where} column")


def _rules(value: object, where: str) -> list[dict]:
    """One raw value's mapping: a target string, or a list of ``{when: {col: value},
    to}`` rules tried in order (a rule without `when` is the fallback, last)."""
    if isinstance(value, str):
        return [{"when": {}, "to": value}]
    if not isinstance(value, list) or not value:
        raise QualificationError(f"{where} must be a target value or a list of rules")
    out = []
    for i, rule in enumerate(value):
        if not isinstance(rule, dict) or not isinstance(rule.get("to"), str):
            raise QualificationError(f"{where}[{i}] needs `to`")
        when = rule.get("when") or {}
        if not isinstance(when, dict):
            raise QualificationError(f"{where}[{i}].when must map column -> value")
        if not when and i != len(value) - 1:
            raise QualificationError(f"{where}[{i}]: a rule without `when` must be last")
        out.append({"when": {_ident(c, f"{where}.when"): str(v) for c, v in when.items()},
                    "to": rule["to"]})
    return out


def site_harmonization(cfg: Mapping, site: str) -> dict:
    """The validated declarations of ONE site (empty sections when undeclared)."""
    every = cfg.get("site_harmonization") or {}
    if not isinstance(every, dict):
        raise QualificationError("site_harmonization must map site -> declarations")
    raw = every.get(site) or {}
    where = f"site_harmonization.{site}"
    out: dict[str, Any] = {"declared": site in every, "category_map": {},
                           "exact_duplicates": [], "column_aliases": {},
                           "declared_exceptions": {}, "bp_method": None,
                           "concept_renames": {}}
    for key, mapping in sorted((raw.get("category_map") or {}).items()):
        _column_key(key, cfg, f"{where}.category_map")
        if not isinstance(mapping, dict):
            raise QualificationError(f"{where}.category_map.{key} must map raw value -> target")
        out["category_map"][key] = {
            (NULL_KEY if source is None else str(source)): _rules(target,
                                                                  f"{where}.category_map.{key}")
            for source, target in sorted(mapping.items(), key=lambda kv: str(kv[0]))}
    for i, rule in enumerate(raw.get("exact_duplicates") or ()):
        w = f"{where}.exact_duplicates[{i}]"
        table = rule.get("table") if isinstance(rule, dict) else None
        spec = (cfg.get("tables") or {}).get(table) or {}
        if not spec.get("dose"):
            raise QualificationError(f"{w}: table must be a dose table")
        drop = [norm_name(v) for v in rule.get("drop") or ()]
        keep = norm_name(rule.get("keep") or "")
        match = [_ident(c, f"{w}.match") for c in rule.get("match") or ()]
        if not drop or not keep or keep in drop or not match:
            raise QualificationError(f"{w} needs `drop`, a different `keep` and `match` columns")
        out["exact_duplicates"].append({"table": table, "drop": drop, "keep": keep,
                                        "match": match})
    for table, aliases in sorted((raw.get("column_aliases") or {}).items()):
        _table_entity(cfg, table)
        out["column_aliases"][table] = {
            _ident(k, f"{where}.column_aliases.{table}"):
                _ident(v, f"{where}.column_aliases.{table}")
            for k, v in sorted((aliases or {}).items())}
    for key, values in sorted((raw.get("declared_exceptions") or {}).items()):
        _column_key(key, cfg, f"{where}.declared_exceptions")
        if not isinstance(values, dict) or any(not str(n or "").strip() for n in values.values()):
            raise QualificationError(
                f"{where}.declared_exceptions.{key} must map value -> note (the reason)")
        out["declared_exceptions"][key] = {str(v): str(n).strip()
                                           for v, n in sorted(values.items())}
    for table, renames in sorted((raw.get("concept_renames") or {}).items()):
        _table_entity(cfg, table)
        out["concept_renames"][table] = {
            _ident(k, f"{where}.concept_renames.{table}"):
                _ident(v, f"{where}.concept_renames.{table}")
            for k, v in sorted((renames or {}).items())}
    bp = raw.get("bp_method")
    if bp is not None:
        out["bp_method"] = _site_bp(bp, f"{where}.bp_method", cfg)
    return out


def bp_config(cfg: Mapping) -> dict | None:
    """The global `bp_method` block (validated) or None."""
    block = cfg.get("bp_method")
    if block is None:
        return None
    methods = list(block.get("methods") or BP_METHODS)
    if not set(methods) <= set(BP_METHODS) or "unknown" not in methods:
        raise QualificationError(f"bp_method.methods must be a subset of {BP_METHODS} "
                                 "including unknown")
    concepts = [str(c) for c in block.get("concepts") or ()]
    if not concepts or block.get("table") not in (cfg.get("tables") or {}):
        raise QualificationError("bp_method needs a configured `table` and `concepts`")
    return {"table": block["table"], "concepts": concepts, "methods": methods,
            "token_concept": _ident(block.get("token_concept", "bp_method"),
                                    "bp_method.token_concept"),
            "require_site_declaration": bool(block.get("require_site_declaration", True))}


def _site_bp(bp: object, where: str, cfg: Mapping) -> dict:
    glob = bp_config(cfg)
    if glob is None:
        raise QualificationError(f"{where} declared but the config has no bp_method block")
    if not isinstance(bp, dict):
        raise QualificationError(f"{where} must be a mapping")
    source = _ident(bp.get("source_column"), f"{where}.source_column")
    patterns = []
    for i, entry in enumerate(bp.get("patterns") or ()):
        method = entry.get("method") if isinstance(entry, dict) else None
        if method not in glob["methods"]:
            raise QualificationError(f"{where}.patterns[{i}].method must be one of "
                                     f"{glob['methods']}")
        try:
            re.compile(entry["pattern"])
        except (KeyError, re.error, TypeError) as exc:
            raise QualificationError(f"{where}.patterns[{i}].pattern is not a regex") from exc
        patterns.append({"pattern": entry["pattern"], "method": method})
    null = bp.get("null")
    if null is not None and null not in glob["methods"]:
        raise QualificationError(f"{where}.null must be one of {glob['methods']}")
    if not patterns:
        raise QualificationError(f"{where} needs patterns")
    return {"source_column": source, "patterns": patterns, "null": null}


def bp_method_of(name: object, decl: Mapping) -> str | None:
    """The method a BP source name maps to (first matching pattern, case-insensitive);
    a null name -> the declared `null` method; None when unmapped."""
    if name is None:
        return decl.get("null")
    for entry in decl["patterns"]:
        if re.search(entry["pattern"], str(name), re.I):
            return entry["method"]
    return None


# ------------------------------------------------------------------ SQL

def parquet_source(fp: Path, aliases: Mapping[str, str] | None,
                   present: set[str] | None = None) -> str:
    """FROM expression reading `fp` with the site's column aliases applied (the site's
    column is read under the config's name). `present`: the parquet's columns (lowercase),
    to skip an alias whose site column the file lacks."""
    base = f"read_parquet('{fp}')"
    pairs = [(cfg_name, site_name) for cfg_name, site_name in (aliases or {}).items()
             if present is None or site_name.lower() in present]
    if not pairs:
        return base
    exclude = ", ".join(site for _, site in pairs)
    renamed = ", ".join(f"{site} AS {name}" for name, site in pairs)
    return f"(SELECT * EXCLUDE ({exclude}), {renamed} FROM {base})"


def _when_sql(when: Mapping[str, str]) -> str:
    return " AND ".join(f"{name_sql(col)} = {_quote(norm_name(val))}"
                       for col, val in when.items()) or "TRUE"


def mapped_sql(column: str, rules: Mapping[str, list] | None) -> str:
    """VARCHAR SQL of `column` after the site's category map (raw values compared by
    `norm_name`; `<null>` matches a missing value; a value no rule matches is kept)."""
    raw = f"CAST({column} AS VARCHAR)"
    if not rules:
        return raw
    branches = []
    for source, value_rules in rules.items():
        match = (f"{column} IS NULL" if source == NULL_KEY
                 else f"{name_sql(column)} = {_quote(norm_name(source))}")
        for rule in value_rules:
            branches.append(f"WHEN {match} AND {_when_sql(rule['when'])} "
                            f"THEN {_quote(rule['to'])}")
    return f"(CASE {' '.join(branches)} ELSE {raw} END)"


def rule_columns(rules: Mapping[str, list] | None) -> set[str]:
    return {c for value_rules in (rules or {}).values() for r in value_rules for c in r["when"]}


def flag_sql(column: str) -> str:
    """A CLIF 0/1 flag (stored INT 0/1 or Boolean) as one spelling, ``1`` / ``0``."""
    v = f"lower(trim(CAST({column} AS VARCHAR)))"
    return (f"(CASE WHEN {v} IN ('1', '1.0', 'true') THEN '1' "
            f"WHEN {v} IN ('0', '0.0', 'false') THEN '0' ELSE NULL END)")


# ------------------------------------------------------------------ global rules

def _conformance_block(cfg: Mapping) -> dict | None:
    block = cfg.get("clif_conformance")
    if block is None:
        return None
    if not isinstance(block, dict) or not block.get("snapshot"):
        raise QualificationError("clif_conformance needs a `snapshot` directory")
    return block


def global_record(cfg: Mapping) -> dict:
    """Every SITE-INDEPENDENT rule that shapes tokens (hashed into the vocabulary)."""
    block = _conformance_block(cfg)
    record: dict[str, Any] = {}
    if block is not None:
        snap = load_snapshot(block["snapshot"])
        record["mcide"] = {"version": snap["version"], "tag": snap["tag"],
                           "commit": snap["commit"], "snapshot_sha256": snap["sha256"],
                           "table_variants": dict(sorted((block.get("table_variants")
                                                          or {}).items())),
                           "checked": dict(sorted((block.get("checked") or {}).items())),
                           "extensions": block.get("extensions") or {}}
    if cfg.get("gcs_not_testable") is not None:
        record["gcs_not_testable"] = cfg["gcs_not_testable"]
    bp = bp_config(cfg)
    if bp is not None:
        record["bp_method"] = bp
    flags = {name: list(spec.get("flag_cols") or ())
             for name, spec in sorted((cfg.get("tables") or {}).items())
             if spec.get("flag_cols")}
    if flags:
        record["flag_cols"] = flags
    weights = {name: spec["dose"]["weight_source"].get("plausible_kg")
               for name, spec in sorted((cfg.get("tables") or {}).items())
               if (spec.get("dose") or {}).get("weight_source", {}).get("plausible_kg")}
    if weights:
        record["weight_plausible_kg"] = weights
    floors = (cfg.get("dose_plausibility") or {}).get("floors")
    if floors:
        record["dose_floors"] = dict(sorted(floors.items()))
    equivalent = (cfg.get("unit_normalization") or {}).get("equivalent_units")
    if equivalent:
        record["equivalent_units"] = equivalent
    for key in ("csv_coverage", "literature_coverage"):
        coverage = (cfg.get("value_binning") or {}).get(key)
        if coverage:
            record[key] = coverage
    if cfg.get("derived_concepts"):
        record["derived_concepts"] = cfg["derived_concepts"]
    return record


def harmonization_record(cfg: Mapping, site: str) -> dict:
    """The vocabulary's hashed harmonization record: the global rules and the reference
    site's own declarations. Empty when the config declares none of them."""
    glob = global_record(cfg)
    decl = site_harmonization(cfg, site)
    if not glob and not decl["declared"]:
        return {}
    site_part = {k: v for k, v in decl.items() if k != "declared"}
    return {"global": glob, "reference_site": site, "site": site_part}


# ------------------------------------------------------------------ the gate

def _columns_read(spec: Mapping) -> tuple[list[str], list[str]]:
    """(required, optional) columns a table spec reads."""
    required, optional = [], []
    if spec.get("key", "hospitalization") == "patient":
        required.append(spec.get("patient_id_col", "patient_id"))
    else:
        required.append("hospitalization_id")
    for key in ("availability_col", "concept_col", "value_col", "unit_col"):
        if spec.get(key):
            required.append(spec[key])
    if (spec.get("dose") or {}).get("action_col"):
        required.append(spec["dose"]["action_col"])
    strict = spec.get("on_missing_column") == "error"
    for key in ("value_cols", "categorical_value_cols"):
        (required if strict else optional).extend(spec.get(key) or ())
    for key in ("categorical_value_col", "concept_qualifier_col"):
        if spec.get(key):
            (required if strict else optional).append(spec[key])
    return required, optional


def _spec_columns(snapshot: Mapping, table: str, variant: str | None) -> set[str] | None:
    if variant:
        cols = (snapshot["variants"].get(table) or {}).get(variant)
        if cols is None:
            raise QualificationError(f"clif_conformance.table_variants: {table} has no "
                                     f"variant {variant!r} in the snapshot")
        return set(cols)
    cols = snapshot["tables"].get(table)
    return None if cols is None else set(cols)


def check_config_columns(cfg: Mapping, snapshot: Mapping) -> list[str]:
    """Columns the config reads that are not CLIF 2.1 columns of their table (or of its
    pinned variant). Returns the problems (empty = conformant)."""
    block = _conformance_block(cfg) or {}
    variants = block.get("table_variants") or {}
    problems = []
    for name, spec in (cfg.get("tables") or {}).items():
        table = clif_table(cfg, name)
        allowed = _spec_columns(snapshot, table, variants.get(table))
        if allowed is None:
            problems.append(f"{name}: {table!r} is not a CLIF 2.1 table")
            continue
        required, optional = _columns_read(spec)
        for col in [*required, *optional]:
            if col not in allowed:
                problems.append(f"{name}: column {col!r} is not a CLIF 2.1 "
                                f"{table} column{' (' + variants[table] + ')' if table in variants else ''}")
    return problems


def check_conformance(con, base: Path, cfg: Mapping, site: str,
                      keep_ids: list | None = None, *, min_cell: int = 10,
                      raise_on_failure: bool | None = None) -> dict:
    """Run the gate for one site; returns the aggregate record (per checked column the
    rows per status, and every non-compliant value with its status and count).

    Fails closed (QualificationError) on a configured column that is not a 2.1 column, a
    required column the site lacks, or a value that is neither permissible, aliased to a
    permissible value, nor declared, unless `clif_conformance.on_violation: report`."""
    block = _conformance_block(cfg)
    if block is None:
        return {}
    snapshot = load_snapshot(block["snapshot"])
    decl = site_harmonization(cfg, site)
    fail = (block.get("on_violation", "error") == "error" if raise_on_failure is None
            else raise_on_failure)
    failures: list[str] = list(check_config_columns(cfg, snapshot))
    record: dict[str, Any] = {
        "site": site, "mcide_version": snapshot["version"], "mcide_tag": snapshot["tag"],
        "snapshot_sha256": snapshot["sha256"], "site_declared": decl["declared"],
        "tables": {}, "columns": {}, "missing_optional_columns": {}}
    # 1. table / column presence (after the site's column aliases).
    present_by_table: dict[str, set[str]] = {}
    raw_by_table: dict[str, set[str]] = {}
    for name in [*(cfg.get("tables") or {}), *STATIC_ENTITIES]:
        stem, spec = _table_entity(cfg, name)
        fp = base / f"{stem}.parquet"
        if not fp.exists():
            record["tables"][name] = "absent"
            continue
        raw_cols = {r[0].lower() for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{fp}')").fetchall()}
        aliases = decl["column_aliases"].get(name) or {}
        present = (raw_cols - {s.lower() for s in aliases.values()}) | {
            c.lower() for c, s in aliases.items() if s.lower() in raw_cols}
        present_by_table[name] = present
        raw_by_table[name] = raw_cols
        record["tables"][name] = "present"
        if not spec:
            continue
        required, optional = _columns_read(spec)
        missing = [c for c in required if c.lower() not in present]
        if missing:
            failures.append(f"{name} ({stem}): required column(s) missing: {', '.join(missing)}")
        absent = [c for c in optional if c.lower() not in present]
        if absent:
            record["missing_optional_columns"][name] = absent
    # 2. category values.
    extensions = block.get("extensions") or {}
    checked = block.get("checked") or {}
    for key, spec_key in sorted(checked.items()):
        name, column = _column_key(key, cfg, "clif_conformance.checked")
        if name not in present_by_table:
            continue
        if column.lower() not in present_by_table[name]:
            record["columns"][key] = {"status": "column_absent"}
            continue
        permitted = {norm_value(v) for v in snapshot["permissible"].get(spec_key, ())}
        if spec_key not in snapshot["permissible"]:
            raise QualificationError(f"clif_conformance.checked.{key}: snapshot has no "
                                     f"permissible list {spec_key!r}")
        for extra in (extensions.get(key) or {}).get("also_permit") or ():
            permitted |= {norm_value(v) for v in snapshot["permissible"][extra]}
        rules = decl["category_map"].get(key)
        missing_rule_cols = sorted(c for c in rule_columns(rules)
                                   if c.lower() not in present_by_table[name])
        if missing_rule_cols:
            failures.append(f"{key}: category_map conditions read missing column(s) "
                            f"{', '.join(missing_rule_cols)}")
            continue
        stem, spec = _table_entity(cfg, name)
        aliases = decl["column_aliases"].get(name) or {}
        source = parquet_source(base / f"{stem}.parquet", aliases, raw_by_table[name])
        has_ids = "hospitalization_id" in present_by_table[name]
        id_sql, params = id_filter(keep_ids) if has_ids else ("", [])
        rows = con.execute(
            f"SELECT CAST({column} AS VARCHAR) AS raw, {mapped_sql(column, rules)} AS mapped, "
            f"count(*) FROM {source} WHERE TRUE {id_sql} GROUP BY ALL", params).fetchall()
        exceptions = {norm_value(v): v for v in decl["declared_exceptions"].get(key, {})}
        counts = dict.fromkeys(STATUSES, 0)
        counts["null"] = 0
        values: dict[str, dict] = {}
        for raw, mapped, n in rows:
            if mapped is None:
                counts["null"] += n
                if raw is not None:   # mapped to nothing is impossible; kept for safety
                    counts["failing"] += n
                continue
            if norm_value(mapped) in permitted:
                status = "compliant" if raw is not None and norm_value(raw) == norm_value(
                    mapped) else "aliased"
            elif norm_value(mapped) in exceptions:
                status = "declared_exception"
            else:
                status = "failing"
            counts[status] += n
            if status != "compliant":
                label = mapped if status != "aliased" else f"{raw} -> {mapped}"
                entry = values.setdefault(label, {"status": status, "rows": 0})
                entry["rows"] += n
        for label, entry in values.items():
            if entry["status"] == "failing":
                failures.append(f"{key}: value {label!r} ({_cell(entry['rows'], min_cell)} "
                                f"rows) is not a permissible mCIDE {spec_key} value; alias it "
                                f"(site_harmonization.{site}.category_map.{key}) or declare it "
                                f"(site_harmonization.{site}.declared_exceptions.{key})")
        record["columns"][key] = {
            "permissible": spec_key, **counts,
            "values": {label: {"status": e["status"], "rows": e["rows"]}
                       for label, e in sorted(values.items())}}
    # 3. 0/1 flag domains.
    for key, domain in sorted((block.get("flag_domains") or {}).items()):
        name, column = _column_key(key, cfg, "clif_conformance.flag_domains")
        if column.lower() not in present_by_table.get(name, set()):
            continue
        stem, _ = _table_entity(cfg, name)
        id_sql, params = id_filter(keep_ids)
        aliases = decl["column_aliases"].get(name) or {}
        source = parquet_source(base / f"{stem}.parquet", aliases, raw_by_table[name])
        rows = con.execute(
            f"SELECT {flag_sql(column)} AS v, CAST({column} AS VARCHAR) AS raw, count(*) "
            f"FROM {source} WHERE TRUE {id_sql} GROUP BY ALL", params).fetchall()
        allowed = {str(int(d)) for d in domain}
        entry = {"permissible": sorted(allowed), "compliant": 0, "aliased": 0, "failing": 0,
                 "null": 0, "values": {}}
        for v, raw, n in rows:
            if raw is None:
                entry["null"] += n
            elif v in allowed:
                status = "compliant" if raw == v else "aliased"
                entry[status] += n
                if status == "aliased":
                    label = f"{raw} -> {v}"
                    entry["values"][label] = {"status": status,
                                              "rows": entry["values"].get(label, {}).get(
                                                  "rows", 0) + n}
            else:
                entry["failing"] += n
                failures.append(f"{key}: value {raw!r} ({_cell(n, min_cell)} rows) is not "
                                f"a 0/1 flag")
        record["columns"][key] = entry
    record["failures"] = len(failures)
    if failures and fail:
        raise QualificationError(f"site {site!r} is not CLIF 2.1 / mCIDE {snapshot['version']} "
                                 "conformant: " + "; ".join(failures))
    record["failure_messages"] = failures
    return record


def compliance_table(record: Mapping, min_cell: int = 10) -> dict:
    """Per table: rows compliant / aliased / declared_exception / failing (suppressed)."""
    table: dict[str, dict] = {}
    for key, entry in (record.get("columns") or {}).items():
        name = key.split(".", 1)[0]
        row = table.setdefault(name, dict.fromkeys(STATUSES, 0))
        for status in STATUSES:
            row[status] += int(entry.get(status, 0) or 0)
    return {name: {k: _cell(v, min_cell) for k, v in row.items()}
            for name, row in sorted(table.items())}


def suppress(record: Mapping, min_cell: int = 10) -> dict:
    """The record with every row count below `min_cell` written as ``<min_cell``."""
    def walk(node: Any, key: str | None = None) -> Any:
        if isinstance(node, dict):
            return {k: walk(v, k) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        if isinstance(node, int) and not isinstance(node, bool) and key not in ("failures",):
            return _cell(node, min_cell)
        return node
    return walk(dict(record))


def main(argv: list[str] | None = None) -> int:
    import duckdb

    ap = argparse.ArgumentParser(description="CLIF 2.1 / mCIDE conformance (aggregate only)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--report-only", action="store_true",
                    help="print failures instead of raising")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    policy = yaml.safe_load((ROOT / cfg["artifact_policy"]).read_text())
    min_cell = int(policy["classes"]["aggregate_no_phi"]["minimum_cell_size"])
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    record = check_conformance(con, Path(args.data).expanduser(), cfg, args.site,
                               min_cell=min_cell,
                               raise_on_failure=False if args.report_only else None)
    out = suppress({k: v for k, v in record.items() if k != "failure_messages"}, min_cell)
    out["compliance_table"] = compliance_table(record, min_cell)
    out["failure_messages"] = record.get("failure_messages", [])
    print(json.dumps(out, indent=2))
    return 1 if record.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
