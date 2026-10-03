"""Aggregate-only tokenization report and the real-data gate (U7; R12, R16; KTD9).

`tokenize_site` writes one report per run beside its events:
``tokenization_report.json`` for the 24 h artifact and ``gem_tokenization_report.json``
for the full-hospitalization GEM artifact. A report holds ONLY aggregates:

- events and stays per source table, with ``low_coverage`` (fewer than
  `LOW_COVERAGE_STAYS` stays: the source is present but unverified);
- token kinds (binned numeric, fused categorical, presence, numeric-but-bare, skipped
  missing numeric) and the numeric concepts emitted bare, split by fit / non-fit
  partition;
- vocabulary size, binning sources per concept count, and the single-bin concepts;
- ``binning``: per binning source (csv, literature, ordinal, quantile, single) the
  concept count and this run's binned events and event share; ``literature``: the
  literature-grounded concepts and those whose sources or edges are unverified;
- ``data_quality``: the site's declared unit conversions and, per unit-less column
  (`column_units`), values outside the plausible range after conversion (quantiles only
  for cells of at least the minimum size), semantic flags (e.g. CRRT ultrafiltration),
  and per dose concept the implausible running doses and applied unit corrections;
- where `bin_index` placed each binned value (inside a segment, in a gap, clamped low
  or high) in total and per concept;
- dose-conversion status counts per dose table (unconverted doses by reason);
- unit mismatches against the expected units, and concepts charted in several units;
- ``<unk>`` counts and rates per partition, on the non-fit partitions, and per concept;
- events per stay (mean, p99) against the context window; GEM windows per stay;
- the availability semantics and lag of every configured table.

Disclosure control (the artifact policy's ``classes.aggregate_no_phi.minimum_cell_size``,
10 in configs/artifact_policy.yaml; `policy_min_cell`): every patient-derived count below
it is written as ``"<10"``, and a mean, percentile or rate whose base has fewer than 10
stays/tokens is withheld (``None``), as is a rate whose numerator is a suppressed count
(1-9; rate x tokens would give it back). Complementary suppression: in every section whose
cells sum to a published total (token kinds and per-source events vs the event total, value
placements vs binned events, ``<unk>`` per partition / per concept vs the overall, GEM
dispositions and admission types vs stays), a lone suppressed nonzero cell would be the
total minus the others, so the next-smallest published cell is withheld too
(``"suppressed"``), iterated until no equation has exactly one hidden nonzero cell; the
non-fit ``<unk>`` count the gate reads is withheld last. Structural numbers (vocabulary
size, concepts per binning source, configuration) are not patient counts. Before writing,
the report is scanned for identifier keys and for any string equal to a hospitalization or
patient id; a hit fails closed.

The gate (`gate_report`, ``python -m src.data.tokenization_report --report ...``) is the
real-data verification step: availability declared for every configured table, no
numeric concept emitted bare in the fit partition, ``<unk>`` rate under 1% on the
non-fit partitions, events per stay within the context window, and no unsuppressed
small cell. It prints only check names and aggregate details.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import polars as pl

from src.data.cohort import QualificationError
from src.data.segments import UNK_ID, PLACEMENTS, as_segments, classify_values

REPORT_VERSION = 1
REPORT_FILE = "tokenization_report.json"
GEM_REPORT_FILE = "gem_tokenization_report.json"
# Default suppression threshold when no artifact policy is supplied (`policy_min_cell`).
MIN_CELL = 10
SUPPRESSED = f"<{MIN_CELL}"
# A cell at or above the threshold withheld so a suppressed one cannot be recovered by
# subtraction from a published total (complementary suppression).
COMPLEMENTARY = "suppressed"
LOW_COVERAGE_STAYS = 50
DEFAULT_CONTEXT = 8192
MAX_NON_FIT_UNK_RATE = 0.01
# Sections whose integers are patient-derived counts (suppressed below MIN_CELL).
COUNT_SECTIONS = ("sources", "events", "token_kinds", "unk", "value_placement",
                  "dose_conversion", "multi_unit_concepts", "gem", "binning",
                  "data_quality")
# Integer fields inside those sections that are configuration, not counts.
NOT_COUNTS = frozenset({"context", "max_tokens", "low_coverage_threshold", "concepts"})
IDENTIFIER_KEYS = frozenset({"hosp_id", "hospitalization_id", "patient_id",
                             "hospitalization_joined_id", "episode_key", "mrn",
                             "encounter_id"})


# ------------------------------------------------------------------ disclosure control

def policy_min_cell(policy: Mapping | None) -> int:
    """The report's suppression threshold: the artifact policy's
    ``classes.aggregate_no_phi.minimum_cell_size`` (`MIN_CELL` when no policy is given).
    A policy that does not declare a positive integer fails closed."""
    if policy is None:
        return MIN_CELL
    try:
        value = policy["classes"]["aggregate_no_phi"]["minimum_cell_size"]
    except (KeyError, TypeError) as exc:
        raise QualificationError(
            "artifact policy does not declare classes.aggregate_no_phi.minimum_cell_size; "
            "refusing to guess a suppression threshold") from exc
    if not _is_count(value) or value < 1:
        raise QualificationError(f"minimum_cell_size must be a positive integer, got {value!r}")
    return value


def _report_min_cell(report: Mapping) -> int:
    value = (report.get("suppression") or {}).get("min_cell_size")
    return value if _is_count(value) and value >= 1 else MIN_CELL


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _suppress(node: Any, min_cell: int, key: str | None = None) -> Any:
    if isinstance(node, dict):
        return {k: _suppress(v, min_cell, k) for k, v in node.items()}
    if isinstance(node, list):
        return [_suppress(v, min_cell, key) for v in node]
    if _is_count(node) and key not in NOT_COUNTS and node < min_cell:
        return f"<{min_cell}"
    return node


CellPath = tuple   # a cell's key path from the report root


def _get(node: Any, path: CellPath) -> Any:
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set(node: dict, path: CellPath, value: Any) -> None:
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


# Withheld only when nothing else in an equation can be (the gate reads the non-fit
# <unk> rate).
_LAST_RESORT = frozenset({("unk", "non_fit", "unk"), ("unk", "non_fit", "tokens")})


def _equations(report: Mapping) -> list[list[CellPath]]:
    """Published totals and the cells summing to them, ``[total, *cells]`` (a solvable
    linear equation: any ONE unknown member is recoverable from the others)."""
    eqs: list[list[CellPath]] = []

    def children(path: CellPath, *leaf: str) -> list[CellPath]:
        node = _get(report, path)
        return ([(*path, k, *leaf) for k in node] if isinstance(node, dict) else [])

    fit = report.get("fit_partition")
    by_partition = _get(report, ("unk", "by_partition")) or {}
    for field in ("unk", "tokens"):
        if fit in by_partition:        # overall = fit + non_fit
            eqs.append([("unk", "overall", field), ("unk", "by_partition", fit, field),
                        ("unk", "non_fit", field)])
        eqs.append([("unk", "overall", field), *children(("unk", "by_partition"), field)])
        eqs.append([("unk", "non_fit", field),
                    *(("unk", "by_partition", p, field) for p in by_partition if p != fit)])
    eqs.append([("unk", "overall", "unk"), *children(("unk", "by_concept"))])
    eqs.append([("events", "total"), *children(("token_kinds",))])
    eqs.append([("events", "total"),
                *(p for p in children(("sources",), "events")
                  if isinstance(_get(report, p[:-1]), dict))])
    eqs.append([("token_kinds", "binned"), *children(("value_placement", "total"))])
    by_concept = _get(report, ("value_placement", "by_concept")) or {}
    for placement in PLACEMENTS:
        if placement != "inside":
            eqs.append([("value_placement", "total", placement),
                        *(("value_placement", "by_concept", c, placement)
                          for c in by_concept)])
    for section in ("dispositions", "admission_types"):
        eqs.append([("gem", "stays"), *children(("gem", section))])
    eqs.append([("token_kinds", "binned"), *children(("binning", "by_source"), "events")])
    return [eq for eq in eqs if len(eq) > 2]


def _complementary(report: Mapping, min_cell: int) -> tuple[set, set]:
    """(hidden, complementary) cell paths: hidden = every count below `min_cell` plus the
    complementary cells chosen so no equation has exactly ONE hidden nonzero member (a
    hidden zero discloses nobody and may be inferable, so it protects nothing)."""
    eqs = []
    for eq in _equations(report):
        cells = [(path, _get(report, path)) for path in eq]
        cells = [(path, v) for path, v in cells if _is_count(v)]
        if len(cells) > 1:
            eqs.append(cells)
    hidden = {path for eq in eqs for path, v in eq if v < min_cell}
    complementary: set = set()
    changed = True
    while changed:
        changed = False
        for cells in eqs:
            exposed = [path for path, v in cells if path in hidden and v != 0]
            if len(exposed) != 1:
                continue
            candidates = sorted((path in _LAST_RESORT, v, path)
                                for path, v in cells if path not in hidden)
            if candidates:
                path = candidates[0][2]
                hidden.add(path)
                complementary.add(path)
                changed = True
    return hidden, complementary


def suppress_small_cells(report: dict, min_cell: int | None = None) -> dict:
    """Replace every patient-derived count below `min_cell` (default: the report's
    recorded ``suppression.min_cell_size``, else `MIN_CELL`) with ``"<{min_cell}"``,
    withhold complementary cells (``"suppressed"``) and every ``<unk>`` rate whose
    count or base is withheld."""
    min_cell = _report_min_cell(report) if min_cell is None else min_cell
    hidden, complementary = _complementary(report, min_cell)
    out = {k: (_suppress(v, min_cell) if k in COUNT_SECTIONS else v)
           for k, v in report.items()}
    for path in complementary:
        _set(out, path, COMPLEMENTARY)

    def withhold_rates(node: Any, path: CellPath) -> None:
        if not isinstance(node, dict):
            return
        if "rate" in node and node["rate"] is not None and (
                ((*path, "unk") in hidden and _get(report, (*path, "unk")) != 0)
                or (*path, "tokens") in hidden):
            node["rate"] = None
        for key, value in node.items():
            withhold_rates(value, (*path, key))

    withhold_rates(out.get("unk"), ("unk",))
    return out


def _walk(node: Any):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)
    elif isinstance(node, str):
        yield node


def assert_aggregate_only(report: Mapping, identifiers: Iterable[str]) -> None:
    """Fail closed on an identifier-named key, or any key / string value equal to a
    hospitalization or patient identifier."""
    ids = set(identifiers)
    for item in _walk(report):
        if item in IDENTIFIER_KEYS:
            raise QualificationError(
                f"tokenization report contains an identifier field ({item!r}); "
                "reports are aggregate-only")
        if item in ids:
            raise QualificationError(
                "tokenization report contains a patient or hospitalization identifier; "
                "reports are aggregate-only")


def write_report(report: dict, path: Path, identifiers: Iterable[str]) -> dict:
    """Suppress small cells (at the report's recorded threshold), scan for identifiers,
    then write (sorted keys, stable)."""
    final = suppress_small_cells(report)
    assert_aggregate_only(final, identifiers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


# ------------------------------------------------------------------ building

def _rate(num: int, den: int, min_cell: int = MIN_CELL) -> float | None:
    """`num / den`, withheld when the base is a small cell or the numerator is a
    suppressed (nonzero, below `min_cell`) count: rate x base would give it back."""
    if den < min_cell or 0 < num < min_cell:
        return None
    return round(num / den, 6)


def _stay_stats(lengths: np.ndarray, context: int, min_cell: int = MIN_CELL) -> dict:
    n = int(lengths.size)
    enough = n >= min_cell
    return {
        "stays": n,
        "per_stay_mean": round(float(lengths.mean()), 2) if enough else None,
        "per_stay_p99": round(float(np.percentile(lengths, 99)), 2) if enough else None,
        "stays_over_context": int((lengths > context).sum()),
        "context": int(context),
    }


def _finite() -> pl.Expr:
    return pl.col("value").is_finite().fill_null(False)


def _has_category() -> pl.Expr:
    if_null = pl.col("cat_value").cast(pl.String).str.strip_chars()
    return (if_null.is_not_null() & (if_null != "")).fill_null(False)


def _token_kinds(events: pl.DataFrame, binned: list[str]) -> dict[str, int]:
    finite, cat = _finite(), _has_category()
    in_edges = pl.col("concept").is_in(binned)
    kinds = events.select(
        (finite & in_edges).sum().alias("binned"),
        (finite & ~in_edges).sum().alias("numeric_bare"),
        (~finite & cat).sum().alias("categorical"),
        (~finite & ~cat & ~in_edges).sum().alias("presence"),
        (~finite & ~cat & in_edges).sum().alias("skipped_missing_numeric"),
    )
    return {k: int(v) for k, v in kinds.row(0, named=True).items()}


def _bare_concepts(events: pl.DataFrame, fit_mask: pl.Expr, binned: list[str]) -> dict:
    """Numeric concepts emitted bare (finite value, no bins), in the fit partition and
    outside it."""
    bare = events.filter(_finite() & ~pl.col("concept").is_in(binned))

    def names(frame: pl.DataFrame) -> list[str]:
        return sorted(frame["concept"].drop_nulls().unique().to_list())

    return {"fit": names(bare.filter(fit_mask)), "non_fit": names(bare.filter(~fit_mask))}


def _value_placement(events: pl.DataFrame, segments: Mapping) -> dict:
    numeric = events.filter(_finite() & pl.col("concept").is_in(list(segments)))
    total = dict.fromkeys(PLACEMENTS, 0)
    by_concept: dict[str, dict[str, int]] = {}
    for (name,), frame in numeric.partition_by("concept", as_dict=True).items():
        counts = classify_values(frame["value"].to_numpy(), as_segments(segments[name]))
        for k, v in counts.items():
            total[k] += v
        off = {k: v for k, v in counts.items() if k != "inside" and v}
        if off:
            by_concept[str(name)] = {k: counts[k] for k in PLACEMENTS if k != "inside"}
    return {"total": total, "by_concept": dict(sorted(by_concept.items()))}


def _sources(events: pl.DataFrame, tables: Iterable[str]) -> dict:
    counts = {
        source: (int(n), int(stays))
        for source, n, stays in events.group_by("source")
        .agg(pl.len(), pl.col("hosp_id").n_unique()).iter_rows()
    }
    out = {}
    for table in sorted(set(tables) | set(counts)):
        n, stays = counts.get(table, (0, 0))
        out[table] = {"events": n, "stays": stays,
                      "low_coverage": stays < LOW_COVERAGE_STAYS}
    return out


def _multi_unit(events: pl.DataFrame, unit_key: Callable[[str], str]) -> dict:
    if "unit" not in events.columns:
        return {}
    charted = (events.filter(pl.col("unit").is_not_null() & (pl.col("unit") != ""))
               .group_by("concept", "unit").len())
    grouped: dict[str, dict[str, int]] = {}
    for concept, unit, n in charted.iter_rows():
        key = unit_key(unit)
        grouped.setdefault(concept, {})
        grouped[concept][key] = grouped[concept].get(key, 0) + int(n)
    return {c: dict(sorted(u.items())) for c, u in sorted(grouped.items()) if len(u) > 1}


def _unk(token_frame: pl.DataFrame, fit_partition: str | None,
         min_cell: int = MIN_CELL) -> dict:
    per = (token_frame.select(
        "partition",
        pl.col("token").list.len().alias("tokens"),
        pl.col("token").list.count_matches(UNK_ID).alias("unk"),
    ).group_by("partition").agg(pl.col("tokens").sum(), pl.col("unk").sum()).sort("partition"))
    by_partition = {}
    totals = {"overall": [0, 0], "non_fit": [0, 0]}
    for partition, tokens, unk in per.iter_rows():
        by_partition[str(partition)] = {"tokens": int(tokens), "unk": int(unk),
                                        "rate": _rate(int(unk), int(tokens), min_cell)}
        totals["overall"][0] += int(tokens)
        totals["overall"][1] += int(unk)
        if partition != fit_partition:
            totals["non_fit"][0] += int(tokens)
            totals["non_fit"][1] += int(unk)
    return {
        "by_partition": by_partition,
        **{name: {"tokens": t, "unk": u, "rate": _rate(u, t, min_cell)}
           for name, (t, u) in totals.items()},
    }


def _binning(events: pl.DataFrame, binning_sources: Mapping, min_cell: int) -> dict:
    """Per binning source: concepts (structural), this run's binned events and their
    share of all binned events."""
    binned = sorted(binning_sources)
    counts = dict(events.filter(_finite() & pl.col("concept").is_in(binned))
                  .group_by("concept").len().iter_rows())
    total = int(sum(counts.values()))
    by_source: dict[str, dict] = {}
    for concept, source in binning_sources.items():
        row = by_source.setdefault(source, {"concepts": 0, "events": 0})
        row["concepts"] += 1
        row["events"] += int(counts.get(concept, 0))
    for row in by_source.values():
        row["event_share"] = _rate(row["events"], total, min_cell)
    return {"by_source": dict(sorted(by_source.items()))}


def _literature(binning_sources: Mapping, literature: Mapping | None) -> dict:
    """The literature-grounded concepts (configuration, not patient counts): decisions,
    and those with unverified sources or edges supported only by unverified sources."""
    out: dict[str, Any] = {}
    if literature:
        concepts = literature.get("concepts") or {}
        out = {
            "fragments": sorted(literature.get("fragments") or ()),
            "concepts": sorted(c for c, s in binning_sources.items() if s == "literature"),
            "decisions": {c: e.get("decision") for c, e in sorted(concepts.items())},
            "unverified_sources": {c: int(e.get("unverified", 0))
                                   for c, e in sorted(concepts.items()) if e.get("unverified")},
            "unverified_only_edges": {c: list(e["unverified_only_edges"])
                                      for c, e in sorted(concepts.items())
                                      if e.get("unverified_only_edges")},
            "ignored_csv": sorted(literature.get("ignored_csv") or ()),
            "absent": sorted(literature.get("absent") or ()),
        }
    return out


def _data_quality(quality: Mapping | None, min_cell: int) -> dict:
    """The site's unit-repair record with every statistic of a small cell withheld:
    quantiles when fewer than `min_cell` values, a share whose base or numerator is a
    small cell. (Counts are suppressed with the rest of the section.)"""
    if not quality:
        return {}
    out = json.loads(json.dumps(dict(quality)))
    for key, bad in (("columns", "out_of_range"), ("doses", "implausible")):
        for entry in (out.get(key) or {}).values():
            n, k = int(entry.get("n", 0)), int(entry.get(bad, 0))
            entry["share"] = _rate(k, n, min_cell)
            if n < min_cell:
                for q in ("p01", "p50", "p99"):
                    if q in entry:
                        entry[q] = None
    return out


def build_report(*, trajectory: str, events: pl.DataFrame, records: pl.DataFrame,
                 vocab: Mapping, segments: Mapping, binning_sources: Mapping,
                 availability: Mapping, tables: Iterable[str], dose_conversion: Mapping,
                 unit_mismatches: list[str], unk_by_concept: Mapping[str, int],
                 fit_partition: str | None, binding: Mapping, vocab_sample: bool,
                 run_sample_size: int | None, context: int,
                 unit_key: Callable[[str], str] = lambda u: u.strip().lower(),
                 gem: Mapping | None = None, min_cell: int = MIN_CELL,
                 missing_columns: Mapping[str, list[str]] | None = None,
                 matched_granularity: Mapping | None = None,
                 data_quality: Mapping | None = None,
                 literature: Mapping | None = None) -> dict:
    """Aggregate-only report (unsuppressed; `write_report` applies disclosure control at
    `min_cell`, recorded under ``suppression``). `missing_columns`: configured wide-table
    columns a site's parquet lacks (skipped, never a binder error), per table.

    `events`: the windowed, ordered events (hosp_id, source, concept, value, cat_value,
    unit, partition). `records`: the written artifact rows (token, partition, n_events,
    hosp_id; GEM rows are windows). `fit_partition` is the partition the vocabulary was
    fit on when this site built it, else None (every partition is non-fit)."""
    binned = sorted(segments)
    sources_count: dict[str, int] = {}
    for concept in binned:
        source = binning_sources.get(concept, "unknown")
        sources_count[source] = sources_count.get(source, 0) + 1
    # `matched_granularity`: a decile-arm vocabulary's KTD11 record (concepts at the
    # clinical arm's bin count, and every exception), listed under ``vocab``.
    fit_mask = (pl.col("partition") == fit_partition) if fit_partition is not None \
        else pl.lit(False)
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "trajectory": trajectory,
        "role": "reference" if fit_partition is not None else "imported",
        "fit_partition": fit_partition,
        "binding": dict(binding),
        "sample": {"vocab_sample": bool(vocab_sample),
                   "run_sample": run_sample_size is not None,
                   "run_sample_size": run_sample_size},
        "suppression": {"min_cell_size": int(min_cell), "marker": f"<{min_cell}",
                        "complementary_marker": COMPLEMENTARY,
                        "applies_to": "patient-derived counts"},
        "availability": {name: dict(spec) for name, spec in sorted(availability.items())},
        "vocab": {
            "size": len(vocab),
            "binned_concepts": len(binned),
            "binning_sources": dict(sorted(sources_count.items())),
            "single_bin_concepts": sorted(c for c in binned
                                          if binning_sources.get(c) == "single"),
        },
        "sources": _sources(events, tables),
        "token_kinds": _token_kinds(events, binned),
        "numeric_bare_concepts": _bare_concepts(events, fit_mask, binned),
        "value_placement": _value_placement(events, segments),
        "dose_conversion": {t: dict(sorted(c.items()))
                            for t, c in sorted(dose_conversion.items())},
        "unit_mismatches": sorted(unit_mismatches),
        "multi_unit_concepts": _multi_unit(events, unit_key),
        "missing_columns": {t: sorted(c) for t, c in sorted((missing_columns or {}).items())
                            if c},
        "unk": {**_unk(records, fit_partition, min_cell),
                "by_concept": {c: int(n) for c, n in sorted(unk_by_concept.items())}},
        "binning": _binning(events, binning_sources, min_cell),
        "literature": _literature(binning_sources, literature),
        "data_quality": _data_quality(data_quality, min_cell),
    }
    if matched_granularity:
        report["vocab"]["matched_granularity"] = {
            "reference_scheme": matched_granularity.get("reference_scheme"),
            "forced_edges": bool(matched_granularity.get("forced_edges")),
            "matched": int(matched_granularity.get("matched", 0)),
            "exceptions": {c: dict(row) for c, row in sorted(
                (matched_granularity.get("exceptions") or {}).items())},
            "not_fit": sorted(matched_granularity.get("not_fit") or ()),
        }
    if trajectory == "hospitalization":
        per_stay = (records.group_by("hosp_id")
                    .agg(pl.col("n_events").sum().alias("tokens"), pl.len().alias("windows")))
        tokens = per_stay["tokens"].to_numpy() if len(per_stay) else np.array([], int)
        windows = per_stay["windows"].to_numpy() if len(per_stay) else np.array([], int)
        lengths = records["n_events"].to_numpy() if len(records) else np.array([], int)
        enough = tokens.size >= min_cell
        report["events"] = {"total": int(len(events))}
        report["gem"] = {
            **{k: v for k, v in (gem or {}).items() if k not in ("stays", "windows")},
            "stays": int(tokens.size),
            "windows": int(lengths.size),
            "max_tokens": int(context),
            "windows_per_stay_mean": round(float(windows.mean()), 3) if enough else None,
            "stays_with_multiple_windows": int((windows > 1).sum()),
            "tokens_per_stay_mean": round(float(tokens.mean()), 2) if enough else None,
            "tokens_per_stay_p99": (round(float(np.percentile(tokens, 99)), 2)
                                    if enough else None),
            "window_tokens_p99": (round(float(np.percentile(lengths, 99)), 2)
                                  if lengths.size >= min_cell else None),
        }
    else:
        lengths = records["n_events"].to_numpy() if len(records) else np.array([], int)
        report["events"] = {"total": int(len(events)),
                            **_stay_stats(lengths, context, min_cell)}
    return report


# ------------------------------------------------------------------ the gate

def _check(name: str, passed: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(passed), "detail": detail}


def gate_report(report: Mapping, cfg: Mapping, *,
                max_unk_rate: float = MAX_NON_FIT_UNK_RATE) -> list[dict]:
    """Real-data verification checks over one (written, suppressed) report. Returns
    ``[{check, passed, detail}]``; every detail is aggregate-only."""
    from src.data.tokenize import AVAILABILITY_SEMANTICS, validate_table_availability

    checks: list[dict] = []
    tables = (cfg or {}).get("tables") or {}
    try:
        validate_table_availability(tables)
        declared = report.get("availability") or {}
        missing = sorted(t for t in tables
                         if (declared.get(t) or {}).get("availability")
                         not in AVAILABILITY_SEMANTICS)
        checks.append(_check(
            "availability_declared", not missing,
            "every configured table declares availability" if not missing
            else f"report lacks availability for: {', '.join(missing)}"))
    except QualificationError as exc:
        checks.append(_check("availability_declared", False, str(exc)))

    try:
        assert_aggregate_only(report, ())
        checks.append(_check("aggregate_only", True, "no identifier fields"))
    except QualificationError as exc:
        checks.append(_check("aggregate_only", False, str(exc)))

    min_cell = _report_min_cell(report)
    small = [section for section in COUNT_SECTIONS
             if _has_small_cell(report.get(section), min_cell)]
    checks.append(_check("small_cells_suppressed", not small,
                         f"every count under {min_cell} suppressed" if not small
                         else f"unsuppressed small counts in: {', '.join(small)}"))

    if report.get("role") == "reference":
        bare = (report.get("numeric_bare_concepts") or {}).get("fit") or []
        checks.append(_check("numeric_concepts_binned", not bare,
                             "0 numeric concepts bare in the fit partition" if not bare
                             else f"{len(bare)} numeric concept(s) bare: {', '.join(bare)}"))

    non_fit = (report.get("unk") or {}).get("non_fit") or {}
    rate, nf_tokens = non_fit.get("rate"), non_fit.get("tokens")
    if rate is not None:
        passed = rate < max_unk_rate
        detail = f"<unk> rate on non-fit partitions = {rate} (limit {max_unk_rate})"
    elif (_is_count(nf_tokens) and nf_tokens >= min_cell
          and non_fit.get("unk") == f"<{min_cell}"):
        # The rate is withheld because the <unk> count is a small cell; the published
        # values still bound it by (min_cell - 1) / tokens.
        bound = (min_cell - 1) / nf_tokens
        passed = bound < max_unk_rate
        detail = (f"<unk> rate on non-fit partitions < {round(bound, 6)} "
                  f"(count under {min_cell}; limit {max_unk_rate})")
    else:
        passed = False
        detail = f"non-fit <unk> rate not measurable (base under {min_cell} or withheld)"
    checks.append(_check("unk_rate_non_fit", passed, detail))

    if report.get("trajectory") == "hospitalization":
        gem = report.get("gem") or {}
        p99, limit = gem.get("window_tokens_p99"), gem.get("max_tokens", DEFAULT_CONTEXT)
        checks.append(_check(
            "windows_within_context", p99 is not None and p99 <= limit,
            f"window length p99 = {p99} (limit {limit}); windows per stay mean = "
            f"{gem.get('windows_per_stay_mean')}"))
    else:
        events = report.get("events") or {}
        p99, limit = events.get("per_stay_p99"), events.get("context", DEFAULT_CONTEXT)
        checks.append(_check(
            "events_per_stay_within_context", p99 is not None and p99 <= limit,
            f"events per stay mean = {events.get('per_stay_mean')}, p99 = {p99} "
            f"(context {limit})"))

    low = sorted(t for t, s in (report.get("sources") or {}).items()
                 if isinstance(s, Mapping) and s.get("low_coverage"))
    checks.append(_check("low_coverage_sources_flagged", True,
                         f"unverified (under {LOW_COVERAGE_STAYS} stays): "
                         f"{', '.join(low) or 'none'}"))
    single = (report.get("vocab") or {}).get("single_bin_concepts") or []
    checks.append(_check("single_bin_concepts_listed", True,
                         f"{len(single)} single-bin concept(s): {', '.join(single) or 'none'}"))
    return checks


def _has_small_cell(node: Any, min_cell: int = MIN_CELL, key: str | None = None) -> bool:
    if isinstance(node, dict):
        return any(_has_small_cell(v, min_cell, k) for k, v in node.items())
    if isinstance(node, list):
        return any(_has_small_cell(v, min_cell, key) for v in node)
    return _is_count(node) and key not in NOT_COUNTS and node < min_cell


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Real-data tokenization gate (aggregate-only)")
    ap.add_argument("--report", action="append", required=True,
                    help="tokenization_report.json / gem_tokenization_report.json "
                         "(repeatable)")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--max-unk-rate", type=float, default=MAX_NON_FIT_UNK_RATE)
    args = ap.parse_args(argv)
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    ok = True
    for path in args.report:
        report = json.loads(Path(path).read_text())
        print(f"[{report.get('trajectory', '?')}] {Path(path).name}")
        for check in gate_report(report, cfg, max_unk_rate=args.max_unk_rate):
            ok &= check["passed"]
            print(f"  {'PASS' if check['passed'] else 'FAIL'}  {check['check']}: "
                  f"{check['detail']}")
    print("GATE PASS" if ok else "GATE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
