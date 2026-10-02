"""Closure-aware value segments, precedence policy v1, and THE binning function.

A binned concept is an ordered list of segments, each a plain JSON-serializable dict
``{"lo": float | None, "hi": float | None, "lo_closed": bool, "hi_closed": bool}``.
A point bin has ``lo == hi`` with both ends closed. ``None`` is an unbounded end and is
only used by quantile (decile) segments; physician CSV segments are always finite. The
fused token for a value is ``concept=<segment index>``.

`bin_index` is the only code path that maps a value to a bin. Everything else
(`tokenize._bin_of`, soft discretization, and later the threshold query, plausibility,
viewer, generation and the site package) must delegate to it.

Physician CSV interval flags: ``[`` is ``>=``, ``(`` is ``>``, ``]`` is ``<=``, ``)`` is
``<``. A row with ``exact_dose_token == 1`` matches only ``value == min_value``.

PRECEDENCE POLICY v1 (applied in this order; recorded in the vocabulary artifact):

1. Boundaries within 1e-6 relative of each other are merged onto one value (a forced
   edge wins the merge; otherwise the smallest value does).
2. Overlap: the earlier segment keeps its upper endpoint and closure; the later segment
   starts there with the complementary closure. (This, not step 1, is what resolves the
   CSV's temp_c ``(37.111,37.5]`` / ``(37.499,37.611]`` pair: 0.001/37.5 is ~2.7e-5
   relative, far above the step-1 tolerance.)
3. Forced edges split the containing segment, closure set by the target's outcome
   direction so the threshold value lands on the non-event side: ``below`` puts the
   edge value in the bin ABOVE (``[t, ...)``), ``above`` puts it in the bin BELOW
   (``(..., t]``). Non-target forced edges use ``[`` (edge value goes above). A forced
   edge that already is a boundary only has its closure set; one inside a gap extends
   both neighbours to it; one outside the segment range fails closed.
4. Exact point rows (and any closed ``[v, v]`` row) become point segments and win over
   any interval containing the same value.
5. A value in a gap goes to the nearest segment; an equidistant value goes to the lower.
6. A value beyond the floor or ceiling clamps to the end segment.
7. CSV measurement prefix ``angiotension`` is aliased to CLIF ``angiotensin``.

Steps 5 and 6 are applied by `bin_index`; the rest when segments are built.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

POLICY_VERSION = 1
MERGE_REL_TOL = 1e-6
CSV_MEASUREMENT_ALIASES: tuple[tuple[str, str], ...] = (("angiotension", "angiotensin"),)
DIRECTIONS = ("above", "below")

_LO_FLAGS = {"[": True, "(": False}
_HI_FLAGS = {"]": True, ")": False}


def make_segment(lo: float | None, hi: float | None, lo_closed: bool, hi_closed: bool) -> dict:
    return {
        "lo": None if lo is None else float(lo),
        "hi": None if hi is None else float(hi),
        "lo_closed": bool(lo_closed),
        "hi_closed": bool(hi_closed),
    }


def is_point(seg: Mapping) -> bool:
    return seg["lo"] is not None and seg["lo"] == seg["hi"]


def _is_empty(seg: Mapping) -> bool:
    lo, hi = seg["lo"], seg["hi"]
    if lo is None or hi is None:
        return False
    if lo > hi:
        return True
    return lo == hi and not (seg["lo_closed"] and seg["hi_closed"])


def _lo_key(seg: Mapping) -> float:
    return -math.inf if seg["lo"] is None else seg["lo"]


def _hi_key(seg: Mapping) -> float:
    return math.inf if seg["hi"] is None else seg["hi"]


def contains(seg: Mapping, value: float) -> bool:
    """True if `value` lies inside `seg` under its closure flags."""
    lo, hi = seg["lo"], seg["hi"]
    if lo is not None and (value < lo or (value == lo and not seg["lo_closed"])):
        return False
    if hi is not None and (value > hi or (value == hi and not seg["hi_closed"])):
        return False
    return True


def validate_partition(segments: Sequence[Mapping]) -> None:
    """Raise ValueError unless `segments` is a sorted, non-empty, non-overlapping list."""
    if not segments:
        raise ValueError("a binned concept needs at least one segment")
    last = len(segments) - 1
    for i, seg in enumerate(segments):
        if set(seg) != {"lo", "hi", "lo_closed", "hi_closed"}:
            raise ValueError(f"segment {i} has unexpected keys: {sorted(seg)}")
        if (seg["lo"] is None and i != 0) or (seg["hi"] is None and i != last):
            raise ValueError(f"segment {i} is unbounded but not an end segment")
        for bound in (seg["lo"], seg["hi"]):
            if bound is not None and not math.isfinite(bound):
                raise ValueError(f"segment {i} has a non-finite bound")
        if _is_empty(seg):
            raise ValueError(f"segment {i} is empty: {seg}")
    for i, (a, b) in enumerate(zip(segments, segments[1:])):
        if a["hi"] > b["lo"]:
            raise ValueError(f"segments {i} and {i + 1} overlap")
        if a["hi"] == b["lo"] and a["hi_closed"] and b["lo_closed"]:
            raise ValueError(f"segments {i} and {i + 1} share a closed endpoint")


def bin_index(value: float | None, segments: Sequence[Mapping]) -> int | None:
    """The single binning function: index of the segment `value` belongs to.

    Point segments first, then intervals (policy steps 1-4 already made these a strict
    partition), then the gap rule (nearest; equidistant -> lower), then clamping to the
    end segment. None, NaN and +/-inf get no bin."""
    if value is None or not segments:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    for i, seg in enumerate(segments):
        if is_point(seg) and seg["lo"] == v:
            return i
    for i, seg in enumerate(segments):
        if not is_point(seg) and contains(seg, v):
            return i
    first, last = segments[0], segments[-1]
    if first["lo"] is not None and v <= first["lo"]:
        return 0
    if last["hi"] is not None and v >= last["hi"]:
        return len(segments) - 1
    for i in range(len(segments) - 1):
        below, above = segments[i]["hi"], segments[i + 1]["lo"]
        if below <= v <= above:
            return i if (v - below) <= (above - v) else i + 1
    raise ValueError(f"segments are not a sorted partition; cannot bin {v!r}")


def _neighbor_width(segments: Sequence[Mapping], start: int, step: int) -> float:
    j = start + step
    while 0 <= j < len(segments):
        lo, hi = segments[j]["lo"], segments[j]["hi"]
        if lo is not None and hi is not None and hi > lo:
            return hi - lo
        j += step
    return 1e-12


def soft_bins(value: float | None, segments: Sequence[Mapping],
              kernel_bins: int) -> list[tuple[int | None, float]]:
    """Fixed-width (2*kernel_bins+1) Gaussian soft assignment around the hard bin.

    The kernel is centred at ``hard_bin + fractional position in the segment - 0.5``
    (sigma = max(k/2, 0.5)). Point segments sit at their centre; end segments borrow
    their neighbour's width because clamped values make their own width meaningless.
    A missing/categorical value pads a single ``(None, 1.0)`` assignment so every event
    yields the same-length list for the [B,T,K] tensor the encoder expects."""
    k = max(int(kernel_bins), 0)
    width = 2 * k + 1
    hard = bin_index(value, segments)

    def uniform(b: int | None) -> list[tuple[int | None, float]]:
        return [(b, 1.0)] + [(b, 0.0)] * (width - 1)

    n = len(segments)
    if hard is None or k == 0 or n < 2:
        return uniform(hard)

    v = float(value)
    seg = segments[hard]
    frac = 0.5
    if not is_point(seg):
        lo = None if hard == 0 else seg["lo"]
        hi = None if hard == n - 1 else seg["hi"]
        if lo is None and hi is not None:
            lo = hi - _neighbor_width(segments, hard, +1)
        elif hi is None and lo is not None:
            hi = lo + _neighbor_width(segments, hard, -1)
        if lo is not None and hi is not None and hi > lo:
            frac = min(max((v - lo) / (hi - lo), 0.0), 1.0)
    center = hard + frac - 0.5

    candidates = range(max(0, hard - k), min(n - 1, hard + k) + 1)
    sigma = max(k / 2, 0.5)
    raw = [math.exp(-0.5 * ((c - center) / sigma) ** 2) for c in candidates]
    total = sum(raw)
    bins: list[int | None] = [hard] * width
    weights = [0.0] * width
    for c, w in zip(candidates, raw):
        slot = c - (hard - k)
        bins[slot] = int(c)
        weights[slot] = float(w / total)
    return list(zip(bins, weights))


# ---- building segments under precedence policy v1 -------------------------------------

def _close(a: float, b: float) -> bool:
    return abs(a - b) <= MERGE_REL_TOL * max(abs(a), abs(b))


def _merge_map(values: Iterable[float], forced: Iterable[float]) -> dict[float, float]:
    """Step 1: map every boundary value onto its cluster representative."""
    forced_set = set(forced)
    mapping: dict[float, float] = {}
    cluster: list[float] = []

    def flush() -> None:
        if not cluster:
            return
        rep = next((v for v in cluster if v in forced_set), cluster[0])
        for v in cluster:
            mapping[v] = rep

    for v in sorted(set(values)):
        if cluster and not _close(cluster[0], v):
            flush()
            cluster = []
        cluster.append(v)
    flush()
    return mapping


def _check_direction(direction: str | None) -> None:
    if direction is not None and direction not in DIRECTIONS:
        raise ValueError(f"unknown outcome direction {direction!r}; expected one of {DIRECTIONS}")


def _apply_forced_edge(segments: list[dict], t: float, direction: str | None) -> list[dict]:
    """Step 3: make `t` a boundary whose value lands on the non-event side."""
    lower_owns = direction == "above"
    out = [dict(s) for s in segments]
    for i, s in enumerate(out):
        if not is_point(s) and _lo_key(s) < t < _hi_key(s):
            left = make_segment(s["lo"], t, s["lo_closed"], lower_owns)
            right = make_segment(t, s["hi"], not lower_owns, s["hi_closed"])
            return out[:i] + [left, right] + out[i + 1:]
    intervals = [i for i, s in enumerate(out) if not is_point(s)]
    if not intervals:
        raise ValueError(f"forced edge {t} has no interval segment to split")
    first, last = out[intervals[0]], out[intervals[-1]]
    if t < _lo_key(first) or t > _hi_key(last):
        raise ValueError(f"forced edge {t} lies outside the segment range "
                         f"[{first['lo']}, {last['hi']}]")
    if t == first["lo"] or t == last["hi"]:
        return out  # already the outer bound; values beyond it clamp
    # t is an existing boundary, or lies in a gap between two consecutive intervals.
    for a_i, b_i in zip(intervals, intervals[1:]):
        a, b = out[a_i], out[b_i]
        if a["hi"] <= t <= b["lo"]:
            a["hi"], a["hi_closed"] = t, lower_owns
            b["lo"], b["lo_closed"] = t, not lower_owns
            return out
    raise ValueError(f"could not place forced edge {t}")  # pragma: no cover - unreachable


def _insert_point(segments: list[dict], v: float) -> list[dict]:
    """Step 4: a point segment wins over any interval containing the same value."""
    out: list[dict] = []
    for s in segments:
        if is_point(s):
            if s["lo"] != v:
                out.append(dict(s))
            continue
        lo, hi = _lo_key(s), _hi_key(s)
        if lo < v < hi:
            out.append(make_segment(s["lo"], v, s["lo_closed"], False))
            out.append(make_segment(v, s["hi"], False, s["hi_closed"]))
        elif v == lo:
            out.append(make_segment(s["lo"], s["hi"], False, s["hi_closed"]))
        elif v == hi:
            out.append(make_segment(s["lo"], s["hi"], s["lo_closed"], False))
        else:
            out.append(dict(s))
    out.append(make_segment(v, v, True, True))
    out.sort(key=lambda s: (_lo_key(s), _hi_key(s)))
    return out


def build_segments(rows: Sequence[Mapping], forced_edges: Iterable[float] = (),
                   direction: str | None = None) -> list[dict]:
    """Apply precedence policy v1 to one concept's raw rows -> strict partition.

    Each row is ``{"lo", "hi", "lo_closed", "hi_closed"}`` plus optional ``"exact"``
    (CSV ``exact_dose_token == 1``: matches only ``value == lo``)."""
    _check_direction(direction)
    forced = sorted({float(e) for e in forced_edges if math.isfinite(float(e))})
    finite_bounds = [
        float(r[k]) for r in rows for k in ("lo", "hi") if r[k] is not None
    ]
    canon = _merge_map(finite_bounds + forced, forced)

    def snap(x: float | None) -> float | None:
        return None if x is None else canon[float(x)]

    points: set[float] = set()
    intervals: list[dict] = []
    for r in rows:
        lo, hi = snap(r["lo"]), snap(r["hi"])
        if r.get("exact"):
            if lo is None:
                raise ValueError("an exact point row needs a finite min_value")
            points.add(lo)
            continue
        seg = make_segment(lo, hi, r["lo_closed"], r["hi_closed"])
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"inverted segment row: {dict(r)}")
        if is_point(seg):
            if seg["lo_closed"] and seg["hi_closed"]:
                points.add(lo)
            continue  # an open point row is empty
        intervals.append(seg)

    # Step 2: overlaps -> earlier segment owns the shared region.
    intervals.sort(key=lambda s: (_lo_key(s), _hi_key(s), not s["lo_closed"]))
    out: list[dict] = []
    for s in intervals:
        if out:
            p = out[-1]
            p_hi = _hi_key(p)
            if _lo_key(s) < p_hi or (s["lo"] == p["hi"] and s["lo_closed"] and p["hi_closed"]):
                if p["hi"] is None:
                    continue  # swallowed by an unbounded earlier segment
                s = make_segment(p["hi"], s["hi"], not p["hi_closed"], s["hi_closed"])
                if _is_empty(s):
                    continue
        out.append(s)

    # Step 3: forced edges, direction-aware closure.
    for t in forced:
        t = canon[t]
        if not out:
            raise ValueError(f"forced edge {t} has no interval segment to split")
        out = _apply_forced_edge(out, t, direction)

    # Step 4: point rows win.
    for v in sorted(points):
        out = _insert_point(out, v)

    validate_partition(out)
    return out


def segments_from_edges(edges: Sequence[float], forced_edges: Iterable[float] = (),
                        direction: str | None = None) -> list[dict]:
    """Quantile (decile-arm) interior edges -> unbounded-end ``[a, b)`` segments.

    ``(-inf, e0), [e0, e1), ..., [e_n, +inf)`` with ``None`` for the unbounded ends; a
    forced edge then gets direction-aware closure (policy step 3)."""
    _check_direction(direction)
    bounds = sorted({float(e) for e in edges})
    if any(not math.isfinite(e) for e in bounds):
        raise ValueError("quantile edges must be finite")
    if not bounds:
        segs = [make_segment(None, None, False, False)]
    else:
        segs = [make_segment(None, bounds[0], False, False)]
        segs += [make_segment(a, b, True, False) for a, b in zip(bounds, bounds[1:])]
        segs.append(make_segment(bounds[-1], None, True, False))
    for t in sorted({float(e) for e in forced_edges if math.isfinite(float(e))}):
        segs = _apply_forced_edge(segs, t, direction)
    validate_partition(segs)
    return segs


def ordinal_segments(values: Iterable[float]) -> list[dict]:
    """One closed point segment per distinct value (integer ordinal scales: GCS, RASS,
    Braden). Values between or beyond the points follow the gap and clamp rules."""
    points = sorted({float(v) for v in values})
    if not points or any(not math.isfinite(v) for v in points):
        raise ValueError("ordinal segments need at least one finite value")
    segs = [make_segment(v, v, True, True) for v in points]
    validate_partition(segs)
    return segs


def is_ordinal(values: Sequence[float], max_distinct: int) -> bool:
    """Integer-valued with at most `max_distinct` distinct values (KTD3 source 2)."""
    distinct = {float(v) for v in values}
    return (0 < len(distinct) <= max_distinct
            and all(math.isfinite(v) and v == round(v) for v in distinct))


def has_zero_point(segments: Sequence[Mapping]) -> bool:
    return any(is_point(s) and s["lo"] == 0.0 for s in segments)


def with_zero_point(segments: Sequence[Mapping]) -> list[dict]:
    """A dose concept's ``[0, 0]`` stop bin (KTD3). Segments that already carry one (the
    CSV medication rows do) are returned unchanged, so it is never duplicated."""
    if has_zero_point(segments):
        return [dict(s) for s in segments]
    out = _insert_point([dict(s) for s in segments], 0.0)
    validate_partition(out)
    return out


def dose_segments_from_edges(edges: Sequence[float], forced_edges: Iterable[float] = (),
                             direction: str | None = None) -> list[dict]:
    """Quantile edges fit on strictly positive doses -> ``[0,0], (0, e0), [e0, e1), ...,
    [e_n, +inf)``. The zero bin keeps a stopped infusion distinct from any running dose;
    a (nonsensical) negative dose clamps into it."""
    forced = [float(e) for e in forced_edges if math.isfinite(float(e))]
    if any(e <= 0.0 for e in list(edges) + forced):
        raise ValueError("dose quantile and forced edges must be strictly positive")
    segs = segments_from_edges(edges, forced, direction)
    first = dict(segs[0])
    first["lo"], first["lo_closed"] = 0.0, False
    out = [make_segment(0.0, 0.0, True, True), first] + [dict(s) for s in segs[1:]]
    validate_partition(out)
    return out


def as_segments(obj: Sequence) -> list:
    """Compatibility: accept segments, or a legacy interior-edge list (``[a, b)`` rule)."""
    if obj and all(isinstance(x, Mapping) for x in obj):
        return obj  # type: ignore[return-value]
    return segments_from_edges(obj)


def n_bins(obj: Sequence) -> int:
    return len(as_segments(obj))


# ---- physician CSV loader ---------------------------------------------------------------

def alias_measurement(name: str) -> str:
    """Policy step 7: map CSV spellings onto CLIF concept names."""
    for old, new in CSV_MEASUREMENT_ALIASES:
        if name.startswith(old):
            return new + name[len(old):]
    return name


def read_csv_rows(csv_path: str | Path) -> dict[str, list[dict]]:
    """Raw interval rows per (aliased) measurement from the physician segmentation CSV."""
    rows: dict[str, list[dict]] = {}
    owner: dict[str, str] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = alias_measurement((row.get("measurement") or "").strip())
            if not name:
                continue
            category = (row.get("category") or "").strip()
            if owner.setdefault(name, category) != category:
                raise ValueError(f"measurement {name!r} appears under two categories")
            lo_flag = (row.get("min_interval") or "").strip()
            hi_flag = (row.get("max_interval") or "").strip()
            if lo_flag not in _LO_FLAGS or hi_flag not in _HI_FLAGS:
                raise ValueError(f"{name}: unknown interval flags {lo_flag!r}/{hi_flag!r}")
            exact = (row.get("exact_dose_token") or "0").strip()
            rows.setdefault(name, []).append({
                "lo": float(row["min_value"]),
                "hi": float(row["max_value"]),
                "lo_closed": _LO_FLAGS[lo_flag],
                "hi_closed": _HI_FLAGS[hi_flag],
                "exact": exact not in ("", "0", "0.0"),
            })
    return rows


def load_csv_segments(
    csv_path: str | Path,
    concepts: Iterable[str] | None = None,
    forced_edges: Mapping[str, Iterable[float]] | None = None,
    directions: Mapping[str, str] | None = None,
) -> dict[str, list[dict]]:
    """Per-concept strict partitions from the physician CSV under policy v1."""
    forced_edges = forced_edges or {}
    directions = directions or {}
    wanted = None if concepts is None else set(concepts)
    out: dict[str, list[dict]] = {}
    for name, rows in sorted(read_csv_rows(csv_path).items()):
        if wanted is not None and name not in wanted:
            continue
        out[name] = build_segments(rows, forced_edges.get(name, ()), directions.get(name))
    return out
