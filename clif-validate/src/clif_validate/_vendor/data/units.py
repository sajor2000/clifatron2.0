"""Dose-unit parsing and conversion (U4; R6, R7; KTD5).

One small, explicit conversion table, re-implemented here rather than taken from clifpy
(not a dependency). A unit string is ``quantity[/kg][/time]``:

- quantity: mass (pg, ng, mcg, mg, g), units (u, mu), meq, mmol, or volume (ul, ml, l);
- ``/kg``: a per-weight dose;
- time: ``/min``, ``/hr`` or ``/day`` (a rate). No time part = an amount (intermittent).

`normalize_unit` spells every unit one way (``units/hour`` -> ``u/hr``) and `unit_suffix`
turns it into a concept-safe suffix (``u_hr``), which matches the physician CSV's
medication measurement names (``vasopressin_u_min``). `convert_dose` returns
``(value, status)`` with status one of `STATUSES`:

- ``converted``: the target unit differs and the conversion succeeded;
- ``passthrough``: no target unit, or the unit already is the target;
- ``no_weight``: the conversion needs a weight and there is none (native value kept);
- ``unconvertible``: incompatible or unknown units (native value kept).

The tokenizer applies the same arithmetic vectorized: `dose_plan` resolves each distinct
(medication, unit) pair once to a factor and a weight power, so the 45M-row tables are
never converted row by row in Python.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from clif_validate._vendor.data.segments import alias_measurement

STATUSES = ("converted", "passthrough", "no_weight", "unconvertible")

# quantity spelling -> (canonical spelling, kind, factor to the kind's base unit)
_QUANTITIES: dict[str, tuple[str, str, float]] = {}
for _names, _canon, _kind, _factor in (
    (("pg", "picogram", "picograms"), "pg", "mass", 1e-9),
    (("ng", "nanogram", "nanograms"), "ng", "mass", 1e-6),
    (("mcg", "ug", "microgram", "micrograms"), "mcg", "mass", 1e-3),
    (("mg", "milligram", "milligrams"), "mg", "mass", 1.0),
    (("g", "gm", "gram", "grams"), "g", "mass", 1e3),
    (("u", "unit", "units", "iu"), "u", "units", 1.0),
    (("mu", "milliunit", "milliunits"), "mu", "units", 1e-3),
    (("meq",), "meq", "meq", 1.0),
    (("mmol",), "mmol", "mmol", 1.0),
    (("ul", "microliter", "microliters"), "ul", "volume", 1e-3),
    (("ml", "milliliter", "milliliters"), "ml", "volume", 1.0),
    (("l", "liter", "liters"), "l", "volume", 1e3),
):
    for _name in _names:
        _QUANTITIES[_name] = (_canon, _kind, _factor)

# time spelling -> (canonical spelling, minutes)
_TIMES: dict[str, tuple[str, float]] = {}
for _names, _canon, _minutes in (
    (("min", "mins", "minute", "minutes"), "min", 1.0),
    (("hr", "hrs", "h", "hour", "hours"), "hr", 60.0),
    (("day", "days", "d"), "day", 1440.0),
):
    for _name in _names:
        _TIMES[_name] = (_canon, _minutes)

_WEIGHT = {"kg"}
# Canonical intermittent quantity per kind (R7). Volume has none: mL of what is unknown.
_CANONICAL_QUANTITY = {"mass": "mg", "units": "u", "meq": "meq", "mmol": "mmol"}
_MICRO = ("¬µ", "µ", "μ")


def normalize_unit(raw: object) -> str:
    """One spelling per unit: lowercase, ``µ`` -> ``u``, aliases collapsed, parts joined by
    ``/``. Unknown parts are kept (sanitized) so they still name a fallback concept;
    a missing/blank unit is ``unknown``."""
    if raw is None:
        return "unknown"
    text = str(raw).strip().lower()
    for micro in _MICRO:
        text = text.replace(micro, "u")
    if not text:
        return "unknown"
    parts = []
    for part in text.split("/"):
        part = part.strip()
        if part in _QUANTITIES:
            parts.append(_QUANTITIES[part][0])
        elif part in _TIMES:
            parts.append(_TIMES[part][0])
        else:
            part = re.sub(r"[^a-z0-9]+", "_", part).strip("_")
            if part:
                parts.append(part)
    return "/".join(parts) or "unknown"


def unit_suffix(raw: object) -> str:
    """Concept-safe suffix for a unit (``mcg/kg/hour`` -> ``mcg_kg_hr``)."""
    return re.sub(r"[^a-z0-9]+", "_", normalize_unit(raw)).strip("_") or "unknown"


def normalize_name(raw: object) -> str:
    """Concept-safe medication / group name (lowercase, non-alphanumerics -> ``_``)."""
    return re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")


def parse_unit(raw: object) -> dict | None:
    """``{"kind", "factor", "per_kg", "minutes"}`` for a recognised unit, else None.
    ``minutes`` is None for an amount (no time part)."""
    parts = normalize_unit(raw).split("/")
    if parts[0] not in _QUANTITIES:
        return None
    _, kind, factor = _QUANTITIES[parts[0]]
    per_kg, minutes = False, None
    for part in parts[1:]:
        if part in _WEIGHT and not per_kg:
            per_kg = True
        elif part in _TIMES and minutes is None:
            minutes = _TIMES[part][1]
        else:
            return None
    return {"kind": kind, "factor": factor, "per_kg": per_kg, "minutes": minutes}


def conversion(src: object, dst: object) -> tuple[float, int] | None:
    """``(factor, weight_power)`` with ``dst = src * factor * weight_kg ** weight_power``,
    or None when the units cannot be converted (different kinds, rate vs amount, or an
    unrecognised unit)."""
    s, d = parse_unit(src), parse_unit(dst)
    if s is None or d is None or s["kind"] != d["kind"]:
        return None
    if (s["minutes"] is None) != (d["minutes"] is None):
        return None
    factor = s["factor"] / d["factor"]
    if s["minutes"] is not None:
        factor *= d["minutes"] / s["minutes"]
    power = int(s["per_kg"]) - int(d["per_kg"])  # per-kg source -> flat target: * weight
    return factor, power


def _valid_weight(weight_kg: float | None) -> bool:
    return weight_kg is not None and math.isfinite(weight_kg) and weight_kg > 0


def convert_dose(value: float, src: object, dst: object | None,
                 weight_kg: float | None = None) -> tuple[float, str]:
    """Convert one dose from its data unit to `dst` (None = no target unit)."""
    if dst is None or normalize_unit(src) == normalize_unit(dst):
        return value, "passthrough"
    conv = conversion(src, dst)
    if conv is None:
        return value, "unconvertible"
    factor, power = conv
    if power and not _valid_weight(weight_kg):
        return value, "no_weight"
    return value * factor * (weight_kg ** power if power else 1.0), "converted"


def canonical_unit(raw: object) -> str | None:
    """The intermittent canonical unit (R7): mass -> mg, units -> u (meq/mmol kept),
    keeping any ``/kg`` and time parts. None for volume and unrecognised units
    (``dose``, ``mL``) -> the native ``category_unit`` fallback."""
    parsed = parse_unit(raw)
    if parsed is None or parsed["kind"] not in _CANONICAL_QUANTITY:
        return None
    parts = [_CANONICAL_QUANTITY[parsed["kind"]]]
    if parsed["per_kg"]:
        parts.append("kg")
    if parsed["minutes"] is not None:
        parts.append(next(c for c, m in _TIMES.values() if m == parsed["minutes"]))
    return "/".join(parts)


def dose_plan(med: object, unit: object, target: object | None) -> dict:
    """How every row of one (medication, data unit) pair is tokenized.

    ``target_concept`` is the concept a successful conversion emits (the CSV
    measurement name when `target` comes from the CSV); ``native_concept`` is the
    native-unit fallback ``{med}_{unit}``. ``status`` is the pre-weight status:
    ``converted`` / ``passthrough`` / ``unconvertible`` (``no_weight`` is decided per row
    from ``weight_power``)."""
    name = normalize_name(med)
    native = f"{name}_{unit_suffix(unit)}"
    plan = {"native_concept": native, "target_concept": native, "factor": 1.0,
            "weight_power": 0, "status": "passthrough",
            "target_unit": normalize_unit(unit)}
    if target is None or normalize_unit(unit) == normalize_unit(target):
        if target is not None:
            plan["target_concept"] = f"{name}_{unit_suffix(target)}"
        return plan
    conv = conversion(unit, target)
    if conv is None:
        plan["status"] = "unconvertible"
        return plan
    plan.update(target_concept=f"{name}_{unit_suffix(target)}", factor=conv[0],
                weight_power=conv[1], status="converted",
                target_unit=normalize_unit(target))
    return plan


def split_med_measurement(name: str) -> tuple[str, str] | None:
    """``norepinephrine_mcg_kg_min`` -> ``("norepinephrine", "mcg/kg/min")``; None when
    the name does not end in a rate unit."""
    tokens = name.split("_")
    for n in (3, 2):
        if len(tokens) <= n:
            continue
        unit = "/".join(tokens[-n:])
        parsed = parse_unit(unit)
        if parsed is not None and parsed["minutes"] is not None:
            return "_".join(tokens[:-n]), normalize_unit(unit)
    return None


def preferred_units_from_csv(csv_path: str | Path) -> dict[str, str]:
    """``{med_category: preferred unit}`` from the physician CSV's medication rows
    (measurement names aliased per policy step 7, e.g. angiotension -> angiotensin)."""
    units: dict[str, str] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("category") or "").strip() != "medications":
                continue
            split = split_med_measurement(alias_measurement((row.get("measurement") or "").strip()))
            if split is not None:
                units[split[0]] = split[1]
    return units
