"""Per-site profile, site-local declarations and timestamp ingest (Rush onboarding).

Three things a second development site needs that one reference site never did:

- A per-site PROFILE in configs/data.yaml `sites.<site>`: `site_timezone` (IANA name; the
  clock a date-only or naive timestamp was charted on), `extraction_dttm` (ISO-8601 UTC;
  the data cut, at which a stay still in hospital - a null discharge time - is censored)
  and `expected_absent_tables` (configured tables the site does not have, e.g. Rush has no
  ECMO/MCS table: skipped quietly and reported; any OTHER absent table fails the build).
- SITE-LOCAL declarations: a site's mappings can name its own source strings (BP source
  names, local unit labels), which must not be committed. They live in the git-ignored
  ``configs/sites/<site>.local.yaml`` (template: ``configs/sites/rush.example.yaml``) and
  are merged at load time into ``site_harmonization.<site>``, ``site_unit_conversions.<site>``
  and ``sites.<site>`` (`with_site_local`). What was merged is hashed into the site's
  shard binding (`tokenize.site_binding`), so a shard records the declarations it was
  built with.
- TIMESTAMP INGEST (`to_utc`): CLIF 2.1 timestamps are UTC. A naive column is read as the
  site's local wall clock and converted (``replace_time_zone(tz, ambiguous="earliest",
  non_existent="null").convert_time_zone("UTC")``); a column in another zone is
  converted; a UTC column is unchanged. Wall-clock times that do not exist (the spring
  DST gap) become null and are COUNTED.

Free-text `note` fields (and the reasons of declared exceptions) are documentation:
`strip_notes` removes them from every hashed record, so editing a note never invalidates a
vocabulary or a shard.

Aggregate-only: nothing here reads or prints a row.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

ROOT = Path(__file__).parents[2]
SITE_CONFIG_DIR = "configs/sites"
SITE_CONFIG_ENV = "CLIF_SITE_CONFIG_DIR"
LOCAL_SUFFIX = ".local.yaml"
# Sections of a site-local file and where each one is merged.
LOCAL_SECTIONS = ("profile", "site_harmonization", "site_unit_conversions")
PROFILE_KEYS = ("site_timezone", "extraction_dttm", "expected_absent_tables")
NOTE_KEYS = ("note", "notes")
# The one site whose artifacts may be built from defaults (the reference site, staged on
# the node first). Every other site must name its episode artifact explicitly.
REFERENCE_SITE = "mimic"


class SiteConfigError(ValueError):
    """A site profile or site-local declaration is malformed."""


def _sha256(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def strip_notes(obj: Any) -> Any:
    """`obj` without free-text documentation: every ``note`` / ``notes`` key removed, and a
    ``declared_exceptions`` mapping (value -> reason) reduced to its sorted values."""
    if isinstance(obj, Mapping):
        out = {}
        for key, value in obj.items():
            if key in NOTE_KEYS:
                continue
            if key == "declared_exceptions" and isinstance(value, Mapping):
                out[key] = {k: sorted(str(v) for v in (inner or {}))
                            for k, inner in value.items()} if all(
                    isinstance(inner, Mapping) for inner in value.values()) else sorted(value)
                continue
            out[key] = strip_notes(value)
        return out
    if isinstance(obj, list):
        return [strip_notes(v) for v in obj]
    return obj


def local_config_dir() -> Path:
    override = os.environ.get(SITE_CONFIG_ENV)
    return Path(override) if override else ROOT / SITE_CONFIG_DIR


def local_config_path(site: str) -> Path:
    return local_config_dir() / f"{site}{LOCAL_SUFFIX}"


def _merge(base: Any, extra: Any) -> Any:
    """Deep merge: mappings merged key by key, anything else replaced by `extra`."""
    if isinstance(base, Mapping) and isinstance(extra, Mapping):
        out = dict(base)
        for key, value in extra.items():
            out[key] = _merge(base.get(key), value) if key in base else copy.deepcopy(value)
        return out
    return copy.deepcopy(extra)


def with_site_local(cfg: Mapping, site: str) -> dict:
    """`cfg` with ``configs/sites/<site>.local.yaml`` merged in (a copy; `cfg` unchanged).

    The file's `profile` is merged into ``sites.<site>``, `site_harmonization` into
    ``site_harmonization.<site>`` and `site_unit_conversions` into
    ``site_unit_conversions.<site>`` (local keys win). Idempotent: a config already merged
    for `site` is returned as is. No file -> `cfg` (copied) unchanged."""
    merged = set((cfg.get("_site_local_merged") or {}))
    out = copy.deepcopy(dict(cfg))
    if site in merged:
        return out
    path = local_config_path(site)
    record = dict(cfg.get("_site_local_merged") or {})
    if path.is_file():
        local = yaml.safe_load(path.read_text()) or {}
        if not isinstance(local, Mapping):
            raise SiteConfigError(f"{path.name} must be a mapping")
        if local.get("site") != site:
            raise SiteConfigError(f"{path.name} declares site {local.get('site')!r}, not {site!r}")
        unknown = sorted(set(local) - {"site", *LOCAL_SECTIONS})
        if unknown:
            raise SiteConfigError(f"{path.name}: unknown section(s) {unknown}; expected "
                                  f"{', '.join(LOCAL_SECTIONS)}")
        targets = {"profile": "sites", "site_harmonization": "site_harmonization",
                   "site_unit_conversions": "site_unit_conversions"}
        for section, key in targets.items():
            block = local.get(section)
            if block is None:
                continue
            if not isinstance(block, Mapping):
                raise SiteConfigError(f"{path.name}: {section} must be a mapping")
            every = dict(out.get(key) or {})
            every[site] = _merge(every.get(site) or {}, block)
            out[key] = every
        record[site] = _sha256(local)
    else:
        record[site] = None
    out["_site_local_merged"] = record
    site_profile(out, site)          # validate what was merged
    return out


def _timezone(name: object, where: str) -> str | None:
    if name is None:
        return None
    if not isinstance(name, str) or not name.strip():
        raise SiteConfigError(f"{where}.site_timezone must be an IANA time-zone name")
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise SiteConfigError(f"{where}.site_timezone {name!r} is not an IANA time zone") from exc
    return name


def parse_extraction(value: object, where: str = "sites") -> dt.datetime | None:
    """`extraction_dttm` as an aware UTC datetime (an ISO string with an offset or ``Z``;
    a naive string is refused: the data cut must name its clock)."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        stamp = value
    elif isinstance(value, str):
        try:
            stamp = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise SiteConfigError(f"{where}.extraction_dttm {value!r} is not ISO-8601") from exc
    else:
        raise SiteConfigError(f"{where}.extraction_dttm must be an ISO-8601 string")
    if stamp.tzinfo is None:
        raise SiteConfigError(f"{where}.extraction_dttm must carry a UTC offset (e.g. 'Z')")
    return stamp.astimezone(dt.UTC)


def site_profile(cfg: Mapping, site: str) -> dict:
    """The validated profile of `site` (configs/data.yaml `sites.<site>`, after the local
    merge): ``{site_timezone: str|None, extraction_dttm: datetime|None,
    expected_absent_tables: [config table names]}``. An undeclared site gets the defaults
    (no zone, no extraction time, no expected absence)."""
    every = cfg.get("sites") or {}
    if not isinstance(every, Mapping):
        raise SiteConfigError("data config `sites` must map site -> profile")
    raw = every.get(site) or {}
    where = f"sites.{site}"
    if not isinstance(raw, Mapping):
        raise SiteConfigError(f"{where} must be a mapping")
    unknown = sorted(set(raw) - set(PROFILE_KEYS) - set(NOTE_KEYS))
    if unknown:
        raise SiteConfigError(f"{where}: unknown key(s) {unknown}; expected {PROFILE_KEYS}")
    tables = cfg.get("tables") or {}
    by_file = {}
    for name, spec in tables.items():
        stem = str((spec or {}).get("file", ""))
        by_file[stem[5:] if stem.startswith("clif_") else stem] = name
    absent = []
    entries = raw.get("expected_absent_tables") or ()
    if isinstance(entries, str) or not all(isinstance(e, str) for e in entries):
        raise SiteConfigError(f"{where}.expected_absent_tables must be a list of table names")
    for entry in entries:
        # A configured table (data.yaml key) or its CLIF table name. A name this config
        # does not configure declares nothing (a misspelt name therefore cannot excuse a
        # missing table: the build still fails on it).
        name = entry if entry in tables else by_file.get(str(entry))
        if name is not None:
            absent.append(name)
    return {"site_timezone": _timezone(raw.get("site_timezone"), where),
            "extraction_dttm": parse_extraction(raw.get("extraction_dttm"), where),
            "expected_absent_tables": sorted(set(absent))}


def profile_record(cfg: Mapping, site: str) -> dict:
    """The profile as JSON-ready values (hashed into the site binding)."""
    profile = site_profile(cfg, site)
    stamp = profile["extraction_dttm"]
    return {**profile, "extraction_dttm": None if stamp is None else stamp.isoformat()}


def to_utc(frame, columns, tz: str | None):
    """``(frame, nulled)``: every listed Datetime column of `frame` in UTC (module
    docstring). `nulled` counts values that were non-null before and null after (a
    wall-clock time inside a DST gap). A naive column with no declared `tz` is refused."""
    import polars as pl

    nulled = 0
    for column in columns:
        if column not in frame.columns:
            continue
        dtype = frame.schema[column]
        if not isinstance(dtype, pl.Datetime):
            continue
        if dtype.time_zone == "UTC":
            continue
        if dtype.time_zone is None:
            if tz is None:
                raise SiteConfigError(
                    f"column {column!r} holds naive timestamps and the site declares no "
                    "site_timezone (configs/data.yaml sites.<site>.site_timezone or the "
                    "site-local profile)")
            converted = (pl.col(column).dt.replace_time_zone(tz, ambiguous="earliest",
                                                             non_existent="null")
                         .dt.convert_time_zone("UTC"))
        else:
            converted = pl.col(column).dt.convert_time_zone("UTC")
        before = int(frame[column].is_not_null().sum())
        frame = frame.with_columns(converted.alias(column))
        nulled += before - int(frame[column].is_not_null().sum())
    return frame, nulled


def censor_open_stays(frame, extraction: dt.datetime | None, column: str = "discharge_dttm"):
    """``(frame, n)``: a null `column` (a patient still in hospital at the data cut) set to
    the site's `extraction_dttm`; `n` stays censored. Without a declared extraction time
    the frame is unchanged (and `n` counts the open stays)."""
    import polars as pl

    if column not in frame.columns:
        return frame, 0
    n = int(frame[column].is_null().sum())
    if not n or extraction is None:
        return frame, n
    return frame.with_columns(pl.col(column).fill_null(
        pl.lit(extraction).cast(frame.schema[column]))), n


def require_explicit_episodes(site: str, episodes: object, what: str) -> None:
    """A site other than the reference site never falls back to the reference site's
    default episode artifact: refuse unless `episodes` was given."""
    if site != REFERENCE_SITE and not episodes:
        raise SiteConfigError(
            f"{what}: site {site!r} needs an explicit --episodes (its own episode artifact); "
            f"the default artifact is the {REFERENCE_SITE} site's")


def episode_site(episodes) -> str | None:
    """The site an episode artifact was built for (its `site` column), or None for an
    artifact that predates the column."""
    if "site" not in episodes.columns:
        return None
    sites = episodes["site"].drop_nulls().unique().to_list()
    if len(sites) != 1:
        raise SiteConfigError("episode artifact names more than one site")
    return str(sites[0])


def require_site_match(recorded: str | None, site: str, what: str) -> None:
    """Refuse an artifact recorded for another site. An artifact that predates the `site`
    column records none and is accepted (every artifact built now records its site, and
    the `--data` provenance checks still bind it to its source tables)."""
    if recorded is None:
        return
    if recorded != site:
        raise SiteConfigError(f"{what} was built for site {recorded!r}, not {site!r}")
