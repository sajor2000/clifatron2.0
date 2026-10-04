"""Context-length report: how much pre-anchor history exceeds the trunk's context window.

Aggregate only (shares and totals). A share is withheld (``suppressed``) when its base
has fewer than `MIN_CELL` units, or when the units over (or not over) a threshold number
1-9: the share times the published total would give that small count back. Two views of
one full-hospitalization shard (`gem_events.parquet`):

- ``anchors``: every token position of every stay is a candidate in-stream anchor (the
  training sampler draws anchors along the whole stay), so the share of candidate anchors
  whose history (tokens up to and including the anchor) exceeds N is
  ``sum(max(0, L - N)) / sum(L)`` over stay lengths L. The stay's ICU + 24 h anchor
  (`anchor_idx`) is reported beside it.
- ``prompts``: the extubation study's time-zero prompts. A prompt is the stay's history up
  to time zero (tokens with ``pos_min <= time-zero minute``); time zero is placed on the
  stay clock by the cohort's own admission time (``pos_min`` is minutes since hospital
  admission). Only ELIGIBLE cohort patients are counted; no outcome is read. An eligible
  extubation whose index stay is not in the shard is COUNTED (``missing_from_shard``),
  never silently dropped: tokenize with ``--extubation-cohort`` adds those stays.

A site other than the reference site (mimic) must name its own episode artifact
(``--episodes``), and every artifact must record that site.

    uv run python -m src.eval.context_length --site mimic \
        --shard output/intermediate_phi/mimic/gem_events.parquet \
        --cohort output/intermediate_phi/extubation_cohort.parquet \
        --episodes output/intermediate_phi/episodes.parquet
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

CONTEXT_THRESHOLDS = (4096, 8192, 16384)
MIN_CELL = 10
SUPPRESSED = "suppressed"


def _key(frame: pl.DataFrame) -> str:
    return "episode_key" if "episode_key" in frame.columns else "hosp_id"


def _small(count: int) -> bool:
    return 0 < int(count) < MIN_CELL


def _count(n: int) -> int | str:
    return f"<{MIN_CELL}" if _small(n) else int(n)


def _share(over: int, n: int) -> float | str:
    """``over / n``, withheld when the base is a small cell or when either side of the
    split (over, not over) is a count of 1-9."""
    if n < MIN_CELL or _small(over) or _small(n - over):
        return SUPPRESSED
    return round(over / n, 4)


def _shares(lengths: np.ndarray, thresholds: Sequence[int]) -> dict[str, Any]:
    n = int(lengths.size)
    if n < MIN_CELL:
        return {"n": SUPPRESSED, "share_over": {str(t): SUPPRESSED for t in thresholds}}
    return {"n": n,
            "share_over": {str(t): _share(int((lengths > t).sum()), n) for t in thresholds}}


def shard_context_report(shard: str | Path, *, thresholds: Sequence[int] = CONTEXT_THRESHOLDS,
                         partition: str | None = None) -> dict[str, Any]:
    """`anchors` view (module docstring) of one shard."""
    frame = pl.scan_parquet(shard)
    columns = frame.collect_schema().names()
    key = "episode_key" if "episode_key" in columns else "hosp_id"
    if partition is not None:
        frame = frame.filter(pl.col("partition") == partition)
    stays = (frame.select(key, pl.col("token").list.len().alias("_n"), "anchor_idx")
             .group_by(key).agg(pl.col("_n").sum().alias("length"),
                                pl.col("anchor_idx").first())
             .collect())
    lengths = stays["length"].to_numpy().astype(np.int64)
    total = int(lengths.sum())
    report: dict[str, Any] = {"stays": int(lengths.size) if lengths.size >= MIN_CELL else SUPPRESSED}
    if lengths.size < MIN_CELL or total == 0:
        report["candidate_anchors_share_over"] = {str(t): SUPPRESSED for t in thresholds}
    else:
        # The excess-token share is driven by the stays longer than t: withheld when
        # 1-9 stays (or all but 1-9) are over t.
        report["candidate_anchors_share_over"] = {
            str(t): (SUPPRESSED if _small(int((lengths > t).sum()))
                     or _small(int((lengths <= t).sum()))
                     else round(float(np.maximum(lengths - t, 0).sum() / total), 4))
            for t in thresholds}
    anchors = stays["anchor_idx"].drop_nulls().to_numpy().astype(np.int64) + 1
    report["icu24h_anchor"] = _shares(anchors, thresholds)
    report["stay_length"] = _shares(lengths, thresholds)
    return report


def prompt_lengths(shard: str | Path, cohort: pl.DataFrame,
                   episodes: pl.DataFrame | None = None) -> tuple[np.ndarray, int]:
    """Tokens in each eligible cohort patient's time-zero prompt (module docstring), and
    the number of eligible patients whose index stay is not in the shard."""
    need = {"hospitalization_id", "time_zero_dttm", "eligible", "admission_dttm"}
    missing = need - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort is missing {sorted(missing)}")
    prompts = (cohort.filter(pl.col("eligible").fill_null(False))
               .select("hospitalization_id", "time_zero_dttm", "admission_dttm"))
    if prompts.is_empty():
        return np.zeros(0, dtype=np.int64), 0
    windows = pl.read_parquet(shard, columns=["hosp_id", "continuation_index", "pos_min"])
    ids = prompts["hospitalization_id"].to_list()
    windows = windows.filter(pl.col("hosp_id").is_in(ids))
    present = set(windows["hosp_id"].unique().to_list())
    absent = sum(1 for i in set(ids) if i not in present)
    joined = (windows.join(prompts.rename({"hospitalization_id": "hosp_id"}), on="hosp_id")
              .with_columns((pl.col("time_zero_dttm") - pl.col("admission_dttm"))
                            .dt.total_minutes().alias("_t0")))
    counts = (joined.select("hosp_id", "_t0", pl.col("pos_min"))
              .explode("pos_min")
              .group_by("hosp_id")
              .agg((pl.col("pos_min") <= pl.col("_t0")).sum().alias("length")))
    return counts["length"].to_numpy().astype(np.int64), absent


def prompt_context_report(shard: str | Path, cohort: str | Path | pl.DataFrame,
                          episodes: str | Path | pl.DataFrame | None = None, *,
                          thresholds: Sequence[int] = CONTEXT_THRESHOLDS) -> dict[str, Any]:
    """`prompts` view: share of extubation time-zero prompts over each threshold, and the
    eligible extubations whose index stay is missing from the shard."""
    cohort = cohort if isinstance(cohort, pl.DataFrame) else pl.read_parquet(
        cohort, columns=["hospitalization_id", "time_zero_dttm", "eligible", "admission_dttm"])
    lengths, absent = prompt_lengths(shard, cohort)
    return {**_shares(lengths, thresholds), "missing_from_shard": _count(absent)}


def _site_of(frame: pl.DataFrame) -> str | None:
    if "site" not in frame.columns:
        return None
    sites = frame["site"].drop_nulls().unique().to_list()
    return str(sites[0]) if len(sites) == 1 else None


def check_sites(site: str, *, cohort: pl.DataFrame | None, episodes: pl.DataFrame | None,
                episodes_given: bool) -> None:
    """A non-reference site names its own episode artifact; every artifact read records
    `site` (an artifact predating the `site` column is accepted for the reference site)."""
    from src.data.site_config import require_explicit_episodes, require_site_match

    require_explicit_episodes(site, episodes_given, "context-length report")
    for frame, what in ((cohort, "extubation cohort"), (episodes, "episode artifact")):
        if frame is not None:
            require_site_match(_site_of(frame), site, what)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", default="mimic",
                    help="the shard's site; any site but mimic needs --episodes")
    ap.add_argument("--shard", required=True, help="a site's gem_events.parquet")
    ap.add_argument("--cohort", help="the site's extubation cohort artifact")
    ap.add_argument("--episodes", help="the site's episode artifact")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--thresholds", type=int, nargs="+", default=list(CONTEXT_THRESHOLDS))
    args = ap.parse_args(argv)
    cohort = pl.read_parquet(args.cohort) if args.cohort else None
    episodes = (pl.read_parquet(args.episodes, columns=None) if args.episodes else None)
    try:
        check_sites(args.site, cohort=cohort, episodes=episodes,
                    episodes_given=bool(args.episodes))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    report = {"site": args.site,
              "anchors": shard_context_report(args.shard, thresholds=args.thresholds,
                                              partition=args.partition)}
    if cohort is not None:
        report["prompts"] = prompt_context_report(args.shard, cohort,
                                                  thresholds=args.thresholds)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
