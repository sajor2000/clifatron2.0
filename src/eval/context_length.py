"""Context-length report: how much pre-anchor history exceeds the trunk's context window.

Aggregate only (shares and totals; a share over fewer than `MIN_CELL` units is
suppressed). Two views of one full-hospitalization shard (`gem_events.parquet`):

- ``anchors``: every token position of every stay is a candidate in-stream anchor (the
  training sampler draws anchors along the whole stay), so the share of candidate anchors
  whose history (tokens up to and including the anchor) exceeds N is
  ``sum(max(0, L - N)) / sum(L)`` over stay lengths L. The stay's ICU + 24 h anchor
  (`anchor_idx`) is reported beside it.
- ``prompts``: the extubation study's time-zero prompts. A prompt is the stay's history up
  to time zero (tokens with ``pos_min <= time-zero minute``); time zero is placed on the
  stay clock through the episode artifact's anchor (``anchor_dttm`` is ``anchor_min``
  minutes after admission). Only ELIGIBLE cohort patients are counted; no outcome is read.

    uv run python -m src.eval.context_length --shard output/intermediate_phi/mimic/gem_events.parquet \
        --cohort output/intermediate_phi/extubation_cohort.parquet --episodes output/intermediate_phi/episodes.parquet
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


def _shares(lengths: np.ndarray, thresholds: Sequence[int]) -> dict[str, Any]:
    n = int(lengths.size)
    if n < MIN_CELL:
        return {"n": SUPPRESSED, "share_over": {str(t): SUPPRESSED for t in thresholds}}
    return {"n": n, "share_over": {str(t): round(float(np.mean(lengths > t)), 4) for t in thresholds}}


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
        report["candidate_anchors_share_over"] = {
            str(t): round(float(np.maximum(lengths - t, 0).sum() / total), 4) for t in thresholds}
    anchors = stays["anchor_idx"].drop_nulls().to_numpy().astype(np.int64) + 1
    report["icu24h_anchor"] = _shares(anchors, thresholds)
    report["stay_length"] = _shares(lengths, thresholds)
    return report


def prompt_lengths(shard: str | Path, cohort: pl.DataFrame, episodes: pl.DataFrame) -> np.ndarray:
    """Tokens in each eligible cohort patient's time-zero prompt (module docstring)."""
    for frame, need, what in ((cohort, {"hospitalization_id", "time_zero_dttm", "eligible"}, "cohort"),
                              (episodes, {"hospitalization_id", "anchor_dttm"}, "episodes")):
        missing = need - set(frame.columns)
        if missing:
            raise ValueError(f"{what} is missing {sorted(missing)}")
    prompts = (cohort.filter(pl.col("eligible").fill_null(False))
               .select("hospitalization_id", "time_zero_dttm")
               .join(episodes.select("hospitalization_id", "anchor_dttm"), on="hospitalization_id",
                     how="inner"))
    if prompts.is_empty():
        return np.zeros(0, dtype=np.int64)
    windows = pl.read_parquet(shard, columns=["hosp_id", "continuation_index", "pos_min", "anchor_min"])
    windows = windows.filter(pl.col("hosp_id").is_in(prompts["hospitalization_id"].to_list()))
    joined = (windows.join(prompts.rename({"hospitalization_id": "hosp_id"}), on="hosp_id")
              .with_columns(
                  (pl.col("anchor_min")
                   + (pl.col("time_zero_dttm") - pl.col("anchor_dttm")).dt.total_minutes())
                  .alias("_t0")))
    counts = (joined.select("hosp_id", "_t0", pl.col("pos_min"))
              .explode("pos_min")
              .group_by("hosp_id")
              .agg((pl.col("pos_min") <= pl.col("_t0")).sum().alias("length")))
    return counts["length"].to_numpy().astype(np.int64)


def prompt_context_report(shard: str | Path, cohort: str | Path | pl.DataFrame,
                          episodes: str | Path | pl.DataFrame, *,
                          thresholds: Sequence[int] = CONTEXT_THRESHOLDS) -> dict[str, Any]:
    """`prompts` view: share of extubation time-zero prompts over each threshold."""
    cohort = cohort if isinstance(cohort, pl.DataFrame) else pl.read_parquet(
        cohort, columns=["hospitalization_id", "time_zero_dttm", "eligible"])
    episodes = episodes if isinstance(episodes, pl.DataFrame) else pl.read_parquet(
        episodes, columns=["hospitalization_id", "anchor_dttm"])
    return _shares(prompt_lengths(shard, cohort, episodes), thresholds)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", required=True, help="a site's gem_events.parquet")
    ap.add_argument("--cohort", help="the site's extubation cohort artifact")
    ap.add_argument("--episodes", help="the site's episode artifact (needed with --cohort)")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--thresholds", type=int, nargs="+", default=list(CONTEXT_THRESHOLDS))
    args = ap.parse_args(argv)
    report = {"anchors": shard_context_report(args.shard, thresholds=args.thresholds,
                                              partition=args.partition)}
    if args.cohort:
        if not args.episodes:
            raise SystemExit("--cohort needs --episodes (time zero is placed on the stay clock)")
        report["prompts"] = prompt_context_report(args.shard, args.cohort, args.episodes,
                                                  thresholds=args.thresholds)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
