"""Context-length report (aggregate) and the threshold evaluation's --max-context option."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.eval import context_length as cl
from src.eval import threshold_eval as te


def _shard(path, lengths, *, window=4000):
    rows = []
    for s, n in enumerate(lengths):
        pos = list(range(n))                       # one token per minute
        for c, lo in enumerate(range(0, n, window)):
            hi = min(n, lo + window)
            rows.append({"hosp_id": f"h{s}", "continuation_index": c, "partition": "train",
                         "token": [5] * (hi - lo), "pos_min": pos[lo:hi],
                         "anchor_idx": min(n - 1, 1440), "anchor_min": min(n - 1, 1440)})
    pl.DataFrame(rows).write_parquet(path)


def test_shard_report_shares_of_candidate_anchors_and_stays(tmp_path):
    lengths = [1000] * 20 + [10000] * 10 + [20000] * 10
    _shard(tmp_path / "gem.parquet", lengths)
    report = cl.shard_context_report(tmp_path / "gem.parquet")
    assert report["stays"] == 40
    total = sum(lengths)
    expect = {t: sum(max(0, n - t) for n in lengths) / total for t in (4096, 8192, 16384)}
    for t, share in expect.items():
        assert report["candidate_anchors_share_over"][str(t)] == pytest.approx(share, abs=1e-4)
    assert report["stay_length"]["share_over"]["8192"] == pytest.approx(0.5)
    assert report["icu24h_anchor"]["share_over"]["4096"] == 0.0


def test_a_share_that_reveals_a_count_under_ten_is_suppressed(tmp_path):
    # 2 of 40 stays over 8,192 (and 16,384): share x n would give back the count 2.
    lengths = [1000] * 38 + [10000, 20000]
    _shard(tmp_path / "gem.parquet", lengths)
    report = cl.shard_context_report(tmp_path / "gem.parquet")
    assert report["stays"] == 40
    assert report["stay_length"]["share_over"]["8192"] == cl.SUPPRESSED
    assert report["candidate_anchors_share_over"]["8192"] == cl.SUPPRESSED
    assert report["candidate_anchors_share_over"]["16384"] == cl.SUPPRESSED
    # And the complement: 38 of 40 over a threshold would leave 2 not over.
    lengths = [20000] * 38 + [100, 100]
    _shard(tmp_path / "gem2.parquet", lengths)
    report = cl.shard_context_report(tmp_path / "gem2.parquet")
    assert report["stay_length"]["share_over"]["4096"] == cl.SUPPRESSED


def test_prompts_whose_index_stay_is_missing_are_reported_not_dropped(tmp_path):
    _shard(tmp_path / "gem.parquet", [3000] * 10)
    anchor = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    ids = [f"h{i}" for i in range(10)] + [f"absent{i}" for i in range(12)]
    cohort = pl.DataFrame({"hospitalization_id": ids, "eligible": [True] * 22,
                           "admission_dttm": [anchor] * 22,
                           "time_zero_dttm": [anchor + dt.timedelta(minutes=100)] * 22})
    report = cl.prompt_context_report(tmp_path / "gem.parquet", cohort)
    assert report["n"] == 10 and report["missing_from_shard"] == 12
    few = cohort.head(13)                    # 3 missing: a small cell
    assert cl.prompt_context_report(tmp_path / "gem.parquet", few)["missing_from_shard"] == "<10"


def test_a_non_reference_site_needs_its_own_episode_artifact(tmp_path):
    with pytest.raises(SystemExit, match="explicit --episodes"):
        cl.main(["--site", "rush", "--shard", str(tmp_path / "x.parquet")])
    episodes = tmp_path / "episodes.parquet"
    pl.DataFrame({"hospitalization_id": ["h0"], "site": ["mimic"]}).write_parquet(episodes)
    with pytest.raises(SystemExit, match="built for site 'mimic'"):
        cl.main(["--site", "rush", "--shard", str(tmp_path / "x.parquet"),
                 "--episodes", str(episodes)])


def test_small_shards_are_suppressed(tmp_path):
    _shard(tmp_path / "gem.parquet", [100] * 3)
    report = cl.shard_context_report(tmp_path / "gem.parquet")
    assert report["stays"] == cl.SUPPRESSED
    assert set(report["candidate_anchors_share_over"].values()) == {cl.SUPPRESSED}


def test_prompt_report_places_time_zero_on_the_stay_clock(tmp_path):
    lengths = [20000] * 10 + [3000] * 10
    _shard(tmp_path / "gem.parquet", lengths)
    anchor = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    episodes = pl.DataFrame({"hospitalization_id": [f"h{i}" for i in range(20)],
                             "anchor_dttm": [anchor] * 20})
    # Time zero 9,000 minutes after the anchor (anchor_min 1440) for the long stays:
    # 10,441 tokens of history; the short stays end before it (all 3,000 tokens).
    # pos_min counts minutes since admission: admission is the anchor minus 1,440 min.
    cohort = pl.DataFrame({"hospitalization_id": [f"h{i}" for i in range(20)],
                           "admission_dttm": [anchor - dt.timedelta(minutes=1440)] * 20,
                           "time_zero_dttm": [anchor + dt.timedelta(minutes=9000)] * 20,
                           "eligible": [True] * 20,
                           "reintubation_hours": [1.0] * 20})      # never read
    report = cl.prompt_context_report(tmp_path / "gem.parquet", cohort, episodes)
    assert report["n"] == 20 and report["missing_from_shard"] == 0
    assert report["share_over"] == {"4096": 0.5, "8192": 0.5, "16384": 0.0}
    lengths_, absent = cl.prompt_lengths(tmp_path / "gem.parquet", cohort, episodes)
    assert sorted(set(lengths_.tolist())) == [3000, 10441] and absent == 0


def test_max_context_option_defaults_to_the_trunk_and_overrides_it():
    mcfg = {"trunk": {"max_tokens": 8192}}
    assert te.context_tokens(None, mcfg) == 8192
    assert te.context_tokens(4096, mcfg) == 4096
    with pytest.raises(SystemExit):
        te.main(["--run-dir", "x", "--checkpoint", "x", "--vocab", "x", "--shards", "x",
                 "--max-context", "1"])


def test_preflight_context_check_warns_when_many_prompts_exceed_the_context(tmp_path):
    from src.train import preflight as pf

    site_dir = tmp_path / "mimic"
    site_dir.mkdir()
    _shard(site_dir / pf.GEM_EVENTS, [20000] * 10 + [3000] * 10)
    anchor = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
    ids = [f"h{i}" for i in range(20)]
    pl.DataFrame({"hospitalization_id": ids, "anchor_dttm": [anchor] * 20}).write_parquet(
        tmp_path / "episodes.parquet")
    pl.DataFrame({"hospitalization_id": ids, "eligible": [True] * 20,
                  "admission_dttm": [anchor - dt.timedelta(minutes=1440)] * 20,
                  "time_zero_dttm": [anchor + dt.timedelta(minutes=9000)] * 20}).write_parquet(
        tmp_path / "cohort.parquet")
    arm = pf.ArmData("clinical_soft", {"mimic": site_dir}, tmp_path / "stats.json")
    (check,) = pf.context_checks(arm, episodes={"mimic": tmp_path / "episodes.parquet"},
                                 cohorts={"mimic": tmp_path / "cohort.parquet"})
    assert check.status == pf.WARN and "exceed 8192" in check.detail
    (info,) = pf.context_checks(arm, episodes={}, cohorts={})
    assert info.status == pf.PASS and "candidate anchors" in info.detail
