"""Parallel per-stay encoding (`tokenize_site(..., workers=N)`, CLI `--workers`).

The hot path (per-stay encode) runs in a spawn-context process pool over contiguous
chunks of the already-ordered events; a stay is never split across chunks and chunks are
concatenated in order. Hard requirement: byte-identical artifacts for any worker count,
for both trajectories, including more workers than stays. Encode memory is bounded by
a fixed per-chunk event budget (`encode_chunk_events`), in-process and in the pool, and
the artifacts are byte-identical for any budget too. All data is synthetic.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import polars as pl


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ParallelEncodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from test_gem_artifact import _build_site
        except ImportError:  # pragma: no cover
            from tests.test_gem_artifact import _build_site
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import FIXTURE_POLICY, SYNTHETIC_SITE

        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old_cwd = os.getcwd()
        os.chdir(work)
        try:
            site, episodes, cfg = _build_site(work)
            kw = {"episodes": episodes, "artifact_policy": FIXTURE_POLICY}
            cls.outs = {}
            # (workers, encode_chunk_events): the default budget, a tiny one (many
            # chunks, several per stay boundary run) and a huge one (a single chunk).
            for workers, budget in ((1, None), (3, None), (1, 7), (1, 10**9), (3, 7)):
                out = work / f"output/intermediate_phi/w{workers}_b{budget}"
                extra = {} if budget is None else {"encode_chunk_events": budget}
                tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, workers=workers,
                              **extra, **kw)
                blob = json.loads((out / "vocab.json").read_text())
                tokenize_site(cfg, SYNTHETIC_SITE, site, out, blob, workers=workers,
                              trajectory="hospitalization", max_tokens=64, **extra, **kw)
                cls.outs[(workers, budget)] = out
            # More workers than stays (a 2-stay sample).
            small = work / "output/intermediate_phi/w_many"
            tokenize_site(cfg, SYNTHETIC_SITE, site, small, None, workers=16,
                          sample_episodes=2, **kw)
            small_one = work / "output/intermediate_phi/w_many_one"
            tokenize_site(cfg, SYNTHETIC_SITE, site, small_one, None, workers=1,
                          sample_episodes=2, **kw)
            cls.small, cls.small_one = small, small_one
        finally:
            os.chdir(old_cwd)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_icu_24h_artifacts_are_byte_identical_across_worker_counts(self):
        for name in ("events.parquet", "vocab.json", "tokenization_report.json"):
            self.assertEqual(_sha(self.outs[(1, None)] / name),
                             _sha(self.outs[(3, None)] / name), name)

    def test_gem_artifacts_are_byte_identical_across_worker_counts(self):
        for name in ("gem_events.parquet", "gem_tokenization_report.json"):
            self.assertEqual(_sha(self.outs[(1, None)] / name),
                             _sha(self.outs[(3, None)] / name), name)

    def test_artifacts_are_byte_identical_across_chunk_budgets(self):
        """Tiny budget (many chunks) vs huge budget (one chunk) vs 3 workers."""
        reference = self.outs[(1, None)]
        for key in ((1, 7), (1, 10**9), (3, 7)):
            for name in ("events.parquet", "gem_events.parquet", "vocab.json",
                         "tokenization_report.json", "gem_tokenization_report.json"):
                self.assertEqual(_sha(reference / name), _sha(self.outs[key] / name),
                                 (key, name))

    def test_more_workers_than_stays(self):
        events = pl.read_parquet(self.small / "events.parquet")
        self.assertEqual(events["hosp_id"].n_unique(), 2)
        self.assertEqual(_sha(self.small / "events.parquet"),
                         _sha(self.small_one / "events.parquet"))

    def test_stays_are_never_split_and_order_is_preserved(self):
        events = pl.read_parquet(self.outs[(3, 7)] / "events.parquet")
        ids = events["hosp_id"].to_list()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))


def _toy_events(sizes: dict[str, int]) -> pl.DataFrame:
    """Ordered, stay-contiguous encode input: `sizes` = {stay: event count}."""
    rows = [(stay, "hr", float(60 + k), None, k, True, "train")
            for stay, n in sizes.items() for k in range(n)]
    return pl.DataFrame(rows, orient="row", schema={
        "hosp_id": pl.String, "concept": pl.String, "value": pl.Float64,
        "cat_value": pl.String, "pos_min": pl.Int64, "target_eligible": pl.Boolean,
        "partition": pl.String})


BIN_CFG = {"soft_discretization": False, "soft_kernel_bins": 1}


class ChunkingTest(unittest.TestCase):
    def test_chunks_cut_only_at_stay_boundaries_within_the_event_budget(self):
        from src.data.tokenize import _stay_chunks

        frame = pl.DataFrame({"hosp_id": ["a"] * 5 + ["b"] * 1 + ["c"] * 7 + ["d"] * 2})
        for budget in (1, 2, 6, 7, 8, 15, 10**9):
            bounds = _stay_chunks(frame, budget)
            self.assertEqual(bounds[0][0], 0)
            self.assertEqual(bounds[-1][1], len(frame))
            for (lo, hi), (lo2, _) in zip(bounds, bounds[1:]):
                self.assertEqual(hi, lo2)
                self.assertNotEqual(frame["hosp_id"][hi - 1], frame["hosp_id"][hi])
            self.assertTrue(all(hi > lo for lo, hi in bounds))
            # Over budget only when a single stay is larger than the budget.
            for lo, hi in bounds:
                if hi - lo > budget:
                    self.assertEqual(frame["hosp_id"][lo:hi].n_unique(), 1)
        self.assertEqual(_stay_chunks(frame, 10**9), [(0, len(frame))])

    def test_in_process_encode_runs_chunk_by_chunk_within_the_budget(self):
        """workers=1 must not encode the whole frame as one Python-object batch."""
        from unittest import mock

        from src.data import tokenize as T

        events = _toy_events({f"s{i:02d}": 4 + i % 5 for i in range(30)})
        seen: list[int] = []
        real = T._encode_events

        def spy(frame, *args):
            seen.append(len(frame))
            return real(frame, *args)

        with mock.patch.object(T, "_encode_events", spy):
            chunked, unk = T._parallel_encode(events, {}, {}, BIN_CFG, False, 1,
                                              chunk_events=20)
        self.assertGreater(len(seen), 1)
        self.assertLessEqual(max(seen), 20)
        self.assertEqual(sum(seen), len(events))
        whole, whole_unk = real(events, {}, {}, BIN_CFG, False)
        self.assertTrue(chunked.equals(whole))
        self.assertEqual(unk, dict(sorted(whole_unk.items())))

    def test_a_worker_exception_propagates(self):
        """A chunk that fails inside a spawn-context worker raises in the parent."""
        from src.data.tokenize import _parallel_encode

        events = _toy_events({f"s{i:02d}": 3 for i in range(8)})
        broken = {"soft_discretization": True}       # soft_kernel_bins missing
        with self.assertRaises(KeyError):
            _parallel_encode(events, {}, {"hr": [
                {"lo": None, "hi": None, "lo_closed": False, "hi_closed": False}]},
                broken, False, 2, chunk_events=6)

    def test_a_failed_encode_leaves_no_artifacts_and_invalid_workers_fail_fast(self):
        try:
            from test_gem_artifact import _build_site
        except ImportError:  # pragma: no cover
            from tests.test_gem_artifact import _build_site
        import copy
        from unittest import mock

        from src.data import tokenize as T
        from src.eval.synthetic_bundle import FIXTURE_POLICY, SYNTHETIC_SITE

        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            old_cwd = os.getcwd()
            os.chdir(work)
            try:
                site, episodes, cfg = _build_site(work)
                kw = {"episodes": episodes, "artifact_policy": FIXTURE_POLICY}
                out = work / "output/intermediate_phi/failed"
                broken = copy.deepcopy(cfg)
                broken["value_binning"]["soft_discretization"] = True
                broken["value_binning"].pop("soft_kernel_bins", None)
                # The pool path: the worker's KeyError reaches the caller.
                with self.assertRaises(KeyError):
                    T.tokenize_site(broken, SYNTHETIC_SITE, site, out, None, workers=2,
                                    encode_chunk_events=50, **kw)
                self.assertFalse(out.exists() and any(out.iterdir()))
                # Fail fast: an invalid worker count or budget never reads a table.
                with mock.patch.object(T, "_read_source") as read:
                    for bad in ({"workers": -1}, {"encode_chunk_events": 0}):
                        with self.assertRaises(ValueError):
                            T.tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, **bad, **kw)
                    read.assert_not_called()
            finally:
                os.chdir(old_cwd)

    def test_worker_count_resolution(self):
        from src.data.tokenize import resolve_workers

        self.assertEqual(resolve_workers(1), 1)
        self.assertEqual(resolve_workers(0), os.cpu_count() or 1)
        with self.assertRaises(ValueError):
            resolve_workers(-2)

    def test_cli_flag(self):
        from src.data.tokenize import build_arg_parser

        base = ["--site", "s", "--in", "i", "--out", "o", "--episodes", "e"]
        self.assertEqual(build_arg_parser().parse_args(base).workers, 1)
        self.assertEqual(build_arg_parser().parse_args([*base, "--workers", "0"]).workers, 0)
        args = build_arg_parser().parse_args([*base, "--encode-chunk-events", "500"])
        self.assertEqual(args.encode_chunk_events, 500)
