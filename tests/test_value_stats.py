"""Per-token value-head normalization stats (src/data/value_stats.py).

Locks in the fix for the unnormalized value-head loss (val≈46000 on real MIMIC):
per-token standardization must collapse wildly different raw magnitudes (creatinine ~1,
platelets ~2e5) to ~N(0,1) so the Gaussian NLL is well-scaled and TargetBuilder accepts
every numeric token.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from src.data.value_stats import (
    compute_value_stats,
    compute_value_stats_from_events,
    load_value_stats,
    vocab_hash,
    write_value_stats,
)


def _multi_magnitude_data(n=500, seed=0):
    """3 concepts spanning ~5 orders of magnitude — the real failure mode."""
    rng = np.random.default_rng(seed)
    tokens, values = [], []
    for _ in range(n):
        tokens.append([10, 20, 30])
        values.append([
            float(rng.normal(1.2, 0.5)),          # creatinine ~1
            float(rng.normal(200000, 60000)),      # platelets ~2e5
            float(abs(rng.normal(2.0, 1.5))),      # lactate ~2
        ])
    return tokens, values


class ValueStatsTest(unittest.TestCase):
    def test_per_token_center_and_scale_recovered(self):
        tokens, values = _multi_magnitude_data()
        stats = compute_value_stats(tokens, values, min_count=20)
        self.assertEqual(set(stats), {10, 20, 30})
        # centers land near each concept's true center, across 5 orders of magnitude
        self.assertAlmostEqual(stats[10][0], 1.2, delta=0.3)
        self.assertAlmostEqual(stats[20][0], 200000, delta=20000)
        # every scale is strictly positive (TargetBuilder rejects non-positive)
        for _, scale in stats.values():
            self.assertGreater(scale, 0.0)

    def test_standardization_collapses_magnitude_to_order_one(self):
        """The load-bearing property: standardized squared error is O(1), not O(1e9)."""
        tokens, values = _multi_magnitude_data()
        stats = compute_value_stats(tokens, values, min_count=20)
        raw_sq, std_sq = [], []
        for toks, vals in zip(tokens, values):
            for tok, val in zip(toks, vals):
                raw_sq.append(val ** 2)
                c, s = stats[tok]
                std_sq.append(((val - c) / s) ** 2)
        self.assertGreater(np.mean(raw_sq), 1e6)      # unnormalized NLL driver is huge
        self.assertLess(np.mean(std_sq), 3.0)          # standardized NLL driver is O(1)

    def test_event_stats_fit_training_partition_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.parquet"
            pl.DataFrame({
                "token": [[10], [10], [10]],
                "value": [[1.0], [3.0], [999999.0]],
                "partition": ["train", "train", "internal_test"],
            }).write_parquet(path)

            stats = compute_value_stats_from_events(path, min_count=1)

        self.assertEqual(stats[10][0], 2.0)
        self.assertLess(stats[10][1], 10.0)

    def test_rare_tokens_still_get_stats_coverage_contract(self):
        """Rare numeric tokens must NOT be dropped — TargetBuilder aborts without them.
        min_count only widens the fallback scale; every numeric token gets an entry."""
        # token 10 appears 30x; token 99 appears only 3x (well under min_count)
        tokens = [[10, 99]] * 3 + [[10]] * 27
        values = [[1.0, 5.0]] * 3 + [[1.1]] * 27
        stats = compute_value_stats(tokens, values, min_count=20)
        self.assertIn(10, stats)
        self.assertIn(99, stats)                 # rare, but still covered
        self.assertGreater(stats[99][1], 0.0)    # positive fallback scale

    def test_categorical_tokens_without_values_are_omitted(self):
        tokens = [[10, 40, 40]] * 30
        values = [[1.0, None, None]] * 30   # token 40 is categorical (no numeric value)
        stats = compute_value_stats(tokens, values, min_count=20)
        self.assertIn(10, stats)
        self.assertNotIn(40, stats)

    def test_constant_valued_token_gets_positive_scale(self):
        tokens = [[10]] * 30
        values = [[7.0]] * 30            # zero variance → IQR and std both 0
        stats = compute_value_stats(tokens, values, min_count=20)
        self.assertIn(10, stats)
        self.assertGreater(stats[10][1], 0.0)  # never a zero/negative scale

    def test_nonfinite_values_ignored(self):
        tokens = [[10, 10, 10]] * 30
        values = [[1.0, float("nan"), float("inf")]] * 30
        stats = compute_value_stats(tokens, values, min_count=20)
        self.assertIn(10, stats)
        # only the finite 1.0 observations contribute → center ~1.0
        self.assertAlmostEqual(stats[10][0], 1.0, delta=1e-6)

    def test_robust_vs_mean_std(self):
        rng = np.random.default_rng(1)
        base = list(rng.normal(0, 1, 200))
        outliers = [1000.0] * 10          # heavy contamination
        vals = base + outliers
        tokens = [[10]] * len(vals)
        values = [[v] for v in vals]
        robust = compute_value_stats(tokens, values, min_count=20, robust=True)
        naive = compute_value_stats(tokens, values, min_count=20, robust=False)
        # robust scale resists the outliers; mean/std is inflated by them
        self.assertLess(robust[10][1], naive[10][1])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_value_stats([[1, 2]], [[1.0]], min_count=1)

    def test_json_round_trip(self):
        tokens, values = _multi_magnitude_data(n=100)
        stats = compute_value_stats(tokens, values, min_count=20)
        with tempfile.TemporaryDirectory() as d:
            path = write_value_stats(stats, Path(d) / "value_stats.json")
            reloaded = load_value_stats(path)
            self.assertEqual(set(reloaded), set(stats))
            for tok in stats:
                self.assertAlmostEqual(reloaded[tok][0], stats[tok][0], places=6)
                self.assertAlmostEqual(reloaded[tok][1], stats[tok][1], places=6)

    def test_legacy_bare_map_still_loads(self):
        """Back-compat: a legacy {token_id: [center, scale]} file loads without identity."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "legacy.json"
            p.write_text(json.dumps({"10": [1.2, 0.5], "20": [200.0, 60.0]}))
            reloaded = load_value_stats(p)
            self.assertEqual(reloaded[10], (1.2, 0.5))

    def test_legacy_bare_map_rejected_when_vocab_hash_expected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "legacy.json"
            p.write_text(json.dumps({"10": [1.2, 0.5]}))
            with self.assertRaisesRegex(ValueError, "legacy value-stats map cannot be verified"):
                load_value_stats(p, expected_vocab_hash="abcd" * 16)

    def test_unbound_schema2_rejected_when_vocab_hash_expected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "unbound.json"
            p.write_text(json.dumps({
                "schema": 2,
                "vocab_hash": None,
                "stats": {"10": [1.2, 0.5]},
            }))
            with self.assertRaisesRegex(ValueError, "unbound"):
                load_value_stats(p, expected_vocab_hash="abcd" * 16)

    def test_vocab_hash_binding_and_mismatch_detection(self):
        tokens, values = _multi_magnitude_data(n=100)
        stats = compute_value_stats(tokens, values, min_count=20)
        vocab = {"map=0": 10, "platelets=0": 20, "lactate=0": 30}
        vsha = vocab_hash(vocab)
        with tempfile.TemporaryDirectory() as d:
            p = write_value_stats(stats, Path(d) / "vs.json", vocab=vocab)
            # correct hash loads fine
            ok = load_value_stats(p, expected_vocab_hash=vsha)
            self.assertEqual(set(ok), set(stats))
            # a different vocab's hash is rejected (stale / cross-vocabulary artifact)
            with self.assertRaises(ValueError):
                load_value_stats(p, expected_vocab_hash="deadbeef" * 8)

    def test_fit_partition_provenance_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_value_stats(
                {10: (1.0, 0.5)},
                Path(directory) / "value_stats.json",
                fit_partition_name="train",
            )
            self.assertEqual(
                load_value_stats(path, expected_fit_partition="train"),
                {10: (1.0, 0.5)},
            )
            with self.assertRaisesRegex(ValueError, "fit partition mismatch"):
                load_value_stats(path, expected_fit_partition="calibration")

    def test_stats_accepted_by_target_builder(self):
        """Frozen stats must satisfy TargetBuilder's value_stats contract end to end."""
        from src.data.targets import TargetBuilder

        tokens, values = _multi_magnitude_data(n=100)
        stats = compute_value_stats(tokens, values, min_count=20)
        vocab = max(max(t) for t in tokens) + 1
        # TargetBuilder.__post_init__ validates every (token, scale); must not raise.
        tb = TargetBuilder(
            vocab_size=vocab, n_time_bins=16, horizon_hours=48.0, value_stats=stats
        )
        self.assertIsInstance(tb, TargetBuilder)


def _legacy_stats_from_events(path, *, partition="train", min_count=20, robust=True):
    """The pre-streaming implementation (whole shard -> Python lists), the reference the
    streamed result must match byte for byte."""
    from src.data.splits import fit_partition

    df = fit_partition(pl.read_parquet(path), partition)
    return compute_value_stats(df["token"].to_list(), df["value"].to_list(),
                               min_count=min_count, robust=robust)


def _synthetic_shard(path, *, stays=3000, seed=11, row_group_size=257):
    """A gem-shaped shard: varied magnitudes, rare (< min_count) and constant tokens,
    categorical tokens, nulls, NaN / inf, empty stays, three partitions, many row groups
    and chunks (sizes not multiples of each other)."""
    rng = np.random.default_rng(seed)
    centers = {t: 10.0 ** rng.uniform(-2, 5) for t in range(5, 60)}
    tokens, values, parts = [], [], []
    for i in range(stays):
        n = int(rng.integers(0, 40))
        toks = rng.integers(1, 64, size=n).tolist()
        vals = []
        for t in toks:
            r = rng.random()
            if t < 5 or r < 0.1:
                vals.append(None)                       # categorical / missing
            elif r < 0.12:
                vals.append(float(rng.choice([np.nan, np.inf, -np.inf])))
            elif t == 60:
                vals.append(7.0)                        # constant token
            elif t >= 61:
                vals.append(float(rng.integers(0, 5)))  # spiky discrete (degenerate IQR)
            else:
                vals.append(float(rng.normal(centers[t], centers[t] / 3)))
        tokens.append(toks)
        values.append(vals)
        parts.append(["train", "validation", "internal_test"][i % 7 % 3])
    # A rare token seen only a few times in train.
    tokens[0] = tokens[0] + [63, 63]
    values[0] = values[0] + [1.5, 2.5]
    pl.DataFrame({"hosp_id": [str(i) for i in range(stays)], "token": tokens,
                  "value": values, "partition": parts},
                 schema_overrides={"value": pl.List(pl.Float64)}).write_parquet(
        path, row_group_size=row_group_size)
    return path


class StreamingValueStatsTest(unittest.TestCase):
    """`compute_value_stats_from_events` streams the shard; its artifact is byte-identical
    to the whole-shard implementation."""

    def test_streamed_artifact_is_byte_identical_to_the_whole_shard_one(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = _synthetic_shard(Path(directory) / "gem_events.parquet")
            for robust in (True, False):
                for chunk_rows in (1, 97, 4096):
                    with self.subTest(robust=robust, chunk_rows=chunk_rows):
                        streamed = compute_value_stats_from_events(
                            shard, robust=robust, chunk_rows=chunk_rows)
                        legacy = _legacy_stats_from_events(shard, robust=robust)
                        self.assertEqual(streamed, legacy)
                        a = write_value_stats(streamed, Path(directory) / "a.json",
                                              vocab_sha="v", segments_sha="s",
                                              fit_partition_name="train", robust=robust)
                        b = write_value_stats(legacy, Path(directory) / "b.json",
                                              vocab_sha="v", segments_sha="s",
                                              fit_partition_name="train", robust=robust)
                        self.assertEqual(a.read_bytes(), b.read_bytes())
            stats = compute_value_stats_from_events(shard)
            self.assertIn(63, stats)          # rare token still covered
            self.assertNotIn(1, stats)        # categorical only: nothing to normalize
            self.assertEqual(stats[60][0], 7.0)

    def test_streaming_refuses_what_the_whole_shard_path_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.parquet"
            pl.DataFrame({"token": [[1, 2]], "value": [[1.0]],
                          "partition": ["train"]}).write_parquet(path)
            with self.assertRaisesRegex(ValueError, "must align within a stay"):
                compute_value_stats_from_events(path)
            pl.DataFrame({"token": [[1]], "value": [[1.0]],
                          "partition": ["validation"]}).write_parquet(path)
            with self.assertRaisesRegex(ValueError, "has zero rows"):
                compute_value_stats_from_events(path)
            pl.DataFrame({"token": [[1]], "value": [[1.0]]}).write_parquet(path)
            with self.assertRaisesRegex(ValueError, "partition column is required"):
                compute_value_stats_from_events(path)
            pl.DataFrame({"token": [[1]], "partition": ["train"]}).write_parquet(path)
            with self.assertRaisesRegex(ValueError, "missing required column 'value'"):
                compute_value_stats_from_events(path)


if __name__ == "__main__":
    unittest.main()
