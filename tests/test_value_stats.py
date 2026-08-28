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

from src.data.value_stats import (
    compute_value_stats,
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


if __name__ == "__main__":
    unittest.main()
