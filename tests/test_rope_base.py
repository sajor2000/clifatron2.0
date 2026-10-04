"""RoPE base for admission-minute positions (pre-training config decision, 2026-10-03).

Positions are minutes since admission; stays reach ~100,000 minutes. The slowest rotary
dimension must not wrap within that span, or two far-apart minutes share an angle there.
"""

import math
import unittest
from pathlib import Path

import torch
import yaml

from src.model.encoder import build_rope_cache

ROOT = Path(__file__).resolve().parents[1]
TRUNK = yaml.safe_load((ROOT / "configs/model.yaml").read_text())["trunk"]
HEAD_DIM = TRUNK["d_model"] // TRUNK["n_heads"]
LONGEST_STAY_MIN = 100_000


def slowest_angles(base: float, positions: torch.Tensor) -> torch.Tensor:
    cos, sin = build_rope_cache(positions[None, :], HEAD_DIM, base)
    half = HEAD_DIM // 2
    # Slowest frequency is the last of the first half (inv_freq is decreasing).
    return torch.atan2(sin[0, :, half - 1], cos[0, :, half - 1])


class RopeBaseTest(unittest.TestCase):
    def test_slowest_wavelength_covers_the_longest_stay(self):
        base = float(TRUNK["rope_base"])
        half = HEAD_DIM // 2
        wavelength = 2 * math.pi * base ** ((half - 1) / half)
        self.assertGreater(wavelength, 2 * LONGEST_STAY_MIN)
        # The old base wrapped inside a long stay.
        self.assertLess(2 * math.pi * 10000.0 ** ((half - 1) / half), LONGEST_STAY_MIN)

    def test_positions_up_to_the_longest_stay_map_to_distinct_slowest_angles(self):
        positions = torch.arange(0, LONGEST_STAY_MIN + 1, 60, dtype=torch.long)   # hourly
        angles = slowest_angles(float(TRUNK["rope_base"]), positions)
        unwrapped = torch.remainder(angles, 2 * math.pi)
        self.assertTrue(bool((unwrapped[1:] > unwrapped[:-1]).all()), "slowest angle wraps")
        self.assertEqual(torch.unique(unwrapped).numel(), positions.numel())
        old = torch.remainder(slowest_angles(10000.0, positions), 2 * math.pi)
        self.assertFalse(bool((old[1:] > old[:-1]).all()))       # 1e4 wraps

    def test_short_range_resolution_is_unchanged(self):
        # The fastest dimension turns 1 rad per minute whatever the base.
        cos, sin = build_rope_cache(torch.tensor([[0, 1]]), HEAD_DIM, float(TRUNK["rope_base"]))
        self.assertAlmostEqual(float(torch.atan2(sin[0, 1, 0], cos[0, 1, 0])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
