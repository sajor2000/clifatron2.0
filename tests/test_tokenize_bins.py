import unittest

import polars as pl
import torch

from src.data.tokenize import _soft_bins, build_value_bins
from src.model.encoder import CLIFEncoder


class TokenizeBinsTest(unittest.TestCase):
    def test_forced_edges_stay_if_outside_reference_range(self):
        events = pl.DataFrame(
            {"concept": ["lactate"] * 10, "value": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}
        )
        edges_lo = build_value_bins(events, 4, {"lactate": [0.1, 2.0]})["lactate"]
        self.assertIn(0.1, edges_lo)

        edges_hi = build_value_bins(events, 4, {"lactate": [2.0, 12.0]})["lactate"]
        self.assertIn(12.0, edges_hi)

    def test_forced_edges_replace_with_exact_values(self):
        events = pl.DataFrame(
            {"concept": ["lactate"] * 100, "value": [float(v) for v in range(100)]}
        )
        edges = build_value_bins(events, 10, {"lactate": [2.0, 4.0]})["lactate"]
        self.assertEqual(len(edges), 9)
        self.assertIn(2.0, edges)
        self.assertIn(4.0, edges)

    def test_rejects_nonfinite_values(self):
        events = pl.DataFrame(
            {
                "concept": ["lactate"] * 5,
                "value": [0.0, 1.0, 2.0, 3.0, float("nan")],
            }
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_value_bins(events, 3)

    def test_soft_bins_are_fixed_width_and_normalized(self):
        assignments = _soft_bins(
            4.5, "lactate", {"lactate": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]}, kernel_bins=1
        )
        self.assertEqual(len(assignments), 3)
        self.assertAlmostEqual(sum(w for _, w in assignments), 1.0)
        bins = [b for b, _ in assignments]
        self.assertIn(3, bins)

    def test_soft_bins_lowest_bin_weights_stay_normalized(self):
        assignments = _soft_bins(
            -1.0, "lactate", {"lactate": [0.0, 2.0, 4.0]}, kernel_bins=1
        )
        self.assertEqual(len(assignments), 3)
        self.assertAlmostEqual(sum(w for _, w in assignments), 1.0)
        self.assertLessEqual(max(w for _, w in assignments), 1.0)

    def test_encoder_rejects_2d_with_weights(self):
        cfg = {
            "trunk": {
                "d_model": 8,
                "n_layers": 1,
                "n_heads": 2,
                "ffn_mult": 2,
                "dropout": 0.0,
                "tied_embeddings": False,
            }
        }
        encoder = CLIFEncoder(12, cfg)
        with self.assertRaisesRegex(ValueError, "both be 3D"):
            encoder(torch.tensor([[3, 4]]), torch.tensor([[0, 1]]), torch.tensor([[0.5, 0.5]]))

    def test_encoder_accepts_weighted_3d_tokens(self):
        cfg = {
            "trunk": {
                "d_model": 8,
                "n_layers": 1,
                "n_heads": 2,
                "ffn_mult": 2,
                "dropout": 0.0,
                "tied_embeddings": False,
            }
        }
        encoder = CLIFEncoder(12, cfg)
        token = torch.tensor([[[3, 4], [5, 6]]])
        weight = torch.tensor([[[0.75, 0.25], [0.5, 0.5]]])
        output = encoder(token, torch.tensor([[0, 1]]), weight)
        self.assertEqual(output.shape, (1, 2, 8))
        with self.assertRaisesRegex(ValueError, "token_weight is required"):
            encoder(token, torch.tensor([[0, 1]]))


if __name__ == "__main__":
    unittest.main()
