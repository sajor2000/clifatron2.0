import unittest

import polars as pl
import torch

from src.data.tokenize import _soft_bins, build_value_bins
from src.model.encoder import CLIFEncoder


class TokenizeBinsTest(unittest.TestCase):
    def test_forced_edges_replace_quantiles_without_growing_vocab(self):
        events = pl.DataFrame(
            {"concept": ["lactate"] * 100, "value": [float(value) for value in range(100)]}
        )
        edges = build_value_bins(events, 10, {"lactate": [2.0, 4.0]})["lactate"]

        self.assertEqual(len(edges), 9)
        self.assertIn(2.0, edges)
        self.assertIn(4.0, edges)

    def test_soft_bins_are_local_and_normalized(self):
        assignments = _soft_bins(4.5, "lactate", {"lactate": [2.0, 4.0, 6.0]}, 1)

        self.assertAlmostEqual(sum(weight for _, weight in assignments), 1.0)
        self.assertEqual([bin_idx for bin_idx, _ in assignments], [1, 2, 3])

    def test_encoder_accepts_weighted_tokens(self):
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
