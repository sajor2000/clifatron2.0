import unittest

import polars as pl
import torch

from src.data.segments import bin_index, segments_from_edges, validate_partition
from src.data.tokenize import (
    ROOT,
    _bin_of,
    _soft_bins,
    build_clinical_segment_bins,
    build_edges,
    build_value_bins,
)
from src.model.encoder import CLIFEncoder

_SEGMENT_CSV = (
    "external/clifatron/tokenETL/config/"
    "critical_illness_tokenization_final_with_intervals.csv"
)


def _is_boundary(segments: list[dict], value: float) -> bool:
    """`value` is the shared endpoint of two consecutive segments."""
    return any(a["hi"] == value == b["lo"] for a, b in zip(segments, segments[1:]))


class ClinicalSegmentBinsTest(unittest.TestCase):
    """The primary v2 scheme: physician-designed clinical-segment bins from the CSV."""

    def test_reads_physician_segments_for_target_concepts(self):
        edges = build_clinical_segment_bins(
            ROOT / _SEGMENT_CSV, ["lactate", "map", "creatinine"]
        )
        self.assertEqual(set(edges), {"lactate", "map", "creatinine"})
        # lactate has the clinician-designed granularity (many bins, sepsis-zone edges)
        self.assertGreater(len(edges["lactate"]), 10)
        for segments in edges.values():
            validate_partition(segments)  # strictly sorted, non-overlapping segments
            self.assertTrue(all(a["hi"] <= b["lo"] for a, b in zip(segments, segments[1:])))

    def test_forced_edges_pinned_onto_segment_grid(self):
        edges = build_clinical_segment_bins(
            ROOT / _SEGMENT_CSV, ["lactate"], forced_edges={"lactate": [2.0, 4.0]},
            directions={"lactate": "above"},
        )["lactate"]
        self.assertTrue(_is_boundary(edges, 2.0))
        self.assertTrue(_is_boundary(edges, 4.0))
        # direction "above": the threshold value stays in the bin below the edge
        self.assertEqual(_bin_of(4.0, "lactate", {"lactate": edges}),
                         _bin_of(3.9, "lactate", {"lactate": edges}))
        self.assertNotEqual(_bin_of(4.0, "lactate", {"lactate": edges}),
                            _bin_of(4.01, "lactate", {"lactate": edges}))

    def test_map_65_decision_threshold_is_a_bin_edge(self):
        edges = build_clinical_segment_bins(
            ROOT / _SEGMENT_CSV, ["map"], forced_edges={"map": [65.0]},
            directions={"map": "below"},
        )["map"]
        self.assertTrue(_is_boundary(edges, 65.0))
        # a MAP of 64 and 66 fall on opposite sides of the 65 edge
        self.assertNotEqual(_bin_of(64.0, "map", {"map": edges}),
                            _bin_of(66.0, "map", {"map": edges}))
        # direction "below": MAP 65 itself is on the non-event (upper) side
        self.assertEqual(_bin_of(65.0, "map", {"map": edges}),
                         _bin_of(66.0, "map", {"map": edges}))

    def test_build_edges_dispatches_on_scheme(self):
        events = pl.DataFrame({"concept": ["lactate"] * 40,
                               "value": [float(v) / 8 for v in range(40)]})
        clinical = build_edges(
            {"scheme": "clinical_segment", "segment_source": _SEGMENT_CSV},
            events, ["lactate"],
        )
        decile = build_edges(
            {"scheme": "decile", "n_bins": 4}, events, ["lactate"],
        )
        self.assertIn("lactate", clinical)
        self.assertIn("lactate", decile)
        self.assertNotEqual(clinical["lactate"], decile["lactate"])  # different schemes

    def test_unknown_scheme_and_missing_params_fail_closed(self):
        events = pl.DataFrame({"concept": ["lactate"], "value": [1.0]})
        with self.assertRaises(ValueError):
            build_edges({"scheme": "nonsense"}, events, ["lactate"])
        with self.assertRaises(ValueError):
            build_edges({"scheme": "clinical_segment"}, events, ["lactate"])  # no source
        with self.assertRaises(ValueError):
            build_edges({"scheme": "decile"}, events, ["lactate"])  # no n_bins

    def test_decile_ablation_scheme_uses_n_bins(self):
        events = pl.DataFrame({"concept": ["lactate"] * 40,
                               "value": [float(v) / 8 for v in range(40)]})
        edges = build_edges({"scheme": "decile_ablation", "n_bins": 10}, events, ["lactate"])
        self.assertIn("lactate", edges)
        self.assertEqual(len(edges["lactate"]), 10)  # 10 bins -> 10 [a, b) segments
        validate_partition(edges["lactate"])
        interior = build_value_bins(events, 10)["lactate"]
        self.assertEqual(edges["lactate"], segments_from_edges(interior))
        # [a, b): a value exactly on a quantile edge goes to the bin above it
        self.assertEqual(bin_index(interior[3], edges["lactate"]), 4)

    def test_decile_forced_edges_take_outcome_direction_closure(self):
        events = pl.DataFrame({"concept": ["lactate"] * 100,
                               "value": [float(v) / 10 for v in range(100)]})
        edges = build_edges(
            {"scheme": "decile", "n_bins": 10, "forced_edges": {"lactate": [2.0]}},
            events, ["lactate"], directions={"lactate": "above"},
        )["lactate"]
        self.assertTrue(_is_boundary(edges, 2.0))
        self.assertEqual(bin_index(2.0, edges), bin_index(1.99, edges))
        self.assertNotEqual(bin_index(2.0, edges), bin_index(2.01, edges))


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
