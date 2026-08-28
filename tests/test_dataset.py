import unittest

import torch
from torch.utils.data import SequentialSampler

from src.data.collate import ModelCollator
from src.data.dataset import PACKED_SCHEMA_VERSION, ModelDataset, make_dataloader
from src.data.targets import TargetBuilder, TargetContractError


HASHES = {"vocabulary": "v1", "outcome_spec": "o1"}


def target(key, tokens, *, anchor_idx):
    return {
        "episode_key": key,
        "token": tokens,
        "pos_min": list(range(len(tokens))),
        "value": [None] * len(tokens),
        "target_eligible": [True] * len(tokens),
        "anchor_idx": anchor_idx,
        "anchor_min": len(tokens),
        "outcomes": [
            {
                "target_idx": 1,
                "status": "negative",
                "time_from_anchor_hours": 48,
                "threshold_bin": 2,
                "direction": "below",
            }
        ],
    }


class ModelDatasetTest(unittest.TestCase):
    def setUp(self):
        self.builder = TargetBuilder(32, 48, 48, {})

    def test_value_and_ntp_and_anchor_equivalence_decile_vs_packed(self):
        canonical = target("opaque-a", [3, 4, 5], anchor_idx=2) | {
            "artifact_hashes": HASHES,
            "soft_token": [[1, 2, 3]] * 3,
            "soft_weight": [0.1, 0.2, 0.3],
        }
        decile = ModelDataset(
            [canonical],
            representation="decile",
            target_builder=self.builder,
            expected_hashes=HASHES,
        )[0]
        packed_record = {
            "packed_schema_version": PACKED_SCHEMA_VERSION,
            "artifact_hashes": HASHES,
            "input_ids": [3, 4, 5],
            "attention_mask": [1, 1, 1],
            "pos_min": [0, 1, 2],
            "segments": [
                {
                    "episode_key": "opaque-a",
                    "source_start": 0,
                    "source_end": 3,
                    "packed_start": 0,
                    "packed_end": 3,
                    "continuation_index": 0,
                    "continues_from_previous": False,
                    "continues_to_next": False,
                }
            ],
        }
        packed = ModelDataset(
            [packed_record],
            representation="clifatron_packed",
            target_builder=self.builder,
            expected_hashes=HASHES,
            episode_targets={"opaque-a": canonical},
        )[0]

        self.assertEqual(decile["value_target"], packed["value_target"])
        self.assertEqual(decile["value_mask"], packed["value_mask"])
        self.assertEqual(decile["ntp_target"], packed["ntp_target"])
        self.assertEqual(decile["ntp_mask"], packed["ntp_mask"])
        self.assertEqual(decile["ntp_delta_min"], packed["ntp_delta_min"])
        self.assertIsNotNone(decile["soft_token"])
        self.assertIsNotNone(decile["soft_weight"])
        decile_seg = decile["segments"][0]
        packed_seg = packed["segments"][0]
        self.assertEqual(decile_seg["outcome_labels"], packed_seg["outcome_labels"])
        self.assertEqual(decile_seg["threshold_query"], packed_seg["threshold_query"])
        self.assertEqual(decile_seg["anchor_offset"], packed_seg["anchor_offset"])

    def test_all_masked_ntp_targets_yield_finite_loss_denominator(self):
        record = target("opaque-a", [3, 4, 5], anchor_idx=2)
        record["target_eligible"] = [False, False, False]
        record["artifact_hashes"] = HASHES
        sample = ModelDataset(
            [record],
            representation="decile",
            target_builder=self.builder,
            expected_hashes=HASHES,
        )[0]

        import torch
        ntp_mask = torch.tensor(sample["ntp_mask"])
        self.assertFalse(ntp_mask.any(), "all-masked sample should have no active NTP positions")
        n_tokens = len(sample["input_ids"])
        self.assertGreaterEqual(n_tokens, 1, "loss denominator must be finite (>= 1 token)")

    def test_decile_and_packed_adapters_produce_equivalent_targets(self):
        canonical = target("opaque-a", [3, 4, 5], anchor_idx=2) | {
            "artifact_hashes": HASHES
        }
        decile = ModelDataset(
            [canonical],
            representation="decile",
            target_builder=self.builder,
            expected_hashes=HASHES,
        )[0]
        packed_record = {
            "packed_schema_version": PACKED_SCHEMA_VERSION,
            "artifact_hashes": HASHES,
            "input_ids": [3, 4, 5],
            "attention_mask": [1, 1, 1],
            "pos_min": [0, 1, 2],
            "segments": [
                {
                    "episode_key": "opaque-a",
                    "source_start": 0,
                    "source_end": 3,
                    "packed_start": 0,
                    "packed_end": 3,
                    "continuation_index": 0,
                    "continues_from_previous": False,
                    "continues_to_next": False,
                }
            ],
        }
        packed = ModelDataset(
            [packed_record],
            representation="clifatron_packed",
            target_builder=self.builder,
            expected_hashes=HASHES,
            episode_targets={"opaque-a": canonical},
        )[0]

        self.assertEqual(decile["ntp_target"], packed["ntp_target"])
        self.assertEqual(decile["ntp_mask"], packed["ntp_mask"])
        self.assertEqual(decile["segments"][0]["outcome_labels"], packed["segments"][0]["outcome_labels"])

    def test_continuation_emits_document_labels_only_on_anchor_segment(self):
        canonical = target("opaque-a", [3, 4, 5, 6], anchor_idx=3)
        rows = [
            {
                "packed_schema_version": PACKED_SCHEMA_VERSION,
                "artifact_hashes": HASHES,
                "input_ids": [3, 4],
                "attention_mask": [1, 1],
                "segments": [{
                    "episode_key": "opaque-a", "source_start": 0, "source_end": 2,
                    "packed_start": 0, "packed_end": 2, "continuation_index": 0,
                    "continues_from_previous": False, "continues_to_next": True,
                }],
            },
            {
                "packed_schema_version": PACKED_SCHEMA_VERSION,
                "artifact_hashes": HASHES,
                "input_ids": [5, 6],
                "attention_mask": [1, 1],
                "segments": [{
                    "episode_key": "opaque-a", "source_start": 2, "source_end": 4,
                    "packed_start": 0, "packed_end": 2, "continuation_index": 1,
                    "continues_from_previous": True, "continues_to_next": False,
                }],
            },
        ]
        dataset = ModelDataset(
            rows,
            representation="clifatron_packed",
            target_builder=self.builder,
            expected_hashes=HASHES,
            episode_targets={"opaque-a": canonical},
        )

        self.assertIsNone(dataset[0]["segments"][0]["anchor_offset"])
        self.assertEqual(dataset[0]["segments"][0]["outcome_labels"], [])
        self.assertEqual(dataset[1]["segments"][0]["anchor_offset"], 1)
        self.assertEqual(len(dataset[1]["segments"][0]["outcome_labels"]), 1)

    def test_fails_closed_on_hash_or_target_join_mismatch(self):
        canonical = target("opaque-a", [3], anchor_idx=0) | {"artifact_hashes": HASHES}
        with self.assertRaisesRegex(TargetContractError, "hash mismatch"):
            ModelDataset(
                [canonical], representation="decile", target_builder=self.builder,
                expected_hashes={"vocabulary": "wrong"},
            )

    def test_loader_honors_sampler_shuffle_exclusivity(self):
        canonical = target("opaque-a", [3, 4], anchor_idx=1) | {"artifact_hashes": HASHES}
        dataset = ModelDataset(
            [canonical], representation="decile", target_builder=self.builder,
            expected_hashes=HASHES,
        )
        sampler = SequentialSampler(dataset)
        loader = make_dataloader(
            dataset, batch_size=1, collate_fn=ModelCollator(), sampler=sampler,
        )
        self.assertIs(loader.sampler, sampler)
        self.assertTrue(all(not value.is_cuda for value in next(iter(loader)).values() if isinstance(value, torch.Tensor)))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            make_dataloader(
                dataset, batch_size=1, collate_fn=ModelCollator(), sampler=sampler, shuffle=True,
            )


if __name__ == "__main__":
    unittest.main()
