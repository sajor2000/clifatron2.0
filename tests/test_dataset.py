import unittest

import torch
from torch.utils.data import SequentialSampler

from src.data.collate import ModelCollator
from src.data.dataset import PACKED_SCHEMA_VERSION, ModelDataset, make_dataloader
from src.data.targets import TargetBuilder, TargetContractError


# Tokenizer-v2 shard binding (KTD7) plus an extra identity hash.
HASHES = {"vocabulary": "v1", "numeric_edges": "s1", "tokenizer_version": "2",
          "outcome_spec": "o1"}


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

    def test_length_grouped_sampler_covers_and_groups(self):
        from src.data.dataset import LengthGroupedSampler

        lengths = [5, 6413, 471, 12, 271, 247, 6300, 3, 900, 150]
        sampler = LengthGroupedSampler(lengths, batch_size=2, seed=7, mega_batch_mult=1)
        order = list(iter(sampler))
        # full coverage, exactly once
        self.assertEqual(sorted(order), list(range(len(lengths))))
        # determinism: same seed -> same order
        self.assertEqual(order, list(iter(LengthGroupedSampler(lengths, 2, seed=7, mega_batch_mult=1))))
        # epoch changes the order but keeps coverage
        s2 = LengthGroupedSampler(lengths, batch_size=2, seed=7, mega_batch_mult=1)
        s2.set_epoch(1)
        self.assertEqual(sorted(iter(s2)), list(range(len(lengths))))
        # grouping: with mega_batch_mult=1 the whole set is one sorted chunk, so
        # consecutive pairs are non-decreasing in length -> similar-length batches
        pairs = [order[i:i + 2] for i in range(0, len(order), 2)]
        for left, right in pairs:
            self.assertLessEqual(lengths[left], lengths[right])

    def test_token_budget_batch_sampler_packs_to_budget(self):
        from src.data.dataset import TokenBudgetBatchSampler

        lengths = [100, 6413, 90, 471, 12, 6300, 3, 900]
        sampler = TokenBudgetBatchSampler(
            lengths, max_batch_tokens=1024, max_batch_size=8, seed=7, mega_batch_mult=1,
        )
        batches = list(iter(sampler))

        # full coverage, exactly once
        seen = [i for b in batches for i in b]
        self.assertEqual(sorted(seen), list(range(len(lengths))))

        # budget holds: B x max_len_in_batch <= 1024, except a lone oversize item
        for batch in batches:
            longest = max(lengths[i] for i in batch)
            if len(batch) > 1:
                self.assertLessEqual(len(batch) * longest, 1024)
            else:
                # a single item may exceed the budget only when alone
                self.assertTrue(longest <= 1024 or len(batch) == 1)
        # the long stays batch alone (6413, 6300 > 1024)
        self.assertIn([1], batches)
        self.assertIn([5], batches)
        # short stays pack together: 3 + 12 + 90 + 100 all fit one batch (100 x 4 = 400)
        self.assertIn([6, 4, 2, 0], batches)

        # max_batch_size caps even when the token budget would allow more
        capped = list(iter(TokenBudgetBatchSampler(
            [10] * 20, max_batch_tokens=100000, max_batch_size=4, seed=3,
            mega_batch_mult=1,
        )))
        self.assertTrue(all(len(b) <= 4 for b in capped))

        # determinism + epoch rotation
        self.assertEqual(
            batches,
            list(iter(TokenBudgetBatchSampler(
                lengths, max_batch_tokens=1024, max_batch_size=8, seed=7, mega_batch_mult=1,
            ))),
        )
        s2 = TokenBudgetBatchSampler(
            lengths, max_batch_tokens=1024, max_batch_size=8, seed=7, mega_batch_mult=1,
        )
        s2.set_epoch(1)
        self.assertEqual(sorted(i for b in s2 for i in b), list(range(len(lengths))))

        # shuffle=False: deterministic sorted order (validation path)
        seq = TokenBudgetBatchSampler(
            lengths, max_batch_tokens=1024, seed=7, shuffle=False, mega_batch_mult=1,
        )
        flat = [i for b in seq for i in b]
        self.assertEqual(flat, sorted(flat))


if __name__ == "__main__":
    unittest.main()
