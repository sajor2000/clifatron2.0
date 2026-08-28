import pickle
import unittest

import torch

from src.data.collate import ModelCollator, document_isolated_forward
from src.data.dataset import ModelDataset
from src.data.targets import TargetBuilder


def record(key, tokens):
    return {
        "episode_key": key,
        "artifact_hashes": {"vocabulary": "v1"},
        "token": tokens,
        "pos_min": list(range(len(tokens))),
        "value": [None] * len(tokens),
        "target_eligible": [True] * len(tokens),
        "anchor_idx": len(tokens) - 1,
        "anchor_min": len(tokens),
        "outcomes": [],
    }


class PrefixSumModel(torch.nn.Module):
    def forward(self, token, pos_min):
        return token.float().cumsum(dim=1).unsqueeze(-1)


class CollateTest(unittest.TestCase):
    def setUp(self):
        dataset = ModelDataset(
            [record("opaque-a", [3, 4]), record("opaque-b", [5, 6, 7])],
            representation="decile",
            target_builder=TargetBuilder(16, 48, 48, {}),
            expected_hashes={"vocabulary": "v1"},
        )
        self.samples = [dataset[0], dataset[1]]

    def test_collator_is_picklable_cpu_only_and_emits_flash_lengths(self):
        collator = pickle.loads(pickle.dumps(ModelCollator(pad_token_id=0)))
        batch = collator(self.samples)

        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertEqual(batch["cu_seqlens"].tolist(), [0, 2, 5])
        self.assertEqual(batch["max_seqlen"], 3)
        self.assertEqual(batch["flash_input_ids"].tolist(), [3, 4, 5, 6, 7])
        self.assertEqual(batch["flash_anchor_idx"].tolist(), [1, 4])
        self.assertTrue(all(not value.is_cuda for value in batch.values() if isinstance(value, torch.Tensor)))

    def test_collator_preserves_3d_soft_token_fields(self):
        samples = [dict(self.samples[0]), dict(self.samples[1])]
        samples[0]["soft_token"] = [[3, 4, 0], [4, 5, 0]]
        samples[0]["soft_weight"] = [[0.8, 0.2, 0.0], [0.7, 0.3, 0.0]]
        samples[1]["soft_token"] = [[5, 6, 0], [6, 7, 0], [7, 8, 0]]
        samples[1]["soft_weight"] = [[0.9, 0.1, 0.0], [0.6, 0.4, 0.0], [1.0, 0.0, 0.0]]
        batch = ModelCollator()(samples)

        self.assertEqual(batch["soft_token"].shape, (2, 3, 3))
        self.assertEqual(batch["soft_weight"].shape, (2, 3, 3))
        self.assertEqual(batch["soft_token"][0, 1].tolist(), [4, 5, 0])
        self.assertEqual(batch["soft_token"][0, 2].tolist(), [0, 0, 0])

    def test_reference_forward_isolates_documents_without_dense_mask(self):
        collator = ModelCollator()
        model = PrefixSumModel()
        first = document_isolated_forward(model, collator(self.samples))
        changed = [dict(self.samples[0], input_ids=[10, 11]), self.samples[1]]
        second = document_isolated_forward(model, collator(changed))

        self.assertFalse(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertNotIn("document_attention_mask", collator(self.samples))


if __name__ == "__main__":
    unittest.main()
