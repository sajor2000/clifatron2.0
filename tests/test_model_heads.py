import unittest

import torch
from torch import nn

from src.model.heads import NextEventHead, ValueRegressionHead, next_event_loss


class NextEventHeadTest(unittest.TestCase):
    def test_untied_by_default_and_tied_for_ablation(self):
        embedding = nn.Embedding(11, 4)
        untied = NextEventHead(4, 11, input_embedding=embedding)
        tied = NextEventHead(4, 11, tie_weights=True, input_embedding=embedding)

        self.assertNotEqual(untied.projection.weight.data_ptr(), embedding.weight.data_ptr())
        self.assertEqual(tied.projection.weight.data_ptr(), embedding.weight.data_ptr())
        self.assertEqual(untied(torch.zeros(2, 3, 4)).shape, (2, 3, 11))

    def test_masked_next_event_loss_ignores_masked_positions(self):
        logits = torch.zeros(1, 3, 5)
        target = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[True, False, False]])

        base = next_event_loss(logits, target, mask)
        changed = target.clone()
        changed[0, 1:] = torch.tensor([4, 4])
        self.assertTrue(torch.allclose(base, next_event_loss(logits, changed, mask)))

    def test_value_loss_aligned_all_masked_is_finite_zero(self):
        head = ValueRegressionHead(4, 8)
        h = torch.zeros(2, 3, 4)
        target_tok = torch.zeros(2, 3, dtype=torch.long)
        target_val = torch.zeros(2, 3)
        mask = torch.zeros(2, 3, dtype=torch.bool)

        loss = head.loss_aligned(h, target_tok, target_val, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
