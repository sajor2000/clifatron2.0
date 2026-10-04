import unittest

import torch
from torch import nn

from src.model.heads import (
    NextEventHead,
    ThresholdHazardHead,
    ValueRegressionHead,
    next_event_loss,
)


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

    def test_value_loss_is_finite_for_extreme_negative_log_variance(self):
        head = ValueRegressionHead(4, 8)
        for parameter in head.parameters():
            parameter.data.zero_()
        head.mlp[-1].bias.data[1] = -100.0

        loss = head.loss_aligned(
            torch.zeros(1, 2, 4),
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
        )

        self.assertTrue(torch.isfinite(loss))

    def test_threshold_loss_is_finite_for_saturated_hazard(self):
        head = ThresholdHazardHead(4, 1, 4, 2)
        for parameter in head.parameters():
            parameter.data.zero_()
        head.mlp[-1].bias.data.fill_(20.0)

        loss = head.loss(
            torch.zeros(1, 4),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([-1]),
            torch.tensor([4]),
        )

        self.assertTrue(torch.isfinite(loss))

    def test_threshold_censoring_credits_only_the_observed_bins(self):
        """KTD4: `observed_bin` is the number of FULLY observed bins; a query censored
        after ten of them contributes survival for hours 0..9 and nothing later."""
        torch.manual_seed(5)
        head = ThresholdHazardHead(4, 1, 48, 2)
        h = torch.randn(1, 4)
        zeros = torch.zeros(1, dtype=torch.long)
        logits = head._logits(h, zeros, zeros, zeros).float()

        loss = head.loss(h, zeros, zeros, zeros, torch.tensor([-1]), torch.tensor([10]))
        expected = -torch.nn.functional.logsigmoid(-logits[0, :10]).sum()
        self.assertAlmostEqual(float(loss), float(expected), places=5)

    def test_threshold_event_in_hour_bin_5_is_survival_then_the_event(self):
        torch.manual_seed(5)
        head = ThresholdHazardHead(4, 1, 48, 2)
        h = torch.randn(1, 4)
        zeros = torch.zeros(1, dtype=torch.long)
        logits = head._logits(h, zeros, zeros, zeros).float()

        loss = head.loss(h, zeros, zeros, zeros, torch.tensor([5]), torch.tensor([5]))
        logsig = torch.nn.functional.logsigmoid
        expected = -(logsig(-logits[0, :5]).sum() + logsig(logits[0, 5]))
        self.assertAlmostEqual(float(loss), float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
