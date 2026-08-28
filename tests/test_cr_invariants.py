import unittest
import torch
from src.model.heads import CompetingRiskHead, ThresholdHazardHead


class CompetingRiskInvariants(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.d, self.K, self.B = 8, 3, 16
        self.head = CompetingRiskHead(self.d, self.K, self.B)

    def test_cif_plus_event_free_equals_one(self):
        h = torch.randn(5, self.d)
        cif, ef = self.head.cif(h)
        total = cif.sum(-2) + ef
        self.assertTrue(torch.allclose(total, torch.ones_like(ef), atol=1e-5),
                        f"max deviation: {(total - 1).abs().max():.2e}")

    def test_cifs_are_nonnegative_and_monotone(self):
        h = torch.randn(10, self.d)
        cif, ef = self.head.cif(h)
        self.assertTrue((cif >= 0).all(), "negative CIF entry")
        diffs = cif[..., 1:] - cif[..., :-1]
        self.assertTrue((diffs >= -1e-7).all(), "non-monotone CIF")

    def test_event_free_is_nonnegative_and_non_increasing(self):
        h = torch.randn(10, self.d)
        cif, ef = self.head.cif(h)
        self.assertTrue((ef >= 0).all() and (ef <= 1).all(),
                        "event-free probability out of [0,1]")
        diffs = ef[..., 1:] - ef[..., :-1]
        self.assertTrue((diffs <= 1e-7).all(), "event-free probability increased")

    def test_loss_gradients_flow_to_head(self):
        h = torch.randn(4, self.d, requires_grad=True)
        event_type = torch.tensor([0, 1, 2, 0])
        dt_bin = torch.randint(0, self.B, (4,))
        loss = self.head.loss(h, event_type, dt_bin)
        loss.backward()
        self.assertIsNotNone(self.head.fc.weight.grad)
        self.assertTrue((self.head.fc.weight.grad.abs().sum() > 0),
                        "no gradients flowed to CR head")

    def test_censored_sample_contributes_event_free_likelihood(self):
        h = torch.randn(4, self.d, requires_grad=True)
        dt_bin = torch.randint(0, self.B, (4,))
        event_type = torch.tensor([-1, -1, -1, -1])
        loss = self.head.loss(h, event_type, dt_bin)
        loss.backward()
        self.assertFalse(torch.isnan(loss), "NaN loss on censored-only batch")

    def test_censored_mask_in_constructor(self):
        h = torch.randn(4, self.d, requires_grad=True)
        dt_bin = torch.tensor([1, 3, 5, 10])
        event_type = torch.tensor([0, 1, 2, 1])
        censored = torch.tensor([False, True, False, True])
        loss = self.head.loss(h, event_type, dt_bin, censored=censored)
        loss.backward()
        self.assertFalse(torch.isnan(loss), "NaN with explicit censored mask")

    def test_recovery_of_known_hazards(self):
        torch.manual_seed(1)
        head = CompetingRiskHead(self.d, 1, 4)
        with torch.no_grad():
            head.fc.weight.zero_()
            head.fc.bias.zero_()
            head.fc.bias[0] = 2.0          # cause 0, bin 0  (elevated event prob)

        h = torch.randn(3, self.d)
        q = head._distribution(h)             # [3, 2, 4] -> (cause0, noevent)
        self.assertTrue((q[:, 0, 0] > q[:, 1, 0]).all(),
                        "elevated event bias should produce cause>noevent in bin 0")
        # With zero weights, all logits are bias only, so softmax reduces to a
        # per-bin contrast of the biased channel versus the neutral one.  The bin-0
        # bias is the only non-zero, so bin 0 stands out; other bins are uniform.

    def test_single_cause_equivalent_to_binary_survival(self):
        head = CompetingRiskHead(self.d, 1, self.B)
        h = torch.randn(5, self.d)
        cif, ef = head.cif(h)
        self.assertTrue(torch.allclose(cif.squeeze(-2) + ef, torch.ones_like(ef),
                                        atol=1e-5))

    def test_censored_branch_differs_from_event_branch(self):
        h = torch.randn(4, self.d)
        event_type = torch.zeros(4, dtype=torch.long)
        dt_bin = torch.tensor([2, 2, 2, 2])
        loss_event = self.head.loss(h, event_type, dt_bin, censored=torch.zeros(4, dtype=torch.bool))
        loss_censored = self.head.loss(h, event_type, dt_bin, censored=torch.ones(4, dtype=torch.bool))
        self.assertFalse(torch.isclose(loss_event, loss_censored, rtol=1e-3),
                         "event and censored branches must produce different losses")

    def test_rejects_zero_cause_types(self):
        h = torch.randn(2, self.d)
        with self.assertRaises(ValueError):
            CompetingRiskHead(self.d, 0, self.B).loss(h, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))

    def test_competing_event_label_no_cause_idx_raises(self):
        from src.data.targets import TargetBuilder, TargetContractError
        builder = TargetBuilder(vocab_size=100, n_time_bins=self.B, horizon_hours=48,
                                value_stats={}, run_seed=0)
        row_good = {"status": "competing_event", "target_idx": 1, "cause_idx": 2,
                     "time_from_anchor_hours": 12.0, "direction": "above", "tte_mask": True,
                     "threshold_bin": 3}
        label = builder._outcome_label(row_good)
        self.assertEqual(label["event_cause"], 2,
                         "competing_event cause_idx should propagate as event_cause")

        row_bad = {"status": "competing_event", "target_idx": 1,
                    "time_from_anchor_hours": 12.0, "direction": "above", "tte_mask": True,
                    "threshold_bin": 3}
        with self.assertRaises(TargetContractError):
            builder._outcome_label(row_bad)

    def test_threshold_no_crossing_contributes_survival_loss(self):
        head = ThresholdHazardHead(d=4, n_targets=1, n_time_bins=3, n_value_bins=4)
        h = torch.zeros(2, 4)
        target = torch.zeros(2, dtype=torch.long)
        tau = torch.zeros(2, dtype=torch.long)
        direction = torch.zeros(2, dtype=torch.long)
        crossed = -torch.ones(2, dtype=torch.long)

        loss = head.loss(h, target, tau, direction, crossed)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_threshold_censoring_uses_observed_window(self):
        head = ThresholdHazardHead(d=4, n_targets=1, n_time_bins=3, n_value_bins=4)
        h = torch.zeros(1, 4)
        target = torch.zeros(1, dtype=torch.long)
        tau = torch.zeros(1, dtype=torch.long)
        direction = torch.zeros(1, dtype=torch.long)
        crossed = -torch.ones(1, dtype=torch.long)

        early = head.loss(h, target, tau, direction, crossed, torch.ones(1, dtype=torch.long))
        full = head.loss(h, target, tau, direction, crossed, torch.full((1,), 3, dtype=torch.long))
        self.assertLess(float(early.detach()), float(full.detach()))


if __name__ == "__main__":
    unittest.main()
