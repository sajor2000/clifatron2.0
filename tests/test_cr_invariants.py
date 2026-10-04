import math
import unittest
import torch
from src.model.heads import CompetingRiskHead, ThresholdHazardHead, time_bin


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


class CensoredCreditInvariants(unittest.TestCase):
    """KTD4: a censored interval is credited only through its last FULLY observed bin.

    For a censored sample `dt_bin` is the number of fully observed bins (the index of
    the bin the censoring time fell in); its likelihood is the event-free probability
    through those bins and nothing later."""

    def setUp(self):
        torch.manual_seed(3)
        self.d, self.K, self.B = 6, 2, 5
        self.head = CompetingRiskHead(self.d, self.K, self.B)
        self.h = torch.randn(1, self.d)
        _, self.event_free = self.head.cif(self.h)          # [1, B]

    def _censored_loss(self, bins: int) -> float:
        return float(self.head.loss(self.h, torch.tensor([-1]), torch.tensor([bins])))

    def test_censored_credit_stops_before_the_partially_observed_bin(self):
        for bins in range(1, self.B + 1):
            with self.subTest(fully_observed_bins=bins):
                expected = -torch.log(self.event_free[0, bins - 1])
                self.assertAlmostEqual(self._censored_loss(bins), float(expected), places=5)

    def test_censoring_inside_the_first_bin_gives_no_credit(self):
        self.assertEqual(self._censored_loss(0), 0.0)

    def test_censored_credit_equals_one_minus_total_incidence(self):
        cif, _ = self.head.cif(self.h)
        for bins in range(1, self.B + 1):
            survival = 1.0 - float(cif[0, :, bins - 1].sum())
            self.assertAlmostEqual(math.exp(-self._censored_loss(bins)), survival, places=5)

    def test_a_longer_censored_interval_never_costs_less(self):
        losses = [self._censored_loss(bins) for bins in range(self.B + 1)]
        self.assertEqual(losses, sorted(losses))

    def test_fully_observed_horizon_is_credited_through_every_bin(self):
        beyond = self._censored_loss(self.B + 3)             # clamped to the horizon
        self.assertAlmostEqual(beyond, self._censored_loss(self.B), places=6)

    def test_event_likelihood_is_prior_survival_times_cause_probability(self):
        q = self.head._distribution(self.h)                  # [1, K+1, B]
        for b in range(self.B):
            prior = 1.0 if b == 0 else float(self.event_free[0, b - 1])
            expected = -math.log(prior * float(q[0, 1, b]))
            got = float(self.head.loss(self.h, torch.tensor([1]), torch.tensor([b])))
            with self.subTest(event_bin=b):
                self.assertAlmostEqual(got, expected, places=5)

    def test_censoring_and_an_event_in_the_same_bin_share_the_prior_survival(self):
        """Censored in bin b: S(b-1). Event in bin b: S(b-1) * q. Same prior, so the
        event likelihood can never exceed the censored one."""
        for b in range(self.B):
            event = float(self.head.loss(self.h, torch.tensor([0]), torch.tensor([b])))
            self.assertGreater(event, self._censored_loss(b))


class ManyCauseLikelihoodTest(unittest.TestCase):
    """The production head has 11 causes over 16 bins. At initialization the event-free
    probability through the horizon is about 12**-16: the likelihood must still be exact
    and still carry a gradient (it used to be clamped at 1e-8, which froze every
    event-free and every late-event sample)."""

    def setUp(self):
        torch.manual_seed(0)
        self.head = CompetingRiskHead(8, 11, 16)
        self.h = torch.randn(3, 8)

    def test_full_horizon_survival_is_exact_and_has_a_gradient(self):
        loss = self.head.loss(self.h, torch.full((3,), -1), torch.full((3,), 16))
        self.head.double()                                  # reference in float64
        _, event_free = self.head.cif(self.h.double())
        expected = -torch.log(event_free[:, 15]).mean()
        self.head.float()
        self.assertLess(float(event_free[:, 15].max()), 1e-8)      # below the old clamp
        self.assertAlmostEqual(float(loss), float(expected), places=3)
        loss.backward()
        self.assertGreater(float(self.head.fc.weight.grad.abs().sum()), 0.0)

    def test_late_event_is_exact_and_has_a_gradient(self):
        loss = self.head.loss(self.h, torch.tensor([0, 5, 10]), torch.tensor([15, 14, 13]))
        self.assertGreater(float(loss), -math.log(1e-8))           # past the old clamp
        loss.backward()
        self.assertGreater(float(self.head.fc.weight.grad.abs().sum()), 0.0)


class TimeBinTest(unittest.TestCase):
    """KTD4: each head bins minutes since the anchor on ITS OWN grid (floor)."""

    def test_five_hours_is_hour_bin_5_and_three_hour_bin_1(self):
        minutes = torch.tensor([300])
        self.assertEqual(time_bin(minutes, 48, 48).tolist(), [5])
        self.assertEqual(time_bin(minutes, 16, 48).tolist(), [1])

    def test_bins_are_left_closed(self):
        minutes = torch.tensor([0, 59, 60, 179, 180, 181])
        self.assertEqual(time_bin(minutes, 48, 48).tolist(), [0, 0, 1, 2, 3, 3])
        self.assertEqual(time_bin(minutes, 16, 48).tolist(), [0, 0, 0, 0, 1, 1])

    def test_ten_hours_is_ten_full_hour_bins_and_three_full_three_hour_bins(self):
        minutes = torch.tensor([600])
        self.assertEqual(time_bin(minutes, 48, 48).tolist(), [10])
        self.assertEqual(time_bin(minutes, 16, 48).tolist(), [3])

    def test_the_horizon_itself_is_one_past_the_last_bin(self):
        minutes = torch.tensor([2880, 5000])
        self.assertEqual(time_bin(minutes, 48, 48).tolist(), [48, 83])
        self.assertEqual(time_bin(minutes, 16, 48).tolist(), [16, 27])

    def test_grid_must_be_positive(self):
        with self.assertRaises(ValueError):
            time_bin(torch.tensor([1]), 0, 48)
        with self.assertRaises(ValueError):
            time_bin(torch.tensor([1]), 4, 0)


if __name__ == "__main__":
    unittest.main()
