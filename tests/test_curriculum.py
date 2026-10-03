"""U4 (KTD5): the next-token -> time-to-event weight schedule and its config gate."""

import unittest
from pathlib import Path

import yaml

from src.train.curriculum import curriculum_enabled, curriculum_weights

ROOT = Path(__file__).resolve().parents[1]
# total 1000: warm-up is updates 0..149, the transition 150..199, configured from 200.
TOTAL = 1000


class CurriculumTest(unittest.TestCase):
    def test_phases_through_schedule(self):
        # warmup (step 0): NTP only
        mix = curriculum_weights(0, TOTAL)
        self.assertEqual(tuple(mix[:4]), (1.0, 0.0, 0.0, 0.0))
        self.assertFalse(mix.train_heads)

        # last warm-up update
        mix = curriculum_weights(149, TOTAL)
        self.assertEqual(tuple(mix[:4]), (1.0, 0.0, 0.0, 0.0))
        self.assertFalse(mix.train_heads)

        # transition
        mix = curriculum_weights(170, TOTAL)
        self.assertGreater(mix.w_cr, 0.0)
        self.assertGreater(mix.w_th, 0.0)
        self.assertTrue(mix.train_heads)

        # mixed
        mix = curriculum_weights(400, TOTAL)
        self.assertEqual(tuple(mix[:4]), (0.2, 1.0, 1.0, 0.5))
        self.assertTrue(mix.train_heads)

    def test_last_step_is_mixed(self):
        mix = curriculum_weights(999, 1000)
        self.assertEqual(mix.w_cr, 1.0)
        self.assertTrue(mix.train_heads)

    def test_transition_blends_linearly_from_next_token_only_to_the_target(self):
        target = (0.3, 2.0, 0.8, 0.7)
        for step, progress in ((150, 0.0), (160, 0.2), (175, 0.5), (199, 0.98)):
            mix = curriculum_weights(step, TOTAL, target=target)
            with self.subTest(step=step):
                self.assertAlmostEqual(mix.w_ntp, 1.0 + progress * (0.3 - 1.0))
                self.assertAlmostEqual(mix.w_cr, progress * 2.0)
                self.assertAlmostEqual(mix.w_th, progress * 0.8)
                self.assertAlmostEqual(mix.w_val, progress * 0.7)
        # The blend is linear: equal steps move every weight by the same amount.
        a, b, c = (curriculum_weights(s, TOTAL, target=target) for s in (160, 170, 180))
        for lo, mid, hi in zip(a[:4], b[:4], c[:4]):
            self.assertAlmostEqual(mid - lo, hi - mid)

    def test_after_the_transition_the_weights_are_exactly_the_target(self):
        target = (0.3, 2.0, 0.8, 0.7)
        for step in (200, 201, 999, 5000):   # also past the planned update count
            self.assertEqual(tuple(curriculum_weights(step, TOTAL, target=target)[:4]), target)

    def test_a_head_the_target_removes_never_gets_weight(self):
        target = (0.2, 1.0, 1.0, 0.0)        # the minus-value arm
        for step in range(0, TOTAL, 7):
            self.assertEqual(curriculum_weights(step, TOTAL, target=target).w_val, 0.0)

    def test_next_token_weight_is_never_zero(self):
        for step in range(0, 260):
            self.assertGreater(curriculum_weights(step, TOTAL).w_ntp, 0.0)

    def test_a_run_too_short_for_a_warm_up_starts_on_the_target(self):
        # int(total * 0.15) == 0: there is no warm-up update to spend.
        for total in (1, 2, 6):
            self.assertEqual(tuple(curriculum_weights(0, total)[:4]), (0.2, 1.0, 1.0, 0.5))

    def test_target_must_name_the_four_weights(self):
        with self.assertRaises(ValueError):
            curriculum_weights(0, TOTAL, target=(0.2, 1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "negative"):
            curriculum_weights(0, TOTAL, target=(0.2, -1.0, 1.0, 0.5))


class ObjectiveConfigGateTest(unittest.TestCase):
    """Loss balancing is fixed weights; an option nothing implements is refused."""

    def test_fixed_and_absent_loss_balancing_pass(self):
        self.assertTrue(curriculum_enabled({"loss_balancing": "fixed",
                                            "curriculum": "ntp_then_tte"}))
        self.assertFalse(curriculum_enabled({"loss_balancing": "fixed", "curriculum": "none"}))
        self.assertFalse(curriculum_enabled({}))    # no curriculum unless one is asked for

    def test_unknown_loss_balancing_fails_closed(self):
        for value in ("uncertainty", "gradnorm", "", None, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "loss_balancing.*fixed"):
                    curriculum_enabled({"loss_balancing": value, "curriculum": "none"})

    def test_unknown_curriculum_fails_closed(self):
        for value in ("tte_then_ntp", "ntp", None, False):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "curriculum.*ntp_then_tte"):
                    curriculum_enabled({"curriculum": value})

    def test_repo_model_configs_use_fixed_weights(self):
        for name, enabled in (("model.yaml", True), ("model.gem-ntp.yaml", False)):
            mcfg = yaml.safe_load((ROOT / "configs" / name).read_text())
            with self.subTest(config=name):
                self.assertEqual(mcfg["loss_balancing"], "fixed")
                self.assertEqual(curriculum_enabled(mcfg), enabled)

    def test_model_refuses_an_unknown_loss_balancing_value(self):
        from src.train.pretrain import Model

        try:
            from test_ddp_multihead import tiny_mcfg
        except ImportError:  # pragma: no cover - run from the repo root as a package
            from tests.test_ddp_multihead import tiny_mcfg
        mcfg = tiny_mcfg()
        mcfg["loss_balancing"] = "uncertainty"
        with self.assertRaisesRegex(ValueError, "loss_balancing"):
            Model(32, 2, mcfg, n_value_bins=5)


if __name__ == "__main__":
    unittest.main()
