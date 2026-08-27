import unittest

from src.train.curriculum import curriculum_weights


class CurriculumTest(unittest.TestCase):
    def test_phases_through_schedule(self):
        total = 1000

        # warmup (step 0): NTP only
        mix = curriculum_weights(0, total)
        self.assertAlmostEqual(mix.w_ntp, 1.0)
        self.assertAlmostEqual(mix.w_cr, 0.0)
        self.assertAlmostEqual(mix.w_th, 0.0)
        self.assertFalse(mix.train_heads)

        # mid-warmup
        mix = curriculum_weights(75, total)
        self.assertAlmostEqual(mix.w_ntp, 1.0)
        self.assertFalse(mix.train_heads)

        # transition
        mix = curriculum_weights(170, total)
        self.assertAlmostEqual(mix.w_ntp, 0.2)
        self.assertGreater(mix.w_cr, 0.0)
        self.assertGreater(mix.w_th, 0.0)
        self.assertTrue(mix.train_heads)

        # mixed
        mix = curriculum_weights(400, total)
        self.assertAlmostEqual(mix.w_ntp, 0.2)
        self.assertAlmostEqual(mix.w_cr, 1.0)
        self.assertAlmostEqual(mix.w_th, 1.0)
        self.assertAlmostEqual(mix.w_val, 0.5)
        self.assertTrue(mix.train_heads)

    def test_last_step_is_mixed(self):
        mix = curriculum_weights(999, 1000)
        self.assertEqual(mix.w_cr, 1.0)
        self.assertTrue(mix.train_heads)


if __name__ == "__main__":
    unittest.main()