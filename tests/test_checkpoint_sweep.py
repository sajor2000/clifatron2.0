import unittest

from src.eval.checkpoint_sweep import select_best


class CheckpointSweepTest(unittest.TestCase):
    def test_selects_lowest_finite_total_validation_loss(self):
        results = [
            {"step": 2040, "losses": {"total": 5.0}},
            {"step": 4080, "losses": {"total": 3.0}},
            {"step": 6000, "losses": {"total": float("nan")}},
        ]

        self.assertEqual(select_best(results)["step"], 4080)

    def test_rejects_sweep_without_finite_result(self):
        with self.assertRaisesRegex(ValueError, "no finite"):
            select_best([{"step": 1, "losses": {"total": float("nan")}}])


if __name__ == "__main__":
    unittest.main()
