"""Calibration-partition isolation and subgroup suppression (src/eval/metrics.py).

Covers U5 defects D4 and D5.

D4: `full_panel(..., recalibrate=True)` fitted `temperature_scale(logits, y)` on the
same `y` it then scored, so calibration slope, ECE, ICI, Brier and DCA were every one
of them fitted on their own test labels. The leak was the DEFAULT argument, not an
unusual call, which is why the fix inverts the default rather than documenting a caveat.
scikit-learn states the same invariant for its own prefit path -- "The user has to take
care manually that data for model fitting and calibration are disjoint" -- so
disjointness is the caller's responsibility and must therefore be a named argument.

D5: `subgroup_panel` silently dropped cells with n < 30 -- no status emitted, no
complementary suppression, and a threshold that disagreed with the landed
`minimum_cell_size: 10` for no recorded reason.
"""

import unittest

import numpy as np

from src.eval import metrics as M
from src.eval import schema as S


def _separable(n=400, seed=0, shift=1.2):
    """Logits with real signal, so calibration changes numbers without destroying order."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    logits = rng.normal(0.0, 1.0, size=n) + shift * y
    return logits, y


class CalibrationPartitionTest(unittest.TestCase):
    def test_full_panel_does_not_calibrate_by_default(self):
        """The regression guard for D4. If this fails, the leak is back as a default."""
        logits, y = _separable()
        p = M._sigmoid(logits)
        panel = M.full_panel(p, y, logits=logits)
        self.assertEqual(panel["temperature"], 1.0,
                         "full_panel must not fit temperature on the labels it scores")

    def test_fit_temperature_is_a_two_argument_api(self):
        """Calibration is fitted on one partition and applied to another. The signature
        itself is the control: there is no way to call it with a single array."""
        cal_logits, cal_y = _separable(seed=1)
        cal = M.fit_temperature(cal_logits, cal_y)
        self.assertIsInstance(cal, M.Calibrator)
        self.assertGreater(float(cal), 0.0)

    def test_temperature_fitted_on_calibration_applies_to_disjoint_test(self):
        logits, y = _separable(n=800, seed=2)
        cal_logits, cal_y = logits[:400], y[:400]
        test_logits, test_y = logits[400:], y[400:]

        T = M.fit_temperature(cal_logits, cal_y)
        uncal = M.full_panel(M._sigmoid(test_logits), test_y, logits=test_logits)
        cal = M.full_panel(M._sigmoid(test_logits), test_y, logits=test_logits, temperature=T)

        self.assertEqual(cal["temperature"], float(T))
        self.assertAlmostEqual(uncal["auroc"], cal["auroc"], places=10,
                               msg="temperature scaling is monotone; it cannot change ranking")
        self.assertNotAlmostEqual(uncal["brier"], cal["brier"], places=12,
                                  msg="calibration must actually change calibration numbers")

    def test_calibration_cannot_change_test_labels_or_discrimination_ordering(self):
        logits, y = _separable(n=600, seed=3)
        base = M.full_panel(M._sigmoid(logits), y, logits=logits)
        for T in (0.5, 1.7, 3.0):
            scaled = M.full_panel(M._sigmoid(logits), y, logits=logits, temperature=T)
            self.assertAlmostEqual(base["auroc"], scaled["auroc"], places=10)
            self.assertAlmostEqual(base["auprc"], scaled["auprc"], places=10)
            self.assertEqual(base["n"], scaled["n"])
            self.assertEqual(base["prevalence"], scaled["prevalence"])

    def test_fitting_on_evaluated_labels_requires_an_explicit_unsafe_name(self):
        """The single-array helper survives for synthetic tests, but a caller has to
        name what it is doing. It cannot be reached by accident."""
        logits, y = _separable(seed=4)
        leaked = M.full_panel(M._sigmoid(logits), y, logits=logits,
                              unsafe_fit_on_eval_labels=True)
        self.assertNotEqual(leaked["temperature"], 1.0)

    def test_temperature_and_unsafe_flag_are_mutually_exclusive(self):
        logits, y = _separable(seed=5)
        with self.assertRaises(ValueError):
            M.full_panel(M._sigmoid(logits), y, logits=logits, temperature=1.5,
                         unsafe_fit_on_eval_labels=True)

    def test_calibrator_refuses_to_be_applied_to_the_rows_it_was_fitted_on(self):
        """Review finding #19. Inverting the default made the leak inconvenient, not
        impossible: a caller could fit on the test rows and pass the scalar straight
        back. The calibrator now carries the identity of the rows it saw."""
        logits, y = _separable(n=300, seed=9)
        leaked = M.fit_temperature(logits, y)
        with self.assertRaises(ValueError) as ctx:
            M.full_panel(M._sigmoid(logits), y, logits=logits, temperature=leaked)
        self.assertIn("disjoint", str(ctx.exception))

    def test_disjoint_rows_sharing_values_are_not_falsely_rejected(self):
        """Greptile PR #4. `_clamp_saturated_logits` maps every saturated logit to the
        SAME bound, so two confident predictions sharing a label collide by construction
        -- and a single collision used to reject a genuinely disjoint split."""
        cal_logits = np.array([np.inf, np.inf, -np.inf, 0.3, 0.7, -0.2])
        cal_y = np.array([1, 1, 0, 1, 0, 1])
        test_logits = np.array([np.inf, -np.inf, 0.9, -0.9])
        test_y = np.array([1, 0, 1, 0])
        cal = M.fit_temperature(cal_logits, cal_y)
        M.full_panel(M._sigmoid(test_logits), test_y, logits=test_logits, temperature=cal)

    def test_declared_partitions_are_the_exact_check(self):
        """Naming the partition removes the heuristic entirely: no false positives,
        no false negatives."""
        logits, y = _separable(n=200, seed=21)
        cal = M.fit_temperature(logits, y, partition="SITE-01:calibration")
        with self.assertRaises(ValueError) as ctx:
            M.full_panel(M._sigmoid(logits), y, logits=logits, temperature=cal,
                         partition="SITE-01:calibration")
        self.assertIn("SITE-01:calibration", str(ctx.exception))
        # A different partition is accepted even though the ROWS are identical --
        # the declaration is authoritative over the values.
        M.full_panel(M._sigmoid(logits), y, logits=logits, temperature=cal,
                     partition="SITE-01:test")

    def test_calibrator_applies_cleanly_to_disjoint_rows(self):
        logits, y = _separable(n=800, seed=10)
        cal = M.fit_temperature(logits[:400], y[:400])
        M.full_panel(M._sigmoid(logits[400:]), y[400:], logits=logits[400:], temperature=cal)

    def test_fit_temperature_refuses_a_single_class_calibration_partition(self):
        with self.assertRaises(ValueError):
            M.fit_temperature(np.array([0.2, 0.4, 0.6]), np.array([1, 1, 1]))

    def test_nan_handling_survives_the_split(self):
        """Regression guard: NaN policy landed in 79684c5 and must not be lost."""
        p = np.array([0.2, 0.8, np.nan, 0.6])
        y = np.array([0, 1, 1, 0])
        panel = M.full_panel(p, y, nan_policy="drop")
        self.assertEqual(panel["n_dropped_nan"], 1)
        self.assertEqual(panel["n"], 3)
        with self.assertRaises(ValueError):
            M.full_panel(p, y, nan_policy="raise")


class SubgroupSuppressionTest(unittest.TestCase):
    def _groups(self, n_by_cat):
        vals, ys, ps = [], [], []
        rng = np.random.default_rng(7)
        for cat, n in n_by_cat.items():
            vals += [cat] * n
            y = rng.integers(0, 2, size=n)
            ys.append(y)
            ps.append(rng.uniform(size=n) * 0.5 + 0.25 * y)
        return (np.concatenate(ps), np.concatenate(ys), {"sex": np.array(vals)})

    def test_small_cell_is_reported_as_a_status_not_silently_dropped(self):
        """D5: a dropped cell is indistinguishable from a cell that was never
        evaluated, and it is recoverable by differencing."""
        p, y, groups = self._groups({"F": 400, "M": 300, "X": 4})
        panel = M.subgroup_panel(p, y, groups)
        self.assertIn("X", panel["sex"], "suppressed cells must still appear, with a status")
        self.assertIn(panel["sex"]["X"]["status"], S.NON_EVALUABLE_STATUSES)
        self.assertNotIn("auroc", panel["sex"]["X"])

    def test_threshold_is_the_landed_policy_value_not_a_local_30(self):
        """A cell of n=20 was silently dropped by the old hard-coded 30 despite
        clearing the repo-wide minimum_cell_size of 10."""
        p, y, groups = self._groups({"F": 400, "M": 300, "Y": 20})
        panel = M.subgroup_panel(p, y, groups)
        self.assertEqual(panel["sex"]["Y"]["status"], S.EVALUABLE,
                         f"n=20 clears MIN_CELL_SIZE={S.min_cell_size()}")

    def test_a_lone_suppressed_cell_cannot_be_recovered_by_differencing(self):
        p, y, groups = self._groups({"F": 400, "M": 300, "X": 4})
        panel = M.subgroup_panel(p, y, groups)
        suppressed = [c for c, v in panel["sex"].items()
                      if v["status"] in S.NON_EVALUABLE_STATUSES]
        self.assertGreaterEqual(len(suppressed), 2,
                                "one suppressed cell among released siblings is a "
                                "subtraction away from the attribute total")

    def test_exported_prevalence_is_rounded(self):
        p, y, groups = self._groups({"F": 400, "M": 300})
        panel = M.subgroup_panel(p, y, groups)
        for cell in panel["sex"].values():
            if cell["status"] == S.EVALUABLE:
                self.assertEqual(cell["prevalence"],
                                 S.round_prevalence(cell["prevalence"]),
                                 "exported prevalence must already be quantised")

    def test_cells_carry_no_patient_level_fields(self):
        p, y, groups = self._groups({"F": 400, "M": 300})
        panel = M.subgroup_panel(p, y, groups)
        for cell in panel["sex"].values():
            for key in cell:
                self.assertIn(key, S.CELL_FIELDS)


class NetBenefitResolutionTest(unittest.TestCase):
    def test_small_cell_gets_no_curve(self):
        rng = np.random.default_rng(0)
        n = S.curve_release_min() - 1
        y = rng.integers(0, 2, size=n)
        p = rng.uniform(size=n)
        self.assertIsNone(M.net_benefit_releasable(p, y))

    def test_large_cell_gets_a_curve(self):
        rng = np.random.default_rng(0)
        n = S.curve_release_min() * 4
        y = rng.integers(0, 2, size=n)
        p = rng.uniform(size=n)
        curve = M.net_benefit_releasable(p, y)
        self.assertIsNotNone(curve)
        self.assertIn("thresholds", curve)


if __name__ == "__main__":
    unittest.main()
