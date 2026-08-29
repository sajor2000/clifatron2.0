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
        cal = M.fit_temperature(cal_logits, cal_y, partition="S:calibration")
        self.assertIsInstance(cal, M.Calibrator)
        self.assertGreater(float(cal), 0.0)

    def test_temperature_fitted_on_calibration_applies_to_disjoint_test(self):
        logits, y = _separable(n=800, seed=2)
        cal_logits, cal_y = logits[:400], y[:400]
        test_logits, test_y = logits[400:], y[400:]

        T = M.fit_temperature(cal_logits, cal_y, partition="S:calibration")
        uncal = M.full_panel(M._sigmoid(test_logits), test_y, logits=test_logits)
        cal = M.full_panel(M._sigmoid(test_logits), test_y, logits=test_logits,
                           temperature=T, partition="S:test")

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

    def test_calibrator_refuses_to_be_applied_to_its_own_partition(self):
        """Review finding #19. Inverting the default made the leak inconvenient, not
        impossible: a caller could fit on the test rows and pass the scalar straight back."""
        logits, y = _separable(n=300, seed=9)
        leaked = M.fit_temperature(logits, y, partition="S:test")
        with self.assertRaises(ValueError) as ctx:
            M.full_panel(M._sigmoid(logits), y, logits=logits, temperature=leaked,
                         partition="S:test")
        self.assertIn("disjoint", str(ctx.exception))

    def test_applying_a_calibrator_without_naming_the_partition_is_refused(self):
        """Greptile PR #4, round 2. Identity cannot be inferred from values -- in either
        direction -- so a calibrator that cannot verify disjointness says so instead of
        guessing. A control that guesses is worse than one that asks."""
        logits, y = _separable(n=200, seed=12)
        cal = M.fit_temperature(logits, y, partition="S:calibration")
        with self.assertRaises(ValueError) as ctx:
            M.full_panel(M._sigmoid(logits), y, logits=logits, temperature=cal)
        self.assertIn("were not named", str(ctx.exception))

    def test_fit_temperature_requires_a_partition_name(self):
        logits, y = _separable(n=100, seed=13)
        with self.assertRaises(TypeError):
            M.fit_temperature(logits, y)
        with self.assertRaises(ValueError):
            M.fit_temperature(logits, y, partition="")

    def test_identical_values_across_partitions_are_accepted(self):
        """A value check false-REJECTS here. Every saturated logit clamps to one bound,
        so confident predictions sharing a label collide by construction, and a small
        disjoint scoring set can consist entirely of colliding values."""
        cal_logits = np.array([np.inf, np.inf, -np.inf, 0.3, 0.7, -0.2])
        cal_y = np.array([1, 1, 0, 1, 0, 1])
        test_logits = np.array([np.inf, -np.inf])   # every value also in the cal set
        test_y = np.array([1, 0])
        cal = M.fit_temperature(cal_logits, cal_y, partition="S:calibration")
        M.full_panel(M._sigmoid(test_logits), test_y, logits=test_logits,
                     temperature=cal, partition="S:test")

    def test_a_diluted_reuse_is_still_caught(self):
        """A value check false-ACCEPTS here: genuinely reused calibration rows diluted by
        unrelated scored rows fall under any proportional threshold. Declaration does not
        care about the mix."""
        logits, y = _separable(n=400, seed=22)
        cal = M.fit_temperature(logits[:20], y[:20], partition="S:calibration")
        with self.assertRaises(ValueError):
            M.full_panel(M._sigmoid(logits), y, logits=logits, temperature=cal,
                         partition="S:calibration")

    def test_declaration_is_authoritative_over_values(self):
        logits, y = _separable(n=200, seed=21)
        cal = M.fit_temperature(logits, y, partition="SITE-01:calibration")
        # Identical ROWS, different declared partition -> accepted. The declaration is
        # the contract; the values are not evidence either way.
        M.full_panel(M._sigmoid(logits), y, logits=logits, temperature=cal,
                     partition="SITE-01:test")

    def test_calibrator_applies_cleanly_to_disjoint_rows(self):
        logits, y = _separable(n=800, seed=10)
        cal = M.fit_temperature(logits[:400], y[:400], partition="S:calibration")
        M.full_panel(M._sigmoid(logits[400:]), y[400:], logits=logits[400:],
                     temperature=cal, partition="S:test")

    def test_fit_temperature_refuses_a_single_class_calibration_partition(self):
        with self.assertRaises(ValueError):
            M.fit_temperature(np.array([0.2, 0.4, 0.6]), np.array([1, 1, 1]),
                              partition="S:calibration")

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


class NanCohortConsistencyTest(unittest.TestCase):
    """greploop review 1, P1: one report must describe one cohort.

    `full_panel` drops undefined predictions internally; `net_benefit` does not. Handing
    both the same un-narrowed array gives the scalar metrics and the decision curve
    different denominators -- and in the curve, `NaN >= threshold` is False, so every
    undefined row is silently counted as a negative decision. The caller must narrow
    once, before either.
    """

    def _arrays(self, n=200, nan_every=5, seed=3):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=n)
        p = rng.uniform(0.05, 0.95, size=n)
        p[::nan_every] = np.nan
        return p, y

    def test_unnarrowed_arrays_give_the_curve_a_different_cohort(self):
        """The bug this guards against, demonstrated rather than asserted in prose."""
        p, y = self._arrays()
        panel = M.full_panel(p, y)                      # drops NaN internally
        curve_raw = M.net_benefit_releasable(p, y)      # does not

        self.assertEqual(panel["n_dropped_nan"], 40)
        self.assertEqual(panel["n"], 160)
        self.assertIsNotNone(curve_raw)
        # net_benefit divides by len(y), not by the defined count.
        self.assertNotEqual(len(y), panel["n"],
                            "raw arrays carry 200 rows while the panel scored 160")

    def test_narrowing_once_makes_every_number_agree(self):
        p, y = self._arrays()
        defined = ~np.isnan(p)
        p_n, y_n = p[defined], y[defined]

        panel = M.full_panel(p_n, y_n)
        curve = M.net_benefit_releasable(p_n, y_n)

        self.assertEqual(panel["n_dropped_nan"], 0)
        self.assertEqual(panel["n"], len(y_n))
        self.assertIsNotNone(curve)
        # treat_all is prevalence-based, so it pins the curve's cohort to the panel's.
        self.assertAlmostEqual(float(curve["treat_all"][0]),
                               y_n.mean() - (1 - y_n.mean()) * (0.01 / 0.99), places=9)

    def test_nan_reads_as_a_negative_decision_in_the_curve(self):
        """Why leaving NaN in is not merely a denominator problem: the comparison is False."""
        p = np.array([np.nan, np.nan, 0.9, 0.9])
        y = np.array([1, 1, 1, 0])
        self.assertFalse(bool(np.nan >= 0.5), "NaN comparisons are False, not skipped")
        curve = M.net_benefit(p, y, thresholds=np.array([0.5]))
        # 2 of 4 rows are undefined and score as "do not treat", deflating TP.
        self.assertLess(float(curve["model"][0]), 0.5)


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
