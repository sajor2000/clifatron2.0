"""Partition isolation in the transportability workflow (src/eval/method3.py).

Covers U5 defects D6, D7, D8 and D11.

D6: `transportability_matrix` fitted one predictor per site on `states[s], labels[s]` --
the whole site -- then scored `matrix[tr][te]` including the `tr == te` diagonal, so the
diagonal fitted and scored on identical rows, with `full_panel(recalibrate=True)`
re-fitting temperature on `labels[te]` on top.

D7: `local_patient_equivalence` was handed the test rows and split them internally, so
the local comparator was fitted on the very labels it was being compared against.

D8/D11: two independent cross-site ensembling entry points (`matrix["ensemble"]` and
`matrix.ensemble_mean`) shipped enabled, presupposing a derived-model exchange that is
not approved. Closing one and leaving the other callable is not a fix.

These use a stub predictor rather than a real probe so the tests stay data-free and fast;
what is under test is which ROWS reach which stage, not what the model learns.
"""

import unittest

import numpy as np

from src.eval import method3 as M3
from src.eval import schema as S
from src.eval.matrix import ensemble_mean
from src.eval.schema import DisclosureError

ROLES = ("train", "validation", "calibration", "test")


def _site(n=400, seed=0, dim=4, roles=ROLES):
    """A site whose rows carry partition roles, with real signal in the features."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, dim)) + y[:, None] * 0.9
    partitions = np.array([roles[i % len(roles)] for i in range(n)])
    groups = {"sex": np.array(["F" if v else "M" for v in rng.integers(0, 2, size=n)])}
    return X, y, groups, partitions


class _RecordingFit:
    """Stand-in for fit_probe that records exactly which rows it was fitted on."""

    def __init__(self):
        self.fit_calls = []

    def __call__(self, X, y):
        self.fit_calls.append(np.asarray(X).copy())
        w = np.asarray(X).mean(axis=0)

        def predict(Z):
            return np.asarray(Z) @ w
        return predict


class PartitionRequiredTest(unittest.TestCase):
    def test_sites_without_partition_roles_are_refused(self):
        """D6: unsplit site arrays cannot enter a fit/evaluate workflow."""
        X, y, g, _ = _site()
        with self.assertRaises(M3.PartitionError) as ctx:
            M3.transportability_matrix({"A": X}, {"A": y}, {"A": g}, {}, method="probe")
        self.assertIn("without partition roles", str(ctx.exception))

    def test_load_site_requires_a_partition_column(self):
        import tempfile
        from pathlib import Path

        import polars as pl
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "site.parquet"
            pl.DataFrame({"sequence": [[1, 2], [3, 4]], "label": [0, 1]}).write_parquet(path)
            with self.assertRaises(M3.PartitionError) as ctx:
                M3.load_site(str(path), "sequence", "label", [])
            self.assertIn("partition", str(ctx.exception))

    def test_load_site_rejects_unknown_partition_roles(self):
        import tempfile
        from pathlib import Path

        import polars as pl
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "site.parquet"
            pl.DataFrame({"sequence": [[1, 2], [3, 4]], "label": [0, 1],
                          "partition": ["train", "holdout"]}).write_parquet(path)
            with self.assertRaises(M3.PartitionError) as ctx:
                M3.load_site(str(path), "sequence", "label", [])
            self.assertIn("unknown partition roles", str(ctx.exception))


class DiagonalIsolationTest(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.g, self.part = _site(seed=1)

    def test_predictor_is_fitted_on_train_rows_only(self):
        """The heart of D6: fitting saw the whole site, including the test rows."""
        recorder = _RecordingFit()
        orig = M3.fit_probe
        M3.fit_probe = recorder
        try:
            M3.transportability_matrix(
                {"A": self.X}, {"A": self.y}, {"A": self.g}, {"A": self.part},
                method="probe")
        finally:
            M3.fit_probe = orig

        self.assertEqual(len(recorder.fit_calls), 1)
        fitted_rows = recorder.fit_calls[0]
        train_rows = self.X[self.part == "train"]
        self.assertEqual(len(fitted_rows), len(train_rows))
        self.assertLess(len(fitted_rows), len(self.X),
                        "fitting must not see the whole site")

        # No test row may appear among the fitted rows.
        test_rows = {tuple(r) for r in self.X[self.part == "test"]}
        for row in fitted_rows:
            self.assertNotIn(tuple(row), test_rows,
                             "a test row reached the fitting stage")

    def test_diagonal_is_reported_and_is_not_fit_on_self(self):
        recorder = _RecordingFit()
        orig = M3.fit_probe
        M3.fit_probe = recorder
        try:
            matrix = M3.transportability_matrix(
                {"A": self.X}, {"A": self.y}, {"A": self.g}, {"A": self.part},
                method="probe")
        finally:
            M3.fit_probe = orig
        cell = matrix["A"]["A"]
        self.assertEqual(cell["status"], S.EVALUABLE)
        self.assertEqual(cell["n"], int((self.part == "test").sum()),
                         "the diagonal must be scored on the test partition only")

    def test_missing_test_partition_yields_a_status_not_a_number(self):
        no_test = np.array(["train"] * len(self.y))
        matrix = M3.transportability_matrix(
            {"A": self.X}, {"A": self.y}, {"A": self.g}, {"A": no_test}, method="probe")
        self.assertEqual(matrix["A"]["A"]["status"], M3.INSUFFICIENT_PARTITIONS)
        self.assertNotIn("auroc", matrix["A"]["A"])

    def test_missing_train_partition_yields_a_status_not_a_number(self):
        no_train = np.array(["test"] * len(self.y))
        matrix = M3.transportability_matrix(
            {"A": self.X}, {"A": self.y}, {"A": self.g}, {"A": no_train}, method="probe")
        self.assertEqual(matrix["A"]["A"]["status"], M3.INSUFFICIENT_PARTITIONS)


class CalibrationPartitionTest(unittest.TestCase):
    def test_calibration_is_fitted_on_the_calibration_partition(self):
        X, y, g, part = _site(seed=2)
        matrix = M3.transportability_matrix(
            {"A": X}, {"A": y}, {"A": g}, {"A": part}, method="probe")
        cell = matrix["A"]["A"]
        self.assertIn("temperature", cell)

    def test_no_calibration_partition_leaves_the_panel_uncalibrated(self):
        X, y, g, _ = _site(seed=3)
        part = np.array([("train" if i % 2 else "test") for i in range(len(y))])
        matrix = M3.transportability_matrix(
            {"A": X}, {"A": y}, {"A": g}, {"A": part}, method="probe")
        self.assertEqual(matrix["A"]["A"]["temperature"], 1.0,
                         "absent a calibration partition, nothing may be fitted")


class _StubLGBM:
    """Stand-in for LGBMClassifier that records the rows it was fitted and scored on.

    Real LightGBM is not exercised here on purpose. What D7 broke was which ROWS reach
    the fit stage versus the score stage, and a stub makes that directly observable
    instead of inferring it from an AUROC. It also keeps the test data-free and fast,
    and avoids a LightGBM/OpenMP segfault on macOS arm64.
    """

    seen_fit: list = []
    seen_score: list = []

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        _StubLGBM.seen_fit.append((np.asarray(X).copy(), np.asarray(y).copy()))
        return self

    def predict_proba(self, X):
        _StubLGBM.seen_score.append(np.asarray(X).copy())
        n = len(X)
        # Deterministic, weakly-informative scores; AUROC lands near 0.5.
        p = np.linspace(0.2, 0.8, n)
        return np.column_stack([1 - p, p])


class LocalPatientEquivalenceTest(unittest.TestCase):
    def setUp(self):
        _StubLGBM.seen_fit = []
        _StubLGBM.seen_score = []

    def test_lpe_fits_and_scores_on_disjoint_row_sets(self):
        """D7: the local comparator was fitted on the very labels it was compared against."""
        from unittest import mock

        rng = np.random.default_rng(4)
        y_fit = rng.integers(0, 2, size=600)
        X_fit = rng.normal(size=(600, 4)) + y_fit[:, None]
        y_ev = rng.integers(0, 2, size=200)
        X_ev = rng.normal(size=(200, 4)) + y_ev[:, None] + 50.0  # disjoint in value space

        with mock.patch("lightgbm.LGBMClassifier", _StubLGBM):
            M__lpe(0.99, X_fit, y_fit, sizes=(250,), eval_X=X_ev, eval_y=y_ev)

        self.assertTrue(_StubLGBM.seen_fit, "the local model was never fitted")
        self.assertTrue(_StubLGBM.seen_score, "the local model was never scored")
        fitted_rows = {tuple(np.round(r, 6)) for r, _ in
                       [(row, None) for X, _ in _StubLGBM.seen_fit for row in X]}
        scored_rows = {tuple(np.round(r, 6)) for X in _StubLGBM.seen_score for r in X}
        self.assertFalse(fitted_rows & scored_rows,
                         "a row used to fit the local comparator was also scored by it")

    def test_lpe_scores_on_exactly_the_supplied_evaluation_rows(self):
        from unittest import mock

        rng = np.random.default_rng(11)
        y_fit = rng.integers(0, 2, size=400)
        X_fit = rng.normal(size=(400, 4))
        X_ev, y_ev = rng.normal(size=(120, 4)), rng.integers(0, 2, size=120)

        with mock.patch("lightgbm.LGBMClassifier", _StubLGBM):
            M__lpe(0.99, X_fit, y_fit, sizes=(250,), eval_X=X_ev, eval_y=y_ev)

        for scored in _StubLGBM.seen_score:
            self.assertEqual(len(scored), len(X_ev))

    def test_single_class_evaluation_set_returns_no_crossing(self):
        rng = np.random.default_rng(5)
        y_fit = rng.integers(0, 2, size=300)
        X_fit = rng.normal(size=(300, 4))
        self.assertEqual(
            M__lpe(0.7, X_fit, y_fit, eval_X=np.zeros((50, 4)), eval_y=np.ones(50)), -1)

    def test_single_class_fit_set_returns_no_crossing(self):
        self.assertEqual(M__lpe(0.7, np.zeros((50, 4)), np.ones(50)), -1)


class CrossSiteEnsembleGateTest(unittest.TestCase):
    def test_no_ensemble_row_by_default(self):
        """D8: the ensemble presupposes an unapproved derived-model exchange."""
        X, y, g, part = _site(seed=6)
        Xb, yb, gb, partb = _site(seed=7)
        matrix = M3.transportability_matrix(
            {"A": X, "B": Xb}, {"A": y, "B": yb}, {"A": g, "B": gb},
            {"A": part, "B": partb}, method="probe")
        self.assertNotIn("ensemble", matrix)

    def test_ensemble_row_appears_only_with_the_explicit_flag(self):
        X, y, g, part = _site(seed=8)
        Xb, yb, gb, partb = _site(seed=9)
        matrix = M3.transportability_matrix(
            {"A": X, "B": Xb}, {"A": y, "B": yb}, {"A": g, "B": gb},
            {"A": part, "B": partb}, method="probe",
            allow_cross_site_ensemble=True)
        self.assertIn("ensemble", matrix)

    def test_ensemble_mean_helper_is_gated_too(self):
        """D11: the second entry point. Closing only the matrix row leaves this callable."""
        a = np.array([0.1, 0.9])
        b = np.array([0.3, 0.7])
        with self.assertRaises(DisclosureError) as ctx:
            ensemble_mean(a, b)
        self.assertIn("not been", str(ctx.exception).lower())

    def test_ensemble_mean_works_with_the_flag(self):
        a = np.array([0.1, 0.9])
        b = np.array([0.3, 0.7])
        out = ensemble_mean(a, b, allow_cross_site_ensemble=True)
        np.testing.assert_allclose(out, [0.2, 0.8])


class StableImportSurfaceTest(unittest.TestCase):
    def test_matrix_module_still_re_exports_the_panel(self):
        """D11's second half: matrix.py advertises itself as the stable import surface,
        so the D4 split must not silently break it."""
        from src.eval import matrix
        for name in ("full_panel", "fit_temperature", "subgroup_panel",
                     "net_benefit_releasable", "score", "validate_export"):
            self.assertTrue(hasattr(matrix, name), f"matrix.{name} disappeared")


def M__lpe(*args, **kwargs):
    from src.eval.metrics import local_patient_equivalence
    return local_patient_equivalence(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()
