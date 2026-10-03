"""U7: count/token baselines, the frozen-state probe and the aggregate task table.

What is under test is which ROWS reach which stage (fit / selection / calibration /
evaluation), that a frozen trunk stays frozen, and that the report states a status
instead of a number whenever a number cannot be released. The end-to-end class runs the
one-command synthetic site once; nothing here touches CLIF data.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.data.cohort import QualificationError
from src.eval import baselines as B
from src.eval import clif_tasks as T
from src.eval import probe as P
from src.eval import schema as S
from src.eval.method3 import PartitionError

ROLES = {"fit": "train", "selection": "validation", "calibration": "calibration",
         "evaluation": "internal_test"}
PARTITIONS = ("train", "train", "train", "validation", "calibration", "internal_test")
MARKER = 0          # feature column carrying each row's unique id (as a "count")
FAST = {
    "seed": 3,
    "lightgbm": {"n_estimators": 60, "early_stopping_rounds": 10, "learning_rate": 0.1,
                 "min_child_samples": 5, "colsample_bytree": 1.0, "n_jobs": 1,
                 "default": {"num_leaves": 7}, "grid": {"num_leaves": [7, 15]}},
    "logistic": {"features": "indicators", "max_iter": 500,
                 "default": {"C": 0.1}, "grid": {"C": [0.01, 1.0]}},
    "probe": {"epochs": 40, "lr": 1.0e-2, "default": {"wd": 1.0e-4},
              "grid": {"wd": [1.0e-4, 1.0e-2]}},
}


def _site(n=600, seed=0, vocab=24):
    """Token-count rows with real signal, a unique marker per row, and partition roles."""
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(float)
    counts = rng.poisson(1.0, size=(n, vocab)).astype(np.float32)
    counts[:, 1:6] += (y[:, None] * rng.poisson(1.5, size=(n, 5))).astype(np.float32)
    counts[:, MARKER] = np.arange(1, n + 1)
    partitions = np.array([PARTITIONS[i % len(PARTITIONS)] for i in range(n)])
    return csr_matrix(counts), y, partitions


def _markers(X) -> set[int]:
    return {int(v) for v in np.asarray(X[:, MARKER].todense()).ravel()}


def _role_markers(X, partitions, name) -> set[int]:
    return _markers(X[np.flatnonzero(partitions == name)])


def _tiny_encoder(vocab_size=32, seed=0):
    from src.model.encoder import CLIFEncoder

    torch.manual_seed(seed)
    return CLIFEncoder(vocab_size, {"trunk": {"d_model": 16, "n_heads": 2, "n_layers": 1,
                                              "ffn_mult": 2, "dropout": 0.0}})


class PartitionIsolationTest(unittest.TestCase):
    """Baselines never see validation, calibration or test rows during fitting."""

    def test_the_fitter_is_handed_fit_rows_and_selection_rows_only(self):
        X, y, partitions = _site()
        seen = {}

        def recording_fit(X_fit, y_fit, X_sel, y_sel):
            seen["fit"], seen["selection"] = _markers(X_fit), _markers(X_sel)
            return (lambda Z: np.zeros(Z.shape[0])), {}

        T.partitioned_cell(recording_fit, X, y, partitions, ROLES, site="S")
        self.assertEqual(seen["fit"], _role_markers(X, partitions, "train"))
        self.assertEqual(seen["selection"], _role_markers(X, partitions, "validation"))
        held_out = (_role_markers(X, partitions, "calibration")
                    | _role_markers(X, partitions, "internal_test"))
        self.assertFalse((seen["fit"] | seen["selection"]) & held_out)

    def test_lightgbm_trains_on_fit_rows_and_stops_early_on_selection_rows(self):
        import lightgbm as lgb

        X, y, partitions = _site()
        calls = []
        real_fit = lgb.LGBMClassifier.fit

        def spy(self, X_arg, y_arg, **kwargs):
            eval_X = kwargs.get("eval_X")
            if eval_X is None and kwargs.get("eval_set"):
                eval_X = kwargs["eval_set"][0][0]
            calls.append((_markers(X_arg), _markers(eval_X) if eval_X is not None else set()))
            return real_fit(self, X_arg, y_arg, **kwargs)

        lgb.LGBMClassifier.fit = spy
        try:
            cell = T.partitioned_cell(B.lightgbm_fitter(FAST["lightgbm"], seed=3),
                                      X, y, partitions, ROLES, site="S")
        finally:
            lgb.LGBMClassifier.fit = real_fit
        self.assertEqual(cell["status"], S.EVALUABLE)
        self.assertEqual(len(calls), 2)                      # one per grid candidate
        for trained_on, stopped_on in calls:
            self.assertEqual(trained_on, _role_markers(X, partitions, "train"))
            self.assertEqual(stopped_on, _role_markers(X, partitions, "validation"))

    def test_logistic_regression_trains_on_fit_rows_only(self):
        from sklearn.linear_model import LogisticRegression

        X, y, partitions = _site()
        n_rows = []
        real_fit = LogisticRegression.fit

        def spy(self, X_arg, y_arg, *args, **kwargs):
            n_rows.append(X_arg.shape[0])
            return real_fit(self, X_arg, y_arg, *args, **kwargs)

        LogisticRegression.fit = spy
        try:
            cell = T.partitioned_cell(B.logistic_fitter(FAST["logistic"], seed=3),
                                      X, y, partitions, ROLES, site="S")
        finally:
            LogisticRegression.fit = real_fit
        self.assertEqual(cell["status"], S.EVALUABLE)
        self.assertEqual(n_rows, [int((partitions == "train").sum())] * 2)

    def test_held_out_rows_cannot_change_a_fitted_baseline(self):
        """Corrupt every calibration and test row: the fitted predictors do not move.
        Corrupt the train rows instead and they do — so the first result is not vacuous."""
        X, y, partitions = _site()
        probe_rows = X[:50]

        def fitted_logits(fitter, X_site, y_site):
            captured = []

            def spy(*args):
                logits_fn, info = fitter(*args)
                captured.append(logits_fn)
                return logits_fn, info

            T.partitioned_cell(spy, X_site, y_site, partitions, ROLES, site="S")
            return captured[0](probe_rows)

        def corrupt(names):
            rng = np.random.default_rng(99)
            dense, labels = X.toarray(), y.copy()
            rows = np.flatnonzero(np.isin(partitions, names))
            dense[rows, 1:] = rng.poisson(3.0, size=(len(rows), dense.shape[1] - 1))
            labels[rows] = 1.0 - labels[rows]
            return type(X)(dense), labels

        for make in (lambda: B.lightgbm_fitter(FAST["lightgbm"], seed=3),
                     lambda: B.logistic_fitter(FAST["logistic"], seed=3)):
            baseline = fitted_logits(make(), X, y)
            held_out = fitted_logits(make(), *corrupt(["calibration", "internal_test"]))
            np.testing.assert_array_equal(baseline, held_out)
            retrained = fitted_logits(make(), *corrupt(["train"]))
            self.assertFalse(np.allclose(baseline, retrained))

    def test_scoring_a_fit_or_calibration_partition_is_refused(self):
        X, y, partitions = _site()
        fit = B.logistic_fitter(FAST["logistic"], seed=3)
        for bad in ("train", "calibration"):
            with self.assertRaises(PartitionError):
                T.partitioned_cell(fit, X, y, partitions, {**ROLES, "evaluation": bad},
                                   site="S")

    def test_a_sealed_partition_needs_the_final_evaluation_flag(self):
        contract = {"sealed_partitions": ["internal_test", "external_confirmation"]}
        with self.assertRaises(PartitionError) as ctx:
            B.check_evaluation_partition(ROLES, contract, final_evaluation=False)
        self.assertIn("sealed", str(ctx.exception))
        B.check_evaluation_partition(ROLES, contract, final_evaluation=True)
        B.check_evaluation_partition({**ROLES, "evaluation": "validation"}, contract,
                                     final_evaluation=False)


class NotEvaluableTest(unittest.TestCase):
    """A task with no positives in a partition is reported as not evaluable."""

    def setUp(self):
        self.X, self.y, self.partitions = _site()
        self.fit = B.logistic_fitter(FAST["logistic"], seed=3)

    def _cell(self, y):
        return T.partitioned_cell(self.fit, self.X, y, self.partitions, ROLES, site="S")

    def test_no_positives_in_the_evaluation_partition(self):
        y = self.y.copy()
        y[self.partitions == "internal_test"] = 0.0
        cell = self._cell(y)
        self.assertEqual(cell["status"], S.SINGLE_CLASS)
        self.assertEqual(cell["display"], "not evaluable")
        self.assertNotIn("auroc", cell)
        self.assertNotIn("n", cell)

    def test_no_positives_in_the_fit_partition(self):
        y = self.y.copy()
        y[self.partitions == "train"] = 0.0
        cell = self._cell(y)
        self.assertEqual(cell["status"], S.SINGLE_CLASS)
        self.assertEqual(cell["display"], "not evaluable")
        self.assertIn("train", cell["reason"])

    def test_fewer_than_ten_positives_is_suppressed_not_scored(self):
        y = self.y.copy()
        test_pos = np.flatnonzero((self.partitions == "internal_test") & (y == 1))
        y[test_pos[3:]] = 0.0
        cell = self._cell(y)
        self.assertEqual(cell["status"], S.SMALL_CELL_SUPPRESSED)
        self.assertEqual(cell["display"], "suppressed (small cell)")
        self.assertNotIn("auroc", cell)
        self.assertNotIn("n", cell)
        self.assertEqual(cell["n_band"], f">={S.min_cell_size()}")

    def test_stays_outside_the_task_are_not_scored(self):
        """NaN labels (prevalent, not ascertainable) leave the at-risk rows only."""
        y = self.y.copy()
        outside = np.flatnonzero(self.partitions == "internal_test")[::2]
        y[outside] = np.nan
        cell = self._cell(y)
        self.assertEqual(cell["status"], S.EVALUABLE)
        self.assertEqual(cell["n"], int((self.partitions == "internal_test").sum())
                         - len(outside))

    def test_a_single_class_selection_partition_falls_back_to_defaults(self):
        y = self.y.copy()
        y[self.partitions == "validation"] = 0.0
        cell = self._cell(y)
        self.assertEqual(cell["status"], S.EVALUABLE)
        self.assertEqual(cell["selection"]["C"], FAST["logistic"]["default"]["C"])
        self.assertIn("defaults", cell["selection"]["note"])

    def test_a_single_class_calibration_partition_leaves_the_cell_uncalibrated(self):
        y = self.y.copy()
        y[self.partitions == "calibration"] = 1.0
        cell = self._cell(y)
        self.assertEqual(cell["status"], S.EVALUABLE)
        self.assertFalse(cell["calibrated"])
        self.assertEqual(cell["temperature"], 1.0)


class FrozenProbeTest(unittest.TestCase):
    """The probe leaves the trunk's parameters without gradients."""

    def _data(self, n=240, seed=1):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, size=n).astype(float)
        sequences = [[int(4 + 10 * label + rng.integers(0, 6)) for _ in range(6)]
                     for label in y]
        positions = [list(range(0, 60 * len(s), 60)) for s in sequences]
        partitions = np.array([PARTITIONS[i % len(PARTITIONS)] for i in range(n)])
        return sequences, positions, y, partitions

    def _assert_frozen(self, trunk, before):
        for name, parameter in trunk.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
            self.assertTrue(torch.equal(parameter, before[name]), name)
        self.assertFalse(trunk.training)

    def test_the_trunk_keeps_no_gradients_and_no_new_weights(self):
        sequences, positions, y, partitions = self._data()
        trunk = _tiny_encoder()
        self.assertTrue(all(p.requires_grad for p in trunk.parameters()))
        before = {k: v.detach().clone() for k, v in trunk.named_parameters()}
        cell = P.frozen_probe(y, partitions, ROLES, site="S", trunk=trunk,
                              sequences=sequences, positions=positions,
                              cfg=FAST["probe"], seed=3)
        self._assert_frozen(trunk, before)
        self.assertEqual(cell["status"], S.EVALUABLE)
        self.assertGreater(cell["auroc"], 0.8)       # the planted token signal is linear

    def test_a_transformers_backbone_is_frozen_the_same_way(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        sequences = self._data()[0]
        torch.manual_seed(0)
        trunk = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=16, n_embd=16,
                                           n_layer=1, n_head=2, bos_token_id=0,
                                           eos_token_id=0))
        before = {k: v.detach().clone() for k, v in trunk.named_parameters()}
        states = P.trunk_states(trunk, sequences)
        self._assert_frozen(trunk, before)
        self.assertEqual(states.shape, (len(sequences), 16))

    def test_padding_does_not_change_a_stays_state(self):
        trunk = _tiny_encoder()
        short, long = [5, 6, 7], [5, 6, 7, 8, 9, 10, 11]
        alone = P.trunk_states(trunk, [short], [[0, 10, 20]])
        batched = P.trunk_states(trunk, [short, long],
                                 [[0, 10, 20], [0, 10, 20, 30, 40, 50, 60]])
        np.testing.assert_allclose(alone[0], batched[0], atol=1e-5)

    def test_precomputed_states_and_a_trunk_are_alternatives(self):
        _, _, y, partitions = self._data()
        states = np.random.default_rng(0).normal(size=(len(y), 8)) + y[:, None]
        cell = P.frozen_probe(y, partitions, ROLES, site="S", states=states,
                              cfg=FAST["probe"], seed=3)
        self.assertEqual(cell["status"], S.EVALUABLE)
        with self.assertRaises(ValueError):
            P.frozen_probe(y, partitions, ROLES, site="S", cfg=FAST["probe"])
        with self.assertRaises(ValueError):
            P.frozen_probe(y, partitions, ROLES, site="S", states=states,
                           trunk=_tiny_encoder(), cfg=FAST["probe"])

    def test_an_empty_sequence_is_refused(self):
        with self.assertRaises(ValueError):
            P.trunk_states(_tiny_encoder(), [[5], []], [[0], []])

    def test_the_probe_is_fitted_on_fit_rows_only(self):
        _, _, y, partitions = self._data()
        rng = np.random.default_rng(0)
        states = rng.normal(size=(len(y), 8)) + y[:, None]
        marked = np.column_stack([np.arange(1, len(y) + 1), states])
        seen = []
        real = P.fit_probe

        def spy(X_tr, y_tr, **kwargs):
            seen.append(len(y_tr))
            return real(X_tr, y_tr, **kwargs)

        P.fit_probe = spy
        try:
            P.frozen_probe(y, partitions, ROLES, site="S", states=marked,
                           cfg=FAST["probe"], seed=3)
        finally:
            P.fit_probe = real
        self.assertEqual(set(seen), {int((partitions == "train").sum())})


class SyntheticSiteReportTest(unittest.TestCase):
    """The one command: baselines + probe end to end on the synthetic site."""

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()
        cls.report = B.run_synthetic()
        cls.rows = {row["row"]: row for row in cls.report["rows"]}
        cls.table = B.render_table(cls.report)

    def test_the_run_leaves_the_callers_working_directory_alone(self):
        self.assertEqual(os.getcwd(), self.cwd)

    def test_baselines_and_probe_are_scored_on_the_same_tasks(self):
        active = {t.name for t in T.load_task_suite().active()}
        for name in (B.ROW_LIGHTGBM, B.ROW_LOGISTIC, B.ROW_SYNTHETIC_PROBE):
            row = self.rows[name]
            self.assertEqual(row["status"], "available", name)
            self.assertEqual(set(row["cells"]), active)
        evaluable = [task for task, cell in self.rows[B.ROW_LIGHTGBM]["cells"].items()
                     if cell["status"] == S.EVALUABLE]
        self.assertGreaterEqual(len(evaluable), 3)
        for task in evaluable:
            cells = [self.rows[name]["cells"][task]
                     for name in (B.ROW_LIGHTGBM, B.ROW_LOGISTIC, B.ROW_SYNTHETIC_PROBE)]
            self.assertEqual({c["n"] for c in cells}, {cells[0]["n"]})   # identical rows
            self.assertEqual({c["prevalence"] for c in cells}, {cells[0]["prevalence"]})

    def test_the_baselines_recover_the_planted_signal(self):
        for name in (B.ROW_LIGHTGBM, B.ROW_LOGISTIC):
            self.assertGreater(self.rows[name]["mean"]["auroc"], 0.65, name)

    def test_a_task_with_no_positives_is_reported_as_not_evaluable(self):
        for name in (B.ROW_LIGHTGBM, B.ROW_LOGISTIC, B.ROW_SYNTHETIC_PROBE):
            cell = self.rows[name]["cells"]["hyponatremia"]
            self.assertEqual(cell["status"], S.SINGLE_CLASS)
            self.assertEqual(cell["display"], "not evaluable")
            self.assertNotIn("auroc", cell)
        self.assertNotIn("hyponatremia", self.rows[B.ROW_LIGHTGBM]["mean"]["tasks"])
        line = next(l for l in self.table.splitlines() if l.startswith("| hyponatremia"))
        self.assertIn("not evaluable", line)

    def test_a_rare_task_is_suppressed_not_scored(self):
        cell = self.rows[B.ROW_LIGHTGBM]["cells"]["hypertension"]
        self.assertEqual(cell["status"], S.SMALL_CELL_SUPPRESSED)
        self.assertNotIn("n", cell)
        line = next(l for l in self.table.splitlines() if l.startswith("| hypertension"))
        self.assertIn("suppressed", line)

    def test_unstaged_comparators_read_not_available_and_the_report_still_builds(self):
        for name in ("clifatron_0p5b_probe", "decile_ntp_probe"):
            row = self.rows[name]
            self.assertEqual(row["status"], "not_available")
            self.assertEqual(row["display"], "not available")
            self.assertIn("no checkpoint staged", row["reason"])
            self.assertNotIn("cells", row)
        self.assertIn("CLIFATRON 0.5B", self.table)
        self.assertIn("not available", self.table)

    def test_the_report_states_which_tasks_were_matched(self):
        comparison = self.report["comparison"]
        self.assertEqual(comparison["counts"]["excluded"], 3)
        self.assertEqual(comparison["tasks"]["vasopressors"]["status"], "excluded")
        self.assertNotIn("vasopressors", self.rows[B.ROW_LIGHTGBM]["cells"])
        self.assertIs(self.report["synthetic"], True)

    def test_the_table_is_aggregate_only(self):
        blob = json.dumps(self.report, allow_nan=False)      # strict JSON, no NaN
        self.assertNotIn("synth-", blob)                      # no stay or patient id
        self.assertNotIn(tempfile.gettempdir(), blob)         # no local path
        floor = S.min_cell_size()
        for row in self.report["rows"]:
            for task, cell in row.get("cells", {}).items():
                if cell["status"] != S.EVALUABLE:
                    self.assertNotIn("n", cell, (row["row"], task))
                    continue
                positives = cell["n"] * cell["prevalence"]
                self.assertGreaterEqual(round(positives), floor - 1)
                self.assertGreaterEqual(cell["n"] - round(positives), floor - 1)


class RunSiteTest(unittest.TestCase):
    """The site entry point: files in, aggregate table out, sealed partition guarded."""

    @classmethod
    def setUpClass(cls):
        import yaml

        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import FIXTURE_POLICY

        cls._td = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._td.cleanup)
        work = Path(cls._td.name).resolve()
        cwd = os.getcwd()
        os.chdir(work)          # tokenizer CWD contract: shards under output/intermediate_phi
        try:
            cls.site = work / "site"
            cls.episodes = B.build_synthetic_task_site(cls.site)
            data_cfg = B.synthetic_data_config(work)
            cls.shard_dir = work / "output/intermediate_phi/site"
            tokenize_site(data_cfg, B.SYNTHETIC_TASK_SITE, cls.site, cls.shard_dir, None,
                          episodes=pl.read_parquet(cls.episodes),
                          artifact_policy=FIXTURE_POLICY, report=False)
            cls.data_config = work / "data_config.yaml"
            cls.data_config.write_text(yaml.safe_dump(data_cfg))
        finally:
            os.chdir(cwd)

    def _run(self, **kwargs):
        return B.run_site(site=B.SYNTHETIC_TASK_SITE, data_dir=self.site,
                          episodes_path=self.episodes,
                          shards_path=self.shard_dir / "events.parquet",
                          vocab_path=self.shard_dir / "vocab.json",
                          data_config=self.data_config, **kwargs)

    def test_the_sealed_test_partition_is_refused_by_default(self):
        with self.assertRaises(PartitionError) as ctx:
            self._run()
        self.assertIn("internal_test", str(ctx.exception))

    def test_the_development_view_scores_validation_and_says_so(self):
        report = self._run(eval_partition="validation")
        self.assertEqual(report["roles"]["evaluation"], "validation")
        self.assertIs(report["synthetic"], False)
        self.assertTrue(any("optimistic" in note for note in report["notes"]))
        self.assertTrue(any("time zero is approximated" in note for note in report["notes"]))
        rows = {row["row"]: row for row in report["rows"]}
        self.assertEqual(rows[B.ROW_LIGHTGBM]["cells"]["expired"]["status"], S.EVALUABLE)
        self.assertEqual(rows["clifatron_0p5b_probe"]["status"], "not_available")
        self.assertNotIn(str(self.site), json.dumps(report))

    def test_the_final_evaluation_matches_the_one_command_run(self):
        """Same site, same seed: the file-based entry point and --synthetic agree."""
        final = self._run(final_evaluation=True)
        synthetic = B.run_synthetic()
        for index in (0, 1):                                  # the two baselines
            self.assertEqual(final["rows"][index]["cells"],
                             synthetic["rows"][index]["cells"])


class ComparatorRowTest(unittest.TestCase):
    def _labels(self, n=240):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, size=n)
        ids = [f"stay-{i:04d}" for i in range(n)]
        labels = pl.DataFrame({
            "hospitalization_id": ids,
            "partition": [PARTITIONS[i % len(PARTITIONS)] for i in range(n)],
            "expired": y.astype(bool),
            "expired_status": ["positive" if v else "negative" for v in y],
        })
        shards = pl.DataFrame({
            "hosp_id": ids,
            "token": [[int(4 + 10 * v + rng.integers(0, 6)) for _ in range(5)] for v in y],
        })
        return labels, shards

    def _suite(self):
        import yaml

        blob = yaml.safe_load(Path(T.DEFAULT_TASKS).read_text())
        for name, task in blob["tasks"].items():
            if name != "expired" and task["active"]:
                task["active"] = False
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tasks.yaml"
            path.write_text(yaml.safe_dump(blob))
            return T.load_task_suite(path)

    def test_a_missing_checkpoint_or_sequence_file_reads_not_available(self):
        labels, _ = self._labels()
        suite = self._suite()
        spec = {"label": "CLIFATRON 0.5B frozen probe", "checkpoint": None, "shards": None}
        row = B.comparator_row("clifatron_0p5b_probe", spec, labels, None, suite, ROLES,
                               site="S", cfg=FAST)
        self.assertEqual((row["status"], row["display"]), ("not_available", "not available"))
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "ckpt").mkdir()
            spec = {**spec, "checkpoint": str(Path(td) / "ckpt"), "shards": None}
            row = B.comparator_row("clifatron_0p5b_probe", spec, labels, None, suite,
                                   ROLES, site="S", cfg=FAST)
        self.assertEqual(row["status"], "not_available")
        self.assertIn("token sequences", row["reason"])
        self.assertNotIn(td, json.dumps(row))                 # no local path in the row

    def test_a_staged_checkpoint_fills_the_row(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        labels, shards = self._labels()
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "ckpt"
            torch.manual_seed(0)
            GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=16, n_embd=16, n_layer=1,
                                       n_head=2, bos_token_id=0, eos_token_id=0)
                            ).save_pretrained(ckpt)
            shards.write_parquet(Path(td) / "sequences.parquet")
            spec = {"label": "CLIFATRON 0.5B frozen probe", "checkpoint": str(ckpt),
                    "shards": str(Path(td) / "sequences.parquet")}
            row = B.comparator_row("clifatron_0p5b_probe", spec, labels, None,
                                   self._suite(), ROLES, site="S", cfg=FAST)
        self.assertEqual(row["status"], "available")
        self.assertEqual(row["cells"]["expired"]["status"], S.EVALUABLE)
        self.assertGreater(row["cells"]["expired"]["auroc"], 0.7)
        self.assertEqual(row["feature_window"], "as tokenized by the comparator (not re-checked)")


class WriteReportTest(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._td = tempfile.TemporaryDirectory()
        os.chdir(self._td.name)
        self.addCleanup(self._td.cleanup)
        self.addCleanup(os.chdir, self._cwd)
        self.report = {"report": "clif_task_suite", "rows": [
            {"row": "lightgbm_counts", "label": "LightGBM", "status": "available",
             "cells": {}}]}

    def test_the_table_is_written_under_the_aggregate_directory(self):
        out = Path("output/final_no_phi/clif_task_baselines.json")
        B.write_report(self.report, out, identifiers=["stay-1"])
        self.assertEqual(json.loads(out.read_text())["report"], "clif_task_suite")

    def test_a_destination_outside_the_aggregate_class_is_refused(self):
        with self.assertRaises(ValueError):
            B.write_report(self.report, Path("output/intermediate_phi/table.json"),
                           identifiers=[])
        with self.assertRaises(ValueError):
            B.write_report(self.report, Path("output/final_no_phi/table.parquet"),
                           identifiers=[])

    def test_an_identifier_in_the_table_is_refused(self):
        leaky = {**self.report, "note": "stay-1"}
        with self.assertRaises(QualificationError):
            B.write_report(leaky, Path("output/final_no_phi/table.json"),
                           identifiers=["stay-1"])
        self.assertFalse(Path("output/final_no_phi/table.json").exists())


if __name__ == "__main__":
    unittest.main()
