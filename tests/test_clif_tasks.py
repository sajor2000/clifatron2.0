"""U7: the post-24-hour CLIF task suite (configs/tasks.yaml, src/eval/clif_tasks.py).

Covers the published-table record, the hard-rule-1 gate on treatment-initiation tasks,
and the two windows: labels read only what became available after hour 24, features only
what was available by hour 24. Everything is hand-built — no CLIF data is touched.
"""

import copy
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.data.cohort import QualificationError
from src.eval import clif_tasks as T

ROOT = Path(__file__).parents[1]
TASKS = ROOT / "configs/tasks.yaml"

T0 = datetime(2026, 1, 1, tzinfo=UTC)          # ICU admission of every hand-built stay
ANCHOR = T0 + timedelta(hours=24)
DISCHARGE = T0 + timedelta(days=5)

# The tables the hand-built sites carry. `meds` is input-only, as in configs/data.yaml.
DATA_CFG = {
    "tables": {
        "vitals": {"file": "clif_vitals", "availability_col": "recorded_dttm",
                   "availability": "missing_storetime", "concept_col": "vital_category",
                   "value_col": "vital_value"},
        "labs": {"file": "clif_labs", "availability_col": "lab_result_dttm",
                 "availability": "result", "concept_col": "lab_category",
                 "value_col": "lab_value_numeric", "unit_col": "reference_unit"},
        "meds": {"file": "clif_medication_admin_continuous", "availability_col": "admin_dttm",
                 "availability": "recorded", "concept_col": "med_category",
                 "value_col": "med_dose", "input_only": True},
    },
}


def _suite(mutate=None, data_cfg=DATA_CFG):
    """Load the repo suite, optionally after editing a copy of its YAML."""
    blob = yaml.safe_load(TASKS.read_text())
    if mutate is not None:
        blob = copy.deepcopy(blob)
        mutate(blob)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tasks.yaml"
        path.write_text(yaml.safe_dump(blob))
        return T.load_task_suite(path, data_cfg=data_cfg)


def _episodes(ids, *, discharge_category=None, eligible=None):
    n = len(ids)
    return pl.DataFrame({
        "hospitalization_id": ids,
        "partition": ["train"] * n,
        "eligible": eligible or [True] * n,
        "icu_admit_dttm": [T0] * n,
        "anchor_dttm": [ANCHOR] * n,
        "discharge_dttm": [DISCHARGE] * n,
        "discharge_category": discharge_category or ["Home"] * n,
    })


def _obs(rows):
    """(stay, offset from ICU admission, concept, value[, unit]) -> observation frame."""
    return pl.DataFrame({
        "hospitalization_id": [r[0] for r in rows],
        "dttm": [T0 + r[1] for r in rows],
        "concept": [r[2] for r in rows],
        "value": [float(r[3]) for r in rows],
        "unit": [r[4] if len(r) > 4 else None for r in rows],
    }, schema={"hospitalization_id": pl.String, "dttm": pl.Datetime("us", "UTC"),
               "concept": pl.String, "value": pl.Float64, "unit": pl.String})


def _status(labels, task):
    return dict(zip(labels["hospitalization_id"], labels[f"{task}_status"]))


def _label(labels, task):
    return dict(zip(labels["hospitalization_id"], labels[task]))


H = timedelta(hours=1)


class PublishedTableTest(unittest.TestCase):
    def test_the_repo_suite_records_all_twelve_published_tasks(self):
        suite = T.load_task_suite(TASKS)
        self.assertEqual(len(suite.tasks), suite.source["n_published_tasks"])
        self.assertEqual(len(suite.tasks), 12)
        self.assertIn(suite.source["status"], ("retrieved", "unverified"))
        for task in suite.tasks.values():
            self.assertTrue(task.published, task.name)
            self.assertIn(task.status, T.STATUSES)
        excluded = {t.name for t in suite.tasks.values() if t.status == "excluded"}
        self.assertEqual(excluded, {"crrt", "imv", "vasopressors"})
        self.assertEqual({t.name for t in suite.active()},
                         set(suite.tasks) - excluded)

    def test_every_excluded_task_says_why_and_is_inactive(self):
        suite = T.load_task_suite(TASKS)
        for name in ("crrt", "imv", "vasopressors"):
            task = suite.tasks[name]
            self.assertFalse(task.active)
            self.assertEqual(task.kind, "treatment_initiation")
            self.assertIn("rule #1", task.reason)

    def test_the_comparison_states_which_tasks_were_matched(self):
        comparison = T.load_task_suite(TASKS).comparison()
        self.assertEqual(comparison["counts"],
                         {"matched": 9, "approximated": 0, "excluded": 3})
        self.assertEqual(comparison["anchor"]["alignment"], "approximated")
        self.assertEqual(comparison["tasks"]["imv"]["status"], "excluded")
        self.assertIn("published", comparison["tasks"]["anemia"])
        self.assertIn("ours", comparison["tasks"]["anemia"])

    def test_an_unverified_table_is_carried_into_the_comparison(self):
        suite = _suite(lambda b: b["source"].update(status="unverified"))
        self.assertEqual(suite.comparison()["source"]["status"], "unverified")


class TreatmentGateTest(unittest.TestCase):
    def test_an_active_treatment_initiation_task_is_rejected(self):
        for name in ("imv", "crrt", "vasopressors"):
            with self.assertRaises(T.TreatmentTargetError) as ctx:
                _suite(lambda b, name=name: b["tasks"][name].update(active=True))
            self.assertIn(name, str(ctx.exception))
            self.assertIn("rule #1", str(ctx.exception))

    def test_relabelling_the_kind_does_not_get_a_treatment_past_the_gate(self):
        """A threshold task whose source is an input-only table is the same violation."""
        def add(blob):
            blob["tasks"]["new_vasopressor"] = {
                "status": "approximated", "how": "any norepinephrine dose",
                "active": True, "kind": "threshold",
                "published": "administration of vasopressors",
                "ours": "norepinephrine dose > 0",
                "any_of": [{"source": "meds", "concept": "norepinephrine", "op": "gt",
                            "threshold": 0.0}],
            }
            blob["source"]["n_published_tasks"] = 13
        with self.assertRaises(T.TreatmentTargetError) as ctx:
            _suite(add)
        self.assertIn("new_vasopressor", str(ctx.exception))

    def test_an_excluded_task_cannot_be_active(self):
        with self.assertRaises(T.TaskConfigError):
            _suite(lambda b: b["tasks"]["anemia"].update(status="excluded",
                                                         reason="left out"))

    def test_an_approximated_task_must_say_how(self):
        with self.assertRaises(T.TaskConfigError) as ctx:
            _suite(lambda b: b["tasks"]["anemia"].update(status="approximated"))
        self.assertIn("how", str(ctx.exception))

    def test_unknown_status_operator_and_source_are_rejected(self):
        with self.assertRaises(T.TaskConfigError):
            _suite(lambda b: b["tasks"]["anemia"].update(status="close_enough"))
        with self.assertRaises(T.TaskConfigError):
            _suite(lambda b: b["tasks"]["anemia"]["any_of"][0].update(op="approx"))
        with self.assertRaises(T.TaskConfigError):
            _suite(lambda b: b["tasks"]["anemia"]["any_of"][0].update(source="notes"))

    def test_the_recorded_task_count_must_match_the_published_one(self):
        with self.assertRaises(T.TaskConfigError):
            _suite(lambda b: b["tasks"].pop("tachycardia"))


class LabelWindowTest(unittest.TestCase):
    """Labels read only what became available after hour 24."""

    @classmethod
    def setUpClass(cls):
        cls.suite = T.load_task_suite(TASKS, data_cfg=DATA_CFG)

    def _labels(self, rows, ids, **kwargs):
        frame = _obs(rows)
        events = {"labs": frame, "vitals": frame}
        return T.derive_task_labels(_episodes(ids, **kwargs), events, self.suite)

    def test_a_label_uses_only_information_after_hour_24(self):
        ids = ["before", "at-anchor", "after", "never", "post-discharge"]
        rows = [
            ("before", 3 * H, "hemoglobin", 6.5, "g/dL"),
            ("at-anchor", 24 * H, "hemoglobin", 6.5, "g/dL"),
            ("after", 24 * H + timedelta(minutes=1), "hemoglobin", 6.5, "g/dL"),
            ("never", 30 * H, "hemoglobin", 9.0, "g/dL"),
            ("post-discharge", 24 * 5 * H + H, "hemoglobin", 6.5, "g/dL"),
        ]
        labels = self._labels(rows, ids)
        self.assertEqual(_status(labels, "anemia"), {
            "before": "prevalent", "at-anchor": "prevalent", "after": "positive",
            "never": "negative", "post-discharge": "negative"})
        self.assertEqual(_label(labels, "anemia"), {
            "before": None, "at-anchor": None, "after": True, "never": False,
            "post-discharge": False})

    def test_pre_anchor_values_cannot_move_a_label(self):
        """Any pre-anchor history that is not the outcome itself leaves labels unchanged."""
        ids = ["a", "b"]
        post = [("a", 30 * H, "hemoglobin", 6.0, "g/dL"),
                ("b", 30 * H, "hemoglobin", 11.0, "g/dL")]
        calm = [("a", 2 * H, "hemoglobin", 12.0, "g/dL"),
                ("b", 2 * H, "hemoglobin", 12.0, "g/dL")]
        stormy = [("a", 2 * H, "hemoglobin", 7.1, "g/dL"),
                  ("b", 2 * H, "hemoglobin", 7.0, "g/dL"),
                  ("b", 5 * H, "hemoglobin", 30.0, "g/dL")]
        self.assertTrue(
            self._labels(post + calm, ids).equals(self._labels(post + stormy, ids)))

    def test_cut_offs_are_closed_exactly_as_published(self):
        ids = ["k-at", "k-below", "hgb-at", "na-at"]
        rows = [
            ("k-at", 30 * H, "potassium", 6.5, "mmol/L"),       # >= 6.5
            ("k-below", 30 * H, "potassium", 6.4, "mmol/L"),
            ("hgb-at", 30 * H, "hemoglobin", 7.0, "g/dL"),      # < 7.0 is strict
            ("na-at", 30 * H, "sodium", 120.0, "mmol/L"),       # < 120 is strict
        ]
        labels = self._labels(rows, ids)
        self.assertEqual(_status(labels, "hyperkalemia")["k-at"], "positive")
        self.assertEqual(_status(labels, "hyperkalemia")["k-below"], "negative")
        self.assertEqual(_status(labels, "anemia")["hgb-at"], "negative")
        self.assertEqual(_status(labels, "hyponatremia")["na-at"], "negative")

    def test_a_composite_task_fires_on_either_condition(self):
        ids = ["map", "sbp", "neither", "sbp-early"]
        rows = [
            ("map", 30 * H, "map", 60, None), ("map", 30 * H, "sbp", 110, None),
            ("sbp", 30 * H, "map", 80, None), ("sbp", 30 * H, "sbp", 85, None),
            ("neither", 30 * H, "map", 80, None), ("neither", 30 * H, "sbp", 110, None),
            # one arm before the anchor is a prior occurrence of the composite outcome
            ("sbp-early", 2 * H, "sbp", 85, None), ("sbp-early", 30 * H, "map", 60, None),
        ]
        self.assertEqual(_status(self._labels(rows, ids), "hypotension"), {
            "map": "positive", "sbp": "positive", "neither": "negative",
            "sbp-early": "prevalent"})

    def test_a_missing_value_never_crosses_a_threshold(self):
        ids = ["nan"]
        rows = [("nan", 30 * H, "sodium", float("nan"), "mmol/L"),
                ("nan", 30 * H, "potassium", float("nan"), "mmol/L")]
        labels = self._labels(rows, ids)
        self.assertEqual(_status(labels, "hypernatremia")["nan"], "negative")
        self.assertEqual(_status(labels, "hyperkalemia")["nan"], "negative")

    def test_death_is_read_from_the_discharge_disposition(self):
        ids = ["died", "home", "unknown", "blank"]
        labels = self._labels([], ids,
                              discharge_category=["Expired", "Home", None, "Missing"])
        self.assertEqual(_status(labels, "expired"), {
            "died": "positive", "home": "negative",
            "unknown": "not_ascertainable", "blank": "not_ascertainable"})
        self.assertIsNone(_label(labels, "expired")["unknown"])

    def test_a_missing_source_table_is_unsupported_not_negative(self):
        frame = _obs([("a", 30 * H, "map", 60, None)])
        labels = T.derive_task_labels(_episodes(["a"]), {"vitals": frame, "labs": None},
                                      self.suite)
        self.assertEqual(_status(labels, "anemia")["a"], "unsupported_at_site")
        self.assertIsNone(_label(labels, "anemia")["a"])
        self.assertEqual(_status(labels, "hypotension")["a"], "positive")

    def test_a_non_canonical_unit_fails_closed(self):
        with self.assertRaises(QualificationError) as ctx:
            self._labels([("a", 30 * H, "hemoglobin", 65.0, "g/L")], ["a"])
        self.assertIn("anemia", str(ctx.exception))

    def test_ineligible_stays_get_no_row(self):
        labels = self._labels([("in", 30 * H, "hemoglobin", 6.0, "g/dL"),
                               ("out", 30 * H, "hemoglobin", 6.0, "g/dL")],
                              ["in", "out"], eligible=[True, False])
        self.assertEqual(labels["hospitalization_id"].to_list(), ["in"])

    def test_no_treatment_task_ever_reaches_the_labels(self):
        labels = self._labels([], ["a"])
        for name in ("crrt", "imv", "vasopressors"):
            self.assertNotIn(name, labels.columns)


class AvailabilityOrderTest(unittest.TestCase):
    """The clock is the table's availability column (plus its lag), never chart time."""

    def _site(self, td, lab_rows, lag=0):
        base = Path(td)
        pl.DataFrame({
            "hospitalization_id": [r[0] for r in lab_rows],
            "lab_collect_dttm": [T0 + r[1] for r in lab_rows],
            "lab_result_dttm": [T0 + r[2] for r in lab_rows],
            "lab_category": ["hemoglobin"] * len(lab_rows),
            "lab_value_numeric": [r[3] for r in lab_rows],
            "reference_unit": ["g/dL"] * len(lab_rows),
        }, schema_overrides={"lab_collect_dttm": pl.Datetime("us", "UTC"),
                             "lab_result_dttm": pl.Datetime("us", "UTC")},
        ).write_parquet(base / "clif_labs.parquet")
        cfg = copy.deepcopy(DATA_CFG)
        cfg["tables"]["labs"]["availability_lag_minutes"] = lag
        suite = T.load_task_suite(TASKS, data_cfg=cfg)
        events = T.load_task_events(base, suite, cfg)
        return T.derive_task_labels(_episodes(["a"]), events, suite), events

    def test_a_result_collected_before_but_available_after_the_anchor_is_a_label(self):
        with tempfile.TemporaryDirectory() as td:
            labels, events = self._site(td, [("a", 23 * H, 25 * H, 6.0)])
        self.assertEqual(_status(labels, "anemia")["a"], "positive")
        self.assertIsNone(events["vitals"])          # table absent at this site

    def test_the_declared_availability_lag_is_applied_before_windowing(self):
        row = [("a", 23 * H, 23 * H + timedelta(minutes=30), 6.0)]
        with tempfile.TemporaryDirectory() as td:
            unlagged, _ = self._site(td, row)
        with tempfile.TemporaryDirectory() as td:
            lagged, _ = self._site(td, row, lag=60)
        self.assertEqual(_status(unlagged, "anemia")["a"], "prevalent")
        self.assertEqual(_status(lagged, "anemia")["a"], "positive")


class FeatureWindowTest(unittest.TestCase):
    """Features read only what was available by hour 24."""

    def _shards(self, tokens, pos_min, ids=("a",)):
        return pl.DataFrame({
            "hosp_id": list(ids), "partition": ["train"] * len(ids),
            "token": tokens, "pos_min": pos_min,
        }, schema_overrides={"token": pl.List(pl.Int64), "pos_min": pl.List(pl.Int64)})

    def test_features_use_only_information_up_to_hour_24(self):
        base = self._shards([[5, 6, 7]], [[0, 600, 1440]])
        leaky = self._shards([[5, 6, 7, 8, 9]], [[0, 600, 1440, 1441, 4000]])
        episodes = _episodes(["a"])
        kept = T.observation_sequences(leaky, episodes)
        self.assertEqual(kept["token"].to_list(), [[5, 6, 7]])
        self.assertTrue(kept.equals(T.observation_sequences(base, episodes)))
        counts = T.token_count_matrix(kept["token"].to_list(), vocab_size=12)
        self.assertEqual(counts[0, 8] + counts[0, 9], 0)

    def test_the_anchor_is_each_stays_own(self):
        """The cut is the episode's anchor, not a constant: a late anchor keeps more."""
        shards = self._shards([[5, 6, 7], [5, 6, 7]], [[0, 1500, 3000]] * 2, ids=("a", "b"))
        episodes = _episodes(["a", "b"]).with_columns(
            pl.when(pl.col("hospitalization_id") == "b")
            .then(pl.col("icu_admit_dttm") + pl.duration(hours=26))
            .otherwise(pl.col("anchor_dttm")).alias("anchor_dttm"))
        kept = T.observation_sequences(shards, episodes).sort("hosp_id")
        self.assertEqual(kept["token"].to_list(), [[5], [5, 6]])

    def test_full_hospitalization_windows_are_cut_at_the_anchor(self):
        """GEM rows carry the stay's anchor index; tokens after it are dropped."""
        gem = pl.DataFrame({
            "hosp_id": ["a", "a"], "partition": ["train", "train"],
            "trajectory": ["hospitalization"] * 2,
            # written out of order: the windows are stitched by continuation_index
            "token": [[7, 8, 30, 2], [1, 20, 5, 6]],
            "pos_min": [[1500, 2000, 7200, 7200], [0, 0, 100, 900]],
            "continuation_index": [1, 0], "anchor_idx": [3, 3],
        }, schema_overrides={"token": pl.List(pl.Int64), "pos_min": pl.List(pl.Int64)})
        kept = T.observation_sequences(gem, _episodes(["a"]))
        self.assertEqual(kept["token"].to_list(), [[1, 20, 5, 6]])
        self.assertEqual(kept["pos_min"].to_list(), [[0, 0, 100, 900]])

    def test_a_stay_without_an_episode_is_refused(self):
        with self.assertRaises(QualificationError):
            T.observation_sequences(self._shards([[5]], [[0]], ids=("ghost",)),
                                    _episodes(["a"]))

    def test_token_counts_are_counts(self):
        counts = T.token_count_matrix([[4, 4, 7], [], [9]], vocab_size=10)
        self.assertEqual(counts.shape, (3, 10))
        np.testing.assert_array_equal(counts.toarray()[0, [4, 7, 9]], [2, 1, 0])
        self.assertEqual(counts[1].sum(), 0)
        with self.assertRaises(ValueError):
            T.token_count_matrix([[10]], vocab_size=10)


if __name__ == "__main__":
    unittest.main()
