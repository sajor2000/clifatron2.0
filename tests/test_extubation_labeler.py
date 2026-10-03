import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import yaml

from src.data.cohort import QualificationError
from src.data.extubation_cohort import (
    build_extubation_artifact,
    build_extubation_cohort,
    load_extubation_config,
    table_availability,
)
from src.data.targets import OUTCOME_STATUSES
from src.data.tokenize import with_availability_lag
from src.eval.causal.estimators import cumulative_incidence
from src.eval.extubation_labeler import (
    CAUSE_CODES,
    STATUS_CODES,
    EndpointDeclarationError,
    build_extubation_labels,
    build_label_artifact,
    declared_study_endpoints,
    estimator_arrays,
    label_counts,
    resolve_endpoint,
    resolve_window,
    suppress_counts,
)
from src.eval.extubation_labeler import (
    main as labeler_main,
)
from tests.fixtures_extubation import (
    PARTITIONS,
    SPLIT_SEED,
    build_extubation_fixture,
    hospitalization_id,
    patient_id,
    write_extubation_fixture,
)

ROOT = Path(__file__).resolve().parents[1]


def single_row_config():
    """The repo contract with the single-row look-forward rule: the fixture's hand-built
    scenario patients chart ONE non-invasive row after extubation (the primary since
    2026-10-03 requires two; see test_primary_requires_two_non_invasive_rows)."""
    config = load_extubation_config(ROOT / "configs/extubation.yaml")
    config["detection"]["look_forward"]["min_non_invasive_rows"] = 1
    return config
SPLIT = {"partitions": PARTITIONS, "split_seed": SPLIT_SEED}
COMPOSITE = "reintubation_or_death"


class ExtubationLabelerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = build_extubation_fixture(outcome_scenarios=True)
        cls.config = single_row_config()
        cls.data_config = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
        cls.cohort_config = yaml.safe_load((ROOT / "configs/cohort.yaml").read_text())
        # Scenario hours assume no availability lag (the repo lags respiratory support
        # 30 min since 2026-10-03; the shift is tested in test_extubation_cohort).
        cls.availability = table_availability(cls.config, with_availability_lag(
            cls.data_config, 0))
        cls.cohort = build_extubation_cohort(
            cls.fixture.tables,
            cls.config,
            availability=cls.availability,
            split=SPLIT,
            site="mimic",
            episodes=cls.fixture.episodes,
            calendar_periods=cls.fixture.calendar_periods,
        ).cohort
        cls.labels = cls.build()
        cls.rows = {row["patient_id"]: row for row in cls.labels.to_dicts()}

    @classmethod
    def build(cls, *, cohort=None, tables=None, config=None, data_config=None, cohort_config=None):
        return build_extubation_labels(
            cls.cohort if cohort is None else cohort,
            cls.fixture.tables if tables is None else tables,
            cls.config if config is None else config,
            availability=cls.availability,
            data_config=cls.data_config if data_config is None else data_config,
            cohort_config=cls.cohort_config if cohort_config is None else cohort_config,
        )

    def row(self, name):
        return self.rows[patient_id(name)]

    def state(self, name, horizon, *, event=COMPOSITE, rule="censor"):
        resolved = resolve_window(self.labels, horizon, event=event, discharge_alive_rule=rule)
        return resolved.filter(pl.col("patient_id") == patient_id(name)).to_dicts()[0]

    def assert_state(self, state, status, cause, hours):
        self.assertEqual((state["status"], state["cause"]), (status, cause))
        self.assertAlmostEqual(state["time_hours"], hours, places=6)
        self.assertEqual(state["event_type"], CAUSE_CODES[cause or "none"])

    def with_table(self, name, frame):
        return {**self.fixture.tables, name: frame}

    # ------------------------------------------------------------ plan scenarios

    def test_reintubation_at_80_hours_is_positive_at_7_days_and_negative_at_72_hours(self):
        row = self.row("out_reintubated_80h")
        self.assertAlmostEqual(row["reintubation_hours"], 80.0)
        self.assertIsNone(row["death_hours"])
        self.assert_state(self.state("out_reintubated_80h", 168), "positive", "reintubation", 80.0)
        # Still in hospital, not yet reintubated: event-free through the 72-hour window.
        self.assert_state(self.state("out_reintubated_80h", 72), "negative", None, 72.0)
        self.assert_state(
            self.state("out_reintubated_80h", 168, event="reintubation"),
            "positive", "reintubation", 80.0,
        )  # fmt: skip

    def test_death_on_day_3_without_reintubation_is_a_composite_event_with_cause_death(self):
        row = self.row("out_death_day3")
        self.assertIsNone(row["reintubation_hours"])
        self.assertAlmostEqual(row["death_hours"], 60.0)
        self.assertEqual(row["death_time_source"], "death_dttm")
        self.assertEqual(row["followup_end_reason"], "death")
        self.assert_state(self.state("out_death_day3", 168), "positive", "death", 60.0)
        self.assert_state(self.state("out_death_day3", 72), "positive", "death", 60.0)
        # Death prevents a later reintubation: it competes for the reintubation endpoint.
        self.assert_state(
            self.state("out_death_day3", 168, event="reintubation"),
            "competing_event", "death", 60.0,
        )  # fmt: skip

    def test_discharge_alive_with_a_death_timestamp_in_the_window_is_a_composite_event(self):
        name = "out_discharged_day2_died_day5"
        row = self.row(name)
        self.assertAlmostEqual(row["discharge_hours"], 48.0)
        self.assertAlmostEqual(row["death_hours"], 120.0)
        self.assertEqual(row["followup_end_reason"], "discharge_alive")
        for rule in ("censor", "event_free"):
            self.assert_state(self.state(name, 168, rule=rule), "positive", "death", 120.0)
        # A window that ends before the death timestamp sees only the discharge.
        self.assert_state(self.state(name, 72), "censored", None, 48.0)
        self.assert_state(self.state(name, 72, rule="event_free"), "negative", None, 72.0)

    def test_discharge_alive_without_a_death_timestamp_follows_the_registered_rule(self):
        name = "out_discharged_day2_alive"
        self.assertIsNone(self.row(name)["death_hours"])
        self.assert_state(self.state(name, 168), "censored", None, 48.0)  # default rule
        self.assert_state(self.state(name, 168, rule="censor"), "censored", None, 48.0)
        self.assert_state(self.state(name, 168, rule="event_free"), "negative", None, 168.0)
        with self.assertRaisesRegex(QualificationError, "discharge_alive_rule"):
            resolve_window(self.labels, 168, discharge_alive_rule="impute")

    def test_unknown_disposition_is_never_read_as_survival(self):
        name = "out_discharge_unknown"
        self.assertEqual(self.row(name)["discharge_disposition"], "unknown")
        self.assertEqual(self.row(name)["followup_end_reason"], "discharge_unknown")
        for rule in ("censor", "event_free"):
            self.assert_state(self.state(name, 168, rule=rule), "censored", None, 48.0)

    def test_niv_or_hfnc_after_extubation_is_never_a_failure(self):
        # NIV on day 1, rescue NIV at 5 h, NIV alternating with HFNC, extubation to NIV.
        for name in ("out_niv_day1", "ae1_rescue_niv", "alternating", "hypercapnic_niv"):
            row = self.row(name)
            self.assertIsNone(row["reintubation_hours"], name)
            self.assertIsNone(row["death_hours"], name)
            for event in (COMPOSITE, "reintubation", "death"):
                state = self.state(name, 168, event=event)
                self.assert_state(state, "negative", None, 168.0)

    # ------------------------------------------------------------ ordering and ties

    def test_first_event_inside_the_window_decides_the_cause(self):
        name = "out_reintubated_then_died"
        row = self.row(name)
        self.assertAlmostEqual(row["reintubation_hours"], 30.0)
        self.assertAlmostEqual(row["death_hours"], 90.0)
        self.assert_state(self.state(name, 168), "positive", "reintubation", 30.0)
        # Reintubation does not prevent death: the death endpoint still sees it.
        self.assert_state(self.state(name, 168, event="death"), "positive", "death", 90.0)
        self.assert_state(self.state(name, 72, event="death"), "negative", None, 72.0)

    def test_reintubation_and_death_at_the_same_instant_count_as_reintubation(self):
        name = "out_reintubated_at_death"
        row = self.row(name)
        self.assertAlmostEqual(row["reintubation_hours"], 50.0)
        self.assertAlmostEqual(row["death_hours"], 50.0)
        self.assert_state(self.state(name, 168), "positive", "reintubation", 50.0)
        self.assertFalse(row["death_within_24h_without_reintubation"])

    def test_event_exactly_at_the_horizon_is_inside_the_window(self):
        name = "out_reintubated_72h"
        self.assertAlmostEqual(self.row(name)["reintubation_hours"], 72.0)
        self.assert_state(self.state(name, 72), "positive", "reintubation", 72.0)
        self.assert_state(self.state(name, 71.5), "negative", None, 71.5)
        # Follow-up that ends exactly at the horizon covers the window.
        self.assert_state(self.state("out_discharged_day2_alive", 48), "negative", None, 48.0)
        self.assert_state(self.state("out_discharged_day2_alive", 48.5), "censored", None, 48.0)

    def test_hospice_discharge_is_a_competing_event(self):
        name = "out_hospice_day3"
        row = self.row(name)
        self.assertAlmostEqual(row["hospice_hours"], 60.0)
        self.assertAlmostEqual(row["death_hours"], 100.0)
        self.assertEqual(row["followup_end_reason"], "hospice_discharge")
        # The death after the hospice discharge does not turn it into a composite event.
        for event in (COMPOSITE, "reintubation", "death"):
            for rule in ("censor", "event_free"):
                state = self.state(name, 168, event=event, rule=rule)
                self.assert_state(state, "competing_event", "hospice", 60.0)
        self.assert_state(self.state(name, 48), "negative", None, 48.0)

    def test_death_within_24_hours_without_reintubation_is_flagged_not_removed(self):
        self.assertTrue(self.row("out_died_10h")["death_within_24h_without_reintubation"])
        self.assert_state(self.state("out_died_10h", 168), "positive", "death", 10.0)
        # Reintubated first, then died inside 24 h: not a terminal-extubation pattern.
        row = self.row("out_reintubated_died_20h")
        self.assertAlmostEqual(row["reintubation_hours"], 6.0)
        self.assertAlmostEqual(row["death_hours"], 20.0)
        self.assertFalse(row["death_within_24h_without_reintubation"])
        self.assertFalse(self.row("out_death_day3")["death_within_24h_without_reintubation"])
        self.assertFalse(self.row("out_niv_day1")["death_within_24h_without_reintubation"])
        # The comfort-care extubation (ineligible in the cohort) is labelled and flagged.
        comfort = self.row("ae2_comfort_care")
        self.assertAlmostEqual(comfort["death_hours"], 6.0)
        self.assertTrue(comfort["death_within_24h_without_reintubation"])
        self.assertEqual(self.labels.height, self.cohort.height)

    def test_inconsistent_death_record_is_flagged_in_the_labels(self):
        # Consistent fixture: nobody is flagged.
        self.assertFalse(self.labels["death_record_inconsistent"].any())
        name = "out_death_day3"
        admission = (
            self.fixture.tables["hospitalization"]
            .filter(pl.col("patient_id") == patient_id(name))["admission_dttm"]
            .min()
        )
        # A death three days before the admission: the labeler's own check flags it,
        # even for a cohort built before the check existed (no cohort column).
        patient = self.fixture.tables["patient"].with_columns(
            pl.when(pl.col("patient_id") == patient_id(name))
            .then(pl.lit(admission - timedelta(days=3)).cast(pl.Datetime("us", "UTC")))
            .otherwise(pl.col("death_dttm"))
            .alias("death_dttm")
        )
        tables = {**self.fixture.tables, "patient": patient}
        labels = self.build(cohort=self.cohort.drop("death_record_inconsistent"), tables=tables)
        flagged = labels.filter(pl.col("death_record_inconsistent"))["patient_id"].to_list()
        self.assertEqual(flagged, [patient_id(name)])
        # The cohort's flag is carried when the cohort has it.
        cohort = self.cohort.with_columns(
            (pl.col("patient_id") == patient_id("out_niv_day1")).alias("death_record_inconsistent")
        )
        labels = self.build(cohort=cohort)
        self.assertEqual(
            labels.filter(pl.col("death_record_inconsistent"))["patient_id"].to_list(),
            [patient_id("out_niv_day1")],
        )
        counts = label_counts(cohort, labels, [168])
        self.assertEqual(counts["patient_death_record_inconsistent"], 1)
        self.assertEqual(counts["eligible_death_record_inconsistent"], 1)

    # ------------------------------------------------------------ reintubation

    def test_stitched_ventilator_gap_is_not_a_reintubation(self):
        # The cohort stitches the 30-minute gap at hour 30, so time zero is hour 50 and the
        # IMV rows from 30.5 precede it.
        self.assertIsNone(self.row("stitched_gap")["reintubation_hours"])
        self.assert_state(self.state("stitched_gap", 168), "negative", None, 168.0)
        # A real return to IMV 15 h after the first extubation is a reintubation.
        self.assertAlmostEqual(self.row("two_extubations")["reintubation_hours"], 15.0)

    def test_time_zero_the_cohort_would_have_stitched_is_refused(self):
        origin = (
            self.fixture.tables["hospitalization"]
            .filter(pl.col("hospitalization_id") == hospitalization_id("stitched_gap"))[
                "admission_dttm"
            ]
            .item()
        )
        moved = self.cohort.with_columns(
            pl.when(pl.col("patient_id") == patient_id("stitched_gap"))
            .then(pl.lit(origin + timedelta(hours=30)).cast(pl.Datetime("us", "UTC")))
            .otherwise(pl.col("time_zero_dttm"))
            .alias("time_zero_dttm")
        )
        with self.assertRaisesRegex(QualificationError, "stitch"):
            self.build(cohort=moved)
        # With the gap AT the threshold the cohort calls hour 30 an extubation, and the
        # return to IMV half an hour later is then a reintubation.
        config = copy.deepcopy(self.config)
        config["detection"]["stitch_gap_hours"] = 0.5
        labels = self.build(cohort=moved, config=config)
        row = labels.filter(pl.col("patient_id") == patient_id("stitched_gap")).to_dicts()[0]
        self.assertAlmostEqual(row["reintubation_hours"], 0.5)

    def test_tracheostomy_after_time_zero(self):
        # A tracheostomy with no charted return to invasive ventilation is not a
        # reintubation; it is recorded for a sensitivity analysis.
        collar = self.row("out_trach_collar_day3")
        self.assertIsNone(collar["reintubation_hours"])
        self.assertAlmostEqual(collar["tracheostomy_hours"], 60.0)
        self.assert_state(self.state("out_trach_collar_day3", 168), "negative", None, 168.0)
        # Invasive ventilation through a tracheostomy is a return to invasive ventilation.
        via = self.row("out_reintubated_via_trach")
        self.assertAlmostEqual(via["reintubation_hours"], 50.0)
        self.assertAlmostEqual(via["tracheostomy_hours"], 50.0)
        self.assertIsNone(self.row("out_niv_day1")["tracheostomy_hours"])

    # ------------------------------------------------------------ death and follow-up

    def test_death_time_comes_from_the_timestamp_and_the_discharge_category(self):
        # Expired with no death timestamp: death at the discharge time.
        row = self.row("out_expired_no_timestamp")
        self.assertAlmostEqual(row["death_hours"], 60.0)
        self.assertEqual(row["death_time_source"], "discharge_category")
        self.assert_state(self.state("out_expired_no_timestamp", 168), "positive", "death", 60.0)
        # A day-resolution timestamp floored to before a discharge alive: death at discharge.
        row = self.row("out_death_date_floor")
        self.assertAlmostEqual(row["death_hours"], 48.0)
        self.assertEqual(row["death_time_source"], "death_dttm")
        self.assertEqual(row["followup_end_reason"], "discharge_alive")
        self.assert_state(self.state("out_death_date_floor", 168), "positive", "death", 48.0)
        self.assertIsNone(self.row("out_niv_day1")["death_time_source"])

    def test_follow_up_end_and_reason(self):
        expected = {
            "out_niv_day1": (260.0, "discharge_alive"),
            "out_reintubated_80h": (260.0, "discharge_alive"),
            "out_discharged_day2_died_day5": (48.0, "discharge_alive"),
            "out_death_day3": (60.0, "death"),
            "out_reintubated_then_died": (90.0, "death"),
            "out_hospice_day3": (60.0, "hospice_discharge"),
            "out_discharge_unknown": (48.0, "discharge_unknown"),
        }
        for name, (hours, reason) in expected.items():
            row = self.row(name)
            self.assertAlmostEqual(row["followup_end_hours"], hours, msg=name)
            self.assertEqual(row["followup_end_reason"], reason, name)

    def test_discharge_recorded_before_time_zero_is_clamped_and_flagged(self):
        row = self.row("out_charted_after_discharge")
        self.assertEqual(row["discharge_hours"], 0.0)
        self.assertTrue(row["times_clamped_to_time_zero"])
        self.assert_state(self.state("out_charted_after_discharge", 168), "censored", None, 0.0)
        self.assertFalse(self.row("out_niv_day1")["times_clamped_to_time_zero"])

    def test_labels_carry_hours_and_no_timestamp(self):
        self.assertEqual(self.labels["patient_id"].to_list(), self.cohort["patient_id"].to_list())
        self.assertFalse(
            [name for name, dtype in self.labels.schema.items() if dtype.is_temporal()]
        )
        self.assertEqual(
            self.labels["extubation_sha256"].unique().to_list(),
            self.cohort["extubation_sha256"].unique().to_list(),
        )
        for horizon in (72, 168):
            for event in (COMPOSITE, "reintubation", "death"):
                for rule in ("censor", "event_free"):
                    resolved = resolve_window(
                        self.labels, horizon, event=event, discharge_alive_rule=rule
                    )
                    self.assertEqual(resolved.height, self.labels.height)
                    self.assertLessEqual(set(resolved["status"]), OUTCOME_STATUSES)
                    self.assertTrue((resolved["time_hours"] >= 0).all())
                    self.assertTrue((resolved["time_hours"] <= horizon).all())

    # ------------------------------------------------------------ integration

    def test_labels_match_the_planted_truth(self):
        truth = self.fixture.truth
        self.assertEqual(truth.height, 300)
        for horizon in (72.0, 168.0):
            resolved = resolve_window(self.labels, horizon).join(
                truth, on="patient_id", how="inner"
            )
            self.assertEqual(resolved.height, truth.height)
            for row in resolved.to_dicts():
                inside = row["event"] and row["event_hours"] <= horizon
                if inside:
                    self.assertEqual(row["status"], "positive", row["patient_id"])
                    self.assertEqual(row["cause"], row["event_cause"], row["patient_id"])
                    self.assertAlmostEqual(row["time_hours"], row["event_hours"], places=3)
                else:
                    # Every background patient is still in hospital at the horizon.
                    self.assertEqual(row["status"], "negative", row["patient_id"])
                    self.assertIsNone(row["cause"])
                    self.assertEqual(row["time_hours"], horizon)
        # Rescue NIV before the event never moved the event time or its cause.
        planted = truth.filter(pl.col("event_cause") == "reintubation").height
        found = (
            self.labels.join(truth, on="patient_id")
            .filter(pl.col("reintubation_hours").is_not_null())
            .height
        )
        self.assertEqual(found, planted)

    # ------------------------------------------------------------ estimator arrays

    def test_estimator_arrays_document_their_codes_and_follow_up(self):
        self.assertEqual(CAUSE_CODES, {"none": 0, "reintubation": 1, "death": 2, "hospice": 3})
        self.assertEqual(STATUS_CODES, {"none": 0, "event": 1, "competing": 2})
        index = {pid: i for i, pid in enumerate(self.cohort["patient_id"].to_list())}

        arrays = estimator_arrays(self.labels, 168)
        self.assertEqual(arrays.codes, CAUSE_CODES)
        self.assertEqual(arrays.event_time.shape, (self.cohort.height,))
        self.assertTrue(np.issubdtype(arrays.event_type.dtype, np.integer))
        for name, code, hours in (
            ("out_reintubated_80h", 1, 80.0),
            ("out_death_day3", 2, 60.0),
            ("out_hospice_day3", 3, 60.0),
            ("out_niv_day1", 0, 168.0),
            ("out_discharged_day2_alive", 0, 48.0),
        ):
            i = index[patient_id(name)]
            self.assertEqual(int(arrays.event_type[i]), code, name)
            self.assertAlmostEqual(float(arrays.event_time[i]), hours, msg=name)
        # Under `censor` a patient discharged alive leaves before the horizon with no event.
        censored = index[patient_id("out_discharged_day2_alive")]
        self.assertFalse(arrays.resolved[censored])
        incomplete = (arrays.event_type == 0) & (arrays.event_time < 168)
        self.assertTrue(np.array_equal(incomplete, ~arrays.resolved))

        # Status coding: 1 = the event of interest, 2 = any competing event.
        status = estimator_arrays(
            self.labels, 168, discharge_alive_rule="event_free", coding="status"
        )
        self.assertEqual(status.codes, STATUS_CODES)
        self.assertEqual(int(status.event_type[index[patient_id("out_reintubated_80h")]]), 1)
        self.assertEqual(int(status.event_type[index[patient_id("out_death_day3")]]), 1)
        self.assertEqual(int(status.event_type[index[patient_id("out_hospice_day3")]]), 2)
        self.assertTrue(status.resolved[censored])
        self.assertEqual(float(status.event_time[censored]), 168.0)
        # Only the unknown disposition is still unresolved under `event_free`.
        self.assertFalse(status.resolved[index[patient_id("out_discharge_unknown")]])
        keep = status.resolved
        self.assertFalse(np.any((status.event_type[keep] == 0) & (status.event_time[keep] < 168)))

        reintubation = estimator_arrays(self.labels, 168, event="reintubation", coding="status")
        self.assertEqual(int(reintubation.event_type[index[patient_id("out_death_day3")]]), 2)

        # The competing-risk estimator accepts the arrays as they are.
        curve = cumulative_incidence(arrays.event_time, arrays.event_type, horizon=168.0)
        total = sum(curve.incidence[cause][-1] for cause in curve.causes) + curve.survival[-1]
        self.assertAlmostEqual(float(total), 1.0)
        with self.assertRaisesRegex(QualificationError, "coding"):
            estimator_arrays(self.labels, 168, coding="binary")

    # ------------------------------------------------------------ declarations

    def test_repo_endpoints_are_declared_label_only(self):
        endpoints = declared_study_endpoints(self.cohort_config, self.data_config)
        self.assertEqual(
            {name: (e.event, e.horizon_hours) for name, e in endpoints.items()},
            {
                "reintubation_72h": ("reintubation", 72.0),
                "reintubation_7d": ("reintubation", 168.0),
                "death_7d": ("death", 168.0),
                "reintubation_or_death_7d": (COMPOSITE, 168.0),
            },
        )
        self.assertEqual(endpoints["reintubation_or_death_7d"].sources, {"resp_support", "patient"})
        for name, endpoint in endpoints.items():
            resolved = resolve_endpoint(self.labels, name, endpoints)
            expected = resolve_window(
                self.labels, endpoint.horizon_hours, event=endpoint.event
            )
            self.assertTrue(resolved.equals(expected), name)
        primary = resolve_endpoint(
            self.labels, "reintubation_or_death_7d", endpoints, discharge_alive_rule="event_free"
        ).filter(pl.col("patient_id") == patient_id("out_discharged_day2_alive"))
        self.assertEqual(primary["status"].item(), "negative")

    def test_endpoint_not_declared_label_only_is_refused(self):
        endpoints = declared_study_endpoints(self.cohort_config, self.data_config)
        # Not a study endpoint at all: a treatment-initiation task and a trunk outcome.
        for name in ("new_imv_24h", "map_below_65_48h", "reintubation_96h"):
            with self.assertRaisesRegex(EndpointDeclarationError, f"{name} is not declared"):
                resolve_endpoint(self.labels, name, endpoints)

        cohort = copy.deepcopy(self.cohort_config)
        cohort["study_endpoints"]["reintubation_72h"].pop("use")
        with self.assertRaisesRegex(
            EndpointDeclarationError,
            r"reintubation_72h is not declared label_only.*treatment source\(s\) "
            r"\['resp_support'\]",
        ):
            self.build(cohort_config=cohort)

        cohort = copy.deepcopy(self.cohort_config)
        cohort["study_endpoints"]["reintubation_7d"]["use"] = "target"
        with self.assertRaisesRegex(EndpointDeclarationError, "reintubation_7d"):
            declared_study_endpoints(cohort, self.data_config)

        # A composite inherits the treatment source of its components.
        cohort = copy.deepcopy(self.cohort_config)
        cohort["study_endpoints"]["reintubation_or_death_7d"].pop("use")
        with self.assertRaisesRegex(
            EndpointDeclarationError,
            r"reintubation_or_death_7d is not declared label_only.*resp_support",
        ):
            declared_study_endpoints(cohort, self.data_config)

        # No declaration at all: respiratory support is not read as a label source.
        cohort = copy.deepcopy(self.cohort_config)
        del cohort["study_endpoints"]
        with self.assertRaisesRegex(EndpointDeclarationError, "reintubation"):
            self.build(cohort_config=cohort)

    def test_study_endpoints_never_live_in_the_trunk_outcomes_block(self):
        # Declared under `outcomes` instead: not a study endpoint, so refused.
        cohort = copy.deepcopy(self.cohort_config)
        cohort["outcomes"]["reintubation_7d"] = cohort["study_endpoints"].pop("reintubation_7d")
        with self.assertRaises(EndpointDeclarationError):
            self.build(cohort_config=cohort)
        # Declared in both blocks: a trunk target, refused.
        cohort = copy.deepcopy(self.cohort_config)
        cohort["outcomes"]["death_7d"] = {"source": "vitals"}
        with self.assertRaisesRegex(EndpointDeclarationError, "also trunk targets"):
            declared_study_endpoints(cohort, self.data_config)
        cohort = copy.deepcopy(self.cohort_config)
        cohort["treatment_target_policy"] = "target_eligible"
        with self.assertRaisesRegex(EndpointDeclarationError, "context_only"):
            declared_study_endpoints(cohort, self.data_config)
        # The labels do not depend on the trunk's outcome contract.
        cohort = copy.deepcopy(self.cohort_config)
        cohort["outcomes"] = {}
        self.assertTrue(self.build(cohort_config=cohort).equals(self.labels))
        self.assertFalse(set(self.labels.columns) & set(self.cohort_config["outcomes"]))

    def test_malformed_declarations_fail_closed(self):
        def mutated(change):
            cohort = copy.deepcopy(self.cohort_config)
            change(cohort["study_endpoints"])
            return cohort

        cases = {
            "source": lambda e: e["reintubation_7d"].update(source="vitals"),
            "event": lambda e: e["death_7d"].update(event="cardiac_arrest"),
            "time_zero": lambda e: e["death_7d"].update(time_zero="icu_admission"),
            "horizon_hours": lambda e: e["reintubation_72h"].update(horizon_hours=0),
            "composite": lambda e: e["reintubation_or_death_7d"].update(
                composite_of=["reintubation_72h", "death_7d"]
            ),
            "circular": lambda e: e["reintubation_or_death_7d"].update(
                composite_of=["reintubation_or_death_7d"]
            ),
            "neither source nor composite_of": lambda e: e["death_7d"].pop("source"),
        }
        for message, change in cases.items():
            with self.assertRaisesRegex(EndpointDeclarationError, message, msg=message):
                self.build(cohort_config=mutated(change))

    # ------------------------------------------------------------ errors

    def test_inputs_fail_closed(self):
        tables = dict(self.fixture.tables)
        del tables["patient"]
        with self.assertRaisesRegex(QualificationError, "patient"):
            self.build(tables=tables)
        patient = self.fixture.tables["patient"]
        naive = patient.with_columns(pl.col("death_dttm").dt.replace_time_zone(None))
        with self.assertRaisesRegex(QualificationError, "timezone-aware UTC"):
            self.build(tables=self.with_table("patient", naive))
        with self.assertRaisesRegex(QualificationError, "patient table"):
            self.build(tables=self.with_table("patient", patient.head(5)))
        with self.assertRaisesRegex(QualificationError, "unique"):
            self.build(tables=self.with_table("patient", pl.concat([patient, patient.head(1)])))
        hospitalization = self.fixture.tables["hospitalization"]
        with self.assertRaisesRegex(QualificationError, "missing required columns"):
            self.build(
                tables=self.with_table(
                    "hospitalization", hospitalization.drop("discharge_category")
                )
            )
        with self.assertRaisesRegex(QualificationError, "index hospitalization"):
            self.build(tables=self.with_table("hospitalization", hospitalization.head(5)))
        open_stay = hospitalization.with_columns(
            pl.lit(None, dtype=pl.Datetime("us", "UTC")).alias("discharge_dttm")
        )
        with self.assertRaisesRegex(QualificationError, "discharge time"):
            self.build(tables=self.with_table("hospitalization", open_stay))
        with self.assertRaisesRegex(QualificationError, "missing required columns"):
            self.build(cohort=self.cohort.drop("time_zero_dttm"))
        for horizon in (0, -1, float("inf"), float("nan")):
            with self.assertRaisesRegex(QualificationError, "horizon"):
                resolve_window(self.labels, horizon)
        with self.assertRaisesRegex(QualificationError, "event"):
            resolve_window(self.labels, 168, event="respiratory_failure")

    # ------------------------------------------------------------ aggregates

    def test_counts_partition_the_eligible_cohort_per_window(self):
        counts = label_counts(self.cohort, self.labels, [72, 168])
        self.assertTrue(all(isinstance(value, int) for value in counts.values()))
        self.assertFalse(any("SYN-" in key for key in counts))
        self.assertFalse(any("arm" in key or "niv" in key or "hfnc" in key for key in counts))
        eligible = self.cohort.filter(pl.col("eligible"))
        self.assertEqual(counts["patient_first_extubation"], self.cohort.height)
        self.assertEqual(counts["patient_eligible"], eligible.height)
        cells = (
            "event_reintubation",
            "event_death",
            "competing_hospice",
            "event_free_in_hospital",
            "discharged_alive_before_horizon",
            "discharge_unknown_before_horizon",
        )
        for horizon in (72, 168):
            total = sum(counts[f"h{horizon}_{cell}"] for cell in cells)
            self.assertEqual(total, counts["patient_eligible"], horizon)
            for rule, censored_cells in {
                "censor": cells[4:],
                "event_free": cells[5:],
            }.items():
                resolved = resolve_window(
                    self.labels, horizon, discharge_alive_rule=rule
                ).join(eligible.select("patient_id"), on="patient_id")
                self.assertEqual(
                    resolved.filter(pl.col("status") == "censored").height,
                    sum(counts[f"h{horizon}_{cell}"] for cell in censored_cells),
                    (horizon, rule),
                )
        self.assertEqual(
            counts["eligible_discharged_expired"],
            counts["eligible_discharged_expired_with_death_dttm"]
            + counts["eligible_discharged_expired_without_death_dttm"],
        )
        self.assertGreaterEqual(counts["eligible_discharged_expired_without_death_dttm"], 1)
        self.assertGreaterEqual(counts["eligible_death_within_24h_without_reintubation"], 1)
        self.assertGreaterEqual(counts["eligible_tracheostomy_without_reintubation"], 1)
        self.assertGreaterEqual(counts["eligible_times_clamped_to_time_zero"], 1)
        # Death timestamps of patients discharged alive (day 5, after hospice, date-floored).
        self.assertEqual(counts["h168_death_dttm_after_discharge"], 3)
        self.assertEqual(counts["h72_death_dttm_after_discharge"], 1)

    def test_small_cells_are_suppressed_with_a_complementary_cell(self):
        counts = {
            "patient_first_extubation": 400,
            "patient_eligible": 300,
            "h72_event_reintubation": 30,
            "h72_event_death": 12,
            "h72_competing_hospice": 3,
            "h72_event_free_in_hospital": 200,
            "h72_discharged_alive_before_horizon": 55,
            "h72_discharge_unknown_before_horizon": 0,
            "eligible_discharged_expired": 40,
            "eligible_discharged_expired_with_death_dttm": 38,
            "eligible_discharged_expired_without_death_dttm": 2,
        }
        shown = suppress_counts(counts, min_cell=10)
        self.assertEqual(shown["h72_competing_hospice"], "<10")
        self.assertEqual(shown["h72_discharge_unknown_before_horizon"], "<10")
        # One hidden non-zero cell could be recovered from the total: hide one more.
        self.assertEqual(shown["h72_event_death"], "suppressed")
        self.assertEqual(shown["h72_event_reintubation"], 30)
        self.assertEqual(shown["patient_eligible"], 300)
        self.assertEqual(shown["eligible_discharged_expired_without_death_dttm"], "<10")
        self.assertEqual(shown["eligible_discharged_expired_with_death_dttm"], "suppressed")
        self.assertEqual(shown["eligible_discharged_expired"], 40)


class ExtubationLabelArtifactTest(unittest.TestCase):
    def test_labels_are_written_under_the_governed_directory_and_only_counts_return(self):
        fixture = build_extubation_fixture(n_background=60, outcome_scenarios=True)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = write_extubation_fixture(fixture, base / "synthetic")
            old_cwd = Path.cwd()
            os.chdir(base)
            try:
                Path("output/intermediate_phi").mkdir(parents=True)
                fixture.episodes.write_parquet("output/intermediate_phi/episodes.parquet")
                cohort_path = "output/intermediate_phi/extubation_cohort.parquet"
                build_extubation_artifact(
                    data,
                    cohort_path,
                    site="mimic",
                    extubation_config=ROOT / "configs/extubation.yaml",
                    data_config=ROOT / "configs/data.yaml",
                    train_config=ROOT / "configs/train.yaml",
                    artifact_policy=ROOT / "configs/artifact_policy.yaml",
                )
                kwargs = {
                    "cohort_artifact": cohort_path,
                    "extubation_config": ROOT / "configs/extubation.yaml",
                    "data_config": ROOT / "configs/data.yaml",
                    "cohort_config": ROOT / "configs/cohort.yaml",
                    "artifact_policy": ROOT / "configs/artifact_policy.yaml",
                }
                out = "output/intermediate_phi/extubation_cohort_labels.parquet"
                counts = build_label_artifact(data, out, **kwargs)

                self.assertTrue(all(isinstance(v, int) for v in counts.values()))
                self.assertFalse(any("SYN-" in key for key in counts))
                self.assertIn("h72_event_reintubation", counts)
                self.assertIn("h168_event_death", counts)
                cohort = pl.read_parquet(cohort_path)
                labels = pl.read_parquet(out)
                self.assertEqual(labels["patient_id"].to_list(), cohort["patient_id"].to_list())
                self.assertEqual(counts["patient_first_extubation"], cohort.height)
                self.assertFalse([n for n, d in labels.schema.items() if d.is_temporal()])
                self.assertIn("clif_patient", labels["label_provenance_json"][0])

                with self.assertRaisesRegex(ValueError, "output/intermediate_phi"):
                    build_label_artifact(data, "elsewhere/labels.parquet", **kwargs)
                self.assertFalse(Path("elsewhere").exists())
                with self.assertRaisesRegex(QualificationError, "cohort artifact"):
                    missing = {**kwargs, "cohort_artifact": "output/intermediate_phi/no.parquet"}
                    build_label_artifact(data, out, **missing)

                # The CLI prints the suppressed counts and nothing row-level.
                Path(out).unlink()
                argv = [
                    "extubation_labeler", "--data", str(data),
                    "--config", str(ROOT / "configs/extubation.yaml"),
                    "--data-config", str(ROOT / "configs/data.yaml"),
                    "--cohort-config", str(ROOT / "configs/cohort.yaml"),
                    "--artifact-policy", str(ROOT / "configs/artifact_policy.yaml"),
                ]  # fmt: skip
                stdout = io.StringIO()
                with patch("sys.argv", argv), contextlib.redirect_stdout(stdout):
                    labeler_main()
                printed = json.loads(stdout.getvalue())
                self.assertEqual(set(printed), set(counts))
                self.assertNotIn("SYN-", stdout.getvalue())
                self.assertTrue(
                    all(v in ("<10", "suppressed") or v >= 10 for v in printed.values())
                )
                # Default destination: the cohort artifact's sibling.
                self.assertTrue(Path(out).exists())
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
