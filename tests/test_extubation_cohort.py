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

import polars as pl
import yaml

from src.data.cohort import QualificationError, validate_episode_artifact
from src.data.extubation_cohort import (
    build_extubation_artifact,
    build_extubation_cohort,
    load_extubation_config,
    suppress_waterfall,
    table_availability,
    validate_extubation_artifact,
)
from src.data.extubation_cohort import (
    main as extubation_main,
)
from src.data.splits import assign_grouped_splits
from tests.fixtures_extubation import (
    PARTITIONS,
    SPLIT_SEED,
    build_extubation_fixture,
    hospitalization_id,
    patient_id,
    write_extubation_fixture,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT = {"partitions": PARTITIONS, "split_seed": SPLIT_SEED}
REASONS = [
    "missing_age",
    "underage",
    "lookback_insufficient",
    "lookforward_insufficient",
    "tracheostomy",
    "comfort_care",
    "do_not_reintubate",
    "ventilation_under_minimum",
]


class ExtubationCohortTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = build_extubation_fixture()
        cls.config = load_extubation_config(ROOT / "configs/extubation.yaml")
        cls.availability = table_availability(
            cls.config, yaml.safe_load((ROOT / "configs/data.yaml").read_text())
        )
        cls.result = cls.build()
        cls.rows = {row["patient_id"]: row for row in cls.result.cohort.to_dicts()}

    @classmethod
    def build(
        cls,
        *,
        tables=None,
        config=None,
        site="mimic",
        episodes="fixture",
        calendar_periods="fixture",
        grace_hours=None,
    ):
        return build_extubation_cohort(
            cls.fixture.tables if tables is None else tables,
            cls.config if config is None else config,
            availability=cls.availability,
            split=SPLIT,
            site=site,
            episodes=cls.fixture.episodes if isinstance(episodes, str) else episodes,
            calendar_periods=(
                cls.fixture.calendar_periods
                if isinstance(calendar_periods, str)
                else calendar_periods
            ),
            grace_hours=grace_hours,
        )

    def row(self, name, result=None):
        if result is None:
            return self.rows[patient_id(name)]
        return result.cohort.filter(pl.col("patient_id") == patient_id(name)).to_dicts()[0]

    def with_table(self, name, frame):
        return {**self.fixture.tables, name: frame}

    # ------------------------------------------------------------ arms and rescue

    def test_ae1_niv_five_hours_after_extubation_to_face_mask_is_rescue(self):
        # Covers AE1: the default 3 h grace window is shorter than 5 h.
        row = self.row("ae1_rescue_niv")
        self.assertTrue(row["eligible"])
        self.assertEqual(row["time_zero_device_category"], "Face Mask")
        self.assertEqual(row["arm"], "conventional_oxygen")
        self.assertTrue(row["rescue_support"])
        self.assertEqual(row["rescue_arm"], "niv")
        self.assertAlmostEqual(row["rescue_hours_from_time_zero"], 5.0)
        self.assertFalse(row["escalated_in_grace"])

    def test_niv_inside_a_longer_grace_window_is_not_rescue(self):
        result = self.build(grace_hours=6.0)
        row = self.row("ae1_rescue_niv", result)
        # The device at the actual extubation still decides the assigned arm...
        self.assertEqual(row["arm"], "conventional_oxygen")
        self.assertFalse(row["rescue_support"])
        self.assertTrue(row["escalated_in_grace"])
        # ...and the declared sensitivity rule reads the same rows as NIV.
        self.assertEqual(row["arm_highest_support"], "niv")
        config = copy.deepcopy(self.config)
        config["arms"]["assignment"]["rule"] = "highest_support"
        self.assertEqual(
            self.row("ae1_rescue_niv", self.build(config=config, grace_hours=6.0))["arm"], "niv"
        )

    def test_niv_alternating_with_hfnc_is_the_niv_arm(self):
        row = self.row("alternating")
        self.assertEqual(row["time_zero_device_category"], "High Flow NC")
        self.assertEqual(row["arm"], "niv")
        self.assertFalse(row["rescue_support"])
        self.assertEqual(self.row("hypercapnic_niv")["arm"], "niv")

    def test_every_device_row_inside_the_grace_window_is_kept(self):
        devices = self.result.device_rows.filter(
            pl.col("patient_id") == patient_id("ae1_rescue_niv")
        ).sort("hours_from_time_zero")
        self.assertEqual(
            devices["device_category"].to_list(), ["Face Mask", "Nasal Cannula", "NIPPV"]
        )
        self.assertEqual(devices["hours_from_time_zero"].to_list(), [0.0, 2.0, 5.0])
        self.assertEqual(devices["in_grace_window"].to_list(), [True, True, False])
        self.assertEqual(
            devices["device_class"].to_list(), ["conventional_oxygen", "conventional_oxygen", "niv"]
        )
        self.assertEqual(devices["is_rescue"].to_list(), [False, False, True])
        # Rows before time zero are never in the device-row artifact.
        self.assertGreaterEqual(self.result.device_rows["hours_from_time_zero"].min(), 0.0)
        # A return to IMV after time zero is kept too (the estimator needs it to censor).
        second = self.result.device_rows.filter(
            pl.col("patient_id") == patient_id("two_extubations")
        )
        self.assertIn("invasive", second["device_class"].to_list())
        self.assertTrue(second.filter(pl.col("device_class") == "hfnc")["after_invasive_row"].all())

    def test_three_arms_have_planted_structure(self):
        eligible = self.result.cohort.filter(pl.col("eligible"))
        counts = dict(eligible.group_by("arm").len().iter_rows())
        self.assertEqual(set(counts), {"conventional_oxygen", "hfnc", "niv"})
        self.assertGreaterEqual(min(counts.values()), 20)
        truth = self.fixture.truth.join(
            self.result.cohort.select("patient_id", "arm", "eligible"),
            on="patient_id",
            suffix="_cohort",
        )
        self.assertEqual(truth.height, self.fixture.truth.height)
        self.assertTrue(truth["eligible"].all())
        self.assertTrue((truth["arm"] == truth["arm_cohort"]).all())
        # Confounding by indication: NIV patients carry more risk factors.
        by_arm = dict(truth.group_by("arm").agg(pl.col("n_risk_factors").mean()).iter_rows())
        self.assertGreater(by_arm["niv"], by_arm["conventional_oxygen"])

    # ------------------------------------------------------------ exclusions

    def test_ae2_comfort_care_extubation_is_excluded(self):
        # Covers AE2: excluded from the cohort, not counted as an outcome.
        row = self.row("ae2_comfort_care")
        self.assertFalse(row["eligible"])
        self.assertEqual(row["eligibility_status"], "comfort_care")
        self.assertTrue(row["comfort_care_at_time_zero"])
        self.assertFalse(row["eligible_ventilation_12h"])
        self.assertGreaterEqual(self.result.waterfall["patient_excluded_comfort_care"], 1)

    def test_comfort_care_ordered_after_time_zero_is_not_used(self):
        row = self.row("late_comfort_care")
        self.assertTrue(row["eligible"])
        self.assertEqual(row["code_status_at_time_zero"], "full")
        self.assertFalse(row["comfort_care_at_time_zero"])

    def test_tracheostomy_rows_and_trach_collar_transitions_are_excluded(self):
        flagged = self.row("trach_flag")
        self.assertEqual(flagged["eligibility_status"], "tracheostomy")
        self.assertEqual(flagged["time_zero_device_category"], "Face Mask")
        collar = self.row("trach_collar")
        self.assertEqual(collar["eligibility_status"], "tracheostomy")
        self.assertIsNone(collar["arm"])
        self.assertFalse(flagged["eligible"] or collar["eligible"])
        self.assertFalse(flagged["eligible_ventilation_12h"] or collar["eligible_ventilation_12h"])

    def test_do_not_reintubate_is_excluded_and_dnr_alone_is_kept(self):
        self.assertEqual(self.row("dni")["eligibility_status"], "do_not_reintubate")
        kept = self.row("dnr_only")
        self.assertTrue(kept["eligible"])
        self.assertEqual(kept["code_status_at_time_zero"], "dnr")
        self.assertFalse(kept["do_not_reintubate_at_time_zero"])

    def test_missing_code_status_keeps_the_patient_and_sets_a_flag(self):
        row = self.row("missing_code_status")
        self.assertTrue(row["eligible"])
        self.assertTrue(row["code_status_missing"])
        self.assertIsNone(row["code_status_at_time_zero"])
        self.assertFalse(self.row("ae1_rescue_niv")["code_status_missing"])
        self.assertGreaterEqual(self.result.waterfall["eligible_code_status_missing"], 1)

    def test_underage_and_unlocated_extubations_keep_a_reason(self):
        self.assertEqual(self.row("underage")["eligibility_status"], "underage")
        self.assertEqual(self.row("single_imv_row")["eligibility_status"], "lookback_insufficient")
        self.assertNotIn(patient_id("died_on_vent"), self.rows)

    # ------------------------------------------------------------ detection

    def test_ventilator_gap_under_one_hour_is_stitched(self):
        row = self.row("stitched_gap")
        origin = (
            self.fixture.tables["hospitalization"]
            .filter(pl.col("hospitalization_id") == hospitalization_id("stitched_gap"))[
                "admission_dttm"
            ]
            .item()
        )
        self.assertEqual(row["time_zero_dttm"], origin + timedelta(hours=50))
        self.assertAlmostEqual(row["imv_hours"], 48.0)
        self.assertEqual(row["arm"], "hfnc")
        self.assertTrue(row["eligible"])
        self.assertGreaterEqual(self.result.waterfall["extubation_gaps_stitched"], 1)

    def test_gap_at_the_stitch_threshold_is_an_extubation(self):
        config = copy.deepcopy(self.config)
        config["detection"]["stitch_gap_hours"] = 0.5
        row = self.row("stitched_gap", self.build(config=config))
        self.assertAlmostEqual(row["imv_hours"], 28.0)
        self.assertEqual(row["arm"], "conventional_oxygen")

    def test_look_back_and_look_forward_row_requirements_are_parameters(self):
        config = copy.deepcopy(self.config)
        config["detection"]["look_back"] = {
            "min_invasive_rows": 1,
            "max_hours_since_last_invasive_row": 48.0,
        }
        self.assertTrue(self.row("single_imv_row", self.build(config=config))["eligible"])
        config["detection"]["look_back"]["max_hours_since_last_invasive_row"] = 12.0
        row = self.row("single_imv_row", self.build(config=config))
        self.assertAlmostEqual(row["hours_since_last_invasive_row"], 38.0)
        self.assertEqual(row["eligibility_status"], "lookback_insufficient")
        # Asking for a confirming non-invasive row drops a patient with one row only.
        config = copy.deepcopy(self.config)
        config["detection"]["look_forward"]["min_non_invasive_rows"] = 2
        result = self.build(config=config)
        self.assertEqual(
            self.row("missing_code_status", result)["eligibility_status"],
            "lookforward_insufficient",
        )
        self.assertTrue(self.row("ae1_rescue_niv", result)["eligible"])

    def test_first_extubation_only(self):
        row = self.row("two_extubations")
        self.assertAlmostEqual(row["imv_hours"], 38.0)
        self.assertEqual(row["arm"], "conventional_oxygen")
        self.assertEqual(self.result.cohort["patient_id"].n_unique(), self.result.cohort.height)
        self.assertGreaterEqual(self.result.waterfall["extubation_excluded_not_first"], 1)

    def test_unmapped_device_is_not_an_extubation_device(self):
        row = self.row("other_first")
        self.assertEqual(row["time_zero_device_category"], "Nasal Cannula")
        self.assertAlmostEqual(row["imv_hours"], 39.0)

    def test_18_hours_of_ventilation_is_sensitivity_only(self):
        row = self.row("vent_18h")
        self.assertAlmostEqual(row["imv_hours"], 18.0)
        self.assertFalse(row["eligible"])
        self.assertEqual(row["eligibility_status"], "ventilation_under_minimum")
        self.assertTrue(row["eligible_ventilation_12h"])
        self.assertEqual(row["eligibility_status_ventilation_12h"], "eligible")
        water = self.result.waterfall
        self.assertEqual(
            water["ventilation_12h_patient_eligible"] - water["patient_eligible"],
            water["patient_excluded_ventilation_under_minimum"]
            - water["ventilation_12h_patient_excluded_ventilation_under_minimum"],
        )
        cohort = self.result.cohort
        self.assertEqual(
            cohort.filter(pl.col("eligible") & ~pl.col("eligible_ventilation_12h")).height, 0
        )

    # ------------------------------------------------------------ risk factors

    def test_lab_resulted_after_time_zero_is_not_used_for_hypercapnia(self):
        row = self.row("late_lab")
        self.assertEqual(row["paco2_mmhg"], 41.0)
        self.assertIs(row["hypercapnia"], False)
        only_late = self.row("only_late_lab")
        self.assertIsNone(only_late["paco2_mmhg"])
        self.assertIsNone(only_late["hypercapnia"])
        self.assertIs(self.row("hypercapnic_niv")["hypercapnia"], True)

    def test_availability_lag_moves_a_lab_past_time_zero(self):
        lagged = dict(self.availability)
        lagged["labs"] = lagged["labs"]._replace(lag_minutes=24 * 60)
        result = build_extubation_cohort(
            self.fixture.tables,
            self.config,
            availability=lagged,
            split=SPLIT,
            site="mimic",
            episodes=self.fixture.episodes,
        )
        self.assertIsNone(self.row("hypercapnic_niv", result)["hypercapnia"])

    def test_non_canonical_paco2_unit_fails_closed(self):
        labs = self.fixture.tables["labs"].with_columns(pl.lit("kPa").alias("reference_unit"))
        with self.assertRaisesRegex(QualificationError, "unit"):
            self.build(tables=self.with_table("labs", labs))

    def test_bmi_uses_only_measurements_before_time_zero(self):
        row = self.row("obese")
        self.assertEqual(row["weight_kg"], 110.0)
        self.assertAlmostEqual(row["bmi"], 110.0 / 1.7**2, places=6)
        self.assertIs(row["bmi_over_30"], True)
        self.assertIsNone(self.row("late_lab")["bmi"])
        self.assertIsNone(self.row("late_lab")["bmi_over_30"])

    def test_age_and_prolonged_ventilation(self):
        self.assertIs(self.row("ae1_rescue_niv")["age_over_65"], False)
        self.assertIs(self.row("trach_flag")["prolonged_ventilation"], False)
        truth = self.fixture.truth.join(self.result.cohort, on="patient_id", suffix="_cohort")
        self.assertTrue((truth["age_over_65"] == truth["age_over_65_cohort"]).all())
        self.assertTrue((truth["bmi_over_30"] == truth["bmi_over_30_cohort"]).all())
        self.assertTrue((truth["prolonged_ventilation"] == (truth["imv_hours"] >= 168.0)).all())
        with_lab = truth.filter(pl.col("hypercapnia_cohort").is_not_null())
        self.assertTrue((with_lab["hypercapnia"] == with_lab["hypercapnia_cohort"]).all())
        self.assertGreater(truth["hypercapnia_cohort"].null_count(), 0)

    def test_comorbidities_come_from_prior_hospitalizations_only(self):
        history = self.row("copd_history")
        self.assertTrue(history["comorbidity_history_available"])
        self.assertIs(history["comorbidity_copd"], True)
        self.assertIs(history["comorbidity_chronic_respiratory_disease"], True)
        self.assertIs(history["comorbidity_heart_failure"], True)
        index_only = self.row("index_code_only")
        self.assertFalse(index_only["comorbidity_history_available"])
        self.assertIsNone(index_only["comorbidity_heart_failure"])
        self.assertFalse(index_only["comorbidity_source_leaky"])
        truth = self.fixture.truth.join(self.result.cohort, on="patient_id")
        known = truth.filter(pl.col("comorbidity_history_available"))
        self.assertTrue((known["copd"] == known["comorbidity_copd"]).all())
        self.assertTrue((known["heart_failure"] == known["comorbidity_heart_failure"]).all())

    def test_leaky_comorbidity_source_must_be_allowed_and_is_flagged(self):
        config = copy.deepcopy(self.config)
        config["risk_factors"]["comorbidities"]["sources"] = ["index_stay_codes"]
        with self.assertRaisesRegex(QualificationError, "leak"):
            self.build(config=config)
        config["risk_factors"]["comorbidities"]["allow_leaky_sources"] = True
        row = self.row("index_code_only", self.build(config=config))
        self.assertIs(row["comorbidity_heart_failure"], True)
        self.assertTrue(row["comorbidity_source_leaky"])

    # ------------------------------------------------------------ partitions

    def test_partition_is_inherited_from_the_episode_artifact(self):
        validate_episode_artifact(self.fixture.episodes)
        expected = dict(
            self.fixture.episodes.filter(pl.col("partition").is_not_null())
            .select("patient_id", "partition")
            .iter_rows()
        )
        inherited = self.result.cohort.filter(pl.col("partition_source") == "episode_artifact")
        self.assertGreater(inherited.height, 100)
        for pid, partition in inherited.select("patient_id", "partition").iter_rows():
            self.assertEqual(partition, expected[pid])
        # The episode artifact picked the EARLIER stay for this patient; the partition
        # still follows the patient.
        self.assertEqual(self.row("copd_history")["partition_source"], "episode_artifact")
        self.assertEqual(
            self.row("copd_history")["partition"], expected[patient_id("copd_history")]
        )

    def test_patient_absent_from_the_episode_artifact_gets_the_same_rule_and_is_counted(self):
        absent = self.row("absent_from_artifact")
        self.assertNotIn(
            patient_id("absent_from_artifact"), self.fixture.episodes["patient_id"].to_list()
        )
        rule = assign_grouped_splits(
            pl.DataFrame(
                {
                    "hospitalization_id": [absent["hospitalization_id"]],
                    "patient_id": [absent["patient_id"]],
                }
            ),
            PARTITIONS,
            seed=SPLIT_SEED,
        )["partition"].item()
        self.assertEqual(absent["partition"], rule)
        self.assertEqual(absent["partition_source"], "assigned_absent_from_artifact")
        null_partition = self.row("null_partition")
        self.assertEqual(null_partition["partition_source"], "assigned_null_in_artifact")
        self.assertIsNotNone(null_partition["partition"])
        water = self.result.waterfall
        self.assertEqual(water["partition_assigned_absent_from_artifact"], 1)
        self.assertEqual(
            water["partition_assigned_null_in_artifact"],
            self.result.cohort.filter(
                pl.col("partition_source") == "assigned_null_in_artifact"
            ).height,
        )
        self.assertEqual(
            water["partition_inherited"]
            + water["partition_assigned_absent_from_artifact"]
            + water["partition_assigned_null_in_artifact"],
            self.result.cohort.height,
        )
        self.assertEqual(self.result.cohort["partition"].null_count(), 0)

    def test_without_an_episode_artifact_every_partition_comes_from_the_rule(self):
        result = self.build(episodes=None)
        self.assertEqual(result.waterfall["partition_inherited"], 0)
        self.assertEqual(
            result.waterfall["partition_assigned_absent_from_artifact"], result.cohort.height
        )
        # Same deterministic rule, so patients the artifact partitioned agree with it.
        both = result.cohort.select("patient_id", "partition").join(
            self.result.cohort.filter(pl.col("partition_source") == "episode_artifact").select(
                "patient_id", "partition"
            ),
            on="patient_id",
            suffix="_artifact",
        )
        self.assertTrue((both["partition"] == both["partition_artifact"]).all())

    def test_episode_artifact_with_a_patient_in_two_partitions_is_rejected(self):
        episodes = self.fixture.episodes.select("hospitalization_id", "patient_id", "partition")
        clash = (
            episodes.filter(pl.col("partition") == "train")
            .head(1)
            .with_columns(
                pl.lit("validation").alias("partition"),
                pl.lit("SYN-HOSP-clash").alias("hospitalization_id"),
            )
        )
        with self.assertRaisesRegex(QualificationError, "partition"):
            self.build(episodes=pl.concat([episodes, clash]))

    # ------------------------------------------------------------ calendar period

    def test_site_with_no_declared_period_source_has_a_null_period(self):
        # The source table is offered, but the site declares none: still null, and
        # nothing is read from the (shifted) event dates.
        result = self.build(site="rush")
        self.assertEqual(result.cohort["calendar_period"].null_count(), result.cohort.height)
        self.assertEqual(result.cohort.schema["calendar_period"], pl.String)
        self.assertEqual(result.waterfall["eligible_calendar_period_available"], 0)

    def test_declared_period_source_is_joined_on_patient(self):
        expected = dict(self.fixture.calendar_periods.iter_rows())
        for pid, period in self.result.cohort.select("patient_id", "calendar_period").iter_rows():
            self.assertEqual(period, expected[pid])
        self.assertEqual(
            self.result.waterfall["eligible_calendar_period_available"],
            self.result.waterfall["patient_eligible"],
        )

    def test_declared_but_unstaged_period_source_is_null(self):
        result = self.build(calendar_periods=None)
        self.assertEqual(result.cohort["calendar_period"].null_count(), result.cohort.height)

    def test_unknown_site_fails_closed(self):
        with self.assertRaisesRegex(QualificationError, "site"):
            self.build(site="elsewhere")

    def test_unit_at_time_zero_comes_from_adt(self):
        self.assertEqual(self.row("ae1_rescue_niv")["unit_at_time_zero"], "medical_icu")
        self.assertEqual(self.row("absent_from_artifact")["unit_at_time_zero"], "general_ward")

    # ------------------------------------------------------------ waterfall, hashes

    def test_waterfall_accounts_for_every_first_extubation(self):
        water = self.result.waterfall
        cohort = self.result.cohort
        self.assertTrue(all(isinstance(v, int) and not isinstance(v, bool) for v in water.values()))
        self.assertEqual(water["patient_first_extubation"], cohort.height)
        self.assertEqual(
            water["patient_first_extubation"],
            sum(water[f"patient_excluded_{reason}"] for reason in REASONS)
            + water["patient_eligible"],
        )
        self.assertEqual(
            water["patient_first_extubation"],
            sum(water[f"ventilation_12h_patient_excluded_{reason}"] for reason in REASONS)
            + water["ventilation_12h_patient_eligible"],
        )
        self.assertEqual(
            water["patient_eligible"],
            sum(water[f"arm_{arm}"] for arm in ["conventional_oxygen", "hfnc", "niv"]),
        )
        self.assertEqual(
            water["extubation_events"],
            water["extubation_excluded_not_first"] + water["patient_first_extubation"],
        )
        self.assertEqual(water["patient_eligible"], cohort.filter(pl.col("eligible")).height)
        # Ineligible rows are kept, each with a reason.
        ineligible = cohort.filter(~pl.col("eligible"))
        self.assertGreater(ineligible.height, 5)
        self.assertTrue(ineligible["eligibility_status"].is_in(REASONS).all())
        self.assertGreater(water["resp_rows_without_device"], 0)

    def test_cohort_carries_no_post_time_zero_discharge_time(self):
        self.assertNotIn("discharge_dttm", self.result.cohort.columns)
        self.assertFalse([c for c in self.result.cohort.columns if c.startswith("_")])

    def test_content_hash_and_validator(self):
        cohort = self.result.cohort
        validate_extubation_artifact(cohort)
        self.assertEqual(len(cohort["extubation_sha256"][0]), 64)
        self.assertEqual(
            self.build().cohort["extubation_sha256"][0], cohort["extubation_sha256"][0]
        )
        self.assertNotEqual(
            self.build(grace_hours=6.0).cohort["extubation_config_sha256"][0],
            cohort["extubation_config_sha256"][0],
        )
        tampered = cohort.with_columns(
            pl.when(pl.col("arm") == "niv")
            .then(pl.lit("hfnc"))
            .otherwise(pl.col("arm"))
            .alias("arm")
        )
        with self.assertRaisesRegex(QualificationError, "hash"):
            validate_extubation_artifact(tampered)
        with self.assertRaisesRegex(QualificationError, "missing required columns"):
            validate_extubation_artifact(cohort.drop("partition"))

    def test_small_cells_are_suppressed_with_a_complementary_cell(self):
        shown = suppress_waterfall(
            {
                "patient_first_extubation": 100,
                "patient_excluded_underage": 3,
                "patient_excluded_tracheostomy": 12,
                "patient_excluded_comfort_care": 0,
                "patient_eligible": 85,
                "arm_niv": 40,
                "arm_hfnc": 45,
                "arm_conventional_oxygen": 0,
            },
            min_cell=10,
        )
        self.assertEqual(shown["patient_excluded_underage"], "<10")
        self.assertEqual(shown["patient_excluded_comfort_care"], "<10")
        # One hidden non-zero cell could be recovered from the total: hide one more.
        self.assertEqual(shown["patient_excluded_tracheostomy"], "suppressed")
        self.assertEqual(shown["patient_eligible"], 85)
        # A hidden zero discloses nobody, so the arm cells stay.
        self.assertEqual(shown["arm_niv"], 40)
        self.assertEqual(shown["arm_conventional_oxygen"], "<10")

    # ------------------------------------------------------------ errors

    def test_inputs_fail_closed(self):
        resp = self.fixture.tables["respiratory_support"]
        with self.assertRaisesRegex(QualificationError, "missing required columns"):
            self.build(tables=self.with_table("respiratory_support", resp.drop("tracheostomy")))
        naive = resp.with_columns(pl.col("recorded_dttm").dt.replace_time_zone(None))
        with self.assertRaisesRegex(QualificationError, "timezone-aware UTC"):
            self.build(tables=self.with_table("respiratory_support", naive))
        numeric = resp.with_columns(
            pl.col("hospitalization_id").str.len_chars().alias("hospitalization_id")
        )
        with self.assertRaisesRegex(QualificationError, "string identifier"):
            self.build(tables=self.with_table("respiratory_support", numeric))
        status = self.fixture.tables["code_status"].with_columns(
            pl.lit("Partial").alias("code_status_category")
        )
        with self.assertRaisesRegex(QualificationError, "code status"):
            self.build(tables=self.with_table("code_status", status))
        tables = dict(self.fixture.tables)
        del tables["respiratory_support"]
        with self.assertRaisesRegex(QualificationError, "respiratory_support"):
            self.build(tables=tables)

    def test_undeclared_choices_fail_closed(self):
        config = copy.deepcopy(self.config)
        config["arms"]["assignment"]["rule"] = "last_device"
        with self.assertRaisesRegex(QualificationError, "assignment rule"):
            self.build(config=config)
        config = copy.deepcopy(self.config)
        config["time_zero"]["covariate_boundary"] = "at_or_before"
        with self.assertRaisesRegex(QualificationError, "covariate_boundary"):
            self.build(config=config)
        config = copy.deepcopy(self.config)
        config["arms"]["device_categories"]["niv"].append("IMV")
        with self.assertRaisesRegex(QualificationError, "more than one class"):
            self.build(config=config)
        with self.assertRaisesRegex(QualificationError, "availability"):
            table_availability(self.config, {"tables": {"resp_support": {"file": "x"}}})

    def test_optional_tables_may_be_absent(self):
        tables = {
            k: v
            for k, v in self.fixture.tables.items()
            if k in {"hospitalization", "respiratory_support"}
        }
        result = self.build(tables=tables)
        self.assertEqual(result.cohort.height, self.result.cohort.height)
        self.assertTrue(result.cohort["code_status_missing"].all())
        self.assertEqual(result.cohort["hypercapnia"].null_count(), result.cohort.height)
        self.assertEqual(result.cohort["unit_at_time_zero"].null_count(), result.cohort.height)


class ExtubationConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = load_extubation_config(ROOT / "configs/extubation.yaml")
        self.data = yaml.safe_load((ROOT / "configs/data.yaml").read_text())

    def test_registered_definitions(self):
        self.assertEqual(self.config["cohorts"]["primary"]["min_invasive_hours"], 24.0)
        self.assertEqual(
            self.config["cohorts"]["sensitivity"]["ventilation_12h"]["min_invasive_hours"], 12.0
        )
        self.assertEqual(self.config["detection"]["stitch_gap_hours"], 1.0)
        grace = self.config["grace_window"]
        self.assertEqual(grace["hours"], 3.0)
        self.assertNotIn(grace["hours"], grace["sensitivity_hours"])
        self.assertEqual(self.config["arms"]["device_categories"]["niv"], ["NIPPV", "CPAP"])
        self.assertEqual(
            self.config["exclusions"]["tracheostomy"]["device_categories"], ["Trach Collar"]
        )
        self.assertIsNone(self.config["sites"]["rush"]["calendar_period_source"])
        self.assertEqual(
            self.config["sites"]["mimic"]["calendar_period_source"]["period_col"],
            "anchor_year_group",
        )
        comorbidities = self.config["risk_factors"]["comorbidities"]
        for source in comorbidities["sources"]:
            self.assertFalse(comorbidities["available_sources"][source]["leaks_post_time_zero"])
        self.assertFalse(comorbidities["allow_leaky_sources"])

    def test_respiratory_support_stays_input_only(self):
        # Hard rule #1: this contract reads respiratory support to define cohort and
        # exposure; it declares no outcome and no target.
        table = self.config["source_tables"]["respiratory_support"]["data_table"]
        self.assertTrue(self.data["tables"][table]["input_only"])
        self.assertNotIn("outcomes", self.config)
        self.assertNotIn("study_endpoints", self.config)

    def test_availability_ordering_comes_from_the_data_config(self):
        availability = table_availability(self.config, self.data)
        self.assertEqual(availability["labs"].column, "lab_result_dttm")
        self.assertEqual(availability["respiratory_support"].column, "recorded_dttm")
        self.assertEqual(availability["code_status"].column, "start_dttm")
        self.assertEqual(availability["labs"].file, "clif_labs")


class ExtubationArtifactTest(unittest.TestCase):
    def test_artifact_is_written_under_the_governed_directory_and_returns_counts_only(self):
        fixture = build_extubation_fixture(n_background=60)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = write_extubation_fixture(fixture, base / "synthetic")
            old_cwd = Path.cwd()
            os.chdir(base)
            try:
                kwargs = {
                    "site": "mimic",
                    "extubation_config": ROOT / "configs/extubation.yaml",
                    "data_config": ROOT / "configs/data.yaml",
                    "train_config": ROOT / "configs/train.yaml",
                    "artifact_policy": ROOT / "configs/artifact_policy.yaml",
                }
                out = "output/intermediate_phi/extubation_cohort.parquet"
                # No episode artifact: fail closed unless explicitly allowed.
                with self.assertRaisesRegex(QualificationError, "episode artifact"):
                    build_extubation_artifact(data, out, **kwargs)
                Path("output/intermediate_phi").mkdir(parents=True)
                fixture.episodes.write_parquet("output/intermediate_phi/episodes.parquet")
                waterfall = build_extubation_artifact(data, out, **kwargs)

                self.assertTrue(all(isinstance(v, int) for v in waterfall.values()))
                self.assertFalse(any("SYN-" in key for key in waterfall))
                cohort = pl.read_parquet(out)
                validate_extubation_artifact(cohort)
                self.assertEqual(waterfall["patient_first_extubation"], cohort.height)
                self.assertEqual(
                    waterfall["eligible_calendar_period_available"], waterfall["patient_eligible"]
                )
                self.assertEqual(
                    cohort["episode_sha256"].unique().to_list(),
                    fixture.episodes["episode_sha256"].unique().to_list(),
                )
                devices = pl.read_parquet(
                    "output/intermediate_phi/extubation_cohort_device_rows.parquet"
                )
                self.assertEqual(set(devices["patient_id"]), set(cohort["patient_id"]))
                self.assertIn("clif_respiratory_support", cohort["source_provenance_json"][0])

                with self.assertRaisesRegex(ValueError, "output/intermediate_phi"):
                    build_extubation_artifact(data, "elsewhere/extubation_cohort.parquet", **kwargs)
                self.assertFalse(Path("elsewhere").exists())

                # The CLI prints the suppressed waterfall and nothing row-level.
                argv = [
                    "extubation_cohort", "--data", str(data), "--site", "mimic",
                    "--config", str(ROOT / "configs/extubation.yaml"),
                    "--data-config", str(ROOT / "configs/data.yaml"),
                    "--train-config", str(ROOT / "configs/train.yaml"),
                    "--artifact-policy", str(ROOT / "configs/artifact_policy.yaml"),
                ]  # fmt: skip
                stdout = io.StringIO()
                with patch("sys.argv", argv), contextlib.redirect_stdout(stdout):
                    extubation_main()
                printed = json.loads(stdout.getvalue())
                self.assertEqual(set(printed), set(waterfall))
                self.assertNotIn("SYN-", stdout.getvalue())
                self.assertEqual(
                    printed["hospitalization_source"], waterfall["hospitalization_source"]
                )
                self.assertTrue(
                    all(v in ("<10", "suppressed") or v >= 10 for v in printed.values())
                )

                allowed = build_extubation_artifact(
                    data,
                    out,
                    episode_artifact="output/intermediate_phi/not_built.parquet",
                    allow_missing_episode_artifact=True,
                    **kwargs,
                )
                self.assertEqual(allowed["partition_inherited"], 0)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
