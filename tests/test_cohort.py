import json
import unittest
from datetime import UTC, datetime, timedelta

import polars as pl

from src.data.cohort import (
    QualificationError,
    build_cohort,
    derive_outcome_states,
)


def ts(hours: int) -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=hours)


class CohortContractTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "anchor_hours": 24,
            "prediction_horizon_hours": 48,
            "minimum_age": 18,
            "icu_location_category": "icu",
        }
        self.hospitalization = pl.DataFrame(
            {
                "hospitalization_id": ["stay-a", "stay-b"],
                "patient_id": ["patient-a", "patient-b"],
                "hospitalization_joined_id": ["chain-a", "chain-b"],
                "admission_dttm": [ts(0), ts(0)],
                "discharge_dttm": [ts(96), ts(30)],
                "age_at_admission": [50, 60],
                "discharge_category": ["Home", "Home"],
                "hospital_id": ["site-a", "site-a"],
            }
        )
        self.adt = pl.DataFrame(
            {
                "hospitalization_id": ["stay-a", "stay-b"],
                "in_dttm": [ts(0), ts(0)],
                "out_dttm": [ts(72), ts(20)],
                "location_category": ["icu", "icu"],
            }
        )

    def test_hour_24_survivor_cohort_and_waterfall(self):
        episodes, waterfall = build_cohort(
            self.hospitalization, self.adt, self.config, return_waterfall=True
        )
        by_stay = {row["hospitalization_id"]: row for row in episodes.to_dicts()}

        self.assertTrue(by_stay["stay-a"]["eligible"])
        self.assertEqual(by_stay["stay-a"]["anchor_dttm"], ts(24))
        self.assertFalse(by_stay["stay-b"]["eligible"])
        self.assertEqual(by_stay["stay-b"]["eligibility_status"], "not_in_icu_at_anchor")
        self.assertEqual(waterfall["patient_eligible"], 1)
        self.assertEqual(
            waterfall["episode_source"],
            waterfall["episode_excluded_no_icu"]
            + waterfall["episode_excluded_underage"]
            + waterfall["episode_excluded_missing_age"]
            + waterfall["episode_excluded_not_observed_at_anchor"]
            + waterfall["episode_excluded_not_in_icu_at_anchor"]
            + waterfall["episode_eligible_at_anchor"],
        )
        self.assertEqual(
            waterfall["episode_candidates"],
            waterfall["episode_excluded_non_index"] + waterfall["episode_selected"],
        )
        self.assertEqual(
            waterfall["patient_source"],
            waterfall["patient_excluded_no_icu"]
            + waterfall["patient_excluded_underage"]
            + waterfall["patient_excluded_missing_age"]
            + waterfall["patient_excluded_not_observed_at_anchor"]
            + waterfall["patient_excluded_not_in_icu_at_anchor"]
            + waterfall["patient_eligible"],
        )

    def test_contiguous_and_overlapping_icu_rows_are_one_interval(self):
        adt = pl.DataFrame(
            {
                "hospitalization_id": ["stay-a", "stay-a", "stay-a"],
                "in_dttm": [ts(0), ts(12), ts(20)],
                "out_dttm": [ts(12), ts(20), ts(30)],
                "location_category": ["icu", "icu", "icu"],
            }
        )

        episode = build_cohort(self.hospitalization.head(1), adt, self.config).row(
            0, named=True
        )

        self.assertTrue(episode["eligible"])
        self.assertEqual(episode["icu_out_dttm"], ts(30))

    def test_anchor_event_is_prevalent_and_post_anchor_event_is_positive(self):
        episodes = build_cohort(self.hospitalization.head(1), self.adt.head(1), self.config)
        observations = pl.DataFrame(
            {
                "hospitalization_id": ["stay-a", "stay-a"],
                "dttm": [ts(24), ts(30)],
                "concept": ["map", "map"],
                "value": [60.0, 55.0],
                "unit": ["mmHg", "mmHg"],
            }
        )
        spec = {
            "name": "map_below_65_48h",
            "concept": "map",
            "direction": "below",
            "threshold": 65,
            "unit": "mmHg",
            "minimum_post_anchor_measurements": 1,
        }

        states = derive_outcome_states(episodes, observations, spec)

        self.assertEqual(states["status"].item(), "prevalent")
        self.assertIsNone(states["label"].item())

        incident_only = observations.tail(1)
        states = derive_outcome_states(episodes, incident_only, spec)
        self.assertEqual(states["status"].item(), "positive")
        self.assertEqual(states["event_time_hours"].item(), 30.0)

    def test_incomplete_follow_up_is_censored_not_negative(self):
        episodes = build_cohort(self.hospitalization.head(1), self.adt.head(1), self.config)
        episodes = episodes.with_columns(
            pl.lit(ts(40)).alias("followup_end_dttm"),
            pl.lit("discharge").alias("terminal_event"),
        )
        observations = pl.DataFrame(
            {
                "hospitalization_id": ["stay-a"],
                "dttm": [ts(30)],
                "concept": ["map"],
                "value": [80.0],
                "unit": ["mmHg"],
            }
        )
        spec = {
            "name": "map_below_65_48h",
            "concept": "map",
            "direction": "below",
            "threshold": 65,
            "unit": "mmHg",
            "minimum_post_anchor_measurements": 1,
        }

        state = derive_outcome_states(episodes, observations, spec).row(0, named=True)

        self.assertEqual(state["status"], "censored")
        self.assertIsNone(state["label"])
        self.assertEqual(state["event_time_hours"], 40.0)

    def test_missing_outcome_source_is_unsupported(self):
        episodes = build_cohort(self.hospitalization.head(1), self.adt.head(1), self.config)
        state = derive_outcome_states(
            episodes,
            None,
            {"name": "map_below_65_48h"},
            source_available=False,
        ).row(0, named=True)
        self.assertEqual(state["status"], "unsupported_at_site")
        self.assertIsNone(state["label"])

    def test_rejects_naive_datetimes_and_multiple_hospitals(self):
        naive = self.adt.with_columns(pl.col("in_dttm").dt.replace_time_zone(None))
        with self.assertRaisesRegex(QualificationError, "UTC"):
            build_cohort(self.hospitalization, naive, self.config)

        multiple = self.hospitalization.with_columns(
            pl.Series("hospital_id", ["site-a", "site-b"])
        )
        with self.assertRaisesRegex(QualificationError, "one hospital"):
            build_cohort(multiple, self.adt, self.config)

    def test_allows_null_optional_linkage_identifier(self):
        """hospitalization_joined_id is an OPTIONAL CLIF field (all-null in MIMIC);
        nulls there must NOT block qualification — only true primary IDs must be non-null."""
        joined_nulled = self.hospitalization.with_columns(
            pl.lit(None, dtype=pl.String).alias("hospitalization_joined_id")
        )
        # Must not raise (previously rejected; corrected per CLIF 2.1 optionality).
        build_cohort(joined_nulled, self.adt, self.config)

    def test_rejects_null_required_identifier_before_join(self):
        """A null in a TRULY required identifier (patient_id) is still rejected."""
        invalid = self.hospitalization.with_columns(
            pl.when(pl.col("hospitalization_id") == "stay-a")
            .then(None)
            .otherwise(pl.col("patient_id"))
            .alias("patient_id")
        )
        with self.assertRaisesRegex(QualificationError, "null identifiers"):
            build_cohort(invalid, self.adt, self.config)

    def test_selects_first_episode_that_reaches_the_anchor(self):
        hospitalization = pl.concat(
            [
                self.hospitalization.head(1).with_columns(
                    pl.lit("short-stay").alias("hospitalization_id"),
                    pl.lit(ts(20)).alias("discharge_dttm"),
                ),
                self.hospitalization.head(1).with_columns(
                    pl.lit("eligible-stay").alias("hospitalization_id"),
                    pl.lit(ts(100)).alias("admission_dttm"),
                    pl.lit(ts(200)).alias("discharge_dttm"),
                ),
            ]
        )
        adt = pl.DataFrame(
            {
                "hospitalization_id": ["short-stay", "eligible-stay"],
                "in_dttm": [ts(0), ts(100)],
                "out_dttm": [ts(20), ts(180)],
                "location_category": ["icu", "icu"],
            }
        )

        episode = build_cohort(hospitalization, adt, self.config).row(0, named=True)

        self.assertEqual(episode["hospitalization_id"], "eligible-stay")
        self.assertTrue(episode["eligible"])


def _waterfall(**override):
    """A cohort waterfall (`build_cohort`) whose equations all hold."""
    base = {
        "episode_source": 1000, "episode_excluded_no_icu": 400,
        "episode_excluded_invalid_adt": 0, "episode_candidates": 600,
        "episode_excluded_non_index": 150, "episode_selected": 450,
        "patient_source": 700, "patient_excluded_no_icu": 250,
        "episode_excluded_underage": 50, "patient_excluded_underage": 40,
        "episode_excluded_missing_age": 0, "patient_excluded_missing_age": 0,
        "episode_excluded_open_stay_no_extraction_time": 0,
        "patient_excluded_open_stay_no_extraction_time": 0,
        "episode_excluded_not_observed_at_anchor": 100,
        "patient_excluded_not_observed_at_anchor": 80,
        "episode_excluded_not_in_icu_at_anchor": 150,
        "patient_excluded_not_in_icu_at_anchor": 120,
        "episode_eligible_at_anchor": 300, "patient_eligible": 210,
    }
    base.update(override)
    return base


class CohortWaterfallDisplayTest(unittest.TestCase):
    """The printed waterfall hides small cells AND the cells that would reveal them."""

    def recoverable(self, shown):
        """Equations with exactly one withheld non-zero member: that member is recoverable
        by subtraction from the published ones."""
        from src.data.cohort import waterfall_equations

        full = _waterfall()
        return [eq for eq in waterfall_equations(full)
                if sum(isinstance(shown[k], str) and full[k] != 0 for k in eq) == 1]

    def test_the_fixture_waterfall_satisfies_its_equations(self):
        from src.data.cohort import waterfall_equations

        full = _waterfall()
        equations = waterfall_equations(full)
        self.assertEqual(len(equations), 5)
        for total, *cells in equations:
            self.assertEqual(full[total], sum(full[k] for k in cells), total)

    def test_a_small_cell_is_not_recoverable_by_subtraction(self):
        from src.data.cohort import suppress_cohort_waterfall

        # 5 stays without ICU: source (1000) - candidates (995) would give it away.
        full = _waterfall(episode_source=1000, episode_excluded_no_icu=5,
                          episode_candidates=995, episode_excluded_non_index=545)
        shown = suppress_cohort_waterfall(full, 10)
        self.assertEqual(shown["episode_excluded_no_icu"], "<10")
        withheld = [k for k, v in shown.items() if v == "suppressed"]
        self.assertTrue(withheld)       # a complementary cell is withheld
        from src.data.cohort import waterfall_equations
        for eq in waterfall_equations(full):
            self.assertNotEqual(
                sum(isinstance(shown[k], str) and full[k] != 0 for k in eq), 1, eq)
        # The old per-cell mask published everything else in the equation.
        self.assertNotEqual(shown["episode_source"], "<10")

    def test_nothing_is_hidden_when_every_cell_is_large_or_zero(self):
        from src.data.cohort import suppress_cohort_waterfall

        full = _waterfall()
        self.assertEqual(suppress_cohort_waterfall(full, 10), full)

    def test_a_small_status_cell_withholds_the_smallest_released_cell_of_its_equation(self):
        from src.data.cohort import suppress_cohort_waterfall

        full = _waterfall(episode_excluded_underage=6, episode_excluded_not_in_icu_at_anchor=194)
        shown = suppress_cohort_waterfall(full, 10)
        self.assertEqual(shown["episode_excluded_underage"], "<10")
        # candidates = statuses + eligible: the smallest released non-zero member goes too.
        self.assertEqual(shown["episode_excluded_not_observed_at_anchor"], "suppressed")
        self.assertEqual(self.recoverable(shown), [])

    def test_the_cli_prints_the_complementary_suppression(self):
        import contextlib
        import io
        from unittest import mock

        from src.data import cohort

        full = _waterfall(episode_excluded_no_icu=5, episode_candidates=995,
                          episode_source=1000, episode_excluded_non_index=545)
        manifest = {"waterfall": full}
        with mock.patch.object(cohort, "build_cohort_artifact",
                               return_value=(None, manifest)), \
                mock.patch("sys.argv", ["cohort", "--data", "x", "--site", "mimic"]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            cohort.main()
        printed = json.loads(out.getvalue())["waterfall"]
        self.assertEqual(printed["episode_excluded_no_icu"], "<10")
        self.assertIn("suppressed", printed.values())
        self.assertEqual(printed, cohort.suppress_cohort_waterfall(full, 10))


if __name__ == "__main__":
    unittest.main()
