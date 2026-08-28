import unittest

import polars as pl

from src.data.splits import (
    assign_grouped_splits,
    fit_partition,
    validate_grouped_splits,
    validate_required_partitions,
    validate_training_targets,
)


class GroupedSplitTest(unittest.TestCase):
    def test_patient_and_linked_encounters_never_cross_partitions(self):
        episodes = pl.DataFrame(
            {
                "hospitalization_id": ["h1", "h2", "h3", "h4", "h5", "h6"],
                "patient_id": ["p1", "p1", "p2", "p3", "p4", "p5"],
                "hospitalization_joined_id": ["j1", "j2", "j2", "j3", "j4", "j5"],
            }
        )
        split = assign_grouped_splits(
            episodes,
            {"train": 0.5, "validation": 0.2, "calibration": 0.1, "test": 0.2},
            seed=19,
        )

        self.assertEqual(
            split.filter(pl.col("patient_id") == "p1")["partition"].n_unique(), 1
        )
        linked = split.filter(pl.col("hospitalization_id").is_in(["h2", "h3"]))
        self.assertEqual(linked["partition"].n_unique(), 1)

    def test_assignment_is_stable_under_row_reordering(self):
        episodes = pl.DataFrame(
            {
                "hospitalization_id": [f"h{i}" for i in range(20)],
                "patient_id": [f"p{i}" for i in range(20)],
            }
        )
        ratios = {"train": 0.6, "validation": 0.15, "calibration": 0.1, "test": 0.15}
        a = assign_grouped_splits(episodes, ratios, seed=7).sort("hospitalization_id")
        b = assign_grouped_splits(episodes.reverse(), ratios, seed=7).sort("hospitalization_id")
        self.assertEqual(a["partition"].to_list(), b["partition"].to_list())

    def test_artifact_fit_sees_training_partition_only(self):
        rows = pl.DataFrame(
            {
                "hospitalization_id": ["train", "test"],
                "partition": ["train", "test"],
                "value": [1.0, 1_000_000.0],
            }
        )
        fitted = fit_partition(rows)
        self.assertEqual(fitted["value"].to_list(), [1.0])

    def test_required_empty_partition_fails_preflight(self):
        rows = pl.DataFrame({"partition": ["train", "train"]})
        with self.assertRaisesRegex(ValueError, "calibration"):
            validate_required_partitions(rows, ["train", "calibration"])

    def test_enabled_objective_without_training_targets_fails_preflight(self):
        labels = pl.DataFrame(
            {"partition": ["train", "test"], "map_below_65_48h": [None, True]}
        )
        with self.assertRaisesRegex(ValueError, "map_below_65_48h"):
            validate_training_targets(labels, ["map_below_65_48h"])

    def test_null_or_non_string_split_identifiers_fail_before_assignment(self):
        null_ids = pl.DataFrame(
            {"hospitalization_id": ["h1"], "patient_id": [None]}
        ).cast({"patient_id": pl.String})
        with self.assertRaisesRegex(ValueError, "null identifiers"):
            assign_grouped_splits(null_ids, {"train": 1.0}, seed=1)

        numeric_ids = pl.DataFrame({"hospitalization_id": [1], "patient_id": ["p1"]})
        with self.assertRaisesRegex(ValueError, "string identifier"):
            assign_grouped_splits(numeric_ids, {"train": 1.0}, seed=1)

    def test_split_validation_rejects_patient_leakage(self):
        rows = pl.DataFrame(
            {
                "hospitalization_id": ["h1", "h2"],
                "patient_id": ["p1", "p1"],
                "partition": ["train", "test"],
            }
        )
        with self.assertRaisesRegex(ValueError, "multiple partitions"):
            validate_grouped_splits(rows)


if __name__ == "__main__":
    unittest.main()
