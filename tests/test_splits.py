import unittest

import polars as pl

from src.data.splits import (
    HeldOutStratification,
    assign_grouped_splits,
    held_out_shares,
    held_out_stratification,
    load_held_out_strata,
    strata_from_cohort,
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


RATIOS = {"train": 0.60, "validation": 0.15, "calibration": 0.10, "internal_test": 0.15}
HELD = ("validation", "calibration", "internal_test")


def _population(n: int = 4000, niv: int = 120, hfnc: int = 200):
    """Synthetic patients; the first `niv` are NIV and the next `hfnc` HFNC (first device)."""
    episodes = pl.DataFrame({"hospitalization_id": [f"h{i}" for i in range(n)],
                             "patient_id": [f"p{i}" for i in range(n)]})
    strata = {f"p{i}": "niv" for i in range(niv)}
    strata.update({f"p{i}": "hfnc" for i in range(niv, niv + hfnc)})
    return episodes, strata


class HeldOutStratificationTest(unittest.TestCase):
    """Item 46 (product authority, 2026-10-03): NIV and HFNC first-device patients are
    over-sampled into the held-out partitions; deterministic; partition sizes kept."""

    def split(self, episodes, strata, share=0.5, seed=42):
        spec = HeldOutStratification(strata=strata, arms=("niv", "hfnc"), share=share, held_out=HELD)
        return assign_grouped_splits(episodes, RATIOS, seed=seed, held_out=spec)

    def test_each_stratified_arm_reaches_the_configured_held_out_share(self):
        episodes, strata = _population()
        default = assign_grouped_splits(episodes, RATIOS, seed=42)
        stratified = self.split(episodes, strata)
        before = held_out_shares(default, strata, HELD)
        after = held_out_shares(stratified, strata, HELD)
        for arm, n in (("niv", 120), ("hfnc", 200)):
            self.assertEqual(after[arm]["patients"], n)
            self.assertAlmostEqual(after[arm]["held_out_share"], 0.5, delta=0.5 / n + 1e-9)
            self.assertLess(before[arm]["held_out_share"], 0.47)     # default ~0.40
        validate_grouped_splits(stratified)

    def test_assignment_is_deterministic_by_seed_and_row_order(self):
        episodes, strata = _population()
        first = self.split(episodes, strata)
        again = self.split(episodes.reverse(), strata).sort("hospitalization_id")
        self.assertTrue(first.sort("hospitalization_id").equals(again))
        other = self.split(episodes, strata, seed=7)
        self.assertFalse(first["partition"].equals(other["partition"]))

    def test_partition_sizes_are_kept_and_other_patients_only_rebalance(self):
        episodes, strata = _population()
        default = assign_grouped_splits(episodes, RATIOS, seed=42)
        stratified = self.split(episodes, strata)
        self.assertEqual(dict(default.group_by("partition").len().iter_rows()),
                         dict(stratified.group_by("partition").len().iter_rows()))
        joined = default.join(stratified, on="hospitalization_id", suffix="_new")
        others = joined.filter(~pl.col("patient_id").is_in(list(strata)))
        moved = others.filter(pl.col("partition") != pl.col("partition_new"))
        # Unstratified patients move only between a held-out partition and train, never
        # between two held-out partitions, and only as many as the arms moved in.
        self.assertTrue(moved.height > 0)
        self.assertTrue(((moved["partition"] == "train") | (moved["partition_new"] == "train")).all())
        stratified_moved = joined.filter(pl.col("patient_id").is_in(list(strata))
                                         & (pl.col("partition") != pl.col("partition_new"))).height
        self.assertEqual(moved.height, stratified_moved)
        self.assertLess(moved.height / others.height, 0.05)

    def test_a_share_below_the_default_moves_arm_patients_back_to_train(self):
        episodes, strata = _population()
        low = held_out_shares(self.split(episodes, strata, share=0.2), strata, HELD)
        self.assertAlmostEqual(low["hfnc"]["held_out_share"], 0.2, delta=0.01)

    def test_strata_read_only_outcome_blind_membership(self):
        cohort = pl.DataFrame({
            "patient_id": ["a", "b", "c", "d", "e"],
            "eligible": [True, True, True, False, True],
            "arm_first_device": ["niv", "hfnc", "conventional_oxygen", "niv", None],
            "arm": ["hfnc", "hfnc", "conventional_oxygen", "niv", None],
            "reintubation_hours": [5.0, None, None, 3.0, None],
            "death_hours": [None, 10.0, None, None, None],
        })
        self.assertEqual(strata_from_cohort(cohort, arm_column="arm_first_device", arms=("niv", "hfnc")),
                         {"a": "niv", "b": "hfnc"})
        # Changing every outcome column changes nothing.
        flipped = cohort.with_columns(pl.lit(1.0).alias("reintubation_hours"), pl.lit(None).alias("death_hours"))
        self.assertEqual(strata_from_cohort(flipped, arm_column="arm_first_device", arms=("niv", "hfnc")),
                         strata_from_cohort(cohort, arm_column="arm_first_device", arms=("niv", "hfnc")))
        # The loader asks parquet for exactly the outcome-blind columns.
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cohort.parquet"
            cohort.write_parquet(path)
            real = pl.read_parquet
            with mock.patch("src.data.splits.pl.read_parquet", side_effect=lambda *a, **k: real(*a, **k)) as spy:
                loaded = load_held_out_strata(path, arm_column="arm_first_device", arms=("niv", "hfnc"))
            self.assertEqual(spy.call_args.kwargs["columns"], ["patient_id", "eligible", "arm_first_device"])
        self.assertEqual(loaded, {"a": "niv", "b": "hfnc"})

    def test_the_option_is_off_by_default_and_fails_closed(self):
        import yaml
        from pathlib import Path

        contract = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/train.yaml").read_text())[
            "data_contract"]
        block = contract["held_out_stratification"]
        self.assertFalse(block["enabled"])
        self.assertEqual((block["arms"], block["held_out_share"], block["arm_column"]),
                         (["niv", "hfnc"], 0.5, "arm_first_device"))
        self.assertIsNone(held_out_stratification(contract, None))
        with self.assertRaisesRegex(ValueError, "not enabled"):
            held_out_stratification(contract, {"a": "niv"})
        enabled = {**contract, "held_out_stratification": {**block, "enabled": True}}
        with self.assertRaisesRegex(ValueError, "strata source"):
            held_out_stratification(enabled, None)
        spec = held_out_stratification(enabled, {"a": "niv"})
        self.assertEqual((spec.share, spec.held_out), (0.5, ("validation", "calibration", "internal_test")))
        with self.assertRaises(ValueError):
            HeldOutStratification(strata={}, arms=("niv",), share=1.0, held_out=HELD)


if __name__ == "__main__":
    unittest.main()
