import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import polars as pl
import yaml

from src.data.cohort import build_cohort_artifact, validate_artifact_destination
from src.data.tokenize import tokenize_site
from src.eval.clif_auto_labeler import main as auto_label_main

ROOT = Path(__file__).parents[1]


class ArtifactPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = yaml.safe_load((ROOT / "configs/artifact_policy.yaml").read_text())

    def test_patient_level_artifacts_are_local_parquet_only(self):
        rule = self.policy["classes"]["patient_level_phi"]
        self.assertEqual(rule["directory"], "output/intermediate_phi")
        self.assertEqual(rule["formats"], ["parquet"])
        self.assertFalse(rule["export_allowed"])
        validate_artifact_destination(
            "output/intermediate_phi/episodes.parquet", "patient_level_phi", self.policy
        )

    def test_patient_level_export_or_csv_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_artifact_destination(
                "output/final_no_phi/episodes.csv", "patient_level_phi", self.policy
            )
        with self.assertRaises(ValueError):
            validate_artifact_destination(
                "output/intermediate_phi/episodes.parquet",
                "patient_level_phi",
                self.policy,
                for_export=True,
            )
        with self.assertRaises(ValueError):
            validate_artifact_destination(
                "output/intermediate_phi/../final_no_phi/episodes.parquet",
                "patient_level_phi",
                self.policy,
            )

    def test_aggregate_exports_require_disclosure_control(self):
        rule = self.policy["classes"]["aggregate_no_phi"]
        self.assertTrue(rule["export_allowed"])
        self.assertGreaterEqual(rule["minimum_cell_size"], 10)

    def test_patient_level_entry_points_reject_disallowed_destinations_first(self):
        with self.assertRaisesRegex(ValueError, "output/intermediate_phi"):
            tokenize_site(
                {},
                "synthetic-site",
                Path("missing-input"),
                Path("output/final_no_phi/tokens"),
                None,
                None,
                artifact_policy=self.policy,
            )

        argv = [
            "clif_auto_labeler",
            "--data",
            "missing-input",
            "--episodes",
            "missing-episodes.parquet",
            "--out",
            "output/final_no_phi/labels.parquet",
        ]
        with patch("sys.argv", argv), patch(
            "src.eval.clif_auto_labeler.auto_label"
        ) as labeler:
            with self.assertRaisesRegex(ValueError, "output/intermediate_phi"):
                auto_label_main()
            labeler.assert_not_called()

    def test_production_cohort_builder_enforces_policy_and_persists_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "synthetic"
            data.mkdir()
            start = datetime(2026, 1, 1, tzinfo=UTC)
            pl.DataFrame(
                {
                    "hospitalization_id": ["stay-1"],
                    "patient_id": ["patient-1"],
                    "hospitalization_joined_id": ["chain-1"],
                    "admission_dttm": [start],
                    "discharge_dttm": [start + timedelta(hours=96)],
                    "age_at_admission": [50],
                    "discharge_category": ["Home"],
                }
            ).write_parquet(data / "clif_hospitalization.parquet")
            pl.DataFrame(
                {
                    "hospitalization_id": ["stay-1"],
                    "in_dttm": [start],
                    "out_dttm": [start + timedelta(hours=72)],
                    "location_category": ["icu"],
                }
            ).write_parquet(data / "clif_adt.parquet")
            cohort_config = base / "cohort.yaml"
            cohort_config.write_text(
                "contract_version: 1.0.0\nclif_version: 2.1.0\nmcide_version: 2.1.0\n"
                "source_tables:\n  hospitalization: clif_hospitalization\n  adt: clif_adt\n"
                "episode:\n  minimum_age: 18\n  icu_location_category: icu\n"
                "anchor:\n  hours_after_icu_admission: 24\n"
                "windows:\n  prediction:\n    horizon_hours: 48\n"
            )
            train_config = base / "train.yaml"
            train_config.write_text(
                "data_contract:\n  split_seed: 7\n  partitions:\n    train: 1.0\n"
                "  required_partitions: [train]\n"
            )
            old_cwd = Path.cwd()
            os.chdir(base)
            try:
                episodes, manifest = build_cohort_artifact(
                    data,
                    "output/intermediate_phi/episodes.parquet",
                    cohort_config=cohort_config,
                    train_config=train_config,
                    artifact_policy=ROOT / "configs/artifact_policy.yaml",
                )
                self.assertTrue(Path("output/intermediate_phi/episodes.parquet").exists())
                self.assertEqual(episodes["partition"].item(), "train")
                self.assertEqual(len(episodes["split_sha256"].item()), 64)
                self.assertIn("clif_hospitalization", manifest["source_provenance"])
                self.assertIn("source_provenance_json", episodes.columns)
                with self.assertRaisesRegex(ValueError, "output/intermediate_phi"):
                    build_cohort_artifact(
                        data,
                        "elsewhere/episodes.parquet",
                        cohort_config=cohort_config,
                        train_config=train_config,
                        artifact_policy=ROOT / "configs/artifact_policy.yaml",
                    )
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
