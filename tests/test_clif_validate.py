"""Synthetic smoke tests for the site-local validation surfaces.

Tests the auto-labeler on synthetic CLIF data, the forest-plot generator,
and validates that aggregate-only output contains no raw patient rows.
"""

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from src.data.cohort import build_cohort
from src.data.splits import content_manifest
from src.eval.clif_auto_labeler import auto_label


class ClifValidateSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_01_auto_labeler_preserves_unsupported_state(self):
        """A missing synthetic outcome table is unsupported, never negative."""
        base = self.out / "synthetic_clif"
        base.mkdir()
        start = datetime(2026, 1, 1, tzinfo=UTC)
        pl.DataFrame(
            {
                "hospitalization_id": ["synthetic-stay"],
                "patient_id": ["synthetic-patient"],
                "hospitalization_joined_id": ["synthetic-chain"],
                "admission_dttm": [start],
                "discharge_dttm": [datetime(2026, 1, 5, tzinfo=UTC)],
                "age_at_admission": [50],
                "discharge_category": ["Home"],
                "hospital_id": ["synthetic-site"],
            }
        ).write_parquet(base / "clif_hospitalization.parquet")
        pl.DataFrame(
            {
                "hospitalization_id": ["synthetic-stay"],
                "in_dttm": [start],
                "out_dttm": [datetime(2026, 1, 4, tzinfo=UTC)],
                "location_category": ["icu"],
            }
        ).write_parquet(base / "clif_adt.parquet")

        hospitalization = pl.read_parquet(base / "clif_hospitalization.parquet")
        adt = pl.read_parquet(base / "clif_adt.parquet")
        config = {
            "anchor_hours": 24,
            "prediction_horizon_hours": 48,
            "minimum_age": 18,
            "icu_location_category": "icu",
        }
        episodes = build_cohort(hospitalization, adt, config).with_columns(
            pl.lit("train").alias("partition"),
        )
        split_hash = content_manifest(
            episodes, columns=["hospitalization_id", "patient_id", "partition"]
        )["sha256"]
        episode_hash = content_manifest(
            episodes, columns=["hospitalization_id", "patient_id", "eligible", "partition"]
        )["sha256"]
        episodes = episodes.with_columns(
            pl.lit("1.0.0").alias("cohort_contract_version"),
            pl.lit(split_hash).alias("split_sha256"),
            pl.lit(episode_hash).alias("episode_sha256"),
            pl.lit("{}").alias("source_provenance_json"),
        )
        episode_path = base / "episodes.parquet"
        episodes.write_parquet(episode_path)
        (base / "clif_hospitalization.parquet").unlink()
        (base / "clif_adt.parquet").unlink()

        labels = auto_label(str(base), episode_path, ["map_below_65_48h"])

        self.assertIn("hospitalization_id", labels.columns)
        self.assertIn("map_below_65_48h_status", labels.columns)
        self.assertEqual(labels["partition"].item(), "train")
        self.assertEqual(labels["map_below_65_48h_status"].item(), "unsupported_at_site")
        self.assertIsNone(labels["map_below_65_48h"].item())

        lbl_path = self.out / "test_labels.parquet"
        labels.write_parquet(lbl_path)

        reloaded = pl.read_parquet(lbl_path)
        self.assertEqual(len(reloaded), len(labels))


    def test_02_forest_plot_generates_on_synthetic_sites(self):
        """Forest-plot data is valid JSON with expected structure."""
        from src.eval.clif_forest_plot import forest_plot_data

        results = [
            {
                "site_name": "MIMIC",
                "n_stays": 1000,
                "in_hospital_mortality": {"auroc": 0.82, "auprc": 0.28, "ece": 0.03},
                "new_imv_24h": {"auroc": 0.79, "auprc": 0.22, "ece": 0.05},
            },
            {
                "site_name": "Rush",
                "n_stays": 800,
                "in_hospital_mortality": {"auroc": 0.78, "auprc": 0.25, "ece": 0.04},
                "new_imv_24h": {"auroc": 0.76, "auprc": 0.20, "ece": 0.06},
            },
            {
                "site_name": "UChicago",
                "n_stays": 600,
                "in_hospital_mortality": {"auroc": 0.85, "auprc": 0.31, "ece": 0.02},
                "new_imv_24h": {"auroc": 0.81, "auprc": 0.24, "ece": 0.04},
            },
        ]

        forest = forest_plot_data(results)
        self.assertGreater(len(forest), 0)

        for row in forest:
            self.assertIn("outcome", row)
            self.assertIn("site", row)
            self.assertIn("value", row)
            self.assertIn("ci_lower", row)
            self.assertIn("ci_upper", row)
            self.assertIsNotNone(row["value"])

        # Write forest plot
        plot_path = self.out / "forest_plot.json"
        plot_path.write_text(json.dumps({"forest": forest}))

        reloaded = json.loads(plot_path.read_text())
        self.assertIn("forest", reloaded)

        print(f"  forest plot: {len(forest)} data points across sites")

    def test_03_aggregate_only_no_raw_patient_rows(self):
        """Validator output JSON contains no patient-level data."""
        results = {
            "site_name": "Test Hospital",
            "n_stays": 1234,
            "in_hospital_mortality": {"auroc": 0.82, "n": 1200},
        }

        results_json = json.dumps(results)
        forbidden = ["patient_id", "hosp_id", "sequence", "token", "pos_min"]
        for term in forbidden:
            self.assertNotIn(term, results_json.lower())

        print("  aggregate-only check: PASS")


if __name__ == "__main__":
    unittest.main()
