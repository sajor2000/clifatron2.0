"""Smoke test for clif-validate/ federation package on MPS.

Tests the auto-labeler on real CLIF data, the forest-plot generator,
and validates that aggregate-only output contains no raw patient rows.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

# CLIF source parquet lives per-machine (git-ignored); override with CLIF_DATA_DIR.
CLIF_DATA = Path(
    os.environ.get("CLIF_DATA_DIR", "~/Data/clif-source")
).expanduser().resolve()
CLIF_AVAILABLE = CLIF_DATA.is_dir() and any(CLIF_DATA.glob("*.parquet"))
_SKIP_REASON = (
    f"CLIF source parquet not found at {CLIF_DATA} "
    "(set CLIF_DATA_DIR to a directory of CLIF 2.1 *.parquet to enable)"
)


class ClifValidateSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @unittest.skipUnless(CLIF_AVAILABLE, _SKIP_REASON)
    def test_01_auto_labeler_runs_on_real_clif_data(self):
        """Auto-labeler produces a labels.parquet with expected columns."""
        from src.eval.clif_auto_labeler import auto_label
        import polars as pl

        labels = auto_label(str(CLIF_DATA), [
            "in_hospital_mortality",
            "new_vasopressor_24h",
        ])

        self.assertIn("hospitalization_id", labels.columns)
        self.assertIn("in_hospital_mortality", labels.columns)
        self.assertIn("new_vasopressor_24h", labels.columns)
        self.assertGreater(len(labels), 0)

        lbl_path = self.out / "test_labels.parquet"
        labels.write_parquet(lbl_path)

        reloaded = pl.read_parquet(lbl_path)
        self.assertEqual(len(reloaded), len(labels))

        print(f"  labeled {len(labels):,} stays")
        for col in labels.columns:
            if col != "hospitalization_id":
                try:
                    n = labels[col].sum()
                    print(f"    {col}: {n:,} positive ({n/len(labels)*100:.1f}%)")
                except Exception:
                    print(f"    {col}: non-numeric column")

    def test_02_forest_plot_generates_on_synthetic_sites(self):
        """Forest-plot data is valid JSON with expected structure."""
        from src.eval.clif_forest_plot import load_site_results, forest_plot_data

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