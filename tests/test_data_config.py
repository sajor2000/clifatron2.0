import tempfile
import unittest
from pathlib import Path

import duckdb
import polars as pl

from src.data.tokenize import _read_table, validate_units


class DataConfigTest(unittest.TestCase):
    def test_reads_availability_column_and_validates_canonical_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            pl.DataFrame(
                {
                    "hospitalization_id": ["stay-1"],
                    "lab_result_dttm": ["2026-01-01T01:00:00"],
                    "lab_category": ["platelet_count"],
                    "lab_value_numeric": [123.0],
                    "reference_unit": ["10^3/µL"],
                }
            ).with_columns(pl.col("lab_result_dttm").str.to_datetime()).write_parquet(base / "clif_labs.parquet")
            spec = {
                "file": "clif_labs",
                "availability_col": "lab_result_dttm",
                "concept_col": "lab_category",
                "value_col": "lab_value_numeric",
                "unit_col": "reference_unit",
            }
            events = _read_table(duckdb.connect(), base, spec)

        self.assertEqual(events["concept"].to_list(), ["platelet_count"])
        validate_units(
            events,
            {"unit_normalization": {"on_mismatch": "error", "concepts": {"platelet_count": "10^3/uL"}}},
        )

    def test_rejects_noncanonical_units(self):
        events = pl.DataFrame({"concept": ["lactate"], "unit": ["mg/dL"]})
        with self.assertRaisesRegex(ValueError, "Non-canonical CLIF units"):
            validate_units(
                events,
                {"unit_normalization": {"on_mismatch": "error", "concepts": {"lactate": "mmol/L"}}},
            )


if __name__ == "__main__":
    unittest.main()
