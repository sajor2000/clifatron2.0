"""Regression: a skipped numeric event must not desync the per-stay arrays.

`encode()` drops an event whose numeric value is missing for a concept that has bin
edges. Before the fix it still emitted the FULL `pos_min` / `target_eligible` lists,
so every token after the first skip carried the wrong position and eligibility flag and
`n_events` disagreed with the sequence length (CodeRabbit: critical). This builds a
synthetic site, nulls one MAP measurement so the skip path fires, tokenizes, and
asserts every per-stay array is the same length.
"""

import os
import tempfile
import unittest
from pathlib import Path


class TokenizeAlignmentTest(unittest.TestCase):
    def test_a_missing_numeric_event_keeps_all_per_stay_arrays_aligned(self):
        import copy

        import polars as pl

        from src.eval.synthetic_bundle import (
            FIXTURE_COHORT,
            FIXTURE_DATA_CONFIG,
            FIXTURE_POLICY,
            SYNTHETIC_SITE,
            build_synthetic_site,
        )

        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            old_cwd = os.getcwd()
            os.chdir(work)  # artifact policy classifies shards relative to CWD
            try:
                site = work / "site"
                episode_path = build_synthetic_site(site)

                # Null out one MAP measurement so encode() takes its skip branch.
                import yaml
                (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
                (work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))
                vitals_path = site / "clif_vitals.parquet"
                vitals = pl.read_parquet(vitals_path)
                # Set the first row's value to null (its concept `map` has a forced edge).
                vitals = vitals.with_columns(
                    pl.when(pl.int_range(pl.len()) == 0)
                    .then(None)
                    .otherwise(pl.col("vital_value"))
                    .alias("vital_value")
                )
                self.assertEqual(vitals["vital_value"].null_count(), 1)
                vitals.write_parquet(vitals_path)

                cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
                cfg["cohort_contract"] = str((work / "cohort.yaml").resolve())
                cfg["artifact_policy"] = str((work / "artifact_policy.yaml").resolve())

                from src.data.tokenize import tokenize_site

                episodes = pl.read_parquet(episode_path)
                out = Path("output/intermediate_phi/align_build")
                tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, None,
                              episodes=episodes, artifact_policy=FIXTURE_POLICY)
                events = pl.read_parquet(out / "events.parquet")
            finally:
                os.chdir(old_cwd)

            self.assertGreater(len(events), 0)
            saw_skip = False
            for row in events.iter_rows(named=True):
                n = len(row["token"])
                self.assertEqual(len(row["pos_min"]), n, "pos_min desynced from token")
                self.assertEqual(len(row["target_eligible"]), n,
                                 "target_eligible desynced from token")
                self.assertEqual(len(row["value"]), n, "value desynced from token")
                self.assertEqual(len(row["soft_token"]), n)
                self.assertEqual(row["n_events"], n, "n_events disagrees with length")
                # The stay carrying the nulled measurement dropped one event.
                if row["hosp_id"] == "synth-000":
                    saw_skip = True
            self.assertTrue(saw_skip, "the nulled-measurement stay was not tokenized")


class ReadTableEdgeCaseTest(unittest.TestCase):
    def test_empty_keep_ids_matches_nothing_instead_of_invalid_sql(self):
        """keep_ids=[] must not emit `IN ()` (a DuckDB syntax error)."""
        import duckdb
        import polars as pl

        from src.data.tokenize import _read_table

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            pl.DataFrame({
                "hospitalization_id": ["a", "b"],
                "recorded_dttm": ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
                "vital_category": ["map", "map"],
                "vital_value": [70.0, 80.0],
                "vital_unit": ["mmHg", "mmHg"],
            }).write_parquet(base / "clif_vitals.parquet")
            spec = {"file": "clif_vitals", "availability_col": "recorded_dttm",
                    "concept_col": "vital_category", "value_col": "vital_value",
                    "unit_col": "vital_unit"}
            con = duckdb.connect()
            con.execute("SET TimeZone = 'UTC'")
            out = _read_table(con, base, spec, keep_ids=[])
            self.assertEqual(len(out), 0)  # empty allow-list keeps nothing, no error


if __name__ == "__main__":
    unittest.main()
