"""Regression: a skipped numeric event must not desync the per-stay arrays.

`encode()` drops an event whose numeric value is missing for a concept that has bin
edges. Before the fix it still emitted the FULL `pos_min` / `target_eligible` lists,
so every token after the first skip carried the wrong position and eligibility flag and
`n_events` disagreed with the sequence length (CodeRabbit: critical). This builds a
synthetic site, nulls one MAP measurement so the skip path fires, tokenizes, and
asserts every per-stay array is the same length.
"""

import json
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


class UnkReservedTokenGuardTest(unittest.TestCase):
    def test_imported_vocab_must_reserve_unk_id(self):
        """An imported vocab whose <unk> id is wrong is rejected before the unknown-
        concept fallback can emit an incorrect/out-of-range token (CodeRabbit PR #13)."""
        import copy

        import polars as pl

        from src.data.cohort import QualificationError
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
            os.chdir(work)
            try:
                import yaml

                site = work / "site"
                episode_path = build_synthetic_site(site)
                (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
                (work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))
                cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
                cfg["cohort_contract"] = str((work / "cohort.yaml").resolve())
                cfg["artifact_policy"] = str((work / "artifact_policy.yaml").resolve())

                from src.data.tokenize import (
                    SPECIAL,
                    _json_sha256,
                    tokenize_site,
                )

                episodes = pl.read_parquet(episode_path)
                # First build a valid frozen vocab (writes vocab.json with its manifest).
                out = Path("output/intermediate_phi/unk_build")
                tokenize_site(
                    cfg, SYNTHETIC_SITE, site, out, None, None,
                    episodes=episodes, artifact_policy=FIXTURE_POLICY,
                )
                blob = json.loads((out / "vocab.json").read_text())
                vocab, edges, manifest = blob["vocab"], blob["edges"], blob["manifest"]

                # Corrupt the reserved <unk> id and re-import with a manifest hash that
                # still matches the corrupted vocab (so the <unk> guard — not the hash
                # check — is what must reject it). Must fail closed.
                bad_vocab = dict(vocab)
                bad_vocab["<unk>"] = max(vocab.values()) + 999  # not SPECIAL["<unk>"]
                self.assertNotEqual(bad_vocab["<unk>"], SPECIAL["<unk>"])
                manifest["hashes"]["vocabulary"] = _json_sha256(bad_vocab)
                out2 = Path("output/intermediate_phi/unk_reject")
                with self.assertRaisesRegex(QualificationError, "<unk>"):
                    tokenize_site(
                        cfg, SYNTHETIC_SITE, site, out2, bad_vocab, edges,
                        episodes=episodes, vocab_manifest=manifest,
                        artifact_policy=FIXTURE_POLICY,
                    )
            finally:
                os.chdir(old_cwd)


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

    def test_categorical_value_col_is_read_and_absent_specs_yield_null(self):
        import duckdb
        import polars as pl

        from src.data.tokenize import _read_table

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            pl.DataFrame({
                "hospitalization_id": ["a", "a"],
                "recorded_dttm": ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
                "assessment_category": ["cam_total", "gcs_total"],
                "numerical_value": [None, 14.0],
                "categorical_value": ["Negative", None],
            }).write_parquet(base / "clif_patient_assessments.parquet")
            spec = {"file": "clif_patient_assessments", "availability_col": "recorded_dttm",
                    "concept_col": "assessment_category", "value_col": "numerical_value",
                    "categorical_value_col": "categorical_value"}
            con = duckdb.connect()
            out = _read_table(con, base, spec).sort("concept")
            self.assertEqual(out["cat_value"].to_list(), ["Negative", None])
            plain = _read_table(con, base, {k: v for k, v in spec.items()
                                            if k != "categorical_value_col"})
            self.assertEqual(plain.schema["cat_value"], pl.String)
            self.assertEqual(plain["cat_value"].null_count(), len(plain))
            missing = _read_table(con, base, {**spec, "file": "nope"})
            self.assertIn("cat_value", missing.columns)


def _repartition(episodes, validation_ids):
    """Move `validation_ids` to the validation partition and reseal the content hashes."""
    import polars as pl

    from src.data.splits import content_manifest

    episodes = episodes.with_columns(
        pl.when(pl.col("hospitalization_id").is_in(validation_ids))
        .then(pl.lit("validation"))
        .otherwise(pl.col("partition"))
        .alias("partition")
    )
    eligible = episodes.filter(pl.col("eligible"))
    split_hash = content_manifest(
        eligible, columns=["hospitalization_id", "patient_id", "partition"]
    )["sha256"]
    episode_hash = content_manifest(
        episodes, columns=["hospitalization_id", "patient_id", "eligible", "partition"]
    )["sha256"]
    return episodes.with_columns(
        pl.lit(split_hash).alias("split_sha256"),
        pl.lit(episode_hash).alias("episode_sha256"),
    )


class FusedCategoricalAndCoverageTest(unittest.TestCase):
    """U2 (R4, R5; KTD3, KTD4) end to end through `tokenize_site`.

    A synthetic assessments table declares `categorical_value_col`. Braden rows carry a
    numeric score AND a descriptor (one ordinal token, never a descriptor token);
    cam_total rows carry only a categorical result (fused `cam_total=negative`); the
    validation stays chart an unseen CAM string (<unk>) and a numeric concept that the
    train partition never saw (no bins, <unk>)."""

    VALIDATION = [f"synth-{i:03d}" for i in range(20, 24)]
    HOURS = range(2, 22, 2)          # all inside [icu admit, anchor = +24h]

    @classmethod
    def setUpClass(cls):
        import copy
        from datetime import timedelta

        import polars as pl
        import yaml

        from src.data.tokenize import ROOT, tokenize_site
        from src.eval.synthetic_bundle import (
            FIXTURE_COHORT,
            FIXTURE_DATA_CONFIG,
            FIXTURE_POLICY,
            SYNTHETIC_SITE,
            build_synthetic_site,
        )

        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old_cwd = os.getcwd()
        os.chdir(work)
        try:
            site = work / "site"
            episode_path = build_synthetic_site(site)
            (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
            (work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))
            episodes = _repartition(pl.read_parquet(episode_path), cls.VALIDATION)

            rows = {"hospitalization_id": [], "recorded_dttm": [], "assessment_category": [],
                    "numerical_value": [], "categorical_value": []}

            def add(stay, when, concept, number, category):
                rows["hospitalization_id"].append(stay)
                rows["recorded_dttm"].append(when)
                rows["assessment_category"].append(concept)
                rows["numerical_value"].append(number)
                rows["categorical_value"].append(category)

            for ep in episodes.iter_rows(named=True):
                stay, admit = ep["hospitalization_id"], ep["icu_admit_dttm"]
                validation = stay in cls.VALIDATION
                for k, hour in enumerate(cls.HOURS):
                    when = admit + timedelta(hours=hour)
                    add(stay, when, "braden_mobility", float(1 + (k % 4)), "Very Limited")
                    add(stay, when, "cam_total", None,
                        "Unable To Assess" if validation else "Negative")
                    if validation:
                        add(stay, when, "valonly_score", 3.5 + k, None)
            pl.DataFrame(rows, schema={
                "hospitalization_id": pl.String,
                "recorded_dttm": pl.Datetime("us", "UTC"),
                "assessment_category": pl.String,
                "numerical_value": pl.Float64,
                "categorical_value": pl.String,
            }).write_parquet(site / "clif_patient_assessments.parquet")

            cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
            cfg["cohort_contract"] = str((work / "cohort.yaml").resolve())
            cfg["artifact_policy"] = str((work / "artifact_policy.yaml").resolve())
            cfg["tables"]["assessments"] = {
                "file": "clif_patient_assessments",
                "availability_col": "recorded_dttm",
                "concept_col": "assessment_category",
                "value_col": "numerical_value",
                "categorical_value_col": "categorical_value",
            }
            cfg["value_binning"].update({
                "scheme": "clinical_segment",
                "segment_source": str(
                    ROOT / "external/clifatron/tokenETL/config/"
                    "critical_illness_tokenization_final_with_intervals.csv"
                ),
                "coverage": "all",
            })
            out = Path("output/intermediate_phi/fused_build")
            tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, None,
                          episodes=episodes, artifact_policy=FIXTURE_POLICY)
            cls.events = pl.read_parquet(out / "events.parquet")
            cls.blob = json.loads((out / "vocab.json").read_text())
        finally:
            os.chdir(old_cwd)
        cls.vocab = cls.blob["vocab"]
        cls.inv = {i: t for t, i in cls.vocab.items()}

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _tokens(self, stay):
        row = self.events.filter(self.events["hosp_id"] == stay).row(0, named=True)
        return [self.inv[t] for t in row["token"]]

    def test_numeric_assessment_row_emits_only_its_ordinal_bin(self):
        self.assertEqual(self.blob["binning_sources"]["braden_mobility"], "ordinal")
        self.assertEqual(
            [t for t in self.vocab if t.startswith("braden_mobility")],
            [f"braden_mobility={b}" for b in range(4)],
        )
        tokens = self._tokens("synth-000")
        braden = [t for t in tokens if t.startswith("braden_mobility")]
        self.assertEqual(len(braden), len(self.HOURS))          # one token per row
        self.assertEqual(braden[:4], [f"braden_mobility={b}" for b in range(4)])

    def test_value_only_categorical_becomes_a_fused_token(self):
        self.assertIn("cam_total=negative", self.vocab)
        self.assertNotIn("cam_total", self.blob["edges"])
        tokens = self._tokens("synth-000")
        self.assertEqual(tokens.count("cam_total=negative"), len(self.HOURS))
        self.assertNotIn("cam_total", tokens)

    def test_unseen_categorical_and_validation_only_concept_map_to_unk(self):
        self.assertNotIn("cam_total=unable_to_assess", self.vocab)
        self.assertNotIn("valonly_score", self.blob["edges"])
        self.assertNotIn("valonly_score", self.blob["binning_sources"])
        self.assertFalse(any(t.startswith("valonly_score") for t in self.vocab))
        tokens = self._tokens(self.VALIDATION[0])
        self.assertEqual(tokens.count("<unk>"), 2 * len(self.HOURS))
        self.assertEqual(
            sum(t.startswith("braden_mobility=") for t in tokens), len(self.HOURS)
        )

    def test_binning_sources_cover_every_binned_concept_and_are_hashed(self):
        from src.data.tokenize import _json_sha256

        sources = self.blob["binning_sources"]
        self.assertEqual(set(sources), set(self.blob["edges"]))
        self.assertEqual(sources["map"], "csv")
        self.assertEqual(
            self.blob["manifest"]["hashes"]["binning_sources"], _json_sha256(sources)
        )

    def test_per_stay_arrays_stay_aligned_with_categorical_events(self):
        for row in self.events.iter_rows(named=True):
            n = len(row["token"])
            for key in ("pos_min", "target_eligible", "value", "soft_token", "soft_weight"):
                self.assertEqual(len(row[key]), n, key)
            self.assertEqual(row["n_events"], n)


if __name__ == "__main__":
    unittest.main()
