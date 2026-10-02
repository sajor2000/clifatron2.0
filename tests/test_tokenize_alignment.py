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
                "availability": "missing_storetime",
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


# U3 (R11, R12; KTD6): deterministic order and declared availability.
TIE_HOUR = 2                      # no fixture MAP row lands on an even hour
TIE_ROWS = [                      # inserted in this (deliberately unsorted) order
    ("map", 72.0),
    ("heart_rate", 90.0),
    ("map", 68.0),
]
LAB_OFFSETS_MIN = (-15, -5, 0, 1)  # lab rows relative to each stay's anchor


def _tokenize_variant(work, name, *, shuffle_seed=None, lab_lag=None, labs=False,
                      drop_vitals_availability=False):
    """Build the synthetic site, add tie rows (and optionally labs), tokenize it.

    Runs inside `work` (the artifact policy classifies shards relative to the CWD).
    Returns (events.parquet bytes, events frame, vocab blob, episodes)."""
    import copy
    from datetime import timedelta

    import polars as pl
    import yaml

    from src.data.tokenize import tokenize_site
    from src.eval.synthetic_bundle import (
        FIXTURE_COHORT,
        FIXTURE_DATA_CONFIG,
        FIXTURE_POLICY,
        SYNTHETIC_SITE,
        build_synthetic_site,
    )

    site = work / f"site_{name}"
    episode_path = build_synthetic_site(site)
    episodes = pl.read_parquet(episode_path)
    (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
    (work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))

    vitals_path = site / "clif_vitals.parquet"
    vitals = pl.read_parquet(vitals_path)
    extra = []
    for ep in episodes.iter_rows(named=True):
        when = ep["icu_admit_dttm"] + timedelta(hours=TIE_HOUR)
        for concept, value in TIE_ROWS:
            extra.append({
                "hospitalization_id": ep["hospitalization_id"], "recorded_dttm": when,
                "vital_category": concept, "vital_value": value,
                "vital_unit": "mmHg" if concept == "map" else "beats per minute",
            })
    vitals = pl.concat([vitals, pl.DataFrame(extra, schema=vitals.schema)])
    if shuffle_seed is not None:
        vitals = vitals.sample(fraction=1.0, shuffle=True, seed=shuffle_seed)
    vitals.write_parquet(vitals_path)

    cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
    cfg["cohort_contract"] = str((work / "cohort.yaml").resolve())
    cfg["artifact_policy"] = str((work / "artifact_policy.yaml").resolve())
    if drop_vitals_availability:
        cfg["tables"]["vitals"].pop("availability", None)
    if labs:
        lab_rows = []
        for ep in episodes.iter_rows(named=True):
            for k, offset in enumerate(LAB_OFFSETS_MIN):
                lab_rows.append({
                    "hospitalization_id": ep["hospitalization_id"],
                    "lab_result_dttm": ep["anchor_dttm"] + timedelta(minutes=offset),
                    "lab_category": "lactate",
                    "lab_value_numeric": 1.0 + k,
                })
        pl.DataFrame(lab_rows, schema={
            "hospitalization_id": pl.String,
            "lab_result_dttm": pl.Datetime("us", "UTC"),
            "lab_category": pl.String,
            "lab_value_numeric": pl.Float64,
        }).write_parquet(site / "clif_labs.parquet")
        cfg["tables"]["labs"] = {
            "file": "clif_labs",
            "availability_col": "lab_result_dttm",
            "availability": "result",
            "concept_col": "lab_category",
            "value_col": "lab_value_numeric",
        }
        if lab_lag is not None:
            cfg["tables"]["labs"]["availability_lag_minutes"] = lab_lag

    out = Path(f"output/intermediate_phi/order_{name}")
    tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, None,
                  episodes=episodes, artifact_policy=FIXTURE_POLICY)
    raw = (out / "events.parquet").read_bytes()
    return (raw, pl.read_parquet(out / "events.parquet"),
            json.loads((out / "vocab.json").read_text()), episodes)


class DeterministicOrderAndAvailabilityTest(unittest.TestCase):
    """U3: the post-join full-key sort and the per-table availability declaration."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.work = Path(cls._td.name)
        cls._old_cwd = os.getcwd()
        os.chdir(cls.work)
        try:
            cls.runs = {
                "plain": _tokenize_variant(cls.work, "plain"),
                "shuffled": _tokenize_variant(cls.work, "shuffled", shuffle_seed=11),
                "lag10": _tokenize_variant(cls.work, "lag10", labs=True, lab_lag=10),
                "lag0": _tokenize_variant(cls.work, "lag0", labs=True, lab_lag=0),
                "lag_default": _tokenize_variant(cls.work, "lag_default", labs=True),
            }
        finally:
            os.chdir(cls._old_cwd)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _row(self, run, stay="synth-000"):
        events = self.runs[run][1]
        return events.filter(events["hosp_id"] == stay).row(0, named=True)

    def _concepts(self, run, row):
        inv = {i: t for t, i in self.runs[run][2]["vocab"].items()}
        return [inv[t].split("=")[0] for t in row["token"]]

    def test_shuffled_input_rows_give_byte_identical_events(self):
        plain_bytes, plain, _, _ = self.runs["plain"]
        shuffled_bytes, shuffled, _, _ = self.runs["shuffled"]
        self.assertTrue(plain.equals(shuffled), "row order leaked into the token stream")
        self.assertEqual(plain_bytes, shuffled_bytes)

    def test_ties_at_one_timestamp_order_by_concept_then_value(self):
        for run in ("plain", "shuffled"):
            for stay in ("synth-000", "synth-005"):
                row = self._row(run, stay)
                concepts = self._concepts(run, row)
                at_tie = [i for i, p in enumerate(row["pos_min"]) if p == TIE_HOUR * 60]
                self.assertEqual(len(at_tie), len(TIE_ROWS))
                self.assertEqual(
                    [(concepts[i], row["value"][i]) for i in at_tie],
                    [("heart_rate", 90.0), ("map", 68.0), ("map", 72.0)],
                )

    def test_positions_are_nondecreasing_within_every_stay(self):
        for run in self.runs:
            for row in self.runs[run][1].iter_rows(named=True):
                self.assertEqual(row["pos_min"], sorted(row["pos_min"]), run)

    def _lactate(self, run, stay):
        row = self._row(run, stay)
        concepts = self._concepts(run, row)
        return [(row["pos_min"][i], row["value"][i])
                for i, c in enumerate(concepts) if c == "lactate"]

    def _anchor_min(self, run, stay):
        episodes = self.runs[run][3]
        ep = episodes.filter(episodes["hospitalization_id"] == stay).row(0, named=True)
        return int((ep["anchor_dttm"] - ep["icu_admit_dttm"]).total_seconds() // 60)

    def test_a_ten_minute_lag_excludes_events_made_available_after_the_anchor(self):
        for stay in ("synth-000", "synth-007"):
            anchor = self._anchor_min("lag10", stay)
            # -15 min -> available at -5 (kept, positioned at availability); -5 -> +5,
            # 0 -> +10, +1 -> +11 all fall after the end-inclusive anchor.
            self.assertEqual(self._lactate("lag10", stay), [(anchor - 5, 1.0)])

    def test_lag_zero_leaves_windowing_unchanged(self):
        for stay in ("synth-000", "synth-007"):
            anchor = self._anchor_min("lag0", stay)
            self.assertEqual(
                self._lactate("lag0", stay),
                [(anchor - 15, 1.0), (anchor - 5, 2.0), (anchor, 3.0)],
            )
        self.assertEqual(self.runs["lag0"][0], self.runs["lag_default"][0])

    def test_vocab_manifest_records_availability_per_table_outside_hashes(self):
        manifest = self.runs["lag10"][2]["manifest"]
        provenance = manifest["provenance"]
        self.assertEqual(
            provenance["availability"], {"vitals": "missing_storetime", "labs": "result"}
        )
        self.assertEqual(provenance["availability_lag_minutes"], {"vitals": 0, "labs": 10})
        self.assertNotIn("availability", manifest["hashes"])

    def test_a_table_without_availability_is_a_config_error(self):
        from src.data.cohort import QualificationError

        old_cwd = os.getcwd()
        os.chdir(self.work)
        try:
            with self.assertRaisesRegex(QualificationError, "vitals.*availability"):
                _tokenize_variant(self.work, "no_availability",
                                  drop_vitals_availability=True)
        finally:
            os.chdir(old_cwd)

    def test_configured_tables_declare_availability_semantics(self):
        import yaml

        from src.data.tokenize import validate_table_availability

        root = Path(__file__).parents[1]
        tables = yaml.safe_load((root / "configs/data.yaml").read_text())["tables"]
        self.assertEqual(tables["vitals"]["availability"], "missing_storetime")
        semantics = validate_table_availability(tables)
        self.assertEqual(set(semantics), set(tables))

    def test_group_by_maintain_order_keeps_sorted_within_group_order(self):
        """`encode` relies on map_groups seeing each stay's rows in the sorted order."""
        import polars as pl

        frame = pl.DataFrame({
            "hosp_id": ["b", "a", "b", "a", "b", "a"],
            "k": [3, 2, 1, 3, 2, 1],
        }).sort(["hosp_id", "k"], maintain_order=True)
        seen = (
            frame.group_by("hosp_id", maintain_order=True)
            .map_groups(lambda g: pl.DataFrame({"hosp_id": g["hosp_id"][0],
                                                "ks": [g["k"].to_list()]}))
        )
        self.assertEqual(seen["hosp_id"].to_list(), ["a", "b"])
        self.assertEqual(seen["ks"].to_list(), [[1, 2, 3], [1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
