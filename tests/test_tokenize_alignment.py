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
                tokenize_site(cfg, SYNTHETIC_SITE, site, out, None,
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
                    cfg, SYNTHETIC_SITE, site, out, None,
                    episodes=episodes, artifact_policy=FIXTURE_POLICY,
                )
                blob = json.loads((out / "vocab.json").read_text())
                vocab, manifest = blob["vocab"], blob["manifest"]

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
                        cfg, SYNTHETIC_SITE, site, out2,
                        {**blob, "vocab": bad_vocab, "manifest": manifest},
                        episodes=episodes, artifact_policy=FIXTURE_POLICY,
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
            tokenize_site(cfg, SYNTHETIC_SITE, site, out, None,
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
        self.assertNotIn("cam_total", self.blob["segments"])
        tokens = self._tokens("synth-000")
        self.assertEqual(tokens.count("cam_total=negative"), len(self.HOURS))
        self.assertNotIn("cam_total", tokens)

    def test_unseen_categorical_and_validation_only_concept_map_to_unk(self):
        self.assertNotIn("cam_total=unable_to_assess", self.vocab)
        self.assertNotIn("valonly_score", self.blob["segments"])
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
        self.assertEqual(set(sources), set(self.blob["segments"]))
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
    tokenize_site(cfg, SYNTHETIC_SITE, site, out, None,
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



# U4 (R6-R10, R21; KTD5, KTD11): new event sources, end to end through `tokenize_site`.
RESP_NUMERIC = (
    "fio2_set", "lpm_set", "tidal_volume_set", "resp_rate_set", "pressure_control_set",
    "pressure_support_set", "flow_rate_set", "peak_inspiratory_pressure_set",
    "inspiratory_time_set", "peep_set", "tidal_volume_obs", "resp_rate_obs",
    "plateau_pressure_obs", "peak_inspiratory_pressure_obs", "peep_obs", "minute_vent_obs",
    "mean_airway_pressure_obs",
)
RESP_CATEGORICAL = ("device_category", "mode_category", "tracheostomy")
CRRT_NUMERIC = ("blood_flow_rate", "pre_filter_replacement_fluid_rate",
                "post_filter_replacement_fluid_rate", "dialysate_flow_rate",
                "ultrafiltration_out")
U4_VALIDATION = ["synth-001"]      # the stay whose only weight is charted AFTER its dose
STATIC_ORDER = ("age_decile", "sex", "race", "ethnicity", "admission_type")


def _write(frame_rows, schema, path):
    import polars as pl

    pl.DataFrame(frame_rows, schema=schema, orient="row").write_parquet(path)


def _build_u4_site(work, name, *, static_tokens=None):
    """Synthetic site with every U4 source; tokenized with the repo's configs/data.yaml
    table specs. Returns (events, vocab blob, stats, episodes)."""
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

    site = work / f"site_{name}"
    episodes = _repartition(pl.read_parquet(build_synthetic_site(site)), U4_VALIDATION)
    (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
    (work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))
    eps = list(episodes.iter_rows(named=True))
    utc = pl.Datetime("us", "UTC")
    h = lambda ep, hours: ep["icu_admit_dttm"] + timedelta(hours=hours)  # noqa: E731

    # Hospital admission for synth-002 precedes its ICU admission by 6 h (carry-forward).
    hosp = pl.read_parquet(site / "clif_hospitalization.parquet")
    hosp = hosp.with_columns(
        pl.when(pl.col("hospitalization_id") == "synth-002")
        .then(pl.col("admission_dttm") - pl.duration(hours=6))
        .otherwise(pl.col("admission_dttm")).alias("admission_dttm"),
        pl.lit("ed").alias("admission_type_category"),
    )
    hosp.write_parquet(site / "clif_hospitalization.parquet")
    _write([(f"synth-p-{i:03d}", "Female" if i % 2 == 0 else "Male", "White", "Non-Hispanic")
            for i in range(len(eps))],
           {"patient_id": pl.String, "sex_category": pl.String, "race_category": pl.String,
            "ethnicity_category": pl.String}, site / "clif_patient.parquet")

    vitals = pl.read_parquet(site / "clif_vitals.parquet")
    weights = [{"hospitalization_id": ep["hospitalization_id"],
                "recorded_dttm": h(ep, 5 if ep["hospitalization_id"] == "synth-001" else 1),
                "vital_category": "weight_kg", "vital_value": 80.0, "vital_unit": "kg"}
               for ep in eps]
    pl.concat([vitals, pl.DataFrame(weights, schema=vitals.schema)]).write_parquet(
        site / "clif_vitals.parquet")

    meds = []
    for ep in eps:
        stay = ep["hospitalization_id"]
        meds += [
            (stay, "fentanyl", h(ep, 3), "start", 100.0, "mcg/hour"),
            (stay, "vasopressin", h(ep, 4), "start", 2.4, "units/hour"),
            (stay, "fentanyl", h(ep, 8), "stop", 0.0, "mcg/hour"),
            (stay, "propofol", h(ep, 2), "start", 20.0, "mL/hour"),
        ]
        if stay == "synth-001":
            meds.append((stay, "fentanyl", h(ep, 6), "dose_change", 100.0, "mcg/hour"))
    _write(meds, {"hospitalization_id": pl.String, "med_category": pl.String,
                  "admin_dttm": utc, "mar_action_category": pl.String,
                  "med_dose": pl.Float32, "med_dose_unit": pl.String},
           site / "clif_medication_admin_continuous.parquet")

    _write([row for ep in eps for row in (
        (ep["hospitalization_id"], "cefepime", h(ep, 5), "given", 2.0, "grams"),
        (ep["hospitalization_id"], "cefepime", h(ep, 9), "given", 2000.0, "mg"),
        (ep["hospitalization_id"], "acetaminophen", h(ep, 11), "given", 1.0, "dose"),
    )], {"hospitalization_id": pl.String, "med_category": pl.String, "admin_dttm": utc,
         "mar_action_category": pl.String, "med_dose": pl.Float32,
         "med_dose_unit": pl.String},
        site / "clif_medication_admin_intermittent.parquet")

    resp_schema = {"hospitalization_id": pl.String, "recorded_dttm": utc,
                   "device_category": pl.String, "mode_category": pl.String,
                   "tracheostomy": pl.Boolean, **{c: pl.Float64 for c in RESP_NUMERIC}}
    resp = []
    for ep in eps:
        row = dict.fromkeys(resp_schema)
        row.update(hospitalization_id=ep["hospitalization_id"], recorded_dttm=h(ep, 6),
                   mode_category="Assist Control-Volume Control", fio2_set=0.4, peep_set=8.0)
        resp.append(row)
    pl.DataFrame(resp, schema=resp_schema).write_parquet(
        site / "clif_respiratory_support.parquet")

    assessments = []
    for i, ep in enumerate(eps):
        stay = ep["hospitalization_id"]
        assessments += [
            (stay, h(ep, 7), "gcs_total", 8.0, None),
            (stay, h(ep, 10), "gcs_total", float(3 + i % 13), None),
            (stay, h(ep, 7), "RASS", -2.0, None),
            (stay, h(ep, 7), "cam_total", None, "Positive"),
        ]
    _write(assessments, {"hospitalization_id": pl.String, "recorded_dttm": utc,
                         "assessment_category": pl.String, "numerical_value": pl.Float64,
                         "categorical_value": pl.String},
           site / "clif_patient_assessments.parquet")

    _write([(ep["hospitalization_id"], h(ep, 6.5), "cvvhdf", 200.0, None, None, None, None)
            for ep in eps],
           {"hospitalization_id": pl.String, "recorded_dttm": utc,
            "crrt_mode_category": pl.String, **{c: pl.Float32 for c in CRRT_NUMERIC}},
           site / "clif_crrt_therapy.parquet")
    _write([row for ep in eps for row in (
        (ep["hospitalization_id"], h(ep, 6.5), "Other", "ECMO", 3000.0, None, None, None),
        (ep["hospitalization_id"], h(ep, 6.5), "Other", "LVAD", 9000.0, None, None, None),
    )], {"hospitalization_id": pl.String, "recorded_dttm": utc, "device_category": pl.String,
         "mcs_group": pl.String, "device_rate": pl.Float32, "flow": pl.Float32,
         "sweep": pl.Float32, "fdO2": pl.Float32}, site / "clif_ecmo_mcs.parquet")

    admission = dict(zip(hosp["patient_id"], hosp["admission_dttm"]))
    code = []
    for ep in eps:
        pid = ep["patient_id"]
        if pid == "synth-p-000":
            code += [(pid, admission[pid] - timedelta(days=2), "DNR"),
                     (pid, h(ep, 12), "DNR/DNI")]
        elif pid == "synth-p-002":
            code += [(pid, admission[pid] - timedelta(days=1), "Full"),
                     (pid, h(ep, -3), "DNR")]          # after hospital, before ICU admit
        else:
            code.append((pid, admission[pid] - timedelta(days=1), "Full"))
    _write(code, {"patient_id": pl.String, "start_dttm": utc,
                  "code_status_category": pl.String}, site / "clif_code_status.parquet")

    ep0 = eps[0]
    position = [("synth-000", h(ep0, 1) + timedelta(minutes=10 * k), "not_prone")
                for k in range(100)]
    position.append(("synth-000", h(ep0, 18), "prone"))
    _write(position[::-1], {"hospitalization_id": pl.String, "recorded_dttm": utc,
                            "position_category": pl.String}, site / "clif_position.parquet")

    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
    cfg["cohort_contract"] = str((work / "cohort.yaml").resolve())
    cfg["artifact_policy"] = str((work / "artifact_policy.yaml").resolve())
    cfg["tables"] = copy.deepcopy(data_cfg["tables"])
    cfg["static_source"] = copy.deepcopy(data_cfg["static_source"])
    cfg["static_tokens"] = (list(data_cfg["static_tokens"]) if static_tokens is None
                            else static_tokens)
    cfg["value_binning"].update({
        "scheme": "clinical_segment",
        "segment_source": str(ROOT / data_cfg["value_binning"]["segment_source"]),
        "coverage": "all",
    })
    out = Path(f"output/intermediate_phi/u4_{name}")
    stats: dict = {}
    tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, episodes=episodes,
                  artifact_policy=FIXTURE_POLICY, stats=stats)
    return (pl.read_parquet(out / "events.parquet"),
            json.loads((out / "vocab.json").read_text()), stats, episodes)


class NewEventSourcesTest(unittest.TestCase):
    """U4: doses (continuous + intermittent), ventilator, assessments, CRRT, ECMO/MCS,
    code status, position and static admission tokens."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old_cwd = os.getcwd()
        os.chdir(work)
        try:
            cls.events, cls.blob, cls.stats, cls.episodes = _build_u4_site(work, "full")
            cls.nostatic = _build_u4_site(work, "nostatic", static_tokens=[])
        finally:
            os.chdir(old_cwd)
        cls.vocab = cls.blob["vocab"]
        cls.inv = {i: t for t, i in cls.vocab.items()}

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _rows(self, stay, events=None, inv=None):
        """[(token, pos_min, value, target_eligible)] for one stay, in stream order."""
        events = self.events if events is None else events
        inv = self.inv if inv is None else inv
        row = events.filter(events["hosp_id"] == stay).row(0, named=True)
        return [(inv[t], p, v, e) for t, p, v, e in zip(
            row["token"], row["pos_min"], row["value"], row["target_eligible"])]

    def _concept(self, stay, prefix):
        return [r for r in self._rows(stay) if r[0].split("=")[0] == prefix]

    def _zero_bin(self, concept):
        segments = self.blob["segments"][concept]
        return next(i for i, s in enumerate(segments) if s["lo"] == s["hi"] == 0.0)

    # --- continuous doses ---------------------------------------------------------
    def test_per_kg_conversion_uses_the_prior_weight(self):
        fentanyl = self._concept("synth-000", "fentanyl_mcg_kg_hr")
        self.assertEqual([(p, v) for _, p, v, _ in fentanyl], [(180, 1.25), (480, 0.0)])

    def test_no_prior_weight_falls_back_to_native_unit_and_never_uses_a_later_weight(self):
        rows = self._rows("synth-001")
        dose_rows = [(t.split("=")[0], p, v) for t, p, v, _ in rows
                     if t.startswith("fentanyl_")]
        # +3 h: the only weight is charted at +5 h, so it must not be used.
        self.assertEqual(dose_rows[0], ("fentanyl_mcg_hr", 180, 100.0))
        # +6 h: the +5 h weight is now available.
        self.assertEqual(dose_rows[1], ("fentanyl_mcg_kg_hr", 360, 1.25))
        self.assertNotIn("<unk>", [t for t, *_ in rows if t.startswith("fentanyl")])
        self.assertEqual(self.stats["dose_conversion"]["meds"]["no_weight"], 1)

    def test_fallback_concept_is_fitted_from_a_fit_only_shadow(self):
        # Every TRAIN dose was converted, yet the native-unit fallback has frozen bins.
        self.assertIn("fentanyl_mcg_hr", self.blob["segments"])
        self.assertIn("fentanyl_mcg_hr", self.blob["binning_sources"])
        self.assertTrue(any(t.startswith("fentanyl_mcg_hr=") for t in self.vocab))
        self.assertEqual(self.blob["segments"]["fentanyl_mcg_hr"][0],
                         {"lo": 0.0, "hi": 0.0, "lo_closed": True, "hi_closed": True})
        # The shadow never reaches the token stream.
        for stay in ("synth-000", "synth-005"):
            self.assertEqual(self._concept(stay, "fentanyl_mcg_hr"), [])
        provenance = self.blob["manifest"]["provenance"]
        self.assertEqual(provenance["dose_conversion"], self.stats["dose_conversion"])

    def test_units_per_hour_become_units_per_minute(self):
        vaso = self._concept("synth-000", "vasopressin_u_min")
        self.assertEqual(len(vaso), 1)
        self.assertAlmostEqual(vaso[0][2], 0.04, places=6)

    def test_a_stop_row_lands_in_the_zero_bin(self):
        stop = [r for r in self._concept("synth-000", "fentanyl_mcg_kg_hr") if r[1] == 480]
        self.assertEqual(stop[0][0], f"fentanyl_mcg_kg_hr={self._zero_bin('fentanyl_mcg_kg_hr')}")
        running = [r for r in self._concept("synth-000", "fentanyl_mcg_kg_hr") if r[1] == 180]
        self.assertNotEqual(running[0][0], stop[0][0])

    def test_volume_rate_is_an_unconvertible_native_fallback(self):
        self.assertEqual(len(self._concept("synth-000", "propofol_ml_hr")), 1)
        self.assertGreater(self.stats["dose_conversion"]["meds"]["unconvertible"], 0)

    # --- intermittent doses -------------------------------------------------------
    def test_grams_and_milligrams_share_one_concept_and_bin(self):
        cefepime = self._concept("synth-000", "cefepime_mg")
        self.assertEqual(len(cefepime), 2)
        self.assertEqual(cefepime[0][0], cefepime[1][0])
        self.assertEqual([v for _, _, v, _ in cefepime], [2000.0, 2000.0])
        self.assertIn("cefepime_mg", self.blob["segments"])
        self.assertEqual(self._zero_bin("cefepime_mg"), 0)

    def test_dose_unit_falls_back_to_category_unit(self):
        self.assertEqual(len(self._concept("synth-000", "acetaminophen_dose")), 1)
        self.assertEqual(
            self.stats["dose_conversion"]["meds_intermittent"]["unconvertible"],
            len(self.episodes),
        )

    # --- U5: reference units and concept sources ------------------------------------
    def test_reference_units_cover_binned_concepts_dose_targets_and_device_metrics(self):
        units = self.blob["reference_units"]
        self.assertEqual(set(units["concepts"]), set(self.blob["segments"]))
        self.assertEqual(units["concepts"]["fentanyl_mcg_kg_hr"], "mcg/kg/hr")
        # A native-unit fallback concept carries the unit its rows are charted with
        # (the concept's unit suffix, U4).
        self.assertEqual(units["concepts"]["fentanyl_mcg_hr"], "mcg_hr")
        self.assertEqual(units["concepts"]["cefepime_mg"], "mg")
        # CLIF vitals carry no unit column and this fixture config declares no canonical
        # unit, so map has no reference unit to check (the configured one wins when set).
        self.assertIsNone(units["concepts"]["map"])
        self.assertEqual(units["concepts"]["ecmo_device_rate"], "device_metric:device_rate")
        self.assertIsNone(units["concepts"]["age_decile"])
        # Dose target units moved from (unhashed) provenance into the hashed field.
        self.assertEqual(units["dose_targets"]["fentanyl"], "mcg/kg/hr")
        self.assertNotIn("dose_target_units", self.blob["manifest"]["provenance"])

    def test_concept_sources_name_each_concepts_tables_and_the_treatment_tables(self):
        sources = self.blob["concept_sources"]
        self.assertEqual(sources["tables"]["fentanyl_mcg_kg_hr"], ["meds"])
        self.assertEqual(sources["tables"]["mode_category"], ["resp_support"])
        self.assertEqual(sources["tables"]["gcs_total"], ["assessments"])
        self.assertEqual(sources["tables"]["sex"], ["static"])
        self.assertLessEqual({"meds", "meds_intermittent", "resp_support", "crrt", "ecmo",
                              "code_status", "position", "adt", "static"},
                             set(sources["treatment_sources"]))
        self.assertNotIn("assessments", sources["treatment_sources"])
        self.assertNotIn("vitals", sources["treatment_sources"])

    # --- ventilator ---------------------------------------------------------------
    def test_a_resp_row_melts_to_its_settings_plus_a_fused_mode(self):
        resp_concepts = set(RESP_NUMERIC) | set(RESP_CATEGORICAL)
        resp = [r for r in self._rows("synth-000") if r[0].split("=")[0] in resp_concepts]
        self.assertEqual(len(resp), 3)
        self.assertEqual({r[1] for r in resp}, {360})
        tokens = [r[0] for r in resp]
        self.assertIn("mode_category=assist_control-volume_control", tokens)
        fio2 = next(r for r in resp if r[0].startswith("fio2_set="))
        peep = next(r for r in resp if r[0].startswith("peep_set="))
        self.assertEqual((fio2[2], peep[2]), (0.4, 8.0))
        self.assertEqual(self.blob["binning_sources"]["fio2_set"], "csv")

    # --- assessments ----------------------------------------------------------------
    def test_gcs_is_ordinal_and_target_eligible(self):
        from src.data.segments import bin_index

        self.assertEqual(self.blob["binning_sources"]["gcs_total"], "ordinal")
        gcs = self._concept("synth-000", "gcs_total")
        first = next(r for r in gcs if r[1] == 420)
        self.assertEqual(first[0],
                         f"gcs_total={bin_index(8.0, self.blob['segments']['gcs_total'])}")
        self.assertTrue(all(r[3] for r in gcs))

    def test_assessment_concepts_are_lowercased(self):
        self.assertIn("rass", self.blob["segments"])
        self.assertNotIn("RASS", self.blob["segments"])

    def test_value_only_cam_result_is_fused(self):
        cam = self._concept("synth-000", "cam_total")
        self.assertEqual([r[0] for r in cam], ["cam_total=positive"])
        self.assertTrue(cam[0][3])

    # --- CRRT / ECMO-MCS ------------------------------------------------------------
    def test_crrt_settings_and_mode(self):
        self.assertEqual(len(self._concept("synth-000", "blood_flow_rate")), 1)
        self.assertEqual([r[0] for r in self._concept("synth-000", "crrt_mode_category")],
                         ["crrt_mode_category=cvvhdf"])

    def test_mcs_metrics_are_qualified_by_device_group(self):
        ecmo = self._concept("synth-000", "ecmo_device_rate")
        lvad = self._concept("synth-000", "lvad_device_rate")
        self.assertEqual((len(ecmo), len(lvad)), (1, 1))
        self.assertEqual((ecmo[0][2], lvad[0][2]), (3000.0, 9000.0))
        self.assertIn("ecmo_device_rate", self.blob["segments"])
        self.assertIn("lvad_device_rate", self.blob["segments"])

    # --- code status / position -------------------------------------------------------
    def test_code_status_in_effect_at_admission_and_mid_stay_change(self):
        self.assertEqual([(t, p) for t, p, *_ in self._concept("synth-000", "code_status")],
                         [("code_status=dnr", 0), ("code_status=dnr/dni", 720)])
        self.assertEqual([(t, p) for t, p, *_ in self._concept("synth-003", "code_status")],
                         [("code_status=full", 0)])

    def test_code_status_charted_before_icu_admission_carries_forward_to_the_window(self):
        # synth-002: Full at hospital admission (ICU - 6 h), DNR at ICU - 3 h. Only the
        # state in effect at ICU admission is carried, positioned at the window start.
        self.assertEqual([(t, p) for t, p, *_ in self._concept("synth-002", "code_status")],
                         [("code_status=dnr", 0)])

    def test_position_emits_only_transitions(self):
        # Transition semantics: rows are ordered by (recorded_dttm, normalized value); the
        # first observation of a stay is emitted, then only rows whose value differs
        # from the previous row's. 100 not_prone rows then one prone -> two events.
        self.assertEqual([(t, p) for t, p, *_ in self._concept("synth-000", "position")],
                         [("position=not_prone", 60), ("position=prone", 1080)])
        self.assertEqual(self._concept("synth-003", "position"), [])

    # --- static admission tokens --------------------------------------------------------
    def test_static_tokens_lead_every_stay_once_in_fixed_order(self):
        self.assertEqual(self.blob["binning_sources"]["age_decile"], "quantile")
        for i, stay in enumerate(self.events["hosp_id"].to_list()):
            rows = self._rows(stay)
            head = [(t.split("=")[0], p) for t, p, *_ in rows[:len(STATIC_ORDER)]]
            self.assertEqual(head, [(c, 0) for c in STATIC_ORDER], stay)
            for concept in STATIC_ORDER:
                self.assertEqual(sum(t.split("=")[0] == concept for t, *_ in rows), 1)
        tokens = [t for t, *_ in self._rows("synth-000")[:len(STATIC_ORDER)]]
        self.assertEqual(tokens[1:], ["sex=female", "race=white", "ethnicity=non-hispanic",
                                      "admission_type=ed"])

    def test_no_static_tokens_when_disabled(self):
        events, blob, _, _ = self.nostatic
        inv = {i: t for t, i in blob["vocab"].items()}
        for stay in events["hosp_id"].to_list():
            concepts = {t.split("=")[0] for t, *_ in self._rows(stay, events, inv)}
            self.assertFalse(concepts & set(STATIC_ORDER), stay)
        self.assertFalse(any(t.split("=")[0] in STATIC_ORDER for t in blob["vocab"]))

    # --- treatment rule ----------------------------------------------------------------
    def test_treatment_and_context_sources_are_never_targets(self):
        input_only = (set(RESP_NUMERIC) | set(RESP_CATEGORICAL) | set(CRRT_NUMERIC)
                      | set(STATIC_ORDER) | {
                          "fentanyl_mcg_kg_hr", "fentanyl_mcg_hr", "vasopressin_u_min",
                          "propofol_ml_hr", "cefepime_mg", "acetaminophen_dose",
                          "crrt_mode_category", "ecmo_device_rate", "lvad_device_rate",
                          "ecmo_device_category", "lvad_device_category",
                          "code_status", "position", "icu"})
        targets = {"map", "weight_kg", "gcs_total", "rass", "cam_total"}
        seen = set()
        for stay in self.events["hosp_id"].to_list():
            for token, _, _, eligible in self._rows(stay):
                concept = token.split("=")[0]
                seen.add(concept)
                if concept in input_only:
                    self.assertFalse(eligible, token)
                elif concept in targets:
                    self.assertTrue(eligible, token)
                else:
                    self.fail(f"unclassified concept {concept!r}")
        self.assertLessEqual({"fentanyl_mcg_kg_hr", "fentanyl_mcg_hr", "fio2_set",
                              "peep_set", "mode_category", "blood_flow_rate", "code_status",
                              "position", "age_decile", "cefepime_mg", "gcs_total"}, seen)

    def test_per_stay_arrays_stay_aligned(self):
        for row in self.events.iter_rows(named=True):
            n = len(row["token"])
            for key in ("pos_min", "target_eligible", "value", "soft_token", "soft_weight"):
                self.assertEqual(len(row[key]), n, key)
            self.assertEqual(row["pos_min"], sorted(row["pos_min"]))

if __name__ == "__main__":
    unittest.main()
