"""CLIF 2.1 / mCIDE conformance gate and per-site harmonization (configs/data.yaml
`clif_conformance`, `site_harmonization`, `gcs_not_testable`, `bp_method`).

Every fixture here is synthetic (ids ``SYN-*``); nothing reads staged data."""
import copy
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl
import yaml

from src.data.clif_conformance import (
    check_conformance,
    check_config_columns,
    compliance_table,
    global_record,
    harmonization_record,
    load_snapshot,
    mapped_sql,
    site_harmonization,
)
from src.data.cohort import QualificationError
from src.data.tokenize import (
    _check_harmonization_binding,
    _event_sort,
    _read_source,
    apply_bp_method,
    apply_gcs_not_testable,
    bp_config,
    build_segments,
    build_vocab,
    categorical_token,
    gcs_rule,
    harmonization_tokens,
    units_equivalent,
)

ROOT = Path(__file__).resolve().parents[1]
UTC_DT = pl.Datetime("us", "UTC")
T0 = datetime(2031, 5, 1, 8, tzinfo=UTC)
SITE = "SYN-SITE"


def repo_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs/data.yaml").read_text())


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET TimeZone = 'UTC'")
    return c


def write_conformant_site(base: Path, n: int = 12) -> None:
    """A small site whose every checked category is a permissible mCIDE value."""
    ids = [f"SYN-H-{i:03d}" for i in range(n)]
    pids = [f"SYN-P-{i:03d}" for i in range(n)]
    ts = [T0 + timedelta(hours=i) for i in range(n)]
    pl.DataFrame({"hospitalization_id": ids, "patient_id": pids, "admission_dttm": ts,
                  "discharge_dttm": [t + timedelta(days=3) for t in ts],
                  "admission_type_category": ["ed"] * n,
                  "discharge_category": ["Home"] * n},
                 schema_overrides={"admission_dttm": UTC_DT, "discharge_dttm": UTC_DT}
                 ).write_parquet(base / "clif_hospitalization.parquet")
    pl.DataFrame({"patient_id": pids, "sex_category": ["Female"] * n,
                  "race_category": ["White"] * n, "ethnicity_category": ["Non-Hispanic"] * n}
                 ).write_parquet(base / "clif_patient.parquet")
    pl.DataFrame({"hospitalization_id": ids, "recorded_dttm": ts,
                  "vital_category": ["map"] * n, "vital_value": [70.0] * n,
                  "vital_name": ["SYN Arterial MAP"] * n},
                 schema_overrides={"recorded_dttm": UTC_DT}
                 ).write_parquet(base / "clif_vitals.parquet")
    pl.DataFrame({"hospitalization_id": ids, "in_dttm": ts, "location_category": ["icu"] * n,
                  "location_type": ["medical_icu"] * n},
                 schema_overrides={"in_dttm": UTC_DT}).write_parquet(base / "clif_adt.parquet")
    pl.DataFrame({"hospitalization_id": ids, "recorded_dttm": ts,
                  "device_category": ["IMV"] * n, "mode_category": [None] * n,
                  "tracheostomy": [0] * n, "fio2_set": [0.4] * n},
                 schema_overrides={"recorded_dttm": UTC_DT, "mode_category": pl.String,
                                   "tracheostomy": pl.Int32}
                 ).write_parquet(base / "clif_respiratory_support.parquet")


def site_cfg(**site_decl) -> dict:
    """Repo config restricted to the vitals, resp_support and adt tables plus static
    entities, with `site_decl` as the synthetic site's declarations."""
    cfg = repo_cfg()
    cfg["tables"] = {k: cfg["tables"][k] for k in ("vitals", "resp_support", "adt")}
    keep = {k: v for k, v in cfg["clif_conformance"]["checked"].items()
            if k.split(".")[0] in ("vitals", "resp_support", "adt", "hospitalization",
                                   "patient")}
    cfg["clif_conformance"]["checked"] = keep
    cfg["clif_conformance"]["extensions"] = {}
    cfg["site_harmonization"] = {SITE: site_decl} if site_decl else {}
    return cfg


class SnapshotTest(unittest.TestCase):
    def test_snapshot_is_the_pinned_public_release(self):
        snap = load_snapshot(ROOT / "configs/clif_mcide_2.1.1")
        manifest = yaml.safe_load((ROOT / "configs/clif_mcide_2.1.1/manifest.yaml").read_text())
        self.assertEqual(snap["version"], "2.1.1")
        self.assertEqual(manifest["commit"], "356e3f3de5e1c8b687044956d6be4988a6f365dd")
        self.assertIn("v2.1.1", (ROOT / "configs/clif_mcide_2.1.1/README.md").read_text())
        self.assertIn("Nasal Cannula", snap["permissible"]["respiratory_support.device_category"])
        self.assertIn("dextrose_5_water",
                      snap["permissible"]["medication_admin_continuous.med_category"])
        self.assertIn("DNI_only", snap["permissible"]["code_status.code_status_category"])
        self.assertNotIn("cvicu_icu", snap["permissible"]["adt.location_type"])
        self.assertEqual(snap["permissible"]["ecmo_mcs.mcs_group"],
                         sorted(["ECMO", "IABP", "RVAD", "durable_LVAD", "temporary_LVAD"]))
        # The action CSV opens with a blank line; its values still parse.
        self.assertIn("stop",
                      snap["permissible"]["medication_admin_continuous.mar_action_category"])
        self.assertIn("sweep_set", snap["tables"]["ecmo_mcs"])
        self.assertIn("sweep", snap["variants"]["ecmo_mcs"]["clifpy_2.1"])
        self.assertEqual(len(snap["sha256"]), 64)
        self.assertEqual(snap["sha256"], load_snapshot("configs/clif_mcide_2.1.1")["sha256"])

    def test_every_configured_column_is_a_clif_21_column(self):
        cfg = repo_cfg()
        self.assertEqual(check_config_columns(cfg, load_snapshot(cfg["clif_conformance"]
                                                                 ["snapshot"])), [])
        broken = copy.deepcopy(cfg)
        broken["tables"]["vitals"]["value_col"] = "vital_numeric"
        problems = check_config_columns(broken, load_snapshot(cfg["clif_conformance"]
                                                              ["snapshot"]))
        self.assertTrue(any("vital_numeric" in p for p in problems))
        # Without the pinned clifpy variant, the ECMO columns are not DDL columns.
        unpinned = copy.deepcopy(cfg)
        unpinned["clif_conformance"]["table_variants"] = {}
        problems = check_config_columns(unpinned, load_snapshot(cfg["clif_conformance"]
                                                                ["snapshot"]))
        self.assertTrue(any("ecmo" in p and "device_rate" in p for p in problems))


class GateTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.base = Path(self._td.name)
        write_conformant_site(self.base)

    def tearDown(self):
        self._td.cleanup()

    def run_gate(self, cfg, **kw):
        return check_conformance(con(), self.base, cfg, SITE, **kw)

    def test_conformant_site_passes(self):
        record = self.run_gate(site_cfg())
        self.assertEqual(record["failures"], 0)
        table = compliance_table(record)
        self.assertEqual(table["vitals"]["failing"], 0)
        self.assertEqual(table["vitals"]["compliant"], 12)
        # Integer 0/1 tracheostomy is compliant as stored.
        self.assertEqual(record["columns"]["resp_support.tracheostomy"]["compliant"], 12)

    def test_undeclared_non_mcide_category_fails_with_an_aggregate_message(self):
        adt = pl.read_parquet(self.base / "clif_adt.parquet")
        adt.with_columns(pl.when(pl.int_range(pl.len()) < 3).then(pl.lit("cvicu_icu"))
                         .otherwise(pl.col("location_type")).alias("location_type")
                         ).write_parquet(self.base / "clif_adt.parquet")
        with self.assertRaises(QualificationError) as ctx:
            self.run_gate(site_cfg())
        message = str(ctx.exception)
        self.assertIn("adt.location_type", message)
        self.assertIn("'cvicu_icu'", message)
        self.assertIn("<10 rows", message)        # 3 rows: small cell
        self.assertNotIn("SYN-H-", message)       # never a stay identifier
        # report-only mode records it instead
        record = self.run_gate(site_cfg(), raise_on_failure=False)
        self.assertEqual(record["columns"]["adt.location_type"]["failing"], 3)

    def test_declared_exception_passes_and_is_reported(self):
        adt = pl.read_parquet(self.base / "clif_adt.parquet")
        adt.with_columns(pl.lit("cvicu_icu").alias("location_type")).write_parquet(
            self.base / "clif_adt.parquet")
        cfg = site_cfg(declared_exceptions={"adt.location_type": {"cvicu_icu": "product call"}})
        record = self.run_gate(cfg)
        entry = record["columns"]["adt.location_type"]
        self.assertEqual(entry["declared_exception"], 12)
        self.assertEqual(entry["values"]["cvicu_icu"]["status"], "declared_exception")
        # A declaration needs its reason.
        with self.assertRaisesRegex(QualificationError, "note"):
            site_harmonization(site_cfg(declared_exceptions={"adt.location_type":
                                                             {"cvicu_icu": ""}}), SITE)

    def test_aliased_value_is_counted_as_aliased(self):
        resp = pl.read_parquet(self.base / "clif_respiratory_support.parquet")
        resp.with_columns(pl.lit("Ventilator").alias("device_category")).write_parquet(
            self.base / "clif_respiratory_support.parquet")
        with self.assertRaises(QualificationError):
            self.run_gate(site_cfg())
        record = self.run_gate(site_cfg(category_map={"resp_support.device_category":
                                                      {"Ventilator": "IMV"}}))
        entry = record["columns"]["resp_support.device_category"]
        self.assertEqual(entry["aliased"], 12)
        self.assertEqual(entry["values"]["Ventilator -> IMV"]["rows"], 12)

    def test_missing_required_column_fails(self):
        vitals = pl.read_parquet(self.base / "clif_vitals.parquet")
        vitals.drop("vital_value").write_parquet(self.base / "clif_vitals.parquet")
        with self.assertRaisesRegex(QualificationError, "required column.*vital_value"):
            self.run_gate(site_cfg())

    def test_column_alias_satisfies_a_required_column(self):
        vitals = pl.read_parquet(self.base / "clif_vitals.parquet")
        vitals.rename({"vital_value": "vital_numeric"}).write_parquet(
            self.base / "clif_vitals.parquet")
        record = self.run_gate(site_cfg(column_aliases={"vitals": {"vital_value":
                                                                   "vital_numeric"}}))
        self.assertEqual(record["failures"], 0)

    def test_boolean_tracheostomy_is_aliased_and_other_values_fail(self):
        resp = pl.read_parquet(self.base / "clif_respiratory_support.parquet")
        resp.with_columns(pl.lit(True).alias("tracheostomy")).write_parquet(
            self.base / "clif_respiratory_support.parquet")
        entry = self.run_gate(site_cfg())["columns"]["resp_support.tracheostomy"]
        self.assertEqual(entry["aliased"], 12)
        resp.with_columns(pl.lit(2).alias("tracheostomy")).write_parquet(
            self.base / "clif_respiratory_support.parquet")
        with self.assertRaisesRegex(QualificationError, "not a 0/1 flag"):
            self.run_gate(site_cfg())


class MimicDeclarationsTest(unittest.TestCase):
    """Every known MIMIC deviation (output/final_no_phi/clif21_audit_*.md) is either
    aliased or declared - checked against the snapshot, without data."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = repo_cfg()
        cls.snap = load_snapshot(cls.cfg["clif_conformance"]["snapshot"])
        cls.decl = site_harmonization(cls.cfg, "mimic")

    def resolve(self, key, raw, **row):
        rules = self.decl["category_map"].get(key, {})
        for source, value_rules in rules.items():
            if source.lower() == str(raw).lower():
                for rule in value_rules:
                    if all(str(row.get(c, "")).lower() == v.lower()
                           for c, v in rule["when"].items()):
                        return rule["to"]
        return raw

    def status(self, key, raw, **row):
        spec = self.cfg["clif_conformance"]["checked"][key]
        permitted = {v.lower().replace(" ", "_") for v in self.snap["permissible"][spec]}
        for extra in (self.cfg["clif_conformance"]["extensions"].get(key) or {}).get(
                "also_permit", ()):
            permitted |= {v.lower().replace(" ", "_") for v in self.snap["permissible"][extra]}
        mapped = self.resolve(key, raw, **row)
        if str(mapped).lower().replace(" ", "_") in permitted:
            return "aliased" if mapped != raw else "compliant"
        if mapped in self.decl["declared_exceptions"].get(key, {}):
            return "declared_exception"
        return "failing"

    def test_known_deviations_are_aliased_or_declared(self):
        cases = [
            ("meds.med_category", "dextrose_in_water_d5w", {}, "aliased"),
            ("meds.med_category", "dextrose", {"med_name": "Dextrose 10%"}, "aliased"),
            ("meds.med_category", "dextrose", {"med_name": "Dextrose 20%"}, "aliased"),
            ("meds.med_category", "albumin_infusion", {}, "aliased"),
            ("meds.med_category", "aminocaproic", {}, "aliased"),
            ("meds.med_category", "magnesium", {}, "aliased"),
            ("meds.med_category", "acetaminophen", {}, "declared_exception"),
            ("meds.med_category", "alteplase", {}, "declared_exception"),
            ("meds.med_category", "sodium chloride", {}, "compliant"),
            ("meds_intermittent.med_category", "magnesium", {}, "aliased"),
            ("meds_intermittent.med_category", "dextrose", {"med_name": "Dextrose 50%"},
             "aliased"),
            ("meds_intermittent.med_category", "insulin", {}, "compliant"),   # extension
            ("meds_intermittent.med_category", "adenosine", {}, "declared_exception"),
            ("adt.location_type", "cvicu_icu", {}, "aliased"),
            ("ecmo.mcs_group", "LVAD", {"device_category": "HeartMate"}, "aliased"),
            ("ecmo.mcs_group", "LVAD", {"device_name": "2.5 / CP"}, "aliased"),
            ("ecmo.mcs_group", "LVAD", {}, "declared_exception"),
            ("ecmo.mcs_group", "ECMO", {}, "compliant"),
            ("ecmo.device_category", "HeartMate", {}, "aliased"),
            ("ecmo.device_category", "Other", {"device_name": "VV"}, "aliased"),
            ("ecmo.device_category", "Other", {}, "declared_exception"),
        ]
        for key, raw, row, expected in cases:
            with self.subTest(key=key, raw=raw, row=row):
                self.assertEqual(self.status(key, raw, **row), expected)
        # Every declared exception carries its reason.
        for key, values in self.decl["declared_exceptions"].items():
            for value, note in values.items():
                self.assertGreater(len(note), 10, (key, value))

    def test_dextrose_duplicates_and_bp_methods_are_declared(self):
        tables = {r["table"] for r in self.decl["exact_duplicates"]}
        self.assertEqual(tables, {"meds", "meds_intermittent"})
        bp = self.decl["bp_method"]
        self.assertEqual(bp["source_column"], "vital_name")
        from src.data.clif_conformance import bp_method_of
        for name, method in [
            ("Non Invasive Blood Pressure systolic", "noninvasive_auto"),
            ("Non Invasive Blood Pressure mean", "noninvasive_auto"),
            ("Arterial Blood Pressure diastolic", "arterial"),
            ("ART BP Mean", "arterial"),
            ("Manual Blood Pressure Systolic Left", "noninvasive_manual"),
            ("Manual Blood Pressure Diastolic Right", "noninvasive_manual"),
        ]:
            self.assertEqual(bp_method_of(name, bp), method)
        self.assertIsNone(bp_method_of("Some Other BP", bp))
        self.assertIsNone(bp_method_of(None, bp))       # null is not mapped for MIMIC

    def test_harmonization_record_carries_snapshot_and_reference_site(self):
        record = harmonization_record(self.cfg, "mimic")
        self.assertEqual(record["reference_site"], "mimic")
        self.assertEqual(record["global"]["mcide"]["version"], "2.1.1")
        self.assertEqual(record["global"]["mcide"]["snapshot_sha256"], self.snap["sha256"])
        self.assertIn("bp_method", record["site"])
        self.assertEqual(record["global"]["weight_plausible_kg"],
                         {"meds": [25.0, 400.0], "meds_intermittent": [25.0, 400.0]})
        self.assertEqual(record["global"]["dose_floors"]["propofol_mg"], 10.0)
        json.dumps(record)    # JSON-serialisable (hashed into vocab.json)


class SqlMappingTest(unittest.TestCase):
    def test_conditional_rules_and_null_key(self):
        rules = site_harmonization(repo_cfg(), "mimic")["category_map"]["ecmo.mcs_group"]
        frame = pl.DataFrame({
            "mcs_group": ["LVAD", "LVAD", "LVAD", None, None, "ECMO"],
            "device_category": ["HeartMate", "Other", "Other", "Other", "Other", "Other"],
            "device_name": ["HM II", "2.5 / CP", None, "5.5", None, "VV"],
        })
        c = con()
        c.register("t", frame.to_arrow())
        got = [r[0] for r in c.execute(f"SELECT {mapped_sql('mcs_group', rules)} FROM t")
               .fetchall()]
        self.assertEqual(got, ["durable_LVAD", "temporary_LVAD", "LVAD", "temporary_LVAD",
                               None, "ECMO"])


class GcsTest(unittest.TestCase):
    """The staged-MIMIC pattern: intubated verbal = 0 and gcs_total forced to 15."""

    def events(self):
        t1, t2 = T0, T0 + timedelta(hours=1)
        rows = [
            ("SYN-H-1", t1, "gcs_verbal", 0.0, None),     # not testable
            ("SYN-H-1", t1, "gcs_eye", 1.0, None),
            ("SYN-H-1", t1, "gcs_motor", 1.0, None),
            ("SYN-H-1", t1, "gcs_total", 15.0, None),     # forced: dropped
            ("SYN-H-1", t2, "gcs_verbal", 4.0, None),     # tested
            ("SYN-H-1", t2, "gcs_eye", 4.0, None),
            ("SYN-H-1", t2, "gcs_motor", 6.0, None),
            ("SYN-H-1", t2, "gcs_total", 14.0, None),     # kept
            ("SYN-H-2", t1, "gcs_verbal", None, "T"),     # categorical not-testable code
            ("SYN-H-2", t1, "gcs_total", 15.0, None),
            ("SYN-H-2", t1, "rass", -4.0, None),
        ]
        return pl.DataFrame(rows, orient="row", schema={
            "hosp_id": pl.String, "dttm": UTC_DT, "concept": pl.String, "value": pl.Float64,
            "cat_value": pl.String}).with_columns(pl.lit(None, pl.String).alias("unit"))

    def test_not_testable_verbal_is_a_token_and_the_forced_total_is_imputed(self):
        rule = gcs_rule(repo_cfg())
        out, counts = apply_gcs_not_testable(self.events(), rule)
        # SYN-H-1 at t1: eye 1 + motor 1 = 2 -> verbal 1 -> total 3 (not the forced 15).
        # SYN-H-2 at t1: no eye/motor -> its total is dropped.
        self.assertEqual(counts, {"verbal_not_testable": 2, "total_imputed": 1,
                                  "total_dropped": 1})
        h1 = out.filter(pl.col("hosp_id") == "SYN-H-1")
        totals = h1.filter(pl.col("concept") == "gcs_total").sort("dttm")["value"].to_list()
        self.assertEqual(totals, [3.0, 14.0])
        markers = h1.filter(pl.col("concept") == "gcs_total_source")
        self.assertEqual(markers.select("dttm", "cat_value").rows(), [(T0, "imputed")])
        self.assertEqual(categorical_token("gcs_total_source", "imputed"),
                         "gcs_total_source=imputed")
        verbal = h1.filter((pl.col("concept") == "gcs_verbal") & (pl.col("dttm") == T0))
        self.assertIsNone(verbal["value"][0])
        self.assertEqual(categorical_token("gcs_verbal", verbal["cat_value"][0]),
                         "gcs_verbal=not_testable")
        self.assertEqual(out.filter((pl.col("hosp_id") == "SYN-H-2")
                                    & (pl.col("concept") == "gcs_total")).height, 0)
        self.assertIn(4.0, h1.filter(pl.col("concept") == "gcs_verbal")["value"].to_list())
        self.assertEqual(out.filter(pl.col("concept") == "rass").height, 1)

    def test_brennan_imputation_table(self):
        from src.data.tokenize import gcs_imputed_verbal
        table = gcs_rule(repo_cfg())["imputation"]
        expected = {2: 1, 3: 1, 6: 1, 7: 2, 8: 4, 9: 4, 10: 5}
        for em, verbal in expected.items():
            self.assertEqual(gcs_imputed_verbal(em, table), verbal)
        self.assertIsNone(gcs_imputed_verbal(11, table))

    def test_vocabulary_always_has_the_not_testable_token(self):
        tokens = harmonization_tokens(repo_cfg())
        self.assertIn("gcs_verbal=not_testable", tokens)
        self.assertIn("gcs_total_source=imputed", tokens)
        self.assertIn("tracheostomy=cat_1", tokens)
        self.assertIn("bp_method=unknown", tokens)

    def test_labs_fragment_notes_no_longer_assume_verbal_one(self):
        doc = yaml.safe_load((ROOT / "configs/literature_segments/labs.yaml").read_text())
        notes = doc["concepts"]["gcs_total"]["notes"]
        self.assertIn("not_testable", notes)
        self.assertIn("Brennan", notes)
        self.assertNotIn("sites chart them as verbal = 1 (or 'T')", notes)
        # Levels unchanged.
        self.assertEqual(doc["concepts"]["gcs_verbal"]["valid_range"], [1, 5])


class BpMethodTest(unittest.TestCase):
    def frame(self, names):
        rows = []
        for i, (concept, value, name) in enumerate(names):
            rows.append(("SYN-H-1", T0, concept, value, None, name))
        return pl.DataFrame(rows, orient="row", schema={
            "hosp_id": pl.String, "dttm": UTC_DT, "concept": pl.String,
            "value": pl.Float64, "cat_value": pl.String, "_bp_src": pl.String,
        }).with_columns(pl.lit(None, pl.String).alias("unit"))

    def setUp(self):
        self.cfg = repo_cfg()
        self.glob = bp_config(self.cfg)
        self.decl = site_harmonization(self.cfg, "mimic")["bp_method"]

    def test_same_time_arterial_and_cuff_readings_each_get_their_method(self):
        frame = self.frame([
            ("map", 62.0, "Arterial Blood Pressure mean"),
            ("sbp", 95.0, "Arterial Blood Pressure systolic"),
            ("map", 71.0, "Non Invasive Blood Pressure mean"),
            ("sbp", 108.0, "Non Invasive Blood Pressure systolic"),
            ("heart_rate", 90.0, None),
        ])
        out, counts = apply_bp_method(frame, self.glob, self.decl, "mimic")
        self.assertEqual(counts, {"arterial": 2, "noninvasive_auto": 2, "tokens": 2})
        # tokenize_site tags every row of the vitals read with its source table.
        ordered = _event_sort(out.with_columns(pl.lit("vitals").alias("source")))
        seq = [(c, v if v is not None else cat) for c, v, cat in
               ordered.select("concept", "value", "cat_value").iter_rows()]
        # The method token immediately precedes its own readings; both MAPs are kept with
        # their values unchanged (the MAP concept and its bins are untouched).
        self.assertEqual(seq, [
            ("bp_method", "arterial"), ("map", 62.0), ("sbp", 95.0),
            ("bp_method", "noninvasive_auto"), ("map", 71.0), ("sbp", 108.0),
            ("heart_rate", 90.0),
        ])
        self.assertNotIn("_okey", ordered.columns)

    def test_unmapped_name_fails_and_explicit_unknown_passes(self):
        frame = self.frame([("map", 70.0, "Mystery BP"), ("map", 72.0, None)])
        with self.assertRaisesRegex(QualificationError, "Mystery BP.*bp_method"):
            apply_bp_method(frame, self.glob, self.decl, "mimic")
        decl = {**self.decl, "patterns": [*self.decl["patterns"],
                                          {"pattern": "^Mystery", "method": "unknown"}],
                "null": "unknown"}
        out, counts = apply_bp_method(frame, self.glob, decl, "mimic")
        self.assertEqual(counts["unknown"], 2)

    def test_site_without_a_declaration_fails(self):
        frame = self.frame([("map", 70.0, None)]).drop("_bp_src")
        with self.assertRaisesRegex(QualificationError, "declares no BP measurement method"):
            apply_bp_method(frame, self.glob, None, "rush")

    def test_map_concept_bins_are_unchanged(self):
        # The MAP 65 decision edge comes from the physician CSV + forced edge, never from
        # the BP-method tokens.
        self.assertEqual(self.cfg["value_binning"]["forced_edges"]["map"], [65.0])
        self.assertEqual(self.glob["concepts"], ["sbp", "dbp", "map"])
        cohort = yaml.safe_load((ROOT / "configs/cohort.yaml").read_text())
        self.assertIsNone(cohort["outcome_measurement_filters"]["map_below_65_48h"]["bp_method"])
        self.assertNotIn("bp_method", cohort["outcomes"]["map_below_65_48h"])


class ReadSourceTest(unittest.TestCase):
    """Med aliases, the dextrose exact-duplicate rule, the plausible-weight filter and
    the tracheostomy flag, through the real table reader."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.base = Path(self._td.name)
        self.cfg = repo_cfg()
        self.decl = site_harmonization(self.cfg, "mimic")

    def tearDown(self):
        self._td.cleanup()

    def write_meds(self):
        rows = []
        for i in range(4):
            t = T0 + timedelta(hours=i)
            # One D5W administration charted twice (dextrose + dextrose_in_water_d5w).
            rows += [("SYN-H-1", "o1", t, "Dextrose 5%", "dextrose", 50.0, "mL/hour", "start"),
                     ("SYN-H-1", "o1", t, "Dextrose 5%", "dextrose_in_water_d5w", 50.0,
                      "mL/hour", "start")]
        rows += [("SYN-H-1", "o2", T0, "Dextrose 10%", "dextrose", 40.0, "mL/hour", "start"),
                 ("SYN-H-1", "o3", T0, "Dextrose 20%", "dextrose", 30.0, "mL/hour", "start"),
                 ("SYN-H-1", "o4", T0, "Albumin 5%", "albumin_infusion", 100.0, "mL/hour",
                  "start"),
                 ("SYN-H-1", "o5", T0, "Norepi", "norepinephrine", 8.0, "mcg/min", "start"),
                 ("SYN-H-2", "o6", T0, "Norepi", "norepinephrine", 8.0, "mcg/min", "start")]
        pl.DataFrame(rows, orient="row", schema={
            "hospitalization_id": pl.String, "med_order_id": pl.String, "admin_dttm": UTC_DT,
            "med_name": pl.String, "med_category": pl.String, "med_dose": pl.Float64,
            "med_dose_unit": pl.String, "mar_action_category": pl.String,
        }).write_parquet(self.base / "clif_medication_admin_continuous.parquet")
        # SYN-H-1 weighs 80 kg; SYN-H-2's only weight is a 1 kg charting error.
        pl.DataFrame([("SYN-H-1", T0 - timedelta(hours=1), "weight_kg", 80.0),
                      ("SYN-H-2", T0 - timedelta(hours=1), "weight_kg", 1.0)], orient="row",
                     schema={"hospitalization_id": pl.String, "recorded_dttm": UTC_DT,
                             "vital_category": pl.String, "vital_value": pl.Float64}
                     ).write_parquet(self.base / "clif_vitals.parquet")

    def test_aliases_duplicates_and_weights(self):
        from src.data.tokenize import _dose_target_units
        self.write_meds()
        events, _, counts = _read_source(
            con(), self.base, self.cfg["tables"]["meds"], tables=self.cfg["tables"],
            target_units=_dose_target_units(self.cfg, None), harmonization=self.decl,
            name="meds")
        concepts = events["concept"].to_list()
        self.assertEqual(counts["exact_duplicates_removed"], 4)
        self.assertEqual(concepts.count("dextrose_5_water_ml_hr"), 4)   # one per administration
        self.assertIn("dextrose_10_water_ml_hr", concepts)
        self.assertIn("dextrose_other_ml_hr", concepts)
        self.assertIn("albumin_ml_hr", concepts)
        self.assertFalse(any(c.startswith("dextrose_in_water") for c in concepts))
        self.assertEqual(counts["weights"], 2)
        self.assertEqual(counts["weights_excluded"], 1)
        # SYN-H-1 converts per kg; SYN-H-2 (implausible weight only) keeps its native unit.
        h2 = events.filter(pl.col("hosp_id") == "SYN-H-2")["concept"].to_list()
        self.assertEqual(h2, ["norepinephrine_mcg_min"])
        h1 = events.filter((pl.col("hosp_id") == "SYN-H-1")
                           & pl.col("concept").str.starts_with("norepinephrine"))
        self.assertEqual(h1["concept"].to_list(), ["norepinephrine_mcg_kg_min"])
        self.assertAlmostEqual(h1["value"][0], 0.1)

    def test_ketamine_mg_per_kg_and_dose_floors(self):
        from src.data.tokenize import apply_dose_floors, dose_corrections
        rows = [("SYN-H-1", T0 + timedelta(hours=h), "ketamine", dose, unit, "given")
                for h, dose, unit in ((1, 0.2, "mg"), (2, 50.0, "mg"), (3, 0.05, "mg"),
                                      (4, 0.2, "mcg"), (5, 0.4, "mg"))]
        rows.append(("SYN-H-1", T0 + timedelta(hours=6), "fentanyl", 10.0, "mcg", "given"))
        rows.append(("SYN-H-1", T0 + timedelta(hours=7), "fentanyl", 50.0, "mcg", "given"))
        pl.DataFrame(rows, orient="row", schema={
            "hospitalization_id": pl.String, "admin_dttm": UTC_DT, "med_category": pl.String,
            "med_dose": pl.Float64, "med_dose_unit": pl.String,
            "mar_action_category": pl.String,
        }).with_columns(pl.lit("SYN item").alias("med_name"), pl.lit("o").alias("med_order_id")
                        ).write_parquet(self.base / "clif_medication_admin_intermittent.parquet")
        pl.DataFrame([("SYN-H-1", T0, "weight_kg", 80.0)], orient="row",
                     schema={"hospitalization_id": pl.String, "recorded_dttm": UTC_DT,
                             "vital_category": pl.String, "vital_value": pl.Float64}
                     ).write_parquet(self.base / "clif_vitals.parquet")
        events, _, counts = _read_source(
            con(), self.base, self.cfg["tables"]["meds_intermittent"], tables=self.cfg["tables"],
            dose_corrections=dose_corrections(self.cfg, "mimic", "meds_intermittent"),
            harmonization=self.decl, name="meds_intermittent")
        got = sorted(events.select("concept", "value").rows(), key=lambda r: (r[0], r[1]))
        self.assertEqual(got, [("fentanyl_mg", 0.01), ("fentanyl_mg", 0.05),
                               ("ketamine_mcg", 0.2), ("ketamine_mg", 0.05),
                               ("ketamine_mg", 0.4), ("ketamine_mg", 16.0),
                               ("ketamine_mg", 50.0)])
        self.assertEqual(counts["corrected"], 1)       # only 0.2 "mg" is in [0.1, 0.35]
        self.assertEqual(counts["quarantined"], 1)
        kept, below = apply_dose_floors(events, self.cfg)
        # Floors: ketamine 7 mg, fentanyl 0.025 mg -> 0.05 and 0.4 mg ketamine and the
        # 10 mcg fentanyl are charting errors, removed and counted.
        self.assertEqual(below, {"fentanyl_mg": 1, "ketamine_mg": 2})
        self.assertEqual(kept.height, events.height - 3)
        self.assertNotIn(0.05, kept.filter(pl.col("concept") == "ketamine_mg")["value"].to_list())

    def test_literature_fragments_follow_the_renamed_concepts(self):
        doc = yaml.safe_load((ROOT / "configs/literature_segments/medications.yaml")
                             .read_text())["concepts"]
        for new in ("dextrose_5_water_ml_hr", "dextrose_10_water_ml_hr", "dextrose_5_water_ml",
                    "dextrose_other_ml", "dextrose_other_g_min", "albumin_ml_hr",
                    "aminocaproic_acid_g_hr", "magnesium_sulfate_mg", "magnesium_sulfate_ml",
                    "magnesium_sulfate_g_hr"):
            self.assertIn(new, doc)
        for old in ("dextrose_ml_hr", "dextrose_in_water_d5w_ml_hr", "dextrose_ml",
                    "dextrose_g_min", "albumin_infusion_ml_hr", "aminocaproic_g_hr",
                    "magnesium_mg", "magnesium_ml", "magnesium_g_hr"):
            self.assertNotIn(old, doc)
        self.assertEqual(doc["magnesium_sulfate_mg"]["edges"], [2000.0, 4000.0])

    def test_boolean_and_integer_tracheostomy_give_the_same_tokens(self):
        frames = {}
        for kind, values, dtype in (("bool", [True, False], pl.Boolean),
                                    ("int", [1, 0], pl.Int32)):
            base = self.base / kind
            base.mkdir()
            pl.DataFrame({"hospitalization_id": ["SYN-H-1", "SYN-H-1"],
                          "recorded_dttm": [T0, T0 + timedelta(hours=1)],
                          "device_category": ["IMV", "IMV"], "tracheostomy": values},
                         schema_overrides={"recorded_dttm": UTC_DT, "tracheostomy": dtype}
                         ).write_parquet(base / "clif_respiratory_support.parquet")
            events, _, _ = _read_source(con(), base, self.cfg["tables"]["resp_support"],
                                        harmonization=self.decl, name="resp_support")
            trach = events.filter(pl.col("concept") == "tracheostomy").sort("dttm")
            frames[kind] = [categorical_token(c, v) for c, v in
                            trach.select("concept", "cat_value").iter_rows()]
        self.assertEqual(frames["bool"], frames["int"])
        self.assertEqual(frames["int"], ["tracheostomy=cat_1", "tracheostomy=cat_0"])

    def test_ecmo_groups_map_to_the_clifpy_vocabulary(self):
        pl.DataFrame({
            "hospitalization_id": ["SYN-H-1"] * 3, "recorded_dttm": [T0] * 3,
            "device_name": ["HM II", "VV", None], "device_category": ["HeartMate", "Other", "Other"],
            "mcs_group": ["LVAD", "ECMO", None], "device_rate": [9000.0, 3000.0, 2000.0],
            "flow": [5.0, 4.0, 3.0], "sweep": [None, 4.0, None], "fdO2": [None, 100.0, None],
        }, schema_overrides={"recorded_dttm": UTC_DT}).write_parquet(
            self.base / "clif_ecmo_mcs.parquet")
        events, _, _ = _read_source(con(), self.base, self.cfg["tables"]["ecmo"],
                                    harmonization=self.decl, name="ecmo")
        concepts = set(events["concept"])
        self.assertTrue({"durable_lvad_flow", "ecmo_flow", "unknown_flow"} <= concepts)
        self.assertNotIn("lvad_flow", concepts)
        cats = {categorical_token(c, v) for c, v in events.filter(
            pl.col("cat_value").is_not_null()).select("concept", "cat_value").iter_rows()}
        self.assertIn("durable_lvad_device_category=hmii", cats)
        self.assertIn("ecmo_device_category=vv_ecmo", cats)
        # Literature fragments still resolve for the concepts that kept their names.
        doc = yaml.safe_load((ROOT / "configs/literature_segments/support.yaml").read_text())
        for concept in ("ecmo_flow", "lvad_flow", "rvad_flow", "unknown_flow"):
            self.assertIn(concept, doc["concepts"])


class UnitAndCoverageTest(unittest.TestCase):
    def test_monovalent_ions_and_ph_spellings_are_equivalent(self):
        cfg = repo_cfg()
        for concept in ("sodium", "potassium", "chloride", "bicarbonate"):
            self.assertTrue(units_equivalent(concept, "mmol/L", "mEq/L", cfg))
        self.assertTrue(units_equivalent("ph_arterial", "(no units)", "units", cfg))
        # Concept-scoped: mEq/L is not mmol/L for a divalent ion.
        self.assertFalse(units_equivalent("magnesium", "mmol/L", "mEq/L", cfg))
        self.assertFalse(units_equivalent("lactate", "mmol/L", "mg/dL", cfg))
        from src.data.tokenize import unit_mismatches
        events = pl.DataFrame({"concept": ["sodium", "ph_venous", "magnesium"],
                               "unit": ["mmol/L", "(no units)", "mmol/L"]})
        reference = {"concepts": {"sodium": "mEq/L", "ph_venous": "units",
                                  "magnesium": "mEq/L"}}
        self.assertEqual(unit_mismatches(events, cfg, reference),
                         ["magnesium: expected 'mEq/L', found 'mmol/L'"])

    def test_csv_concepts_without_reference_rows_are_in_the_vocabulary(self):
        cfg = repo_cfg()
        bin_cfg = {**cfg["value_binning"], "literature_source": None}
        fit = pl.DataFrame({"hosp_id": ["SYN-H-1"] * 30, "concept": ["map"] * 30,
                            "value": [60.0 + i for i in range(30)], "source": ["vitals"] * 30,
                            "cat_value": [None] * 30}, schema_overrides={"cat_value": pl.String})
        targets = [t["name"] for t in cfg["target_concepts"]]
        segs, sources = build_segments(bin_cfg, fit, targets, tables=cfg["tables"])
        vocab = build_vocab(fit, segs)
        for concept in ("glucose_fingerstick", "neutrophils_absolute", "procalcitonin",
                        "troponin_i", "remifentanil_mcg_kg_min"):
            self.assertEqual(sources[concept], "csv")
            self.assertIn(f"{concept}=0", vocab)
        # A CSV medication absent from the fit still has the [0, 0] stop bin.
        first = segs["remifentanil_mcg_kg_min"][0]
        self.assertEqual((first["lo"], first["hi"]), (0.0, 0.0))
        # The old rule (targets_only) leaves them out.
        old, _ = build_segments({**bin_cfg, "csv_coverage": "targets_only"}, fit, targets,
                                tables=cfg["tables"])
        self.assertNotIn("procalcitonin", old)


class VocabularyBindingTest(unittest.TestCase):
    def test_global_rules_bind_the_vocabulary(self):
        from src.data.segments import json_sha256
        cfg = repo_cfg()
        record = json.loads(json.dumps(harmonization_record(cfg, "mimic")))
        blob = {"harmonization": record}
        _check_harmonization_binding(blob, {"harmonization": json_sha256(record)}, cfg)
        with self.assertRaisesRegex(QualificationError, "predates"):
            _check_harmonization_binding({}, {}, cfg)
        with self.assertRaisesRegex(QualificationError, "hash mismatch"):
            _check_harmonization_binding(blob, {"harmonization": "0" * 64}, cfg)
        changed = copy.deepcopy(cfg)
        changed["gcs_not_testable"]["numeric_codes"] = [0, 1]
        with self.assertRaisesRegex(QualificationError, "different CLIF harmonization"):
            _check_harmonization_binding(blob, {"harmonization": json_sha256(record)}, changed)
        # A config without any rule accepts a vocabulary without the record.
        bare = {k: v for k, v in cfg.items() if k not in (
            "clif_conformance", "gcs_not_testable", "bp_method")}
        bare["tables"] = {k: {kk: vv for kk, vv in v.items() if kk != "flag_cols"}
                          for k, v in cfg["tables"].items()}
        for table in ("meds", "meds_intermittent"):
            bare["tables"][table]["dose"] = {**cfg["tables"][table]["dose"],
                                             "weight_source": {"table": "vitals",
                                                               "concept": "weight_kg"}}
        bare["dose_plausibility"] = {k: v for k, v in cfg["dose_plausibility"].items()
                                     if k != "floors"}
        bare["unit_normalization"] = {k: v for k, v in cfg["unit_normalization"].items()
                                      if k != "equivalent_units"}
        bare["value_binning"] = {k: v for k, v in cfg["value_binning"].items()
                                 if k not in ("csv_coverage", "literature_coverage")}
        bare.pop("derived_concepts")
        bare.pop("vocabulary_allowlist")
        self.assertEqual(global_record(bare), {})
        _check_harmonization_binding({}, {}, bare)


class Clif21UnitsTest(unittest.TestCase):
    """Every literature edge is in the CLIF 2.1 unit of its concept; derived / non-CLIF
    concepts are listed explicitly."""

    # Concepts with no CLIF 2.1 unit: assessment scores (points), the age band token
    # (hospitalization.age_at_admission, years) and derived concepts.
    NON_CLIF = {"gcs_total", "gcs_eye", "gcs_verbal", "gcs_motor", "rass", "braden_total",
                "braden_sensory", "braden_moisture", "braden_mobility", "braden_activity",
                "braden_nutrition", "braden_friction", "age_decile",
                "crrt_effluent_dose_ml_kg_h"}

    def clif_unit(self, concept, cfg, snapshot):
        from src.data.units import unit_suffix
        if concept in snapshot["units"]:
            return snapshot["units"][concept]
        columns = cfg["column_units"]
        for key, spec in columns.items():
            table, column = key.split(".")
            qualified = cfg["tables"][table].get("concept_qualifier_col")
            if concept == column.lower() or (qualified and concept.endswith(f"_{column.lower()}")):
                return spec["unit"]
        doc = yaml.safe_load((ROOT / "configs/literature_segments/medications.yaml")
                             .read_text())["concepts"]
        if concept in doc:
            # CLIF med_dose_unit convention: the concept's own unit suffix (units.py).
            unit = doc[concept]["unit"]
            self.assertTrue(concept.endswith("_" + unit_suffix(unit)), (concept, unit))
            return unit
        return None

    def test_every_literature_fragment_is_in_the_clif_21_unit(self):
        from src.data.tokenize import _unit_text, load_literature_segments
        cfg = repo_cfg()
        snapshot = load_snapshot(cfg["clif_conformance"]["snapshot"])
        loaded = load_literature_segments(cfg["value_binning"]["literature_source"])
        unchecked = []
        for concept, spec in loaded["concepts"].items():
            if concept in self.NON_CLIF:
                continue
            expected = self.clif_unit(concept, cfg, snapshot)
            if expected is None:
                unchecked.append(concept)
                continue
            with self.subTest(concept=concept):
                self.assertTrue(units_equivalent(concept, _unit_text(spec["unit"]),
                                                 _unit_text(expected), cfg),
                                f"{concept}: fragment {spec['unit']!r} vs CLIF {expected!r}")
        self.assertEqual(unchecked, [])

    def test_derived_concepts_are_declared_non_clif(self):
        cfg = repo_cfg()
        from src.data.tokenize import derived_concept_specs
        specs = derived_concept_specs(cfg)
        self.assertEqual(set(specs), {"crrt_effluent_dose_ml_kg_h"})
        self.assertFalse(specs["crrt_effluent_dose_ml_kg_h"]["clif_concept"])
        self.assertIn("crrt_effluent_dose_ml_kg_h", self.NON_CLIF)

    def test_reference_units_are_recorded_in_clif_spelling(self):
        from src.data.tokenize import reference_units
        cfg = repo_cfg()
        fit = pl.DataFrame({"concept": ["sodium", "ph_arterial", "platelet_count", "calcium_ionized"],
                            "unit": ["mEq/L", "units", "K/uL", "mmol/L"],
                            "source": ["labs"] * 4})
        segs = {c: [] for c in fit["concept"]}
        record = reference_units(fit, segs, cfg, {}, "mimic")
        self.assertEqual(record["concepts"]["sodium"], "mmol/L")
        self.assertEqual(record["concepts"]["ph_arterial"], "(no units)")
        self.assertEqual(record["concepts"]["platelet_count"], "10^3/µL")
        # Not equivalent (mmol/L calcium vs CLIF mg/dL): kept and listed as a deviation.
        self.assertEqual(record["clif_unit_deviations"], {"calcium_ionized": "mmol/L"})


class DerivedAndRenamedConceptsTest(unittest.TestCase):
    def test_effluent_dose_needs_every_component_and_a_weight(self):
        from src.data.tokenize import derive_concepts
        cfg = with_lag0(repo_cfg())
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            pl.DataFrame([("SYN-H-1", T0 - timedelta(hours=1), "weight_kg", 80.0),
                          ("SYN-H-3", T0 - timedelta(hours=1), "weight_kg", 2.0)], orient="row",
                         schema={"hospitalization_id": pl.String, "recorded_dttm": UTC_DT,
                                 "vital_category": pl.String, "vital_value": pl.Float64}
                         ).write_parquet(base / "clif_vitals.parquet")
            parts = ["dialysate_flow_rate", "pre_filter_replacement_fluid_rate",
                     "post_filter_replacement_fluid_rate", "ultrafiltration_out"]
            rows = [("SYN-H-1", T0, c, v) for c, v in zip(parts, [1000.0, 600.0, 200.0, 200.0])]
            rows += [("SYN-H-2", T0, c, 500.0) for c in parts[:3]]          # no net UF
            rows += [("SYN-H-3", T0, c, 500.0) for c in parts]              # implausible weight
            events = pl.DataFrame(rows, orient="row", schema={
                "hosp_id": pl.String, "dttm": UTC_DT, "concept": pl.String,
                "value": pl.Float64}).with_columns(pl.lit(None, pl.String).alias("unit"),
                                                   pl.lit(None, pl.String).alias("cat_value"))
            out, counts = derive_concepts(events, "crrt", cfg, con(), base, None)
        dose = out.filter(pl.col("concept") == "crrt_effluent_dose_ml_kg_h")
        self.assertEqual(dose.select("hosp_id", "value").rows(), [("SYN-H-1", 25.0)])
        self.assertEqual(dose["unit"].to_list(), ["mL/kg/hr"])
        self.assertEqual(counts["crrt_effluent_dose_ml_kg_h"],
                         {"rows_with_any_component": 3, "rows_complete": 2, "emitted": 1,
                          "no_weight": 1})

    def test_conventional_troponin_is_its_own_concept_at_mimic(self):
        from src.data.tokenize import apply_concept_renames
        decl = site_harmonization(repo_cfg(), "mimic")
        events = pl.DataFrame({"concept": ["troponin_t", "lactate"], "value": [20.0, 2.0]})
        out = apply_concept_renames(events, decl["concept_renames"]["labs"])
        self.assertEqual(out["concept"].to_list(), ["troponin_t_conventional", "lactate"])
        doc = yaml.safe_load((ROOT / "configs/literature_segments/labs.yaml").read_text())
        self.assertEqual(doc["concepts"]["troponin_t"]["edges"], [5.0, 15.0, 52.0, 140.0])
        self.assertNotIn("troponin_t_conventional", doc["concepts"])   # data-driven bins
        self.assertEqual(doc["concepts"]["so2_mixed_venous"]["edges"], [60.0, 65.0, 75.0])
        support = yaml.safe_load((ROOT / "configs/literature_segments/support.yaml")
                                 .read_text())["concepts"]
        self.assertEqual(support["rvad_flow"]["decision"], "keep_quantile")
        self.assertEqual(support["crrt_effluent_dose_ml_kg_h"]["edges"],
                         [13.0, 20.0, 25.0, 35.0])


def with_lag0(cfg):
    from src.data.tokenize import with_availability_lag
    return with_availability_lag(cfg, 0)


if __name__ == "__main__":
    unittest.main()
