"""Explicit unit repair for unit-less wide-table columns and medication doses.

CRRT and ECMO/MCS charting carries no unit column, so CLIF 2.1's canonical unit and
plausible range per column are declared in the data config (`column_units`). A site
whose values sit on another scale declares the conversion explicitly
(`site_unit_conversions.<site>`); nothing is auto-detected. After conversion a column
with more than `max_out_of_range_share` of its values outside the plausible range is
refused, with a scale hint. Medication doses get the same treatment per converted concept
(`dose_plausibility`), with explicit per-site corrections scoped to a med_category and
its charted unit.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import duckdb
import polars as pl
import yaml

from src.data.cohort import QualificationError

UTC = pl.Datetime("us", "UTC")
CRRT_COLS = ("blood_flow_rate", "pre_filter_replacement_fluid_rate",
             "post_filter_replacement_fluid_rate", "dialysate_flow_rate",
             "ultrafiltration_out")
COLUMN_UNITS = {
    "crrt.blood_flow_rate": {"unit": "mL/min", "range": [10, 500]},
    "crrt.ultrafiltration_out": {
        "unit": "mL/hr", "range": [0, 500], "on_out_of_range": "report",
        "flag": "median above typical net ultrafiltration; may be total effluent"},
    "ecmo.flow": {"unit": "L/min", "range": [0, 10]},
    "ecmo.fdO2": {"unit": "fraction", "range": [0.21, 1.0]},
}
MLH_TO_MLMIN = {"from": "mL/hr", "to": "mL/min", "factor": 1 / 60}
PCT_TO_FRACTION = {"from": "percent", "to": "fraction", "factor": 0.01}


def _write_site(work: Path, name: str, *, blood_flow: list[float], fdo2: list[float],
                uf: float = 350.0) -> tuple[Path, pl.DataFrame]:
    from src.eval.synthetic_bundle import build_synthetic_site

    site = work / name
    episodes = pl.read_parquet(build_synthetic_site(site))
    eps = list(episodes.iter_rows(named=True))
    crrt, ecmo = [], []
    for i, ep in enumerate(eps):
        t = ep["icu_admit_dttm"] + timedelta(hours=2)
        crrt.append({"hospitalization_id": ep["hospitalization_id"], "recorded_dttm": t,
                     "crrt_mode_category": "cvvhdf",
                     "blood_flow_rate": blood_flow[i % len(blood_flow)],
                     "pre_filter_replacement_fluid_rate": 1500.0,
                     "post_filter_replacement_fluid_rate": 200.0,
                     "dialysate_flow_rate": 800.0, "ultrafiltration_out": uf})
        ecmo.append({"hospitalization_id": ep["hospitalization_id"], "recorded_dttm": t,
                     "device_category": "Other", "mcs_group": "ECMO", "device_rate": 3500.0,
                     "flow": 4.5, "sweep": 3.0, "fdO2": fdo2[i % len(fdo2)]})
    pl.DataFrame(crrt, schema={"hospitalization_id": pl.String, "recorded_dttm": UTC,
                               "crrt_mode_category": pl.String,
                               **{c: pl.Float32 for c in CRRT_COLS}}
                 ).write_parquet(site / "clif_crrt_therapy.parquet")
    pl.DataFrame(ecmo, schema={"hospitalization_id": pl.String, "recorded_dttm": UTC,
                               "device_category": pl.String, "mcs_group": pl.String,
                               "device_rate": pl.Float32, "flow": pl.Float32,
                               "sweep": pl.Float32, "fdO2": pl.Float32}
                 ).write_parquet(site / "clif_ecmo_mcs.parquet")
    return site, episodes


def _cfg(work: Path, conversions: dict | None = None) -> dict:
    from src.data.tokenize import ROOT
    from src.eval.synthetic_bundle import FIXTURE_COHORT, FIXTURE_DATA_CONFIG, FIXTURE_POLICY

    (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
    (work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))
    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
    cfg["cohort_contract"] = str((work / "cohort.yaml").resolve())
    cfg["artifact_policy"] = str((work / "artifact_policy.yaml").resolve())
    cfg["tables"]["crrt"] = copy.deepcopy(data_cfg["tables"]["crrt"])
    cfg["tables"]["ecmo"] = copy.deepcopy(data_cfg["tables"]["ecmo"])
    cfg["value_binning"]["n_bins"] = 4
    cfg["unit_normalization"]["max_out_of_range_share"] = 0.01
    cfg["column_units"] = copy.deepcopy(COLUMN_UNITS)
    cfg["site_unit_conversions"] = copy.deepcopy(conversions or {})
    return cfg


def _tokenize(cfg, site_name, site, out, episodes, blob=None):
    from src.data.tokenize import tokenize_site
    from src.eval.synthetic_bundle import FIXTURE_POLICY

    tokenize_site(cfg, site_name, site, out, blob, episodes=episodes,
                  artifact_policy=FIXTURE_POLICY)
    return json.loads((out / "vocab.json").read_text()) if blob is None else blob


def _values(out: Path, blob: dict, concept: str) -> list[float]:
    """Values of `concept` in the written events, rounded to 6 places (the fixture
    stores float32, as CLIF parquet does)."""
    inv = {i: t for t, i in blob["vocab"].items()}
    events = pl.read_parquet(out / "events.parquet")
    vals = []
    for tokens, values in events.select("token", "value").iter_rows():
        vals += [round(v, 6) for t, v in zip(tokens, values)
                 if inv[t].split("=")[0] == concept]
    return vals


class _Work(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.work = Path(self._td.name)
        self._old = os.getcwd()
        os.chdir(self.work)

    def tearDown(self):
        os.chdir(self._old)
        self._td.cleanup()


class ColumnUnitRepairTest(_Work):
    def test_mlh_blood_flow_without_a_declaration_is_refused_with_the_scale_hint(self):
        from src.eval.synthetic_bundle import SYNTHETIC_SITE

        site, episodes = _write_site(self.work, "s", blood_flow=[7200.0, 12000.0, 15000.0],
                                     fdo2=[0.21, 0.5, 1.0])
        with self.assertRaises(QualificationError) as ctx:
            _tokenize(_cfg(self.work), SYNTHETIC_SITE, site, Path("output/intermediate_phi/x"), episodes)
        message = str(ctx.exception)
        self.assertIn("crrt.blood_flow_rate", message)
        self.assertIn(SYNTHETIC_SITE, message)
        self.assertIn("1/60", message)
        self.assertIn("site_unit_conversions", message)
        self.assertNotIn("synth-0", message)          # aggregate only, no identifiers

    def test_declared_conversion_bins_blood_flow_as_ml_min_and_fdo2_as_a_fraction(self):
        from src.eval.synthetic_bundle import SYNTHETIC_SITE

        site, episodes = _write_site(self.work, "s", blood_flow=[7200.0, 12000.0, 15000.0],
                                     fdo2=[21.0, 50.0, 100.0])
        conversions = {SYNTHETIC_SITE: {"crrt.blood_flow_rate": MLH_TO_MLMIN,
                                        "ecmo.fdO2": PCT_TO_FRACTION}}
        out = Path("output/intermediate_phi/conv")
        blob = _tokenize(_cfg(self.work, conversions), SYNTHETIC_SITE, site, out, episodes)
        self.assertEqual(sorted(set(_values(out, blob, "blood_flow_rate"))),
                         [120.0, 200.0, 250.0])
        self.assertEqual(sorted(set(_values(out, blob, "ecmo_fdo2"))), [0.21, 0.5, 1.0])
        units = blob["reference_units"]
        self.assertEqual(units["concepts"]["blood_flow_rate"], "mL/min")
        self.assertEqual(units["concepts"]["ecmo_fdo2"], "fraction")
        # A column with no declared canonical unit keeps its device-metric "unit".
        self.assertEqual(units["concepts"]["ecmo_device_rate"], "device_metric:device_rate")

    def test_a_conditional_conversion_leaves_values_already_canonical(self):
        from src.eval.synthetic_bundle import SYNTHETIC_SITE

        # Mostly percent, some rows already fractions: a blanket x0.01 would turn 0.5
        # into 0.005, below the range but under the refusal share.
        site, episodes = _write_site(self.work, "s", blood_flow=[120.0, 200.0],
                                     fdo2=[100.0, 50.0, 0.5, 21.0])
        when = {**PCT_TO_FRACTION, "when": {"gt": 1.0}}
        cfg = _cfg(self.work, {SYNTHETIC_SITE: {"ecmo.fdO2": when}})
        out = Path("output/intermediate_phi/when")
        blob = _tokenize(cfg, SYNTHETIC_SITE, site, out, episodes)
        self.assertEqual(sorted(set(_values(out, blob, "ecmo_fdo2"))), [0.21, 0.5, 1.0])
        recorded = blob["reference_units"]["site_conversions"]["ecmo.fdO2"]
        self.assertEqual(recorded["when"], {"gt": 1.0})
        # The condition is part of the binding.
        from src.data.tokenize import validate_vocabulary_artifact
        from src.eval.synthetic_bundle import FIXTURE_POLICY

        other = copy.deepcopy(cfg)
        del other["site_unit_conversions"][SYNTHETIC_SITE]["ecmo.fdO2"]["when"]
        with self.assertRaisesRegex(QualificationError, "unit conversions"):
            validate_vocabulary_artifact(copy.deepcopy(blob), other, FIXTURE_POLICY)
        # An unconditional x0.01 on the same data turns 0.5 into 0.005: refused.
        blanket = _cfg(self.work, {SYNTHETIC_SITE: {"ecmo.fdO2": PCT_TO_FRACTION}})
        with self.assertRaisesRegex(QualificationError, "ecmo.fdO2"):
            _tokenize(blanket, SYNTHETIC_SITE, site, Path("output/intermediate_phi/b"),
                      episodes)

    def test_a_site_already_in_canonical_units_needs_no_declaration(self):
        from src.eval.synthetic_bundle import SYNTHETIC_SITE

        site, episodes = _write_site(self.work, "s", blood_flow=[120.0, 200.0, 250.0],
                                     fdo2=[0.21, 0.5, 1.0])
        out = Path("output/intermediate_phi/canon")
        blob = _tokenize(_cfg(self.work), SYNTHETIC_SITE, site, out, episodes)
        self.assertEqual(sorted(set(_values(out, blob, "blood_flow_rate"))),
                         [120.0, 200.0, 250.0])
        self.assertEqual(blob["reference_units"]["site_conversions"], {})

    def test_conversions_are_recorded_and_hashed(self):
        from src.data.segments import json_sha256
        from src.data.tokenize import validate_vocabulary_artifact
        from src.eval.synthetic_bundle import FIXTURE_POLICY, SYNTHETIC_SITE

        site, episodes = _write_site(self.work, "s", blood_flow=[7200.0, 12000.0, 15000.0],
                                     fdo2=[21.0, 50.0, 100.0])
        conversions = {SYNTHETIC_SITE: {"crrt.blood_flow_rate": MLH_TO_MLMIN,
                                        "ecmo.fdO2": PCT_TO_FRACTION}}
        cfg = _cfg(self.work, conversions)
        blob = _tokenize(cfg, SYNTHETIC_SITE, site, Path("output/intermediate_phi/h"), episodes)
        units = blob["reference_units"]
        self.assertEqual(units["site_conversions"]["crrt.blood_flow_rate"]["to"], "mL/min")
        self.assertEqual(units["column_units"]["ecmo.fdO2"]["unit"], "fraction")
        self.assertEqual(blob["manifest"]["hashes"]["reference_units"], json_sha256(units))
        validate_vocabulary_artifact(copy.deepcopy(blob), cfg, FIXTURE_POLICY)
        # A config whose reference-site conversions differ does not bind the vocabulary.
        other = copy.deepcopy(cfg)
        other["site_unit_conversions"][SYNTHETIC_SITE]["ecmo.fdO2"]["factor"] = 0.1
        with self.assertRaisesRegex(QualificationError, "unit conversions"):
            validate_vocabulary_artifact(copy.deepcopy(blob), other, FIXTURE_POLICY)
        # Nor does one whose canonical column units differ.
        other = copy.deepcopy(cfg)
        other["column_units"]["crrt.blood_flow_rate"]["unit"] = "mL/hr"
        with self.assertRaisesRegex(QualificationError, "column units"):
            validate_vocabulary_artifact(copy.deepcopy(blob), other, FIXTURE_POLICY)

    def test_ultrafiltration_flag_appears_in_the_report(self):
        from src.eval.synthetic_bundle import SYNTHETIC_SITE

        site, episodes = _write_site(self.work, "s", blood_flow=[120.0, 200.0],
                                     fdo2=[0.5], uf=800.0)
        out = Path("output/intermediate_phi/uf")
        _tokenize(_cfg(self.work), SYNTHETIC_SITE, site, out, episodes)
        report = json.loads((out / "tokenization_report.json").read_text())
        quality = report["data_quality"]
        uf = quality["columns"]["crrt.ultrafiltration_out"]
        self.assertEqual(uf["status"], "reported")      # out of range, not refused
        self.assertIn("effluent", quality["semantic_flags"]["crrt.ultrafiltration_out"])
        self.assertEqual(quality["columns"]["crrt.blood_flow_rate"]["status"], "ok")
        self.assertIn("quantile", report["binning"]["by_source"])

    def test_per_site_declarations_do_not_leak_across_sites(self):
        from src.eval.synthetic_bundle import SYNTHETIC_SITE

        ref_site, episodes = _write_site(self.work, "ref",
                                         blood_flow=[7200.0, 12000.0, 15000.0],
                                         fdo2=[21.0, 50.0, 100.0])
        conversions = {SYNTHETIC_SITE: {"crrt.blood_flow_rate": MLH_TO_MLMIN,
                                        "ecmo.fdO2": PCT_TO_FRACTION}}
        cfg = _cfg(self.work, conversions)
        blob = _tokenize(cfg, SYNTHETIC_SITE, ref_site, Path("output/intermediate_phi/ref"), episodes)
        # A second site already in canonical units imports the vocabulary: the reference
        # site's declaration is not applied to it.
        canon, episodes_b = _write_site(self.work, "canon", blood_flow=[120.0, 200.0],
                                        fdo2=[0.21, 1.0])
        out = Path("output/intermediate_phi/canon_import")
        _tokenize(cfg, "other", canon, out, episodes_b, blob=blob)
        self.assertEqual(sorted(set(_values(out, blob, "blood_flow_rate"))), [120.0, 200.0])
        self.assertEqual(sorted(set(_values(out, blob, "ecmo_fdo2"))), [0.21, 1.0])
        # A second site in mL/h with no declaration of its own is refused.
        mlh, episodes_c = _write_site(self.work, "mlh", blood_flow=[7200.0, 12000.0],
                                      fdo2=[0.5])
        with self.assertRaisesRegex(QualificationError, "'other'.*crrt.blood_flow_rate"):
            _tokenize(cfg, "other", mlh, Path("output/intermediate_phi/mlh_import"), episodes_c, blob=blob)
        # ... and accepted once it declares its own conversion.
        cfg_c = copy.deepcopy(cfg)
        cfg_c["site_unit_conversions"]["other"] = {"crrt.blood_flow_rate": MLH_TO_MLMIN}
        out = Path("output/intermediate_phi/mlh_ok")
        _tokenize(cfg_c, "other", mlh, out, episodes_c, blob=blob)
        self.assertEqual(sorted(set(_values(out, blob, "blood_flow_rate"))), [120.0, 200.0])

    def test_malformed_declarations_fail_closed(self):
        from src.data.tokenize import site_unit_conversions

        cfg = _cfg(self.work)
        for bad in ({"crrt.nonexistent": MLH_TO_MLMIN},
                    {"crrt.blood_flow_rate": {**MLH_TO_MLMIN, "to": "L/min"}},
                    {"crrt.blood_flow_rate": {**MLH_TO_MLMIN, "factor": -1.0}},
                    {"crrt.blood_flow_rate": {"from": "mL/hr", "to": "mL/min"}},
                    {"crrt.blood_flow_rate": {**MLH_TO_MLMIN, "when": {"between": 1}}}):
            cfg["site_unit_conversions"] = {"s": bad}
            with self.subTest(bad=bad), self.assertRaises(QualificationError):
                site_unit_conversions(cfg, "s")


class DosePlausibilityTest(_Work):
    """Medication doses after conversion, per declared concept range."""

    def _site(self):
        site = self.work / "meds"
        site.mkdir()
        rows = ([("h1", "propofol", 10.0, "mcg")] * 30 + [("h1", "propofol", 50.0, "mg")] * 70
                + [("h1", "ketamine", 0.2, "mcg")] * 10 + [("h1", "ketamine", 50.0, "mg")] * 90)
        pl.DataFrame(
            [(h, m, v, u) for h, m, v, u in rows],
            schema={"hospitalization_id": pl.String, "med_category": pl.String,
                    "med_dose": pl.Float64, "med_dose_unit": pl.String}, orient="row",
        ).with_columns(pl.lit("2026-01-01T00:00:00Z").str.to_datetime(time_zone="UTC")
                       .alias("admin_dttm"),
                       pl.lit("given").alias("mar_action_category")
                       ).write_parquet(site / "clif_medication_admin_intermittent.parquet")
        return site

    def _cfg(self, conversions=None):
        from src.data.tokenize import ROOT

        data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
        return {"tables": {"meds_intermittent":
                           copy.deepcopy(data_cfg["tables"]["meds_intermittent"])},
                "unit_normalization": {"on_mismatch": "error"},
                "dose_plausibility": {"max_implausible_share": 0.05, "ranges": {
                    "propofol_mg": {"unit": "mg", "range": [1, 400]},
                    "ketamine_mg": {"unit": "mg", "range": [1, 500]}}},
                "site_unit_conversions": conversions or {}}

    def _read(self, cfg, site_name="s"):
        from src.data.tokenize import check_dose_plausibility, dose_corrections, _read_source

        con = duckdb.connect()
        con.execute("SET TimeZone = 'UTC'")
        spec = cfg["tables"]["meds_intermittent"]
        events, _, counts = _read_source(
            con, self._base, spec, None, tables=cfg["tables"], target_units={},
            dose_corrections=dose_corrections(cfg, site_name, "meds_intermittent"))
        quality = check_dose_plausibility(
            {"meds_intermittent": events.with_columns(source=pl.lit("meds_intermittent"))},
            cfg, site_name)
        return events, counts, quality

    def test_a_mislabeled_unit_beyond_the_share_is_refused_without_a_declaration(self):
        self._base = self._site()
        with self.assertRaisesRegex(QualificationError, r"site 's' dose propofol_mg"):
            self._read(self._cfg())

    def test_a_declared_correction_relabels_the_charted_unit(self):
        self._base = self._site()
        cfg = self._cfg({"s": {
            "meds_intermittent.propofol[mcg]": {"from": "mcg", "to": "mg", "factor": 1.0,
                                                "note": "mg charted as mcg"}}})
        cfg["dose_plausibility"]["max_implausible_share"] = 0.2
        events, counts, quality = self._read(cfg)
        propofol = events.filter(pl.col("concept") == "propofol_mg")["value"]
        self.assertEqual(sorted(set(propofol.to_list())), [10.0, 50.0])
        self.assertEqual(quality["doses"]["propofol_mg"]["implausible"], 0)
        self.assertEqual(counts["corrected"], 30)
        # Ketamine 0.2 mcg -> 0.0002 mg: counted as implausible (10 of 100).
        self.assertEqual(quality["doses"]["ketamine_mg"]["implausible"], 10)

    def test_a_small_implausible_share_is_reported_not_refused(self):
        self._base = self._site()
        cfg = self._cfg({"s": {
            "meds_intermittent.propofol[mcg]": {"from": "mcg", "to": "mg", "factor": 1.0}}})
        cfg["dose_plausibility"]["max_implausible_share"] = 0.2
        _, _, quality = self._read(cfg)
        self.assertEqual(quality["doses"]["ketamine_mg"]["status"], "reported")

    def test_a_report_only_range_is_never_refused(self):
        self._base = self._site()
        cfg = self._cfg({"s": {
            "meds_intermittent.propofol[mcg]": {"from": "mcg", "to": "mg", "factor": 1.0}}})
        cfg["dose_plausibility"]["max_implausible_share"] = 0.0
        cfg["dose_plausibility"]["ranges"]["ketamine_mg"]["on_implausible"] = "report"
        _, _, quality = self._read(cfg)
        self.assertEqual(quality["doses"]["ketamine_mg"]["status"], "reported")
        self.assertEqual(quality["doses"]["ketamine_mg"]["implausible"], 10)

    def test_quarantine_keeps_unknown_unit_rows_in_their_native_concept(self):
        self._base = self._site()
        cfg = self._cfg({"s": {
            "meds_intermittent.propofol[mcg]": {"from": "mcg", "to": "mg", "factor": 1.0},
            "meds_intermittent.ketamine[mcg]": {"from": "mcg", "action": "quarantine"}}})
        events, counts, quality = self._read(cfg)
        self.assertEqual(events.filter(pl.col("concept") == "ketamine_mcg").height, 10)
        self.assertEqual(events.filter(pl.col("concept") == "ketamine_mg").height, 90)
        self.assertEqual(counts["quarantined"], 10)
        self.assertEqual(quality["doses"]["ketamine_mg"]["implausible"], 0)

    def test_declarations_for_another_site_are_not_applied(self):
        self._base = self._site()
        cfg = self._cfg({"rush": {
            "meds_intermittent.propofol[mcg]": {"from": "mcg", "to": "mg", "factor": 1.0}}})
        with self.assertRaises(QualificationError):
            self._read(cfg, "s")


if __name__ == "__main__":
    unittest.main()
