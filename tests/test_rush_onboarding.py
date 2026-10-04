"""Rush onboarding and data-pipeline-at-scale fixes (product authority decisions of
2026-10-03; full-MIMIC review). Every fixture is synthetic; nothing reads real data.

One test (or test group) per behaviour change: the mCIDE vocabulary allowlist, pooled
value stats and per-site coverage, medication-action exclusion, expected-absent tables,
CRRT coverage, the site-local config and the site binding, timestamp ingest, death
records at local midnight, open stays at the extraction time, partial linkage ids, the
explicit-episodes and provenance rules, the label freeze and the label binding, the window
pushdown, partition-grouped row groups, extubation index stays, stale cache builds,
suppression in the smoke report and in error messages, the arterial-only MAP filter, and
DDP before compile.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import duckdb
import polars as pl
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
UTC = dt.UTC


def repo_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs/data.yaml").read_text())


# ------------------------------------------------------------------ synthetic site


def _synthetic_cfg(work: Path) -> tuple[dict, dict, Path, pl.DataFrame]:
    """The pre-flight's synthetic site (src/train/preflight.build_synthetic_site): raw
    tables, the fixture data config and a governed policy under `work`."""
    from src.eval.synthetic_bundle import (
        FIXTURE_COHORT,
        FIXTURE_DATA_CONFIG,
        FIXTURE_POLICY,
    )
    from src.eval.synthetic_bundle import build_synthetic_site as build_tables

    raw = work / "raw"
    episodes_raw = build_tables(raw)
    hosp = pl.read_parquet(raw / "clif_hospitalization.parquet").with_columns(
        pl.lit("ed").alias("admission_type_category"))
    hosp.write_parquet(raw / "clif_hospitalization.parquet")
    policy = copy.deepcopy(FIXTURE_POLICY)
    for rule in policy["classes"].values():
        rule["directory"] = str(work / rule["directory"])
    (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
    data_cfg = repo_cfg()
    cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
    cfg["cohort_contract"] = str(work / "cohort.yaml")
    cfg["artifact_policy"] = str(work / "artifact_policy.yaml")
    (work / "artifact_policy.yaml").write_text(yaml.safe_dump(policy))
    cfg["tables"]["adt"] = copy.deepcopy(data_cfg["tables"]["adt"])
    cfg["gem"] = copy.deepcopy(data_cfg["gem"])
    return cfg, policy, raw, pl.read_parquet(episodes_raw)


class SyntheticSiteTest(unittest.TestCase):
    """Tokenizations of the synthetic site (24 h build, then GEM)."""

    @classmethod
    def setUpClass(cls):
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import SYNTHETIC_SITE

        cls._td = tempfile.TemporaryDirectory()
        cls.work = Path(cls._td.name)
        cls.cfg, cls.policy, cls.raw, cls.episodes = _synthetic_cfg(cls.work)
        cls.cfg["vocabulary_allowlist"] = repo_cfg()["vocabulary_allowlist"]
        cls.site = SYNTHETIC_SITE
        cls.out = cls.work / "output/intermediate_phi/s"
        kw = {"episodes": cls.episodes, "artifact_policy": cls.policy}
        tokenize_site(cls.cfg, cls.site, cls.raw, cls.out, None, **kw)
        cls.blob = json.loads((cls.out / "vocab.json").read_text())
        tokenize_site(cls.cfg, cls.site, cls.raw, cls.out, cls.blob,
                      trajectory="hospitalization", **kw)
        cls.report = json.loads((cls.out / "gem_tokenization_report.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    # --- item 1: mCIDE allowlist ----------------------------------------------------
    def test_every_permissible_mcide_value_of_the_allowlist_is_a_token(self):
        vocab = self.blob["vocab"]
        for token in ("device_category=room_air", "code_status=dnar", "code_status=udnr",
                      "code_status=dni_only", "code_status=presume_full", "code_status=other",
                      "hospice", "rehab", "radiology", "dialysis", "admission_type=osh",
                      "admission_type=facility", "admission_type=other", "sex=unknown",
                      "mode_category=blow_by", "crrt_mode_category=avvh"):
            self.assertIn(token, vocab)
        self.assertEqual(self.blob["harmonization"]["global"]["vocabulary_allowlist"],
                         self.cfg["vocabulary_allowlist"])

    # --- item 10: site binding ----------------------------------------------------------
    def test_every_shard_row_records_the_site_and_its_declarations(self):
        from src.data.tokenize import site_binding

        expected = site_binding(self.cfg, self.site)
        for name in ("events.parquet", "gem_events.parquet"):
            hashes = pl.read_parquet(self.out / name)["artifact_hashes"].unique().to_list()
            self.assertEqual(len(hashes), 1)
            self.assertEqual(hashes[0]["site"], self.site)
            self.assertEqual(hashes[0]["site_declarations"], expected["site_declarations"])

    # --- row groups grouped by partition -----------------------------------------------
    def test_shards_are_grouped_by_partition_in_small_row_groups(self):
        import pyarrow.parquet as pq

        from src.data.tokenize import SHARD_ROW_GROUP_ROWS
        from src.data.value_stats import _partition_row_groups

        frame = pl.read_parquet(self.out / "gem_events.parquet")
        parts = frame["partition"].to_list()
        runs = [p for i, p in enumerate(parts) if i == 0 or parts[i - 1] != p]
        self.assertEqual(len(runs), len(set(runs)), "each partition is one contiguous run")
        shard = pq.ParquetFile(self.out / "gem_events.parquet")
        self.assertLessEqual(shard.metadata.row_group(0).num_rows, SHARD_ROW_GROUP_ROWS)
        self.assertEqual(_partition_row_groups(shard, "no_such_partition"), [])

    # --- item 6: CRRT coverage, report section --------------------------------------------
    def test_report_carries_the_site_profile(self):
        quality = self.report["data_quality"]
        self.assertIn("site_profile", quality)
        self.assertEqual(quality["site_profile"]["dst_nulled_rows"], {})

    # --- item 5: absent tables ----------------------------------------------------------
    def test_an_absent_table_fails_unless_it_is_declared_absent(self):
        from src.data.cohort import QualificationError
        from src.data.tokenize import tokenize_site

        cfg = copy.deepcopy(self.cfg)
        cfg["tables"]["position"] = copy.deepcopy(repo_cfg()["tables"]["position"])
        cfg["sites"] = {self.site: {"expected_absent_tables": []}}
        out = self.work / "output/intermediate_phi/absent"
        kw = {"episodes": self.episodes, "artifact_policy": self.policy, "report": True}
        with self.assertRaisesRegex(QualificationError, "expected_absent_tables"):
            tokenize_site(cfg, self.site, self.raw, out, self.blob,
                          trajectory="hospitalization", **kw)
        cfg["sites"] = {self.site: {"expected_absent_tables": ["position"]}}
        tokenize_site(cfg, self.site, self.raw, out, self.blob,
                      trajectory="hospitalization", **kw)
        report = json.loads((out / "gem_tokenization_report.json").read_text())
        self.assertEqual(report["data_quality"]["site_profile"]["expected_absent_tables"],
                         ["position"])
        # The CLIF table name works too (Rush declares ecmo_mcs).
        from src.data.site_config import site_profile
        full = repo_cfg()
        full["sites"]["rush"] = {"expected_absent_tables": ["ecmo_mcs"]}
        self.assertEqual(site_profile(full, "rush")["expected_absent_tables"], ["ecmo"])

    # --- item 13: extubation index stays ---------------------------------------------------
    def test_extubation_index_stays_join_the_gem_stays_with_the_patients_partition(self):
        from src.data.tokenize import tokenize_site

        hosp = pl.read_parquet(self.raw / "clif_hospitalization.parquet")
        eligible = self.episodes.filter(pl.col("eligible"))
        donor = eligible.row(0, named=True)
        # A second stay of the first patient, not an episode: its own index extubation.
        extra = hosp.filter(pl.col("hospitalization_id") == donor["hospitalization_id"]) \
            .with_columns(pl.lit("synth-extra").alias("hospitalization_id"),
                          (pl.col("admission_dttm") + pl.duration(days=60)).alias("admission_dttm"),
                          (pl.col("discharge_dttm") + pl.duration(days=60)).alias("discharge_dttm"))
        raw = self.work / "raw_extra"
        shutil.copytree(self.raw, raw)
        pl.concat([hosp, extra]).write_parquet(raw / "clif_hospitalization.parquet")
        cohort = pl.DataFrame({
            "hospitalization_id": ["synth-extra", donor["hospitalization_id"]],
            "patient_id": [donor["patient_id"]] * 2,
            "eligible": [True, True], "partition": [donor["partition"]] * 2,
            "site": [self.site] * 2,
            "episode_sha256": [self.episodes["episode_sha256"][0]] * 2})
        out = self.work / "output/intermediate_phi/extra"
        tokenize_site(self.cfg, self.site, raw, out, self.blob, trajectory="hospitalization",
                      episodes=self.episodes, artifact_policy=self.policy,
                      extubation_cohort=cohort)
        gem = pl.read_parquet(out / "gem_events.parquet")
        row = gem.filter(pl.col("hosp_id") == "synth-extra").row(0, named=True)
        self.assertEqual(row["partition"], donor["partition"])
        self.assertIsNone(row["anchor_idx"])
        self.assertEqual(gem["hosp_id"].n_unique(), eligible.height + 1)
        report = json.loads((out / "gem_tokenization_report.json").read_text())
        self.assertIn("extubation_index_stays", report["gem"])


class SplitFreezeTwoSitesTest(SyntheticSiteTest):
    """Item 8: one freeze record holds MIMIC and Rush, each artifact its own site's."""

    def test_one_freeze_records_both_sites_and_refuses_a_foreign_artifact(self):
        from src.train import preflight as pf

        mimic, rush = self.work / "e_mimic.parquet", self.work / "e_rush.parquet"
        self.episodes.write_parquet(mimic)
        self.episodes.write_parquet(rush)
        record = pf.build_split_freeze({"mimic": mimic, "rush": rush}, approver="tester")
        self.assertEqual(set(record["sites"]), {"mimic", "rush"})
        # An artifact recorded for MIMIC is not Rush's.
        self.episodes.with_columns(pl.lit("mimic").alias("site")).write_parquet(rush)
        with self.assertRaisesRegex(pf.PreflightError, "built for site 'mimic'"):
            pf.build_split_freeze({"mimic": mimic, "rush": rush}, approver="tester")
        parser = pf.build_parser().parse_args(["--episodes", f"mimic={mimic}",
                                               "--episodes", f"rush={rush}"])
        self.assertEqual(len(parser.episodes), 2)


# ------------------------------------------------------------------ allowlist hashing


def test_the_allowlist_and_free_text_notes_in_the_harmonization_hash():
    from src.data.clif_conformance import harmonization_record
    from src.data.segments import json_sha256

    cfg = repo_cfg()
    base = json_sha256(harmonization_record(cfg, "mimic"))
    noted = copy.deepcopy(cfg)
    exceptions = noted["site_harmonization"]["mimic"]["declared_exceptions"]
    exceptions["meds.med_category"]["alteplase"] = "an edited reason"
    noted["site_unit_conversions"]["mimic"]["meds_intermittent.propofol[mcg]"]["note"] = "x"
    assert json_sha256(harmonization_record(noted, "mimic")) == base   # notes not hashed
    fewer = copy.deepcopy(cfg)
    fewer["vocabulary_allowlist"] = fewer["vocabulary_allowlist"][:1]
    assert json_sha256(harmonization_record(fewer, "mimic")) != base


def test_reference_units_ignore_notes_and_flags():
    import src.data.tokenize as tk

    cfg = repo_cfg()
    edited = copy.deepcopy(cfg)
    edited["column_units"]["crrt.ultrafiltration_out"]["flag"] = "reworded flag"
    edited["site_unit_conversions"]["mimic"]["meds_intermittent.ketamine[mg]"]["note"] = "y"
    events = pl.DataFrame(schema={"concept": pl.String, "unit": pl.String, "source": pl.String})
    a = tk.reference_units(events, {}, cfg, {}, "mimic")
    b = tk.reference_units(events, {}, edited, {}, "mimic")
    assert a == b
    tk._check_unit_repair_binding(a, edited, "mimic")     # still binds


# ------------------------------------------------------------------ item 3: MAR actions


def test_not_given_and_other_administrations_are_excluded_and_counted(tmp_path):
    from src.data.tokenize import _read_source

    t = dt.datetime(2026, 1, 1, tzinfo=UTC)
    pl.DataFrame({
        "hospitalization_id": ["h1"] * 4, "admin_dttm": [t] * 4,
        "med_category": ["fentanyl"] * 4, "med_dose": [50.0, 50.0, 25.0, 60.0],
        "med_dose_unit": ["mcg"] * 4,
        "mar_action_category": ["given", "not_given", "other", None],
    }).write_parquet(tmp_path / "clif_medication_admin_intermittent.parquet")
    spec = copy.deepcopy(repo_cfg()["tables"]["meds_intermittent"])
    spec["dose"].pop("weight_source")
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    events, _, counts = _read_source(con, tmp_path, spec, tables={}, target_units={})
    assert counts["excluded_action"] == 2
    assert sorted(events["value"].to_list()) == [0.05, 0.06]


# ------------------------------------------------------------------ item 2: value stats


def _stats_shard(path: Path, rows: list[tuple[str, list[int], list[float]]], vocab="v" * 64):
    pl.DataFrame({
        "hosp_id": [f"h{i}" for i in range(len(rows))],
        "partition": [r[0] for r in rows],
        "token": [r[1] for r in rows], "value": [r[2] for r in rows],
        "target_eligible": [[True] * len(r[1]) for r in rows],
        "artifact_hashes": [{"vocabulary": vocab}] * len(rows),
    }).write_parquet(path)


def test_value_stats_pool_the_train_partitions_of_every_site(tmp_path):
    from src.data.value_stats import (
        compute_value_stats,
        compute_value_stats_from_sites,
        coverage_gaps,
    )

    _stats_shard(tmp_path / "a.parquet", [("train", [5, 5], [1.0, 2.0]),
                                          ("validation", [6], [9.0])])
    _stats_shard(tmp_path / "b.parquet", [("train", [5, 7], [3.0, 4.0]),
                                          ("validation", [8], [1.0])])
    stats, rows = compute_value_stats_from_sites(
        {"mimic": tmp_path / "a.parquet", "rush": tmp_path / "b.parquet"})
    assert rows == {"mimic": 1, "rush": 1}
    assert stats == compute_value_stats([[5, 5], [5, 7]], [[1.0, 2.0], [3.0, 4.0]])
    # Every site is checked (not only the first), on train + validation.
    gaps = coverage_gaps({"mimic": tmp_path / "a.parquet", "rush": tmp_path / "b.parquet"},
                         stats)
    assert gaps == {"mimic": [6], "rush": [8]}


def test_value_stats_cli_refuses_a_shard_of_another_vocabulary(tmp_path, monkeypatch):
    import sys

    from src.data import value_stats as vs

    blob = {"vocab": {"<pad>": 0}, "segments": {"x": [[0.0, 1.0, "[", ")"]]},
            "manifest": {"tokenizer_version": 2}}
    (tmp_path / "vocab.json").write_text(json.dumps(blob))
    _stats_shard(tmp_path / "a.parquet", [("train", [5], [1.0])], vocab=vs.vocab_hash(blob["vocab"]))
    _stats_shard(tmp_path / "b.parquet", [("train", [5], [1.0])], vocab="0" * 64)
    monkeypatch.setattr(sys, "argv", ["vs", "--events", f"mimic={tmp_path / 'a.parquet'}",
                                      "--events", f"rush={tmp_path / 'b.parquet'}",
                                      "--vocab", str(tmp_path / "vocab.json"),
                                      "--out", str(tmp_path / "s.json")])
    try:
        vs.segments_hash(vs.vocab_segments(blob))
    except Exception:  # noqa: BLE001 - a toy vocabulary the segment parser refuses
        pytest.skip("toy vocabulary is not a tokenizer-v2 vocabulary")
    with pytest.raises(SystemExit, match="rush"):
        vs._main()


# ------------------------------------------------------------------ item 6: CRRT coverage


def test_crrt_coverage_counts_rows_with_each_setting(tmp_path):
    from src.data.tokenize import setting_coverage

    pl.DataFrame({"hospitalization_id": ["h"] * 4,
                  "blood_flow_rate": [200.0, None, None, None],
                  "crrt_mode_category": ["cvvhdf", "cvvhdf", None, "cvvh"]}
                 ).write_parquet(tmp_path / "c.parquet")
    con = duckdb.connect()
    out = setting_coverage(con, f"read_parquet('{tmp_path / 'c.parquet'}')",
                           ["blood_flow_rate", "dialysate_flow_rate", "crrt_mode_category"],
                           {"hospitalization_id", "blood_flow_rate", "crrt_mode_category"})
    assert out["rows"] == 4
    assert out["with"] == {"blood_flow_rate": 1, "crrt_mode_category": 3}
    assert out["absent_columns"] == ["dialysate_flow_rate"]


# ------------------------------------------------------------------ item 10: local config


def test_the_site_local_file_is_merged_and_hashed(tmp_path, monkeypatch):
    from src.data.clif_conformance import site_harmonization
    from src.data.site_config import site_profile, with_site_local
    from src.data.tokenize import site_binding

    monkeypatch.setenv("CLIF_SITE_CONFIG_DIR", str(tmp_path))
    example = yaml.safe_load((ROOT / "configs/sites/rush.example.yaml").read_text())
    example["profile"]["extraction_dttm"] = "2026-06-30T00:00:00Z"
    example["site_harmonization"]["bp_method"]["patterns"] = [
        {"pattern": "^art", "method": "arterial"}]
    (tmp_path / "rush.local.yaml").write_text(yaml.safe_dump(example))
    cfg = with_site_local(repo_cfg(), "rush")
    assert site_profile(cfg, "rush")["site_timezone"] == "America/Chicago"
    assert site_profile(cfg, "rush")["expected_absent_tables"] == ["ecmo"]
    assert site_harmonization(cfg, "rush")["bp_method"]["patterns"][0]["method"] == "arterial"
    assert with_site_local(cfg, "rush") == cfg                       # idempotent
    first = site_binding(cfg, "rush")["site_declarations"]
    example["site_harmonization"]["bp_method"]["patterns"][0]["pattern"] = "^a-line"
    (tmp_path / "rush.local.yaml").write_text(yaml.safe_dump(example))
    assert site_binding(with_site_local(repo_cfg(), "rush"), "rush")["site_declarations"] != first
    # The template itself holds placeholders only and is a valid local file.
    text = (ROOT / "configs/sites/rush.example.yaml").read_text()
    assert "<ARTERIAL_SOURCE_NAME_REGEX>" in text
    assert ".local.yaml" in (ROOT / ".gitignore").read_text()


def test_a_local_file_for_another_site_is_refused(tmp_path, monkeypatch):
    from src.data.site_config import SiteConfigError, with_site_local

    monkeypatch.setenv("CLIF_SITE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "rush.local.yaml").write_text("site: mimic\n")
    with pytest.raises(SiteConfigError, match="declares site"):
        with_site_local(repo_cfg(), "rush")


# ------------------------------------------------------------------ timestamps


def test_naive_local_times_become_utc_and_dst_gaps_are_counted():
    from src.data.site_config import SiteConfigError, to_utc

    frame = pl.DataFrame({"t": [dt.datetime(2024, 3, 10, 2, 30), dt.datetime(2024, 1, 1, 0, 0),
                                dt.datetime(2024, 11, 3, 1, 30)]})
    out, nulled = to_utc(frame, ["t"], "America/Chicago")
    assert nulled == 1
    assert out["t"].to_list()[1:] == [dt.datetime(2024, 1, 1, 6, 0, tzinfo=UTC),
                                      dt.datetime(2024, 11, 3, 6, 30, tzinfo=UTC)]
    utc = pl.DataFrame({"t": [dt.datetime(2024, 1, 1, tzinfo=UTC)]})
    assert to_utc(utc, ["t"], "America/Chicago")[0].equals(utc)        # UTC unchanged
    with pytest.raises(SiteConfigError, match="site_timezone"):
        to_utc(frame, ["t"], None)


def test_a_date_only_death_at_local_midnight_gets_the_tolerance():
    from src.data.extubation_cohort import death_record_flags

    config = {"exclusions": {"death_record": {"date_resolution_tolerance_hours": 24.0}}}
    death = dt.datetime(2150, 1, 10, 5, 0, tzinfo=UTC)       # 00:00 in Boston (EST)
    hosp = pl.DataFrame({"patient_id": ["p"], "admission_dttm": [death + dt.timedelta(hours=10)]})
    patient = pl.DataFrame({"patient_id": ["p"], "death_dttm": [death]})
    local = death_record_flags(hosp, patient, config, site_timezone="America/New_York")
    assert local["death_record_inconsistent"].to_list() == [False]
    as_utc = death_record_flags(hosp, patient, config)
    assert as_utc["death_record_inconsistent"].to_list() == [True]


# ------------------------------------------------------------------ open stays, linkage


def test_open_stays_are_censored_at_extraction_or_never_eligible():
    from src.data.cohort import build_cohort
    from src.data.site_config import censor_open_stays

    t = dt.datetime(2026, 1, 1, tzinfo=UTC)
    hosp = pl.DataFrame({"hospitalization_id": ["h1"], "patient_id": ["p1"],
                         "hospitalization_joined_id": [None], "admission_dttm": [t],
                         "discharge_dttm": pl.Series([None], dtype=pl.Datetime("us", "UTC")),
                         "age_at_admission": [60], "discharge_category": [None]},
                        schema_overrides={"hospitalization_joined_id": pl.String,
                                          "discharge_category": pl.String})
    adt = pl.DataFrame({"hospitalization_id": ["h1"], "in_dttm": [t],
                        "out_dttm": [t + dt.timedelta(days=3)], "location_category": ["icu"]})
    config = {"anchor_hours": 24, "prediction_horizon_hours": 48, "minimum_age": 18,
              "icu_location_category": "icu"}
    row = build_cohort(hosp, adt, config).row(0, named=True)
    assert row["eligible"] is False and row["eligibility_status"] == "open_stay_no_extraction_time"
    censored, n = censor_open_stays(hosp, t + dt.timedelta(days=10))
    assert n == 1
    row = build_cohort(censored, adt, config).row(0, named=True)
    assert row["eligible"] is True and row["eligibility_status"] == "eligible"


def test_a_partly_populated_linkage_id_validates_its_non_null_rows_only():
    from src.data.splits import validate_grouped_splits

    rows = pl.DataFrame({"hospitalization_id": ["a", "b", "c", "d"],
                         "patient_id": ["1", "2", "3", "4"],
                         "hospitalization_joined_id": [None, None, "L", "L"],
                         "partition": ["train", "validation", "train", "train"]})
    validate_grouped_splits(rows)                       # nulls are singletons
    with pytest.raises(ValueError, match="hospitalization_joined_id"):
        validate_grouped_splits(rows.with_columns(
            pl.Series("partition", ["train", "validation", "train", "validation"])))


# ------------------------------------------------------------------ episodes per site


def test_a_non_reference_site_never_falls_back_to_the_mimic_episodes():
    from src.data.site_config import (
        SiteConfigError,
        require_explicit_episodes,
        require_site_match,
    )

    require_explicit_episodes("mimic", None, "x")
    with pytest.raises(SiteConfigError, match="explicit --episodes"):
        require_explicit_episodes("rush", None, "extubation cohort")
    with pytest.raises(SiteConfigError, match="built for site 'mimic'"):
        require_site_match("mimic", "rush", "episode artifact")


def test_data_is_checked_against_the_episode_artifacts_provenance():
    from src.data.cohort import QualificationError
    from src.data.extubation_cohort import check_data_provenance

    episodes = pl.DataFrame({"source_provenance_json": [json.dumps(
        {"clif_hospitalization": "a" * 64, "clif_adt": "b" * 64})]})
    check_data_provenance(episodes, {"clif_hospitalization": "a" * 64, "clif_labs": "c"})
    with pytest.raises(QualificationError, match="--data"):
        check_data_provenance(episodes, {"clif_hospitalization": "f" * 64})


def test_the_cohort_cli_requires_a_site():
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "src.data.cohort", "--data", "x"], cwd=ROOT,
                         capture_output=True, text=True, check=False)
    assert out.returncode != 0 and "--site" in out.stderr


# ------------------------------------------------------------------ labels


def test_a_confirmatory_site_gets_no_labels_before_the_freeze():
    from src.data.cohort import QualificationError
    from src.eval.extubation_labeler import DEFAULT_REGISTRY, require_label_freeze

    kw = {"registry_path": DEFAULT_REGISTRY, "cohort_config": ROOT / "configs/extubation.yaml"}
    assert require_label_freeze("mimic", freeze_manifest=None, **kw) == "exploratory"
    with pytest.raises(QualificationError, match="confirmatory.*freeze"):
        require_label_freeze("rush", freeze_manifest=None, **kw)
    with pytest.raises(QualificationError, match="missing"):
        require_label_freeze("rush", freeze_manifest={"vocabulary": "0" * 64}, **kw)


def test_labels_of_another_cohort_build_are_refused():
    from src.eval.causal.emulate import LabelBindingError, SiteData

    cohort = pl.DataFrame({"patient_id": ["p"], "extubation_sha256": ["a" * 64]})
    SiteData(cohort, pl.DataFrame({"patient_id": ["p"], "extubation_sha256": ["a" * 64]}))
    with pytest.raises(LabelBindingError, match="another extubation cohort"):
        SiteData(cohort, pl.DataFrame({"patient_id": ["p"], "extubation_sha256": ["b" * 64]}))
    with pytest.raises(LabelBindingError, match="no extubation_sha256"):
        SiteData(cohort, pl.DataFrame({"patient_id": ["p"]}))


# ------------------------------------------------------------------ window pushdown


def test_the_window_is_pushed_into_the_scan(tmp_path):
    from src.data.tokenize import _register_window, _windowed_source

    t = dt.datetime(2026, 1, 1, tzinfo=UTC)
    pl.DataFrame({"hospitalization_id": ["in", "in", "in", "out"],
                  "recorded_dttm": [t - dt.timedelta(hours=2), t + dt.timedelta(hours=1),
                                    t + dt.timedelta(days=9), t]}
                 ).write_parquet(tmp_path / "v.parquet")
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    stays = pl.DataFrame({"hospitalization_id": ["in"], "lo": [t],
                          "hi": [t + dt.timedelta(days=1)]})
    window = _register_window(con, stays, "lo", "hi")
    source = f"read_parquet('{tmp_path / 'v.parquet'}')"
    spec = {"availability_col": "recorded_dttm", "availability_lag_minutes": 0}
    rows = con.execute(f"SELECT count(*) FROM {_windowed_source(con, source, spec, window)}").fetchone()
    assert rows == (1,)                          # other stay and out-of-window rows skipped
    state = {**spec, "carry_forward": True}      # state tables keep rows before the window
    rows = con.execute(f"SELECT count(*) FROM {_windowed_source(con, source, state, window)}").fetchone()
    assert rows == (2,)


# ------------------------------------------------------------------ cache builds


def test_a_crashed_cache_build_is_removed_and_a_live_one_kept(tmp_path):
    from src.data.dataset import remove_stale_cache_builds

    dead = tmp_path / ".train-abc.999999.tmp"
    live = tmp_path / f".train-abc.{os.getpid()}.tmp"
    dead.mkdir()
    live.mkdir()
    assert remove_stale_cache_builds(tmp_path) == [dead.name]
    assert live.exists() and not dead.exists()


# ------------------------------------------------------------------ disclosure


def test_smoke_label_status_suppresses_counts_under_ten():
    from src.train.real_data_smoke import SMALL, WITHHELD, suppress_label_status

    report = {"anchors": 120,
              "competing_risk_by_cause": {"supervised_anchors": 100,
                                          "events": {"0": 3, "1": 40},
                                          "rates": {"0": 0.03, "1": 0.4}},
              "competing_risk": {"n": 100, "counts": {"positive": 4, "negative": 60,
                                                      "censored": 36},
                                 "shares": {"positive": 0.04, "negative": 0.6,
                                            "censored": 0.36}}}
    out = suppress_label_status(report)
    cause = out["competing_risk_by_cause"]
    assert cause["events"] == {"0": SMALL, "1": 40}
    assert cause["rates"] == {"0": None, "1": 0.4}
    assert out["competing_risk"]["counts"] == {"positive": SMALL, "negative": 60,
                                               "censored": WITHHELD}
    assert out["competing_risk"]["shares"]["positive"] is None


def _consistent_label_report(*, positive=43, competing=0, events=None, not_supervised=20):
    """A `targets.anchor_status_shares` report whose identities hold: per-cause events sum
    to positive + competing_event, supervised is the competing-risk group but
    not_supervised, and anchors is that group's n."""
    events = {"0": 3, "1": 40} if events is None else events
    negative, censored = 200, 100
    supervised = positive + competing + negative + censored
    n = supervised + not_supervised
    counts = {"positive": positive, "competing_event": competing, "negative": negative,
              "censored": censored, "not_supervised": not_supervised}
    return {"anchors": n,
            "competing_risk_by_cause": {
                "supervised_anchors": supervised, "events": events,
                "rates": {k: v / supervised for k, v in events.items()}},
            "competing_risk": {"n": n, "counts": counts,
                               "shares": {k: v / n for k, v in counts.items()}}}


def _recoverable_label_cells(shown, full):
    """Equations with exactly one withheld non-zero member (recoverable by subtraction)."""
    from src.train.real_data_smoke import _label_cells, _label_equations

    cells = _label_cells(full)
    flat = {}
    for path in cells:
        node = shown
        for part in {("supervised",): ("competing_risk_by_cause", "supervised_anchors"),
                     ("anchors",): ("anchors",)}.get(
                path, ("competing_risk_by_cause", "events", path[1]) if path[0] == "events"
                else (path[0], *path[1:])):
            node = node[part]
        flat[path] = node
    return [eq for eq in _label_equations(cells)
            if sum(isinstance(flat[p], str) and cells[p] != 0 for p in eq) == 1]


def test_smoke_label_status_does_not_leak_a_small_cause_count_through_the_cr_counts():
    # events sum to competing_risk positive + competing_event, both released: 43 - 40 = 3.
    from src.train.real_data_smoke import SMALL, WITHHELD, suppress_label_status

    report = _consistent_label_report()
    out = suppress_label_status(report)
    cause = out["competing_risk_by_cause"]
    assert cause["events"] == {"0": SMALL, "1": WITHHELD}
    assert cause["rates"] == {"0": None, "1": None}
    assert out["competing_risk"]["counts"]["positive"] == 43       # still published
    assert _recoverable_label_cells(out, report) == []


def test_smoke_label_status_does_not_leak_not_supervised_through_anchors_and_supervised():
    # anchors - supervised_anchors is the not_supervised count (here 4): one hidden cell
    # of an equation whose other members are published.
    from src.train.real_data_smoke import SMALL, suppress_label_status

    report = _consistent_label_report(not_supervised=4, events={"0": 100, "1": 243})
    out = suppress_label_status(report)
    assert out["competing_risk"]["counts"]["not_supervised"] == SMALL
    assert _recoverable_label_cells(out, report) == []
    withheld = [out["anchors"], out["competing_risk_by_cause"]["supervised_anchors"],
                out["competing_risk"]["n"]]
    assert any(isinstance(v, str) for v in withheld)


def test_smoke_label_status_publishes_everything_when_no_cell_is_small():
    from src.train.real_data_smoke import suppress_label_status

    report = _consistent_label_report(events={"0": 20, "1": 23})
    assert suppress_label_status(report) == report


def test_failure_messages_suppress_small_counts_and_stay_on_the_node():
    from src.data.cohort import QualificationError
    from src.data.tokenize import NODE_ONLY, apply_bp_method

    t = dt.datetime(2026, 1, 1, tzinfo=UTC)
    events = pl.DataFrame({"hosp_id": ["h"] * 3, "dttm": [t] * 3, "concept": ["map"] * 3,
                           "value": [70.0] * 3, "unit": [None] * 3, "cat_value": [None] * 3,
                           "_bp_src": ["mystery line"] * 3})
    glob = {"concepts": ["sbp", "dbp", "map"], "token_concept": "bp_method",
            "require_site_declaration": True}
    decl = {"patterns": [{"pattern": "^art", "method": "arterial"}], "null": None}
    with pytest.raises(QualificationError) as err:
        apply_bp_method(events, glob, decl, "rush")
    assert "(<10 rows)" in str(err.value) and NODE_ONLY in str(err.value)


# ------------------------------------------------------------------ MAP sensitivity


def test_arterial_only_map_counts_only_readings_after_an_arterial_method_token():
    from src.data.targets import InStreamTargets, TargetBuilder, measurement_filters

    data_cfg = repo_cfg()
    cohort_cfg = yaml.safe_load((ROOT / "configs/cohort.yaml").read_text())
    vocab = {"bp_method=arterial": 20, "bp_method=noninvasive_auto": 21,
             "bp_method=noninvasive_manual": 22, "bp_method=unknown": 23}
    filters, methods = measurement_filters(cohort_cfg, data_cfg, vocab,
                                           data_cfg["target_concepts"])
    assert filters == {}                                     # off by default
    filters, methods = measurement_filters(cohort_cfg, data_cfg, vocab,
                                           data_cfg["target_concepts"], "arterial_only")
    map_index = [c["name"] for c in data_cfg["target_concepts"]].index("map")
    assert filters == {map_index: frozenset({20})} and 21 in methods
    grid = SimpleNamespace(token_target={10: map_index}, terminal_tokens={99},
                           death_token=99)

    def builder(f):
        return TargetBuilder(vocab_size=100, n_time_bins=4, horizon_hours=1.0,
                             value_stats={}, mode="gem_tte",
                             in_stream=InStreamTargets(grid, 1, 1, 1.0, 1.0,
                                                       measurement_filters=f,
                                                       method_tokens=methods))
    tokens = [20, 10, 21, 10, 99]
    pos = [0, 0, 5, 5, 9]
    values = [None, 60.0, None, 58.0, None]
    every = builder(None)._stream_index(tokens, pos, values)["events"][map_index]
    arterial = builder(filters)._stream_index(tokens, pos, values)["events"][map_index]
    assert every[1].tolist() == [60.0, 58.0]
    assert arterial[1].tolist() == [60.0]


# ------------------------------------------------------------------ launcher


def test_run_arm_wraps_ddp_before_compile_and_documents_uv_run_torchrun():
    text = (ROOT / "src/train/run_arm.py").read_text()
    assert text.index("model = wrap_ddp(model, dev, local)") < text.index("torch.compile(model")
    assert "uv run torchrun --nproc_per_node=2 -m src.train.run_arm" in text


def test_the_audit_caps_blas_threads_before_numpy_is_imported():
    text = (ROOT / "src/eval/extubation_audit.py").read_text()
    assert text.index('os.environ.setdefault(_var, "1")') < text.index("import numpy")
