"""U7 (R12, R15, R16; KTD9): aggregate-only tokenization report, the verification
sample, the sample-vocabulary training refusal, and the real-data gate.

Everything here is synthetic. The real-data run (`--sample-episodes` on the staged
reference site) is an operator step on the governed node; these tests pin the contract
it relies on:

- `tokenize_site` writes `tokenization_report.json` (24 h) / `gem_tokenization_report.json`
  (GEM) beside the events with no identifiers anywhere (keys or values) and every
  patient-derived count under 10 suppressed to "<10";
- `--sample-episodes N` draws N ELIGIBLE episodes deterministically by a salted hash of
  the hospitalization id (never the first N ids) and marks the vocabulary
  `provenance.sample: true`;
- training (`build_loaders`, non-dry-run) refuses a sample vocabulary;
- the gate fails when a table lacks an availability declaration.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

try:  # pytest puts tests/ on sys.path (rootdir-less test modules)
    from test_tokenize_alignment import _build_u4_site, _repartition
except ImportError:  # pragma: no cover - run from the repo root as a package
    from tests.test_tokenize_alignment import _build_u4_site, _repartition

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER_KEYS = {"hosp_id", "hospitalization_id", "patient_id",
                   "hospitalization_joined_id", "episode_key"}


def _walk(node, path=()):
    """Yield (path, key-or-None, value) for every key and leaf of a JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, None
            yield from _walk(value, (*path, key))
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value, path)
    else:
        yield path, None, node


# ---------------------------------------------------------------- value placement

class ClassifyValuesTest(unittest.TestCase):
    """`segments.classify_values` (gap / clamp counters) follows `bin_index` exactly."""

    def _reference(self, value, segments):
        from src.data import segments as S

        for seg in segments:
            if S.is_point(seg) and seg["lo"] == value:
                return "inside"
        if any(not S.is_point(s) and S.contains(s, value) for s in segments):
            return "inside"
        if segments[0]["lo"] is not None and value <= segments[0]["lo"]:
            return "clamp_low"
        if segments[-1]["hi"] is not None and value >= segments[-1]["hi"]:
            return "clamp_high"
        return "gap"

    def test_counts_match_the_bin_index_code_path(self):
        from src.data import segments as S

        segments = [
            S.make_segment(0.0, 1.0, False, True),     # (0, 1]
            S.make_segment(2.0, 2.0, True, True),      # [2]       gap (1, 2)
            S.make_segment(3.0, 5.0, True, False),     # [3, 5)    gap (2, 3)
            S.make_segment(5.0, 8.0, True, True),      # [5, 8]
        ]
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.uniform(-2, 10, 400),
                                 [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 8.0, 8.5, -1.0]])
        expected = {"inside": 0, "gap": 0, "clamp_low": 0, "clamp_high": 0}
        for v in values:
            expected[self._reference(float(v), segments)] += 1
        self.assertEqual(S.classify_values(values, segments), expected)
        self.assertGreater(expected["gap"], 0)
        self.assertGreater(expected["clamp_low"], 0)
        self.assertGreater(expected["clamp_high"], 0)

    def test_unbounded_quantile_segments_never_clamp(self):
        from src.data import segments as S

        segments = S.segments_from_edges([1.0, 2.0])
        counts = S.classify_values(np.array([-1e9, 0.0, 1.0, 1.5, 3.0, 1e9]), segments)
        self.assertEqual(counts, {"inside": 6, "gap": 0, "clamp_low": 0, "clamp_high": 0})

    def test_non_finite_values_are_not_counted(self):
        from src.data import segments as S

        counts = S.classify_values(np.array([np.nan, np.inf, 0.5]),
                                   [S.make_segment(0.0, 1.0, True, True)])
        self.assertEqual(sum(counts.values()), 1)


# ---------------------------------------------------------------- suppression / gate

class SuppressionAndIdentifierGuardTest(unittest.TestCase):
    def test_patient_derived_counts_below_ten_are_suppressed(self):
        from src.data.tokenization_report import SUPPRESSED, suppress_small_cells

        report = {
            "vocab": {"size": 7, "binning_sources": {"csv": 2, "single": 1}},
            "sources": {"ecmo": {"events": 3, "stays": 0, "low_coverage": True},
                        "vitals": {"events": 12000, "stays": 400, "low_coverage": False}},
            "events": {"total": 12003, "stays": 400, "per_stay_mean": 30.0,
                       "context": 8192},
            "dose_conversion": {"meds": {"converted": 9, "no_weight": 10}},
        }
        out = suppress_small_cells(copy.deepcopy(report))
        self.assertEqual(out["sources"]["ecmo"]["events"], SUPPRESSED)
        self.assertEqual(out["sources"]["ecmo"]["stays"], SUPPRESSED)
        self.assertIs(out["sources"]["ecmo"]["low_coverage"], True)
        self.assertEqual(out["sources"]["vitals"]["stays"], 400)
        self.assertEqual(out["dose_conversion"]["meds"],
                         {"converted": SUPPRESSED, "no_weight": 10})
        # Structural (vocabulary) numbers and configuration are not patient counts.
        self.assertEqual(out["vocab"], report["vocab"])
        self.assertEqual(out["events"]["context"], 8192)
        self.assertEqual(out["events"]["per_stay_mean"], 30.0)

    def test_identifier_keys_and_values_are_refused(self):
        from src.data.cohort import QualificationError
        from src.data.tokenization_report import assert_aggregate_only

        assert_aggregate_only({"sources": {"vitals": {"stays": 12}}}, {"H-1"})
        with self.assertRaisesRegex(QualificationError, "identifier"):
            assert_aggregate_only({"rows": [{"hosp_id": "x"}]}, set())
        with self.assertRaisesRegex(QualificationError, "identifier"):
            assert_aggregate_only({"unk": {"by_concept": {"H-1": 12}}}, {"H-1"})
        with self.assertRaisesRegex(QualificationError, "identifier"):
            assert_aggregate_only({"note": ["H-1"]}, {"H-1"})


class SmallCellRecoveryTest(unittest.TestCase):
    """A suppressed count must not be recoverable by arithmetic: not as rate x tokens,
    and not as a published total minus the other cells of a section summing to it."""

    def test_a_rate_is_withheld_when_its_numerator_is_a_small_cell(self):
        from src.data.tokenization_report import _rate

        self.assertIsNone(_rate(7, 1000))          # 7 = 0.007 x 1000 would leak
        self.assertIsNone(_rate(1, 1000))
        self.assertEqual(_rate(0, 1000), 0.0)      # a zero discloses nobody
        self.assertEqual(_rate(10, 1000), 0.01)
        self.assertIsNone(_rate(50, 9))            # base too small
        self.assertIsNone(_rate(12, 1000, min_cell=20))

    def test_a_small_unk_count_has_no_recoverable_rate_in_the_written_report(self):
        from src.data.tokenization_report import SUPPRESSED, _unk, suppress_small_cells

        records = pl.DataFrame({
            "partition": ["train"] * 2 + ["val"] * 2,
            "token": [[5] * 600, [5] * 400, [3] * 7 + [5] * 493, [5] * 500],
        })
        report = suppress_small_cells({"fit_partition": "train",
                                       "unk": _unk(records, "train")})
        val = report["unk"]["by_partition"]["val"]
        self.assertEqual(val["unk"], SUPPRESSED)
        self.assertIsNone(val["rate"])
        self.assertEqual(report["unk"]["non_fit"]["unk"], SUPPRESSED)
        self.assertIsNone(report["unk"]["non_fit"]["rate"])
        for entry in (val, report["unk"]["non_fit"], report["unk"]["overall"]):
            if isinstance(entry["rate"], float) and isinstance(entry["tokens"], int):
                self.assertNotEqual(round(entry["rate"] * entry["tokens"]), 7)

    def test_a_single_suppressed_disposition_gets_a_complementary_cell(self):
        from src.data.tokenization_report import (
            COMPLEMENTARY,
            SUPPRESSED,
            suppress_small_cells,
        )

        report = {"gem": {"stays": 1000, "windows": 1200, "max_tokens": 64,
                          "dispositions": {"home": 820, "expired": 120, "hospice": 53,
                                           "ama": 7},
                          "admission_types": {"ed": 700, "elective": 300}}}
        out = suppress_small_cells(copy.deepcopy(report))["gem"]
        self.assertEqual(out["dispositions"]["ama"], SUPPRESSED)
        # stays - (home + expired + hospice) would give ama back: hospice (the next
        # smallest) is suppressed too, and nothing else.
        self.assertEqual(out["dispositions"]["hospice"], COMPLEMENTARY)
        self.assertEqual(out["dispositions"]["home"], 820)
        self.assertEqual(out["dispositions"]["expired"], 120)
        self.assertEqual(out["admission_types"], {"ed": 700, "elective": 300})
        self.assertEqual(out["stays"], 1000)

    def test_token_kinds_summing_to_the_event_total_are_protected(self):
        from src.data.tokenization_report import COMPLEMENTARY, SUPPRESSED, \
            suppress_small_cells

        report = {"events": {"total": 5000, "stays": 400, "context": 8192},
                  "token_kinds": {"binned": 4000, "numeric_bare": 0, "categorical": 600,
                                  "presence": 396, "skipped_missing_numeric": 4}}
        out = suppress_small_cells(copy.deepcopy(report))["token_kinds"]
        self.assertEqual(out["skipped_missing_numeric"], SUPPRESSED)
        self.assertEqual(out["numeric_bare"], SUPPRESSED)       # a zero hides nothing
        self.assertEqual(out["presence"], COMPLEMENTARY)
        self.assertEqual(out["categorical"], 600)

    def test_a_small_fit_partition_unk_hides_the_overall_not_the_gated_non_fit(self):
        """overall - non_fit = fit: with a 1..9 fit-partition <unk> count one of them
        goes; the gate needs the non-fit rate, so the overall is withheld."""
        from src.data.tokenization_report import gate_report, suppress_small_cells

        report = {"fit_partition": "train", "role": "reference",
                  "unk": {"by_partition": {
                      "train": {"tokens": 80000, "unk": 5, "rate": None},
                      "val": {"tokens": 10000, "unk": 40, "rate": 0.004},
                      "test": {"tokens": 10000, "unk": 60, "rate": 0.006}},
                      "overall": {"tokens": 100000, "unk": 105, "rate": 0.00105},
                      "non_fit": {"tokens": 20000, "unk": 100, "rate": 0.005},
                      "by_concept": {"position": 50, "code_status": 55}}}
        out = suppress_small_cells(copy.deepcopy(report))["unk"]
        self.assertEqual(out["non_fit"], report["unk"]["non_fit"])
        self.assertNotIsInstance(out["overall"]["unk"], int)
        self.assertIsNone(out["overall"]["rate"])
        # by_concept sums to the overall: one of its cells is withheld too.
        self.assertEqual(sum(isinstance(v, int) for v in out["by_concept"].values()), 1)
        # train = overall - val - test is not recoverable either.
        hidden = [p for p, e in out["by_partition"].items() if not isinstance(e["unk"], int)]
        self.assertGreaterEqual(len(hidden), 1)
        rate = gate_report({**report, "unk": out}, {"tables": {}})
        self.assertTrue(next(c for c in rate if c["check"] == "unk_rate_non_fit")["passed"])

    def test_the_threshold_comes_from_the_artifact_policy(self):
        from src.data.cohort import QualificationError
        from src.data.tokenization_report import (
            MIN_CELL,
            policy_min_cell,
            suppress_small_cells,
        )
        from src.eval.synthetic_bundle import FIXTURE_POLICY

        self.assertEqual(policy_min_cell(None), MIN_CELL)
        self.assertEqual(policy_min_cell(FIXTURE_POLICY), 10)
        policy = copy.deepcopy(FIXTURE_POLICY)
        policy["classes"]["aggregate_no_phi"]["minimum_cell_size"] = 25
        self.assertEqual(policy_min_cell(policy), 25)
        del policy["classes"]["aggregate_no_phi"]["minimum_cell_size"]
        with self.assertRaisesRegex(QualificationError, "minimum_cell_size"):
            policy_min_cell(policy)
        out = suppress_small_cells({"suppression": {"min_cell_size": 25},
                                    "sources": {"vitals": {"events": 24, "stays": 30}}})
        self.assertEqual(out["sources"]["vitals"], {"events": "<25", "stays": 30})


class PolicyThresholdReportTest(unittest.TestCase):
    """tokenize_site reads the report's suppression threshold from the artifact policy."""

    def test_tokenize_site_suppresses_at_the_policy_minimum_cell_size(self):
        try:
            from test_gem_artifact import _build_site
        except ImportError:  # pragma: no cover
            from tests.test_gem_artifact import _build_site
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import FIXTURE_POLICY, SYNTHETIC_SITE

        policy = copy.deepcopy(FIXTURE_POLICY)
        policy["classes"]["aggregate_no_phi"]["minimum_cell_size"] = 30
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            old_cwd = os.getcwd()
            os.chdir(work)
            try:
                site, episodes, cfg = _build_site(work)
                out = work / "output/intermediate_phi/policy30"
                tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, episodes=episodes,
                              artifact_policy=policy)
                report = json.loads((out / "tokenization_report.json").read_text())
            finally:
                os.chdir(old_cwd)
        self.assertEqual(report["suppression"]["min_cell_size"], 30)
        stays = report["sources"]["vitals"]["stays"]
        self.assertEqual(stays, "<30")                   # 24 synthetic stays


def _good_report() -> dict:
    return {
        "report_version": 1,
        "trajectory": "icu_24h",
        "role": "reference",
        "availability": {"vitals": {"availability": "missing_storetime", "lag_minutes": 0}},
        "numeric_bare_concepts": {"fit": [], "non_fit": []},
        "events": {"stays": 4000, "per_stay_mean": 900.0, "per_stay_p99": 4000.0,
                   "context": 8192},
        "unk": {"non_fit": {"tokens": 100000, "unk": 50, "rate": 0.0005}},
        "sources": {"vitals": {"events": 100000, "stays": 4000, "low_coverage": False}},
        "vocab": {"single_bin_concepts": []},
    }


class RealDataGateTest(unittest.TestCase):
    CFG = {"tables": {"vitals": {"availability": "missing_storetime"}}}

    def _failed(self, report, cfg=None):
        from src.data.tokenization_report import gate_report

        checks = gate_report(report, cfg or self.CFG)
        return {c["check"] for c in checks if not c["passed"]}

    def test_a_good_report_passes(self):
        self.assertEqual(self._failed(_good_report()), set())

    def test_a_table_without_availability_fails_the_gate(self):
        cfg = {"tables": {"vitals": {"availability": "missing_storetime"},
                          "labs": {"file": "clif_labs"}}}
        self.assertIn("availability_declared", self._failed(_good_report(), cfg))

    def test_a_report_missing_a_tables_availability_fails_the_gate(self):
        report = _good_report()
        report["availability"] = {}
        self.assertIn("availability_declared", self._failed(report))

    def test_bare_numeric_concepts_high_unk_and_overlong_stays_fail(self):
        report = _good_report()
        report["numeric_bare_concepts"]["fit"] = ["potassium"]
        report["unk"]["non_fit"]["rate"] = 0.05
        report["events"]["per_stay_p99"] = 9000.0
        self.assertTrue({"numeric_concepts_binned", "unk_rate_non_fit",
                         "events_per_stay_within_context"} <= self._failed(report))

    def test_unmeasured_unk_rate_or_unsuppressed_cells_fail(self):
        report = _good_report()
        report["unk"]["non_fit"] = {"tokens": "<10", "unk": "<10", "rate": None}
        report["sources"]["vitals"]["stays"] = 7
        self.assertTrue({"unk_rate_non_fit", "small_cells_suppressed"}
                        <= self._failed(report))

    def test_a_suppressed_unk_count_on_a_large_base_gates_on_its_upper_bound(self):
        report = _good_report()
        report["unk"]["non_fit"] = {"tokens": 2_000_000, "unk": "<10", "rate": None}
        self.assertNotIn("unk_rate_non_fit", self._failed(report))
        report["unk"]["non_fit"] = {"tokens": 500, "unk": "<10", "rate": None}
        self.assertIn("unk_rate_non_fit", self._failed(report))  # bound 9/500 > 0.01

    def test_gate_cli_exits_nonzero_on_failure(self):
        import yaml

        from src.data.tokenization_report import main

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tokenization_report.json"
            cfg_path = Path(td) / "data.yaml"
            cfg_path.write_text(yaml.safe_dump(self.CFG))
            path.write_text(json.dumps(_good_report()))
            self.assertEqual(main(["--report", str(path), "--config", str(cfg_path)]), 0)
            bad = _good_report()
            bad["availability"] = {}
            path.write_text(json.dumps(bad))
            self.assertEqual(main(["--report", str(path), "--config", str(cfg_path)]), 1)


# ---------------------------------------------------------------- report from tokenize_site

class TokenizationReportTest(unittest.TestCase):
    """The report the multi-source U4 fixture tokenization writes beside events.parquet."""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old_cwd = os.getcwd()
        os.chdir(work)
        try:
            cls.events, cls.blob, cls.stats, cls.episodes = _build_u4_site(work, "report")
            out = work / "output/intermediate_phi/u4_report"
            cls.report = json.loads((out / "tokenization_report.json").read_text())
            cls.raw_text = (out / "tokenization_report.json").read_text()
        finally:
            os.chdir(old_cwd)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_report_has_no_identifiers_in_keys_or_values(self):
        ids = {str(v) for col in ("hospitalization_id", "patient_id")
               for v in self.episodes[col].to_list()}
        for _, key, value in _walk(self.report):
            if key is not None:
                self.assertNotIn(key, IDENTIFIER_KEYS)
                self.assertNotIn(key, ids)
            if isinstance(value, str):
                self.assertNotIn(value, ids)
        for identifier in ids:
            self.assertNotIn(f'"{identifier}"', self.raw_text)

    def test_every_patient_derived_count_under_ten_is_suppressed(self):
        from src.data.tokenization_report import COUNT_SECTIONS, NOT_COUNTS

        for section in COUNT_SECTIONS:
            for path, key, value in _walk(self.report.get(section, {})):
                if key is None and isinstance(value, int) and not isinstance(value, bool):
                    if path and path[-1] in NOT_COUNTS:
                        continue
                    self.assertGreaterEqual(value, 10, (section, path))
        # Only one synthetic stay doses fentanyl before any weight is charted, so the
        # no-weight conversion count is a suppressed small cell.
        self.assertEqual(self.report["dose_conversion"]["meds"]["no_weight"], "<10")

    def test_report_contents(self):
        report = self.report
        self.assertEqual(report["trajectory"], "icu_24h")
        self.assertEqual(report["role"], "reference")
        cfg_tables = set(json.loads(json.dumps(self.blob["manifest"]["provenance"]
                                               ["availability"])))
        self.assertEqual(set(report["availability"]), cfg_tables)
        self.assertEqual(report["availability"]["vitals"]["availability"],
                         "missing_storetime")
        # Every configured source (plus static) has events/stays and a coverage flag.
        for table in [*cfg_tables, "static"]:
            entry = report["sources"][table]
            self.assertIn("events", entry)
            self.assertIn("stays", entry)
            self.assertIs(entry["low_coverage"], True)       # 24 stays < 50
        vocab = report["vocab"]
        self.assertEqual(vocab["size"], len(self.blob["vocab"]))
        self.assertEqual(sum(vocab["binning_sources"].values()), len(self.blob["segments"]))
        self.assertEqual(sorted(vocab["single_bin_concepts"]), sorted(
            c for c, s in self.blob["binning_sources"].items() if s == "single"))
        self.assertEqual(set(report["dose_conversion"]), {"meds", "meds_intermittent"})
        self.assertEqual(report["numeric_bare_concepts"]["fit"], [])
        for key in ("binned", "categorical", "presence", "numeric_bare",
                    "skipped_missing_numeric"):
            self.assertIn(key, report["token_kinds"])
        for key in ("inside", "gap", "clamp_low", "clamp_high"):
            self.assertIn(key, report["value_placement"]["total"])
        self.assertIn("non_fit", report["unk"])
        self.assertIn("overall", report["unk"])
        self.assertIn("per_stay_p99", report["events"])
        self.assertIsInstance(report["unit_mismatches"], list)
        self.assertEqual(report["binding"]["numeric_edges"],
                         self.blob["manifest"]["hashes"]["numeric_edges"])
        self.assertIs(report["sample"]["vocab_sample"], False)


# ---------------------------------------------------------------- verification sample

def _sample_site(work: Path):
    """24-stay synthetic site; four stays ineligible (resealed), four in validation."""
    import yaml

    from src.data.splits import content_manifest
    from src.eval.synthetic_bundle import (
        FIXTURE_COHORT,
        FIXTURE_DATA_CONFIG,
        FIXTURE_POLICY,
        build_synthetic_site,
    )

    site = work / "site"
    episodes = pl.read_parquet(build_synthetic_site(site))
    episodes = _repartition(episodes, [f"synth-{i:03d}" for i in range(20, 24)])
    ineligible = [f"synth-{i:03d}" for i in range(4)]       # the first four ids
    episodes = episodes.with_columns(
        pl.when(pl.col("hospitalization_id").is_in(ineligible)).then(False)
        .otherwise(pl.col("eligible")).alias("eligible"))
    eligible = episodes.filter(pl.col("eligible"))
    episodes = episodes.with_columns(
        pl.lit(content_manifest(eligible, columns=["hospitalization_id", "patient_id",
                                                   "partition"])["sha256"])
        .alias("split_sha256"),
        pl.lit(content_manifest(episodes, columns=["hospitalization_id", "patient_id",
                                                   "eligible", "partition"])["sha256"])
        .alias("episode_sha256"),
    )
    (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
    (work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))
    cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
    cfg["cohort_contract"] = str((work / "cohort.yaml").resolve())
    cfg["artifact_policy"] = str((work / "artifact_policy.yaml").resolve())
    return site, episodes, cfg, ineligible


class SampleEpisodesTest(unittest.TestCase):
    N = 10

    @classmethod
    def setUpClass(cls):
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import FIXTURE_POLICY, SYNTHETIC_SITE

        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old_cwd = os.getcwd()
        os.chdir(work)
        try:
            cls.site, cls.episodes, cls.cfg, cls.ineligible = _sample_site(work)
            cls.outs = []
            for run in ("a", "b"):
                out = work / f"output/intermediate_phi/sample_{run}"
                tokenize_site(cls.cfg, SYNTHETIC_SITE, cls.site, out, None,
                              episodes=cls.episodes, artifact_policy=FIXTURE_POLICY,
                              sample_episodes=cls.N)
                cls.outs.append(out)
            full = work / "output/intermediate_phi/full"
            tokenize_site(cls.cfg, SYNTHETIC_SITE, cls.site, full, None,
                          episodes=cls.episodes, artifact_policy=FIXTURE_POLICY)
            cls.full = full
            cls.work = work
        finally:
            os.chdir(old_cwd)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_sample_ids_are_eligible_deterministic_and_not_the_first_n(self):
        from src.data.tokenize import sample_episode_ids

        ids = sample_episode_ids(self.episodes, self.N)
        self.assertEqual(len(ids), self.N)
        self.assertEqual(len(set(ids)), self.N)
        eligible = set(self.episodes.filter(pl.col("eligible"))["hospitalization_id"])
        self.assertTrue(set(ids) <= eligible)
        self.assertFalse(set(ids) & set(self.ineligible))
        self.assertEqual(ids, sample_episode_ids(self.episodes.reverse(), self.N))
        first_eligible = sorted(eligible)[: self.N]
        self.assertNotEqual(sorted(ids), first_eligible)

    def test_sample_size_must_be_positive(self):
        from src.data.tokenize import sample_episode_ids

        for bad in (0, -1):
            with self.assertRaises(ValueError):
                sample_episode_ids(self.episodes, bad)

    def test_sampled_tokenization_covers_only_the_sample_and_is_deterministic(self):
        from src.data.tokenize import sample_episode_ids

        a, b = (pl.read_parquet(out / "events.parquet") for out in self.outs)
        self.assertEqual(set(a["hosp_id"]), set(sample_episode_ids(self.episodes, self.N)))
        self.assertTrue(a.equals(b))
        self.assertEqual((self.outs[0] / "vocab.json").read_bytes(),
                         (self.outs[1] / "vocab.json").read_bytes())

    def test_sample_vocab_is_marked_in_provenance_and_the_report(self):
        sample = json.loads((self.outs[0] / "vocab.json").read_text())
        full = json.loads((self.full / "vocab.json").read_text())
        self.assertIs(sample["manifest"]["provenance"]["sample"], True)
        self.assertEqual(sample["manifest"]["provenance"]["sample_size"], self.N)
        self.assertIs(full["manifest"]["provenance"]["sample"], False)
        self.assertIsNone(full["manifest"]["provenance"]["sample_size"])
        report = json.loads((self.outs[0] / "tokenization_report.json").read_text())
        self.assertIs(report["sample"]["vocab_sample"], True)
        self.assertIs(report["sample"]["run_sample"], True)

    def test_cli_flag_is_wired(self):
        from src.data.tokenize import build_arg_parser

        args = build_arg_parser().parse_args(
            ["--site", "s", "--in", "i", "--out", "o", "--episodes", "e", "--build-vocab",
             "--sample-episodes", "5000"])
        self.assertEqual(args.sample_episodes, 5000)

    def test_sample_and_limit_stays_are_mutually_exclusive(self):
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import FIXTURE_POLICY, SYNTHETIC_SITE

        old_cwd = os.getcwd()
        os.chdir(self.work)
        try:
            with self.assertRaisesRegex(ValueError, "sample"):
                tokenize_site(self.cfg, SYNTHETIC_SITE, self.site,
                              self.work / "output/intermediate_phi/both", None,
                              episodes=self.episodes, artifact_policy=FIXTURE_POLICY,
                              sample_episodes=3, limit_stays=3)
        finally:
            os.chdir(old_cwd)

    # ---- training refuses a sample vocabulary (dry-run allowed) ----------------------
    def _loaders(self, out: Path, *, dry_run: bool):
        from src.data.segments import artifact_binding
        from src.train.pretrain import build_loaders

        blob = json.loads((out / "vocab.json").read_text())
        tcfg = {"batch": {"per_gpu": 2}, "runtime": {"num_workers": 0}}
        mcfg = {"heads": {"competing_risk": {"n_time_bins": 4, "horizon_hours": 48}}}
        return build_loaders(out / "events.parquet", binding=artifact_binding(blob),
                             vocab_blob=blob, tcfg=tcfg, mcfg=mcfg, vocab_size=512,
                             dry_run=dry_run)

    def test_training_refuses_a_sample_vocab_but_dry_run_allows_it(self):
        with self.assertRaisesRegex(SystemExit, "sample"):
            self._loaders(self.outs[0], dry_run=False)
        loaders = self._loaders(self.outs[0], dry_run=True)
        self.assertGreater(len(loaders.train_dataset), 0)

    def test_a_full_vocab_is_not_refused_for_being_a_sample(self):
        with self.assertRaises(SystemExit) as caught:
            self._loaders(self.full, dry_run=False)
        # It still fails closed for the pre-existing reasons, never for "sample".
        self.assertNotIn("sample", str(caught.exception))

    def test_loader_refuses_a_vocab_that_does_not_match_the_binding(self):
        from src.data.segments import ArtifactBindingError, artifact_binding
        from src.train.pretrain import build_loaders

        sample = json.loads((self.outs[0] / "vocab.json").read_text())
        full = json.loads((self.full / "vocab.json").read_text())
        with self.assertRaises(ArtifactBindingError):
            build_loaders(self.full / "events.parquet", binding=artifact_binding(full),
                          vocab_blob=sample,
                          tcfg={"batch": {"per_gpu": 2}, "runtime": {"num_workers": 0}},
                          mcfg={"heads": {"competing_risk": {"n_time_bins": 4}}},
                          vocab_size=512, dry_run=True)


# ---------------------------------------------------------------- GEM report

class GemReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from test_gem_artifact import _build_site
        except ImportError:  # pragma: no cover
            from tests.test_gem_artifact import _build_site
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import FIXTURE_POLICY, SYNTHETIC_SITE

        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old_cwd = os.getcwd()
        os.chdir(work)
        try:
            site, episodes, cfg = _build_site(work)
            kw = {"episodes": episodes, "artifact_policy": FIXTURE_POLICY}
            out = work / "output/intermediate_phi/gem_report"
            tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, **kw)
            cls.icu_report = json.loads((out / "tokenization_report.json").read_text())
            blob = json.loads((out / "vocab.json").read_text())
            tokenize_site(cfg, SYNTHETIC_SITE, site, out, blob,
                          trajectory="hospitalization", max_tokens=64, **kw)
            cls.report = json.loads((out / "gem_tokenization_report.json").read_text())
            cls.icu_report_after = json.loads(
                (out / "tokenization_report.json").read_text())
            cls.ids = {str(v) for v in episodes["hospitalization_id"].to_list()}
        finally:
            os.chdir(old_cwd)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_gem_report_is_separate_and_aggregate_only(self):
        self.assertEqual(self.report["trajectory"], "hospitalization")
        self.assertEqual(self.icu_report, self.icu_report_after)
        for _, key, value in _walk(self.report):
            self.assertNotIn(key, IDENTIFIER_KEYS)
            if isinstance(value, str):
                self.assertNotIn(value, self.ids)

    def test_gem_report_has_windows_and_terminal_counts(self):
        gem = self.report["gem"]
        for key in ("stays", "windows", "windows_per_stay_mean", "tokens_per_stay_mean",
                    "tokens_per_stay_p99", "window_tokens_p99", "dispositions",
                    "admission_types"):
            self.assertIn(key, gem)
        self.assertLessEqual(gem["window_tokens_p99"], 64)
        self.assertEqual(self.report["gem"]["max_tokens"], 64)


def test_report_module_is_vendored():
    sync = (ROOT / "clif-validate/scripts/sync_vendor.py").read_text()
    assert '"src/data/tokenization_report.py"' in sync


class HarmonizationQualityTest(unittest.TestCase):
    """CLIF harmonization counts (GCS, BP method, duplicates, weights) in the report's
    data-quality section: shares per BP method, small cells withheld."""

    def test_bp_method_shares_and_suppression(self):
        from src.data.tokenization_report import _data_quality, suppress_small_cells

        quality = {"harmonization": {
            "bp_method": {"arterial": 300, "noninvasive_auto": 696, "noninvasive_manual": 4,
                          "tokens": 500},
            "gcs": {"verbal_not_testable": 120, "total_dropped": 118, "eye_motor_emitted": 110},
            "dose_tables": {"meds": {"exact_duplicates_removed": 40, "weights": 900,
                                     "weights_excluded": 3}}}}
        out = _data_quality(quality, 10)
        shares = out["harmonization"]["bp_method_shares"]
        self.assertAlmostEqual(shares["arterial"], 0.3)
        self.assertAlmostEqual(shares["noninvasive_auto"], 0.696)
        self.assertIsNone(shares["noninvasive_manual"])      # 4 readings: withheld
        report = suppress_small_cells({"suppression": {"min_cell_size": 10},
                                       "data_quality": out}, 10)
        harm = report["data_quality"]["harmonization"]
        self.assertEqual(harm["bp_method"]["noninvasive_manual"], "<10")
        # 4 manual readings would be total - others (total = arterial / its share): every
        # share is withheld and the smallest released count is withheld as its complement.
        self.assertEqual(set(harm["bp_method_shares"].values()), {None})
        self.assertEqual(harm["bp_method"]["arterial"], "suppressed")
        self.assertEqual(harm["bp_method"]["noninvasive_auto"], 696)
        self.assertEqual(harm["dose_tables"]["meds"]["weights_excluded"], "<10")
        self.assertEqual(harm["gcs"]["total_dropped"], 118)


class BinningShareLeakTest(unittest.TestCase):
    """A binning share times the binned total recovers its count (PR review)."""

    def report(self, sources):
        from src.data.tokenization_report import _rate
        total = sum(sources.values())
        return {"suppression": {"min_cell_size": 10},
                "token_kinds": {"binned": total, "categorical": 1000},
                "events": {"total": total + 1000},
                "binning": {"by_source": {
                    s: {"concepts": 1, "events": n, "event_share": _rate(n, total, 10)}
                    for s, n in sources.items()}}}

    def test_small_source_hides_every_share(self):
        from src.data.tokenization_report import suppress_small_cells
        out = suppress_small_cells(self.report({"csv": 5, "quantile": 20}), 10)
        rows = out["binning"]["by_source"]
        self.assertEqual(rows["csv"]["events"], "<10")
        self.assertEqual(rows["quantile"]["events"], "suppressed")
        self.assertIsNone(rows["csv"]["event_share"])
        self.assertIsNone(rows["quantile"]["event_share"])

    def test_shares_are_kept_when_nothing_is_suppressed(self):
        from src.data.tokenization_report import suppress_small_cells
        out = suppress_small_cells(self.report({"csv": 50, "quantile": 150}), 10)
        rows = out["binning"]["by_source"]
        self.assertAlmostEqual(rows["csv"]["event_share"], 0.25)
        self.assertAlmostEqual(rows["quantile"]["event_share"], 0.75)

    def test_bp_shares_kept_without_a_small_method(self):
        from src.data.tokenization_report import _data_quality, suppress_small_cells
        quality = _data_quality({"harmonization": {"bp_method": {
            "arterial": 300, "noninvasive_auto": 700, "tokens": 500}}}, 10)
        out = suppress_small_cells({"suppression": {"min_cell_size": 10},
                                    "data_quality": quality}, 10)
        harm = out["data_quality"]["harmonization"]
        self.assertAlmostEqual(harm["bp_method_shares"]["arterial"], 0.3)
        self.assertEqual(harm["bp_method"]["arterial"], 300)
