"""The literature-grounded segment source (configs/literature_segments/*.yaml).

Precedence: csv -> literature -> ordinal -> quantile -> single. A fragment concept bins on
its published edges with the declared closure; a physician-CSV concept is never
overridden; the unit must equal the concept's reference unit; every edge needs a source.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.data.cohort import QualificationError
from src.data.segments import bin_index
from src.data.tokenize import ROOT, BINNING_SOURCES, build_segments, load_literature_segments

TARGETS = ["map", "lactate", "spo2", "respiratory_rate", "creatinine", "bilirubin_total",
           "platelet_count", "heart_rate", "sbp", "temp_c"]


def _source(supports, verified=True, ident="PMID:1"):
    return {"id": ident, "supports": supports, "quote": "q", "verified": verified}


FRAGMENT = {
    "group": "test", "reviewed": "2026-10-03", "status": "literature_proposed",
    "concepts": {
        "troponin_t": {"unit": "ng/L", "decision": "segments", "edges": [14.0, 52.0],
                       "closure": "left", "rationale": "r",
                       "sources": [_source([14.0]), _source([52.0], False, "DOI:x")]},
        "eosinophils_percent": {"unit": "%", "decision": "segments", "edges": [2.0],
                                "closure": "right", "rationale": "r",
                                "sources": [_source([2.0])]},
        "rass": {"unit": "points (-5 to +4)", "decision": "keep_ordinal",
                 "valid_range": [-5, 4], "clinical_cutoffs": ["<= -3 deep"],
                 "rationale": "r", "sources": [_source(["-5 to +4"])]},
        "pt": {"unit": "sec", "decision": "keep_quantile", "rationale": "no threshold",
               "sources": []},
        "heparin_u_hr": {"unit": "u/hr", "decision": "segments",
                         "edges": [500.0, 1000.0], "closure": "left", "rationale": "r",
                         "sources": [_source([500.0, 1000.0])]},
        "lactate": {"unit": "mmol/L", "decision": "segments", "edges": [3.0],
                    "closure": "left", "rationale": "r", "sources": [_source([3.0])]},
        "absent_lab": {"unit": "mg/dL", "decision": "segments", "edges": [1.0],
                       "closure": "left", "rationale": "r", "sources": [_source([1.0])]},
    },
}


def _fit() -> pl.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for v in rng.uniform(1, 200, 300):
        rows.append(("troponin_t", float(v), "ng/L", "labs"))
    for v in rng.uniform(0, 10, 300):
        rows.append(("eosinophils_percent", float(v), "%", "labs"))
    for v in rng.integers(-3, 2, 300):          # only 5 of the 10 RASS levels observed
        rows.append(("rass", float(v), None, "assessments"))
    for v in rng.integers(10, 14, 300):         # integer-valued: ordinal by data alone
        rows.append(("pt", float(v), "sec", "labs"))
    for v in rng.uniform(100, 2000, 300):
        rows.append(("heparin_u_hr", float(v), "u_hr", "meds"))
    rows += [("heparin_u_hr", 0.0, "u_hr", "meds")] * 20
    for v in rng.lognormal(0.3, 0.6, 300):
        rows.append(("lactate", float(v), "mmol/L", "labs"))
    return pl.DataFrame(rows, schema={"concept": pl.String, "value": pl.Float64,
                                      "unit": pl.String, "source": pl.String}, orient="row")


def _binning(directory: Path, **overrides) -> dict:
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    return {**cfg["value_binning"], "literature_source": str(directory), **overrides}


def _directions() -> dict:
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    return {t["name"]: t["direction"] for t in cfg["target_concepts"]}


class LiteratureSegmentsTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)
        self._write(FRAGMENT)

    def tearDown(self):
        self._td.cleanup()

    def _write(self, fragment, name="test.yaml"):
        (self.dir / name).write_text(yaml.safe_dump(fragment))

    def _build(self, fit=None, **overrides):
        record: dict = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            segments, sources = build_segments(
                _binning(self.dir, **overrides), _fit() if fit is None else fit, TARGETS,
                _directions(), literature_out=record)
        self.warnings = [str(w.message) for w in caught]
        return segments, sources, record

    def test_precedence_order(self):
        self.assertEqual(BINNING_SOURCES, ("csv", "literature", "ordinal", "quantile", "single"))

    def test_a_literature_concept_bins_on_its_edges_with_the_declared_closure(self):
        segments, sources, record = self._build()
        self.assertEqual(sources["troponin_t"], "literature")
        trop = segments["troponin_t"]
        self.assertEqual(len(trop), 3)
        # closure left: [14, 52) -> 14 goes up, 52 goes up.
        self.assertEqual(bin_index(13.99, trop), 0)
        self.assertEqual(bin_index(14.0, trop), 1)
        self.assertEqual(bin_index(52.0, trop), 2)
        eos = segments["eosinophils_percent"]
        # closure right: (-inf, 2] -> 2 stays in the lower bin.
        self.assertEqual(bin_index(2.0, eos), 0)
        self.assertEqual(bin_index(2.01, eos), 1)
        lit = record["concepts"]["troponin_t"]
        self.assertEqual(lit["edges"], [14.0, 52.0])
        self.assertEqual(lit["closure"], "left")
        self.assertEqual(lit["sources"], ["PMID:1", "DOI:x"])
        self.assertEqual((lit["verified"], lit["unverified"]), (1, 1))
        self.assertEqual(lit["unverified_only_edges"], [52.0])

    def test_a_csv_concept_named_in_a_fragment_is_ignored_with_a_warning(self):
        segments, sources, record = self._build()
        self.assertEqual(sources["lactate"], "csv")
        self.assertIn("lactate", record["ignored_csv"])
        self.assertTrue(any("lactate" in w and "CSV" in w for w in self.warnings))
        baseline, _ = build_segments(
            {k: v for k, v in _binning(self.dir).items() if k != "literature_source"},
            _fit(), TARGETS, _directions())
        self.assertEqual(segments["lactate"], baseline["lactate"])   # byte-identical

    def test_a_fragment_concept_absent_from_the_data_is_skipped_with_a_warning(self):
        segments, _, record = self._build()
        self.assertNotIn("absent_lab", segments)
        self.assertIn("absent_lab", record["absent"])
        self.assertTrue(any("absent_lab" in w for w in self.warnings))

    def test_a_unit_mismatch_is_refused(self):
        bad = copy.deepcopy(FRAGMENT)
        bad["concepts"]["troponin_t"]["unit"] = "ng/mL"
        self._write(bad)
        with self.assertRaisesRegex(QualificationError, "troponin_t.*ng/mL"):
            self._build()

    def test_unit_spelling_and_a_parenthetical_qualifier_are_not_mismatches(self):
        segments, sources, record = self._build()
        self.assertEqual(sources["heparin_u_hr"], "literature")     # u/hr vs charted u_hr
        self.assertEqual(record["concepts"]["rass"]["unit_check"], "assumed")

    def test_an_edge_without_a_source_is_refused(self):
        bad = copy.deepcopy(FRAGMENT)
        bad["concepts"]["troponin_t"]["edges"] = [14.0, 52.0, 140.0]
        self._write(bad)
        with self.assertRaisesRegex(QualificationError, "troponin_t.*140"):
            load_literature_segments(self.dir)

    def test_malformed_fragments_are_refused(self):
        for mutate in (
            lambda c: c["troponin_t"].__setitem__("edges", [52.0, 14.0]),
            lambda c: c["troponin_t"].__setitem__("closure", "both"),
            lambda c: c["troponin_t"].__setitem__("decision", "guess"),
            lambda c: c["troponin_t"].__setitem__("edges", []),
        ):
            bad = copy.deepcopy(FRAGMENT)
            mutate(bad["concepts"])
            self._write(bad)
            with self.subTest(), self.assertRaises(QualificationError):
                load_literature_segments(self.dir)
        self._write(FRAGMENT)
        self._write({"concepts": {"pt": FRAGMENT["concepts"]["pt"]}}, "dup.yaml")
        with self.assertRaisesRegex(QualificationError, "pt"):
            load_literature_segments(self.dir)

    def test_keep_ordinal_uses_every_level_of_the_scale(self):
        segments, sources, record = self._build()
        self.assertEqual(sources["rass"], "ordinal")
        self.assertEqual([s["lo"] for s in segments["rass"]],
                         [float(v) for v in range(-5, 5)])
        self.assertEqual(record["concepts"]["rass"]["clinical_cutoffs"], ["<= -3 deep"])

    def test_keep_quantile_overrides_the_ordinal_rule(self):
        segments, sources, record = self._build()
        self.assertEqual(sources["pt"], "quantile")
        self.assertEqual(record["concepts"]["pt"]["rationale"], "no threshold")

    def test_a_dose_concept_keeps_its_stop_bin_below_the_first_edge(self):
        segments, _, _ = self._build()
        heparin = segments["heparin_u_hr"]
        self.assertEqual(heparin[0], {"lo": 0.0, "hi": 0.0, "lo_closed": True,
                                      "hi_closed": True})
        self.assertEqual(bin_index(0.0, heparin), 0)
        self.assertEqual(bin_index(1.0, heparin), 1)        # running, never "stopped"
        self.assertEqual(bin_index(500.0, heparin), 2)
        self.assertEqual(len(heparin), 4)

    def test_fragment_change_changes_the_record_hash(self):
        from src.data.segments import json_sha256

        _, _, first = self._build()
        changed = copy.deepcopy(FRAGMENT)
        changed["concepts"]["pt"]["rationale"] = "edited"
        self._write(changed)
        _, _, second = self._build()
        self.assertNotEqual(json_sha256(first), json_sha256(second))
        self.assertNotEqual(first["fragments"], second["fragments"])

    def test_decile_arm_ignores_literature_but_matches_its_bin_count(self):
        report: dict = {}
        segments, sources = build_segments(
            _binning(self.dir, scheme="decile_ablation", matched_granularity=True),
            _fit(), TARGETS, _directions(), granularity=report)
        self.assertEqual(sources["troponin_t"], "quantile")
        self.assertEqual(len(segments["troponin_t"]), 3)            # literature count
        self.assertEqual(len(segments["heparin_u_hr"]), 4)
        self.assertNotIn("literature", set(sources.values()))


class LiteratureReportTest(unittest.TestCase):
    def test_report_lists_sources_shares_and_unverified_literature(self):
        from src.data.tokenization_report import _binning, _literature

        events = pl.DataFrame({"concept": ["troponin_t"] * 30 + ["lactate"] * 70,
                               "value": [1.0] * 100})
        record = {"fragments": {"labs.yaml": "x"}, "ignored_csv": ["lactate"], "absent": [],
                  "concepts": {"troponin_t": {"decision": "segments", "unverified": 1,
                                              "unverified_only_edges": [52.0]}}}
        sources = {"troponin_t": "literature", "lactate": "csv"}
        out = _binning(events, sources, 10)
        self.assertEqual(out["by_source"]["literature"],
                         {"concepts": 1, "events": 30, "event_share": 0.3})
        self.assertEqual(out["by_source"]["csv"]["event_share"], 0.7)
        lit = _literature(sources, record)
        self.assertEqual(lit["unverified_sources"], {"troponin_t": 1})
        self.assertEqual(lit["unverified_only_edges"], {"troponin_t": [52.0]})
        self.assertEqual(lit["concepts"], ["troponin_t"])


class RepoFragmentsTest(unittest.TestCase):
    """Every fragment under configs/literature_segments/ loads and validates."""

    def test_repo_fragments_load_and_validate(self):
        cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
        source = cfg["value_binning"]["literature_source"]
        loaded = load_literature_segments(ROOT / source)
        self.assertGreaterEqual(len(loaded["fragments"]), 2)
        self.assertIn("age_decile", loaded["concepts"])
        for concept, spec in loaded["concepts"].items():
            with self.subTest(concept=concept):
                if spec["decision"] == "segments":
                    self.assertEqual(spec["edges"], sorted(set(spec["edges"])))

    def test_age_uses_the_clinical_bands_not_quantiles(self):
        rng = np.random.default_rng(1)
        fit = pl.DataFrame({"concept": ["age_decile"] * 200,
                            "value": rng.integers(18, 95, 200).astype(float),
                            "source": ["static"] * 200})
        cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
        segments, sources = build_segments(cfg["value_binning"], fit, TARGETS, _directions(),
                                           quantile_concepts={"age_decile"})
        self.assertEqual(sources["age_decile"], "literature")
        self.assertEqual([s["hi"] for s in segments["age_decile"][:-1]],
                         [45.0, 55.0, 65.0, 75.0, 80.0])


if __name__ == "__main__":
    unittest.main()
