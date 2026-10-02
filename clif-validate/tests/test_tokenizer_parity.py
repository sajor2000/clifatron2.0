"""U5 (R13): the site package's vendored tokenizer and the repository tokenizer emit
byte-identical token streams for the same fixture, on both the reference (vocab-building)
path and the import (frozen-vocab) path a site actually runs."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import polars as pl
import yaml

from clif_validate._vendor.data import tokenize as vendored
from clif_validate._vendor.eval.synthetic_bundle import (
    FIXTURE_COHORT,
    FIXTURE_DATA_CONFIG,
    FIXTURE_POLICY,
    SYNTHETIC_SITE,
    build_synthetic_site,
)

try:
    from src.data import tokenize as repo
except ImportError:  # installed site: no repository checkout to compare against
    repo = None


@unittest.skipIf(repo is None, "repository checkout not present")
class TokenizerParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old = os.getcwd()
        os.chdir(work)
        try:
            site = work / "site"
            episodes = pl.read_parquet(build_synthetic_site(site))
            (work / "cohort.yaml").write_text(yaml.safe_dump(FIXTURE_COHORT))
            (work / "artifact_policy.yaml").write_text(yaml.safe_dump(FIXTURE_POLICY))
            cfg = copy.deepcopy(FIXTURE_DATA_CONFIG)
            cfg["cohort_contract"] = str((work / "cohort.yaml").resolve())
            cfg["artifact_policy"] = str((work / "artifact_policy.yaml").resolve())
            cls.out = {}
            for name, module in (("repo", repo), ("vendored", vendored)):
                build = Path(f"output/intermediate_phi/{name}_build")
                module.tokenize_site(copy.deepcopy(cfg), SYNTHETIC_SITE, site, build, None,
                                     episodes=episodes, artifact_policy=FIXTURE_POLICY)
                blob = json.loads((build / "vocab.json").read_text())
                imported = Path(f"output/intermediate_phi/{name}_import")
                module.tokenize_site(copy.deepcopy(cfg), "SITE-B", site, imported, blob,
                                     episodes=episodes, artifact_policy=FIXTURE_POLICY)
                cls.out[name] = {
                    "build_events": (build / "events.parquet").read_bytes(),
                    "vocab": (build / "vocab.json").read_bytes(),
                    "import_events": (imported / "events.parquet").read_bytes(),
                }
        finally:
            os.chdir(old)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def test_reference_build_streams_and_vocab_are_byte_identical(self):
        self.assertEqual(self.out["repo"]["vocab"], self.out["vendored"]["vocab"])
        self.assertEqual(self.out["repo"]["build_events"],
                         self.out["vendored"]["build_events"])

    def test_imported_vocab_streams_are_byte_identical(self):
        self.assertEqual(self.out["repo"]["import_events"],
                         self.out["vendored"]["import_events"])


if __name__ == "__main__":
    unittest.main()
