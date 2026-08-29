"""U9: bundle contract, synthetic fixture, and the wired inference path.

The expensive artifacts (synthetic site, sealed bundle, tiny checkpoint) are built
once per class; mutation tests operate on copies so the pristine bundle stays
pristine. Every fail-closed guard here has its red case exercised — a guard whose
failure mode is never demonstrated is a docstring, not a control.

Note on timezones: this suite runs on a non-UTC host (America/Chicago), which is
itself a regression check — DuckDB used to render TIMESTAMPTZ in the session
timezone, so `tokenize_site` refused every tz-aware parquet outside UTC until the
session pin landed with U9.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.eval import schema as S
from src.eval.bundle import (
    BUNDLE_MANIFEST,
    load_bundle,
    verify_bundle_files,
    write_bundle_manifest,
)
from src.eval.clif_validate import ArtifactMismatch
from src.eval.synthetic_bundle import (
    SYNTHETIC_OUTCOME,
    build_synthetic_bundle,
    build_synthetic_site,
)


def _reset_policy_pin():
    os.environ.pop(S.POLICY_OVERRIDE_ENV, None)
    S.min_cell_size.cache_clear()


class _BundleFixtureCase(unittest.TestCase):
    """Shared once-per-class synthetic site + sealed bundle, built under a scratch CWD."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls._tmp.name)
        cls._old_cwd = os.getcwd()
        os.chdir(cls.work)  # the artifact policy classifies shards relative to CWD
        cls.site = cls.work / "site"
        cls.episodes = build_synthetic_site(cls.site)
        cls.bundle_dir = build_synthetic_bundle(cls.work / "bundle", cls.site,
                                                cls.episodes)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._old_cwd)
        _reset_policy_pin()
        cls._tmp.cleanup()

    def tearDown(self):
        # load_bundle pins the policy process-wide; never leak it into other tests.
        _reset_policy_pin()

    def _mutable_copy(self, name: str) -> Path:
        dst = self.work / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.bundle_dir, dst)
        return dst


class BundleContractTest(_BundleFixtureCase):
    def test_load_bundle_happy_path(self):
        b = load_bundle(self.bundle_dir)
        self.assertEqual(b.provenance["model_bundle_id"], "synthetic-fixture")
        self.assertEqual(b.provenance["clif_version"], "2.1")
        self.assertIn(SYNTHETIC_OUTCOME, b.outcome_queries)
        self.assertIn("map", b.edges)
        # config paths were rewritten to bundled absolutes — nothing repo-relative left
        self.assertTrue(Path(b.data_cfg["cohort_contract"]).is_absolute())
        self.assertTrue(Path(b.data_cfg["cohort_contract"]).exists())
        self.assertTrue(Path(b.data_cfg["artifact_policy"]).is_absolute())

    def test_a_single_flipped_byte_fails_red(self):
        """Prove the guard guards: mutate one byte, watch the hash check refuse."""
        broken = self._mutable_copy("bundle_bitflip")
        target = broken / "vocab.json"
        raw = bytearray(target.read_bytes())
        raw[len(raw) // 2] ^= 0x01
        target.write_bytes(bytes(raw))
        with self.assertRaisesRegex(ArtifactMismatch, "hashes do not match"):
            load_bundle(broken)

    def test_an_unlisted_extra_file_fails(self):
        """A file the manifest never covered is a tamper channel, not a passenger."""
        broken = self._mutable_copy("bundle_extra")
        (broken / "extra_policy.yaml").write_text("minimum_cell_size: 1\n")
        with self.assertRaisesRegex(ArtifactMismatch, "does not cover"):
            load_bundle(broken)

    def test_a_missing_hashed_file_fails(self):
        broken = self._mutable_copy("bundle_missing")
        (broken / "data_config.yaml").unlink()
        with self.assertRaisesRegex(ArtifactMismatch, "missing hashed files"):
            load_bundle(broken)

    def test_a_manifest_without_file_hashes_fails(self):
        broken = self._mutable_copy("bundle_nofiles")
        manifest_path = broken / BUNDLE_MANIFEST
        manifest = json.loads(manifest_path.read_text())
        del manifest["files"]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ArtifactMismatch, "declares no file hashes"):
            load_bundle(broken)

    def test_manifest_identity_must_match_the_shipped_vocabulary(self):
        """The headline vocab_hash cannot attest to a vocabulary the bundle lacks."""
        broken = self._mutable_copy("bundle_identity")
        manifest_path = broken / BUNDLE_MANIFEST
        manifest = json.loads(manifest_path.read_text())
        manifest["vocab_hash"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        # File hashes still pass (the manifest itself is not in its own map), so
        # only the identity cross-check stands between this bundle and a report.
        with self.assertRaisesRegex(ArtifactMismatch, "different vocabulary"):
            load_bundle(broken)

    def test_verify_bundle_files_requires_the_named_roles(self):
        manifest = json.loads((self.bundle_dir / BUNDLE_MANIFEST).read_text())
        del manifest["files"]["vocab.json"]
        with self.assertRaisesRegex(ArtifactMismatch, "vocab.json"):
            verify_bundle_files(self.bundle_dir, manifest)

    def test_load_bundle_pins_the_bundled_policy(self):
        """min_cell_size() must read the BUNDLE's floor after loading, not a checkout's."""
        pinned = self._mutable_copy("bundle_pin17")
        policy_path = pinned / "artifact_policy.yaml"
        policy_path.write_text(
            policy_path.read_text().replace("minimum_cell_size: 10",
                                            "minimum_cell_size: 17")
        )
        manifest = json.loads((pinned / BUNDLE_MANIFEST).read_text())
        write_bundle_manifest(  # re-seal over the edited policy
            pinned,
            model_bundle_id=manifest["model_bundle_id"],
            model_version=manifest["model_version"],
            vocab_hash=manifest["vocab_hash"],
            outcome_spec_hash=manifest["outcome_spec_hash"],
            clif_version=manifest["clif_version"],
            outcome_queries=manifest["outcome_queries"],
        )
        load_bundle(pinned)
        self.assertEqual(S.min_cell_size(), 17)
        self.assertEqual(S.curve_release_min(), 85)


class BundleInferenceTest(_BundleFixtureCase):
    """The wired path: bundle → tokenize → zero-shot inference → schema'd cell."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from src.eval.clif_validate import load_checkpoint

        cls.bundle = load_bundle(cls.bundle_dir)
        cls.model = load_checkpoint(str(cls.bundle_dir))

    def _predict_fn(self, outcome_cfgs, shard="shards_default"):
        from src.eval.bundle_inference import bundle_predict_fn

        return bundle_predict_fn(
            self.bundle, self.model, data_path=self.site,
            episode_artifact=self.episodes, outcome_cfgs=outcome_cfgs,
            shard_dir=Path("output/intermediate_phi") / shard, site_id="SYNTH-A",
        )

    def test_an_undeclared_outcome_refuses_to_improvise_a_query(self):
        with self.assertRaisesRegex(ArtifactMismatch, "no zero-shot query"):
            self._predict_fn([{"name": "made_up_outcome"}])

    def test_end_to_end_evaluation_yields_an_evaluable_aggregate_cell(self):
        from src.eval.clif_validate import evaluate_site

        cfgs = [{"name": SYNTHETIC_OUTCOME}]
        result = evaluate_site(
            str(self.bundle_dir), str(self.site), str(self.episodes), cfgs,
            predict_fn=self._predict_fn(cfgs, shard="shards_e2e"),
            cohort_config=self.bundle.cohort_path,
            data_config=self.bundle.data_cfg_path,
        )
        cell = result["outcomes"][SYNTHETIC_OUTCOME]
        self.assertEqual(cell["status"], S.EVALUABLE)
        metrics = cell["metrics"]
        self.assertEqual(metrics["n"], 24)
        self.assertEqual(metrics["n_dropped_nan"], 0)
        self.assertTrue(0.0 <= metrics["auroc"] <= 1.0)
        # Aggregate-only: no per-stay identifier anywhere in the serialized cell.
        dumped = json.dumps(cell)
        self.assertNotIn("synth-0", dumped)
        self.assertNotIn("hospitalization_id", dumped)

    def test_a_stay_the_tokenizer_never_saw_gets_nan_not_a_guess(self):
        import polars as pl

        cfgs = [{"name": SYNTHETIC_OUTCOME}]
        predict_fn = self._predict_fn(cfgs, shard="shards_nan")
        labels = pl.DataFrame({
            "hospitalization_id": ["synth-000", "never-tokenized-stay"],
        })
        probs = predict_fn(labels)
        self.assertEqual(probs.shape, (2, 1))
        self.assertTrue(np.isfinite(probs[0, 0]))
        self.assertTrue(np.isnan(probs[1, 0]))


if __name__ == "__main__":
    unittest.main()
