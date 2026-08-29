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
import time
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
        # Force a non-UTC session timezone so the DuckDB UTC pin is actually exercised
        # regardless of the host/CI timezone. Without this the "regression check" was a
        # claim, not a control: on a UTC runner the pin in tokenize.py could be deleted
        # and this suite would still pass (review finding). tzset() is POSIX-only, which
        # matches the package's stated platform.
        cls._old_tz = os.environ.get("TZ")
        os.environ["TZ"] = "America/Chicago"
        if hasattr(time, "tzset"):
            time.tzset()
        os.chdir(cls.work)  # the artifact policy classifies shards relative to CWD
        # Build under try/except: build_synthetic_* is fallible, and a raise after the
        # chdir would otherwise skip tearDownClass and leave the whole pytest session
        # in a temp dir that TemporaryDirectory then deletes (review finding).
        try:
            cls.site = cls.work / "site"
            cls.episodes = build_synthetic_site(cls.site)
            cls.bundle_dir = build_synthetic_bundle(cls.work / "bundle", cls.site,
                                                    cls.episodes)
        except BaseException:
            os.chdir(cls._old_cwd)
            if cls._old_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = cls._old_tz
            if hasattr(time, "tzset"):
                time.tzset()
            _reset_policy_pin()
            cls._tmp.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._old_cwd)
        if cls._old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = cls._old_tz
        if hasattr(time, "tzset"):
            time.tzset()
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

    def _reseal(self, d):
        m = json.loads((d / BUNDLE_MANIFEST).read_text())
        write_bundle_manifest(
            d, model_bundle_id=m["model_bundle_id"], model_version=m["model_version"],
            vocab_hash=m["vocab_hash"], outcome_spec_hash=m["outcome_spec_hash"],
            clif_version=m["clif_version"], outcome_queries=m["outcome_queries"],
        )

    def test_a_nested_manifest_named_file_does_not_escape_the_envelope(self):
        """An unlisted `sub/bundle_manifest.json` must be caught, not skipped by name."""
        broken = self._mutable_copy("bundle_nested_manifest")
        (broken / "sub").mkdir()
        (broken / "sub" / BUNDLE_MANIFEST).write_text('{"evil": 1}')
        with self.assertRaisesRegex(ArtifactMismatch, "does not cover"):
            load_bundle(broken)

    def test_a_symlink_in_the_bundle_is_refused(self):
        """A symlink could point outside the bundle or hide files from the hash map."""
        broken = self._mutable_copy("bundle_symlink")
        (broken / "link.json").symlink_to(broken / "vocab.json")
        with self.assertRaisesRegex(ArtifactMismatch, "symlink"):
            load_bundle(broken)

    def test_a_direction_outside_0_1_is_refused_at_load(self):
        """direction indexes an Embedding(2); reject 2 at load, not deep in torch."""
        broken = self._mutable_copy("bundle_dir2")
        manifest = json.loads((broken / BUNDLE_MANIFEST).read_text())
        manifest["outcome_queries"][SYNTHETIC_OUTCOME]["direction"] = 2
        (broken / BUNDLE_MANIFEST).write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ArtifactMismatch, "direction must be 0"):
            load_bundle(broken)

    def test_a_sql_unsafe_column_name_is_refused(self):
        """An untrusted table spec column that could break out of DuckDB SQL fails closed."""
        import yaml

        broken = self._mutable_copy("bundle_sqlcol")
        dc = yaml.safe_load((broken / "data_config.yaml").read_text())
        dc["tables"]["vitals"]["concept_col"] = "x FROM read_text('/etc/passwd') --"
        (broken / "data_config.yaml").write_text(yaml.safe_dump(dc))
        self._reseal(broken)
        with self.assertRaisesRegex(ArtifactMismatch, "SQL identifier"):
            load_bundle(broken)

    def test_a_traversal_file_name_is_refused(self):
        import yaml

        broken = self._mutable_copy("bundle_traversal")
        dc = yaml.safe_load((broken / "data_config.yaml").read_text())
        dc["tables"]["vitals"]["file"] = "../../etc/passwd"
        (broken / "data_config.yaml").write_text(yaml.safe_dump(dc))
        self._reseal(broken)
        with self.assertRaisesRegex(ArtifactMismatch, "unsafe file name"):
            load_bundle(broken)

    def test_a_mismatched_clif_version_is_refused(self):
        """A data config whose CLIF version disagrees with the manifest must fail closed.

        Two guards cover this in depth: validate_vocabulary_artifact checks the vocab
        manifest's clif_version against the data config, and load_bundle's own
        schema_version/clif_version cross-check backs it. This mutation trips the former
        first (QualificationError); the point of the test is that a version mismatch is
        refused, not which guard fires.
        """
        import yaml

        from src.data.cohort import QualificationError

        broken = self._mutable_copy("bundle_clifver")
        dc = yaml.safe_load((broken / "data_config.yaml").read_text())
        dc["schema_version"] = "3.0.0"  # manifest clif_version stays 2.1
        (broken / "data_config.yaml").write_text(yaml.safe_dump(dc))
        self._reseal(broken)
        with self.assertRaises((ArtifactMismatch, QualificationError)):
            load_bundle(broken)

    def test_a_bundle_rejected_late_does_not_leave_its_policy_pinned(self):
        """A bundle refused at an identity check must not lower the process floor."""
        before = os.environ.get(S.POLICY_OVERRIDE_ENV)
        broken = self._mutable_copy("bundle_pinreject")
        pol = broken / "artifact_policy.yaml"
        pol.write_text(pol.read_text().replace("minimum_cell_size: 10",
                                               "minimum_cell_size: 1"))
        # Re-seal the files map over the edited policy, then corrupt an IDENTITY field
        # (vocab_hash) so load_bundle fails AFTER the policy would previously have pinned.
        self._reseal(broken)
        manifest = json.loads((broken / BUNDLE_MANIFEST).read_text())
        manifest["vocab_hash"] = "0" * 64
        (broken / BUNDLE_MANIFEST).write_text(json.dumps(manifest))
        with self.assertRaises(ArtifactMismatch):
            load_bundle(broken)
        self.assertEqual(os.environ.get(S.POLICY_OVERRIDE_ENV), before)

    def test_pin_policy_false_leaves_process_state_untouched(self):
        """The opt-out branch a comparison/introspection caller uses."""
        before = os.environ.get(S.POLICY_OVERRIDE_ENV)
        load_bundle(self.bundle_dir, pin_policy=False)
        self.assertEqual(os.environ.get(S.POLICY_OVERRIDE_ENV), before)


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


class RepoCliEndToEndTest(_BundleFixtureCase):
    """The repo CLI itself, driven end to end against the fixture bundle.

    This is the plan's U9 Verification item verbatim: main()'s predict_fn — the
    seam that used to raise by design — now runs real zero-shot inference from
    the bundle and completes the approved release ceremony.
    """

    def test_main_runs_the_wired_inference_path_and_releases(self):
        from unittest import mock

        from src.eval import attestation as attest
        from src.eval.clif_validate import main

        signing_key = self.work / "cli_signing.key"
        signing_key.write_text("cd" * 32)
        access_key = self.work / "cli_access.key"
        access_key.write_bytes(b"repo-cli-chain-key")
        argv = [
            "clif_validate",
            "--checkpoint", str(self.bundle_dir),
            "--data", str(self.site),
            "--episode-artifact", str(self.episodes),
            "--site-id", "SYNTH-A",
            "--release-id", "rel-repo-cli",
            "--out", "output/final_no_phi/repo_cli.json",
            "--ledger", "output/intermediate_phi/repo_cli_ledger.jsonl",
            "--access-log", "output/intermediate_phi/repo_cli_access.jsonl",
            "--shard-dir", "output/intermediate_phi/repo_cli_shards",
            "--signing-key-file", str(signing_key),
            "--access-log-key-file", str(access_key),
            "--approved",
        ]
        with mock.patch("sys.argv", argv):
            main()
        out = Path("output/final_no_phi/repo_cli.json")
        payload = json.loads(out.read_text())
        self.assertEqual(payload["disclosure_status"], "reviewed_approved")
        self.assertEqual(payload["release_id"], "rel-repo-cli")
        self.assertIn(SYNTHETIC_OUTCOME, payload["outcomes"])
        self.assertTrue(
            attest.verify_report(payload, bytes.fromhex(signing_key.read_text()))
        )
        self.assertEqual(
            attest.confirmed_releases(Path("output/intermediate_phi/repo_cli_ledger.jsonl")),
            {"rel-repo-cli"},
        )


if __name__ == "__main__":
    unittest.main()
