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
    SYNTHETIC_KEY_ID,
    SYNTHETIC_OUTCOME,
    build_synthetic_bundle,
    build_synthetic_site,
)


def _reset_policy_pin():
    os.environ.pop(S.POLICY_OVERRIDE_ENV, None)
    S.min_cell_size.cache_clear()
    S.max_dropped_fraction.cache_clear()


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
            # The signed fixture writes its trust root + releaser key beside the bundle.
            cls.trust_roles = cls.work / "trust_roles.yaml"
            cls.releaser_key = bytes.fromhex((cls.work / "releaser.key").read_text())
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
        b = load_bundle(self.bundle_dir, trust_roles_path=self.trust_roles)
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
        write_bundle_manifest(  # re-seal (and re-sign) over the edited policy
            pinned,
            model_bundle_id=manifest["model_bundle_id"],
            model_version=manifest["model_version"],
            vocab_hash=manifest["vocab_hash"],
            outcome_spec_hash=manifest["outcome_spec_hash"],
            clif_version=manifest["clif_version"],
            outcome_queries=manifest["outcome_queries"],
            signing_key=self.releaser_key,
            key_id=SYNTHETIC_KEY_ID,
        )
        load_bundle(pinned, trust_roles_path=self.trust_roles)
        self.assertEqual(S.min_cell_size(), 17)
        self.assertEqual(S.curve_release_min(), 85)

    def _reseal(self, d):
        m = json.loads((d / BUNDLE_MANIFEST).read_text())
        write_bundle_manifest(
            d, model_bundle_id=m["model_bundle_id"], model_version=m["model_version"],
            vocab_hash=m["vocab_hash"], outcome_spec_hash=m["outcome_spec_hash"],
            clif_version=m["clif_version"], outcome_queries=m["outcome_queries"],
        )

    def test_an_unsigned_reseal_removes_a_stale_signature(self):
        """A re-seal without a signing key must delete any prior .sig, so the bundle never
        carries a signature that matches no manifest (CodeRabbit)."""
        from src.eval.trust import SIGNATURE_FILENAME
        d = self._mutable_copy("bundle_reseal_sig")
        self.assertTrue((d / SIGNATURE_FILENAME).exists())  # fixture ships signed
        self._reseal(d)
        self.assertFalse((d / SIGNATURE_FILENAME).exists())
        # And a governed load now fails as unsigned, not with a confusing signed_by error.
        with self.assertRaisesRegex(ArtifactMismatch, "unsigned"):
            load_bundle(d, trust_roles_path=self.trust_roles)

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
        load_bundle(self.bundle_dir, pin_policy=False,
                    trust_roles_path=self.trust_roles)
        self.assertEqual(os.environ.get(S.POLICY_OVERRIDE_ENV), before)


class BundleInferenceTest(_BundleFixtureCase):
    """The wired path: bundle → tokenize → zero-shot inference → schema'd cell."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from src.eval.clif_validate import load_checkpoint

        cls.bundle = load_bundle(cls.bundle_dir, trust_roles_path=cls.trust_roles)
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

    def _dropping_predict_fn(self, cfgs, n_pos_drop, n_neg_drop, shard):
        """Wrap real inference, then NaN out a controlled count from each class.

        Class-balanced drops keep the survivors evaluable so the test isolates the
        coverage gate / banding, not a suppression side effect.
        """
        base = self._predict_fn(cfgs, shard=shard)
        name = cfgs[0]["name"]

        def pf(labels_df):
            probs = base(labels_df)
            status = labels_df[f"{name}_status"].to_list()
            pos = [i for i, s in enumerate(status) if s == "positive"]
            neg = [i for i, s in enumerate(status) if s == "negative"]
            for i in pos[:n_pos_drop] + neg[:n_neg_drop]:
                probs[i, 0] = np.nan
            return probs

        return pf

    def _evaluate(self, cfgs, predict_fn):
        from src.eval.clif_validate import evaluate_site

        return evaluate_site(
            str(self.bundle_dir), str(self.site), str(self.episodes), cfgs,
            predict_fn=predict_fn, cohort_config=self.bundle.cohort_path,
            data_config=self.bundle.data_cfg_path,
        )["outcomes"][cfgs[0]["name"]]

    def test_a_sub_floor_dropped_count_is_banded_not_released_exactly(self):
        """An exact small count of untokenizable stays is a numerator disclosure."""
        cfgs = [{"name": SYNTHETIC_OUTCOME}]
        # 24 stays (12/12). Drop 2 per class: fraction 4/24=0.167 <= 0.2 (evaluable),
        # survivors 10/10 clear the floor, dropped count 4 is in (0, 10) -> banded.
        cell = self._evaluate(cfgs, self._dropping_predict_fn(cfgs, 2, 2, "shards_band"))
        self.assertEqual(cell["status"], S.EVALUABLE)
        self.assertEqual(cell["metrics"]["n_dropped_nan"], f"<{S.min_cell_size()}")
        self.assertNotIn("4", json.dumps(cell["metrics"]["n_dropped_nan"]))

    def test_near_total_tokenization_failure_is_coverage_insufficient(self):
        """A biased sliver must not release as a score; the PORTER vocab-drop scenario."""
        cfgs = [{"name": SYNTHETIC_OUTCOME}]
        # Drop 6/24 = 0.25 > 0.2: the coverage gate fires before suppression, so the
        # outcome is coverage_insufficient rather than a metric on the survivors.
        cell = self._evaluate(cfgs, self._dropping_predict_fn(cfgs, 3, 3, "shards_cov"))
        self.assertEqual(cell["status"], S.COVERAGE_INSUFFICIENT)
        self.assertNotIn("metrics", cell)  # non-evaluable carries no numbers
        # And the reason must not leak an exact sub-floor count either.
        self.assertNotIn("6", cell["reason"])


class RepoCliEndToEndTest(_BundleFixtureCase):
    """The repo CLI itself, driven end to end against the fixture bundle.

    This is the plan's U9 Verification item verbatim: main()'s predict_fn — the
    seam that used to raise by design — now runs real zero-shot inference from
    the bundle and completes the approved release ceremony.
    """

    def _cli_argv(self, *extra):
        signing_key = self.work / "cli_signing.key"
        signing_key.write_text("cd" * 32)
        access_key = self.work / "cli_access.key"
        access_key.write_bytes(b"repo-cli-chain-key")
        self._cli_signing_key = signing_key
        return [
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
            "--trust-roles", str(self.trust_roles),
            "--rollback-state", "output/intermediate_phi/repo_cli_rollback.json",
            *extra,
        ]

    def test_main_runs_the_wired_inference_path_and_releases(self):
        """The full U9 ceremony under U11's trust + content-hash gates: verify the
        releaser signature, write a draft, then release exactly the reviewed draft."""
        from unittest import mock

        from src.eval import attestation as attest
        from src.eval.clif_validate import main

        # Step 1: the draft run (no --approved) writes the draft + its content hash.
        with mock.patch("sys.argv", self._cli_argv()):
            main()
        draft_hash = Path("output/final_no_phi/repo_cli.json.draft.sha256").read_text().strip()
        self.assertEqual(len(draft_hash), 64)

        # Step 2: approve THAT draft by its hash — the release is bound to reviewed content.
        with mock.patch("sys.argv", self._cli_argv("--approved", "--approved-hash", draft_hash)):
            main()
        out = Path("output/final_no_phi/repo_cli.json")
        payload = json.loads(out.read_text())
        self.assertEqual(payload["disclosure_status"], "reviewed_approved")
        self.assertEqual(payload["release_id"], "rel-repo-cli")
        self.assertIn(SYNTHETIC_OUTCOME, payload["outcomes"])
        self.assertTrue(
            attest.verify_report(payload, bytes.fromhex(self._cli_signing_key.read_text()))
        )
        self.assertEqual(
            attest.confirmed_releases(Path("output/intermediate_phi/repo_cli_ledger.jsonl")),
            {"rel-repo-cli"},
        )

    def test_approved_with_a_wrong_content_hash_fails_closed(self):
        """A release whose payload does not match the approved draft hash is refused."""
        from unittest import mock

        from src.eval.clif_validate import main

        argv = self._cli_argv("--approved", "--approved-hash", "0" * 64)
        with mock.patch("sys.argv", argv):
            with self.assertRaises(SystemExit):
                main()
        self.assertFalse(Path("output/final_no_phi/repo_cli.json").exists())

    def test_approved_without_a_content_hash_fails_closed(self):
        """--approved must bind to reviewed content — there is no waiver (U11 review)."""
        from unittest import mock

        from src.eval.clif_validate import main

        with mock.patch("sys.argv", self._cli_argv("--approved")):
            with self.assertRaises(SystemExit):
                main()
        self.assertFalse(Path("output/final_no_phi/repo_cli.json").exists())

    def test_approved_without_a_rollback_state_fails_closed(self):
        """A governed release must enforce anti-rollback; omitting the state path is refused."""
        from unittest import mock

        from src.eval.clif_validate import main

        argv = [a for a in self._cli_argv("--approved", "--approved-hash", "0" * 64)
                if a != "output/intermediate_phi/repo_cli_rollback.json"]
        argv = [a for a in argv if a != "--rollback-state"]
        with mock.patch("sys.argv", argv):
            with self.assertRaises(SystemExit):
                main()
        self.assertFalse(Path("output/final_no_phi/repo_cli.json").exists())

    def test_allow_unsigned_cannot_produce_an_approved_release(self):
        """The load-bearing bypass (U11 review): --allow-unsigned + --approved is refused,
        so a not-for-release (unverified) bundle can never be published as reviewed_approved."""
        from unittest import mock

        from src.eval.clif_validate import main

        argv = self._cli_argv("--approved", "--approved-hash", "0" * 64, "--allow-unsigned")
        with mock.patch("sys.argv", argv):
            with self.assertRaises(SystemExit):
                main()
        self.assertFalse(Path("output/final_no_phi/repo_cli.json").exists())
        self.assertFalse(Path("output/intermediate_phi/repo_cli_ledger.jsonl").exists())

    def test_a_governed_load_without_a_trust_root_fails_closed(self):
        """Omitting --trust-roles (and --allow-unsigned) refuses to load — no anchor."""
        from unittest import mock

        from src.eval.clif_validate import main

        argv = [a for a in self._cli_argv() if a not in ("--trust-roles",)]
        argv = [a for a in argv if a != str(self.trust_roles)]
        with mock.patch("sys.argv", argv):
            with self.assertRaises(ArtifactMismatch):
                main()


if __name__ == "__main__":
    unittest.main()
