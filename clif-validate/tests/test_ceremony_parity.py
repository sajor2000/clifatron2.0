"""Ceremony parity (U9): the wheel's CLI carries U5's full operational surface.

Everything here drives the VENDORED code — `clif_validate.cli.main` and the
vendored fixture builder — through the same ceremony U5's in-repo tests pin:
draft/--approved two-step, replay rejection by release id, fail-closed access-log
key before anything publishes, and a signed, ledger-confirmed release.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from clif_validate._vendor.eval import attestation as attest
from clif_validate._vendor.eval import schema as schema
from clif_validate._vendor.eval.synthetic_bundle import (
    build_synthetic_bundle,
    build_synthetic_site,
)
from clif_validate.cli import main


class CeremonyParityTest(unittest.TestCase):
    """One synthetic site + bundle per class; each test runs the CLI in its own workdir slice."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls._tmp.name)
        cls._old_cwd = os.getcwd()
        os.chdir(cls.work)
        cls.site = cls.work / "site"
        cls.episodes = build_synthetic_site(cls.site)
        cls.bundle = build_synthetic_bundle(cls.work / "bundle", cls.site, cls.episodes)
        cls.signing_key = cls.work / "signing.key"
        cls.signing_key.write_text("ab" * 32)
        cls.access_key = cls.work / "access.key"
        cls.access_key.write_bytes(b"synthetic-chain-key-for-tests")

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._old_cwd)
        os.environ.pop(schema.POLICY_OVERRIDE_ENV, None)
        os.environ.pop("CLIF_ACCESS_LOG_KEY_FILE", None)
        schema.min_cell_size.cache_clear()
        schema.max_dropped_fraction.cache_clear()
        cls._tmp.cleanup()

    def tearDown(self):
        os.environ.pop(schema.POLICY_OVERRIDE_ENV, None)
        os.environ.pop("CLIF_ACCESS_LOG_KEY_FILE", None)
        schema.min_cell_size.cache_clear()
        schema.max_dropped_fraction.cache_clear()

    def _argv(self, tag: str, *extra: str) -> list[str]:
        return [
            "clif-validate",
            "--checkpoint", str(self.bundle),
            "--data", str(self.site),
            "--episode-artifact", str(self.episodes),
            "--site-id", "SYNTH-A",
            "--release-id", f"rel-{tag}",
            "--out", f"output/final_no_phi/{tag}.json",
            "--ledger", f"output/intermediate_phi/{tag}_ledger.jsonl",
            "--access-log", f"output/intermediate_phi/{tag}_access.jsonl",
            "--shard-dir", f"output/intermediate_phi/{tag}_shards",
            "--signing-key-file", str(self.signing_key),
            "--access-log-key-file", str(self.access_key),
            *extra,
        ]

    def test_draft_then_approved_two_step(self):
        """Unapproved run: a local draft, nothing released, nothing ledgered."""
        with mock.patch("sys.argv", self._argv("twostep")):
            main()
        out = Path("output/final_no_phi/twostep.json")
        draft_path = Path(str(out) + ".draft")
        self.assertTrue(draft_path.exists())
        self.assertFalse(out.exists())
        self.assertFalse(Path("output/intermediate_phi/twostep_ledger.jsonl").exists())
        draft = json.loads(draft_path.read_text())
        self.assertEqual(draft["disclosure_status"], schema.DRAFT_DISCLOSURE_STATUS)
        self.assertNotIn("signature", draft)

        # Approved run: released, signed over the FINAL status, ledger confirmed.
        with mock.patch("sys.argv", self._argv("twostep", "--approved")):
            main()
        self.assertTrue(out.exists())
        payload = json.loads(out.read_text())
        self.assertEqual(payload["disclosure_status"], "reviewed_approved")
        self.assertTrue(
            attest.verify_report(payload, bytes.fromhex(self.signing_key.read_text()))
        )
        ledger = Path("output/intermediate_phi/twostep_ledger.jsonl")
        self.assertEqual(attest.confirmed_releases(ledger), {"rel-twostep"})
        self.assertEqual(attest.unconfirmed_releases(ledger), set())
        # Write-ahead access record exists for the export.
        access = Path("output/intermediate_phi/twostep_access.jsonl")
        self.assertTrue(access.exists())
        self.assertTrue(attest.verify_access_log(access))

    def test_a_replayed_release_id_is_rejected(self):
        with mock.patch("sys.argv", self._argv("replay", "--approved")):
            main()
        out = Path("output/final_no_phi/replay.json")
        first = out.read_bytes()
        with mock.patch("sys.argv", self._argv("replay", "--approved")):
            with self.assertRaises(schema.DisclosureError):
                main()
        self.assertEqual(out.read_bytes(), first)  # nothing overwrote the release

    def test_a_missing_access_log_key_fails_before_anything_publishes(self):
        argv = self._argv("nokey", "--approved")
        # Strip the key flag and make sure no ambient variable leaks in.
        idx = argv.index("--access-log-key-file")
        del argv[idx:idx + 2]
        os.environ.pop("CLIF_ACCESS_LOG_KEY_FILE", None)
        with mock.patch("sys.argv", argv):
            with self.assertRaises(attest.AuthenticationError):
                main()
        self.assertFalse(Path("output/final_no_phi/nokey.json").exists())
        self.assertFalse(Path("output/final_no_phi/nokey.json.draft").exists())
        self.assertFalse(Path("output/intermediate_phi/nokey_ledger.jsonl").exists())

    def test_a_shard_dir_outside_the_policy_class_is_refused(self):
        with mock.patch("sys.argv",
                        self._argv("badshard", "--approved", "--shard-dir",
                                   "somewhere_unclassified/shards")):
            with self.assertRaisesRegex(ValueError, "must be stored under"):
                main()
        self.assertFalse(Path("output/final_no_phi/badshard.json").exists())


if __name__ == "__main__":
    unittest.main()
