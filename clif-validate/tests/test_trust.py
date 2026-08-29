"""U11 fail-closed load integration, driving the VENDORED code end to end.

Where clif-validate/tests/test_trust.py's sibling in the repo (tests/test_trust.py)
unit-tests the trust primitives, this exercises the whole `load_bundle` boundary on a
real signed synthetic bundle: an absent, untrusted, revoked, tampered, or rolled-back
signature each fails closed; a governed load with no trust root refuses; and the
synthetic escape hatch loads unsigned but marks the bundle not-for-release.
"""

import os
import tempfile
import unittest
from pathlib import Path

import yaml

from clif_validate._vendor.eval import bundle as v_bundle
from clif_validate._vendor.eval import trust as v_trust
from clif_validate._vendor.eval.clif_validate import ArtifactMismatch
from clif_validate._vendor.eval.synthetic_bundle import (
    SYNTHETIC_KEY_ID,
    build_synthetic_bundle,
    build_synthetic_site,
)


class LoadBundleTrustTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls._tmp.name)
        cls._old_cwd = os.getcwd()
        os.chdir(cls.work)
        try:
            cls.site = cls.work / "site"
            cls.episodes = build_synthetic_site(cls.site)
            cls.bundle_dir = build_synthetic_bundle(cls.work / "bundle", cls.site,
                                                    cls.episodes)
            cls.trust_roles = cls.work / "trust_roles.yaml"
            root = yaml.safe_load(cls.trust_roles.read_text())["release_signing"]
            cls.public_key_hex = root["trusted_keys"][0]["public_key_hex"]
        except BaseException:
            os.chdir(cls._old_cwd)
            cls._tmp.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._old_cwd)
        cls._reset_pins()
        cls._tmp.cleanup()

    @staticmethod
    def _reset_pins():
        from clif_validate._vendor.eval import schema as v_schema
        os.environ.pop(v_schema.POLICY_OVERRIDE_ENV, None)
        v_schema.min_cell_size.cache_clear()
        v_schema.max_dropped_fraction.cache_clear()

    def tearDown(self):
        self._reset_pins()

    def _copy(self, name: str) -> Path:
        import shutil
        dst = self.work / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(self.bundle_dir, dst)
        return dst

    def _trust_root(self, name: str, *, key_id=SYNTHETIC_KEY_ID, public_key_hex=None,
                    revoked_keys=(), revoked_bundles=()) -> Path:
        path = self.work / name
        path.write_text(yaml.safe_dump({
            "release_signing": {
                "trusted_keys": [{
                    "key_id": key_id,
                    "public_key_hex": public_key_hex or self.public_key_hex,
                }],
                "revoked_key_ids": list(revoked_keys),
                "revoked_bundle_ids": list(revoked_bundles),
            }
        }))
        return path

    # -------------------------------------------------------------- happy path
    def test_a_signed_bundle_loads_against_its_trust_root(self):
        b = v_bundle.load_bundle(self.bundle_dir, trust_roles_path=self.trust_roles)
        self.assertTrue(b.for_release)

    # -------------------------------------------------------------- fail closed
    def test_a_governed_load_with_no_trust_root_is_refused(self):
        with self.assertRaises(ArtifactMismatch):
            v_bundle.load_bundle(self.bundle_dir)  # verify_signature default, no root

    def test_an_unsigned_bundle_is_refused(self):
        unsigned = self._copy("unsigned")
        (unsigned / v_trust.SIGNATURE_FILENAME).unlink()
        with self.assertRaisesRegex(ArtifactMismatch, "unsigned"):
            v_bundle.load_bundle(unsigned, trust_roles_path=self.trust_roles)

    def test_a_signature_by_an_untrusted_key_is_refused(self):
        """The manifest is signed by SYNTHETIC_KEY_ID; a root that trusts a different id."""
        root = self._trust_root("otherkey.yaml", key_id="some-other-releaser")
        with self.assertRaisesRegex(ArtifactMismatch, "not in the trust root"):
            v_bundle.load_bundle(self.bundle_dir, trust_roles_path=root)

    def test_a_wrong_public_key_for_the_signer_is_refused(self):
        """Same key_id, wrong public key: the signature cannot verify."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        other_pub = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
        root = self._trust_root("wrongpub.yaml", public_key_hex=other_pub)
        with self.assertRaisesRegex(ArtifactMismatch, "does not verify"):
            v_bundle.load_bundle(self.bundle_dir, trust_roles_path=root)

    def test_a_revoked_key_is_refused(self):
        root = self._trust_root("revokedkey.yaml", revoked_keys=[SYNTHETIC_KEY_ID])
        with self.assertRaisesRegex(ArtifactMismatch, "revoked"):
            v_bundle.load_bundle(self.bundle_dir, trust_roles_path=root)

    def test_a_revoked_bundle_is_refused(self):
        manifest = self.bundle_dir / v_bundle.BUNDLE_MANIFEST
        import json
        bundle_id = json.loads(manifest.read_text())["model_bundle_id"]
        root = self._trust_root("revokedbundle.yaml", revoked_bundles=[bundle_id])
        with self.assertRaisesRegex(ArtifactMismatch, "revoked"):
            v_bundle.load_bundle(self.bundle_dir, trust_roles_path=root)

    def test_rewriting_signed_by_to_an_unknown_key_is_refused(self):
        """signed_by is not in the signed subset, so an attacker can rewrite it — but the
        rewritten id must resolve in the trust root, and a bogus/alias id fails closed
        (U11 review; belt-and-suspenders with the trust root's duplicate-key refusal)."""
        import json
        tampered = self._copy("swapped_signer")
        mpath = tampered / v_bundle.BUNDLE_MANIFEST
        m = json.loads(mpath.read_text())
        m["signed_by"] = "some-unknown-releaser"
        mpath.write_text(json.dumps(m))
        with self.assertRaisesRegex(ArtifactMismatch, "not in the trust root"):
            v_bundle.load_bundle(tampered, trust_roles_path=self.trust_roles)

    def test_a_tampered_signed_field_is_refused(self):
        """Mutating a signed manifest field breaks the signature (not just a file hash)."""
        import json
        tampered = self._copy("tampered")
        mpath = tampered / v_bundle.BUNDLE_MANIFEST
        m = json.loads(mpath.read_text())
        m["model_version"] = "9.9.9-forged"  # a signed field
        mpath.write_text(json.dumps(m))
        with self.assertRaises(ArtifactMismatch):
            v_bundle.load_bundle(tampered, trust_roles_path=self.trust_roles)

    # -------------------------------------------------------------- anti-rollback
    def test_a_downgrade_below_the_floor_is_refused(self):
        state = self.work / "rollback_refuse.json"
        state.write_text('{"highest_trusted_version": "9.9.9"}')
        with self.assertRaisesRegex(ArtifactMismatch, "rollback"):
            v_bundle.load_bundle(self.bundle_dir, trust_roles_path=self.trust_roles,
                                 rollback_state_path=state)

    def test_acceptance_advances_the_rollback_floor(self):
        state = self.work / "rollback_advance.json"
        self.assertIsNone(v_trust.read_rollback_floor(state))
        v_bundle.load_bundle(self.bundle_dir, trust_roles_path=self.trust_roles,
                             rollback_state_path=state)
        import json
        manifest_version = json.loads(
            (self.bundle_dir / v_bundle.BUNDLE_MANIFEST).read_text())["model_version"]
        self.assertEqual(v_trust.read_rollback_floor(state), manifest_version)

    # -------------------------------------------------------------- explicit escape
    def test_allow_unsigned_loads_but_marks_not_for_release(self):
        b = v_bundle.load_bundle(self.bundle_dir, verify_signature=False)
        self.assertFalse(b.for_release)


if __name__ == "__main__":
    unittest.main()
