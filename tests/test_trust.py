"""U11: Ed25519 bundle signing, revocation, and anti-rollback — every gate fails closed.

Data-free; uses a throwaway keypair. Each control has its red case exercised: a tampered
manifest, an untrusted key, a revoked key/bundle, and a rolled-back or corrupt version
floor all fail closed.
"""

import json
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src.eval import trust


def _keypair():
    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes_raw()
    pub_hex = priv.public_key().public_bytes_raw().hex()
    return priv_bytes, pub_hex


def _manifest(**over):
    m = {
        "model_bundle_id": "b1",
        "model_version": "1.2.0",
        "vocab_hash": "a" * 64,
        "outcome_spec_hash": "b" * 64,
        "clif_version": "2.1",
        "files": {"vocab.json": "c" * 64},
        "outcome_queries": {"map_below_65_48h": {"target_index": 0, "tau_bin": 1,
                                                 "direction": 0}},
    }
    m.update(over)
    return m


class SignVerifyTest(unittest.TestCase):
    def test_sign_then_verify_roundtrips(self):
        priv, pub_hex = _keypair()
        m = _manifest()
        sig = trust.sign_manifest(m, priv)
        trust.verify_manifest_signature(m, sig, bytes.fromhex(pub_hex))  # no raise

    def test_a_tampered_manifest_fails_verification(self):
        priv, pub_hex = _keypair()
        m = _manifest()
        sig = trust.sign_manifest(m, priv)
        m["outcome_queries"]["map_below_65_48h"]["direction"] = 1  # tamper after signing
        with self.assertRaises(trust.TrustError):
            trust.verify_manifest_signature(m, sig, bytes.fromhex(pub_hex))

    def test_a_signature_from_another_key_fails(self):
        priv_a, _ = _keypair()
        _, pub_b_hex = _keypair()
        m = _manifest()
        sig = trust.sign_manifest(m, priv_a)
        with self.assertRaises(trust.TrustError):
            trust.verify_manifest_signature(m, sig, bytes.fromhex(pub_b_hex))

    def test_non_hex_signature_fails_closed(self):
        _, pub_hex = _keypair()
        with self.assertRaises(trust.TrustError):
            trust.verify_manifest_signature(_manifest(), "not-hex", bytes.fromhex(pub_hex))


class TrustRootTest(unittest.TestCase):
    def _root_file(self, td, pub_hex, *, key_id="releaser-2026",
                   revoked_keys=(), revoked_bundles=()):
        path = Path(td) / "trust_roles.yaml"
        import yaml
        path.write_text(yaml.safe_dump({
            "release_signing": {
                "trusted_keys": [{"key_id": key_id, "public_key_hex": pub_hex}],
                "revoked_key_ids": list(revoked_keys),
                "revoked_bundle_ids": list(revoked_bundles),
            }
        }))
        return path

    def test_verify_against_trust_root_happy_path(self):
        priv, pub_hex = _keypair()
        m = _manifest()
        sig = trust.sign_manifest(m, priv)
        with tempfile.TemporaryDirectory() as td:
            root = trust.load_trust_roots(self._root_file(td, pub_hex))
            trust.verify_against_trust_root(m, sig, "releaser-2026", root)  # no raise

    def test_an_untrusted_key_id_is_refused(self):
        priv, pub_hex = _keypair()
        m = _manifest()
        sig = trust.sign_manifest(m, priv)
        with tempfile.TemporaryDirectory() as td:
            root = trust.load_trust_roots(self._root_file(td, pub_hex))
            with self.assertRaisesRegex(trust.TrustError, "not in the trust root"):
                trust.verify_against_trust_root(m, sig, "some-other-key", root)

    def test_a_revoked_key_is_refused_despite_a_valid_signature(self):
        priv, pub_hex = _keypair()
        m = _manifest()
        sig = trust.sign_manifest(m, priv)
        with tempfile.TemporaryDirectory() as td:
            root = trust.load_trust_roots(
                self._root_file(td, pub_hex, revoked_keys=["releaser-2026"]))
            with self.assertRaisesRegex(trust.TrustError, "revoked"):
                trust.verify_against_trust_root(m, sig, "releaser-2026", root)

    def test_a_revoked_bundle_id_is_refused(self):
        priv, pub_hex = _keypair()
        m = _manifest(model_bundle_id="bad-bundle")
        sig = trust.sign_manifest(m, priv)
        with tempfile.TemporaryDirectory() as td:
            root = trust.load_trust_roots(
                self._root_file(td, pub_hex, revoked_bundles=["bad-bundle"]))
            with self.assertRaisesRegex(trust.TrustError, "revoked"):
                trust.verify_against_trust_root(m, sig, "releaser-2026", root)

    def test_a_trust_root_with_no_keys_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            import yaml
            path = Path(td) / "empty.yaml"
            path.write_text(yaml.safe_dump({"release_signing": {"trusted_keys": []}}))
            with self.assertRaises(trust.TrustError):
                trust.load_trust_roots(path)


class AntiRollbackTest(unittest.TestCase):
    def test_newer_advances_and_older_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "rollback.json"
            self.assertIsNone(trust.read_rollback_floor(state))
            trust.advance_rollback_floor("1.2.0", state)
            self.assertEqual(trust.read_rollback_floor(state), "1.2.0")
            trust.enforce_anti_rollback("1.3.0", state)  # newer: ok
            with self.assertRaisesRegex(trust.TrustError, "rollback"):
                trust.enforce_anti_rollback("1.1.0", state)  # older: refused

    def test_advance_does_not_lower_the_floor(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "rollback.json"
            trust.advance_rollback_floor("2.0.0", state)
            trust.advance_rollback_floor("1.0.0", state)  # older: no-op
            self.assertEqual(trust.read_rollback_floor(state), "2.0.0")

    def test_a_corrupt_state_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "rollback.json"
            state.write_text("{ not json")
            with self.assertRaises(trust.TrustError):
                trust.read_rollback_floor(state)
            with self.assertRaises(trust.TrustError):
                trust.enforce_anti_rollback("9.9.9", state)


if __name__ == "__main__":
    unittest.main()
