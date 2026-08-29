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

    def test_a_64_char_non_hex_public_key_fails_as_TrustError(self):
        """The malformed-key guard checks length; a 64-char non-hex value must still land
        as TrustError (an ArtifactMismatch), not a bare ValueError that escapes the load
        boundary's `except ArtifactMismatch` (U11 review)."""
        with tempfile.TemporaryDirectory() as td:
            import yaml
            path = Path(td) / "nonhex.yaml"
            path.write_text(yaml.safe_dump({"release_signing": {
                "trusted_keys": [{"key_id": "k", "public_key_hex": "z" * 64}]}}))
            with self.assertRaises(trust.TrustError):
                trust.load_trust_roots(path)

    def test_a_missing_trust_root_file_fails_as_TrustError(self):
        """An absent --trust-roles path must land as TrustError, not a bare
        FileNotFoundError that escapes the load boundary's except (CodeRabbit)."""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(trust.TrustError):
                trust.load_trust_roots(Path(td) / "does_not_exist.yaml")

    def test_a_scalar_revocation_list_fails_closed_not_open(self):
        """`revoked_key_ids: releaser-2026` (a scalar) would become set('releaser-2026') —
        single chars that never match a full key_id, silently failing revocation OPEN. It
        must be refused (CodeRabbit)."""
        _, pub_hex = _keypair()
        with tempfile.TemporaryDirectory() as td:
            import yaml
            path = Path(td) / "scalar.yaml"
            path.write_text(yaml.safe_dump({"release_signing": {
                "trusted_keys": [{"key_id": "releaser-2026", "public_key_hex": pub_hex}],
                "revoked_key_ids": "releaser-2026",  # scalar, not a list
            }}))
            with self.assertRaisesRegex(trust.TrustError, "list of strings"):
                trust.load_trust_roots(path)

    def test_a_non_mapping_trusted_key_entry_fails_as_TrustError(self):
        """A bare string in trusted_keys would raise AttributeError on .get (outside the
        TrustError contract); it must be refused as a malformed root (CodeRabbit)."""
        with tempfile.TemporaryDirectory() as td:
            import yaml
            path = Path(td) / "badentry.yaml"
            path.write_text(yaml.safe_dump({"release_signing": {
                "trusted_keys": ["not-a-mapping"]}}))
            with self.assertRaises(trust.TrustError):
                trust.load_trust_roots(path)

    def test_a_duplicate_public_key_under_two_key_ids_is_refused(self):
        """Same public key under two ids would let a revoked id be dodged via its alias;
        the trust root refuses it at load (U11 review)."""
        _, pub_hex = _keypair()
        with tempfile.TemporaryDirectory() as td:
            import yaml
            path = Path(td) / "dup.yaml"
            path.write_text(yaml.safe_dump({"release_signing": {"trusted_keys": [
                {"key_id": "primary", "public_key_hex": pub_hex},
                {"key_id": "alias", "public_key_hex": pub_hex},
            ]}}))
            with self.assertRaisesRegex(trust.TrustError, "duplicate|both"):
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

    def test_equal_versions_of_differing_segment_counts_are_not_a_rollback(self):
        """'1.2' and '1.2.0' are the same version; the raw tuple compare (1,2)<(1,2,0)
        would wrongly refuse '1.2' as a rollback. Padding fixes it (U11 review)."""
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "rollback.json"
            trust.advance_rollback_floor("1.2.0", state)
            trust.enforce_anti_rollback("1.2", state)      # equal: must NOT raise
            trust.advance_rollback_floor("1.2", state)     # equal: no lowering
            self.assertEqual(trust.read_rollback_floor(state), "1.2.0")
            # A genuinely older 2-segment version is still refused.
            with self.assertRaisesRegex(trust.TrustError, "rollback"):
                trust.enforce_anti_rollback("1.1", state)

    def test_deleting_the_state_file_resets_the_floor_accepted_residual(self):
        """DOCUMENTED RESIDUAL (U11 review): an ABSENT state file is 'no floor', so deleting
        it re-opens downgrades. This is out of the bundle-trust threat model (it needs local
        write to the site's own state dir). Corruption still fails closed; deletion does not.
        This test pins the accepted behavior so a future change is a conscious decision."""
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "rollback.json"
            trust.advance_rollback_floor("2.0.0", state)
            state.unlink()                                  # attacker/ops deletes it
            self.assertIsNone(trust.read_rollback_floor(state))
            trust.enforce_anti_rollback("1.0.0", state)     # no floor => does NOT raise


class SignatureFilenameContractTest(unittest.TestCase):
    def test_bundle_and_trust_agree_on_the_signature_filename(self):
        """bundle.py keeps a local _SIGNATURE_FILENAME to avoid importing trust at module
        load; assert it equals the authoritative trust.SIGNATURE_FILENAME so drift fails
        loudly instead of the writer and reader disagreeing on the file name (U11 review)."""
        from src.eval import bundle
        self.assertEqual(bundle._SIGNATURE_FILENAME, trust.SIGNATURE_FILENAME)


if __name__ == "__main__":
    unittest.main()
