"""Releaser-to-site bundle trust: Ed25519 signatures, revocation, anti-rollback (U11).

U5/U9 sign REPORTS and the access-log chain with HMAC (symmetric) — correct for the
site->aggregator direction. This module is the opposite direction: the releaser signs a
bundle manifest with a PRIVATE key; every site verifies with the releaser's PUBLIC key,
distributed out of band via `configs/trust_roles.yaml`. A symmetric secret cannot model
a trust root (both sides would hold it, so any site could forge a bundle), so this is
asymmetric — Ed25519 via `cryptography`.

Three fail-closed controls, each closing a gap U9's review deferred here:
  1. **Signature.** The self-hashed manifest proves internal consistency, not
     provenance — a fully re-hashed replacement bundle passes `load_bundle`'s file
     checks. A detached signature over the manifest's identity + files map +
     outcome_queries, verified against the trust root, is the anchor that makes a
     replacement detectable.
  2. **Revocation.** A signed revocation list names compromised releaser key ids and/or
     withdrawn bundle ids; a revoked signer or bundle fails closed even with a valid
     signature.
  3. **Anti-rollback.** The site persists the highest release version it has accepted; a
     validly-signed but OLDER bundle (a forced downgrade to a superseded, possibly
     weakened bundle) fails closed. A corrupt state file fails closed rather than
     resetting the floor.

`cryptography` is imported lazily inside the sign/verify functions so importing this
module never requires it — only signing or verifying does.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from clif_validate._vendor.eval.clif_validate import ArtifactMismatch

SIGNATURE_FILENAME = "bundle_manifest.sig"

# The manifest fields the signature binds. Everything load_bundle trusts flows from
# these: the identity block, the per-file hash map (so file contents are transitively
# covered), and the zero-shot query parameters.
_SIGNED_FIELDS = (
    "model_bundle_id", "model_version", "vocab_hash", "outcome_spec_hash",
    "clif_version", "files", "outcome_queries",
)


class TrustError(ArtifactMismatch):
    """A bundle failed a releaser-to-site trust control (signature/revocation/rollback).

    Subclasses ArtifactMismatch so every existing `except ArtifactMismatch` at the load
    boundary already fails closed on a trust failure.
    """


def _signable_bytes(manifest: dict) -> bytes:
    """Canonical JSON of exactly the signed subset — deterministic across processes."""
    subset = {}
    for field in _SIGNED_FIELDS:
        if field not in manifest:
            raise TrustError(f"manifest cannot be signed: missing field {field!r}")
        subset[field] = manifest[field]
    return json.dumps(subset, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sign_manifest(manifest: dict, private_key_bytes: bytes) -> str:
    """Releaser-side: return the hex Ed25519 signature over the manifest's signed subset."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    return key.sign(_signable_bytes(manifest)).hex()


def verify_manifest_signature(manifest: dict, signature_hex: str,
                              public_key_bytes: bytes) -> None:
    """Verify the manifest signature against one public key. Fail closed on any mismatch."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        signature = bytes.fromhex(signature_hex)
    except (ValueError, TypeError) as exc:
        raise TrustError("bundle signature is not valid hex") from exc
    key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
    try:
        key.verify(signature, _signable_bytes(manifest))
    except InvalidSignature as exc:
        raise TrustError(
            "bundle manifest signature does not verify against this trusted key — the "
            "bundle was tampered with or signed by an untrusted releaser"
        ) from exc


def load_trust_roots(path: str | Path) -> dict:
    """Load the trust root: {key_id: public_key_bytes}, plus revoked key/bundle ids.

    Shape (configs/trust_roles.yaml):
        release_signing:
          trusted_keys:
            - key_id: releaser-2026
              public_key_hex: <64 hex chars = 32-byte Ed25519 public key>
          revoked_key_ids: [<key_id>, ...]
          revoked_bundle_ids: [<model_bundle_id>, ...]
    """
    doc = yaml.safe_load(Path(path).read_text())
    block = (doc or {}).get("release_signing")
    if not isinstance(block, dict):
        raise TrustError(
            f"trust roles at {path} declare no release_signing block; refusing to verify "
            "a bundle against an undefined trust root"
        )
    keys: dict[str, bytes] = {}
    for entry in block.get("trusted_keys") or []:
        key_id = entry.get("key_id")
        hex_pub = entry.get("public_key_hex")
        if not key_id or not isinstance(hex_pub, str) or len(hex_pub) != 64:
            raise TrustError(f"trusted key entry is malformed: {entry!r}")
        keys[key_id] = bytes.fromhex(hex_pub)
    if not keys:
        raise TrustError(f"trust roles at {path} declare no trusted signing keys")
    return {
        "keys": keys,
        "revoked_key_ids": set(block.get("revoked_key_ids") or []),
        "revoked_bundle_ids": set(block.get("revoked_bundle_ids") or []),
    }


def verify_against_trust_root(manifest: dict, signature_hex: str, key_id: str,
                              trust_root: dict) -> None:
    """Full releaser-to-site verification: revocation, then signature by a trusted key."""
    if key_id in trust_root["revoked_key_ids"]:
        raise TrustError(f"signing key {key_id!r} is revoked; refusing the bundle")
    if manifest.get("model_bundle_id") in trust_root["revoked_bundle_ids"]:
        raise TrustError(
            f"bundle {manifest.get('model_bundle_id')!r} is revoked; refusing it"
        )
    public_key = trust_root["keys"].get(key_id)
    if public_key is None:
        raise TrustError(
            f"bundle is signed by key {key_id!r}, which is not in the trust root; a "
            "re-hashed replacement bundle signed by an untrusted key is refused here"
        )
    verify_manifest_signature(manifest, signature_hex, public_key)


# ---------------------------------------------------------------- anti-rollback
def _version_tuple(version: str) -> tuple:
    """Parse a dotted version into an int tuple for ordering; non-numeric parts sort low."""
    parts = []
    for chunk in str(version).split("."):
        parts.append(int(chunk) if chunk.isdigit() else -1)
    return tuple(parts)


def read_rollback_floor(state_path: str | Path) -> str | None:
    """The highest release version this site has accepted, or None if never set.

    A malformed/corrupt state file FAILS CLOSED (raises) rather than being treated as
    "no floor" — a downgrade attack could otherwise just corrupt the file to reset it.
    """
    path = Path(state_path)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text())
        floor = state["highest_trusted_version"]
    except (ValueError, KeyError, TypeError) as exc:
        raise TrustError(
            f"anti-rollback state at {path} is corrupt; refusing rather than resetting "
            "the version floor"
        ) from exc
    if not isinstance(floor, str):
        raise TrustError(f"anti-rollback floor at {path} is not a version string")
    return floor


def enforce_anti_rollback(model_version: str, state_path: str | Path) -> None:
    """Fail closed if `model_version` is older than the persisted floor."""
    floor = read_rollback_floor(state_path)
    if floor is not None and _version_tuple(model_version) < _version_tuple(floor):
        raise TrustError(
            f"bundle version {model_version!r} is older than the highest accepted "
            f"version {floor!r}; refusing a rollback to a superseded bundle"
        )


def advance_rollback_floor(model_version: str, state_path: str | Path) -> None:
    """Record `model_version` as the new floor if it is newer (call AFTER acceptance)."""
    path = Path(state_path)
    floor = read_rollback_floor(path)
    if floor is None or _version_tuple(model_version) > _version_tuple(floor):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"highest_trusted_version": str(model_version)}))


__all__ = [
    "SIGNATURE_FILENAME",
    "TrustError",
    "advance_rollback_floor",
    "enforce_anti_rollback",
    "load_trust_roots",
    "read_rollback_floor",
    "sign_manifest",
    "verify_against_trust_root",
    "verify_manifest_signature",
]
