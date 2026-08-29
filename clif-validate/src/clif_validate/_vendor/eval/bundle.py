"""The bundle contract — the sealed box an external site receives (U9).

A bundle is a directory holding everything inference needs and nothing else:

    bundle_manifest.json   identity + per-file SHA-256 + zero-shot outcome queries
    head_weights.pt        trained head parameters (loaded strict=True)
    config.json, model.*   the HF backbone checkpoint
    vocab.json             {"vocab", "edges", "manifest"} — the frozen vocabulary
    data_config.yaml       resolved data config (tables, binning, versions)
    cohort.yaml            the frozen outcome contract the vocabulary was hashed against
    artifact_policy.yaml   disclosure policy pinned to THIS bundle

Three rules govern this module:

1. **Everything is hash-covered, including files nobody asked about.** The manifest's
   `files` map must list every regular file in the bundle; an unlisted file is a
   tamper channel (drop in a different `artifact_policy.yaml`, nothing notices), so
   its presence is a hard failure, not a warning.

2. **No repo-relative path survives into a wheel.** `schema.DEFAULT_ARTIFACT_POLICY`,
   `tokenize.ROOT / cfg["cohort_contract"]`, and `auto_label`'s config defaults all
   resolve against the repository checkout and dangle inside site-packages. The bundle
   carries its own copies, and `load_bundle` rewrites the config paths to the bundled
   absolute files (``Path(root) / "/abs"`` is ``"/abs"``, so ROOT-relative resolution
   becomes a no-op) and pins the disclosure policy through
   ``schema.POLICY_OVERRIDE_ENV`` so every downstream ``min_cell_size()`` call reads
   the bundle's floor, not a checkout that is not there.

3. **Verify before parse.** File hashes are checked before any bundled file's content
   is interpreted; the vocabulary then re-verifies its own compatibility hashes via
   `validate_vocabulary_artifact`; and the manifest's identity fields are cross-checked
   against the vocabulary manifest so one bundle cannot carry two identities.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from clif_validate._vendor.eval import schema as _schema
from clif_validate._vendor.eval.clif_validate import ArtifactMismatch, verify_bundle_compatibility

BUNDLE_MANIFEST = "bundle_manifest.json"

# Roles the manifest must cover by exact name. The backbone checkpoint files
# (config.json + weights) vary by format, so they are covered by the walk in
# `hash_bundle_files` rather than named here.
REQUIRED_BUNDLE_FILES = (
    "head_weights.pt",
    "vocab.json",
    "data_config.yaml",
    "cohort.yaml",
    "artifact_policy.yaml",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_bundle_files(bundle_dir: str | Path) -> dict[str, str]:
    """SHA-256 of every regular file under the bundle except the manifest itself.

    Keys are POSIX relative paths, sorted, so the map is deterministic and the
    manifest that embeds it is byte-stable across builds of identical content.
    """
    root = Path(bundle_dir)
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != BUNDLE_MANIFEST:
            out[p.relative_to(root).as_posix()] = _sha256_file(p)
    return out


def verify_bundle_files(bundle_dir: str | Path, manifest: dict) -> None:
    """Check the manifest's file map against the directory. All three directions.

    Missing file: the bundle is incomplete. Hash mismatch: the file was altered.
    Unlisted file: something was ADDED that the manifest never covered — the exact
    shape of a swapped policy or vocabulary, so it fails just as hard.
    """
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise ArtifactMismatch(
            "bundle manifest declares no file hashes. An unhashed bundle offers no "
            "way to tell the released files from substituted ones."
        )
    for required in REQUIRED_BUNDLE_FILES:
        if required not in declared:
            raise ArtifactMismatch(
                f"bundle manifest does not cover required file {required!r}"
            )
    actual = hash_bundle_files(bundle_dir)
    missing = sorted(set(declared) - set(actual))
    if missing:
        raise ArtifactMismatch(f"bundle is missing hashed files: {', '.join(missing)}")
    unlisted = sorted(set(actual) - set(declared))
    if unlisted:
        raise ArtifactMismatch(
            f"bundle contains files the manifest does not cover: {', '.join(unlisted)}. "
            "An unlisted file is outside the bundle's integrity envelope; refusing."
        )
    altered = sorted(name for name in declared if declared[name] != actual[name])
    if altered:
        raise ArtifactMismatch(f"bundle file hashes do not match: {', '.join(altered)}")


def _validated_outcome_queries(manifest: dict) -> dict[str, dict]:
    """Zero-shot query parameters per outcome: {name: {target_index, tau_bin, direction}}.

    These are the model-facing integers, not the human-readable strings in
    `cohort.yaml`: per ThresholdHazardHead, `tau_bin` is the queried threshold's
    VALUE-bin index (0..n_value_bins) and `direction` is 0=below / 1=above.
    """
    queries = manifest.get("outcome_queries")
    if not isinstance(queries, dict) or not queries:
        raise ArtifactMismatch(
            "bundle manifest declares no outcome_queries. Without per-outcome "
            "target_index/tau_bin/direction the zero-shot head cannot be asked "
            "anything, and guessing them would score the wrong question."
        )
    for name, q in queries.items():
        if not isinstance(q, dict):
            raise ArtifactMismatch(f"outcome query {name!r} is not a mapping")
        for field in ("target_index", "tau_bin", "direction"):
            value = q.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ArtifactMismatch(
                    f"outcome query {name!r} field {field!r} must be a non-negative "
                    f"integer, got {value!r}"
                )
    return queries


def pin_bundle_policy(policy_path: str | Path) -> None:
    """Make the bundle's artifact policy the one every downstream check reads.

    `schema.min_cell_size()` is lru_cached and called with no arguments throughout
    the export path, so pinning is two steps: point the override env var at the
    bundled policy AND drop the cache so a value resolved earlier (possibly from a
    repo checkout that only exists on the releaser's machine) cannot linger.
    """
    os.environ[_schema.POLICY_OVERRIDE_ENV] = str(Path(policy_path).resolve())
    _schema.min_cell_size.cache_clear()


@dataclass(frozen=True)
class Bundle:
    path: Path
    provenance: dict            # the identity block verify_bundle_compatibility returns
    outcome_queries: dict       # {outcome_name: {target_index, tau_bin, direction}}
    vocab: dict
    edges: dict
    vocab_manifest: dict
    data_cfg: dict              # config-path fields rewritten to bundled absolutes
    policy: dict
    policy_path: Path
    cohort_path: Path
    data_cfg_path: Path


def load_bundle(path: str | Path, *, pin_policy: bool = True) -> Bundle:
    """Verify and load a bundle, fail-closed at every step.

    Order matters: file hashes are verified before any bundled file is parsed, the
    policy is pinned before the vocabulary validation that reads it, and the
    manifest's identity is cross-checked against the vocabulary's own manifest last —
    after both have independently proven internally consistent.
    """
    root = Path(path)
    manifest_path = root / BUNDLE_MANIFEST
    if not manifest_path.exists():
        raise ArtifactMismatch(
            f"bundle_manifest.json is absent from {root}. Provenance cannot be "
            "established, so no result from this bundle is attributable."
        )
    manifest = json.loads(manifest_path.read_text())
    verify_bundle_files(root, manifest)
    provenance = verify_bundle_compatibility(str(root))
    outcome_queries = _validated_outcome_queries(manifest)

    policy_path = (root / "artifact_policy.yaml").resolve()
    policy = yaml.safe_load(policy_path.read_text())
    if pin_policy:
        pin_bundle_policy(policy_path)

    cohort_path = (root / "cohort.yaml").resolve()
    data_cfg_path = (root / "data_config.yaml").resolve()
    data_cfg = yaml.safe_load(data_cfg_path.read_text())
    # Rewrite the config-path fields to the bundled absolute files. tokenize.py
    # resolves them as `ROOT / cfg[...]`, and pathlib joins an absolute right-hand
    # side by discarding the left — so after this rewrite the repo-relative
    # resolution is a no-op and the wheel never needs a repository checkout.
    data_cfg["cohort_contract"] = str(cohort_path)
    data_cfg["artifact_policy"] = str(policy_path)

    blob = json.loads((root / "vocab.json").read_text())
    from clif_validate._vendor.data.tokenize import validate_vocabulary_artifact

    vocab, edges, vocab_manifest = validate_vocabulary_artifact(blob, data_cfg, policy)

    # One bundle, one identity: the manifest's headline hashes must be the ones the
    # vocabulary actually carries, or the provenance block would attest to a
    # different artifact than the one inference used.
    hashes = vocab_manifest["hashes"]
    if provenance["vocab_hash"] != hashes["vocabulary"]:
        raise ArtifactMismatch(
            "bundle manifest vocab_hash does not match the bundled vocabulary's own "
            "hash — the manifest attests to a different vocabulary than it ships."
        )
    if provenance["outcome_spec_hash"] != hashes["outcome_spec"]:
        raise ArtifactMismatch(
            "bundle manifest outcome_spec_hash does not match the bundled cohort "
            "contract's hash — the manifest attests to different outcomes than it ships."
        )
    # The manifest carries the major.minor CLIF version verify_bundle_compatibility
    # gates on ("2.1"); the data config carries the full schema_version ("2.1.0").
    if data_cfg["schema_version"].split(".")[:2] != provenance["clif_version"].split(".")[:2]:
        raise ArtifactMismatch(
            f"bundle manifest clif_version {provenance['clif_version']!r} does not "
            f"match the bundled data config's schema_version "
            f"{data_cfg['schema_version']!r}"
        )

    return Bundle(
        path=root,
        provenance=provenance,
        outcome_queries=outcome_queries,
        vocab=vocab,
        edges=edges,
        vocab_manifest=vocab_manifest,
        data_cfg=data_cfg,
        policy=policy,
        policy_path=policy_path,
        cohort_path=cohort_path,
        data_cfg_path=data_cfg_path,
    )


def write_bundle_manifest(bundle_dir: str | Path, *, model_bundle_id: str,
                          model_version: str, vocab_hash: str,
                          outcome_spec_hash: str, clif_version: str,
                          outcome_queries: dict[str, dict]) -> Path:
    """Releaser-side: hash the bundle's files and seal the manifest over them.

    Shared by the synthetic fixture and any real release path so there is exactly one
    implementation of "what a manifest contains". Must be called LAST, after every
    other file is in place — the file map covers whatever is present at call time.
    """
    root = Path(bundle_dir)
    manifest = {
        "model_bundle_id": model_bundle_id,
        "model_version": model_version,
        "vocab_hash": vocab_hash,
        "outcome_spec_hash": outcome_spec_hash,
        "clif_version": clif_version,
        "outcome_queries": outcome_queries,
        "files": hash_bundle_files(root),
    }
    out = root / BUNDLE_MANIFEST
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out


__all__ = [
    "BUNDLE_MANIFEST",
    "REQUIRED_BUNDLE_FILES",
    "Bundle",
    "hash_bundle_files",
    "load_bundle",
    "pin_bundle_policy",
    "verify_bundle_files",
    "write_bundle_manifest",
]
