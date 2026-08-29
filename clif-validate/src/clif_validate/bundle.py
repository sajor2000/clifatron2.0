"""Bundle loading and verification — stable re-export of the vendored contract."""

from clif_validate._vendor.eval.bundle import (
    BUNDLE_MANIFEST,
    REQUIRED_BUNDLE_FILES,
    Bundle,
    hash_bundle_files,
    load_bundle,
    pin_bundle_policy,
    verify_bundle_files,
    write_bundle_manifest,
)
# ArtifactMismatch is the failure mode of every function here (load_bundle,
# verify_bundle_files); re-export it so callers catch it without reaching into _vendor.
from clif_validate._vendor.eval.clif_validate import ArtifactMismatch

__all__ = [
    "BUNDLE_MANIFEST",
    "REQUIRED_BUNDLE_FILES",
    "ArtifactMismatch",
    "Bundle",
    "hash_bundle_files",
    "load_bundle",
    "pin_bundle_policy",
    "verify_bundle_files",
    "write_bundle_manifest",
]
