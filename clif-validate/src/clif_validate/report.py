"""Export assembly, signing, and the disclosure ceremony — stable re-exports."""

from clif_validate._vendor.eval.attestation import (
    AuthenticationError,
    confirm_publication,
    confirmed_releases,
    preflight_access_log,
    reconcile_ledger,
    record_access,
    sign_report,
    unconfirmed_releases,
    verify_access_log,
    verify_report,
)
from clif_validate._vendor.eval.clif_validate import (
    ArtifactMismatch,
    build_export,
    write_export,
)
from clif_validate._vendor.eval.schema import DisclosureError, validate_export

__all__ = [
    # The full ceremony surface an agent/operator branches on, re-exported here so
    # callers never have to reach into `_vendor` (which the package tells them not to):
    # the exceptions that ARE the failure modes, plus ledger/access-log verification.
    "ArtifactMismatch",
    "AuthenticationError",
    "DisclosureError",
    "build_export",
    "confirm_publication",
    "confirmed_releases",
    "preflight_access_log",
    "reconcile_ledger",
    "record_access",
    "sign_report",
    "unconfirmed_releases",
    "validate_export",
    "verify_access_log",
    "verify_report",
    "write_export",
]
