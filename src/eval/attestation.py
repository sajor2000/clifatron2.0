"""Site-to-aggregator report authentication and the cumulative disclosure ledger (U5).

Two controls, both on the site->aggregator direction of the trust model. The
releaser->site direction — release signing, trust root, revocation, anti-rollback,
key custody — belongs to U11 and is deliberately absent here.

**Report authentication.** Aggregate artifacts arriving at the aggregator are otherwise
unauthenticated, so a forged or altered site report silently poisons the cross-site
comparison and the headline forest figure with no way to attribute it afterward. Reports
are signed with an HMAC over a canonical serialization; the aggregator verifies before
reading. HMAC (shared secret per site) rather than asymmetric signing because the
aggregator and sites are inside one governed consortium with an existing key-exchange
path — U11 revisits this if the trust boundary widens.

**Cumulative disclosure ledger.** One-shot suppression is not enough across repeated
releases: U6/U7/U8 emit repeated aggregate releases over the same cohorts at the same
sites, and a cell suppressed in release 2 can be recovered by differencing against
release 1. The ledger is append-only, written at every export, and read by the
differencing check before the next export. Reconstructing it afterward from surviving
artifacts is exactly the failure this exists to prevent, which is why it is created
here at the first release boundary and not at U10.

The ledger stores cell *definitions and sizes*, never cell contents.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

from src.eval.schema import (
    EVALUABLE,
    MIN_CELL_SIZE,
    NON_EVALUABLE_STATUSES,
    DisclosureError,
)

_SIGNATURE_FIELD = "signature"


class AuthenticationError(ValueError):
    """Raised when a site report fails signature verification."""


# ---------------------------------------------------------------- canonical form
def canonical_bytes(payload: dict) -> bytes:
    """Deterministic serialization used for both signing and verification.

    Sorted keys and no insignificant whitespace, so two structurally identical reports
    produce identical bytes regardless of construction order. The signature field is
    excluded — a signature cannot cover itself.
    """
    body = {k: v for k, v in payload.items() if k != _SIGNATURE_FIELD}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sign_report(payload: dict, key: bytes) -> dict:
    """Return a copy of `payload` carrying a detached HMAC signature."""
    if not key:
        raise AuthenticationError("refusing to sign with an empty key")
    mac = hmac.new(key, canonical_bytes(payload), hashlib.sha256).hexdigest()
    return {**payload, _SIGNATURE_FIELD: mac}


def verify_report(payload: dict, key: bytes) -> dict:
    """Verify a site report's signature. Returns it unchanged, or raises.

    Fails closed on an absent signature as well as a wrong one: an unsigned report is
    not "unverified but probably fine", it is a report whose origin nobody can attest.
    """
    if _SIGNATURE_FIELD not in payload:
        raise AuthenticationError(
            "site report carries no signature; an unauthenticated report cannot enter "
            "the cross-site comparison"
        )
    expected = hmac.new(key, canonical_bytes(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(payload[_SIGNATURE_FIELD]), expected):
        raise AuthenticationError(
            "site report signature does not match its content; the report was altered "
            "in transit or signed with a different key"
        )
    return payload


# ---------------------------------------------------------------- access log
def record_access(log_path: str | Path, *, model_version: str, actor_role: str,
                  artifact_id: str, action: str) -> None:
    """Append a tamper-evident access record, keyed by model version.

    Each line carries a hash chained to the previous line, so a deleted or edited
    record breaks the chain and is detectable by `verify_access_log`. This is what
    makes "accesses are logged by model version" auditable rather than asserted.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = "0" * 64
    if path.exists():
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if lines:
            prev = json.loads(lines[-1])["chain"]
    entry = {
        "model_version": model_version,
        "actor_role": actor_role,
        "artifact_id": artifact_id,
        "action": action,
        "prev": prev,
    }
    entry["chain"] = hashlib.sha256(
        (prev + json.dumps(entry, sort_keys=True, separators=(",", ":"))).encode("utf-8")
    ).hexdigest()
    with path.open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")


def verify_access_log(log_path: str | Path) -> bool:
    """Return True when the access log's hash chain is intact."""
    path = Path(log_path)
    if not path.exists():
        return True
    prev = "0" * 64
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["prev"] != prev:
            return False
        chain = entry.pop("chain")
        recomputed = hashlib.sha256(
            (entry["prev"] + json.dumps(entry, sort_keys=True, separators=(",", ":"))).encode("utf-8")
        ).hexdigest()
        if chain != recomputed:
            return False
        prev = chain
    return True


# ---------------------------------------------------------------- disclosure ledger
def _cell_key(site_id: str, outcome: str, attribute: str, category: str) -> str:
    return f"{site_id}|{outcome}|{attribute}|{category}"


def ledger_entries(payload: dict) -> list[dict]:
    """Extract the ledger records implied by one export.

    Records the *shape* of what was released — site, model version, outcome, cell
    definition, reported n, suppression status — and never a metric value.
    """
    site = payload["site_id"]
    version = payload["model_version"]
    out: list[dict] = []
    for outcome, block in (payload.get("outcomes") or {}).items():
        metrics = block.get("metrics") or {}
        out.append({
            "site_id": site, "model_version": version, "outcome": outcome,
            "cell": _cell_key(site, outcome, "_overall", "_all"),
            "n": metrics.get("n"), "status": block.get("status"),
        })
        for attr, cells in (block.get("subgroups") or {}).items():
            for cat, cell in cells.items():
                out.append({
                    "site_id": site, "model_version": version, "outcome": outcome,
                    "cell": _cell_key(site, outcome, attr, cat),
                    "n": cell.get("n"), "status": cell.get("status"),
                })
    return out


def read_ledger(ledger_path: str | Path) -> list[dict]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def check_cross_release_differencing(payload: dict, ledger_path: str | Path) -> None:
    """Block an export that would expose a previously suppressed cell by differencing.

    The attack this stops: release 1 suppresses cell X; release 2 covers the same site,
    outcome, and cell but releases it (or releases a total that makes X a subtraction
    away). Each release is individually compliant; together they are not. Suppression
    that only ever looks at the current report cannot see this.
    """
    prior = read_ledger(ledger_path)
    suppressed_before = {
        e["cell"] for e in prior
        if e.get("status") in NON_EVALUABLE_STATUSES
    }
    if not suppressed_before:
        return
    for entry in ledger_entries(payload):
        if entry["cell"] in suppressed_before and entry.get("status") == EVALUABLE:
            raise DisclosureError(
                f"cell {entry['cell']!r} was suppressed in a prior release and would be "
                f"released now (model_version {entry['model_version']!r}). Differencing "
                "the two releases recovers the suppressed value. Keep it suppressed, or "
                "obtain a documented disclosure-review exception."
            )


def append_to_ledger(payload: dict, ledger_path: str | Path) -> None:
    """Append this export's records to the append-only ledger."""
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for entry in ledger_entries(payload):
            fh.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")


__all__ = [
    "AuthenticationError", "MIN_CELL_SIZE",
    "canonical_bytes", "sign_report", "verify_report",
    "record_access", "verify_access_log",
    "ledger_entries", "read_ledger", "check_cross_release_differencing", "append_to_ledger",
]
