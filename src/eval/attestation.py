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
import os
from pathlib import Path

from src.eval.schema import (
    EVALUABLE,
    NON_EVALUABLE_STATUSES,
    DisclosureError,
    min_cell_size,
)

_SIGNATURE_FIELD = "signature"

# Environment variable naming the file that holds the access-log chain key. The key
# itself is never stored in the repo. An unkeyed SHA-256 chain is only tamper-EVIDENT
# against an editor who cannot recompute it; anyone with write access to the log could
# rewrite entries and re-chain them (U5 review #10). Keying it closes that.
_CHAIN_KEY_ENV = "CLIF_ACCESS_LOG_KEY_FILE"
_DEV_CHAIN_KEY = b"clif-access-log-unconfigured-development-key"


def _chain_key() -> bytes:
    """Key for the access-log HMAC chain.

    Falls back to a well-known development key when unconfigured, so tests and dry runs
    work. That fallback is NOT a security control: a site running for real must set
    CLIF_ACCESS_LOG_KEY_FILE, and U11 owns provisioning it alongside the release trust
    root. Documented here rather than silently pretending the chain is protected.
    """
    key_file = os.environ.get(_CHAIN_KEY_ENV)
    if key_file:
        return Path(key_file).read_bytes().strip()
    return _DEV_CHAIN_KEY


def _durably_append(path: Path, line: str) -> None:
    """Append a line and return only once it is on stable storage.

    A buffered write is not a record. `write_export` publishes the artifact immediately
    after the ledger append, so a crash between the two used to leave the report published
    with its ledger entry still in the OS page cache -- and a later differencing check
    would then miss a real prior release.

    fsync on the file makes the bytes durable; fsync on the parent directory makes the
    entry durable when the file was newly created. Both are needed for the guarantee to
    hold across a power loss, not just a process kill.
    """
    existed = path.exists()
    with path.open("a") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    if not existed:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _head_path(log_path: Path) -> Path:
    """Sidecar anchor naming the chain's expected terminal record."""
    return log_path.with_suffix(log_path.suffix + ".head")


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
    # `allow_nan=False` and no `default=` (U5 review #27). Bare NaN/Infinity is invalid
    # JSON that no non-Python consumer can verify, and `default=str` silently stringified
    # unserializable values straight into the signed bytes -- authenticating something
    # nobody could reproduce. Both now raise instead.
    try:
        return json.dumps(body, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except ValueError as exc:
        raise AuthenticationError(
            f"payload is not strictly serializable and cannot be signed: {exc}. "
            "Non-finite floats must be emitted as null before signing."
        ) from exc
    except TypeError as exc:
        raise AuthenticationError(
            f"payload contains a non-JSON value and cannot be signed: {exc}"
        ) from exc


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
    if not key:
        raise AuthenticationError(
            "refusing to verify with an empty key. An aggregator that loaded a missing "
            "secret as b'' would verify anything signed with b'' -- the asymmetry with "
            "sign_report's guard silently disabled authentication."
        )
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
    prev, seq = "0" * 64, 0
    if path.exists():
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if lines:
            try:
                last = json.loads(lines[-1])
                prev, seq = last["chain"], int(last["seq"]) + 1
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                # An interrupted write leaves a partial last line. Crashing with a raw
                # JSONDecodeError on every subsequent run bricks the audit trail behind
                # an unreadable error (U5 review #9); name the cause instead.
                raise AuthenticationError(
                    f"access log at {path.name} is corrupt at its final record ({exc}). "
                    "A write was interrupted. Do not append to it: preserve it for audit "
                    "and start a new log with a recorded reason."
                ) from exc
    entry = {
        "seq": seq,
        "model_version": model_version,
        "actor_role": actor_role,
        "artifact_id": artifact_id,
        "action": action,
        "prev": prev,
    }
    entry["chain"] = hmac.new(
        _chain_key(),
        (prev + json.dumps(entry, sort_keys=True, separators=(",", ":"))).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    _durably_append(path, json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    # Head anchor, written outside the log. Truncating or deleting the log no longer
    # produces a self-consistent shorter chain, because the head still names the record
    # count and terminal hash it must end at (U5 review #10).
    head = _head_path(path)
    head.write_text(json.dumps({"seq": seq, "chain": entry["chain"]},
                               sort_keys=True, separators=(",", ":")))


def verify_access_log(log_path: str | Path) -> bool:
    """Return True when the access log's hash chain is intact."""
    path = Path(log_path)
    head_path = _head_path(path)

    if not path.exists():
        # FAIL CLOSED. Returning True for an absent log meant deleting the audit trail
        # was indistinguishable from never having written one (U5 review #10). An absent
        # log is only intact if no head anchor claims otherwise.
        return not head_path.exists()

    prev, expected_seq = "0" * 64, 0
    last_chain, last_seq = prev, -1
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return False
        if entry.get("prev") != prev or int(entry.get("seq", -1)) != expected_seq:
            return False
        chain = entry.pop("chain", None)
        recomputed = hmac.new(
            _chain_key(),
            (entry["prev"] + json.dumps(entry, sort_keys=True, separators=(",", ":"))).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if chain != recomputed:
            return False
        prev, last_chain, last_seq = chain, chain, expected_seq
        expected_seq += 1

    if head_path.exists():
        try:
            head = json.loads(head_path.read_text())
        except json.JSONDecodeError:
            return False
        # Tail truncation produces a valid prefix; only the anchor catches it.
        if head.get("chain") != last_chain or int(head.get("seq", -1)) != last_seq:
            return False
    return True


# ---------------------------------------------------------------- disclosure ledger
def _cell_key(site_id: str, outcome: str, attribute: str, category: str) -> str:
    return f"{site_id}|{outcome}|{attribute}|{category}"


def _cell_n(block: dict) -> int | None:
    """The count a ledger entry records, whether the cell was released or suppressed."""
    metrics = block.get("metrics") or {}
    if "n" in metrics:
        return metrics["n"]
    return block.get("n")


def ledger_entries(payload: dict) -> list[dict]:
    """Extract the ledger records implied by one export.

    Records the *shape* of what was released — site, model version, outcome, cell
    definition, reported n, suppression status — and never a metric value.
    """
    site = payload["site_id"]
    version = payload["model_version"]
    release = payload.get("release_id")
    spec = payload.get("outcome_spec_hash")
    out: list[dict] = []
    for outcome, block in (payload.get("outcomes") or {}).items():
        out.append({
            "site_id": site, "model_version": version, "release_id": release,
            "outcome_spec_hash": spec, "outcome": outcome,
            "cell": _cell_key(site, outcome, "_overall", "_all"),
            "n": _cell_n(block), "status": block.get("status"),
        })
        for attr, cells in (block.get("subgroups") or {}).items():
            for cat, cell in cells.items():
                out.append({
                    "site_id": site, "model_version": version, "release_id": release,
                    "outcome_spec_hash": spec, "outcome": outcome,
                    "cell": _cell_key(site, outcome, attr, cat),
                    "n": _cell_n(cell), "status": cell.get("status"),
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
    if not prior:
        return

    # Two DIFFERENT questions, and round 3 wrongly answered both with "confirmed only"
    # (Greptile PR #4, round 4):
    #
    #   "may this release id be reused?"      -> only a CONFIRMED release blocks, so a
    #                                            crashed attempt stays retryable
    #   "might this cell already be public?"  -> an UNCONFIRMED entry must still count,
    #                                            because the crash window sits between
    #                                            publication and confirmation. During it
    #                                            the artifact is visible while its record
    #                                            is unconfirmed, and ignoring it let a
    #                                            later report release a cell that was
    #                                            already published as suppressed.
    #
    # Conflating them inverted the fail-safe direction: it turned a blocked release into a
    # possible disclosure. Differencing now assumes an unconfirmed entry MIGHT be public.
    live = confirmed_releases(ledger_path)
    prior_by_cell: dict[str, dict] = {}
    for e in prior:
        if "confirm_release_id" in e:
            continue
        prior_by_cell[e["cell"]] = e          # confirmed or not -- assume it may be public
    seen_releases = set(live)                 # replay gate: confirmed only

    current = ledger_entries(payload)

    # #13: a replayed release must not be re-counted into the aggregate.
    release = payload.get("release_id")
    if release and release in seen_releases:
        raise DisclosureError(
            f"release_id {release!r} has already been published and recorded. A replayed "
            "report would be counted as an additional site, biasing the cross-site "
            "aggregate. (An unconfirmed entry from a failed earlier attempt does not "
            "trigger this -- that release id is safe to retry.)"
        )

    floor = min_cell_size()
    for entry in current:
        before = prior_by_cell.get(entry["cell"])
        if before is None:
            continue
        was_suppressed = before.get("status") in NON_EVALUABLE_STATUSES

        # The original check: suppressed -> released.
        if was_suppressed and entry.get("status") == EVALUABLE:
            raise DisclosureError(
                f"cell {entry['cell']!r} was suppressed in a prior release and would be "
                f"released now (model_version {entry['model_version']!r}). Differencing "
                "the two releases recovers the suppressed value. Keep it suppressed, or "
                "obtain a documented disclosure-review exception."
            )

        # #11: a still-suppressed cell whose recorded n moves between releases leaks the
        # delta. Two suppressed observations of the same cell at n=4 and n=9 bound the
        # membership change to five patients; enough such deltas reconstruct the cell.
        if was_suppressed and entry.get("status") in NON_EVALUABLE_STATUSES:
            n_before, n_now = before.get("n"), entry.get("n")
            if (isinstance(n_before, int) and isinstance(n_now, int)
                    and n_before != n_now and abs(n_now - n_before) < floor):
                raise DisclosureError(
                    f"cell {entry['cell']!r} is suppressed in both releases but its "
                    f"recorded size moved from {n_before} to {n_now}. A delta smaller "
                    f"than {floor} discloses the membership change. Hold the cohort "
                    "fixed across releases, or obtain a disclosure-review exception."
                )

        # #11: a released cell whose n shrinks while siblings stay released leaks the
        # difference against the earlier total.
        if (not was_suppressed and entry.get("status") == EVALUABLE):
            n_before, n_now = before.get("n"), entry.get("n")
            if (isinstance(n_before, int) and isinstance(n_now, int)
                    and 0 < abs(n_now - n_before) < floor):
                raise DisclosureError(
                    f"cell {entry['cell']!r} changed size from {n_before} to {n_now} "
                    f"across releases, a delta below {floor}. The difference identifies "
                    "the patients who entered or left the cell."
                )


def confirm_publication(payload: dict, ledger_path: str | Path) -> None:
    """Mark a release's ledger entries as actually published.

    Two durable resources -- the ledger and the artifact -- cannot be updated atomically
    without a transaction, so every ordering has a crash window. Rather than pretend
    otherwise, the ledger records intent first (`published: false`) and confirms after the
    artifact is visible. That makes each window recoverable instead of terminal:

      - crash before the artifact is published -> unconfirmed entries, which are inert:
        they gate nothing and the release id can be retried
      - crash after publication, before confirmation -> `reconcile_ledger` finds it

    Only confirmed entries count as prior releases, because an artifact nobody could see
    cannot have been differenced against.
    """
    release = payload.get("release_id")
    if not release:
        return
    _durably_append(Path(ledger_path), json.dumps(
        {"confirm_release_id": release}, sort_keys=True, separators=(",", ":")) + "\n")


def confirmed_releases(ledger_path: str | Path) -> set:
    """Release ids whose artifact is known to have been published."""
    return {e["confirm_release_id"] for e in read_ledger(ledger_path)
            if "confirm_release_id" in e}


def reconcile_ledger(ledger_path: str | Path, published_release_ids: set) -> list[str]:
    """Release ids that were published but never confirmed, and need a confirmation.

    Run after a crash. `published_release_ids` comes from whatever the operator can see on
    the export volume. Anything in that set without a confirmation record is the narrow
    residue of a crash between publication and confirmation, and must be confirmed before
    the next release so the differencing check counts it.
    """
    return sorted(published_release_ids - confirmed_releases(ledger_path))


def append_to_ledger(payload: dict, ledger_path: str | Path) -> None:
    """Record this export's intent to release. Not yet a published release.

    Entries are written unconfirmed; `confirm_publication` marks them live once the
    artifact is actually visible.
    """
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps({**entry, "published": False}, sort_keys=True, separators=(",", ":")) + "\n"
        for entry in ledger_entries(payload)
    )
    # Durable before returning (Greptile PR #4). write_export publishes the artifact as
    # soon as this returns, so an entry sitting in the page cache is an entry that a
    # crash erases while the report stays published.
    _durably_append(path, lines)


__all__ = [
    "AuthenticationError", "confirm_publication", "confirmed_releases", "reconcile_ledger",
    "canonical_bytes", "sign_report", "verify_report",
    "record_access", "verify_access_log",
    "ledger_entries", "read_ledger", "check_cross_release_differencing", "append_to_ledger",
]
