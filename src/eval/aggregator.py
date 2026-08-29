"""Coordinating-center aggregator (U15): ingest signed site reports into one cross-site,
cross-release aggregate — fail closed at every step.

This runs at the COORDINATING CENTER, never at a site, so it is deliberately NOT part of
the `clif-validate` site package (not vendored). It is the read side of the site->aggregator
trust boundary that `src/eval/attestation.py` defines and `clif_validate.py` writes to, and
it is designed as an INDEPENDENT second line of defence: it re-checks everything itself
rather than trusting that the site ran its own gates.

Gates, in order, per report — verification and admission are SEPARATED so a bad report in a
batch is caught before ANY report touches the ledger:
  1. **Attribution + schema (reused, single trust path).** `clif_forest_plot.load_site_results`
     already performs the exact site-in-keys check, `attestation.verify_report` (HMAC), the
     `schema.validate_export` allow-list, and in-batch `release_id` replay dedup — and it reads
     each report fresh from disk, so the verified bytes are a private snapshot that no caller
     can mutate between verification and admission. Delegating to it keeps the "is this report
     trustworthy" decision in ONE place, so a future hardening there also protects the
     aggregator (and vice versa).
  2. **Releasable-status (independent of the site).** The site's `write_export` refuses to
     release anything but `reviewed_approved`, but a compromised-but-keyed site could still
     `sign_report` a `pending_review` payload with its valid secret. The aggregator re-enforces
     the releasable-status gate itself, so a report that never crossed a recorded disclosure
     review cannot enter the aggregate even with a valid signature.
  3. **Cumulative disclosure.** Under one ledger lock, `check_cross_release_differencing` runs
     against the aggregator's OWN cumulative ledger before the entry is appended and confirmed.
     The aggregator re-derives the ledger records from each signed report and checks
     differencing against its own history, so a site whose local ledger was reset or never kept
     cannot slip a previously-suppressed cell through by releasing it later (the second line of
     defence U10 names).

The aggregator stores cell *definitions and sizes* in its ledger, never metric values — the
same discipline as the site ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval import attestation as _attest
from src.eval.attestation import AuthenticationError
from src.eval.clif_forest_plot import build_forest_table, load_site_results
from src.eval.schema import RELEASABLE_DISCLOSURE_STATUSES, DisclosureError

__all__ = ["ingest_report", "aggregate_site_reports"]


def _verify_reports(paths: list, signing_keys: dict[str, bytes]) -> list[dict]:
    """Gates 1 + 2 for a whole batch, with NO ledger side effects.

    Reuses `load_site_results` for attribution + schema + in-batch replay (reading each report
    fresh from disk = an immutable snapshot), then independently enforces the releasable-status
    gate. Raises before any report is admitted, so an all-or-nothing verification precedes the
    (stateful) ledger commit. An unreadable/malformed report is unattributable -> fail closed as
    `AuthenticationError`, not a bare I/O error, matching this module's contract.
    """
    try:
        reports = load_site_results([str(p) for p in paths], signing_keys=signing_keys)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise AuthenticationError(
            f"a site report could not be read or parsed ({exc}); an unreadable report is "
            "unattributable and cannot enter the aggregate"
        ) from exc
    for report in reports:
        status = report.get("disclosure_status")
        if status not in RELEASABLE_DISCLOSURE_STATUSES:
            raise DisclosureError(
                f"report for site {report.get('site_id')!r} carries disclosure_status "
                f"{status!r}; the aggregator independently admits only "
                f"{sorted(RELEASABLE_DISCLOSURE_STATUSES)}. A validly-signed but un-reviewed "
                "report must not enter the cross-site aggregate — the signature attests "
                "origin, not that a disclosure review occurred."
            )
    return reports


def _admit_to_ledger(payload: dict, cumulative_ledger_path: str | Path) -> None:
    """Gate 3: differencing check then append-intent then confirm, under ONE lock.

    The lock spans all three exactly as the site's `write_export` does: the differencing check
    reads a snapshot and the append writes one, so an unlocked gap would let two concurrent
    ingests each pass against the same history. check/append/confirm do not self-lock.
    """
    with _attest.ledger_lock(cumulative_ledger_path):
        _attest.check_cross_release_differencing(payload, cumulative_ledger_path)
        _attest.append_to_ledger(payload, cumulative_ledger_path)
        _attest.confirm_publication(payload, cumulative_ledger_path)


def ingest_report(report_path, *, signing_keys: dict[str, bytes],
                  cumulative_ledger_path: str | Path) -> dict:
    """Verify one signed site report and admit it to the cumulative ledger, or fail closed.

    `report_path` is a path to a report JSON. `signing_keys` maps `site_id` -> the site's shared
    HMAC secret. Returns the verified payload. Raises `AuthenticationError` on an
    unattributable/altered/unreadable report and `schema.DisclosureError` on a non-releasable
    status or a cross-release differencing leak; nothing is recorded when either fires.
    """
    [payload] = _verify_reports([report_path], signing_keys)
    _admit_to_ledger(payload, cumulative_ledger_path)
    return payload


def aggregate_site_reports(paths, *, signing_keys: dict[str, bytes],
                           cumulative_ledger_path: str | Path,
                           metrics: list[str] | None = None) -> dict:
    """Verify every signed site report, admit each to the cumulative ledger, build the panel.

    Verification (gates 1-2) runs for the WHOLE batch first, so a report that fails attribution,
    schema, in-batch replay, or the releasable-status gate aborts the aggregate before any
    ledger write. The reports are also required to share one `outcome_spec_hash` — pooling cells
    defined under different outcome specs into one headline figure is an analytic error, not a
    comparison.

    NON-ATOMIC across the ledger commit (documented, fail-closed): the differencing gate (3) is
    stateful, so if report[k] trips differencing, reports[0..k-1] are already confirmed in the
    cumulative ledger. That is correct — those reports were valid and are now prior history — but
    a naive retry of the same batch replays them; resume with only the not-yet-admitted reports.
    A cross-host aggregator sharing the ledger over a network filesystem needs a real coordinator
    (the `ledger_lock` flock is single-host advisory, per attestation.py).
    """
    reports = _verify_reports(list(paths), signing_keys)
    specs = {r.get("outcome_spec_hash") for r in reports}
    if len(specs) > 1:
        raise DisclosureError(
            f"site reports declare more than one outcome_spec_hash {sorted(specs)}; refusing to "
            "pool cells defined under different outcome specifications into one aggregate"
        )
    for payload in reports:
        _admit_to_ledger(payload, cumulative_ledger_path)
    return {
        "n_sites": len(reports),
        "site_ids": [r.get("site_id") for r in reports],
        "table": build_forest_table(reports, metrics),
    }
