"""Coordinating-center aggregator (U15): ingest signed site reports into one cross-site,
cross-release aggregate — fail closed at every step.

This runs at the COORDINATING CENTER, never at a site, so it is deliberately NOT part of
the `clif-validate` site package (not vendored). It is the read side of the site->aggregator
trust boundary that `src/eval/attestation.py` defines and `clif_validate.py` writes to.

Three fail-closed gates, in order, per report:
  1. **Attribution.** The report's `site_id` must have a registered signing secret, and its
     detached HMAC signature must verify against it (`attestation.verify_report`). An
     unsigned, altered, or unregistered report cannot enter the aggregate — otherwise a
     forged report counts as another site and biases every cross-site figure.
  2. **Schema allow-list.** `schema.validate_export` — the report may carry only the fields
     the export contract allows, so a site JSON cannot smuggle a stray key in as an outcome.
  3. **Cumulative disclosure.** Under one ledger lock, `check_cross_release_differencing`
     runs against the aggregator's OWN cumulative ledger before the entry is appended and
     confirmed. This is the independent second line of defence U10 names: the site keeps a
     local ledger, but the aggregator re-derives the ledger records from each signed report
     and checks differencing against its own history, so a site whose local ledger was reset
     or never kept cannot slip a previously-suppressed cell through by releasing it later.

The aggregator stores cell *definitions and sizes* in its ledger, never metric values — the
same discipline as the site ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval import attestation as _attest
from src.eval.attestation import AuthenticationError
from src.eval.clif_forest_plot import build_forest_table
from src.eval.schema import validate_export

__all__ = ["ingest_report", "aggregate_site_reports"]


def ingest_report(report, *, signing_keys: dict[str, bytes],
                  cumulative_ledger_path: str | Path) -> dict:
    """Verify one signed site report and admit it to the cumulative ledger, or fail closed.

    `report` is a report dict or a path to one. `signing_keys` maps `site_id` -> the site's
    shared HMAC secret. Returns the verified payload. Raises `AuthenticationError` on an
    unattributable/altered report and `schema.DisclosureError` on a cross-release
    differencing leak; nothing is recorded when either fires.
    """
    payload = report if isinstance(report, dict) else json.loads(Path(report).read_text())

    site = payload.get("site_id")
    if not signing_keys or site not in signing_keys:
        raise AuthenticationError(
            f"no signing key registered for site {site!r}; an unattributable report cannot "
            "enter the cross-site aggregate"
        )
    _attest.verify_report(payload, signing_keys[site])  # HMAC — raises on tamper/absent/empty
    validate_export(payload)                            # allow-list, not accept-anything

    # One lock spans check -> append-intent -> confirm, exactly as the site's write_export
    # does: the differencing check reads a snapshot and the append writes one, so an
    # unlocked gap between them would let two concurrent ingests each pass against the same
    # history. append/confirm/check do not self-lock (the caller owns the lock).
    with _attest.ledger_lock(cumulative_ledger_path):
        _attest.check_cross_release_differencing(payload, cumulative_ledger_path)
        _attest.append_to_ledger(payload, cumulative_ledger_path)
        _attest.confirm_publication(payload, cumulative_ledger_path)
    return payload


def aggregate_site_reports(paths, *, signing_keys: dict[str, bytes],
                           cumulative_ledger_path: str | Path,
                           metrics: list[str] | None = None) -> dict:
    """Ingest every signed site report, then build the multi-site panel.

    Each report passes the three gates in `ingest_report` before it contributes to the
    panel, so a single bad report aborts the aggregate rather than silently biasing it. The
    panel carries only the schema's allow-listed metric blocks (via `build_forest_table`) —
    no patient-level fields, and a non-evaluable cell contributes its status, never a number.
    """
    accepted = [ingest_report(p, signing_keys=signing_keys,
                              cumulative_ledger_path=cumulative_ledger_path) for p in paths]
    return {
        "n_sites": len(accepted),
        "site_ids": [r.get("site_id") for r in accepted],
        "table": build_forest_table(accepted, metrics),
    }
