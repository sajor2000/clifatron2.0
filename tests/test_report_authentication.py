"""Site-to-aggregator report authentication and the cumulative disclosure ledger.

U5 owns this direction of the trust model; releaser-to-site distribution trust is U11's.
Without these controls a forged or altered site report silently poisons the cross-site
comparison, and repeated releases defeat single-report suppression by differencing.
"""

import tempfile
import unittest
from pathlib import Path

from src.eval import attestation as A
from src.eval import schema as S
from src.eval.schema import DisclosureError

KEY = b"site-01-shared-secret"


def _label_validity():
    return {
        "outcome_definition_id": "map_below_65_48h",
        "outcome_definition_version": "1.0.0",
        "status_counts": {state: 10 for state in S.U1_OUTCOME_STATES},
        "evaluable_denominator_fraction": 0.82,
    }


def _report(site="SITE-01", version="v0", outcomes=None, release=None):
    return {
        "schema_version": S.METRIC_SCHEMA_VERSION,
        "metric_version": "1.0.0",
        "model_bundle_id": "bundle-abc",
        "model_version": version,
        "vocab_hash": "a" * 16,
        "outcome_spec_hash": "b" * 16,
        "clif_version": "2.1",
        "site_id": site,
        "site_role": "development",
        "partition_role": "test",
        "disclosure_status": "reviewed_approved",
        "release_id": release or f"rel-{site}-{version}",
        "outcomes": outcomes if outcomes is not None else {
            "map_below_65_48h": {
                "status": S.EVALUABLE,
                "label_validity": _label_validity(),
                "metrics": {"auroc": 0.78, "n": 400, "prevalence": 0.2},
            }
        },
    }


class ReportSigningTest(unittest.TestCase):
    def test_signed_report_verifies(self):
        signed = A.sign_report(_report(), KEY)
        self.assertIs(A.verify_report(signed, KEY), signed)

    def test_unsigned_report_is_rejected(self):
        """An unsigned report is not 'unverified but probably fine' -- it is a report
        whose origin nobody can attest."""
        with self.assertRaises(A.AuthenticationError) as ctx:
            A.verify_report(_report(), KEY)
        self.assertIn("no signature", str(ctx.exception))

    def test_altered_report_is_rejected(self):
        signed = A.sign_report(_report(), KEY)
        signed["outcomes"]["map_below_65_48h"]["metrics"]["auroc"] = 0.95
        with self.assertRaises(A.AuthenticationError) as ctx:
            A.verify_report(signed, KEY)
        self.assertIn("altered", str(ctx.exception))

    def test_report_signed_with_a_different_key_is_rejected(self):
        signed = A.sign_report(_report(), b"some-other-site-key")
        with self.assertRaises(A.AuthenticationError):
            A.verify_report(signed, KEY)

    def test_signature_is_order_independent(self):
        """Two structurally identical reports must produce the same signature
        regardless of construction order."""
        a = _report()
        b = {k: a[k] for k in reversed(list(a))}
        self.assertEqual(A.canonical_bytes(a), A.canonical_bytes(b))

    def test_signature_does_not_cover_itself(self):
        signed = A.sign_report(_report(), KEY)
        self.assertNotIn(b"signature", A.canonical_bytes(signed))

    def test_empty_key_refuses_to_sign(self):
        with self.assertRaises(A.AuthenticationError):
            A.sign_report(_report(), b"")

    def test_signed_report_still_passes_the_export_allow_list(self):
        """A base64/hex signature must not trip the local-path heuristic."""
        S.validate_export(A.sign_report(_report(), KEY))


class AccessLogTest(unittest.TestCase):
    def test_accesses_are_recorded_and_the_chain_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "access.jsonl"
            for i in range(3):
                A.record_access(log, model_version="v0", actor_role="aggregator",
                                artifact_id=f"artifact-{i}", action="unseal")
            self.assertTrue(A.verify_access_log(log))
            self.assertEqual(len(log.read_text().strip().splitlines()), 3)

    def test_a_deleted_record_breaks_the_chain(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "access.jsonl"
            for i in range(3):
                A.record_access(log, model_version="v0", actor_role="aggregator",
                                artifact_id=f"artifact-{i}", action="unseal")
            lines = log.read_text().splitlines()
            log.write_text("\n".join([lines[0], lines[2]]) + "\n")
            self.assertFalse(A.verify_access_log(log), "tampering must be detectable")

    def test_an_edited_record_breaks_the_chain(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "access.jsonl"
            A.record_access(log, model_version="v0", actor_role="site_operator",
                            artifact_id="a1", action="export")
            log.write_text(log.read_text().replace("site_operator", "aggregator"))
            self.assertFalse(A.verify_access_log(log))

    def test_absent_log_is_trivially_intact(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertTrue(A.verify_access_log(Path(td) / "nope.jsonl"))


class DisclosureLedgerTest(unittest.TestCase):
    def test_ledger_records_cell_shape_not_cell_contents(self):
        entries = A.ledger_entries(_report())
        self.assertTrue(entries)
        for e in entries:
            self.assertNotIn("auroc", e)
            self.assertNotIn("prevalence", e)
            self.assertIn("n", e)
            self.assertIn("status", e)

    def test_subgroup_cells_get_their_own_ledger_records(self):
        report = _report()
        report["outcomes"]["map_below_65_48h"]["subgroups"] = {
            "sex": {"F": {"status": S.EVALUABLE, "n": 200, "prevalence": 0.2},
                    "M": {"status": S.INSUFFICIENT_N, "n_band": "<10"}}
        }
        cells = {e["cell"] for e in A.ledger_entries(report)}
        self.assertTrue(any("sex|F" in c for c in cells))
        self.assertTrue(any("sex|M" in c for c in cells))

    def test_first_release_passes_the_differencing_check(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"
            A.check_cross_release_differencing(_report(), ledger)  # no prior releases

    def test_releasing_a_previously_suppressed_cell_is_blocked(self):
        """Each release is individually compliant; together they are not. Suppression
        that only looks at the current report cannot see this."""
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"

            first = _report(version="v0")
            first["outcomes"]["map_below_65_48h"]["subgroups"] = {
                "sex": {"M": {"status": S.INSUFFICIENT_N, "n": 4}}}
            A.append_to_ledger(first, ledger)
            A.confirm_publication(first, ledger)   # it was actually published

            second = _report(version="v1")
            second["outcomes"]["map_below_65_48h"]["subgroups"] = {
                "sex": {"M": {"status": S.EVALUABLE, "n": 60}}}
            with self.assertRaises(DisclosureError) as ctx:
                A.check_cross_release_differencing(second, ledger)
            self.assertIn("suppressed in a prior release", str(ctx.exception))

    def test_a_cell_that_stays_suppressed_is_allowed(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"
            first = _report(version="v0")
            first["outcomes"]["map_below_65_48h"]["subgroups"] = {
                "sex": {"M": {"status": S.INSUFFICIENT_N, "n": 4}}}
            A.append_to_ledger(first, ledger)
            A.confirm_publication(first, ledger)

            second = _report(version="v1")
            second["outcomes"]["map_below_65_48h"]["subgroups"] = {
                "sex": {"M": {"status": S.INSUFFICIENT_N, "n": 4}}}
            A.check_cross_release_differencing(second, ledger)

    def test_a_different_site_is_not_confused_with_the_suppressed_one(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"
            first = _report(site="SITE-01")
            first["outcomes"]["map_below_65_48h"]["subgroups"] = {
                "sex": {"M": {"status": S.INSUFFICIENT_N, "n": 4}}}
            A.append_to_ledger(first, ledger)
            A.confirm_publication(first, ledger)

            other = _report(site="SITE-02")
            other["outcomes"]["map_below_65_48h"]["subgroups"] = {
                "sex": {"M": {"status": S.EVALUABLE, "n": 60}}}
            A.check_cross_release_differencing(other, ledger)

    def test_an_unconfirmed_entry_gates_nothing_and_is_retryable(self):
        """Greptile PR #4, round 3. A release recorded but never published must not
        strand its release id or block later releases -- nobody could have differenced
        against an artifact that never became visible."""
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"
            attempt = _report(version="v0", release="rel-A")
            attempt["outcomes"]["map_below_65_48h"]["subgroups"] = {
                "sex": {"M": {"status": S.INSUFFICIENT_N, "n": 4}}}
            A.append_to_ledger(attempt, ledger)      # publication then failed

            # Same release id retries cleanly.
            A.check_cross_release_differencing(attempt, ledger)

            # But it DOES still protect the cell: the crash window sits between
            # publication and confirmation, so an unconfirmed entry might already be
            # public and must be assumed so (Greptile PR #4, round 4).
            later = _report(version="v1", release="rel-B")
            later["outcomes"]["map_below_65_48h"]["subgroups"] = {
                "sex": {"M": {"status": S.EVALUABLE, "n": 60}}}
            with self.assertRaises(DisclosureError):
                A.check_cross_release_differencing(later, ledger)

    def test_retryability_and_cell_protection_are_separate_questions(self):
        """Round 3 answered both with "confirmed only", which inverted the fail-safe
        direction: a blocked release became a possible disclosure."""
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"
            attempt = _report(release="rel-A")
            attempt["outcomes"]["map_below_65_48h"]["subgroups"] = {
                "sex": {"M": {"status": S.INSUFFICIENT_N, "n": 4}}}
            A.append_to_ledger(attempt, ledger)      # published, then crashed

            # Retry of the SAME release id: allowed.
            A.check_cross_release_differencing(attempt, ledger)

            # A DIFFERENT release un-suppressing that cell: refused.
            other = _report(release="rel-B")
            other["outcomes"]["map_below_65_48h"]["subgroups"] = {
                "sex": {"M": {"status": S.EVALUABLE, "n": 60}}}
            with self.assertRaises(DisclosureError):
                A.check_cross_release_differencing(other, ledger)

    def test_reconcile_names_published_but_unconfirmed_releases(self):
        """The one remaining crash window is recoverable rather than silent."""
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"
            A.append_to_ledger(_report(release="rel-A"), ledger)
            self.assertEqual(A.reconcile_ledger(ledger, {"rel-A"}), ["rel-A"])
            A.confirm_publication(_report(release="rel-A"), ledger)
            self.assertEqual(A.reconcile_ledger(ledger, {"rel-A"}), [])

    def test_ledger_is_append_only_across_releases(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "ledger.jsonl"
            A.append_to_ledger(_report(version="v0"), ledger)
            first_count = len(A.read_ledger(ledger))
            A.append_to_ledger(_report(version="v1"), ledger)
            self.assertEqual(len(A.read_ledger(ledger)), first_count * 2)


if __name__ == "__main__":
    unittest.main()
