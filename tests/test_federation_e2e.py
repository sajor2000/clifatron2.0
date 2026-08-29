"""U15: the synthetic federation loop, end to end and fail-closed.

Releaser -> site -> aggregator on synthetic fixtures, driving the REAL releaser
(`synthetic_bundle`), the REAL site CLI (`clif_validate.main`), and the new
`src/eval/aggregator.py`. No mocks of either trust boundary: the releaser->site
Ed25519 signature (U11) and the site->aggregator HMAC signature (U5) are both
exercised as distinct gates, and the aggregator keeps its OWN cumulative
cross-release ledger — a second line of defence that does not trust a site to have
kept its local ledger honestly.

Data-free, CPU, tiny GPT-2. Expensive artifacts (site, bundle) build once per class;
one site's real signed report is produced once and the fail-closed cases derive from
copies, so the whole file runs a bounded number of real CLI ceremonies.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src.eval import aggregator as agg
from src.eval import attestation as attest
from src.eval import schema as S
from src.eval.synthetic_bundle import build_synthetic_bundle, build_synthetic_site


def _reset_policy_pin():
    os.environ.pop(S.POLICY_OVERRIDE_ENV, None)
    S.min_cell_size.cache_clear()
    S.max_dropped_fraction.cache_clear()


class FederationE2ETest(unittest.TestCase):
    """One synthetic site + signed bundle; SITE-A's real report built once and reused."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.work = Path(cls._tmp.name)
        cls._old_cwd = os.getcwd()
        cls._old_tz = os.environ.get("TZ")
        os.environ["TZ"] = "America/Chicago"  # exercise the DuckDB UTC pin (as U9 tests do)
        if hasattr(time, "tzset"):
            time.tzset()
        os.chdir(cls.work)
        try:
            cls.site = cls.work / "site"
            cls.episodes = build_synthetic_site(cls.site)
            cls.bundle_dir = build_synthetic_bundle(cls.work / "bundle", cls.site, cls.episodes)
            cls.trust_roles = cls.work / "trust_roles.yaml"
            cls.access_key = cls.work / "access.key"
            cls.access_key.write_bytes(b"synthetic-access-chain-key")
            # Per-site report-signing secrets (site -> aggregator HMAC). The aggregator
            # holds the public registry {site_id: secret}.
            cls.site_keys = {"SITE-A": b"site-a-report-secret",
                             "SITE-B": b"site-b-report-secret"}
            cls._keyfiles = {}
            for sid, secret in cls.site_keys.items():
                kf = cls.work / f"{sid}.key"
                kf.write_text(secret.hex())
                cls._keyfiles[sid] = kf
            # SITE-A's real signed report, produced once via the full ceremony.
            cls.report_a = cls._run_site("SITE-A", "rel-a", "site_a")
        except BaseException:
            cls._restore_env()
            cls._tmp.cleanup()
            raise

    @classmethod
    def _restore_env(cls):
        os.chdir(cls._old_cwd)
        if cls._old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = cls._old_tz
        if hasattr(time, "tzset"):
            time.tzset()
        _reset_policy_pin()

    @classmethod
    def tearDownClass(cls):
        cls._restore_env()
        cls._tmp.cleanup()

    def tearDown(self):
        _reset_policy_pin()

    @classmethod
    def _run_site(cls, site_id: str, release_id: str, tag: str) -> Path:
        """Drive the REAL site CLI through draft -> approve and return the report path."""
        from src.eval.clif_validate import main

        out = f"output/final_no_phi/{tag}.json"
        base = [
            "clif_validate",
            "--checkpoint", str(cls.bundle_dir),
            "--data", str(cls.site),
            "--episode-artifact", str(cls.episodes),
            "--site-id", site_id,
            "--release-id", release_id,
            "--out", out,
            "--ledger", f"output/intermediate_phi/{tag}_ledger.jsonl",
            "--access-log", f"output/intermediate_phi/{tag}_access.jsonl",
            "--shard-dir", f"output/intermediate_phi/{tag}_shards",
            "--signing-key-file", str(cls._keyfiles[site_id]),
            "--access-log-key-file", str(cls.access_key),
            "--trust-roles", str(cls.trust_roles),
            "--rollback-state", f"output/intermediate_phi/{tag}_rollback.json",
        ]
        with mock.patch("sys.argv", base):  # draft
            main()
        draft_hash = Path(out + ".draft.sha256").read_text().strip()
        with mock.patch("sys.argv", base + ["--approved", "--approved-hash", draft_hash]):
            main()
        _reset_policy_pin()
        return Path(out)

    def _resign(self, payload: dict, site_id: str) -> dict:
        """Re-sign a (possibly mutated) payload with a site's key — a valid signature over
        different content, the way a compromised-but-keyed site or a re-release would look."""
        return attest.sign_report({k: v for k, v in payload.items() if k != "signature"},
                                  self.site_keys[site_id])

    # ----------------------------------------------------------------- happy path
    def test_full_loop_two_sites_aggregate_and_ledger(self):
        """Releaser -> two sites -> aggregator: both reports verify, the panel carries two
        sites, and the aggregator's cumulative ledger records the released cells — with no
        patient-level field, metric-less suppressed cells aside, anywhere in the output."""
        report_b = self._run_site("SITE-B", "rel-b", "site_b")
        cum = "output/intermediate_phi/aggregate_ledger.jsonl"
        panel = agg.aggregate_site_reports(
            [str(self.report_a), str(report_b)],
            signing_keys=self.site_keys, cumulative_ledger_path=cum)

        self.assertEqual(panel["n_sites"], 2)
        self.assertEqual(set(panel["site_ids"]), {"SITE-A", "SITE-B"})
        # The cumulative ledger recorded both releases as confirmed.
        self.assertEqual(attest.confirmed_releases(cum), {"rel-a", "rel-b"})
        # No patient-level fields leaked into the aggregate: no local path anywhere, and the
        # per-metric blocks carry only summary statistics — never a raw patient count `n`.
        blob = json.dumps(panel)
        self.assertNotIn(str(self.site), blob)      # no site data path
        self.assertNotIn("/Users", blob)            # no absolute local path of any kind
        allowed = {"values", "mean", "std", "min", "max", "n_sites"}
        for row in panel["table"]:  # list of per-outcome rows
            for metric in ("auroc", "auprc", "ece"):
                stats = row[metric]
                self.assertTrue(set(stats).issubset(allowed),
                                f"panel stats leaked a non-allow-listed key: {set(stats)}")
                self.assertNotIn("n", stats)        # summary carries n_sites, never patient n

    # ----------------------------------------------------------------- fail closed
    def test_a_tampered_report_is_refused(self):
        payload = json.loads(self.report_a.read_text())
        # Change a signed field without re-signing: the HMAC must no longer verify.
        payload["model_version"] = str(payload["model_version"]) + "-forged"
        tampered = self.work / "tampered.json"
        tampered.write_text(json.dumps(payload))
        cum = "output/intermediate_phi/tamper_ledger.jsonl"
        with self.assertRaises(attest.AuthenticationError):
            agg.ingest_report(str(tampered), signing_keys=self.site_keys,
                              cumulative_ledger_path=cum)
        self.assertEqual(attest.read_ledger(cum), [])  # nothing entered the ledger

    def test_an_unregistered_site_is_refused(self):
        payload = self._resign({**json.loads(self.report_a.read_text()),
                                "site_id": "SITE-UNKNOWN"}, "SITE-A")
        # Signed with A's key but claims an unregistered site: no key to attribute it.
        rp = self.work / "unregistered.json"
        rp.write_text(json.dumps(payload))
        cum = "output/intermediate_phi/unreg_ledger.jsonl"
        with self.assertRaises(attest.AuthenticationError):
            agg.ingest_report(str(rp), signing_keys=self.site_keys, cumulative_ledger_path=cum)
        self.assertEqual(attest.read_ledger(cum), [])  # attribution precedes any ledger write

    def test_a_report_relabeled_to_a_different_registered_site_is_refused(self):
        """A's real content relabeled SITE-B but still signed with A's key: verifying against
        B's registered key fails, so a cross-key confusion cannot enter the aggregate."""
        payload = {**json.loads(self.report_a.read_text()), "site_id": "SITE-B"}
        payload = self._resign(payload, "SITE-A")  # signed with A's key, claims B
        rp = self.work / "relabeled.json"
        rp.write_text(json.dumps(payload))
        cum = "output/intermediate_phi/relabel_ledger.jsonl"
        with self.assertRaises(attest.AuthenticationError):
            agg.ingest_report(str(rp), signing_keys=self.site_keys, cumulative_ledger_path=cum)
        self.assertEqual(attest.read_ledger(cum), [])

    def test_a_non_releasable_status_report_is_refused(self):
        """The independent approval gate: a validly-signed pending_review (draft) report is
        refused by the aggregator, which does not trust the site to have run its own gate."""
        payload = {k: v for k, v in json.loads(self.report_a.read_text()).items()
                   if k != "signature"}
        payload["disclosure_status"] = S.DRAFT_DISCLOSURE_STATUS
        payload = attest.sign_report(payload, self.site_keys["SITE-A"])  # a genuinely signed draft
        rp = self.work / "draft.json"
        rp.write_text(json.dumps(payload))
        cum = "output/intermediate_phi/draft_ledger.jsonl"
        with self.assertRaises(S.DisclosureError):
            agg.ingest_report(str(rp), signing_keys=self.site_keys, cumulative_ledger_path=cum)
        self.assertEqual(attest.read_ledger(cum), [])

    def test_a_schema_violating_report_is_refused(self):
        """Gate 2: a validly-signed report smuggling a stray top-level key fails the schema
        allow-list before any ledger write."""
        payload = {k: v for k, v in json.loads(self.report_a.read_text()).items()
                   if k != "signature"}
        payload["smuggled_field"] = "not in the export allow-list"
        payload = attest.sign_report(payload, self.site_keys["SITE-A"])
        rp = self.work / "smuggled.json"
        rp.write_text(json.dumps(payload))
        cum = "output/intermediate_phi/smuggle_ledger.jsonl"
        with self.assertRaises(S.DisclosureError):
            agg.ingest_report(str(rp), signing_keys=self.site_keys, cumulative_ledger_path=cum)
        self.assertEqual(attest.read_ledger(cum), [])

    def test_a_replayed_release_is_refused(self):
        cum = "output/intermediate_phi/replay_ledger.jsonl"
        agg.ingest_report(str(self.report_a), signing_keys=self.site_keys,
                          cumulative_ledger_path=cum)  # first ingest ok
        before = attest.read_ledger(cum)
        with self.assertRaises(S.DisclosureError):
            agg.ingest_report(str(self.report_a), signing_keys=self.site_keys,
                              cumulative_ledger_path=cum)  # same release id again
        # The refused replay recorded nothing new; only the first release is confirmed.
        self.assertEqual(attest.confirmed_releases(cum), {"rel-a"})
        self.assertEqual(attest.read_ledger(cum), before)

    def test_the_aggregator_blocks_cross_release_differencing(self):
        """The load-bearing scenario (KTD-U15d): a site suppresses a cell in release 1 then
        would release it in release 2. Even with NO site-local ledger, the aggregator's own
        cumulative ledger catches the differencing leak and refuses the second report."""
        base = json.loads(self.report_a.read_text())
        outcome = next(iter(base["outcomes"]))
        # Precondition (asserted, not assumed): the base cell is EVALUABLE, so release 2's cell
        # trips the sticky-suppression branch against release 1's suppressed cell.
        self.assertEqual(base["outcomes"][outcome]["status"], S.EVALUABLE)
        # Reuse the real cell's label_validity so the suppressed block is schema-valid; a
        # suppressed cell just carries status + reason + label_validity (no metrics).
        label_validity = base["outcomes"][outcome]["label_validity"]
        # Release 1: the headline cell suppressed (coverage-insufficient, no metrics).
        r1 = {k: v for k, v in base.items() if k != "signature"}
        r1["release_id"] = "rel-diff-1"
        r1["outcomes"] = {outcome: {"status": S.COVERAGE_INSUFFICIENT,
                                    "reason": "synthetic suppression",
                                    "label_validity": label_validity}}
        r1 = attest.sign_report(r1, self.site_keys["SITE-A"])
        # Release 2: the SAME cell now released (evaluable) — differencing recovers it.
        r2 = {k: v for k, v in base.items() if k != "signature"}
        r2["release_id"] = "rel-diff-2"
        r2 = attest.sign_report(r2, self.site_keys["SITE-A"])
        p1, p2 = self.work / "diff1.json", self.work / "diff2.json"
        p1.write_text(json.dumps(r1))
        p2.write_text(json.dumps(r2))
        cum = "output/intermediate_phi/diff_ledger.jsonl"
        agg.ingest_report(str(p1), signing_keys=self.site_keys, cumulative_ledger_path=cum)
        with self.assertRaises(S.DisclosureError):
            agg.ingest_report(str(p2), signing_keys=self.site_keys, cumulative_ledger_path=cum)
        # rel-diff-2 must not have been recorded as confirmed.
        self.assertNotIn("rel-diff-2", attest.confirmed_releases(cum))

    def test_an_unsigned_bundle_cannot_produce_a_report_to_aggregate(self):
        """Composition of the two gates: a site given an unsigned bundle cannot complete an
        approved release (U11), so nothing exists to aggregate. Assert the site refuses."""
        from src.eval.clif_validate import main

        out = "output/final_no_phi/unsigned_site.json"
        argv = [
            "clif_validate", "--checkpoint", str(self.bundle_dir), "--data", str(self.site),
            "--episode-artifact", str(self.episodes), "--site-id", "SITE-A",
            "--release-id", "rel-unsigned", "--out", out,
            "--ledger", "output/intermediate_phi/unsigned_ledger.jsonl",
            "--access-log", "output/intermediate_phi/unsigned_access.jsonl",
            "--shard-dir", "output/intermediate_phi/unsigned_shards",
            "--signing-key-file", str(self._keyfiles["SITE-A"]),
            "--access-log-key-file", str(self.access_key),
            "--allow-unsigned", "--approved", "--approved-hash", "0" * 64,
            "--rollback-state", "output/intermediate_phi/unsigned_rollback.json",
        ]
        with mock.patch("sys.argv", argv):
            with self.assertRaises(SystemExit):
                main()
        self.assertFalse(Path(out).exists())


if __name__ == "__main__":
    unittest.main()
