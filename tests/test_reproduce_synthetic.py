"""U19: the one-command synthetic reproduction runs end to end and stays disclosure-safe.

Smoke-first: the entrypoint's value is that it RUNS the full releaser -> site -> aggregator
loop on synthetic fixtures and yields a two-site aggregate with no patient-level data. The
deep fail-closed coverage lives in tests/test_federation_e2e.py (the gates this composes);
this test only proves the reproduction wires them, stays governed + data-free, and leaves the
caller's process untouched.
"""

import json
import os
import sys
import unittest
from pathlib import Path

from src.eval import reproduce_synthetic as repro
from src.eval import schema as S

_SENTINEL_POLICY = "/sentinel/policy.yaml"
_SENTINEL_ACCESS = "/sentinel/access.key"


class ReproduceSyntheticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Snapshot process state and plant sentinels so we can prove the run leaves the caller
        # pristine — restoring any prior values afterward (save/restore, never blind delete).
        cls._argv_before = list(sys.argv)
        cls._cwd_before = os.getcwd()
        prior_env = {k: os.environ.get(k)
                     for k in (S.POLICY_OVERRIDE_ENV, "CLIF_ACCESS_LOG_KEY_FILE")}

        def _restore():
            for key, prior in prior_env.items():
                if prior is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prior
            S.min_cell_size.cache_clear()
            S.max_dropped_fraction.cache_clear()

        # Register restoration BEFORE mutating the env, so a failure in the run below still
        # restores the caller's state (a raise in setUpClass would skip tearDownClass).
        cls.addClassCleanup(_restore)
        os.environ[S.POLICY_OVERRIDE_ENV] = _SENTINEL_POLICY
        os.environ["CLIF_ACCESS_LOG_KEY_FILE"] = _SENTINEL_ACCESS
        cls.panel = repro.run_synthetic_federation()  # builds fixtures under its own temp dir
        cls._argv_after = list(sys.argv)
        cls._cwd_after = os.getcwd()
        cls._policy_after = os.environ.get(S.POLICY_OVERRIDE_ENV)
        cls._access_after = os.environ.get("CLIF_ACCESS_LOG_KEY_FILE")

    def test_produces_a_two_site_aggregate(self):
        self.assertEqual(self.panel["n_sites"], 2)
        self.assertEqual(set(self.panel["site_ids"]), {"SITE-A", "SITE-B"})

    def test_the_aggregate_carries_no_patient_level_data(self):
        blob = json.dumps(self.panel)
        self.assertNotIn("/Users", blob)      # no absolute local path
        self.assertNotIn(".key", blob)        # no key file path leaked
        allowed = {"values", "mean", "std", "min", "max", "n_sites"}
        for row in self.panel["table"]:       # list of per-outcome rows
            for metric in ("auroc", "auprc", "ece"):
                stats = row[metric]
                self.assertTrue(set(stats).issubset(allowed),
                                f"panel stats leaked a non-allow-listed key: {set(stats)}")
                self.assertNotIn("n", stats)  # summary carries n_sites, never a patient n

    def test_the_reproduction_uses_the_governed_path_not_an_escape_hatch(self):
        """A produced aggregate already implies the governed ceremony (an unsigned bundle
        could not approve), but pin it statically too: the entrypoint never reaches for
        --allow-unsigned, so the reproduction can never demonstrate an ungoverned release."""
        source = Path(repro.__file__).read_text()
        self.assertNotIn("--allow-unsigned", source)

    def test_the_run_leaves_the_caller_process_pristine(self):
        """The site CLIs run out-of-process, so their sys.argv/env mutations never reach here;
        the in-process fixture build changes cwd + the policy pin but restores both."""
        self.assertEqual(self._cwd_after, self._cwd_before)
        self.assertEqual(self._argv_after, self._argv_before)
        self.assertEqual(self._policy_after, _SENTINEL_POLICY)   # restored by the runner
        self.assertEqual(self._access_after, _SENTINEL_ACCESS)   # never touched (child-only)


if __name__ == "__main__":
    unittest.main()
