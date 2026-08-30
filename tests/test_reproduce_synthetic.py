"""U19: the one-command synthetic reproduction runs end to end and stays disclosure-safe.

Smoke-first: the entrypoint's value is that it RUNS the full releaser -> site -> aggregator
loop on synthetic fixtures and yields a two-site aggregate with no patient-level data. The
deep fail-closed coverage lives in tests/test_federation_e2e.py (the gates this composes);
this test only proves the reproduction wires them and stays governed + data-free.
"""

import json
import unittest
from pathlib import Path

from src.eval import reproduce_synthetic as repro


class ReproduceSyntheticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = repro.run_synthetic_federation()  # builds fixtures under its own temp dir

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


if __name__ == "__main__":
    unittest.main()
