"""Packaging consistency guards (U11) — cheap, offline, deterministic.

A full offline-wheelhouse clean-install test (build a wheelhouse, install into a fresh
env with no network, validate a signed bundle) needs torch-sized downloads and a real
build host, so it lives in PACKAGING.md as a documented CI/ops procedure rather than a
unit test. What IS cheap to guard here is the discipline that clean-install depends on:
the committed uv.lock and SBOM.json must stay in sync with pyproject, and the security
floor the review set (cryptography) must actually be reflected in the pinned artifacts —
so a dependency bump that forgets to re-lock / re-SBOM fails HERE, not at a site.
"""

import json
import tomllib
import unittest
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


class PackagingArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.pyproject = PKG_ROOT / "pyproject.toml"
        self.lock = PKG_ROOT / "uv.lock"
        self.sbom = PKG_ROOT / "SBOM.json"
        for p in (self.pyproject, self.lock, self.sbom):
            if not p.exists():
                self.skipTest(f"{p.name} not present (packaging artifacts not built here)")
        self.declared = tomllib.loads(self.pyproject.read_text())["project"]["dependencies"]

    def _dep_names(self):
        # "torch>=2.4" -> "torch"; strip version/extras/markers.
        import re
        out = []
        for spec in self.declared:
            out.append(_norm(re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0]))
        return out

    def test_uv_lock_pins_every_declared_dependency(self):
        """Every runtime dependency in pyproject must appear as a locked package, so the
        lock a site installs from actually covers what the package declares."""
        locked = {_norm(pkg["name"])
                  for pkg in tomllib.loads(self.lock.read_text()).get("package", [])}
        missing = [d for d in self._dep_names() if d not in locked]
        self.assertEqual(missing, [], f"declared deps absent from uv.lock: {missing} "
                                      "(re-run `uv lock` after changing dependencies)")

    def test_cryptography_floor_is_honoured_in_the_lock(self):
        """The review raised the cryptography floor past known advisories; the pinned
        version in the lock must clear it, not silently resolve to an older wheel."""
        pkgs = {_norm(p["name"]): p["version"]
                for p in tomllib.loads(self.lock.read_text()).get("package", [])}
        self.assertIn("cryptography", pkgs)
        ver = tuple(int(x) for x in pkgs["cryptography"].split(".")[:3])
        self.assertGreaterEqual(ver, (44, 0, 1),
                                f"locked cryptography {pkgs['cryptography']} is below the "
                                "44.0.1 advisory floor set in review")

    def test_sbom_is_valid_cyclonedx_and_lists_cryptography(self):
        doc = json.loads(self.sbom.read_text())
        self.assertEqual(doc.get("bomFormat"), "CycloneDX")
        self.assertTrue(str(doc.get("specVersion", "")).startswith("1."))
        names = {_norm(c.get("name", "")) for c in doc.get("components", [])}
        self.assertIn("cryptography", names,
                      "SBOM.json is stale — re-generate it from uv.lock (see PACKAGING.md)")


if __name__ == "__main__":
    unittest.main()
