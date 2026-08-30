"""Reproducibility guard (U16): the committed root uv.lock must stay in sync with pyproject.

CI installs the data-free suites from this lock with `uv run --frozen`, so a lock that has
drifted from `pyproject.toml` (a dependency added or bumped without re-locking) would make CI
reproduce a DIFFERENT environment than the one the results were produced in. This guard is
cheap and offline (parse the lock + pyproject); a MISSING lock FAILS rather than skips, so a
packaging change that deletes it cannot report green. Mirrors clif-validate/tests/test_packaging.py.
"""

import os
import re
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _norm(name: str) -> str:
    # PEP 503 name normalization: collapse runs of -_. to a single - and lowercase.
    return re.sub(r"[-_.]+", "-", name).lower()


class RootLockConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.pyproject = REPO_ROOT / "pyproject.toml"
        self.lock = REPO_ROOT / "uv.lock"
        # A missing committed lock is a FAILURE, not a skip: `uv run --frozen` in CI depends on
        # it. Skipping is allowed only via an explicit opt-out (a non-repro test mode).
        if os.environ.get("CLIF_SKIP_REPRO_LOCK"):
            self.skipTest("CLIF_SKIP_REPRO_LOCK set (explicit non-reproducibility mode)")
        self.assertTrue(self.pyproject.exists(), "pyproject.toml missing")
        self.assertTrue(self.lock.exists(),
                        "root uv.lock is missing — run `uv lock` and commit it (CI installs "
                        "from it with --frozen)")
        self.project = tomllib.loads(self.pyproject.read_text())["project"]

    def _declared_dep_names(self):
        names = []
        for spec in self.project.get("dependencies", []):
            names.append(_norm(re.split(r"[<>=!~;\[ ]", spec, maxsplit=1)[0]))
        return names

    def _locked(self):
        return {_norm(p["name"]): p["version"]
                for p in tomllib.loads(self.lock.read_text()).get("package", [])}

    def test_lock_pins_every_declared_dependency(self):
        locked = set(self._locked())
        missing = [d for d in self._declared_dep_names() if d not in locked]
        self.assertEqual(missing, [], f"declared deps absent from root uv.lock: {missing} "
                                      "(run `uv lock` after changing dependencies)")

    def test_cryptography_floor_is_honoured_in_the_lock(self):
        locked = self._locked()
        self.assertIn("cryptography", locked)
        ver = tuple(int(x) for x in locked["cryptography"].split(".")[:3])
        self.assertGreaterEqual(ver, (44, 0, 1),
                                f"locked cryptography {locked['cryptography']} is below the "
                                "44.0.1 advisory floor set in the U11 review")

    def test_pytest_is_available_from_the_dev_group_lock(self):
        """CI runs `uv run --frozen --group dev pytest`; the runner must be locked."""
        self.assertIn("pytest", self._locked(),
                      "pytest is not in uv.lock — add it to the [dependency-groups] dev group "
                      "and re-run `uv lock`")


if __name__ == "__main__":
    unittest.main()
