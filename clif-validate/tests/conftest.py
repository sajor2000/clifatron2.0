"""Make the package importable in-repo and expose the repo root for the drift guard.

The suite runs from the repository (`pytest clif-validate/tests`); at an installed
site the package is on sys.path already and the vendor-drift guard skips itself
when no repository checkout is present.
"""

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent

for p in (str(PKG_ROOT / "src"), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
