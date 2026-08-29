"""clif-validate: federated validation of a frozen CLIFATRON bundle at a CLIF site.

The logic lives in `clif_validate._vendor.*` — byte-identical copies of the
repository's `src/` modules, produced by `scripts/sync_vendor.py`. The modules at
this level are thin, stable entry points; import from here, not from `_vendor`.
"""

__version__ = "0.1.0"
