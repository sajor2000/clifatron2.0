"""Vendor-drift guard (U9): the wheel and the repo must run the same bytes.

`scripts/sync_vendor.py` copies the repo's `src/` closure into
`clif_validate/_vendor/` with a mechanical import rewrite and records source
SHA-256s in `vendor_manifest.json`. This suite asserts the guard in both
directions on the REAL trees, and then — prove-the-guard-guards — that the
checker actually goes red for every drift class, exercised on throwaway copies:
a byte flipped in a vendored file, a source edited after sync, and a module
added that nobody vendored.
"""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
VENDOR_ROOT = PKG_ROOT / "src" / "clif_validate" / "_vendor"

_spec = importlib.util.spec_from_file_location(
    "sync_vendor", PKG_ROOT / "scripts" / "sync_vendor.py"
)
sync_vendor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_vendor)


@unittest.skipUnless((REPO_ROOT / "src" / "eval").is_dir(),
                     "no repository checkout beside the package (installed site)")
class VendorDriftGuardTest(unittest.TestCase):
    def test_vendored_tree_matches_the_repo_sources(self):
        """The shipped guard: any drift between src/ and _vendor/ fails here."""
        problems = sync_vendor.check(repo_root=REPO_ROOT, vendor_root=VENDOR_ROOT)
        self.assertEqual(problems, [])

    def test_manifest_covers_the_declared_closure(self):
        import json

        manifest = json.loads((VENDOR_ROOT / "vendor_manifest.json").read_text())
        self.assertEqual(sorted(manifest), sorted(sync_vendor.VENDOR_FILES))

    def _mirrored_trees(self):
        """Throwaway copies of the 14 sources + the vendor tree, check-clean."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo_copy = Path(tmp.name) / "repo"
        vendor_copy = Path(tmp.name) / "vendor"
        for rel in sync_vendor.VENDOR_FILES:
            dest = repo_copy / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / rel, dest)
        shutil.copytree(VENDOR_ROOT, vendor_copy)
        self.assertEqual(
            sync_vendor.check(repo_root=repo_copy, vendor_root=vendor_copy), []
        )
        return repo_copy, vendor_copy

    def test_the_guard_goes_red_for_a_flipped_vendored_byte(self):
        repo_copy, vendor_copy = self._mirrored_trees()
        target = vendor_copy / "eval" / "schema.py"
        raw = bytearray(target.read_bytes())
        raw[len(raw) // 2] ^= 0x01
        target.write_bytes(bytes(raw))
        problems = sync_vendor.check(repo_root=repo_copy, vendor_root=vendor_copy)
        self.assertTrue(any("not the mechanical rewrite" in p for p in problems),
                        problems)

    def test_the_guard_goes_red_for_a_source_edited_after_sync(self):
        repo_copy, vendor_copy = self._mirrored_trees()
        target = repo_copy / "src" / "eval" / "schema.py"
        target.write_text(target.read_text() + "\n# post-sync edit\n")
        problems = sync_vendor.check(repo_root=repo_copy, vendor_root=vendor_copy)
        self.assertTrue(any("stale" in p for p in problems), problems)

    def test_the_guard_goes_red_for_an_unvendored_module(self):
        repo_copy, vendor_copy = self._mirrored_trees()
        (vendor_copy / "eval" / "sneaky.py").write_text("PASS = True\n")
        problems = sync_vendor.check(repo_root=repo_copy, vendor_root=vendor_copy)
        self.assertTrue(any("unexpected vendored module" in p for p in problems),
                        problems)


class VendoredImportSurfaceTest(unittest.TestCase):
    """The wheel's public modules import and expose the ceremony surface."""

    def test_public_shims_import(self):
        import clif_validate
        from clif_validate import bundle, cli, inference, report

        self.assertTrue(callable(cli.main))
        self.assertTrue(callable(bundle.load_bundle))
        self.assertTrue(callable(inference.bundle_predict_fn))
        self.assertTrue(callable(report.write_export))
        self.assertEqual(clif_validate.__version__, "0.1.0")

    def test_no_vendored_module_still_imports_from_src(self):
        """Runtime companion to the textual rewrite check: import the vendored
        entry module and confirm its module graph never reaches `src.*`."""
        import sys

        import clif_validate._vendor.eval.clif_validate  # noqa: F401

        vendored = [name for name in sys.modules
                    if name.startswith("clif_validate._vendor.")]
        self.assertTrue(vendored)
        for name in vendored:
            mod = sys.modules[name]
            for attr in vars(mod).values():
                origin = getattr(attr, "__module__", "") or ""
                self.assertFalse(origin == "src" or origin.startswith("src."),
                                 f"{name} leaked a src.* object: {attr!r}")


if __name__ == "__main__":
    unittest.main()
