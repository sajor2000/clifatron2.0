"""U14: checkpoint atomicity, RNG round-trip, and fail-closed loading.

The engine claims exact-resume; that rests on the checkpoint (a) writing atomically
so an interrupted save never replaces a good checkpoint with a half-written one,
(b) round-tripping the RNG state so stochastic ops continue identically after resume,
and (c) failing closed on a missing or corrupt file rather than silently starting
fresh. This proves each claim data-free on CPU.
"""

import os
import tempfile
import unittest
from pathlib import Path

import torch

from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.train.engine import _restore_rng_states
from src.train.manifest import Manifest


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)


def _save(path, **over):
    model = over.pop("model", None) or _Tiny()
    opt = over.pop("opt", None) or torch.optim.SGD(_Tiny().parameters(), lr=0.01)
    sched = over.pop("sched", None) or torch.optim.lr_scheduler.StepLR(opt, 1000)
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched,
                    epoch=over.pop("epoch", 1), step=over.pop("step", 5),
                    manifest=Manifest("t", {}, seed=42, ckpt_dir="/tmp"), **over)


class CheckpointAtomicityTest(unittest.TestCase):
    def test_save_leaves_no_temp_residue_and_a_complete_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            _save(path)
            self.assertTrue(path.exists())
            # No leftover ckpt_*.pt temp files beside the final checkpoint.
            residue = [p for p in Path(td).iterdir() if p.name != "ckpt.pt"]
            self.assertEqual(residue, [], f"temp residue left behind: {residue}")
            loaded = load_checkpoint(path)
            for key in ("model", "optimizer", "scheduler", "epoch", "step", "manifest"):
                self.assertIn(key, loaded)

    def test_a_failed_save_does_not_replace_an_existing_good_checkpoint(self):
        """tmp-then-rename: a save that dies before the move leaves the old file intact."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            _save(path, epoch=1)
            good = load_checkpoint(path)
            self.assertEqual(good["epoch"], 1)

            # A model whose state_dict() raises mid-save: torch.save never completes,
            # shutil.move never runs, and the pre-existing checkpoint is untouched.
            class _Explodes(torch.nn.Module):
                def state_dict(self, *a, **k):
                    raise RuntimeError("disk full")

            with self.assertRaises(RuntimeError):
                _save(path, model=_Explodes(),
                      opt=torch.optim.SGD(_Tiny().parameters(), lr=0.01))
            still = load_checkpoint(path)
            self.assertEqual(still["epoch"], 1, "a failed save clobbered the good file")
            residue = [p for p in Path(td).iterdir() if p.name != "ckpt.pt"]
            self.assertEqual(residue, [], f"failed save left temp residue: {residue}")


class RngRoundTripTest(unittest.TestCase):
    def test_rng_state_survives_save_and_restore(self):
        torch.manual_seed(123)
        _ = torch.rand(10)  # advance the generator to a non-initial state
        captured = torch.get_rng_state()
        expected_next = torch.rand(5)  # the values that must follow the captured state

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            _save(path, rng_states=[{"cpu": captured.tolist(), "cuda": {}}])
            # Perturb the global RNG so a no-op restore would be detectable.
            torch.manual_seed(999)
            torch.rand(7)
            loaded = load_checkpoint(path)
            _restore_rng_states(loaded["rng_states"])

        self.assertTrue(torch.equal(torch.get_rng_state(), captured))
        self.assertTrue(torch.equal(torch.rand(5), expected_next),
                        "post-resume RNG stream diverged from the straight run")


class FailClosedLoadTest(unittest.TestCase):
    def test_absent_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_checkpoint("/nonexistent/dir/ckpt.pt")

    def test_truncated_checkpoint_raises_not_silent_fresh_start(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ckpt.pt"
            _save(path)
            raw = bytearray(path.read_bytes())
            path.write_bytes(bytes(raw[: len(raw) // 2]))  # truncate to half
            with self.assertRaises(Exception):
                load_checkpoint(path)

    def test_a_pre_v1_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "old.pt"
            torch.save({"schema_version": 0, "model": {}}, path)
            with self.assertRaisesRegex(ValueError, "schema version"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
