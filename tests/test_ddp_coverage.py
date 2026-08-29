"""U14: DDP sample coverage — two ranks shard the data without overlap.

Two independent proofs of the "no cross-rank double-counting" claim, both data-free
on CPU:
  1. A deterministic single-process check of DistributedSampler's partition property
     (always runs) — the invariant the engine relies on.
  2. A genuine two-process gloo smoke test (spawns real processes, inits the process
     group) that each rank's DistributedSampler union is the whole dataset with no
     overlap. Skips cleanly if the sandbox forbids process spawning — never passes
     silently.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler

N = 10  # divisible by world_size=2, so no sampler padding — a clean partition


class DistributedSamplerPartitionTest(unittest.TestCase):
    def test_two_ranks_partition_the_dataset_with_no_overlap(self):
        data = list(range(N))
        r0 = list(DistributedSampler(data, num_replicas=2, rank=0, shuffle=True, seed=0))
        s0 = DistributedSampler(data, num_replicas=2, rank=1, shuffle=True, seed=0)
        r1 = list(s0)
        self.assertEqual(len(r0), N // 2)
        self.assertEqual(len(r1), N // 2)
        self.assertEqual(set(r0) & set(r1), set(), "the two ranks share a sample")
        self.assertEqual(set(r0) | set(r1), set(range(N)), "coverage is not the full set")

    def test_non_divisible_length_pads_and_duplicates_one_sample(self):
        """The 'no overlap' guarantee holds only for a divisible length (or drop_last).

        `src/train/pretrain.py` uses DistributedSampler with the default
        drop_last=False, so for an ODD dataset length over two ranks the sampler PADS
        the index list to make it even — meaning one sample is seen twice across ranks.
        This documents that reality so the divisible-N tests above are not mistaken for
        a universal guarantee (CodeRabbit).
        """
        odd = 7
        data = list(range(odd))
        r0 = list(DistributedSampler(data, num_replicas=2, rank=0, shuffle=False))
        r1 = list(DistributedSampler(data, num_replicas=2, rank=1, shuffle=False))
        self.assertEqual(set(r0) | set(r1), set(range(odd)), "coverage is still complete")
        # 7 -> ceil(7/2)*2 = 8 index slots, so exactly one sample is duplicated.
        self.assertEqual(len(r0) + len(r1), odd + 1)
        self.assertEqual(len(set(r0) & set(r1)), 1,
                         "padding should duplicate exactly one sample across ranks")

    def test_set_epoch_reshuffles_deterministically(self):
        data = list(range(N))
        s = DistributedSampler(data, num_replicas=2, rank=0, shuffle=True, seed=0)
        s.set_epoch(0)
        e0 = list(s)
        s.set_epoch(1)
        e1 = list(s)
        s.set_epoch(0)
        e0_again = list(s)
        self.assertNotEqual(e0, e1, "set_epoch did not change the shuffle")
        self.assertEqual(e0, e0_again, "the same epoch did not reproduce the same order")


def _rank_worker(rank: int, world_size: int, out_dir: str, port: str):
    """Runs in a spawned process: init gloo, shard, dump this rank's indices."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        sampler = DistributedSampler(list(range(N)), num_replicas=world_size,
                                     rank=rank, shuffle=True, seed=0)
        sampler.set_epoch(0)
        indices = list(sampler)
        Path(out_dir, f"rank_{rank}.json").write_text(json.dumps(indices))
        dist.barrier()
    finally:
        dist.destroy_process_group()


class TwoProcessGlooSmokeTest(unittest.TestCase):
    def test_two_gloo_processes_cover_the_dataset_without_overlap(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as td:
            # A high, fixed-ish port derived from the pid to avoid collisions.
            port = str(29500 + (os.getpid() % 2000))
            try:
                mp.spawn(_rank_worker, args=(2, td, port), nprocs=2, join=True)
            except (RuntimeError, OSError, PermissionError) as exc:
                self.skipTest(f"process spawning unavailable in this sandbox: {exc}")

            r0 = json.loads(Path(td, "rank_0.json").read_text())
            r1 = json.loads(Path(td, "rank_1.json").read_text())
            self.assertEqual(set(r0) & set(r1), set(),
                             "the two gloo ranks trained on a shared sample")
            self.assertEqual(set(r0) | set(r1), set(range(N)),
                             "the two gloo ranks did not cover the dataset")


if __name__ == "__main__":
    unittest.main()
