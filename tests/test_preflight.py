"""U19: the L40 pre-flight (`src/train/preflight.py`).

What is proven, on a synthetic site tokenized through the real pipeline:
  - the split freeze (KTD6) is required: a missing freeze, a freeze of another split, and
    a split whose configured proportions changed since the freeze are refused; a freeze is
    never silently replaced; held-out arm counts are read from a blind-stage export;
  - a shard bound to another vocabulary, and a vocabulary fit on a sample, are refused;
    value stats must be bound to the vocabulary and cover the sample's numeric tokens;
  - without CUDA the GPU checks fail (they do not crash) and the command exits non-zero;
    `--skip-gpu` runs the two-rank gloo rehearsal and the whole pre-flight passes;
  - the memory projection arithmetic and its warn/fail thresholds;
  - the edge check names the refused control and the arm.
"""

import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

import polars as pl
import yaml

from src.train import preflight as pf

ROOT = Path(__file__).resolve().parents[1]


def _rewrite_split(path: Path, out: Path) -> None:
    """A valid episode artifact with ONE eligible episode moved to another partition (a
    different split, content hashes recomputed)."""
    from src.data.splits import content_manifest

    episodes = pl.read_parquet(path)
    first = episodes.filter(pl.col("eligible"))["hospitalization_id"][0]
    episodes = episodes.with_columns(
        pl.when(pl.col("hospitalization_id") == first).then(pl.lit("validation"))
        .otherwise(pl.col("partition")).alias("partition"))
    eligible = episodes.filter(pl.col("eligible"))
    split = content_manifest(eligible, columns=["hospitalization_id", "patient_id",
                                                "partition"])["sha256"]
    episode = content_manifest(episodes, columns=["hospitalization_id", "patient_id",
                                                  "eligible", "partition"])["sha256"]
    episodes.with_columns(pl.lit(split).alias("split_sha256"),
                          pl.lit(episode).alias("episode_sha256")).write_parquet(out)


class PreflightSyntheticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls.site = pf.build_synthetic_site(cls._td.name)
        cls.work = cls.site["work"]
        cls.episodes = {pf.SYNTHETIC_SITE_NAME: cls.site["episodes"]}

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def arm(self, directory: Path | None = None, stats: Path | None = None) -> pf.ArmData:
        directory = directory or self.site["shard_dir"]
        return pf.ArmData("synthetic", {pf.SYNTHETIC_SITE_NAME: directory},
                          stats or directory / "gem_value_stats.json", None)

    def copy_shard_dir(self, name: str) -> Path:
        out = self.work / "output/intermediate_phi" / name
        shutil.copytree(self.site["shard_dir"], out, dirs_exist_ok=True)
        return out

    # ---------------------------------------------------------------- split freeze

    def freeze(self, name: str, episodes=None, **kw) -> Path:
        record = pf.build_split_freeze(episodes or self.episodes, approver="tester", **kw)
        return pf.write_split_freeze(self.work / f"output/final_no_phi/{name}.json", record)

    def test_split_freeze_missing_is_refused(self):
        for path in (None, self.work / "does_not_exist.json"):
            check = pf.check_split_freeze(path, self.episodes)
            self.assertEqual(check.status, pf.FAIL)
            self.assertIn("--write-split-freeze", check.detail)

    def test_split_freeze_matches_the_artifact(self):
        check = pf.check_split_freeze(self.freeze("match"), self.episodes)
        self.assertEqual(check.status, pf.PASS, check.detail)

    def test_split_freeze_of_another_split_is_refused(self):
        path = self.freeze("other_split")
        changed = self.work / "output/intermediate_phi/episodes_changed.parquet"
        _rewrite_split(self.site["episodes"], changed)
        check = pf.check_split_freeze(path, {pf.SYNTHETIC_SITE_NAME: changed})
        self.assertEqual(check.status, pf.FAIL)
        self.assertIn("split hash", check.detail)

    def test_changed_proportions_are_refused(self):
        path = self.freeze("proportions")
        config = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
        config["data_contract"]["partitions"] = {"train": 0.5, "validation": 0.25,
                                                 "calibration": 0.1, "internal_test": 0.15}
        changed = self.work / "train_changed.yaml"
        changed.write_text(yaml.safe_dump(config))
        check = pf.check_split_freeze(path, self.episodes, train_config=changed)
        self.assertEqual(check.status, pf.FAIL)
        self.assertIn("partitions or split_seed changed", check.detail)

    def test_site_missing_from_freeze_is_refused(self):
        path = self.freeze("one_site")
        check = pf.check_split_freeze(path, {"rush": self.site["episodes"]})
        self.assertEqual(check.status, pf.FAIL)
        self.assertIn("site rush is not in the freeze", check.detail)

    def test_freeze_is_not_silently_replaced_and_needs_an_approver(self):
        path = self.freeze("replace")
        changed = self.work / "output/intermediate_phi/episodes_replace.parquet"
        _rewrite_split(self.site["episodes"], changed)
        other = pf.build_split_freeze({pf.SYNTHETIC_SITE_NAME: changed}, approver="tester")
        with self.assertRaises(pf.PreflightError):
            pf.write_split_freeze(path, other)
        pf.write_split_freeze(path, other, force=True)
        with self.assertRaises(pf.PreflightError):
            pf.build_split_freeze(self.episodes, approver="  ")

    def test_held_out_arm_counts_from_the_blind_export(self):
        blind = self.work / "blind.json"
        blind.write_text(json.dumps({"stage": "blind", "release_id": "r1", "cells": {
            "mimic|held_out|eligible": {"n": 120, "status": "evaluable"},
            "mimic|held_out|eligible|arm=hfnc": {"n": 80, "status": "evaluable"},
            "mimic|held_out|eligible|arm=niv": {"n": None, "status": "suppressed"},
            "mimic|all|eligible|arm=hfnc": {"n": 400, "status": "evaluable"},
        }}))
        record = pf.build_split_freeze(self.episodes, approver="tester", audit_blind=blind)
        self.assertEqual(record["audit_blind"]["held_out_arms"],
                         {"mimic": {"hfnc": 80, "niv": "suppressed"}})
        self.assertEqual(record["audit_blind"]["release_id"], "r1")
        summary = record["sites"][pf.SYNTHETIC_SITE_NAME]
        self.assertEqual(set(summary), {"file", "file_sha256", "split_sha256",
                                        "episode_sha256", "eligible_episodes",
                                        "observed_shares", "configured_proportions",
                                        "split_seed"})

    def test_shard_partitions_must_equal_the_episode_artifact(self):
        shard = self.site["shard_dir"] / pf.GEM_EVENTS
        ok = pf.check_shard_partitions("s", shard, self.site["episodes"])
        self.assertEqual(ok.status, pf.PASS, ok.detail)
        changed = self.work / "output/intermediate_phi/episodes_partitions.parquet"
        _rewrite_split(self.site["episodes"], changed)
        bad = pf.check_shard_partitions("s", shard, changed)
        self.assertEqual(bad.status, pf.FAIL)
        self.assertIn("1 differ", bad.detail)

    # ---------------------------------------------------------------- binding

    def test_bound_arm_passes(self):
        checks = pf.check_arm_binding(self.arm())
        self.assertEqual([c.status for c in checks], [pf.PASS, pf.PASS],
                         [c.detail for c in checks])

    def test_shard_bound_to_another_vocabulary_is_refused(self):
        directory = self.copy_shard_dir("other_vocab")
        blob = json.loads((directory / "vocab.json").read_text())
        blob["vocab"]["map=extra"] = max(blob["vocab"].values()) + 1
        (directory / "vocab.json").write_text(json.dumps(blob))
        binding = pf.check_arm_binding(self.arm(directory))[-1]
        self.assertEqual(binding.status, pf.FAIL)
        self.assertIn("vocabulary mismatch", binding.detail)
        stats = pf.check_value_stats(self.arm(directory))
        self.assertEqual(stats.status, pf.FAIL)
        self.assertIn("vocabulary hash mismatch", stats.detail)

    def test_second_site_bound_to_another_vocabulary_is_refused(self):
        directory = self.copy_shard_dir("site_b")
        blob = json.loads((directory / "vocab.json").read_text())
        blob["vocab"]["map=extra"] = max(blob["vocab"].values()) + 1
        (directory / "vocab.json").write_text(json.dumps(blob))
        arm = pf.ArmData("two_sites", {"a": self.site["shard_dir"], "b": directory},
                         self.site["value_stats"], None)
        binding = pf.check_arm_binding(arm)[-1]
        self.assertEqual(binding.status, pf.FAIL)
        self.assertIn("site b vocab.json", binding.detail)

    def test_sample_vocabulary_is_refused(self):
        directory = self.copy_shard_dir("sample_vocab")
        blob = json.loads((directory / "vocab.json").read_text())
        blob["manifest"].setdefault("provenance", {})["sample"] = True
        (directory / "vocab.json").write_text(json.dumps(blob))
        vocabulary = pf.check_arm_binding(self.arm(directory))[0]
        self.assertEqual(vocabulary.status, pf.FAIL)
        self.assertIn("verification sample", vocabulary.detail)

    def test_value_stats_missing_or_incomplete_are_refused(self):
        missing = pf.check_value_stats(self.arm(stats=self.work / "nope.json"))
        self.assertEqual(missing.status, pf.FAIL)
        stats = json.loads(self.site["value_stats"].read_text())
        dropped = next(iter(stats["stats"]))
        del stats["stats"][dropped]
        partial = self.work / "partial_stats.json"
        partial.write_text(json.dumps(stats))
        check = pf.check_value_stats(self.arm(stats=partial),
                                     self.site["shard_dir"] / pf.GEM_EVENTS)
        self.assertEqual(check.status, pf.FAIL)
        self.assertIn("have no stats", check.detail)

    # ---------------------------------------------------------------- schedule

    def test_schedule_counts_updates_and_refuses_a_run_without_a_checkpoint(self):
        tcfg = {"batch": {"per_gpu": 4, "grad_accum": 1},
                "runtime": {"token_budget": 0, "ckpt_every": 2},
                "schedule": {"warmup_steps": 1, "total_steps": 99}}
        # 24 single-window stays: 6 global batches, 3 per rank per pass.
        batches, updates = pf.updates_per_budget([10] * 24, tcfg,
                                                 {"screening": 0.05, "full": 1.0})
        self.assertEqual((batches, updates), (3, {"screening": 1, "full": 3}))
        check = pf.schedule_check(self.arm(), tcfg, {"screening": 0.05, "full": 1.0})
        self.assertEqual(check.status, pf.FAIL)
        self.assertIn("screening run(s) would write NO checkpoint", check.detail)
        ok = pf.schedule_check(self.arm(), tcfg, {"screening": 1.0, "full": 2.0})
        self.assertEqual(ok.status, pf.PASS, ok.detail)
        tcfg["schedule"]["warmup_steps"] = 10
        self.assertEqual(pf.schedule_check(self.arm(), tcfg,
                                           {"screening": 1.0, "full": 2.0}).status, pf.WARN)

    # ---------------------------------------------------------------- thresholds

    def test_edge_check_names_the_refused_control(self):
        from src.data.segments import segments_from_edges
        from src.data.threshold_grid import load_thresholds

        blob = json.loads((self.site["shard_dir"] / "vocab.json").read_text())
        control = next(t for t in load_thresholds()["control"] if t.concept == "map")
        on_edge = copy.deepcopy(blob)
        on_edge["segments"]["map"] = segments_from_edges([40.0, control.value, 65.0, 90.0])
        checks = {c.name: c for c in pf.threshold_checks({"clean": blob,
                                                          "decile_like": on_edge})}
        edge = checks["thresholds: edge check"]
        self.assertEqual(edge.status, pf.FAIL)
        self.assertIn(f"control map {control.value:g}", edge.detail)
        self.assertIn("'decile_like'", edge.detail)
        self.assertNotIn("'clean'", edge.detail)
        self.assertEqual(checks["thresholds: registry"].status, pf.PASS)
        self.assertEqual(checks["thresholds: competing-risk causes"].status, pf.WARN)
        self.assertEqual(pf.threshold_checks({"clean": blob})[1].status, pf.PASS)

    # ---------------------------------------------------------------- GPU / end to end

    def test_without_cuda_gpu_checks_fail_and_the_command_exits_nonzero(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            checks = pf.gpu_checks(skip_gpu=False)
            self.assertTrue(checks)
            self.assertTrue(all(c.status == pf.FAIL for c in checks),
                            [(c.name, c.status) for c in checks])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = pf.main(["--synthetic", "--allow-dirty"])
        self.assertEqual(code, 1)
        table = out.getvalue()
        self.assertRegex(table, r"gpu: CUDA available\s+FAIL")
        self.assertRegex(table, r"gpu: two-rank DDP smoke\s+FAIL")
        self.assertIn("PRE-FLIGHT FAILED", table)

    def test_skip_gpu_runs_the_gloo_rehearsal_and_passes_on_the_synthetic_site(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = pf.main(["--synthetic", "--skip-gpu", "--allow-dirty"])
        table = out.getvalue()
        self.assertEqual(code, 0, table)
        self.assertRegex(table, r"gpu: two-rank DDP smoke \(gloo, CPU\)\s+PASS")
        self.assertRegex(table, r"split: frozen hash \(KTD6\)\s+PASS")
        self.assertRegex(table, r"memory: projection.*\s+PASS")
        self.assertIn("PRE-FLIGHT PASSED", table)
        self.assertNotIn("synth-0", table)          # no stay identifier is printed


class MemoryProjectionTest(unittest.TestCase):
    SAMPLE: ClassVar[dict] = {"events": 1000, "mapped_bytes": 60_000, "private_bytes": 10_000,
              "build_peak_bytes": 400_000}
    TOTALS: ClassVar[list] = [
        {"events": 1_000_000, "multi_window_events": 200_000,
         "train": {"events": 800_000}, "validation": {"events": 200_000}},
        {"events": 500_000, "multi_window_events": 100_000,
         "train": {"events": 400_000}, "validation": {"events": 100_000}},
    ]

    def test_projection_arithmetic(self):
        p = pf.project_memory(self.SAMPLE, self.TOTALS, ranks=2, target_bytes_per_event=26)
        self.assertEqual(p["events"], 1_500_000)
        self.assertEqual(p["mapped_bytes_per_event"], 60.0)
        self.assertEqual(p["mapped_bytes"], 60.0 * 1_500_000)
        self.assertEqual(p["per_rank_bytes"], 10.0 * 1_500_000 + 26 * 300_000)
        self.assertEqual(p["build_bytes"], 400.0 * 800_000)       # largest partition
        self.assertEqual(p["node_steady_bytes"], p["mapped_bytes"] + 2 * p["per_rank_bytes"])
        self.assertEqual(p["node_peak_bytes"], p["node_steady_bytes"] + p["build_bytes"])
        built = pf.project_memory(self.SAMPLE, self.TOTALS, ranks=2,
                                  target_bytes_per_event=26, caches_built=True)
        self.assertEqual(built["build_bytes"], 0.0)

    def test_target_cache_bytes_follow_the_dataset(self):
        self.assertEqual(pf.target_cache_bytes_per_event(), 8 + 1 + 8 + 8 + 1)

    def test_thresholds_warn_above_70_and_fail_above_90_percent(self):
        self.assertEqual(pf.memory_status(69, 100), (pf.PASS, 0.69))
        self.assertEqual(pf.memory_status(71, 100)[0], pf.WARN)
        self.assertEqual(pf.memory_status(90, 100)[0], pf.WARN)
        self.assertEqual(pf.memory_status(91, 100)[0], pf.FAIL)


class FilesystemTest(unittest.TestCase):
    def test_temporary_directory_is_local(self):
        with tempfile.TemporaryDirectory() as td:
            fstype, local = pf.filesystem_type(td)
            self.assertTrue(local, fstype)
            self.assertEqual(pf.check_local_fs(Path(td) / "gem_cache", "x").status, pf.PASS)

    def test_mount_escapes_and_network_types(self):
        self.assertEqual(pf._unescape_mount(r"/mnt/my\040data"), "/mnt/my data")
        self.assertIn("nfs4", pf.NETWORK_FS)
        self.assertIn("cifs", pf.NETWORK_FS)


if __name__ == "__main__":
    unittest.main()
