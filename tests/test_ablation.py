import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.eval.ablation_compare import (
    build_headroom_table,
    build_outcome_table,
    build_transfer_table,
)


class AblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arms = {
            "frozen_backbone_head_only": {
                "description": "frozen backbone",
                "tags": ["finetune", "frozen-encoder"],
            },
            "joint_finetune": {
                "description": "joint",
                "tags": ["finetune", "joint-training"],
            },
            "from_scratch": {
                "description": "from scratch",
                "tags": ["from-scratch"],
            },
            "no_pretrain_baseline": {
                "description": "baseline",
                "tags": ["baseline"],
            },
        }

    def test_outcome_table_includes_all_arms(self):
        results = {
            "no_pretrain_baseline": {
                "tasks": {
                    "mortality": {"auroc": 0.70, "auprc": 0.15, "ece": 0.08},
                }
            },
            "frozen_backbone_head_only": {
                "tasks": {
                    "mortality": {"auroc": 0.82, "auprc": 0.28, "ece": 0.03},
                }
            },
            "joint_finetune": {
                "tasks": {
                    "mortality": {"auroc": 0.79, "auprc": 0.24, "ece": 0.05},
                }
            },
        }
        table = build_outcome_table(self.arms, results)
        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]["outcome"], "mortality")
        self.assertEqual(table[0]["frozen_backbone_head_only"]["auroc"], 0.82)
        self.assertEqual(table[0]["joint_finetune"]["auroc"], 0.79)
        self.assertNotIn("from_scratch", table[0])

    def test_headroom_is_positive_for_trained_arms(self):
        results = {
            "no_pretrain_baseline": {
                "tasks": {"mortality": {"auroc": 0.65, "auprc": 0.10}},
            },
            "frozen_backbone_head_only": {
                "tasks": {"mortality": {"auroc": 0.80, "auprc": 0.25}},
            },
        }
        headroom = build_headroom_table(results)
        self.assertEqual(headroom[0]["gain"], 0.15)

    def test_transfer_gap_positive_when_domain_better(self):
        results = {
            "frozen_backbone_head_only": {
                "tasks": {
                    "mortality": {"auroc": 0.82},
                    "delirium": {"auroc": 0.74},
                }
            },
        }
        transfer = build_transfer_table(
            results,
            in_domain_outcomes=["mortality"],
            zero_shot_outcomes=["delirium"],
        )
        self.assertGreater(transfer[0]["transfer_gap"], 0)

    def test_ablation_config_has_all_required_arms(self):
        config_path = Path("configs/ablation.yaml")
        abl = yaml.safe_load(config_path.read_text())
        required = ["frozen_backbone_head_only", "joint_finetune",
                     "from_scratch", "no_pretrain_baseline"]
        for arm in required:
            self.assertIn(arm, abl["arms"])
            self.assertIn("description", abl["arms"][arm])
            self.assertIn("lr", abl["arms"][arm])
            self.assertIn("total_steps", abl["arms"][arm])
            self.assertIn("tags", abl["arms"][arm])

    def test_ablation_config_has_no_treatment_initiation_outcome(self):
        """Hard rule #1: treatments are model inputs, never trunk prediction targets."""
        try:
            from test_data_config import FORBIDDEN_TRUNK_TASKS
        except ImportError:  # pragma: no cover - run from the repo root as a package
            from tests.test_data_config import FORBIDDEN_TRUNK_TASKS

        shared = yaml.safe_load(Path("configs/ablation.yaml").read_text())["shared"]
        named = set(shared["outcomes"]) | set(shared.get("zero_shot_outcomes") or ())
        self.assertFalse(named & set(FORBIDDEN_TRUNK_TASKS), sorted(named))
        self.assertIn("new_imv_24h", FORBIDDEN_TRUNK_TASKS)   # the helper still lists them

    def test_every_arm_names_a_curriculum_the_engine_runs(self):
        from src.train.curriculum import curriculum_enabled

        abl = yaml.safe_load(Path("configs/ablation.yaml").read_text())
        for name, arm in abl["arms"].items():
            with self.subTest(arm=name):
                enabled = curriculum_enabled({"curriculum": arm.get("curriculum", "none")})
                # A frozen CLIFATRON probe has no next-token objective to warm up.
                if arm.get("freeze_trunk") and arm["trunk"] == "clifatron_checkpoint":
                    self.assertFalse(enabled)

    def test_run_arm_refuses_a_cpu_distributed_launch_without_the_flag(self):
        import os
        import subprocess
        import sys

        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", RANK="0", LOCAL_RANK="0",
                   WORLD_SIZE="2", MASTER_ADDR="127.0.0.1", MASTER_PORT="29599")
        done = subprocess.run(
            [sys.executable, "-m", "src.train.run_arm", "--arm", "from_scratch",
             "--data", "/nonexistent"], env=env, capture_output=True, text=True, timeout=180)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("CUDA is unavailable", done.stderr)
        self.assertIn("--allow-cpu-ddp", done.stderr)

    def test_dry_run_arm_loads(self):
        import subprocess

        # n_value_bins is derived from the data's tokenizer-v2 vocab.json (U5, KTD7).
        with tempfile.TemporaryDirectory() as data:
            (Path(data) / "vocab.json").write_text(json.dumps({
                "vocab": {"<pad>": 0, "map=0": 1, "map=1": 2},
                "segments": {"map": [
                    {"lo": None, "hi": 65.0, "lo_closed": False, "hi_closed": False},
                    {"lo": 65.0, "hi": None, "lo_closed": True, "hi_closed": False},
                ]},
                "manifest": {"tokenizer_version": 2},
            }))
            result = subprocess.run(
                ["uv", "run", "python", "-m", "src.train.run_arm",
                 "--arm", "from_scratch", "--data", data,
                 "--ablation-config", "configs/ablation.yaml",
                 "--dry-run"],
                capture_output=True, text=True,
            )
            missing = subprocess.run(
                ["uv", "run", "python", "-m", "src.train.run_arm",
                 "--arm", "from_scratch", "--data", str(Path(data) / "absent"),
                 "--ablation-config", "configs/ablation.yaml", "--dry-run"],
                capture_output=True, text=True,
            )
        self.assertNotEqual(result.returncode, 1)
        self.assertIn("CLIFEncoder from scratch", result.stdout)
        # Without a vocabulary there is no n_value_bins to derive: refuse, never default.
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("vocab.json", missing.stderr)


if __name__ == "__main__":
    unittest.main()