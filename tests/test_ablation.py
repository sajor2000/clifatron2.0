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

    def test_dry_run_arm_loads(self):
        import subprocess
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.train.run_arm",
             "--arm", "from_scratch", "--data", "/tmp",
             "--ablation-config", "configs/ablation.yaml",
             "--dry-run"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 1)
        self.assertIn("CLIFEncoder from scratch", result.stdout)


if __name__ == "__main__":
    unittest.main()