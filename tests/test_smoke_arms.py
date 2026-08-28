"""End-to-end smoke test on real CLIF 2.1 data using MPS.

Tokenizes 100 real ICU stays from ~/Data/clif-source, builds a frozen decile
vocab, constructs a single batch with all required keys, and forward-passes
every ablation arm. Runs on Apple Silicon MPS. Catches shape mismatches,
import errors, and basic numerical issues before the L40 box runs.
"""

import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

# CLIF source parquet lives per-machine (git-ignored); override with CLIF_DATA_DIR.
CLIF_DATA = Path(
    os.environ.get("CLIF_DATA_DIR", "~/Data/clif-source")
).expanduser().resolve()
DATA_CFG = Path("configs/data.yaml")
MODEL_CFG = Path("configs/model.yaml")
ABL_CFG = Path("configs/ablation.yaml")

N_STAYS = 100
MAX_EVENTS_PER_STAY = 1024
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _clif_data_available(base: Path) -> bool:
    """True only when the CLIF source dir has at least one parquet to tokenize."""
    return base.is_dir() and any(base.glob("*.parquet"))


CLIF_AVAILABLE = _clif_data_available(CLIF_DATA)
_SKIP_REASON = (
    f"CLIF source parquet not found at {CLIF_DATA} "
    "(set CLIF_DATA_DIR to a directory of CLIF 2.1 *.parquet to enable)"
)


@unittest.skipUnless(CLIF_AVAILABLE, _SKIP_REASON)
class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out_dir = Path(cls.tmp.name)
        cls.cfg = yaml.safe_load(DATA_CFG.read_text())
        cls.mcfg = yaml.safe_load(MODEL_CFG.read_text())

        cls._build_vocab()

        cls.shards = _read_shards()
        print(f"Loaded {len(cls.shards):,} stays with vocab size {cls.vocab_size:,}")

        cls.batch = _make_batch(cls.shards, cls.vocab_size)
        print(f"Batch shapes: token={list(cls.batch['token'].shape)}, "
              f"soft_token={list(cls.batch['soft_token'].shape)}, "
              f"pos_min={list(cls.batch['pos_min'].shape)}")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def _build_vocab(cls):
        from src.data.tokenize import tokenize_site

        tokenize_site(cls.cfg, "mimic", CLIF_DATA, cls.out_dir,
                      vocab=None, edges=None, limit_stays=N_STAYS * 3)

        blob = json.loads((cls.out_dir / "vocab.json").read_text())
        cls.vocab = blob["vocab"]
        cls.edges = blob["edges"]
        cls.vocab_size = len(cls.vocab)
        print(f"Built vocab: {cls.vocab_size:,} tokens, {len(cls.edges)} numeric concepts")

    # ----------------------------------------------------------------- arms
    def test_01_from_scratch_forward_pass(self):
        """Random-init CLIFEncoder + heads, one step."""
        _run_arm("from_scratch", self.mcfg, self.batch, self.vocab_size,
                 n_targets=10, freeze_trunk=False)

    def test_02_frozen_backbone_no_pretrain_baseline(self):
        """Frozen random encoder + TaskHead baseline."""
        _run_arm("no_pretrain_baseline", self.mcfg, self.batch, self.vocab_size,
                 n_targets=10, freeze_trunk=True)

    def test_03_joint_full_model_ntp_forward(self):
        """Unfrozen CLIFEncoder + heads, full loss."""
        _run_arm("joint_finetune", self.mcfg, self.batch, self.vocab_size,
                 n_targets=10, freeze_trunk=False)


class DataFreeSmokeTest(unittest.TestCase):
    """Numeric/config smoke checks that need no CLIF source data.

    Kept separate from SmokeTest so they still run on a box without staged
    CLIF parquet (the arm-forward tests below require a real vocab).
    """

    def test_04_curriculum_scheduler_phases(self):
        """All three curriculum phases produce valid Mix values."""
        from src.train.curriculum import curriculum_weights

        for step in (0, 15000, 40000):
            mix = curriculum_weights(step, 60000)
            self.assertGreaterEqual(mix.w_ntp, 0)
            self.assertTrue(0 <= mix.w_cr <= 1)

    def test_05b_nan_predictions_are_dropped_not_fabricated(self):
        """NaN predictions must NOT be silently coerced to a neutral 0.5 and scored.

        Regression guard (Greptile PR #1 P1): undefined model output is dropped and
        counted, saturated ±inf predictions are kept, and an all-NaN vector yields NaN
        metrics rather than fabricated ones."""
        from src.eval import metrics as M

        np.random.seed(1)
        y = (np.random.random(200) > 0.7).astype(int)
        p = np.clip(y.astype(float) + np.random.normal(0, 0.15, 200), 0, 1)

        # (a) ±inf saturated logits are kept (clamped), nothing dropped
        logits = M._logit(p)
        logits[p == 0] = -np.inf
        logits[p == 1] = np.inf
        pan = M.full_panel(p.copy(), y, logits=logits.copy(), recalibrate=True)
        self.assertEqual(pan["n_dropped_nan"], 0)

        # (b) NaN predictions are dropped and counted, not fabricated to 0.5
        p_nan = p.copy()
        p_nan[:10] = np.nan
        pan_nan = M.full_panel(p_nan, y.copy(), recalibrate=False)
        self.assertEqual(pan_nan["n_dropped_nan"], 10)
        self.assertEqual(pan_nan["n"], 190)

        # (c) nan_policy='raise' fails loud
        with self.assertRaises(ValueError):
            M.full_panel(p_nan, y.copy(), recalibrate=False, nan_policy="raise")

        # (d) all-NaN => NaN metrics, never fabricated
        pan_all = M.full_panel(np.full(50, np.nan),
                               (np.random.random(50) > 0.5).astype(int),
                               recalibrate=False)
        self.assertTrue(np.isnan(pan_all["auroc"]))
        self.assertEqual(pan_all["n"], 0)

    def test_05_metrics_panel_on_quick_synthetic(self):
        """Metrics panel returns all expected keys on synthetic data."""
        from src.eval import metrics as M

        np.random.seed(42)
        y = (np.random.random(200) > 0.8).astype(int)
        p = np.clip(y.astype(float) + np.random.normal(0, 0.15, 200), 0, 1)
        logits = M._logit(p)  # robust prob->logit (clips 0/1 to avoid ±inf)

        panel = M.full_panel(p, y, logits=logits, recalibrate=True)
        for key in ("auroc", "auprc", "ece", "brier", "calib_slope",
                     "calib_intercept", "ici", "temperature"):
            self.assertIn(key, panel, f"missing metric: {key}")

    def test_06_cr_d_calibration_runs(self):
        """D-calibration produces valid chi-squared p-value."""
        from src.eval import metrics as M

        np.random.seed(42)
        cif = np.random.random((60, 3, 16))
        cif /= cif.sum(axis=1, keepdims=True)
        events = np.random.randint(0, 3, 60)
        times = np.random.randint(0, 16, 60)

        dcal = M.cr_d_calibration(cif, events, times)
        self.assertIn("d_calib_p", dcal)
        self.assertGreater(dcal.get("n", 0), 0)

    def test_07_ablation_config_references_correct_evidence(self):
        """Config tags match evidence anchors."""
        abl = yaml.safe_load(ABL_CFG.read_text())

        arm_tags = {
            "frozen_backbone_head_only": ["finetune", "frozen-encoder"],
            "joint_finetune": ["finetune", "joint-training"],
            "from_scratch": ["from-scratch", "pretraining"],
            "no_pretrain_baseline": ["baseline", "no-pretraining"],
        }
        for arm_name, expected_tags in arm_tags.items():
            actual = abl["arms"][arm_name]["tags"]
            for tag in expected_tags:
                self.assertIn(tag, actual, f"{arm_name} missing tag {tag}")


# ------------------------------------------------------------------- helpers
def _read_shards():
    import polars as pl

    df = pl.read_parquet(SmokeTest.out_dir / "events.parquet")
    shards = df.sort("n_events").to_dicts()[:N_STAYS]
    return shards


def _make_batch(shards: list, vocab_size: int):
    max_len = max(s["n_events"] for s in shards)

    token = torch.zeros((len(shards), max_len), dtype=torch.long)
    soft_token = torch.zeros((len(shards), max_len, 3), dtype=torch.long)
    soft_weight = torch.zeros((len(shards), max_len, 3))
    pos_min = torch.zeros((len(shards), max_len), dtype=torch.long)
    val = torch.zeros((len(shards), max_len))
    val_mask = torch.zeros((len(shards), max_len))

    for i, s in enumerate(shards):
        n = s["n_events"]
        token[i, :n] = torch.tensor(s["token"][:n], dtype=torch.long)
        _soft = np.array(s["soft_token"][:n])
        _w = np.array(s["soft_weight"][:n])
        soft_token[i, :n] = torch.tensor(_soft, dtype=torch.long)
        soft_weight[i, :n] = torch.tensor(_w)
        pos_min[i, :n] = torch.tensor(s["pos_min"][:n], dtype=torch.long)
        _val = np.array(s["value"][:n])
        mask = np.isfinite(_val)
        val[i, :n] = torch.tensor(np.nan_to_num(_val))
        val_mask[i, :n] = torch.tensor(mask.astype(float))

    n_targets = 10
    batch = {
        "token": token,
        "soft_token": soft_token,
        "soft_weight": soft_weight,
        "pos_min": pos_min,
        "value": val,
        "val_mask": val_mask,
        "last_idx": torch.tensor([s["n_events"] - 1 for s in shards]),
        "cr_type": torch.randint(0, n_targets, (len(shards),)),
        "cr_bin": torch.randint(0, 16, (len(shards),)),
        "th_target": torch.randint(0, n_targets, (len(shards),)),
        "th_tau": torch.randint(1, 10, (len(shards),)),
        "th_dir": torch.randint(0, 2, (len(shards),)),
        "th_crossed": torch.where(
            torch.rand(len(shards)) > 0.7,
            torch.randint(1, 48, (len(shards),)),
            -torch.ones(len(shards), dtype=torch.long),
        ),
    }
    return batch


def _run_arm(name: str, mcfg: dict, batch: dict, vocab_size: int,
             n_targets: int, freeze_trunk: bool):
    from src.model.encoder import CLIFEncoder, count_params
    from src.model.heads import (
        CompetingRiskHead, ThresholdHazardHead, ValueRegressionHead,
        next_event_loss,
    )
    from src.train.run_arm import FromScratchModel
    from src.train.curriculum import curriculum_weights

    dev = torch.device(DEVICE)
    print(f"\n--- {name} ({dev}) ---")

    model = FromScratchModel(vocab_size, n_targets, mcfg).to(dev)

    if freeze_trunk:
        for p in model.enc.parameters():
            p.requires_grad = False
        for p in model.enc.lm_head.parameters():
            p.requires_grad = False

    total = count_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    b = {k: v.to(dev) if isinstance(v, torch.Tensor) else v
         for k, v in batch.items()}
    losses = model(b)
    total_loss = (
        losses["ntp"] * 0.2 + losses["cr"] * 1.0 + losses["th"] * 1.0
        + (losses["val"] or 0) * 0.5
    )
    print(f"  losses: ntp={losses['ntp']:.4f} cr={losses['cr']:.4f} "
          f"th={losses['th']:.4f} val={losses.get('val', 0) or 0:.4f} "
          f"total={total_loss:.4f}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    opt.zero_grad()
    total_loss.backward()
    opt.step()

    grads = sum(p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters() if p.requires_grad)
    print(f"  gradients: {grads} params with non-zero grad")

    assert grads > 0, f"{name}: no gradients flowed — dead model"
    print(f"  PASS")


if __name__ == "__main__":
    unittest.main()