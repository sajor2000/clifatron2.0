"""The real-data smoke driver (`src.train.real_data_smoke`) on synthetic data.

The ablation-arm half reuses `run_tokenization_ablation.setup_arm` / `train_arm`, which
`tests/test_tokenization_ablation.py` covers end to end. This pins the GEM half (next-event
steps on train windows + rollouts from a held-out stay's anchor prefix), the smoke trunk /
schedule overrides, and that the driver refuses a non-sample vocabulary.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TINY_HEADS = {
    "next_event": {"enabled": True, "weight": 0.2},
    "competing_risk": {"enabled": True, "weight": 1.0, "n_time_bins": 4, "horizon_hours": 48},
    "threshold_hazard": {"enabled": True, "weight": 1.0, "horizon_hours": 48,
                         "n_time_bins": 4, "threshold_embed_dim": 4},
    "value_regression": {"enabled": True, "weight": 0.5},
}


class RealDataSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from test_gem_artifact import _build_site
            from test_tokenize_alignment import _repartition
        except ImportError:  # pragma: no cover
            from tests.test_gem_artifact import _build_site
            from tests.test_tokenize_alignment import _repartition
        from src.data.tokenize import tokenize_site
        from src.eval.synthetic_bundle import FIXTURE_POLICY, SYNTHETIC_SITE

        cls._td = tempfile.TemporaryDirectory()
        work = Path(cls._td.name)
        old_cwd = os.getcwd()
        os.chdir(work)
        try:
            site, episodes, cfg = _build_site(work)
            episodes = _repartition(episodes, ["synth-020", "synth-021"])
            kw = {"episodes": episodes, "artifact_policy": FIXTURE_POLICY}
            out = work / "output/intermediate_phi/smoke"
            tokenize_site(cfg, SYNTHETIC_SITE, site, out, None, sample_episodes=20, **kw)
            blob = json.loads((out / "vocab.json").read_text())
            tokenize_site(cfg, SYNTHETIC_SITE, site, out, blob, sample_episodes=20,
                          trajectory="hospitalization", max_tokens=64, **kw)
            full = work / "output/intermediate_phi/full"
            tokenize_site(cfg, SYNTHETIC_SITE, site, full, None, **kw)
            cls.out, cls.full, cls.cfg, cls.site, cls.episodes = out, full, cfg, site, episodes
        finally:
            os.chdir(old_cwd)

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    def _mcfg(self):
        return {"trunk": {"d_model": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2,
                          "dropout": 0.0, "rope_base": 10000.0, "tied_embeddings": False,
                          "target_vocab": 256},
                "heads": json.loads(json.dumps(TINY_HEADS))}

    def test_gem_steps_and_rollouts(self):
        import torch

        from src.train.real_data_smoke import run_gem

        result = run_gem(self.out, mcfg=self._mcfg(), cfg=self.cfg,
                         device=torch.device("cpu"), steps=2, rollouts=2,
                         max_new_tokens=4, batch=2)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["ntp_steps"], 2)
        self.assertEqual(result["rollouts"], 2)
        self.assertEqual(sum(result["rollout_stop_reasons"].values()), 2)

    def test_smoke_configs_shrink_the_trunk_and_never_checkpoint(self):
        from src.train.real_data_smoke import SMOKE_TRUNK, _smoke_configs

        mcfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
        tcfg = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
        small_m, small_t = _smoke_configs(mcfg, tcfg, Path("/tmp/x"), steps=2, batch=2)
        for key, value in SMOKE_TRUNK.items():
            self.assertEqual(small_m["trunk"][key], value)
        self.assertEqual(small_m["trunk"]["target_vocab"], mcfg["trunk"]["target_vocab"])
        self.assertEqual(small_t["schedule"]["total_steps"], 2)
        self.assertGreater(small_t["runtime"]["ckpt_every"], 2)
        self.assertGreater(small_t["eval_schedule"]["val_every"], 2)
        self.assertNotEqual(mcfg["trunk"]["d_model"], small_m["trunk"]["d_model"])

    def test_prepare_arms_refuses_a_non_sample_vocabulary(self):
        import polars as pl

        from src.eval.synthetic_bundle import FIXTURE_COHORT, FIXTURE_POLICY
        from src.train.real_data_smoke import prepare_arms

        with self.assertRaisesRegex(SystemExit, "sample"):
            prepare_arms(self.full, data_dir=self.site, episodes=self.episodes,
                         labels=pl.DataFrame(), cfg=self.cfg, cohort=FIXTURE_COHORT,
                         policy=FIXTURE_POLICY, site="x")
