"""U4 (R31, KTD5): objective arms and the curriculum driven by the engine's update counter.

Data-free: the tiny `pretrain.Model` and synthetic microbatches of test_ddp_multihead,
trained single-process on CPU through `engine.train`. The two-rank (gloo) proof of the
warm-up lives in tests/test_ddp_multihead.py.
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.train import engine
from src.train.curriculum import (
    HEADS,
    apply_objective_arm,
    compute_budget,
    curriculum_weights,
    load_objective_arms,
    resolve_objective_arm,
)
from src.train.engine import TrainConfig
from src.train.pretrain import build_optimizer, build_scheduler

try:  # pytest puts tests/ on sys.path (rootdir-less test modules)
    from test_ddp_multihead import (
        build_model, head_state, record_weights, synthetic_batch, tiny_mcfg, tiny_tcfg)
except ImportError:  # pragma: no cover - run from the repo root as a package
    from tests.test_ddp_multihead import (
        build_model, head_state, record_weights, synthetic_batch, tiny_mcfg, tiny_tcfg)

ROOT = Path(__file__).resolve().parents[1]
ARMS_CONFIG = ROOT / "configs/objective_arms.yaml"
BINDING = {"tokenizer_version": "2", "vocabulary": "a" * 64, "numeric_edges": "b" * 64}
CPU = torch.device("cpu")
TOTAL = 20          # warm-up: updates 0..2; transition: update 3 (blend 0); configured: 4..19
FULL = (0.2, 1.0, 1.0, 0.5)
NTP_ONLY = (1.0, 0.0, 0.0, 0.0)

EXPECTED_ARMS = {
    "full": (FULL, True),
    "next_token_only": (NTP_ONLY, False),
    "minus_value": ((0.2, 1.0, 1.0, 0.0), True),
    "minus_competing_risk": ((0.2, 0.0, 1.0, 0.5), True),
    "minus_threshold": ((0.2, 1.0, 0.0, 0.5), True),
    "no_curriculum": (FULL, False),
}


def loader(n: int = 4):
    batches = [synthetic_batch(500 + i) for i in range(n)]
    return torch.utils.data.DataLoader(batches, batch_size=None, shuffle=False)


def run(mcfg: dict, ckpt_dir, *, total_steps: int = TOTAL, grad_accum: int = 1,
        ckpt_every: int = 1000, n_batches: int = 4, resume=None, seed: int = 0):
    """`engine.train` over a fresh tiny model; returns (model, weights per forward,
    manifest, stdout)."""
    model = build_model(mcfg, seed=seed)
    used = record_weights(model)
    tcfg = tiny_tcfg(ckpt_dir, grad_accum=grad_accum, ckpt_every=ckpt_every)
    opt = build_optimizer(model, lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _, manifest = engine.train(
            model, loader(n_batches), None, opt, build_scheduler(opt, total_steps, 1),
            TrainConfig({}, tcfg, mcfg, total_steps), CPU, vocab_binding=BINDING,
            resume_ckpt=resume)
    return model, used, manifest, out.getvalue(), opt


class ArmConfigTest(unittest.TestCase):
    def test_each_arm_resolves_to_its_weights_and_curriculum_flag(self):
        arms = load_objective_arms(ARMS_CONFIG)
        self.assertEqual(set(arms), set(EXPECTED_ARMS))
        for name, (weights, curriculum) in EXPECTED_ARMS.items():
            arm = resolve_objective_arm(name, ARMS_CONFIG)
            with self.subTest(arm=name):
                self.assertEqual(tuple(arm.weights[h] for h in HEADS), weights)
                self.assertIs(arm.curriculum, curriculum)

    def test_unknown_arm_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown objective arm 'minus_everything'"):
            resolve_objective_arm("minus_everything", ARMS_CONFIG)

    def test_all_arms_share_steps_batch_size_and_token_budget(self):
        tcfg = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
        mcfg = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
        budgets = {}
        for name, arm in load_objective_arms(ARMS_CONFIG).items():
            applied = apply_objective_arm(mcfg, arm)
            # An arm changes the objective and nothing else in the model config.
            for key in set(mcfg) - {"heads", "curriculum"}:
                self.assertEqual(applied[key], mcfg[key], f"{name} changed {key}")
            self.assertEqual({k: {kk: vv for kk, vv in v.items() if kk != "weight"}
                              for k, v in applied["heads"].items()},
                             {k: {kk: vv for kk, vv in v.items() if kk != "weight"}
                              for k, v in mcfg["heads"].items()})
            budgets[name] = compute_budget(tcfg)
        self.assertEqual(len({tuple(b.items()) for b in budgets.values()}), 1, budgets)
        self.assertEqual(budgets["full"]["total_steps"], tcfg["schedule"]["total_steps"])

    def test_an_arm_that_sets_compute_is_refused(self):
        spec = yaml.safe_load(ARMS_CONFIG.read_text())
        for key, value in (("total_steps", 1000), ("per_gpu_batch", 8),
                           ("token_budget", 4096), ("lr", 1e-3)):
            bad = {"arms": {"full": dict(spec["arms"]["full"], **{key: value})}}
            with tempfile.TemporaryDirectory() as td, self.subTest(key=key):
                path = Path(td) / "arms.yaml"
                path.write_text(yaml.safe_dump(bad))
                with self.assertRaisesRegex(ValueError, "equal compute"):
                    load_objective_arms(path)

    def test_an_arm_must_set_every_head_and_a_known_curriculum(self):
        spec = yaml.safe_load(ARMS_CONFIG.read_text())["arms"]["full"]
        cases = {
            "heads must set": dict(spec, heads={"next_event": 1.0}),
            "curriculum": dict(spec, curriculum="tte_first"),
            "negative": dict(spec, heads=dict(spec["heads"], value_regression=-1.0)),
        }
        for message, arm in cases.items():
            with tempfile.TemporaryDirectory() as td, self.subTest(case=message):
                path = Path(td) / "arms.yaml"
                path.write_text(yaml.safe_dump({"arms": {"x": arm}}))
                with self.assertRaisesRegex(ValueError, message):
                    load_objective_arms(path)

    def test_pretrain_cli_refuses_an_unknown_arm_before_touching_data(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            env.pop(key, None)
        done = subprocess.run(
            [sys.executable, "-m", "src.train.pretrain", "--data", "/nonexistent",
             "--site", "synthetic", "--objective-arm", "minus_everything"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("unknown objective arm", done.stderr)
        self.assertNotIn("vocab.json", done.stderr)


class CurriculumByUpdateTest(unittest.TestCase):
    def test_accumulation_of_four_advances_the_schedule_once_per_update(self):
        with tempfile.TemporaryDirectory() as td:
            _, used, manifest, _, _ = run(tiny_mcfg(curriculum="ntp_then_tte"), td,
                                          grad_accum=4, n_batches=8)
        self.assertEqual(manifest.ledger["optimizer_updates"], TOTAL)
        self.assertEqual(len(used), 4 * TOTAL)
        for update in range(TOTAL):
            expected = tuple(curriculum_weights(update, TOTAL, target=FULL)[:4])
            with self.subTest(update=update):
                self.assertEqual(used[4 * update: 4 * update + 4], [expected] * 4)
        # Counted per microbatch, the configured weights would arrive at forward 4; per
        # update they arrive at update 4 = forward 16.
        self.assertEqual(used[:16], [NTP_ONLY] * 16)
        self.assertEqual(used[16:], [FULL] * (4 * TOTAL - 16))

    def test_inside_the_transition_the_weights_blend_and_after_it_are_configured(self):
        total = 100   # warm-up 0..14, transition 15..19, configured from 20
        with tempfile.TemporaryDirectory() as td:
            _, used, _, _, _ = run(tiny_mcfg(curriculum="ntp_then_tte"), td, total_steps=total)
        self.assertEqual(used[:15], [NTP_ONLY] * 15)
        for update, progress in zip(range(15, 20), (0.0, 0.2, 0.4, 0.6, 0.8)):
            ntp, cr, th, val = used[update]
            with self.subTest(update=update):
                self.assertAlmostEqual(ntp, 1.0 - 0.8 * progress)
                self.assertAlmostEqual(cr, progress)
                self.assertAlmostEqual(th, progress)
                self.assertAlmostEqual(val, 0.5 * progress)
        self.assertEqual(used[20:], [FULL] * (total - 20))

    def test_no_curriculum_uses_the_configured_weights_from_the_first_update(self):
        arm = resolve_objective_arm("no_curriculum", ARMS_CONFIG)
        with tempfile.TemporaryDirectory() as td:
            _, used, _, _, _ = run(apply_objective_arm(tiny_mcfg(), arm), td)
        self.assertEqual(used, [FULL] * TOTAL)

    def test_resume_after_the_transition_continues_with_the_configured_weights(self):
        mcfg = tiny_mcfg(curriculum="ntp_then_tte")
        with tempfile.TemporaryDirectory() as td:
            _, first, _, _, _ = run(mcfg, td, total_steps=8, ckpt_every=6)
            ckpt = Path(td) / "ckpt_ep1_step6.pt"
            self.assertTrue(ckpt.exists(), sorted(p.name for p in Path(td).iterdir()))
            self.assertEqual(first[5], FULL)          # update 5 is past the transition
            # A fresh model and engine resume the 20-update plan at update 6.
            model, used, manifest, _, opt = run(mcfg, td, resume=ckpt, seed=1)
        self.assertEqual(manifest.ledger["optimizer_updates"], TOTAL)
        self.assertEqual(len(used), TOTAL - 6)
        self.assertEqual(used[0], FULL)               # not the warm-up's (1, 0, 0, 0)
        self.assertEqual(set(used), {FULL})
        # The per-head optimizer groups came back from the checkpoint, decaying again.
        heads = [g for g in opt.param_groups if "head" in g]
        self.assertEqual([g["head"] for g in heads],
                         ["competing_risk", "threshold_hazard", "value_regression"])
        self.assertTrue(all(g["weight_decay"] == 0.1 for g in heads))
        self.assertEqual(model.loss_weights, dict(zip(HEADS, FULL)))


class HeadWeightDecayTest(unittest.TestCase):
    def test_heads_are_bit_identical_through_the_warm_up_and_move_after_it(self):
        mcfg = tiny_mcfg(curriculum="ntp_then_tte")
        initial = build_model(mcfg).state_dict()
        with tempfile.TemporaryDirectory() as td:
            run(mcfg, td, ckpt_every=1)
            saved = {step: torch.load(Path(td) / f"ckpt_ep{(step - 1) // 4 + (step % 4 == 0)}"
                                      f"_step{step}.pt", weights_only=False)["model"]
                     for step in range(1, 7)}
        for step in (1, 2, 3, 4):     # warm-up updates 0..2 and transition update 3 (blend 0)
            for name, value in head_state(saved[step]).items():
                self.assertTrue(torch.equal(value, initial[name]), f"step {step}: {name}")
            self.assertFalse(torch.equal(saved[step]["enc.blocks.0.qkv.weight"],
                                         initial["enc.blocks.0.qkv.weight"]))
        for name in ("cr.fc.weight", "th.mlp.0.weight", "vr.mlp.0.weight"):
            self.assertFalse(torch.equal(saved[6][name], initial[name]), name)

    def test_a_flat_adamw_group_would_decay_a_zero_weight_head(self):
        """Why the head groups exist: the same warm-up with one AdamW group shrinks the
        heads (zero gradient, nonzero decoupled decay); the engine refuses that setup."""
        mcfg = tiny_mcfg(curriculum="ntp_then_tte")
        model = build_model(mcfg)
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
        with tempfile.TemporaryDirectory() as td:
            cfg = TrainConfig({}, tiny_tcfg(td), mcfg, TOTAL)
            engine._train_one_epoch(model, loader(2), opt, build_scheduler(opt, TOTAL, 1),
                                    0, cfg, CPU, rank=0)
            self.assertEqual(model.loss_weights["competing_risk"], 0.0)
            self.assertFalse(torch.equal(model.state_dict()["cr.fc.weight"],
                                         initial["cr.fc.weight"]))
            with self.assertRaisesRegex(ValueError, "build_optimizer"):
                engine.train(model, loader(), None, opt,
                             build_scheduler(opt, TOTAL, 1), cfg, CPU, vocab_binding=BINDING)

    def test_head_groups_switch_decay_with_the_curriculum_weight(self):
        model = build_model(tiny_mcfg(curriculum="ntp_then_tte"))
        opt = build_optimizer(model, lr=1e-2, weight_decay=0.1, betas=(0.9, 0.95))
        self.assertEqual(len(opt.param_groups), 4)
        owned = sum(len(g["params"]) for g in opt.param_groups)
        self.assertEqual(owned, len(list(model.parameters())))
        decay = lambda: {g.get("head", "trunk"): g["weight_decay"] for g in opt.param_groups}
        engine._apply_objective_step(model, opt, 0, TOTAL)
        self.assertEqual(decay(), {"trunk": 0.1, "competing_risk": 0.0,
                                   "threshold_hazard": 0.0, "value_regression": 0.0})
        engine._apply_objective_step(model, opt, 4, TOTAL)
        self.assertEqual(set(decay().values()), {0.1})


class ArmRunTest(unittest.TestCase):
    def test_short_cpu_run_of_each_arm_logs_its_configured_schedule(self):
        for name, (weights, curriculum) in EXPECTED_ARMS.items():
            arm = resolve_objective_arm(name, ARMS_CONFIG)
            mcfg = apply_objective_arm(tiny_mcfg(), arm)
            initial = build_model(mcfg).state_dict()
            with tempfile.TemporaryDirectory() as td, self.subTest(arm=name):
                model, used, manifest, log, _ = run(mcfg, td)
                target = " ".join(f"{h}={w:g}" for h, w in zip(HEADS, weights))
                if curriculum:
                    line = (f"objective arm={name} loss_balancing=fixed curriculum="
                            f"ntp_then_tte: updates 0-2 next_event only; updates 3-3 "
                            f"linear blend; updates 4-19 at {target}")
                else:
                    line = (f"objective arm={name} loss_balancing=fixed curriculum=none: "
                            f"updates 0-19 at {target}")
                self.assertIn(line, log)
                self.assertIn("w=", log)
                self.assertEqual(used[-1], weights)
                self.assertEqual(used[0], NTP_ONLY if curriculum else weights)
                record = manifest.config["objective"]
                self.assertEqual(record["objective_arm"], name)
                self.assertEqual(record["trained_heads"],
                                 [h for h, w in zip(HEADS, weights) if w > 0])
                # A head the arm removes stays exactly at its initialisation.
                prefixes = tuple(prefix for head, prefix in
                                 zip(HEADS[1:], ("cr.", "th.", "vr.")) if
                                 weights[HEADS.index(head)] == 0.0)
                for pname, value in head_state(model.state_dict(), prefixes).items():
                    self.assertTrue(torch.equal(value, initial[pname]), pname)


if __name__ == "__main__":
    unittest.main()
