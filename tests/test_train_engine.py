import os
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.train.engine import setup_ddp, is_distributed, _prepare_batch, _train_one_epoch, _restore_rng_states, TrainConfig
from src.train.pretrain import _has_supervised_outcomes, _load_decile_records
from src.train.manifest import Manifest
from src.train.checkpoint import save_checkpoint, load_checkpoint


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, batch):
        x = batch["input_ids"].float().mean(dim=1, keepdim=True).expand(-1, 4)
        loss = self.fc(x).sum()
        return {"total": loss, "ntp": loss, "cr": loss * 0.5, "th": loss * 0.3, "val": loss * 0.1}


class TestTrainingEngine(unittest.TestCase):

    def setUp(self):
        self.dev = torch.device("cpu")
        self.tcfg = TrainConfig({}, {
            "batch": {"per_gpu": 2, "grad_accum": 2},
            "runtime": {"ckpt_dir": tempfile.mkdtemp(), "ckpt_every": 1},
            "schedule": {"warmup_steps": 10, "total_steps": 10},
            "optimizer": {"grad_clip": 1.0},
        }, {"compile": False}, total_steps=10)

    def test_one_batch_overfit_decreases_loss(self):
        class SingleBatchDS(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            def __getitem__(self, i):
                return {"input_ids": torch.randint(0, 100, (8,)), "attention_mask": torch.ones(8)}

        dl = torch.utils.data.DataLoader(SingleBatchDS(), batch_size=2, shuffle=False)
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000)

        losses = []
        for _ in range(5):
            ml, _, _, _, _ = _train_one_epoch(model, dl, opt, scheduler, 0, self.tcfg, self.dev, rank=0)
            losses.append(ml.loss_total[-1] if ml.loss_total else float('inf'))

        self.assertLess(losses[-1], losses[0] * 0.95, "loss did not decrease over 5 epochs")

    def test_checkpoint_roundtrip(self):
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000)
        manifest = Manifest("test", {}, seed=42, ckpt_dir="/tmp")

        path = Path(tempfile.mktemp(suffix=".pt"))
        save_checkpoint(path, model=model, optimizer=opt, scheduler=scheduler, epoch=3, step=7, manifest=manifest)

        loaded = load_checkpoint(path)
        self.assertEqual(loaded["epoch"], 3)
        self.assertEqual(loaded["step"], 7)
        model.load_state_dict(loaded["model"])
        opt.load_state_dict(loaded["optimizer"])
        scheduler.load_state_dict(loaded["scheduler"])
        self.assertIn("run_id", loaded["manifest"])

    def test_grad_accumulation_partial_final_normalize(self):
        class TinyDS(torch.utils.data.Dataset):
            def __len__(self):
                return 5  # 2*2 + 1 partial
            def __getitem__(self, i):
                return {"input_ids": torch.randint(0, 100, (4,)), "attention_mask": torch.ones(4)}

        dl = torch.utils.data.DataLoader(TinyDS(), batch_size=2, shuffle=False)
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000)
        _, sm, _, _, updates = _train_one_epoch(model, dl, opt, scheduler, 0, self.tcfg, self.dev, rank=0)
        self.assertEqual(sm, 5, "all 5 samples should be counted")
        self.assertEqual(updates, 2, "one full and one partial accumulation should step")

    def test_missing_ntp_mask_counts_batch_tokens_not_cumulative_tokens(self):
        class TokenDS(torch.utils.data.Dataset):
            def __len__(self):
                return 2
            def __getitem__(self, i):
                return {"input_ids": torch.ones(3, dtype=torch.long), "attention_mask": torch.ones(3)}

        dl = torch.utils.data.DataLoader(TokenDS(), batch_size=1, shuffle=False)
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000)
        _, _, tokens, ntp_tokens, _ = _train_one_epoch(
            model, dl, opt, scheduler, 0, self.tcfg, self.dev, rank=0
        )
        self.assertEqual(tokens, 6)
        self.assertEqual(ntp_tokens, 6)

    def test_partial_accumulation_uses_actual_microbatch_count(self):
        class ConstantLossModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor(1.0))
            def forward(self, batch):
                loss = self.w * batch["input_ids"].float().mean()
                return {"total": loss}

        class TwoMicrobatchDS(torch.utils.data.Dataset):
            def __len__(self):
                return 2
            def __getitem__(self, i):
                return {"input_ids": torch.ones(1, dtype=torch.long), "attention_mask": torch.ones(1)}

        cfg = TrainConfig({}, {
            "batch": {"per_gpu": 1, "grad_accum": 4},
            "runtime": {"ckpt_dir": tempfile.mkdtemp(), "ckpt_every": 99},
            "schedule": {"warmup_steps": 1, "total_steps": 1},
            "optimizer": {"grad_clip": None},
        }, {"compile": False}, total_steps=1)
        model = ConstantLossModel()
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000)
        _train_one_epoch(
            model,
            torch.utils.data.DataLoader(TwoMicrobatchDS(), batch_size=1, shuffle=False),
            opt,
            scheduler,
            0,
            cfg,
            self.dev,
            rank=0,
        )
        self.assertAlmostEqual(float(model.w.detach()), 0.0, places=5)

    def test_train_one_epoch_stops_at_max_updates(self):
        class ManyBatchDS(torch.utils.data.Dataset):
            def __len__(self):
                return 20
            def __getitem__(self, i):
                return {"input_ids": torch.randint(0, 100, (4,)), "attention_mask": torch.ones(4)}

        dl = torch.utils.data.DataLoader(ManyBatchDS(), batch_size=2, shuffle=False)
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000)
        _, _, _, _, updates = _train_one_epoch(
            model, dl, opt, scheduler, 0, self.tcfg, self.dev, rank=0, max_updates=1
        )
        self.assertEqual(updates, 1)

    def test_prepare_batch_bridges_collate_output_to_model_contract(self):
        batch = {
            "input_ids": torch.tensor([[3, 4, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
            "anchor_idx": torch.tensor([1]),
            "value_target": torch.tensor([[0.0, 1.5, 0.0]]),
            "value_mask": torch.tensor([[False, True, False]]),
            "document_labels": [[{
                "tte_mask": True,
                "event_cause": 2,
                "event_bin": 4,
                "observed_bins": 5,
                "censored": False,
            }]],
            "threshold_queries": [{
                "target_idx": 1,
                "threshold_bin": 6,
                "direction": 0,
                "threshold_crossed_bin": 4,
            }],
        }
        prepared = _prepare_batch(batch, self.dev)
        self.assertTrue(torch.equal(prepared["token"], batch["input_ids"]))
        self.assertTrue(torch.equal(prepared["last_idx"], batch["anchor_idx"]))
        self.assertEqual(prepared["cr_type"].tolist(), [2])
        self.assertEqual(prepared["cr_bin"].tolist(), [4])
        self.assertEqual(prepared["th_target"].tolist(), [1])
        self.assertEqual(prepared["th_tau"].tolist(), [6])
        self.assertEqual(prepared["th_crossed"].tolist(), [4])
        self.assertTrue(torch.equal(prepared["value"], batch["value_target"]))
        self.assertTrue(torch.equal(prepared["val_mask"], batch["value_mask"]))

    def test_load_decile_records_normalizes_tokenizer_output(self):
        import polars as pl
        tmp = Path(tempfile.mkdtemp()) / "events.parquet"
        pl.DataFrame({
            "hosp_id": ["stay-a"],
            "token": [[3, 4]],
            "pos_min": [[0, 60]],
            "value": [[1.0, 2.0]],
            "target_eligible": [[True, True]],
            "n_events": [2],
        }).write_parquet(tmp)
        records = _load_decile_records(tmp, drop_values_without_stats=True)
        self.assertEqual(records[0]["episode_key"], "stay-a")
        self.assertEqual(records[0]["anchor_idx"], 1)
        self.assertEqual(records[0]["outcomes"], [])
        self.assertEqual(records[0]["value"], [None, None])
        self.assertFalse(_has_supervised_outcomes(records))

    def test_pretrain_model_masks_unlabeled_tte_losses(self):
        from src.data.collate import collate_model_samples
        from src.data.dataset import ModelDataset
        from src.data.targets import TargetBuilder
        from src.train.pretrain import Model

        cfg = {
            "trunk": {
                "d_model": 8,
                "n_layers": 1,
                "n_heads": 2,
                "ffn_mult": 2,
                "dropout": 0.0,
                "rope_base": 10000.0,
                "tied_embeddings": False,
            },
            "heads": {
                "competing_risk": {"n_time_bins": 4},
                "threshold_hazard": {"n_time_bins": 4, "threshold_embed_dim": 4},
                "value_regression": {"enabled": True},
            },
        }
        record = {
            "episode_key": "stay-a",
            "artifact_hashes": {},
            "token": [3, 4, 5],
            "pos_min": [0, 1, 2],
            "value": [None, None, None],
            "target_eligible": [True, True, True],
            "anchor_idx": 2,
            "anchor_min": 2,
            "outcomes": [],
        }
        ds = ModelDataset([record], representation="decile", target_builder=TargetBuilder(16, 4, 4, {}), expected_hashes={})
        batch = _prepare_batch(collate_model_samples([ds[0]]), torch.device("cpu"))
        model = Model(vocab_size=16, n_targets=2, mcfg=cfg)
        losses = model(batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["cr"]), 0.0)
        self.assertEqual(float(losses["th"]), 0.0)

    def test_pretrain_model_handles_anchorless_packed_chunk(self):
        from src.train.pretrain import Model

        cfg = {
            "trunk": {
                "d_model": 8,
                "n_layers": 1,
                "n_heads": 2,
                "ffn_mult": 2,
                "dropout": 0.0,
                "rope_base": 10000.0,
                "tied_embeddings": False,
            },
            "heads": {
                "competing_risk": {"n_time_bins": 4},
                "threshold_hazard": {"n_time_bins": 4, "threshold_embed_dim": 4},
                "value_regression": {"enabled": True},
            },
        }
        batch = {
            "token": torch.tensor([[3, 4, 5]]),
            "pos_min": torch.tensor([[0, 1, 2]]),
            "document_ids": torch.tensor([[0, 0, 0]]),
            "anchor_batch_idx": torch.tensor([], dtype=torch.long),
            "last_idx": torch.tensor([], dtype=torch.long),
            "ntp_target": torch.tensor([[4, 5, 0]]),
            "ntp_mask": torch.tensor([[True, True, False]]),
            "value": torch.zeros(1, 3),
            "val_mask": torch.zeros(1, 3, dtype=torch.bool),
            "cr_mask": torch.zeros(0, dtype=torch.bool),
            "th_mask": torch.zeros(0, dtype=torch.bool),
            "cr_type": torch.zeros(0, dtype=torch.long),
            "cr_bin": torch.zeros(0, dtype=torch.long),
            "th_target": torch.zeros(0, dtype=torch.long),
            "th_tau": torch.zeros(0, dtype=torch.long),
            "th_dir": torch.zeros(0, dtype=torch.long),
            "th_crossed": torch.zeros(0, dtype=torch.long),
        }
        losses = Model(vocab_size=16, n_targets=2, mcfg=cfg)(batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["cr"]), 0.0)
        self.assertEqual(float(losses["th"]), 0.0)

    def test_pretrain_model_rejects_multi_document_dense_path(self):
        from src.train.pretrain import Model

        cfg = {
            "trunk": {
                "d_model": 8,
                "n_layers": 1,
                "n_heads": 2,
                "ffn_mult": 2,
                "dropout": 0.0,
                "rope_base": 10000.0,
                "tied_embeddings": False,
            },
            "heads": {
                "competing_risk": {"n_time_bins": 4},
                "threshold_hazard": {"n_time_bins": 4, "threshold_embed_dim": 4},
                "value_regression": {"enabled": False},
            },
        }
        batch = {
            "token": torch.tensor([[3, 4, 5, 6]]),
            "pos_min": torch.tensor([[0, 1, 0, 1]]),
            "document_ids": torch.tensor([[0, 0, 1, 1]]),
            "last_idx": torch.tensor([1]),
            "ntp_target": torch.tensor([[4, 0, 6, 0]]),
            "ntp_mask": torch.tensor([[True, False, True, False]]),
            "cr_type": torch.tensor([0]),
            "cr_bin": torch.tensor([1]),
            "th_target": torch.tensor([0]),
            "th_tau": torch.tensor([1]),
            "th_dir": torch.tensor([0]),
            "th_crossed": torch.tensor([-1]),
        }
        with self.assertRaisesRegex(RuntimeError, "multi-document packed rows"):
            Model(vocab_size=16, n_targets=2, mcfg=cfg)(batch)

    def test_train_sets_dataset_epoch_for_threshold_sampling(self):
        class EpochDS(torch.utils.data.Dataset):
            def __init__(self):
                self.epoch = None
            def __len__(self):
                return 1
            def __getitem__(self, i):
                return {"input_ids": torch.ones(2, dtype=torch.long), "attention_mask": torch.ones(2)}
            def set_epoch(self, epoch):
                self.epoch = epoch

        ds = EpochDS()
        dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000)
        from src.train.engine import train
        cfg = TrainConfig({}, {
            "batch": {"per_gpu": 1, "grad_accum": 1},
            "runtime": {"ckpt_dir": tempfile.mkdtemp(), "ckpt_every": 99},
            "schedule": {"warmup_steps": 1, "total_steps": 1},
            "optimizer": {"grad_clip": 1.0},
        }, {"compile": False}, total_steps=1)
        train(model, dl, None, opt, scheduler, cfg, self.dev)
        self.assertEqual(ds.epoch, 0)

    def test_resume_carries_forward_manifest_ledger_counters(self):
        class OneBatchDS(torch.utils.data.Dataset):
            def __len__(self):
                return 1
            def __getitem__(self, i):
                return {"input_ids": torch.ones(2, dtype=torch.long), "attention_mask": torch.ones(2)}

        from src.train.engine import train
        ckpt_dir = Path(tempfile.mkdtemp())
        model = TinyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1000)
        manifest = Manifest("test", {}, seed=42, ckpt_dir=str(ckpt_dir))
        manifest.record_ledger(samples=10, tokens=20, ntp_tokens=12, optimizer_step=3)
        ckpt = ckpt_dir / "resume.pt"
        save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=scheduler, epoch=3, step=3, manifest=manifest)

        cfg = TrainConfig({}, {
            "batch": {"per_gpu": 1, "grad_accum": 1},
            "runtime": {"ckpt_dir": str(ckpt_dir), "ckpt_every": 99},
            "schedule": {"warmup_steps": 1, "total_steps": 4},
            "optimizer": {"grad_clip": 1.0},
        }, {"compile": False}, total_steps=4)
        resumed_model = TinyModel()
        resumed_opt = torch.optim.SGD(resumed_model.parameters(), lr=0.01)
        resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_opt, 1000)
        _, resumed_manifest = train(
            resumed_model,
            torch.utils.data.DataLoader(OneBatchDS(), batch_size=1, shuffle=False),
            None,
            resumed_opt,
            resumed_scheduler,
            cfg,
            self.dev,
            resume_ckpt=ckpt,
        )
        self.assertGreaterEqual(resumed_manifest.ledger.get("samples_seen", 0), 10)


class _DropoutModel(torch.nn.Module):
    """A tiny model with dropout, so training consumes the global RNG — the resume
    path must round-trip that RNG for the two runs to end bit-identical."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.drop = torch.nn.Dropout(0.5)
        self.out = torch.nn.Linear(4, 2)

    def forward(self, batch):
        x = batch["input_ids"].float().mean(dim=1, keepdim=True).expand(-1, 4)
        loss = self.out(self.drop(torch.relu(self.fc(x)))).sum()
        return {"total": loss}


class _FixedDS(torch.utils.data.Dataset):
    """Deterministic data — no RNG in __getitem__ — so the only training-time RNG
    consumer is dropout, isolating the resume RNG round-trip."""

    def __len__(self):
        return 6

    def __getitem__(self, i):
        return {"input_ids": torch.arange(i, i + 8) % 100,
                "attention_mask": torch.ones(8)}


class ResumeEquivalenceTest(unittest.TestCase):
    """The load-bearing U4 claim: resume from an epoch-boundary checkpoint yields the
    SAME final parameters as training straight through. Proves the model/optimizer/
    scheduler state AND the RNG round-trip together, on CPU, data-free."""

    def _cfg(self, ckpt_dir):
        return TrainConfig({}, {
            "batch": {"per_gpu": 2, "grad_accum": 1},
            "runtime": {"ckpt_dir": ckpt_dir, "ckpt_every": 1},
            "schedule": {"warmup_steps": 2, "total_steps": 100},
            "optimizer": {"grad_clip": 1.0},
        }, {"compile": False}, total_steps=100)

    def _fresh(self):
        torch.manual_seed(20260829)
        model = _DropoutModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.9)
        return model, opt, sched

    def _params(self, model):
        return [p.detach().clone() for p in model.parameters()]

    def test_resume_from_epoch_boundary_matches_straight_through(self):
        dev = torch.device("cpu")
        total_epochs, split = 4, 2

        # Straight through: same seed governs init AND the dropout RNG stream.
        torch.manual_seed(7)
        model_s, opt_s, sched_s = self._fresh()
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(td)
            dl = torch.utils.data.DataLoader(_FixedDS(), batch_size=2, shuffle=False)
            for ep in range(total_epochs):
                _train_one_epoch(model_s, dl, opt_s, sched_s, ep, cfg, dev, rank=0)
            straight = self._params(model_s)

        # Split run: train `split` epochs, checkpoint AT the boundary with RNG, then a
        # brand-new model/opt/sched resumes and trains the remainder.
        torch.manual_seed(7)
        model_a, opt_a, sched_a = self._fresh()
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg(td)
            dl = torch.utils.data.DataLoader(_FixedDS(), batch_size=2, shuffle=False)
            for ep in range(split):
                _train_one_epoch(model_a, dl, opt_a, sched_a, ep, cfg, dev, rank=0)
            ckpt = Path(td) / "boundary.pt"
            save_checkpoint(ckpt, model=model_a, optimizer=opt_a, scheduler=sched_a,
                            epoch=split, step=split,
                            rng_states=[{"cpu": torch.get_rng_state().tolist(), "cuda": {}}],
                            manifest=Manifest("t", {}, seed=7, ckpt_dir=td))

            # Perturb global RNG so a missing restore would change the outcome.
            torch.manual_seed(123456)
            model_b = _DropoutModel()
            opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1, momentum=0.9)
            sched_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=2, gamma=0.9)
            loaded = load_checkpoint(ckpt)
            model_b.load_state_dict(loaded["model"])
            opt_b.load_state_dict(loaded["optimizer"])
            sched_b.load_state_dict(loaded["scheduler"])
            _restore_rng_states(loaded["rng_states"])
            for ep in range(split, total_epochs):
                _train_one_epoch(model_b, dl, opt_b, sched_b, ep, cfg, dev, rank=0)
            resumed = self._params(model_b)

        for a, b in zip(straight, resumed):
            self.assertTrue(torch.equal(a, b),
                            f"resume diverged from straight-through: max|d|={ (a-b).abs().max() }")

    def test_production_train_resume_matches_straight_through(self):
        """Drive the real `train(..., resume_ckpt=)` path, not a hand-rolled loop, so a
        regression that stops restoring RNG or mishandles epoch/step is caught. The
        checkpoint must land on an EPOCH BOUNDARY (3 batches/epoch here) — resume
        re-iterates the epoch, so a mid-epoch checkpoint would change the data order and
        is not claimed equivalent."""
        from src.train.engine import train

        dev = torch.device("cpu")

        def cfg(td, total_steps, ckpt_every):
            return TrainConfig({}, {
                "batch": {"per_gpu": 2, "grad_accum": 1},
                "runtime": {"ckpt_dir": td, "ckpt_every": ckpt_every},
                "schedule": {"warmup_steps": 2, "total_steps": total_steps},
                "eval_schedule": {"val_every": 10_000},
                "optimizer": {"grad_clip": 1.0},
            }, {"compile": False}, total_steps=total_steps)

        def build():
            torch.manual_seed(20260829)
            m = _DropoutModel()
            o = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
            s = torch.optim.lr_scheduler.StepLR(o, step_size=2, gamma=0.9)
            return m, o, s

        # 6 samples, batch 2 -> 3 updates/epoch; boundaries at steps 3, 6.
        def loader():
            return torch.utils.data.DataLoader(_FixedDS(), batch_size=2, shuffle=False)

        torch.manual_seed(7)
        m_s, o_s, s_s = build()
        with tempfile.TemporaryDirectory() as td:
            train(m_s, loader(), None, o_s, s_s, cfg(td, 6, 999), dev, seed=7)
            straight = [p.detach().clone() for p in m_s.parameters()]

        torch.manual_seed(7)
        m_a, o_a, s_a = build()
        with tempfile.TemporaryDirectory() as td:
            train(m_a, loader(), None, o_a, s_a, cfg(td, 3, 3), dev, seed=7)  # 1 epoch, saves at step 3
            ckpts = list(Path(td).glob("ckpt_*.pt"))
            self.assertTrue(ckpts, "train() did not write an epoch-boundary checkpoint")
            ckpt = ckpts[0]

            torch.manual_seed(999_999)  # perturb: a missing RNG restore would diverge
            m_b, o_b, s_b = build()
            train(m_b, loader(), None, o_b, s_b, cfg(td, 6, 999), dev,
                  resume_ckpt=ckpt, seed=7)
            resumed = [p.detach().clone() for p in m_b.parameters()]

        for a, b in zip(straight, resumed):
            self.assertTrue(torch.equal(a, b),
                            f"train() resume diverged: max|d|={ (a-b).abs().max() }")


if __name__ == "__main__":
    unittest.main()
