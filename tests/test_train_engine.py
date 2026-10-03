import contextlib
import io
import math
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.train import engine
from src.train.engine import _prepare_batch, _train_one_epoch, _restore_rng_states, TrainConfig
from src.train.pretrain import _has_supervised_outcomes, _load_decile_records
from src.train.manifest import Manifest
from src.train.checkpoint import save_checkpoint, load_checkpoint

try:  # pytest puts tests/ on sys.path (rootdir-less test modules)
    from test_ddp_multihead import build_model, synthetic_batch, tiny_mcfg, tiny_tcfg
except ImportError:  # pragma: no cover - run from the repo root as a package
    from tests.test_ddp_multihead import build_model, synthetic_batch, tiny_mcfg, tiny_tcfg

ROOT = Path(__file__).resolve().parents[1]

# engine.train requires the training vocabulary binding (segments.artifact_binding);
# these data-free tests bind every run (and resume checkpoint) to one fixed binding.
BINDING = {"tokenizer_version": "2", "vocabulary": "a" * 64, "numeric_edges": "b" * 64}


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
        from src.data.collate import collate_model_samples
        from src.data.dataset import ModelDataset
        from src.data.targets import TargetBuilder

        record = {
            "episode_key": "stay-a",
            "artifact_hashes": {"tokenizer_version": "2", "vocabulary": "v",
                                "numeric_edges": "s"},
            "token": [3, 4], "pos_min": [0, 30], "value": [None, 1.5],
            "target_eligible": [True, True], "anchor_idx": 1, "anchor_min": 30,
            "outcomes": [{"target_idx": 1, "status": "positive",
                          "time_from_anchor_hours": 12.5, "threshold_bin": 6,
                          "direction": "below"}],
        }
        ds = ModelDataset([record], representation="decile",
                          target_builder=TargetBuilder(16, 16, 48, {4: (1.0, 1.0)}),
                          expected_hashes={})
        batch = collate_model_samples([ds[0]])
        prepared = _prepare_batch(batch, self.dev)
        self.assertTrue(torch.equal(prepared["token"], batch["input_ids"]))
        self.assertTrue(torch.equal(prepared["last_idx"], batch["anchor_idx"]))
        self.assertTrue(torch.equal(prepared["value"], batch["value_target"]))
        self.assertTrue(torch.equal(prepared["val_mask"], batch["value_mask"]))
        self.assertEqual(prepared["cr_type"].tolist(), [1])
        self.assertEqual(prepared["th_target"].tolist(), [1])
        self.assertEqual(prepared["th_tau"].tolist(), [6])
        # KTD4: the engine passes label times through as minutes since the anchor and
        # bins nothing; it used to hand the hourly threshold head a bin of the
        # competing-risk grid (`th_crossed` 4 for this 12.5 h crossing).
        self.assertEqual(prepared["cr_time_min"].tolist(), [750])
        self.assertEqual(prepared["th_time_min"].tolist(), [750])
        for binned in ("cr_bin", "th_crossed", "th_observed_bin"):
            self.assertNotIn(binned, prepared)

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

    def test_load_decile_records_filters_to_training_partition(self):
        import polars as pl
        tmp = Path(tempfile.mkdtemp()) / "events.parquet"
        pl.DataFrame({
            "hosp_id": ["train-stay", "sealed-stay"],
            "token": [[3, 4], [5, 6]],
            "pos_min": [[0, 60], [0, 60]],
            "value": [[1.0, 2.0], [999999.0, 999999.0]],
            "target_eligible": [[True, True], [True, True]],
            "partition": ["train", "internal_test"],
        }).write_parquet(tmp)

        records = _load_decile_records(tmp, partition="train")

        self.assertEqual([record["episode_key"] for record in records], ["train-stay"])
        self.assertEqual(records[0]["partition"], "train")

    def test_training_partition_filter_fails_closed_without_partition_column(self):
        import polars as pl
        tmp = Path(tempfile.mkdtemp()) / "events.parquet"
        pl.DataFrame({
            "hosp_id": ["stay-a"],
            "token": [[3, 4]],
            "pos_min": [[0, 60]],
            "value": [[1.0, 2.0]],
            "target_eligible": [[True, True]],
        }).write_parquet(tmp)

        with self.assertRaisesRegex(ValueError, "partition column"):
            _load_decile_records(tmp, partition="train")

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
            "artifact_hashes": {"tokenizer_version": "2", "vocabulary": "v",
                                "numeric_edges": "s"},
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
        model = Model(vocab_size=16, n_targets=2, mcfg=cfg, n_value_bins=4)
        losses = model(batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["cr"].detach()), 0.0)
        self.assertEqual(float(losses["th"].detach()), 0.0)

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
        losses = Model(vocab_size=16, n_targets=2, mcfg=cfg, n_value_bins=4)(batch)
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["cr"].detach()), 0.0)
        self.assertEqual(float(losses["th"].detach()), 0.0)

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
            Model(vocab_size=16, n_targets=2, mcfg=cfg, n_value_bins=4)(batch)

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
        train(model, dl, None, opt, scheduler, cfg, self.dev, vocab_binding=BINDING)
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
        save_checkpoint(ckpt, model=model, optimizer=opt, scheduler=scheduler, epoch=3, step=3,
                        manifest=manifest, vocab_binding=BINDING)

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
            vocab_binding=BINDING,
            resume_ckpt=ckpt,
        )
        self.assertGreaterEqual(resumed_manifest.ledger.get("samples_seen", 0), 10)


class SkippedHeadObjectiveTest(unittest.TestCase):
    """KTD2: a skipped head adds a zero-valued term that reaches its parameters, so the
    loss VALUES are what they were before and only the gradient bookkeeping changes."""

    # fp32 `pretrain.Model` losses on `synthetic_batch(11)`, model seed 0, captured on the
    # commit before the zero-valued touch term was added (torch 2.13.0).
    # U3 (KTD4) then changed ONE number on purpose: the competing-risk loss of a censored
    # sample no longer credits the partially observed bin (`full` cr was 1.809082, total
    # 4.623174; batch 11 holds one censored sample). Event-only batches, and the ntp / th
    # / val columns, are unchanged — see tests/test_cr_invariants.py.
    BASELINE_LOSSES = {
        "full": {"ntp": 3.525213, "cr": 0.920133, "th": 1.886438, "val": 0.445222,
                 "total": 3.734225},
        "no_anchor": {"ntp": 3.525213, "cr": 0.0, "th": 0.0, "val": 0.445222,
                      "total": 0.927654},
        "tte_zero": {"ntp": 3.525213, "cr": 0.0, "th": 0.0, "val": 0.445222,
                     "total": 0.927654},
        "all_heads_zero": {"ntp": 3.525213, "cr": 0.0, "th": 0.0, "val": 0.0,
                           "total": 0.705043},
    }
    # Total loss over four AdamW (lr 1e-2, weight decay 0.1) steps on batches 0..3, same
    # commit. Every head is either supervised or skipped on EVERY step in these runs.
    # `full` moved with the censored-credit fix above (batches 0-2 each hold a censored
    # sample; it was [5.227314, 4.269813, 3.331987, 6.591049]).
    BASELINE_TRAJECTORIES = {
        "full": [4.600975, 3.906291, 2.90259, 6.548203],
        "tte_zero": [0.804643, 1.152664, 1.010028, 0.84273],
        "all_heads_zero": [0.705037, 0.689857, 0.723751, 0.717308],
        "no_anchor": [0.804643, 1.152664, 1.010028, 0.84273],
    }

    @staticmethod
    def _case(name):
        mcfg = {
            "full": tiny_mcfg(),
            "no_anchor": tiny_mcfg(),
            "tte_zero": tiny_mcfg(competing_risk=0.0, threshold_hazard=0.0),
            "all_heads_zero": tiny_mcfg(competing_risk=0.0, threshold_hazard=0.0,
                                        value_regression=0.0),
        }[name]
        return mcfg, name != "no_anchor"

    def test_single_process_losses_match_the_pre_change_baseline(self):
        for name, expected in self.BASELINE_LOSSES.items():
            mcfg, supervised = self._case(name)
            batch = _prepare_batch(synthetic_batch(11, supervised=supervised),
                                   torch.device("cpu"))
            losses = build_model(mcfg)(batch)
            for key, value in expected.items():
                with self.subTest(case=name, loss=key):
                    self.assertAlmostEqual(float(losses[key].detach()), value, delta=1e-4)
            # A skipped head reports exactly zero, not a small number.
            for key in ("cr", "th", "val"):
                if expected[key] == 0.0:
                    self.assertEqual(float(losses[key].detach()), 0.0, f"{name} {key}")

    def test_single_process_training_losses_match_the_pre_change_baseline(self):
        for name, expected in self.BASELINE_TRAJECTORIES.items():
            mcfg, supervised = self._case(name)
            model = build_model(mcfg)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1,
                                    betas=(0.9, 0.95))
            observed = []
            for seed in range(4):
                batch = _prepare_batch(synthetic_batch(seed, supervised=supervised),
                                       torch.device("cpu"))
                losses = model(batch)
                opt.zero_grad(set_to_none=True)
                losses["total"].backward()
                opt.step()
                observed.append(float(losses["total"].detach()))
            for step, (got, want) in enumerate(zip(observed, expected, strict=True)):
                with self.subTest(case=name, step=step):
                    self.assertAlmostEqual(got, want, delta=1e-4)

    def test_every_trainable_parameter_gets_a_gradient_whatever_is_skipped(self):
        """The DDP invariant, checked in one process: the set of parameters with a
        gradient is the full trainable set for every weight / supervision combination,
        and a skipped head's gradient is exactly zero."""
        skipped_heads = {
            "full": (),
            "no_anchor": ("cr.", "th."),
            "tte_zero": ("cr.", "th."),
            "all_heads_zero": ("cr.", "th.", "vr."),
        }
        for name, prefixes in skipped_heads.items():
            mcfg, supervised = self._case(name)
            model = build_model(mcfg)
            batch = _prepare_batch(synthetic_batch(11, supervised=supervised),
                                   torch.device("cpu"))
            model(batch)["total"].backward()
            for pname, param in model.named_parameters():
                with self.subTest(case=name, parameter=pname):
                    self.assertIsNotNone(param.grad, "parameter received no gradient")
                    if pname.startswith(prefixes) and prefixes:
                        self.assertEqual(float(param.grad.abs().max()), 0.0)
            if not prefixes:
                for head in (model.cr, model.th, model.vr):
                    self.assertGreater(
                        max(float(p.grad.abs().max()) for p in head.parameters()), 0.0)

    def test_batch_without_head_target_keys_still_touches_the_heads(self):
        """A context-only shard (no outcome join) carries no CR / threshold / value keys."""
        batch = _prepare_batch(synthetic_batch(11), torch.device("cpu"))
        for key in [k for k in batch if k.startswith(("cr_", "th_"))] + ["value", "val_mask",
                                                                         "value_target",
                                                                         "value_mask"]:
            del batch[key]
        model = build_model(tiny_mcfg())
        losses = model(batch)
        self.assertEqual([float(losses[k].detach()) for k in ("cr", "th", "val")], [0.0, 0.0, 0.0])
        losses["total"].backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_zero_touch_is_a_plain_constant_without_trainable_parameters(self):
        from src.train.pretrain import zero_touch

        like = torch.ones(2, dtype=torch.bfloat16)
        head = torch.nn.Linear(3, 2)
        touched = zero_touch(head, like)
        self.assertEqual((float(touched.detach()), touched.dtype), (0.0, torch.bfloat16))
        self.assertTrue(touched.requires_grad)
        for param in head.parameters():
            param.requires_grad_(False)
        for constant in (zero_touch(head, like), zero_touch(None, like)):
            self.assertEqual((float(constant), constant.dtype), (0.0, torch.bfloat16))
            self.assertFalse(constant.requires_grad)

    def test_disabled_value_head_adds_no_parameters_and_reports_zero(self):
        mcfg = tiny_mcfg()
        mcfg["heads"]["value_regression"]["enabled"] = False
        model = build_model(mcfg)
        self.assertIsNone(model.vr)
        losses = model(_prepare_batch(synthetic_batch(11), torch.device("cpu")))
        self.assertEqual(float(losses["val"].detach()), 0.0)
        losses["total"].backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


class DistributedLaunchGuardTest(unittest.TestCase):
    """A distributed launch without CUDA fails closed unless `--allow-cpu-ddp` is given;
    single-process CPU / MPS runs stay allowed."""

    LAUNCH_ENV = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}

    def _environment(self, **env):
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in self.LAUNCH_ENV:
            if key not in env:
                os.environ.pop(key, None)

    def _no_cuda(self):
        patcher = mock.patch("torch.cuda.is_available", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_distributed_launch_without_cuda_is_refused(self):
        self._environment(**self.LAUNCH_ENV)
        self._no_cuda()
        with mock.patch.object(engine.dist, "init_process_group") as init:
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable.*--allow-cpu-ddp"):
                engine.setup_ddp()
        init.assert_not_called()

    def test_world_size_above_one_without_a_rank_is_refused_too(self):
        self._environment(WORLD_SIZE="2")
        self._no_cuda()
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            engine.setup_ddp()

    def test_cpu_flag_joins_a_gloo_group(self):
        self._environment(RANK="1", LOCAL_RANK="1", WORLD_SIZE="2")
        self._no_cuda()
        with mock.patch.object(engine.dist, "init_process_group") as init, \
                mock.patch.object(engine.dist, "get_rank", return_value=1):
            self.assertEqual(engine.setup_ddp(allow_cpu=True), (1, False))
        init.assert_called_once_with("gloo")

    def test_cuda_launch_still_uses_nccl(self):
        self._environment(**self.LAUNCH_ENV)
        with mock.patch("torch.cuda.is_available", return_value=True), \
                mock.patch("torch.cuda.set_device") as set_device, \
                mock.patch.object(engine.dist, "init_process_group") as init, \
                mock.patch.object(engine.dist, "get_rank", return_value=0):
            self.assertEqual(engine.setup_ddp(), (0, True))
        init.assert_called_once_with("nccl")
        set_device.assert_called_once_with(0)

    def test_single_process_launch_needs_no_flag_and_no_process_group(self):
        self._environment()
        self._no_cuda()
        with mock.patch.object(engine.dist, "init_process_group") as init:
            self.assertEqual(engine.setup_ddp(), (0, True))
        init.assert_not_called()
        self.assertIn(engine.select_device(0).type, ("cpu", "mps"))

    def test_cpu_rehearsal_trains_on_cpu_not_mps(self):
        self._no_cuda()
        with mock.patch.object(engine, "is_distributed", return_value=True):
            self.assertEqual(engine.select_device(1), torch.device("cpu"))
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(engine.select_device(1), torch.device("cuda:1"))

    def test_precision_note_says_what_bf16_does_off_cuda(self):
        self.assertIsNone(engine.precision_note(torch.device("cuda:0")))
        self.assertIn("inert on mps", engine.precision_note(torch.device("mps")))
        self._no_cuda()
        self.assertIn("CPU autocast", engine.precision_note(torch.device("cpu")))
        with mock.patch("torch.cuda.is_available", return_value=True):
            # The engine's autocast is cuda-scoped there, so a CPU model runs fp32.
            self.assertIn("inert on cpu", engine.precision_note(torch.device("cpu")))

    def _cli(self, module, *args, **env):
        """Run a training CLI as one rank of a distributed launch on a CUDA-less node."""
        child_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **env)
        return subprocess.run(
            [sys.executable, "-m", module, *args], cwd=ROOT, env=child_env,
            capture_output=True, text=True, timeout=180)

    def test_training_clis_refuse_a_cpu_distributed_launch(self):
        for module, args in (
            ("src.train.pretrain", ("--data", "/nonexistent", "--site", "synthetic")),
            ("src.train.run_tokenization_ablation", ("--arm", "clinical_soft")),
        ):
            with self.subTest(module=module):
                done = self._cli(module, *args, **self.LAUNCH_ENV)
                self.assertNotEqual(done.returncode, 0)
                self.assertIn("CUDA is unavailable", done.stderr)
                self.assertIn("--allow-cpu-ddp", done.stderr)

    @unittest.skipUnless(engine.dist.is_available() and engine.dist.is_gloo_available(),
                         "torch.distributed gloo backend is not built on this platform")
    def test_pretrain_cli_flag_passes_the_guard(self):
        """With the flag, a one-rank CPU launch gets past the guard and stops at the next
        gate (the vocabulary it was pointed at does not exist)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        done = self._cli("src.train.pretrain", "--data", "/nonexistent", "--site", "synthetic",
                         "--allow-cpu-ddp", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
                         MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        self.assertNotEqual(done.returncode, 0)
        self.assertNotIn("CUDA is unavailable", done.stderr)
        self.assertIn("vocab.json is required", done.stderr)


class SingleProcessBf16Test(unittest.TestCase):
    def test_single_process_cpu_run_with_bf16_configured_still_trains(self):
        """Every train config sets `runtime.precision: bf16`; a single-process CPU run
        trains with it and says what bf16 means there."""
        from src.train.pretrain import build_scheduler

        dev = torch.device("cpu")
        mcfg = tiny_mcfg()
        model = build_model(mcfg)
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
        batches = [synthetic_batch(i, supervised=i != 1) for i in range(4)]
        loader = torch.utils.data.DataLoader(batches, batch_size=None, shuffle=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
        with tempfile.TemporaryDirectory() as td:
            tcfg = tiny_tcfg(td, grad_accum=2)
            self.assertEqual(tcfg["runtime"]["precision"], "bf16")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                trained, manifest = engine.train(
                    model, loader, None, opt, build_scheduler(opt, 2, 1),
                    TrainConfig({}, tcfg, mcfg, total_steps=2), dev, vocab_binding=BINDING)
        self.assertIs(trained, model)   # no DDP wrapper in a single process
        self.assertEqual(manifest.ledger["optimizer_updates"], 2)
        self.assertIn(engine.precision_note(dev), out.getvalue())
        self.assertIn("precision: bf16", out.getvalue())
        for name, value in model.state_dict().items():
            self.assertTrue(bool(torch.isfinite(value).all()), name)
        self.assertFalse(torch.equal(initial["th.mlp.0.weight"],
                                     model.state_dict()["th.mlp.0.weight"]))


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
    """Deterministic but EPOCH-DEPENDENT data — no RNG in __getitem__ (so dropout is the
    only training-time RNG consumer, isolating the resume RNG round-trip), but the token
    content shifts with the epoch. `train()` calls `set_epoch(epoch)`, so restoring the
    WRONG epoch on resume produces different batches and a divergent result — that is
    what makes epoch restoration observable in the resume-equivalence tests (CodeRabbit).
    """

    def __init__(self):
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __len__(self):
        return 6

    def __getitem__(self, i):
        return {"input_ids": torch.arange(i + self._epoch, i + self._epoch + 8) % 100,
                "attention_mask": torch.ones(8)}


def _state_dicts_equal(a: dict, b: dict) -> bool:
    """Tensor-aware deep equality for optimizer / scheduler state dicts."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(_state_dicts_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_state_dicts_equal(x, y) for x, y in zip(a, b))
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    return a == b


class FreshScheduleResumeTest(unittest.TestCase):
    """--fresh-schedule: continuation runs keep the NEW config LR schedule — the
    saved, fully-decayed schedule would pin LR at its tail value forever (the
    3000-step continuation experiment, gem-overnight-log 2026-09-26)."""

    def _cfg(self, ckpt_dir, total_steps):
        return TrainConfig({}, {
            "batch": {"per_gpu": 2, "grad_accum": 1},
            "runtime": {"ckpt_dir": ckpt_dir, "ckpt_every": 3},
            "schedule": {"warmup_steps": 2, "total_steps": 100},
            "optimizer": {"grad_clip": 1.0},
        }, {"compile": False}, total_steps=total_steps)

    def test_fresh_schedule_keeps_new_lr_and_default_pins_old(self):
        from src.train.engine import train

        dev = torch.device("cpu")

        def loader():
            return torch.utils.data.DataLoader(_FixedDS(), batch_size=2, shuffle=False)

        # Run A: 3 steps under a schedule that decays to ~zero (0.5 x 0.01^3 = 5e-7).
        torch.manual_seed(7)
        m_a = _DropoutModel()
        o_a = torch.optim.SGD(m_a.parameters(), lr=0.5, momentum=0.9)
        s_a = torch.optim.lr_scheduler.ExponentialLR(o_a, gamma=0.01)
        with tempfile.TemporaryDirectory() as td:
            train(m_a, loader(), None, o_a, s_a, self._cfg(td, 3), dev, vocab_binding=BINDING)
            ckpts = sorted(Path(td).glob("ckpt_*.pt"))
            self.assertTrue(ckpts, "train() did not checkpoint")
            ckpt = ckpts[-1]

            # fresh_schedule: the new CONSTANT schedule drives LR, not the decayed one.
            torch.manual_seed(99)
            m_b = _DropoutModel()
            o_b = torch.optim.SGD(m_b.parameters(), lr=0.5, momentum=0.9)
            s_b = torch.optim.lr_scheduler.StepLR(o_b, step_size=1000)
            train(m_b, loader(), None, o_b, s_b, self._cfg(td, 6), dev,
                  vocab_binding=BINDING, resume_ckpt=ckpt, fresh_schedule=True)
            self.assertAlmostEqual(o_b.param_groups[0]["lr"], 0.5, places=6)

            # Control: default resume LOADS the decayed schedule — LR stays pinned.
            torch.manual_seed(99)
            m_c = _DropoutModel()
            o_c = torch.optim.SGD(m_c.parameters(), lr=0.5, momentum=0.9)
            s_c = torch.optim.lr_scheduler.StepLR(o_c, step_size=1000)
            train(m_c, loader(), None, o_c, s_c, self._cfg(td, 6), dev,
                  vocab_binding=BINDING, resume_ckpt=ckpt)
            self.assertLess(o_c.param_groups[0]["lr"], 1e-4)


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
                dl.dataset.set_epoch(ep)  # epoch-dependent data (mirrors train())
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
                dl.dataset.set_epoch(ep)
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
                dl.dataset.set_epoch(ep)
                _train_one_epoch(model_b, dl, opt_b, sched_b, ep, cfg, dev, rank=0)
            resumed = self._params(model_b)
            resumed_opt, resumed_sched = opt_b.state_dict(), sched_b.state_dict()
        straight_opt, straight_sched = opt_s.state_dict(), sched_s.state_dict()

        for a, b in zip(straight, resumed, strict=True):  # strict: lengths must match
            self.assertTrue(torch.equal(a, b),
                            f"resume diverged from straight-through: max|d|={ (a-b).abs().max() }")
        self.assertTrue(_state_dicts_equal(straight_opt, resumed_opt),
                        "optimizer state diverged after resume")
        self.assertTrue(_state_dicts_equal(straight_sched, resumed_sched),
                        "scheduler state diverged after resume")

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
            train(m_s, loader(), None, o_s, s_s, cfg(td, 6, 999), dev, seed=7,
                  vocab_binding=BINDING)
            straight = [p.detach().clone() for p in m_s.parameters()]
            straight_opt, straight_sched = o_s.state_dict(), s_s.state_dict()

        torch.manual_seed(7)
        m_a, o_a, s_a = build()
        with tempfile.TemporaryDirectory() as td:
            train(m_a, loader(), None, o_a, s_a, cfg(td, 3, 3), dev, seed=7,
                  vocab_binding=BINDING)  # 1 epoch, saves at step 3
            ckpts = list(Path(td).glob("ckpt_*.pt"))
            self.assertTrue(ckpts, "train() did not write an epoch-boundary checkpoint")
            ckpt = ckpts[0]

            torch.manual_seed(999_999)  # perturb: a missing RNG restore would diverge
            m_b, o_b, s_b = build()
            train(m_b, loader(), None, o_b, s_b, cfg(td, 6, 999), dev,
                  vocab_binding=BINDING, resume_ckpt=ckpt, seed=7)
            resumed = [p.detach().clone() for p in m_b.parameters()]
            resumed_opt, resumed_sched = o_b.state_dict(), s_b.state_dict()

        for a, b in zip(straight, resumed, strict=True):  # strict: lengths must match
            self.assertTrue(torch.equal(a, b),
                            f"train() resume diverged: max|d|={ (a-b).abs().max() }")
        self.assertTrue(_state_dicts_equal(straight_opt, resumed_opt),
                        "optimizer state diverged after train() resume")
        self.assertTrue(_state_dicts_equal(straight_sched, resumed_sched),
                        "scheduler state diverged after train() resume")

    def test_production_train_records_validation_loss(self):
        from src.train.engine import train

        dev = torch.device("cpu")
        model = _DropoutModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        loader = torch.utils.data.DataLoader(_FixedDS(), batch_size=2, shuffle=False)

        with tempfile.TemporaryDirectory() as directory:
            config = TrainConfig({}, {
                "batch": {"per_gpu": 2, "grad_accum": 1},
                "runtime": {"ckpt_dir": directory, "ckpt_every": 999},
                "schedule": {"warmup_steps": 0, "total_steps": 1},
                "eval_schedule": {"val_every": 1},
                "optimizer": {"grad_clip": 1.0},
            }, {"compile": False}, total_steps=1)
            _, manifest = train(
                model,
                loader,
                loader,
                optimizer,
                scheduler,
                config,
                dev,
                seed=7,
                vocab_binding=BINDING,
            )

        self.assertEqual(len(manifest.validation), 1)
        self.assertTrue(math.isfinite(manifest.validation[0]["val_loss"]))

    def test_train_requires_a_vocabulary_binding(self):
        """No silent skip: a run (and so its checkpoints) must be bound to a vocabulary."""
        from src.train.engine import train

        model = _DropoutModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        loader = torch.utils.data.DataLoader(_FixedDS(), batch_size=2, shuffle=False)
        with tempfile.TemporaryDirectory() as directory:
            config = TrainConfig({}, {
                "batch": {"per_gpu": 2, "grad_accum": 1},
                "runtime": {"ckpt_dir": directory, "ckpt_every": 999},
                "schedule": {"warmup_steps": 0, "total_steps": 1},
                "optimizer": {"grad_clip": 1.0},
            }, {"compile": False}, total_steps=1)
            args = (model, loader, None, optimizer, scheduler, config, torch.device("cpu"))
            with self.assertRaises(TypeError):
                train(*args)  # vocab_binding is a required keyword
            with self.assertRaisesRegex(ValueError, "requires vocab_binding"):
                train(*args, vocab_binding=None)



class ScheduleResolutionTest(unittest.TestCase):
    """`engine.resolve_schedule`: warm-up and checkpoint interval scale with the run;
    absolute values still work; a schedule that cannot do its job is refused."""

    def _tcfg(self, *, warmup_steps=None, warmup_frac=None, ckpt_every=None,
              checkpoints_per_run=None, final_checkpoint=None):
        schedule = {"warmup_steps": warmup_steps}
        if warmup_frac is not None:
            schedule["warmup_frac"] = warmup_frac
        runtime = {"ckpt_every": ckpt_every}
        if checkpoints_per_run is not None:
            runtime["checkpoints_per_run"] = checkpoints_per_run
        if final_checkpoint is not None:
            runtime["final_checkpoint"] = final_checkpoint
        return {"schedule": schedule, "runtime": runtime}

    def test_warmup_frac_resolves_to_a_share_of_the_run(self):
        plan = engine.resolve_schedule(self._tcfg(warmup_frac=0.05), 300)
        self.assertEqual((plan.warmup_steps, plan.warmup_source), (15, "warmup_frac"))
        # Default share when the key is absent.
        self.assertEqual(engine.resolve_schedule(self._tcfg(), 200).warmup_steps,
                         int(200 * engine.DEFAULT_WARMUP_FRAC))
        # A few-update screening run still gets one warm-up update, never the whole run.
        self.assertEqual(engine.resolve_schedule(self._tcfg(warmup_frac=0.05), 13).warmup_steps, 1)
        self.assertEqual(engine.resolve_schedule(self._tcfg(warmup_frac=0.05), 1).warmup_steps, 0)
        # An absolute warmup_steps wins.
        plan = engine.resolve_schedule(self._tcfg(warmup_steps=7, warmup_frac=0.5), 300)
        self.assertEqual((plan.warmup_steps, plan.warmup_source), (7, "warmup_steps"))

    def test_warmup_as_long_as_the_run_is_refused(self):
        # The pre-fix configs/train.yaml (warmup_steps 2000) on a 13-update sample pass.
        with self.assertRaisesRegex(engine.ScheduleError, "never leave the linear warm-up"):
            engine.resolve_schedule(self._tcfg(warmup_steps=2000), 13)
        with self.assertRaisesRegex(engine.ScheduleError, "never leave"):
            engine.resolve_schedule(self._tcfg(warmup_steps=13), 13)
        with self.assertRaisesRegex(engine.ScheduleError, r"warmup_frac must be in \[0, 1\)"):
            engine.resolve_schedule(self._tcfg(warmup_frac=1.0), 13)
        # Unvalidated (TrainConfig) it resolves as given.
        self.assertEqual(engine.resolve_schedule(self._tcfg(warmup_steps=2000), 13,
                                                 validate=False).warmup_steps, 2000)

    def test_checkpoint_interval_scales_and_guarantees_periodic_checkpoints(self):
        for total in (1, 4, 13, 15, 287, 60000):
            plan = engine.resolve_schedule(self._tcfg(checkpoints_per_run=5), total)
            with self.subTest(total=total):
                self.assertEqual(plan.ckpt_source, "checkpoints_per_run")
                self.assertGreaterEqual(plan.periodic_checkpoints, min(5, total))
                self.assertTrue(plan.final_checkpoint)
        self.assertEqual(engine.resolve_schedule(self._tcfg(), 13).ckpt_every,
                         13 // engine.DEFAULT_CHECKPOINTS_PER_RUN)
        plan = engine.resolve_schedule(self._tcfg(ckpt_every=4), 13)
        self.assertEqual((plan.ckpt_every, plan.ckpt_source, plan.periodic_checkpoints),
                         (4, "ckpt_every", 3))

    def test_a_config_that_writes_no_checkpoint_is_refused(self):
        # Interval longer than the run: the final checkpoint is the only one (allowed) ...
        plan = engine.resolve_schedule(self._tcfg(ckpt_every=2000), 13)
        self.assertEqual(plan.periodic_checkpoints, 0)
        # ... and with the final checkpoint off there is none: refused.
        with self.assertRaisesRegex(engine.ScheduleError, "no checkpoint would be written"):
            engine.resolve_schedule(self._tcfg(ckpt_every=2000, final_checkpoint=False), 13)
        with self.assertRaisesRegex(engine.ScheduleError, "ckpt_every must be >= 1"):
            engine.resolve_schedule(self._tcfg(ckpt_every=0), 13)
        with self.assertRaisesRegex(engine.ScheduleError, "checkpoints_per_run must be >= 1"):
            engine.resolve_schedule(self._tcfg(checkpoints_per_run=0), 13)

    def test_shipped_configs_resolve_for_the_matrix_budgets(self):
        """configs/train.yaml on a 13-update sample pass (0.05 pass -> 1 update) and full
        MIMIC (~180-300 updates per pass): every run leaves warm-up and checkpoints."""
        import yaml

        from src.train.engine import resolve_total_steps

        tcfg = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
        for batches_per_rank in (13 * 32, 180 * 32, 300 * 32):
            for passes in (0.05, 1.0):
                cfg = {**tcfg, "schedule": {**tcfg["schedule"], "passes": passes}}
                total = resolve_total_steps(cfg, batches_per_rank)
                plan = engine.resolve_schedule(cfg, total)
                with self.subTest(batches=batches_per_rank, passes=passes):
                    self.assertTrue(plan.warmup_steps < total or plan.warmup_steps == 0)
                    self.assertGreaterEqual(plan.periodic_checkpoints, 1)
        for name in ("train.smoke.yaml", "train.mps.yaml"):
            cfg = yaml.safe_load((ROOT / "configs" / name).read_text())
            engine.resolve_schedule(cfg, int(cfg["schedule"]["total_steps"]))

    def test_lr_leaves_warmup_on_a_short_run(self):
        """The fixed warmup_steps 2000 pinned a 13-update run in linear warm-up; the
        resolved warm-up reaches the peak LR and then decays."""
        from src.train.pretrain import build_scheduler

        for total in (1, 2, 13):
            plan = engine.resolve_schedule({"schedule": {"warmup_frac": 0.05},
                                            "runtime": {}}, total)
            opt = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)
            sched = build_scheduler(opt, total, plan.warmup_steps)
            lrs = []
            for _ in range(total):
                lrs.append(opt.param_groups[0]["lr"])
                opt.step()
                sched.step()
            with self.subTest(total=total):
                self.assertAlmostEqual(max(lrs), 0.1, places=6)


def _short_cfg(ckpt_dir, total_steps, ckpt_every=None):
    return TrainConfig({}, {
        "batch": {"per_gpu": 2, "grad_accum": 1},
        "runtime": {"ckpt_dir": ckpt_dir, "ckpt_every": ckpt_every},
        "schedule": {"warmup_steps": None, "total_steps": total_steps},
        "eval_schedule": {"val_every": 10_000},
        "optimizer": {"grad_clip": 1.0},
    }, {"compile": False}, total_steps=total_steps)


class FinalCheckpointTest(unittest.TestCase):
    """A run shorter than ckpt_every still ends with a checkpoint that loads and resumes
    (6 samples, batch 2: 3 updates per pass)."""

    def _build(self):
        torch.manual_seed(20260829)
        m = _DropoutModel()
        o = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        s = torch.optim.lr_scheduler.StepLR(o, step_size=2, gamma=0.9)
        return m, o, s

    @staticmethod
    def _loader():
        return torch.utils.data.DataLoader(_FixedDS(), batch_size=2, shuffle=False)

    def test_run_shorter_than_ckpt_every_writes_a_final_checkpoint_that_resumes(self):
        from src.train.engine import latest_checkpoint, train

        dev = torch.device("cpu")
        torch.manual_seed(7)
        m_s, o_s, s_s = self._build()
        with tempfile.TemporaryDirectory() as td:
            train(m_s, self._loader(), None, o_s, s_s, _short_cfg(td, 6, 2000), dev, seed=7,
                  vocab_binding=BINDING)
            straight = [p.detach().clone() for p in m_s.parameters()]
            self.assertEqual([p.name for p in Path(td).glob("*.pt")], ["ckpt_ep2_step6.pt"])

        torch.manual_seed(7)
        m_a, o_a, s_a = self._build()
        with tempfile.TemporaryDirectory() as td:
            # 3 updates (one full pass) with ckpt_every 2000: only the final checkpoint.
            train(m_a, self._loader(), None, o_a, s_a, _short_cfg(td, 3, 2000), dev, seed=7,
                  vocab_binding=BINDING)
            self.assertEqual(sorted(p.name for p in Path(td).glob("*.pt")),
                             ["ckpt_ep1_step3.pt"])
            ckpt = latest_checkpoint(td)
            loaded = load_checkpoint(ckpt)
            self.assertEqual((loaded["epoch"], loaded["step"]), (1, 3))
            self.assertEqual(loaded["vocab_binding"], BINDING)
            self.assertEqual(loaded["manifest"]["ledger"]["optimizer_updates"], 3)

            torch.manual_seed(999_999)
            m_b, o_b, s_b = self._build()
            train(m_b, self._loader(), None, o_b, s_b, _short_cfg(td, 6, 2000), dev,
                  vocab_binding=BINDING, resume_ckpt=ckpt, seed=7)
            resumed = [p.detach().clone() for p in m_b.parameters()]
            self.assertEqual(latest_checkpoint(td).name, "ckpt_ep2_step6.pt")
        for a, b in zip(straight, resumed, strict=True):
            self.assertTrue(torch.equal(a, b), "resume from the final checkpoint diverged")

    def test_final_checkpoint_mid_pass_and_no_duplicate(self):
        from src.train.engine import train

        dev = torch.device("cpu")
        with tempfile.TemporaryDirectory() as td:
            m, o, s = self._build()
            # 4 updates: one pass + 1 update of the next (mid-pass: epoch 1 consumed).
            train(m, self._loader(), None, o, s, _short_cfg(td, 4, 2000), dev,
                  vocab_binding=BINDING)
            self.assertEqual(sorted(p.name for p in Path(td).glob("*.pt")),
                             ["ckpt_ep1_step4.pt"])
        with tempfile.TemporaryDirectory() as td:
            m, o, s = self._build()
            # Scaled interval (null ckpt_every): 6 // 5 = every update, the final save is
            # the periodic one (not written twice).
            train(m, self._loader(), None, o, s, _short_cfg(td, 6), dev, vocab_binding=BINDING)
            names = sorted(p.name for p in Path(td).glob("*.pt"))
            self.assertEqual(len(names), 6)
            self.assertIn("ckpt_ep2_step6.pt", names)
        with tempfile.TemporaryDirectory() as td:
            m, o, s = self._build()
            cfg = _short_cfg(td, 3, 2000)
            cfg.final_checkpoint = False
            train(m, self._loader(), None, o, s, cfg, dev, vocab_binding=BINDING)
            self.assertEqual(list(Path(td).glob("*.pt")), [])

    def test_resume_of_a_finished_run_trains_nothing_and_writes_nothing(self):
        from src.train.engine import train

        dev = torch.device("cpu")
        with tempfile.TemporaryDirectory() as td:
            m, o, s = self._build()
            train(m, self._loader(), None, o, s, _short_cfg(td, 3, 2000), dev,
                  vocab_binding=BINDING)
            ckpt = Path(td) / "ckpt_ep1_step3.pt"
            mtime = ckpt.stat().st_mtime_ns
            m2, o2, s2 = self._build()
            _, manifest = train(m2, self._loader(), None, o2, s2, _short_cfg(td, 3, 2000), dev,
                                vocab_binding=BINDING, resume_ckpt=ckpt)
            self.assertEqual(manifest.ledger["optimizer_updates"], 3)
            self.assertEqual(sorted(p.name for p in Path(td).glob("*.pt")),
                             ["ckpt_ep1_step3.pt"])
            self.assertEqual(ckpt.stat().st_mtime_ns, mtime)

    def test_resume_refuses_a_checkpoint_of_another_run(self):
        from src.train.engine import train

        dev = torch.device("cpu")
        with tempfile.TemporaryDirectory() as td:
            m, o, s = self._build()
            cfg = _short_cfg(td, 3, 2000)
            cfg.tokenization_arm = "clinical_soft"
            train(m, self._loader(), None, o, s, cfg, dev, vocab_binding=BINDING, seed=1)
            ckpt = Path(td) / "ckpt_ep1_step3.pt"
            for attr, value, seed in (("tokenization_arm", "global_deciles", 1),
                                      ("tokenization_arm", "clinical_soft", 2)):
                m2, o2, s2 = self._build()
                other = _short_cfg(td, 6, 2000)
                setattr(other, attr, value)
                with self.subTest(attr=attr, value=value, seed=seed), \
                        self.assertRaisesRegex(ValueError, "different run"):
                    train(m2, self._loader(), None, o2, s2, other, dev, vocab_binding=BINDING,
                          resume_ckpt=ckpt, seed=seed,
                          resume_match=("tokenization_arm", "seed"))


class _SigtermAt(_FixedDS):
    def __init__(self, at):
        super().__init__()
        self.at, self.sent = at, False

    def __getitem__(self, i):
        if i == self.at and not self.sent:
            import signal

            self.sent = True
            os.kill(os.getpid(), signal.SIGTERM)
        return super().__getitem__(i)


class CleanStopTest(unittest.TestCase):
    def test_sigterm_checkpoints_the_current_update_and_returns(self):
        import signal

        from src.train.engine import latest_checkpoint, train

        before = signal.getsignal(signal.SIGTERM)
        torch.manual_seed(0)
        model = _DropoutModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000)
        # Sample 3 is in the second batch (batch 2): the stop lands after update 2.
        loader = torch.utils.data.DataLoader(_SigtermAt(3), batch_size=2, shuffle=False)
        with tempfile.TemporaryDirectory() as td:
            _, manifest = train(model, loader, None, opt, sched, _short_cfg(td, 100, 2000),
                                torch.device("cpu"), vocab_binding=BINDING)
            self.assertEqual(manifest.ledger["optimizer_updates"], 2)
            self.assertEqual(sorted(p.name for p in Path(td).glob("*.pt")),
                             ["ckpt_ep0_step2.pt"])
            self.assertEqual(load_checkpoint(latest_checkpoint(td))["step"], 2)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)


class ResumeArgumentTest(unittest.TestCase):
    def test_latest_picks_the_most_updates_and_refuses_an_empty_directory(self):
        from src.train.engine import latest_checkpoint, resolve_resume

        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(latest_checkpoint(td))
            with self.assertRaisesRegex(SystemExit, "no checkpoint"):
                resolve_resume("latest", td)
            for name in ("ckpt_ep0_step2.pt", "ckpt_ep1_step10.pt", "ckpt_ep1_step9.pt",
                         "ckpt_tmpabc.pt", "notes.txt"):
                (Path(td) / name).write_bytes(b"")
            self.assertEqual(latest_checkpoint(td).name, "ckpt_ep1_step10.pt")
            self.assertEqual(resolve_resume("latest", td).name, "ckpt_ep1_step10.pt")
            self.assertEqual(resolve_resume(Path(td) / "ckpt_ep0_step2.pt", td).name,
                             "ckpt_ep0_step2.pt")
            with self.assertRaisesRegex(SystemExit, "no such checkpoint"):
                resolve_resume(Path(td) / "missing.pt", td)
            self.assertIsNone(resolve_resume(None, td))


if __name__ == "__main__":
    unittest.main()
