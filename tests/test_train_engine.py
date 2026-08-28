import os
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from src.train.engine import setup_ddp, is_distributed, _prepare_batch, _train_one_epoch, TrainConfig
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


if __name__ == "__main__":
    unittest.main()
