"""MPS smoke test: forward-pass all ablation arms on Apple Silicon.

Skips the slow full-stay tokenization (verified by unit tests) and
constructs realistic synthetic batches to test model integrity on MPS.
"""

import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

MODEL_CFG = Path("configs/model.yaml")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class MPSSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mcfg = yaml.safe_load(MODEL_CFG.read_text())
        cls.n_targets = 10
        cls.vocab_size = 500
        cls.batch_size = 8
        cls.seq_len = 128
        cls.dev = torch.device(DEVICE)

        cls._make_batch()
        print(f"MPS smoke: device={DEVICE}, batch={cls.batch['token'].shape}")

    @classmethod
    def _make_batch(cls):
        bsz, slen, v = cls.batch_size, cls.seq_len, cls.vocab_size
        nt = cls.n_targets

        token = torch.randint(1, v, (bsz, slen))
        soft_token = torch.randint(1, v, (bsz, slen, 3))
        soft_weight = torch.rand(bsz, slen, 3)
        soft_weight = soft_weight / soft_weight.sum(-1, keepdim=True)

        cls.batch = {
            "token": token,
            "soft_token": soft_token,
            "soft_weight": soft_weight,
            "pos_min": torch.randint(0, 1440, (bsz, slen)),
            "value": torch.randn(bsz, slen),
            "val_mask": (torch.rand(bsz, slen) > 0.3).float(),
            "last_idx": torch.full((bsz,), slen - 1),
            "cr_type": torch.randint(0, nt, (bsz,)),
            "cr_bin": torch.randint(0, 16, (bsz,)),
            "th_target": torch.randint(0, nt, (bsz,)),
            "th_tau": torch.randint(1, 10, (bsz,)),
            "th_dir": torch.randint(0, 2, (bsz,)),
            "th_crossed": torch.where(
                torch.rand(bsz) > 0.7,
                torch.randint(1, 48, (bsz,)),
                -torch.ones(bsz, dtype=torch.long),
            ),
        }

    # ----------------------------------------------------------------- arms
    def test_01_from_scratch_untied(self):
        """CLIFEncoder random init, untied embeddings, full ORA loss."""
        from src.model.encoder import CLIFEncoder, count_params
        from src.model.heads import CompetingRiskHead, ThresholdHazardHead, ValueRegressionHead, next_event_loss

        cfg = dict(self.mcfg)
        cfg["trunk"]["tied_embeddings"] = False

        enc = CLIFEncoder(self.vocab_size, cfg).to(self.dev)
        d = enc.d_model
        cr = CompetingRiskHead(d, self.n_targets, 16).to(self.dev)
        th = ThresholdHazardHead(d, self.n_targets, 48, n_value_bins=10).to(self.dev)
        vr = ValueRegressionHead(d, self.vocab_size).to(self.dev)

        params = count_params(enc) + sum(p.numel() for p in cr.parameters()) + \
                 sum(p.numel() for p in th.parameters()) + sum(p.numel() for p in vr.parameters())
        print(f"  from_scratch_untied: {params/1e6:.1f}M params on {DEVICE}")

        b = {k: v.to(self.dev) for k, v in self.batch.items()}
        H = enc(b["soft_token"], b["pos_min"], b["soft_weight"])
        h_last = H[torch.arange(H.size(0)), b["last_idx"]]

        la = next_event_loss(enc.lm_logits(H), b["token"])
        lb = cr.loss(h_last, b["cr_type"], b["cr_bin"])
        lc = th.loss(h_last, b["th_target"], b["th_tau"], b["th_dir"], b["th_crossed"])
        ld = vr.loss(H, b["token"], b["value"], b["val_mask"])

        total = 0.2 * la + 1.0 * lb + 1.0 * lc + 0.5 * ld
        total.backward()

        print(f"  losses: ntp={la:.4f} cr={lb:.4f} th={lc:.4f} val={ld:.4f} total={total:.4f}")

        grads = sum(p.grad is not None and p.grad.abs().sum() > 0
                    for p in enc.parameters())
        self.assertGreater(grads, 0, "no gradients flowed in CLIFEncoder")
        print(f"  PASS ({grads} encoder params with gradients)")

    def test_02_tied_encoder_ablation(self):
        """CLIFEncoder with tied embeddings (ablation arm)."""
        from src.model.encoder import CLIFEncoder

        cfg = dict(self.mcfg)
        cfg["trunk"]["tied_embeddings"] = True
        cfg["trunk"]["d_model"] = 32

        enc = CLIFEncoder(20, cfg).to(self.dev)

        token = torch.randint(1, 20, (4, 8)).to(self.dev)
        pos = torch.zeros(4, 8, dtype=torch.long).to(self.dev)
        H = enc(token, pos)
        logits = enc.lm_logits(H)
        self.assertEqual(logits.shape, (4, 8, 20))

        tied = enc.lm_head.projection.weight.data_ptr() == enc.tok_emb.weight.data_ptr()
        print(f"  tied_embeddings=True → weights tied: {tied}")
        self.assertTrue(tied)
        print("  PASS")

    def test_03_frozen_encoder_taskhead(self):
        """Frozen CLIFEncoder + TaskHead baseline (no-pretrain)."""
        from src.model.encoder import CLIFEncoder, count_params
        from src.model.heads import TaskHead

        cfg = dict(self.mcfg)
        cfg["trunk"]["d_model"] = 64
        cfg["trunk"]["n_layers"] = 2
        cfg["trunk"]["n_heads"] = 4
        cfg["trunk"]["tied_embeddings"] = False

        enc = CLIFEncoder(self.vocab_size, cfg).to(self.dev)
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()

        head = TaskHead(cfg["trunk"]["d_model"], 7).to(self.dev)

        token = torch.randint(1, self.vocab_size, (4, 64)).to(self.dev)
        pos = torch.randint(0, 1440, (4, 64)).to(self.dev)
        with torch.no_grad():
            H = enc(token, pos)
        h_last = H[torch.arange(H.size(0)), torch.tensor([63, 63, 63, 63])]

        logits = head(h_last)
        labels = torch.randint(0, 2, (4, 7)).to(self.dev)
        mask = torch.ones_like(labels)
        loss = head.loss(h_last, labels, mask)
        loss.backward()

        self.assertEqual(logits.shape, (4, 7))
        self.assertGreater(loss.item(), 0)
        print(f"  TaskHead: params={count_params(head):,}, loss={loss:.4f}")
        print("  PASS")

    def test_04_hard_token_fallback(self):
        """Encoder accepts hard [B,T] tokens without weights."""
        from src.model.encoder import CLIFEncoder

        cfg = dict(self.mcfg)
        cfg["trunk"]["d_model"] = 32
        cfg["trunk"]["tied_embeddings"] = False

        enc = CLIFEncoder(50, cfg).to(self.dev)
        token = torch.randint(1, 50, (2, 16)).to(self.dev)
        pos = torch.randint(0, 1000, (2, 16)).to(self.dev)
        H = enc(token, pos)
        self.assertEqual(H.shape, (2, 16, 32))
        print(f"  hard-token forward: {H.shape}")
        print("  PASS")


if __name__ == "__main__":
    unittest.main()