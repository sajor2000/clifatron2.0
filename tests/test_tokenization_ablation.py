"""Tokenization ablation smoke tests on MPS.

Forward-pass each of the 5 tokenization arms through a frozen encoder.
Tests that the continuous-fused and textcode code paths produce valid
tensors without import/crash errors.
"""

import unittest

import numpy as np
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
B, T = 4, 32
VOCAB = 200
D_MODEL = 64


class TokenizationSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = torch.device(DEVICE)
        cls.pos = torch.randint(0, 1440, (B, T)).to(cls.dev)
        cls.token = torch.randint(1, VOCAB, (B, T)).to(cls.dev)
        cls.soft_token = torch.randint(1, VOCAB, (B, T, 3)).to(cls.dev)
        cls.soft_weight = torch.rand(B, T, 3).to(cls.dev)
        cls.soft_weight = cls.soft_weight / cls.soft_weight.sum(-1, keepdim=True)
        cls.value = torch.randn(B, T).to(cls.dev)

    def _mini_cfg(self, tied=False):
        return {
            "trunk": {
                "d_model": D_MODEL,
                "n_layers": 1,
                "n_heads": 2,
                "ffn_mult": 2,
                "dropout": 0.0,
                "tied_embeddings": tied,
            }
        }

    def test_01_discrete_hard_tokens(self):
        """Standard CLIFEncoder with hard [B,T] tokens (clinical bins)."""
        from src.model.encoder import CLIFEncoder

        enc = CLIFEncoder(VOCAB, self._mini_cfg()).to(self.dev)
        H = enc(self.token, self.pos)
        logits = enc.lm_logits(H)
        self.assertEqual(H.shape, (B, T, D_MODEL))
        self.assertEqual(logits.shape, (B, T, VOCAB))
        print(f"  discrete hard: H={list(H.shape)} logits={list(logits.shape)}")

    def test_02_discrete_soft_tokens(self):
        """CLIFEncoder with soft [B,T,K] weighted tokens (deciles+soft)."""
        from src.model.encoder import CLIFEncoder

        enc = CLIFEncoder(VOCAB, self._mini_cfg()).to(self.dev)
        H = enc(self.soft_token, self.pos, self.soft_weight)
        self.assertEqual(H.shape, (B, T, D_MODEL))
        print(f"  discrete soft: H={list(H.shape)}")

    def test_03_continuous_fused_forward(self):
        """ContinuousFusedEncoder with concept tokens + value channel."""
        from src.model.encoder_continuous import ContinuousFusedEncoder

        enc = ContinuousFusedEncoder(VOCAB, self._mini_cfg()).to(self.dev)
        H = enc(self.token, self.pos, continuous_value=self.value)
        logits = enc.lm_logits(H)
        self.assertEqual(H.shape, (B, T, D_MODEL))
        self.assertEqual(logits.shape, (B, T, VOCAB))

        grad = torch.sum(H)
        grad.backward()
        grads = sum(p.grad is not None and p.grad.abs().sum() > 0
                    for p in enc.parameters())
        self.assertGreater(grads, 0)
        print(f"  continuous_fused: H={list(H.shape)} grads={grads}")

    def test_04_textcode_embedding_shape(self):
        """TextCode returns correct [B,T,token_dim] shape."""
        from src.data.tokenize_textcode import textcode_embedding

        model_dim = 128
        projection = torch.nn.Linear(model_dim, D_MODEL, bias=False)
        cached = np.random.randn(VOCAB, model_dim).astype(np.float32)

        tokens = torch.randint(0, VOCAB, (2, 16))
        output = textcode_embedding(tokens, cached, projection)
        self.assertEqual(output.shape, (2, 16, D_MODEL))
        print(f"  textcode: {list(output.shape)}")

    def test_05_all_five_arms_dry_run(self):
        """Each arm in tokenization_ablation.yaml loads without error."""
        import yaml
        from src.train.run_tokenization_ablation import TokenizationAblationModel

        abl = yaml.safe_load(
            __import__("pathlib").Path("configs/tokenization_ablation.yaml").read_text()
        )
        mcfg = {
            "trunk": {
                "d_model": 32, "n_layers": 1, "n_heads": 2,
                "ffn_mult": 2, "dropout": 0.0, "tied_embeddings": False,
            },
            "heads": {
                "next_event": {"enabled": True, "weight": 0.2},
                "competing_risk": {"enabled": True, "weight": 1.0, "n_time_bins": 8},
                "threshold_hazard": {
                    "enabled": True, "weight": 1.0,
                    "horizon_hours": 48, "n_time_bins": 24,
                    "threshold_embed_dim": 16,
                },
                "value_regression": {"enabled": True, "weight": 0.5},
            },
        }

        for arm_name in abl["arms"]:
            model = TokenizationAblationModel(50, 5, mcfg, abl["arms"][arm_name])
            total = sum(p.numel() for p in model.parameters())
            print(f"  {arm_name}: {total:,} params, tokenizer={abl['arms'][arm_name]['tokenizer']}")
            self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main()