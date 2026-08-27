"""Continuous-fused encoder variant (McCann 2026).

Replaces the standard token embedding with concept-only embedding +
learned continuous value projection. One token per event (no bin expansion),
giving -34% sequence length vs discrete binning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.encoder import CLIFEncoder as BaseEncoder, build_rope_cache


class ContinuousFusedEncoder(BaseEncoder):
    """CLIFEncoder with continuous-fused value channel.

    Concept tokens are discrete event IDs; values are continuous scalars
    projected through a learned weight matrix and summed with concept embeddings.
    This is the McCann 2026 architecture.
    """

    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__(vocab_size, cfg)
        d = self.d_model
        self.value_proj = nn.Linear(1, d, bias=False)

    def forward(self, token, pos_min, token_weight=None,
                continuous_value=None) -> torch.Tensor:
        if token.ndim != 2:
            raise ValueError("continuous-fused encoder expects 2D [B,T] concept tokens")

        x = self.tok_emb(token)

        if continuous_value is not None:
            x = x + self.value_proj(continuous_value.unsqueeze(-1))

        cos, sin = build_rope_cache(pos_min, self.head_dim, self.rope_base)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.ln_f(x)