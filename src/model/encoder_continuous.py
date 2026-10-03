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

    Concept tokens are discrete (edgeless) event IDs; the event's normalized current
    value and its presence mask are projected through a learned ``Linear(2, d)`` and
    summed with the concept embedding (McCann 2026). A masked (missing / categorical /
    NaN) value contributes exactly zero, so NaN never reaches the trunk.
    """

    uses_value_channel = True

    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__(vocab_size, cfg)
        d = self.d_model
        self.value_proj = nn.Linear(2, d, bias=False)   # [value, present] -> d

    def forward(self, token, pos_min, token_weight=None,
                continuous_value=None, continuous_value_mask=None) -> torch.Tensor:
        if token.ndim != 2:
            raise ValueError("continuous-fused encoder expects 2D [B,T] concept tokens")

        x = self.embed_tokens(token)

        if continuous_value is not None:
            present = torch.isfinite(continuous_value)
            if continuous_value_mask is not None:
                present = present & continuous_value_mask.bool()
            value = torch.where(present, continuous_value,
                                torch.zeros_like(continuous_value))
            feats = torch.stack([value, present.to(value.dtype)], dim=-1)
            x = x + self.value_proj(feats.to(self.value_proj.weight.dtype))

        cos, sin = build_rope_cache(pos_min, self.head_dim, self.rope_base)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.ln_f(x)
