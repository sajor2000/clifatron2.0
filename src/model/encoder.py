"""Flat Llama-style causal decoder over FUSED code=value tokens (~30M params).

REVISED per Lee et al. 2026 (arXiv:2604.16775), the CLIF-native tokenization
ablation (28 matched decoders on MIMIC-IV-Ext-CLIF):
  - FUSED single token per (concept, value-bin) — biggest win (mortality 0.891->0.915).
    The old split (concept-token + value-token) and the dual-level intra-event pool are
    retired; a fused token makes a flat sequence sufficient.
  - admission-relative RoPE at 1-min-resolution position ids  >=  inserted time tokens,
    and ~11% shorter sequences (replaces the continuous-time Delta-t ALiBi bias).
  - context 4096 tokens covers >99.95% of first-24h stays.

Backbone: Llama-style — RMSNorm, RoPE, SwiGLU, tied embeddings, causal SDPA.
Returns per-token hidden states H_t; heads (see heads.py) consume the state at the
anchor/last position (ICareFM-style per-step patient state).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return n * self.weight


def build_rope_cache(pos: torch.Tensor, head_dim: int, base: float = 10000.0):
    """RoPE cos/sin from explicit position ids (admission-relative, 1-min resolution).

    pos: [B, T] integer minutes-since-admission per token (NOT sequence index).
    Returns cos, sin each [B, T, head_dim].
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=pos.device).float() / half))
    ang = pos.float()[..., None] * inv_freq[None, None, :]      # [B, T, half]
    ang = torch.cat([ang, ang], dim=-1)                          # [B, T, head_dim]
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, nh, T, hd]; cos/sin: [B, T, hd]
    cos = cos[:, None]
    sin = sin[:, None]
    half = x.shape[-1] // 2
    x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + x_rot * sin


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.hd = d // n_heads
        self.ln1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = RMSNorm(d)
        hidden = ffn_mult * d
        self.w_gate = nn.Linear(d, hidden, bias=False)   # SwiGLU
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos, sin) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(o))
        g = self.ln2(x)
        x = x + self.w_down(F.silu(self.w_gate(g)) * self.w_up(g))
        return x


class CLIFEncoder(nn.Module):
    """Flat causal decoder over fused tokens. `token` is a single id per event that
    already encodes (concept, value-bin); `pos_min` is minutes-since-admission."""

    def __init__(self, vocab_size: int, cfg: dict):
        super().__init__()
        xe = cfg["trunk"]
        d = xe["d_model"]
        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.n_heads = xe["n_heads"]
        self.head_dim = d // xe["n_heads"]
        self.blocks = nn.ModuleList(
            Block(d, xe["n_heads"], xe["ffn_mult"], xe["dropout"]) for _ in range(xe["n_layers"])
        )
        self.ln_f = RMSNorm(d)
        self.d_model = d
        self.rope_base = xe.get("rope_base", 10000.0)

    def forward(self, token, pos_min) -> torch.Tensor:
        """token: [B, T] fused ids (0=pad). pos_min: [B, T] minutes since admission."""
        x = self.tok_emb(token)                                  # [B, T, d]
        cos, sin = build_rope_cache(pos_min, self.head_dim, self.rope_base)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.ln_f(x)                                      # per-token states H_t

    def lm_logits(self, H: torch.Tensor) -> torch.Tensor:
        """Tied-embedding next-token logits."""
        return F.linear(H, self.tok_emb.weight)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
