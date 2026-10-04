"""Flat Qwen2-arch (Llama-style) causal decoder over FUSED code=value tokens (~30M params).

REVISED per Lee et al. 2026 (arXiv:2604.16775), the CLIF-native tokenization
ablation (28 matched decoders on MIMIC-IV-Ext-CLIF):
  - FUSED single token per (concept, value-bin) — biggest win (mortality 0.891->0.915).
    The old split (concept-token + value-token) and the dual-level intra-event pool are
    retired; a fused token makes a flat sequence sufficient.
  - ICU-admission-relative RoPE at 1-min-resolution position ids  >=  inserted time tokens,
    and ~11% shorter sequences (replaces the continuous-time Delta-t ALiBi bias).
  - context: 8192 tokens (configs/model.yaml max_tokens; 4096 = Lee-tokenizer ablation arm).

Backbone: Qwen2-arch / Llama-style — pre-norm RMSNorm, RoPE, SwiGLU, full multi-head
attention (no GQA, no QK-Norm), untied embeddings by default, causal SDPA.
Note: unlike HF Qwen2, the fused qkv projection has no bias.
Returns per-token hidden states H_t; heads (see heads.py) consume the state at the
anchor/last position (ICareFM-style per-step patient state).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.model.heads import NextEventHead

# Training attention on CUDA: the flash and memory-efficient kernels only. Without this a
# shape or dtype they refuse falls back silently to the math kernel, which materializes
# [B, heads, T, T] scores (an OOM at 8192 tokens); with it that case raises instead.
TRAINING_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return n * self.weight


def residual_input(x: torch.Tensor) -> torch.Tensor:
    """The residual stream starts (and so stays) fp32 in every arm. Under bf16 autocast an
    nn.Embedding returns fp32 but an nn.Linear (the TextCode projection, the continuous
    value channel) returns bf16, and `x + block(x)` would then carry a bf16 residual
    through every layer."""
    return x.float()


def build_rope_cache(pos: torch.Tensor, head_dim: int, base: float = 10000.0):
    """RoPE cos/sin from explicit position ids (admission-relative, 1-min resolution).

    pos: [B, T] integer minutes-since-admission per token (NOT sequence index).
    Returns cos, sin each [B, T, head_dim], in fp32 (angles of large minute positions
    lose their phase in bf16); `apply_rope` casts them to q/k's dtype.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=pos.device).float() / half))
    ang = pos.float()[..., None] * inv_freq[None, None, :]      # [B, T, half]
    ang = torch.cat([ang, ang], dim=-1)                          # [B, T, head_dim]
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, nh, T, hd]; cos/sin: [B, T, hd] fp32 -> x's dtype (bf16 under autocast), so
    # q/k are not promoted to fp32 next to a bf16 v.
    cos = cos[:, None].to(x.dtype)
    sin = sin[:, None].to(x.dtype)
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
        if q.is_cuda and self.training:
            with sdpa_kernel(TRAINING_SDPA_BACKENDS):
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
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
        self.lm_head = NextEventHead(
            d,
            vocab_size,
            tie_weights=xe.get("tied_embeddings", False),
            input_embedding=self.tok_emb,
        )
        self.n_heads = xe["n_heads"]
        self.head_dim = d // xe["n_heads"]
        self.blocks = nn.ModuleList(
            Block(d, xe["n_heads"], xe["ffn_mult"], xe["dropout"]) for _ in range(xe["n_layers"])
        )
        self.ln_f = RMSNorm(d)
        self.d_model = d
        self.rope_base = xe.get("rope_base", 10000.0)

    def forward(self, token, pos_min, token_weight=None) -> torch.Tensor:
        """Encode hard [B,T] IDs or soft [B,T,K] IDs plus normalized weights."""
        if token.ndim == 3 and token_weight is None:
            raise ValueError("token_weight is required for soft token IDs")
        if token_weight is not None:
            if token.ndim != 3 or token_weight.ndim != 3:
                raise ValueError(
                    "soft token and weight tensors must both be 3D [B,T,K]; "
                    f"got token.ndim={token.ndim}, weight.ndim={token_weight.ndim}"
                )
            if token.shape != token_weight.shape:
                raise ValueError("token and token_weight must have matching shapes")
        elif token.ndim == 3:
            raise ValueError(
                f"soft tokens (3D) require token_weight; got token.ndim={token.ndim}"
            )
        x = self.embed_tokens(token)
        if token_weight is not None:
            x = (x * token_weight.unsqueeze(-1)).sum(-2)
        x = residual_input(x)
        cos, sin = build_rope_cache(pos_min, self.head_dim, self.rope_base)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.ln_f(x)                                      # per-token states H_t

    def embed_tokens(self, token: torch.Tensor) -> torch.Tensor:
        """Input embedding of token ids ([B,T] or [B,T,K]); the TextCode arm overrides
        it with a frozen text-embedding table + trainable projection."""
        return self.tok_emb(token)

    def lm_logits(self, H: torch.Tensor) -> torch.Tensor:
        return self.lm_head(H)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
