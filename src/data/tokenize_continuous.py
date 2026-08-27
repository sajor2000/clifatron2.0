"""Continuous-fused value tokenizer (McCann et al. 2026, medRxiv 2026.08.04).

Fused concept token + continuous numeric head: each event gets a discrete
concept embedding plus a learned continuous value projection from a small
numeric MLP. Instead of binning values into discrete tokens, the value is
represented as a continuous embedding concatenated/added to the concept embedding.

McCann: +30% numeric accuracy, -34% sequence length (fewer tokens), but WORSE
calibration than discrete binning. This arm tests the calibration trade-off.

Our implementation: concept token gets a standard embedding; value gets a
1-d convolution (learned projection) that maps scalar -> d-vector; the two
are summed before entering the transformer. No value tokens - sequence length
is number of events, not bin-expanded.

This is the ABLATION arm - our DEFAULT is discrete+soft (Lee 2026).
"""

from __future__ import annotations

import numpy as np


def normalize_value(value: float | None, concept: str) -> float:
    """Z-score normalizer stub. Real version computes mean/std from
    reference-site data and serializes along with frozen vocab."""
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def continuous_fused_embedding(x_concept, x_value, value_proj, embedding):
    """Produce fused concept+value embedding for a batch.

    x_concept: [B,T]  concept token IDs (vocab: concept names, no value bins)
    x_value:   [B,T]  raw normalized continuous values
    value_proj: nn.Linear(1, d) that maps scalar -> d-dimensional
    embedding: nn.Embedding(vocab_size, d)

    Returns: [B,T,d] summed embedding.

    In McCann's architecture the concept embedding and value projection
    are summed; the transformer sees one token per event (no bin expansion).
    This gives -34% sequence length vs discrete binning (10 bins/event).
    """
    import torch

    ce = embedding(x_concept)
    vp = value_proj(x_value.unsqueeze(-1))
    return ce + vp