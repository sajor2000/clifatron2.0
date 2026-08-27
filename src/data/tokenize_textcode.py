"""TextCode tokenizer: frozen BioClinical-ModernBERT code embeddings.

Based on Al Attrach et al. 2025 (arxiv:2512.05217): each CLIF code is mapped
to a human-readable natural language description, encoded by a frozen pretrained
clinical language model, and cached as a fixed embedding table. Only a small
projection layer is trained.

Key findings:
  - Frozen encoders beat trainable by 10+ AUROC points
  - 15x reduction in trainable parameters (14.5M -> <1M)
  - Larger clinical encoders improve further (BioClinical-ModernBERT-L > Tiny-ClinicalBERT)
  - Enhanced mapping coverage (100% descriptions) is required for fair comparison

Our implementation:
  1. Build a per-concept description from CLIF mCIDE vocabulary
  2. Encode all descriptions once with frozen BioClinical-ModernBERT
  3. Cache embeddings as a numpy array
  4. At training time: index cached embeddings + learn projection to model dim

This is the frozen-text-encoder arm. It competes against fused-event-binning
on transportability and calibration per Lee 2026 and Guo/Sung 2026.
"""

from __future__ import annotations

import numpy as np

CACHE_DTYPE = np.float32


def code_description(concept: str, source_table: str) -> str:
    """Map a CLIF concept to a human-readable description — STUB.

    The real implementation must read descriptions from the CLIF mCIDE
    vocabulary CSV files under external/clifatron/mCIDE/. This stub
    generates plausible synthetic descriptions from the concept name,
    which are adequate for smoke-testing the embedding pipeline but
    should NOT be used for the actual TextCode ablation arm.

    ponytail: global stub, implement from real mCIDE CSVs before running
    the textcode ablation arm.
    """
    desc = concept.replace("_", " ")
    if source_table == "labs":
        return f"laboratory measurement of {desc}"
    if source_table == "vitals":
        return f"vital sign measurement of {desc}"
    if source_table == "meds":
        return f"continuous infusion of {desc}"
    if source_table == "resp_support":
        return f"respiratory support using {desc}"
    if source_table == "adt":
        return f"patient admission to {desc}"
    return f"clinical event: {desc}"


def build_textcode_embeddings(vocab: list[str], model_name: str,
                               token_dim: int) -> tuple[np.ndarray, any]:
    """Precompute frozen embeddings for all vocabulary codes.

    Args:
        vocab: list of concept strings (e.g. ["heart_rate", "lactate", ...])
        model_name: HF model to use (default: BioClinical-ModernBERT-large)
        token_dim: target projection dimension

    Returns:
        cached: [len(vocab), model_dim] frozen embeddings
        proj: linear projection layer (model_dim -> token_dim), trainable
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    encoder.eval()

    descriptions = [f"clinical measurement: {c.replace('_', ' ')}" for c in vocab]
    cached = np.zeros((len(descriptions), encoder.config.hidden_size), dtype=CACHE_DTYPE)

    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(descriptions), batch_size):
            batch = descriptions[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=32)
            outputs = encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            cached[i : i + batch_size] = embeddings.astype(CACHE_DTYPE)

    proj = torch.nn.Linear(encoder.config.hidden_size, token_dim, bias=False)
    return cached, proj


def textcode_embedding(x_concept, cached_embeddings, projection):
    """Look up frozen BERT embeddings and project to model dimension.

    x_concept: [B,T] concept index into the cached embedding table
    cached_embeddings: [vocab_size, model_dim] numpy array
    projection: nn.Linear(model_dim, token_dim)

    Returns: [B,T,token_dim]

    This is the frozen-encoder pathway from Al Attrach 2025:
    no gradient through the language model, only through the projection.
    """
    import torch

    cached = torch.tensor(cached_embeddings, device=x_concept.device,
                          dtype=torch.float32)
    x = cached[x_concept]
    return projection(x)