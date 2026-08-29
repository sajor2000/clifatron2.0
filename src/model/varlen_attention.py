"""Document-isolated attention over packed rows (U13).

One output contract, two execution modes. Given the collator's flattened varlen view
(`flash_input_ids`, `cu_seqlens`, `flash_anchor_idx` from
`src/data/collate.py::collate_model_samples`), produce per-token hidden states
`[total_tokens, hidden]` in which **no token attends across a document boundary**, then
gather each document's anchor state.

- **Fallback (CPU; GPT2's real path).** Each document is a separate forward pass, so
  cross-document attention is impossible and NO dense `[batch, heads, len, len]`
  isolation mask is ever materialized. This is the block-diagonal structure realized
  the honest way, and it is what the data-free test suite exercises.
- **FlashAttention-2 (GPU; Qwen2/Qwen3).** One flattened forward. Per HF's
  packing-with-FA2 guidance, isolation is "limited to providing the `position_ids`" —
  FA2 reads per-document-resetting position ids to derive boundaries internally and
  skips cross-document attention with no dense mask. Runs only when the backbone was
  loaded with `attn_implementation="flash_attention_2"` AND a CUDA device + `flash-attn`
  are present; otherwise the fallback runs.

A single-document pack is numerically equivalent to the existing dense
`CLIFATRONHeads.hidden_states` forward, so this path is a faithful generalization of the
dense one, not a behavior change (see `tests/test_varlen_attention.py`).
"""

from __future__ import annotations

import torch


def _backbone_hidden_dim(backbone) -> int:
    cfg = backbone.config
    return getattr(cfg, "n_embd", None) or cfg.hidden_size


def flash_attention_available() -> bool:
    """True only when both a CUDA device and an importable `flash-attn` are present."""
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def _uses_flash_attention_2(backbone) -> bool:
    impl = getattr(getattr(backbone, "config", None), "_attn_implementation", None)
    return impl == "flash_attention_2"


def training_isolation_active(model) -> bool:
    """Whether `model`'s attention actually isolates packed documents during training.

    True only when the backbone runs FlashAttention-2 (which derives document
    boundaries from per-document position ids) AND a CUDA device + `flash-attn` are
    present. Eager/SDPA Qwen2 SFT has no document-isolation path that avoids a dense
    `[batch, heads, len, len]` mask (prohibited), so a multi-document pack there WOULD
    leak across documents and must stay rejected. A training entry point that accepts
    multi-document packs must gate on this — it is the enforced condition behind the
    fail-closed guard, not a comment beside it.
    """
    return _uses_flash_attention_2(model) and flash_attention_available()


def validate_pack(cu_seqlens, total: int) -> list[int]:
    """Fail closed on a malformed pack BEFORE any model forward runs.

    `cu_seqlens` must start at 0, be non-decreasing (each document has length >= 0),
    and end exactly at `total` (the flattened token count) — otherwise a
    `document_ids` / `cu_seqlens` disagreement or a truncated pack would silently
    mis-slice documents. Returns the validated boundaries as a Python list.
    """
    boundaries = [int(x) for x in cu_seqlens.tolist()]
    if not boundaries or boundaries[0] != 0:
        raise ValueError("cu_seqlens must start at 0")
    for prev, nxt in zip(boundaries[:-1], boundaries[1:]):
        if nxt < prev:
            raise ValueError("cu_seqlens must be non-decreasing")
    if boundaries[-1] != total:
        raise ValueError(
            f"cu_seqlens final boundary {boundaries[-1]} does not match the flattened "
            f"length {total}"
        )
    return boundaries


def document_hidden_states(backbone, flash_input_ids, cu_seqlens, *,
                           frozen: bool = True, force_fallback: bool = False):
    """`[total_tokens, hidden]` per-token hidden states with document isolation.

    `force_fallback=True` pins the per-document CPU path regardless of hardware — the
    test suite uses it so the isolation contract is exercised without a GPU.
    """
    total = int(flash_input_ids.shape[0])
    boundaries = validate_pack(cu_seqlens, total)
    device = next(backbone.parameters()).device
    ctx = torch.no_grad() if frozen else torch.enable_grad()
    spans = list(zip(boundaries[:-1], boundaries[1:]))

    use_fa2 = (not force_fallback and _uses_flash_attention_2(backbone)
               and flash_attention_available())
    if use_fa2:
        # Per-document 0-based position ids (0,1,…,0,1,…) are what FA2 reads to find
        # document boundaries; do not reuse admission-minute `pos_min` here.
        position_ids = torch.cat([
            torch.arange(end - start, device=device) for start, end in spans
        ]) if spans else torch.zeros(0, dtype=torch.long, device=device)
        with ctx:
            out = backbone(
                input_ids=flash_input_ids.to(device).unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
                output_hidden_states=True,
            )
        return out.hidden_states[-1].squeeze(0)  # [total, hidden]

    # Fallback: one forward per document. Isolation is structural — no mask needed.
    hidden = None
    with ctx:
        for start, end in spans:
            if end == start:
                continue  # a zero-length document contributes no tokens
            ids = flash_input_ids[start:end].to(device).unsqueeze(0)  # [1, doc_len]
            out = backbone(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
            )
            h = out.hidden_states[-1].squeeze(0)  # [doc_len, hidden]
            if hidden is None:
                hidden = h.new_zeros((total, h.size(-1)))
            hidden[start:end] = h
    if hidden is None:
        return torch.zeros((total, _backbone_hidden_dim(backbone)), device=device)
    return hidden


def gather_anchor_states(flat_hidden, flash_anchor_idx):
    """`[n_anchors, hidden]` — each anchored document's anchor hidden state.

    `flash_anchor_idx` indexes the flattened stream (the collator computes it as the
    document's start offset plus its local anchor). An index outside `[0, total)` is a
    corrupted pack and fails closed rather than gathering a neighbouring document's row.
    """
    total = flat_hidden.size(0)
    if flash_anchor_idx.numel() == 0:
        return flat_hidden.new_zeros((0, flat_hidden.size(-1)))
    lo = int(flash_anchor_idx.min())
    hi = int(flash_anchor_idx.max())
    if lo < 0 or hi >= total:
        raise ValueError(
            f"flash_anchor_idx spans [{lo}, {hi}] outside the flattened stream "
            f"[0, {total}); the pack is malformed"
        )
    return flat_hidden[flash_anchor_idx.to(flat_hidden.device)]


__all__ = [
    "document_hidden_states",
    "flash_attention_available",
    "gather_anchor_states",
    "training_isolation_active",
    "validate_pack",
]
