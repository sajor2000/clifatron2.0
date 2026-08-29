"""Attach our survival/threshold heads to a CLIFATRON backbone (the "Method 3" wedge).

CLIFATRON ships trained HF causal LMs (GPT2 / Qwen2) over ~1,300 fused clinical tokens.
Every one exposes per-token hidden states via `output_hidden_states=True` and takes an
`attention_mask` — so our heads (heads.py) bolt straight onto `H_t` at the hour-24 anchor,
no retokenization. This gives a calibrated, cheap alternative to their Method 1
(XGBoost-on-embeddings) and Method 2 (Monte-Carlo rollout).

Two uses:
  - frozen probe: freeze the backbone, train only the heads on local labels;
  - joint fine-tune: unfreeze, add next-token loss (curriculum NTP -> +TTE heads).

See notes/INTEGRATION.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.heads import (
    CompetingRiskHead,
    NextEventHead,
    ThresholdHazardHead,
    ValueRegressionHead,
    next_event_loss,
)
from src.model.varlen_attention import (
    document_hidden_states,
    gather_anchor_states,
    validate_pack,
)


def load_backbone(checkpoint: str):
    """Load a trained CLIFATRON checkpoint as an HF causal LM (GPT2 or Qwen2)."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(checkpoint)


def hidden_dim(backbone) -> int:
    cfg = backbone.config
    return getattr(cfg, "n_embd", None) or cfg.hidden_size


class CLIFATRONHeads(nn.Module):
    """Backbone + our heads. Anchor = hour-24 token (their benchmark truncates to 24h);
    pass `anchor_idx` explicitly, else the last real token (from attention_mask) is used."""

    def __init__(self, backbone, n_targets: int, *, freeze_backbone: bool = True,
                  cr_bins: int = 16, th_bins: int = 48, n_value_bins: int = 10,
                  enable_value: bool = True, tie_weights: bool = False):
        super().__init__()
        self.backbone = backbone
        d = hidden_dim(backbone)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        self.frozen = freeze_backbone
        self.next_event = NextEventHead(
            d,
            backbone.config.vocab_size,
            tie_weights=tie_weights,
            input_embedding=backbone.get_input_embeddings(),
        )
        if not tie_weights:
            try:
                pretrained = backbone.get_output_embeddings()
                if pretrained is not None:
                    with torch.no_grad():
                        self.next_event.projection.weight.copy_(pretrained.weight)
            except Exception:
                pass
        self.cr = CompetingRiskHead(d, n_targets, cr_bins)
        self.th = ThresholdHazardHead(d, n_targets, th_bins, n_value_bins=n_value_bins)
        self.vr = ValueRegressionHead(d, backbone.config.vocab_size) if enable_value else None

    def hidden_states(self, input_ids, attention_mask):
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        return out.hidden_states[-1]                      # [B, T, d]

    def anchor_state(self, H, attention_mask, anchor_idx=None):
        if anchor_idx is None:
            anchor_idx = attention_mask.long().sum(1) - 1   # last real token
        idx = torch.arange(H.size(0), device=H.device)
        return H[idx, anchor_idx]                          # [B, d]

    def anchor_states_from_pack(self, batch, *, force_fallback: bool = False):
        """Per-document anchor states `[documents, d]` from a packed varlen batch (U13).

        Consumes the collator's flattened view (`flash_input_ids`, `cu_seqlens`,
        `flash_anchor_idx`) and runs the document-isolated attention path, so multiple
        episode-documents can share a packed row without attending across boundaries.
        One anchor row per anchored document — the shape the CR/threshold heads expect.
        """
        flat = document_hidden_states(
            self.backbone, batch["flash_input_ids"], batch["cu_seqlens"],
            frozen=self.frozen, force_fallback=force_fallback)
        boundaries = validate_pack(batch["cu_seqlens"], flat.size(0))
        return gather_anchor_states(flat, batch["flash_anchor_idx"], boundaries)

    # ---- zero-shot inference (no trained head needed for threshold queries) ----
    def threshold_prob(self, input_ids, attention_mask, target_idx, tau_bin, direction,
                       anchor_idx=None) -> torch.Tensor:
        """Cumulative failure F_k(h | H_anchor, τ, direction) — ICareFM zero-shot query.
        Compose with heads.composite_or / composite_and for multivariate events."""
        H = self.hidden_states(input_ids, attention_mask)
        h = self.anchor_state(H, attention_mask, anchor_idx)
        return self.th.cumulative_failure(h, target_idx, tau_bin, direction)

    # ---- training losses ----
    def loss(self, batch, w_ntp: float = 0.2, w_cr: float = 1.0, w_th: float = 1.0,
             w_val: float = 0.5) -> dict:
        H = self.hidden_states(batch["input_ids"], batch["attention_mask"])
        h = self.anchor_state(H, batch["attention_mask"], batch.get("anchor_idx"))
        out = {}
        out["cr"] = self.cr.loss(h, batch["cr_type"], batch["cr_bin"])
        out["th"] = self.th.loss(h, batch["th_target"], batch["th_tau"],
                                 batch["th_dir"], batch["th_crossed"])
        total = w_cr * out["cr"] + w_th * out["th"]
        if self.vr is not None and "value" in batch:
            out["val"] = self.vr.loss(H, batch["input_ids"], batch["value"], batch["val_mask"])
            total = total + w_val * out["val"]
        if not self.frozen:  # joint next-token only makes sense when the backbone trains
            logits = self.next_event(H)
            out["ntp"] = next_event_loss(logits, batch["input_ids"])
            total = total + w_ntp * out["ntp"]
        out["total"] = total
        return out
