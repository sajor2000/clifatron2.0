"""Attach our survival/threshold heads to a CLIFATRON backbone (the "Method 3" wedge).

CLIFATRON ships trained HF causal LMs (GPT2 / Qwen2) over ~1,300 fused clinical tokens.
Every one exposes per-token hidden states via `output_hidden_states=True` and takes an
`attention_mask` — so our heads (heads.py) bolt straight onto `H_t` at the hour-24 anchor,
no retokenization. This gives a calibrated, cheap alternative to their Method 1
(XGBoost-on-embeddings) and Method 2 (Monte-Carlo rollout).

Two uses:
  - frozen probe: freeze the backbone, train only the heads on local labels;
  - joint fine-tune: unfreeze, add next-token loss (driven by `src/train/run_arm.py`,
    whose curriculum is the engine's, KTD5).

Labels follow the per-anchor batch contract of `src/data/collate.py`: times are MINUTES
since the anchor and each head bins them on its own grid with `heads.time_bin` (KTD4).

See notes/INTEGRATION.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from clif_validate._vendor.model.heads import (
    CompetingRiskHead,
    NextEventHead,
    ThresholdHazardHead,
    ValueRegressionHead,
    next_event_loss,
    time_bin,
)
from clif_validate._vendor.model.varlen_attention import (
    document_hidden_states,
    gather_anchor_states,
    validate_pack,
)


def load_backbone(checkpoint: str):
    """Load a trained CLIFATRON checkpoint as an HF causal LM (GPT2 or Qwen2)."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(checkpoint)


def on_grid(minutes: torch.Tensor, event: torch.Tensor, n_bins: int,
            horizon_hours: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Label minutes -> (event, slot) on ONE head's grid (KTD4). `slot` is `time_bin`:
    the event's bin, or the count of FULLY observed bins. An event after this head's
    horizon is not an event for it: the horizon was survived."""
    event = event & (minutes <= round(horizon_hours * 60))
    return event, time_bin(minutes, n_bins, horizon_hours).clamp(min=0, max=n_bins)


def touch(module: nn.Module | None, like: torch.Tensor) -> torch.Tensor:
    """Zero-valued loss reaching every trainable parameter of a skipped head, so DDP sees
    the same gradient set on every rank and step (KTD2)."""
    params = [] if module is None else [p for p in module.parameters() if p.requires_grad]
    if not params:
        return like.new_tensor(0.0)
    return (sum(p.sum() for p in params) * 0.0).to(like.dtype)


def _skipped(weight) -> bool:
    return not torch.is_tensor(weight) and float(weight) == 0.0


def hidden_dim(backbone) -> int:
    cfg = backbone.config
    return getattr(cfg, "n_embd", None) or cfg.hidden_size


class CLIFATRONHeads(nn.Module):
    """Backbone + our heads. Anchor = hour-24 token (their benchmark truncates to 24h);
    pass `anchor_idx` explicitly, else the last real token (from attention_mask) is used.
    `n_value_bins` is required: derive it from the frozen vocabulary
    (`segments.n_value_bins`), so the threshold head matches the bins it is queried with."""

    def __init__(self, backbone, n_targets: int, *, n_value_bins: int,
                 freeze_backbone: bool = True, cr_bins: int = 16, th_bins: int = 48,
                 enable_value: bool = True, tie_weights: bool = False,
                 cr_horizon_hours: float = 48.0, th_horizon_hours: float = 48.0):
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
        if freeze_backbone and not tie_weights:
            # A frozen probe never puts next-token loss in its objective (see `loss`).
            self.next_event.requires_grad_(False)
        self.cr = CompetingRiskHead(d, n_targets + 1, cr_bins)  # +1 for global death cause
        # Each head's own horizon: label minutes are binned per head (KTD4).
        self.cr_horizon_hours = float(cr_horizon_hours)
        self.th_horizon_hours = float(th_horizon_hours)
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
        One row per anchor, in the collator's anchor order (several per document on the
        full-hospitalization representation) — the rows the CR/threshold heads read.
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
        """Weighted head losses on the per-anchor batch contract (`src/data/collate.py`):
        `anchor_batch_idx` / `anchor_idx` per anchor, `cr_type` / `cr_time_min` /
        `cr_mask` per anchor, `th_*` per threshold query with `th_anchor`,
        `th_event`, `th_time_min`, `th_mask`. A weight given as the plain number 0 skips
        its head; a skipped head, or one with no supervised label in the batch, adds a
        zero-valued term that still reaches its parameters (KTD2)."""
        H = self.hidden_states(batch["input_ids"], batch["attention_mask"])
        if "anchor_batch_idx" in batch:
            h = H[batch["anchor_batch_idx"], batch["anchor_idx"]]
        else:
            h = self.anchor_state(H, batch["attention_mask"], batch.get("anchor_idx"))
        out = {"cr": self._cr_loss(batch, h, H, skip=_skipped(w_cr)),
               "th": self._th_loss(batch, h, H, skip=_skipped(w_th))}
        total = w_cr * out["cr"] + w_th * out["th"]
        if self.vr is not None and "value" in batch and not _skipped(w_val):
            if "ntp_target" in batch:  # engine batches: targets aligned at state[t]
                out["val"] = self.vr.loss_aligned(H, batch["ntp_target"], batch["value"],
                                                  batch["val_mask"])
            else:
                out["val"] = self.vr.loss(H, batch["input_ids"], batch["value"],
                                          batch["val_mask"])
        else:
            out["val"] = touch(self.vr, H)
        total = total + w_val * out["val"]
        if not self.frozen:  # joint next-token only makes sense when the backbone trains
            logits = self.next_event(H)
            out["ntp"] = next_event_loss(logits, batch.get("ntp_target", batch["input_ids"]),
                                         batch.get("ntp_mask"))
            total = total + w_ntp * out["ntp"]
        out["total"] = total
        return out

    def _cr_loss(self, batch, h, H, *, skip: bool):
        if skip or "cr_type" not in batch or "cr_time_min" not in batch:
            return touch(self.cr, H)
        event, slot = on_grid(batch["cr_time_min"], batch["cr_type"] >= 0,
                              self.cr.n_bins, self.cr_horizon_hours)
        cr_type = torch.where(event, batch["cr_type"], torch.full_like(slot, -1))
        # Censored before one full bin of this grid: nothing was observed.
        mask = event | (slot > 0)
        if "cr_mask" in batch:
            mask = mask & batch["cr_mask"]
        if not bool(mask.any()):
            return touch(self.cr, H)
        return self.cr.loss(h[mask], cr_type[mask], slot[mask])

    def _th_loss(self, batch, h, H, *, skip: bool):
        if skip or "th_target" not in batch or "th_time_min" not in batch:
            return touch(self.th, H)
        event, slot = on_grid(batch["th_time_min"], batch["th_event"],
                              self.th.n_bins, self.th_horizon_hours)
        crossed = torch.where(event, slot.clamp(max=self.th.n_bins - 1),
                              torch.full_like(slot, -1))
        mask = event | (slot > 0)
        if "th_mask" in batch:
            mask = mask & batch["th_mask"]
        if not bool(mask.any()):
            return touch(self.th, H)
        th_h = h[batch["th_anchor"]] if "th_anchor" in batch else h
        return self.th.loss(th_h[mask], batch["th_target"][mask], batch["th_tau"][mask],
                            batch["th_dir"][mask], crossed[mask], slot[mask])
