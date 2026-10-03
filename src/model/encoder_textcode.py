"""TextCode encoder variant (Al Attrach et al. 2025, arXiv:2512.05217).

The input embedding of every fused vocab id is the FROZEN text embedding of its generated
description (`src/data/tokenize_textcode.py`), held as a non-trainable buffer, mapped to
the model width by a small trainable projection. The trunk, the untied next-event head
and the objective are the shared `CLIFEncoder` / `pretrain.Model` ones, so only the input
representation differs from the fused arms.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.model.encoder import CLIFEncoder


class TextCodeEncoder(CLIFEncoder):
    # The frozen table is a constant: `engine.wrap_ddp` does not broadcast it before every
    # forward (it is synced once when DDP wraps the model and stays in checkpoints).
    static_input_table = True

    def __init__(self, vocab_size: int, cfg: dict, text_table):
        if cfg["trunk"].get("tied_embeddings", False):
            raise ValueError("the TextCode arm needs untied embeddings: its input table "
                             "is frozen text, not a learnable output matrix")
        super().__init__(vocab_size, cfg)
        table = torch.as_tensor(np.asarray(text_table, dtype=np.float32))
        if table.ndim != 2 or table.shape[0] != vocab_size:
            raise ValueError(f"TextCode embedding table must be [vocab_size={vocab_size}, "
                             f"text_dim]; got {tuple(table.shape)}")
        if not bool(torch.isfinite(table).all()):
            raise ValueError("TextCode embedding table contains non-finite values")
        del self.tok_emb  # replaced by the frozen table + projection (no unused params)
        self.register_buffer("text_table", table)
        self.text_proj = nn.Linear(table.shape[1], self.d_model, bias=False)

    def embed_tokens(self, token: torch.Tensor) -> torch.Tensor:
        # proj(table)[token] == proj(table[token]): a batch with at least as many token
        # slots as table rows (training, [B,T,K] soft ids) projects the table once and
        # indexes it; a short batch (a generation step) projects only its own rows.
        if token.numel() >= self.text_table.shape[0]:
            return self.text_proj(self.text_table)[token]
        return self.text_proj(self.text_table[token])
