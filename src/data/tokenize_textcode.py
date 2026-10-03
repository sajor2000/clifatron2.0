"""TextCode tokenizer: frozen BioClinical-ModernBERT code embeddings.

Based on Al Attrach et al. 2025 (arxiv:2512.05217): each code is mapped to a
human-readable natural-language description, encoded by a frozen pretrained clinical
language model, and cached as a fixed embedding table. Only a small projection layer is
trained (`src/model/encoder_textcode.py`).

Key findings:
  - Frozen encoders beat trainable by 10+ AUROC points
  - 15x reduction in trainable parameters (14.5M -> <1M)
  - Enhanced mapping coverage (100% descriptions) is required for fair comparison

Our implementation (U6, KTD8) describes every FUSED vocab id of the frozen tokenizer-v2
vocabulary, so the arm sees exactly the primary arm's tokens:

  1. `code_description`: concept, the source table(s) that chart it (vocab
     ``concept_sources``), the bin interval with its closure brackets
     (`segments.interval_label`), and the reference unit (vocab ``reference_units``).
     The CLIF mCIDE free-text descriptions are not in the repo, so descriptions are
     generated from the vocabulary artifact itself — 100% coverage by construction.
  2. `build_textcode_embeddings`: encode every description once with a frozen encoder.
     The encoder is an injectable callable (``list[str] -> [N, D] array``) so tests need
     no network or model download; the default loads the HF model named in
     `configs/data.yaml` (``thomas-sounack/BioClinical-ModernBERT-base``).
  3. `textcode_table`: a ``[vocab_size, D]`` table (rows past the vocabulary and the
     padding id are zero), consumed by `TextCodeEncoder`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from src.data.segments import (
    ADMISSION_PREFIX,
    DEVICE_METRIC_PREFIX,
    DISCHARGE_PREFIX,
    interval_label,
)

CACHE_DTYPE = np.float32
DEFAULT_TEXTCODE_ENCODER = "thomas-sounack/BioClinical-ModernBERT-base"
TextEncoder = Callable[[Sequence[str]], np.ndarray]

_SPECIAL_DESCRIPTIONS = {
    "<pad>": "padding",
    "<bos>": "beginning of the hospital stay record",
    "<eos>": "end of the hospital stay record",
    "<unk>": "unknown clinical event",
}


def _words(text: str) -> str:
    return str(text).replace("_", " ").strip()


def _tables(concept: str, blob: Mapping) -> str:
    tables = ((blob.get("concept_sources") or {}).get("tables") or {}).get(concept) or []
    return "/".join(str(t) for t in tables)


def _unit(concept: str, blob: Mapping) -> str:
    unit = ((blob.get("reference_units") or {}).get("concepts") or {}).get(concept)
    if not isinstance(unit, str) or not unit:
        return ""
    if unit.startswith(DEVICE_METRIC_PREFIX):
        return f"device metric {_words(unit[len(DEVICE_METRIC_PREFIX):])}"
    return unit


def code_description(token: str, vocab_blob: Mapping) -> str:
    """Natural-language description of one fused vocab token.

    ``lactate=6`` -> ``"lactate (labs): value in (2, 2.2] mmol/L"``;
    ``cam_total=negative`` -> ``"cam total (assessments): negative"``;
    a bare presence token -> ``"icu (adt)"``; GEM terminal tokens and specials get fixed
    phrases. The interval is the bin's segment with its closure brackets."""
    if token in _SPECIAL_DESCRIPTIONS:
        return _SPECIAL_DESCRIPTIONS[token]
    for prefix, phrase in ((ADMISSION_PREFIX, "hospital admission type"),
                           (DISCHARGE_PREFIX, "hospital discharge disposition")):
        if token.startswith(prefix):
            return f"{phrase}: {_words(token[len(prefix):])}"
    segments = vocab_blob.get("segments") or {}
    concept, sep, suffix = token.partition("=")
    tables = _tables(concept, vocab_blob)
    head = f"{_words(concept)} ({tables})" if tables else _words(concept)
    if not sep:
        return head
    if concept in segments and suffix.isdigit() and int(suffix) < len(segments[concept]):
        interval = interval_label(segments[concept][int(suffix)])
        unit = _unit(concept, vocab_blob)
        return f"{head}: value in {interval}" + (f" {unit}" if unit else "")
    return f"{head}: {_words(suffix)}"


def vocab_descriptions(vocab_blob: Mapping) -> dict[int, str]:
    """``{token id: description}`` for every fused id of a vocabulary artifact."""
    return {int(i): code_description(token, vocab_blob)
            for token, i in vocab_blob["vocab"].items()}


def hf_text_encoder(model_name: str = DEFAULT_TEXTCODE_ENCODER, *,
                    batch_size: int = 64, max_length: int = 64) -> TextEncoder:
    """Frozen HF encoder as a ``list[str] -> [N, D]`` callable ([CLS] state, eval, no
    grad). Imported lazily: tests inject their own encoder and never download."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    def encode(texts: Sequence[str]) -> np.ndarray:
        rows = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                inputs = tokenizer(list(texts[i:i + batch_size]), return_tensors="pt",
                                   padding=True, truncation=True, max_length=max_length)
                hidden = encoder(**inputs).last_hidden_state[:, 0, :]
                rows.append(hidden.float().numpy())
        return np.concatenate(rows, axis=0)

    return encode


def build_textcode_embeddings(descriptions: Sequence[str], *,
                              encode: TextEncoder | None = None,
                              model_name: str = DEFAULT_TEXTCODE_ENCODER) -> np.ndarray:
    """Encode every description once with the frozen encoder -> ``[N, D]`` float32."""
    encode = encode or hf_text_encoder(model_name)
    cached = np.asarray(encode(list(descriptions)), dtype=CACHE_DTYPE)
    if cached.ndim != 2 or cached.shape[0] != len(descriptions):
        raise ValueError(f"text encoder returned shape {cached.shape} for "
                         f"{len(descriptions)} descriptions; expected [N, D]")
    if not np.isfinite(cached).all():
        raise ValueError("text encoder returned non-finite embeddings")
    return cached


def textcode_table(vocab_blob: Mapping, vocab_size: int, *,
                   encode: TextEncoder | None = None,
                   model_name: str = DEFAULT_TEXTCODE_ENCODER) -> np.ndarray:
    """``[vocab_size, D]`` frozen input table: row `i` embeds token id `i`'s description;
    the padding id and ids past the vocabulary are zero rows."""
    descriptions = vocab_descriptions(vocab_blob)
    if max(descriptions) >= vocab_size:
        raise ValueError(f"vocabulary id {max(descriptions)} does not fit the model's "
                         f"vocab_size {vocab_size}")
    ids = sorted(i for token, i in vocab_blob["vocab"].items() if token != "<pad>")
    cached = build_textcode_embeddings([descriptions[i] for i in ids], encode=encode,
                                       model_name=model_name)
    table = np.zeros((vocab_size, cached.shape[1]), dtype=CACHE_DTYPE)
    table[ids] = cached
    return table


def textcode_embedding(x_concept, cached_embeddings, projection):
    """Look up frozen BERT embeddings and project to model dimension.

    x_concept: [B,T] index into the cached embedding table
    cached_embeddings: [vocab_size, model_dim] numpy array
    projection: nn.Linear(model_dim, token_dim)

    Returns: [B,T,token_dim] — no gradient through the language model, only through
    the projection.
    """
    import torch

    cached = torch.tensor(cached_embeddings, device=x_concept.device,
                          dtype=torch.float32)
    x = cached[x_concept]
    return projection(x)
