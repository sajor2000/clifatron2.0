"""CPU-only collation and document-isolated reference execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


def _pad_2d(samples: Sequence[dict[str, Any]], field: str, length: int, value: Any, dtype):
    result = torch.full((len(samples), length), value, dtype=dtype)
    for row, sample in enumerate(samples):
        values = sample.get(field)
        if values is not None:
            result[row, : len(values)] = torch.as_tensor(values, dtype=dtype)
    return result


def _pad_3d(samples: Sequence[dict[str, Any]], field: str, length: int, value: Any, dtype):
    width = max((len(sample[field][0]) for sample in samples if sample.get(field)), default=1)
    result = torch.full((len(samples), length, width), value, dtype=dtype)
    for row, sample in enumerate(samples):
        values = sample.get(field)
        if values is not None:
            tensor = torch.as_tensor(values, dtype=dtype)
            result[row, : tensor.shape[0], : tensor.shape[1]] = tensor
    return result


def collate_model_samples(
    samples: Sequence[dict[str, Any]], *, pad_token_id: int = 0
) -> dict[str, Any]:
    """Pad packed rows while retaining a variable-length document view on CPU."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    versions = {sample["packed_schema_version"] for sample in samples}
    if len(versions) != 1:
        raise ValueError("cannot mix packed schema versions")
    max_length = max(len(sample["input_ids"]) for sample in samples)
    batch = {
        "input_ids": _pad_2d(samples, "input_ids", max_length, pad_token_id, torch.long),
        "attention_mask": _pad_2d(samples, "attention_mask", max_length, 0, torch.bool),
        "pos_min": _pad_2d(samples, "pos_min", max_length, 0, torch.long),
        "ntp_target": _pad_2d(samples, "ntp_target", max_length, 0, torch.long),
        "ntp_mask": _pad_2d(samples, "ntp_mask", max_length, False, torch.bool),
        "ntp_delta_min": _pad_2d(samples, "ntp_delta_min", max_length, 0, torch.long),
        "value_target": _pad_2d(samples, "value_target", max_length, 0.0, torch.float32),
        "value_mask": _pad_2d(samples, "value_mask", max_length, False, torch.bool),
        "packed_schema_version": versions.pop(),
    }
    if all(sample.get("soft_token") is not None for sample in samples):
        batch["soft_token"] = _pad_3d(samples, "soft_token", max_length, 0, torch.long)
    if all(sample.get("soft_weight") is not None for sample in samples):
        batch["soft_weight"] = _pad_3d(samples, "soft_weight", max_length, 0.0, torch.float32)

    document_ids = torch.full((len(samples), max_length), -1, dtype=torch.long)
    flat_tokens: list[int] = []
    flat_positions: list[int] = []
    cu_seqlens = [0]
    segment_map: list[list[int]] = []
    anchor_batch_idx: list[int] = []
    anchor_idx: list[int] = []
    flash_anchor_idx: list[int] = []
    document_labels: list[list[dict[str, Any]]] = []
    threshold_queries: list[dict[str, Any] | None] = []
    episode_keys: list[str] = []
    for row, sample in enumerate(samples):
        for segment in sample["segments"]:
            start, end = int(segment["packed_start"]), int(segment["packed_end"])
            if not batch["attention_mask"][row, start:end].all():
                raise ValueError("document segment includes padding")
            document_id = len(segment_map)
            document_ids[row, start:end] = document_id
            flat_tokens.extend(sample["input_ids"][start:end])
            flat_positions.extend(sample["pos_min"][start:end])
            cu_seqlens.append(cu_seqlens[-1] + end - start)
            segment_map.append([row, start, end])
            episode_keys.append(segment["episode_key"])
            if segment.get("anchor_offset") is not None:
                local_anchor = int(segment["anchor_offset"])
                anchor_batch_idx.append(row)
                anchor_idx.append(local_anchor)
                flash_anchor_idx.append(cu_seqlens[-2] + local_anchor - start)
                document_labels.append(segment["outcome_labels"])
                threshold_queries.append(segment["threshold_query"])
    batch.update(
        {
            "document_ids": document_ids,
            "flash_input_ids": torch.tensor(flat_tokens, dtype=torch.long),
            "flash_position_ids": torch.tensor(flat_positions, dtype=torch.long),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "max_seqlen": max((end - start for _, start, end in segment_map), default=0),
            "segment_map": torch.tensor(segment_map, dtype=torch.long),
            "episode_keys": episode_keys,
            "anchor_batch_idx": torch.tensor(anchor_batch_idx, dtype=torch.long),
            "anchor_idx": torch.tensor(anchor_idx, dtype=torch.long),
            "flash_anchor_idx": torch.tensor(flash_anchor_idx, dtype=torch.long),
            "document_labels": document_labels,
            "threshold_queries": threshold_queries,
        }
    )
    if any(tensor.is_cuda for tensor in batch.values() if isinstance(tensor, torch.Tensor)):
        raise RuntimeError("collation must return CPU tensors")
    return batch


@dataclass(frozen=True)
class ModelCollator:
    """Top-level picklable DataLoader collator."""

    pad_token_id: int = 0

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return collate_model_samples(samples, pad_token_id=self.pad_token_id)


def document_isolated_forward(model, batch: dict[str, Any]) -> list[torch.Tensor]:
    """CPU qualification fallback: invoke a sequence model once per document."""
    outputs = []
    for row, start, end in batch["segment_map"].tolist():
        outputs.append(
            model(
                batch["input_ids"][row : row + 1, start:end],
                batch["pos_min"][row : row + 1, start:end],
            )
        )
    return outputs
