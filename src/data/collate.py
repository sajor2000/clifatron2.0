"""CPU-only collation and document-isolated reference execution.

ANCHOR CONTRACT (U3). A batch has A anchors and N threshold-query rows, flattened over
every sample and segment in order:

    anchor_batch_idx [A]  row of the padded batch          anchor_idx [A]  position in it
    flash_anchor_idx [A]  position in the flattened (flash) stream
    cr_mask [A] bool      the anchor has a competing-risk label
    cr_type [A]           cause index of the event, -1 for a censored / event-free label
    cr_time_min [A]       minutes from the anchor to the event, censoring or horizon
    th_anchor [N]         which anchor (0..A-1) each threshold query belongs to
    th_target / th_tau / th_dir [N]   concept index, threshold value bin, direction (0/1)
    th_mask [N] bool      the query has a supervised status
    th_event [N] bool     the threshold was crossed (status ``positive``)
    th_time_min [N]       minutes from the anchor to the crossing, censoring or horizon

Times are minutes, never bins: each head bins them on its own grid (KTD4). A segment
lists its anchors under ``"anchors"`` (`src.data.dataset`); a segment without that key
contributes its ``anchor_offset`` (if any) as one unlabelled anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from src.data.targets import SUPERVISED_STATUSES

# Threshold-query rows of a batch (module docstring) and their dtypes.
_QUERY_FIELDS = {
    "th_anchor": torch.long, "th_mask": torch.bool, "th_target": torch.long,
    "th_tau": torch.long, "th_dir": torch.long, "th_event": torch.bool,
    "th_time_min": torch.long,
}


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
    if all(sample.get("input_value") is not None for sample in samples):
        # Continuous-fused arm: normalized current value + mask (padding is masked).
        batch["input_value"] = _pad_2d(samples, "input_value", max_length, 0.0, torch.float32)
        batch["input_value_mask"] = _pad_2d(samples, "input_value_mask", max_length, False,
                                            torch.bool)

    document_ids = torch.full((len(samples), max_length), -1, dtype=torch.long)
    flat_tokens: list[int] = []
    flat_positions: list[int] = []
    cu_seqlens = [0]
    segment_map: list[list[int]] = []
    anchor_batch_idx: list[int] = []
    anchor_idx: list[int] = []
    flash_anchor_idx: list[int] = []
    cr_mask: list[bool] = []
    cr_type: list[int] = []
    cr_time_min: list[int] = []
    queries: dict[str, list] = {name: [] for name in _QUERY_FIELDS}
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
            anchors = segment.get("anchors")
            if anchors is None:
                offset = segment.get("anchor_offset")
                anchors = [] if offset is None else [{"offset": offset, "cr": None,
                                                      "queries": []}]
            for anchor in anchors:
                local_anchor = int(anchor["offset"])
                if not start <= local_anchor < end:
                    raise ValueError("anchor lies outside its document segment")
                cr = anchor["cr"]
                for query in anchor["queries"]:
                    supervised = query["status"] in SUPERVISED_STATUSES
                    queries["th_anchor"].append(len(anchor_idx))
                    queries["th_mask"].append(supervised)
                    queries["th_target"].append(int(query["target_idx"]))
                    queries["th_tau"].append(int(query["threshold_bin"]))
                    queries["th_dir"].append(int(query["direction"]))
                    queries["th_event"].append(query["status"] == "positive")
                    queries["th_time_min"].append(int(query["minutes"]) if supervised else 0)
                anchor_batch_idx.append(row)
                anchor_idx.append(local_anchor)
                flash_anchor_idx.append(cu_seqlens[-2] + local_anchor - start)
                cr_mask.append(cr is not None)
                cr_type.append(-1 if cr is None else int(cr["cause"]))
                cr_time_min.append(0 if cr is None else int(cr["minutes"]))
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
            "cr_mask": torch.tensor(cr_mask, dtype=torch.bool),
            "cr_type": torch.tensor(cr_type, dtype=torch.long),
            "cr_time_min": torch.tensor(cr_time_min, dtype=torch.long),
            **{name: torch.tensor(queries[name], dtype=dtype)
               for name, dtype in _QUERY_FIELDS.items()},
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
