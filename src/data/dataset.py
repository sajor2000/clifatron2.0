"""Map-style adapters for canonical decile shards and packed CLIFATRON rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl
from torch.utils.data import DataLoader, Dataset, Sampler

import logging

from src.data.targets import TargetBuilder, TargetContractError

try:
    from external.clifatron.AR.qwen2.data.packed_dataset import PACKED_SCHEMA_VERSION  # type: ignore[import-untyped]
except ImportError:
    PACKED_SCHEMA_VERSION = "2.0.0"

logger = logging.getLogger(__name__)


class ModelDataset(Dataset):
    """Deterministic map-style dataset retaining packed document boundaries."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]] | str | Path,
        *,
        representation: str,
        target_builder: TargetBuilder,
        expected_hashes: Mapping[str, str],
        episode_targets: Mapping[str, Mapping[str, Any]] | None = None,
        epoch: int = 0,
    ) -> None:
        if representation not in {"decile", "clifatron_packed"}:
            raise ValueError("representation must be 'decile' or 'clifatron_packed'")
        if isinstance(records, (str, Path)):
            records = pl.read_parquet(records).to_dicts()
        self.records = [dict(record) for record in records]
        self.representation = representation
        self.target_builder = target_builder
        self.expected_hashes = dict(expected_hashes)
        self.episode_targets = dict(episode_targets or {})
        self.epoch = int(epoch)
        for record in self.records:
            self._validate_hashes(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = deepcopy(self.records[index])
        if self.representation == "decile":
            return self._decile_sample(record)
        return self._packed_sample(record)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _validate_hashes(self, record: Mapping[str, Any]) -> None:
        hashes = record.get("artifact_hashes")
        if not isinstance(hashes, Mapping):
            raise TargetContractError("sample is missing artifact_hashes")
        for name, expected in self.expected_hashes.items():
            if hashes.get(name) != expected:
                raise TargetContractError(f"artifact hash mismatch: {name}")

    def _decile_sample(self, record: dict[str, Any]) -> dict[str, Any]:
        built = self.target_builder.build(record, epoch=self.epoch)
        length = len(record["token"])
        return {
            "packed_schema_version": PACKED_SCHEMA_VERSION,
            "input_ids": record["token"],
            "attention_mask": [1] * length,
            "pos_min": record["pos_min"],
            "soft_token": record.get("soft_token"),
            "soft_weight": record.get("soft_weight"),
            "ntp_target": built["ntp_target"],
            "ntp_mask": built["ntp_mask"],
            "ntp_delta_min": built["ntp_delta_min"],
            "value_target": built["value_target"],
            "value_mask": built["value_mask"],
            "segments": [
                {
                    "episode_key": record["episode_key"],
                    "source_start": 0,
                    "source_end": length,
                    "packed_start": 0,
                    "packed_end": length,
                    "continuation_index": 0,
                    "continues_from_previous": False,
                    "continues_to_next": False,
                    "anchor_offset": built["anchor_idx"],
                    "outcome_labels": built["outcome_labels"],
                    "threshold_query": built["threshold_query"],
                }
            ],
        }

    def _packed_sample(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("packed_schema_version") != PACKED_SCHEMA_VERSION:
            raise TargetContractError("unsupported packed schema version")
        input_ids = list(record["input_ids"])
        attention_mask = list(record["attention_mask"])
        if len(input_ids) != len(attention_mask):
            raise TargetContractError("packed input and attention mask lengths differ")
        if "pos_min" in record and record["pos_min"] is not None:
            pos_min = list(record["pos_min"])
        else:
            pos_min = list(range(len(input_ids)))
            logger.warning(
                "packed record is missing pos_min column; "
                "falling back to range(%d). Admission-relative position is degraded.",
                len(input_ids),
            )
        output = {
            "packed_schema_version": PACKED_SCHEMA_VERSION,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pos_min": pos_min,
            "ntp_target": [0] * len(input_ids),
            "ntp_mask": [False] * len(input_ids),
            "ntp_delta_min": [0] * len(input_ids),
            "value_target": [0.0] * len(input_ids),
            "value_mask": [False] * len(input_ids),
            "segments": [],
        }
        occupied: set[int] = set()
        for segment in record.get("segments", []):
            segment = dict(segment)
            key = segment.get("episode_key")
            if not isinstance(key, str) or not key:
                raise TargetContractError("packed segment is missing an opaque episode key")
            required = {
                "source_start",
                "source_end",
                "packed_start",
                "packed_end",
                "continuation_index",
                "continues_from_previous",
                "continues_to_next",
            }
            if not required.issubset(segment):
                raise TargetContractError("packed segment is missing source/continuation metadata")
            source_start, source_end = int(segment["source_start"]), int(segment["source_end"])
            packed_start, packed_end = int(segment["packed_start"]), int(segment["packed_end"])
            if source_end <= source_start or packed_end <= packed_start:
                raise TargetContractError("packed segment spans must be non-empty")
            if source_end - source_start != packed_end - packed_start or packed_end > len(input_ids):
                raise TargetContractError("packed source and destination spans are inconsistent")
            positions = set(range(packed_start, packed_end))
            if occupied & positions:
                raise TargetContractError("packed segments overlap")
            occupied |= positions
            if key not in self.episode_targets:
                raise TargetContractError(f"packed segment has no target join: {key}")
            built = self.target_builder.build(self.episode_targets[key], epoch=self.epoch)
            for field in ("ntp_target", "ntp_mask", "ntp_delta_min", "value_target", "value_mask"):
                output[field][packed_start:packed_end] = built[field][source_start:source_end]
            anchor = built["anchor_idx"]
            contains_anchor = source_start <= anchor < source_end
            segment["anchor_offset"] = packed_start + anchor - source_start if contains_anchor else None
            segment["outcome_labels"] = built["outcome_labels"] if contains_anchor else []
            segment["threshold_query"] = built["threshold_query"] if contains_anchor else None
            output["segments"].append(segment)
        if not output["segments"]:
            raise TargetContractError("packed row contains no document segments")
        return output


def make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    collate_fn,
    sampler: Sampler | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Construct a loader without passing mutually exclusive sampler options."""
    if sampler is not None and shuffle:
        raise ValueError("sampler and shuffle are mutually exclusive")
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
    return DataLoader(**kwargs)
