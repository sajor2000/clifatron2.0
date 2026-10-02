"""Map-style adapters for canonical fused-token shards and packed CLIFATRON rows.

`representation="decile"` is a legacy name for the canonical shards written by
`src/data/tokenize.py`, whatever `value_binning.scheme` produced them (clinical
segments by default).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import logging

from src.data.segments import RETOKENIZE, TOKENIZER_VERSION
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
        if self.representation == "decile" and (
            not hashes.get("numeric_edges")
            or str(hashes.get("tokenizer_version")) != str(TOKENIZER_VERSION)
        ):
            # KTD7: a canonical shard row records the tokenizer version and segments
            # hash it was encoded with. Without them it was built by the previous
            # tokenizer, whose bins no longer mean what the vocabulary says.
            raise TargetContractError(
                f"shard row is not bound to tokenizer-v{TOKENIZER_VERSION} segments "
                f"(no numeric_edges/tokenizer_version in artifact_hashes); {RETOKENIZE}"
            )
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


class LengthGroupedSampler(Sampler):
    """Shuffle, then group similar-length documents into each batch (HF
    group_by_length). Uniform shuffling over heavy-tailed CLIF lengths (mean 271,
    p95 471, max 6,413) pads every batch to its longest member — ~10x wasted
    compute and unbounded per-batch attention shape diversity, which drove the
    MPS caching-allocator watermark to 84 GiB and OOM'd an overnight run
    (gem-overnight-log 2026-09-26). Grouping bounds both: padded shapes recycle
    across batches and long stays only ever share a batch with other long stays.

    Single-process sampler (dev boxes). DDP ranks keep DistributedSampler for
    now; fold world-awareness in during G2 L40 bring-up.
    """

    def __init__(self, lengths, batch_size, *, seed=42, mega_batch_mult=50):
        self.lengths = [int(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.mega_batch_mult = int(mega_batch_mult)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=g).tolist()
        mega = max(self.batch_size * self.mega_batch_mult, self.batch_size)
        batches = []
        for start in range(0, len(order), mega):
            chunk = sorted(order[start:start + mega], key=lambda i: self.lengths[i])
            for k in range(0, len(chunk), self.batch_size):
                batches.append(chunk[k:k + self.batch_size])
        batch_order = torch.randperm(len(batches), generator=g).tolist()
        for b in batch_order:
            yield from batches[b]


class TokenBudgetBatchSampler(Sampler):
    """Batch sampler (yields index LISTS; use as DataLoader(batch_sampler=...)):
    length-groups like LengthGroupedSampler, then packs each batch to a token budget
    — B x max_len_in_batch <= max_batch_tokens. A 6,413-token stay batches ALONE;
    short stays pack tight. MPS math-path attention materializes T^2 buffers per
    layer for backward (~40 GiB for one 4x6413 batch), and uniform per_gpu batching
    over heavy-tailed CLIF lengths OOM'd three overnight runs (86 GiB watermark,
    updates ~205/~410/~2300 — gem-overnight-log). Token budgets bound the live
    transients AND cut padding waste to ~zero. opt-in via runtime.token_budget
    (CUDA flash attention doesn't need it; keep uniform per_gpu batches there).

    A single sequence longer than the budget still forms its own batch (allowed
    oversize; nothing is dropped).
    """

    def __init__(self, lengths, *, max_batch_tokens, max_batch_size=None,
                 seed=42, mega_batch_mult=50, shuffle=True):
        self.lengths = [int(length) for length in lengths]
        self.max_batch_tokens = int(max_batch_tokens)
        self.max_batch_size = int(max_batch_size) if max_batch_size else None
        self.seed = int(seed)
        self.mega_batch_mult = int(mega_batch_mult)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.lengths)

    def _batches(self) -> list[list[int]]:
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(self.lengths), generator=g).tolist()
        else:
            order = list(range(len(self.lengths)))  # deterministic (validation)
        mega = max((self.max_batch_size or 1) * self.mega_batch_mult, 1)
        batches: list[list[int]] = []
        for start in range(0, len(order), mega):
            chunk = sorted(order[start:start + mega], key=lambda i: self.lengths[i])
            batch: list[int] = []
            longest = 0
            for idx in chunk:
                cand = max(longest, self.lengths[idx])
                too_big = (
                    batch
                    and cand * (len(batch) + 1) > self.max_batch_tokens
                )
                too_many = (
                    self.max_batch_size is not None
                    and batch
                    and len(batch) + 1 > self.max_batch_size
                )
                if too_big or too_many:
                    batches.append(batch)
                    batch, longest = [], 0
                batch.append(idx)
                longest = max(longest, self.lengths[idx])
            if batch:
                batches.append(batch)
        return batches

    def __iter__(self):
        batches = self._batches()
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            batch_order = torch.randperm(len(batches), generator=g).tolist()
        else:
            batch_order = list(range(len(batches)))
        for b in batch_order:
            yield batches[b]


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
