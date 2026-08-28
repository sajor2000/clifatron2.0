#!/usr/bin/env python3
"""
packed_dataset.py - Load pre-packed sequences from parquet

Map-style dataset that loads pre-packed sequences with v2 schema preserving
document boundaries, episode keys, source spans, and continuation metadata
for CLIFATRON model experiments.

Works with DataLoader, DDP, and DeepSpeed.
"""

import polars as pl
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Dict, List

PACKED_SCHEMA_VERSION = "2.0.0"

REQUIRED_COLUMNS = {"input_ids", "attention_mask", "labels", "document_ids"}
OPTIONAL_MANIFEST_COLUMNS = {
    "packed_schema_version",
    "episode_keys",
    "segment_source_starts",
    "segment_source_ends",
    "segment_packed_starts",
    "segment_packed_ends",
    "segment_continuation_indices",
    "segment_continues_from_previous",
    "segment_continues_to_next",
    "artifact_hashes",
    "pos_min",
}


class PackedSequenceDataset(Dataset):
    """
    Dataset that loads pre-packed sequences from parquet.

    v2 schema adds episode keys and source-span metadata so that
    continuation segments and cross-episode boundaries are preserved.

    Args:
        parquet_path: Path to packed parquet file
    """

    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        self._version: str | None = None
        self._column_map: dict[str, int] = {}

        print(f"Loading packed sequences from {parquet_path}...")
        self.df = pl.read_parquet(parquet_path)
        self._inspect_schema()
        print(f"  ✓ Loaded {len(self.df)} packed sequences")

    @property
    def packed_schema_version(self) -> str:
        return self._version or PACKED_SCHEMA_VERSION

    def _inspect_schema(self) -> None:
        columns = set(self.df.columns)
        missing_required = REQUIRED_COLUMNS - columns
        if missing_required:
            raise ValueError(
                f"packed parquet is missing required columns: {sorted(missing_required)}"
            )
        if "packed_schema_version" in columns:
            versions = self.df["packed_schema_version"].unique().to_list()
            if len(versions) > 1:
                raise ValueError("packed parquet contains mixed schema versions")
            self._version = str(versions[0])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.row(idx, named=True)
        sample: dict[str, Any] = {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
            "document_ids": row["document_ids"],
            "packed_schema_version": self.packed_schema_version,
        }
        if "artifact_hashes" in row and row["artifact_hashes"] is not None:
            sample["artifact_hashes"] = row["artifact_hashes"]
        if "pos_min" in row and row["pos_min"] is not None:
            sample["pos_min"] = row["pos_min"]
        segments = self._build_segments(row)
        if segments:
            sample["segments"] = segments
        return sample

    def _build_segments(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        episode_keys = row.get("episode_keys")
        if episode_keys is None:
            return []
        source_starts = row.get("segment_source_starts")
        source_ends = row.get("segment_source_ends")
        packed_starts = row.get("segment_packed_starts")
        packed_ends = row.get("segment_packed_ends")
        cont_indices = row.get("segment_continuation_indices")
        cont_from = row.get("segment_continues_from_previous", [])
        cont_to = row.get("segment_continues_to_next", [])
        segments: list[dict[str, Any]] = []
        for i, key in enumerate(episode_keys):
            segments.append({
                "episode_key": key,
                "source_start": source_starts[i] if source_starts is not None else 0,
                "source_end": source_ends[i] if source_ends is not None else 0,
                "packed_start": packed_starts[i] if packed_starts is not None else 0,
                "packed_end": packed_ends[i] if packed_ends is not None else 0,
                "continuation_index": cont_indices[i] if cont_indices is not None else 0,
                "continues_from_previous": bool(cont_from[i]) if cont_from else False,
                "continues_to_next": bool(cont_to[i]) if cont_to else False,
            })
        return segments


def load_packed_dataset(packed_dir: str, split: str) -> PackedSequenceDataset:
    """
    Load pre-packed dataset for train/val split.

    Args:
        packed_dir: Directory containing packed parquet files
        split: 'train' or 'val'

    Returns:
        PackedSequenceDataset instance
    """
    packed_dir = Path(packed_dir)
    parquet_file = packed_dir / f"{split}_packed.parquet"

    if not parquet_file.exists():
        raise FileNotFoundError(
            f"Packed parquet file not found: {parquet_file}\n"
            f"Please run sequence packing first:\n"
            f"  uv run AR/qwen2/scripts/pack_sequences.py"
        )

    return PackedSequenceDataset(str(parquet_file))
