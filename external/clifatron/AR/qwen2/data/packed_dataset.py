#!/usr/bin/env python3
"""
packed_dataset.py - Load pre-packed sequences from parquet

Simple map-style dataset that loads pre-packed sequences.
Works reliably with DataLoader, DDP, and DeepSpeed.
"""

import polars as pl
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List


class PackedSequenceDataset(Dataset):
    """
    Dataset that loads pre-packed sequences from parquet.

    Args:
        parquet_path: Path to packed parquet file
    """

    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path

        print(f"Loading packed sequences from {parquet_path}...")
        self.df = pl.read_parquet(parquet_path)
        print(f"  ✓ Loaded {len(self.df)} packed sequences")

    def __len__(self) -> int:
        """Return number of packed sequences."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """
        Get a single packed sequence.

        Returns:
            Dictionary with input_ids, attention_mask, labels, and document_ids (1D)
        """
        row = self.df.row(idx, named=True)

        return {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": row["labels"],
            "document_ids": row["document_ids"],  # 1D array: document ID for each token
        }


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
