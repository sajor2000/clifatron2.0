#!/usr/bin/env python3
"""
hospitalization_dataset.py - Load Parquet Data for Custom Packing

Loads event-level parquet data, groups by hospitalization_id,
and formats as text strings for custom sequence packing.

The parquet data structure:
- Each row = one clinical event (e.g., vitals, labs, assessments)
- Multiple rows per hospitalization
- Columns: hospitalization_id, event_time, sequence_order, clif_sentence
- CRITICAL: Already sorted by (hospitalization_id, sequence_order) - DO NOT re-sort

ORDERING REQUIREMENT:
The parquet files MUST be pre-sorted by (hospitalization_id, sequence_order).
This is critical because:
1. Some tokens have NO timestamp (e.g., demographics, PREV_NARRATIVE_START)
2. Some tokens have the SAME timestamp but different sequence_order
3. sequence_order is the only reliable way to preserve event order

Example from a single hospitalization:
  Row  | event_time | sequence_order | clif_sentence
  -----|------------|----------------|------------------
  1    | NULL       | 1              | PREV_NARRATIVE_START
  2    | NULL       | 2              | no_patient_history
  3    | NULL       | 3              | PREV_NARRATIVE_END
  4    | NULL       | 4              | age_56_65
  5    | NULL       | 4              | sex_female  (same seq_order as above!)
  6    | 2018-02-09 | 5              | day_1
  7    | 2018-02-09 | 6              | hour_11
  8    | 2018-02-09 | 7              | vitals_height...
  9    | 2018-02-09 | 7              | vitals_weight...  (same timestamp!)

This code preserves that exact order using Polars' maintain_order=True.
"""

import os
from typing import List, Dict
from pathlib import Path

import polars as pl
from torch.utils.data import Dataset


class HospitalizationTextDataset(Dataset):
    """
    Dataset that loads hospitalization narratives from parquet and returns text strings.

    TRL's ConstantLengthDataset will handle tokenization and packing.

    Args:
        parquet_path: Path to parquet file or directory containing train_val/test parquet files
        split: 'train', 'val', or 'test'
        split_mode: 'temporal' (2018-2023 train/val, 2024 test) or 'random'
        train_val_fraction: Fraction for train split when split_mode='temporal' (default: 0.9)
        seed: Random seed for train/val split (default: 42)
    """

    def __init__(
        self,
        parquet_path: str,
        split: str = 'train',
        split_mode: str = 'temporal',
        train_val_fraction: float = 0.9,
        seed: int = 42,
        tokenizer=None,
        max_length: int = 8192,
    ):
        self.parquet_path = parquet_path
        self.split = split
        self.split_mode = split_mode
        self.train_val_fraction = train_val_fraction
        self.seed = seed
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

        # Validate split_mode
        if split_mode not in ['temporal', 'random']:
            raise ValueError(f"Invalid split_mode: {split_mode}. Must be 'temporal' or 'random'")

        print(f"Loading hospitalization dataset ({split} split, {split_mode} mode)...")
        print(f"  Parquet: {parquet_path}")

        # Load hospitalizations
        self.hospitalizations = self._load_hospitalizations()

        print(f"  Loaded {len(self.hospitalizations)} hospitalizations for {split} split")

    def _load_hospitalizations(self) -> List[str]:
        """
        Load hospitalizations from parquet and convert to text strings.

        Returns:
            List of text strings, one per hospitalization
        """
        if self.split_mode == 'temporal':
            return self._load_temporal_split()
        else:
            return self._load_random_split()

    def _load_temporal_split(self) -> List[str]:
        """
        Load from temporal split parquet files.

        For train/val: Loads from train_val_sequences.parquet (2018-2023)
        For test: Loads from test_sequences.parquet (2024)

        Returns:
            List of text strings
        """
        base_dir = Path(self.parquet_path).parent if Path(self.parquet_path).is_file() else Path(self.parquet_path)

        if self.split in ['train', 'val']:
            # Load train_val_sequences.parquet (2018-2023 data)
            parquet_file = base_dir / 'train_val_sequences.parquet'
            print(f"  Loading temporal split: train_val_sequences.parquet (2018-2023)")
        else:  # test
            # Load test_sequences.parquet (2024 data)
            parquet_file = base_dir / 'test_sequences.parquet'
            print(f"  Loading temporal split: test_sequences.parquet (2024)")

        if not parquet_file.exists():
            raise FileNotFoundError(
                f"Temporal split file not found: {parquet_file}\n"
                f"Please run narrative assembly with temporal splits:\n"
                f"  uv run tokenETL/assemble_narratives.py"
            )

        # Load parquet file
        df = pl.read_parquet(parquet_file)

        # Get unique hospitalization IDs (preserves order from parquet)
        hosp_ids = df['hospitalization_id'].unique(maintain_order=True).to_list()
        total_hosps = len(hosp_ids)

        print(f"  Total hospitalizations: {total_hosps:,}")

        # For train/val splits, further split the train_val data
        if self.split in ['train', 'val']:
            import numpy as np
            np.random.seed(self.seed)

            # Shuffle hospitalization IDs for random train/val split
            hosp_ids_array = np.array(hosp_ids)
            np.random.shuffle(hosp_ids_array)
            hosp_ids = hosp_ids_array.tolist()

            train_size = int(total_hosps * self.train_val_fraction)

            if self.split == 'train':
                hosp_ids = hosp_ids[:train_size]
                print(f"  Train split: {len(hosp_ids):,} / {total_hosps:,} hospitalizations ({self.train_val_fraction*100:.1f}%)")
            else:  # val
                hosp_ids = hosp_ids[train_size:]
                print(f"  Val split: {len(hosp_ids):,} / {total_hosps:,} hospitalizations ({(1-self.train_val_fraction)*100:.1f}%)")
        else:  # test
            print(f"  Test split: {len(hosp_ids):,} hospitalizations (2024 data)")

        # Convert each hospitalization to text string
        return self._convert_to_text(df, hosp_ids)

    def _load_random_split(self) -> List[str]:
        """
        Load from single parquet and split randomly.

        Returns:
            List of text strings
        """
        # Load parquet file
        if Path(self.parquet_path).is_file():
            parquet_file = Path(self.parquet_path)
        else:
            parquet_file = Path(self.parquet_path) / 'narrative_sequences.parquet'

        if not parquet_file.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

        df = pl.read_parquet(parquet_file)

        # Get unique hospitalization IDs
        hosp_ids = df['hospitalization_id'].unique(maintain_order=True).to_list()
        total_hosps = len(hosp_ids)

        print(f"  Total hospitalizations: {total_hosps:,}")

        # Split hospitalizations into train/val/test
        import numpy as np
        np.random.seed(self.seed)

        hosp_ids_array = np.array(hosp_ids)
        np.random.shuffle(hosp_ids_array)

        val_size = int(total_hosps * 0.1)
        test_size = int(total_hosps * 0.1)
        train_size = total_hosps - val_size - test_size

        if self.split == 'train':
            hosp_ids = hosp_ids_array[:train_size].tolist()
        elif self.split == 'val':
            hosp_ids = hosp_ids_array[train_size:train_size + val_size].tolist()
        else:  # test
            hosp_ids = hosp_ids_array[train_size + val_size:].tolist()

        print(f"  Split sizes - Train: {train_size:,}, Val: {val_size:,}, Test: {test_size:,}")
        print(f"  Processing {len(hosp_ids):,} hospitalizations for {self.split}...")

        # Convert each hospitalization to text string
        return self._convert_to_text(df, hosp_ids)

    def _convert_to_text(self, df: pl.DataFrame, hosp_ids: List[str]) -> List[str]:
        """
        Convert hospitalization event rows to text strings.

        CRITICAL: This method relies on the parquet data being pre-sorted by
        (hospitalization_id, sequence_order). The Polars group_by with
        maintain_order=True preserves this ordering.

        Args:
            df: Polars DataFrame with all event data (pre-sorted by sequence_order)
            hosp_ids: List of hospitalization IDs to process

        Returns:
            List of text strings (space-separated tokens in correct sequence_order)
        """
        # CRITICAL: DO NOT SORT - parquet rows are already in correct chronological order
        # The sequence_order column exists but resets within different time periods
        # Sorting by sequence_order would BREAK the intended order
        print("  ✓ Preserving parquet row order (NO sorting applied)")

        # PERFORMANCE FIX: Filter once to only needed hospitalizations, then group
        df_filtered = df.filter(pl.col('hospitalization_id').is_in(hosp_ids))

        # Group by hospitalization_id (much faster than filtering in loop)
        # Note: Polars group_by returns (key_tuple, dataframe) pairs
        grouped = df_filtered.group_by('hospitalization_id', maintain_order=True)

        # Create mapping from hosp_id to text
        hosp_id_to_text = {}
        for key_tuple, hosp_df in grouped:
            # key_tuple is a tuple with single element (hosp_id,)
            hosp_id = key_tuple[0] if isinstance(key_tuple, tuple) else key_tuple

            # Extract tokens (clif_sentence column)
            tokens = hosp_df['clif_sentence'].to_list()

            # Filter out None values
            tokens = [str(token) for token in tokens if token is not None]

            # Join into space-separated text
            text = " ".join(tokens)

            hosp_id_to_text[hosp_id] = text

        # Return in original hosp_ids order
        # Handle missing hosp_ids gracefully (shouldn't happen but be safe)
        hospitalizations = []
        for hosp_id in hosp_ids:
            if hosp_id in hosp_id_to_text:
                hospitalizations.append(hosp_id_to_text[hosp_id])
            else:
                # Fallback: empty text for missing hosp_id
                print(f"Warning: Missing hospitalization {hosp_id}, using empty text")
                hospitalizations.append("")

        return hospitalizations

    def __len__(self) -> int:
        """Return number of hospitalizations."""
        return len(self.hospitalizations)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single hospitalization as pre-tokenized data.

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = self.hospitalizations[idx]

        # If no tokenizer provided, return text format
        if self.tokenizer is None:
            return {"text": text}

        # Tokenize the space-separated clinical tokens
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # Packing will handle padding
            return_tensors=None,  # Return lists
        )

        # For causal LM, labels are same as input_ids
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["input_ids"].copy(),
        }


def load_hospitalization_dataset(
    config_path: str,
    split: str = 'train',
    split_mode: str = 'temporal',
    train_val_fraction: float = 0.9,
    seed: int = 42,
    tokenizer=None,
    max_length: int = 8192,
) -> HospitalizationTextDataset:
    """
    Convenience function to load dataset from clif_config.json.

    Args:
        config_path: Path to clif_config.json
        split: Data split ('train', 'val', 'test')
        split_mode: Split strategy ('temporal' or 'random')
        train_val_fraction: Fraction for train split (default: 0.9)
        seed: Random seed (default: 42)
        tokenizer: Optional tokenizer for pre-tokenization
        max_length: Maximum sequence length (default: 8192)

    Returns:
        HospitalizationTextDataset instance
    """
    import json

    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'output_dir' not in config:
        raise ValueError("Required key 'output_dir' missing from clif_config.json")

    # Get narrative parquet path
    output_dir = config['output_dir']
    narratives_dir = os.path.join(output_dir, 'narratives')

    # Create dataset
    return HospitalizationTextDataset(
        parquet_path=narratives_dir,
        split=split,
        split_mode=split_mode,
        train_val_fraction=train_val_fraction,
        seed=seed,
        tokenizer=tokenizer,
        max_length=max_length,
    )
