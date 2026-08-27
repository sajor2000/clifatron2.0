#!/usr/bin/env python3
"""
build_vocab_from_data.py - Build Vocabulary from Parquet Data for GPT2 HF

Extracts vocabulary directly from narrative parquet files, ensuring:
1. Order preservation (critical for tokens without timestamps)
2. Complete time marker coverage (all day_* and hour_* tokens)
3. Locked vocabulary with SHA256 hash for cross-site consistency

Usage:
    python AR/gpt2_hf/scripts/build_vocab_from_data.py
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import polars as pl


def extract_unique_tokens_ordered(parquet_paths: List[str]) -> List[str]:
    """
    Extract unique tokens from parquet files while preserving order.

    CRITICAL: Uses maintain_order=True to preserve exact sequence of first
    occurrence. This is essential because tokens like PREV_NARRATIVE_START,
    demographics (age_*, sex_*), and no_patient_history don't have timestamps
    and rely on sequence_order for correct positioning.

    Args:
        parquet_paths: List of paths to parquet files

    Returns:
        List of unique tokens in order of first occurrence
    """
    all_tokens = []

    for parquet_path in parquet_paths:
        print(f"\n  Reading {os.path.basename(parquet_path)}...")

        # Load parquet with Polars
        df = pl.read_parquet(parquet_path)

        print(f"    Total rows: {len(df):,}")
        print(f"    Unique hospitalizations: {df['hospitalization_id'].n_unique():,}")

        # Extract unique tokens from this file (order-preserving)
        file_tokens = df['clif_sentence'].unique(maintain_order=True).to_list()
        print(f"    Unique tokens in file: {len(file_tokens):,}")

        # Add to master list
        all_tokens.extend(file_tokens)

    # Manual deduplication while preserving order
    print(f"\n  Deduplicating across files...")
    unique_tokens = []
    seen = set()

    for token in all_tokens:
        # Skip None tokens
        if token is None:
            continue
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)

    print(f"  Total unique tokens across all files: {len(unique_tokens):,}")

    return unique_tokens


def build_vocabulary(
    train_val_path: str,
    test_path: str,
    include_time_tokens: bool = True
) -> dict:
    """
    Build complete vocabulary with 3-tier structure.

    Structure:
        1. Special tokens (5):      [PAD], [UNK], [BOS], [EOS], [SEP]
        2. Clinical tokens:          From parquet data, order-preserved
        3. Time tokens (55):         day_1-30, day_30+, hour_1-24

    Args:
        train_val_path: Path to train_val_sequences.parquet
        test_path: Path to test_sequences.parquet
        include_time_tokens: Whether to add explicit time tokens (default: True)

    Returns:
        Dictionary mapping token -> ID
    """
    print("=" * 80)
    print("BUILDING VOCABULARY FROM DATA")
    print("=" * 80)

    # Step 1: Extract unique tokens from parquet files
    print("\nStep 1: Extracting unique tokens from parquet files...")
    clinical_tokens = extract_unique_tokens_ordered([train_val_path, test_path])

    # Step 2: Filter out time tokens (we'll add them explicitly later)
    if include_time_tokens:
        print("\nStep 2: Filtering out time tokens found in data...")
        day_tokens = {f"day_{i}" for i in range(1, 31)} | {"day_30+"}
        hour_tokens = {f"hour_{i}" for i in range(1, 25)}
        time_tokens = day_tokens | hour_tokens

        clinical_tokens_filtered = []
        found_time_tokens = []

        for token in clinical_tokens:
            if token in time_tokens:
                found_time_tokens.append(token)
            else:
                clinical_tokens_filtered.append(token)

        print(f"  Found {len(found_time_tokens)} time tokens in data (will re-add explicitly)")
        print(f"  Remaining clinical tokens: {len(clinical_tokens_filtered):,}")

        clinical_tokens = clinical_tokens_filtered

    # Step 3: Build vocabulary with 3-tier structure
    print("\nStep 3: Building vocabulary...")
    vocab = {}
    current_id = 0

    # Tier 1: Special tokens (5)
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
    for token in special_tokens:
        vocab[token] = current_id
        current_id += 1
    print(f"  Added {len(special_tokens)} special tokens (IDs 0-{current_id-1})")

    # Tier 2: Clinical tokens (from data, order-preserved)
    clinical_start_id = current_id
    for token in clinical_tokens:
        if token not in vocab:  # Safety check
            vocab[token] = current_id
            current_id += 1
    print(f"  Added {len(clinical_tokens):,} clinical tokens (IDs {clinical_start_id}-{current_id-1})")

    # Tier 3: Time tokens (55) - added explicitly
    if include_time_tokens:
        time_start_id = current_id

        # Day tokens: day_1 to day_30, then day_30+
        day_tokens_list = [f"day_{i}" for i in range(1, 31)] + ["day_30+"]
        for token in day_tokens_list:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1

        # Hour tokens: hour_1 to hour_24
        hour_tokens_list = [f"hour_{i}" for i in range(1, 25)]
        for token in hour_tokens_list:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1

        time_tokens_added = current_id - time_start_id
        print(f"  Added {time_tokens_added} time tokens (IDs {time_start_id}-{current_id-1})")
        print(f"    - Day tokens: 31 (day_1 to day_30, day_30+)")
        print(f"    - Hour tokens: 24 (hour_1 to hour_24)")

    print(f"\n  Total vocabulary size: {len(vocab):,}")

    return vocab


def save_vocabulary(vocab: dict, output_paths: List[str]):
    """
    Save vocabulary to multiple locations with metadata.

    Args:
        vocab: Dictionary mapping token -> ID
        output_paths: List of paths where to save vocab_lock.json
    """
    print("\nStep 4: Saving vocabulary...")

    # Prepare vocabulary data
    vocab_data = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "special_tokens": {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "sep_token": "[SEP]"
        }
    }

    # Compute hash for validation (using sorted vocab for determinism)
    vocab_json = json.dumps(vocab, sort_keys=True, ensure_ascii=False)
    vocab_hash = hashlib.sha256(vocab_json.encode('utf-8')).hexdigest()

    # Save to all output paths
    for output_path in output_paths:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("VOCABULARY BUILD COMPLETE")
    print("=" * 80)
    print(f"Vocabulary size: {len(vocab):,}")
    print(f"Vocabulary hash (SHA256): {vocab_hash}")
    print(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def main():
    """Main entry point."""
    # Paths
    root_dir = Path(__file__).parent.parent.parent.parent  # Go to CLIFATRON root
    train_val_path = root_dir / "OutputTokens" / "narratives" / "train_val_sequences.parquet"
    test_path = root_dir / "OutputTokens" / "narratives" / "test_sequences.parquet"

    # Output paths
    output_paths = [
        root_dir / "models" / "gpt2_hf" / "vocab_lock.json",  # Central location
        root_dir / "AR" / "gpt2_hf" / "vocab_lock.json",      # Local lock
    ]

    # Verify input files exist
    if not train_val_path.exists():
        raise FileNotFoundError(f"Train/val parquet not found: {train_val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test parquet not found: {test_path}")

    print(f"Input files:")
    print(f"  Train/Val: {train_val_path}")
    print(f"  Test:      {test_path}")
    print(f"\nOutput locations:")
    for path in output_paths:
        print(f"  {path}")

    # Build vocabulary
    vocab = build_vocabulary(
        train_val_path=str(train_val_path),
        test_path=str(test_path),
        include_time_tokens=True
    )

    # Save vocabulary
    save_vocabulary(vocab, [str(p) for p in output_paths])


if __name__ == "__main__":
    main()
