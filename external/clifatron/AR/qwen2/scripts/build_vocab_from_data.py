#!/usr/bin/env python3
"""
Build vocabulary from parquet data files (no token_registry dependency).

This script:
1. Loads train_val and test parquet files
2. Extracts all unique tokens from clif_sentence column
3. Adds special tokens and time markers
4. Saves vocabulary to:
   - /home/vchaudha/CLIFATRON/models/qwen2_vocab.json
   - /home/vchaudha/CLIFATRON/AR/qwen2/vocab_lock.json

CRITICAL: Preserves order of first occurrence to ensure consistent vocab across sites.
"""

import json
import polars as pl
from pathlib import Path
from typing import Dict, List
import argparse


def extract_unique_tokens_ordered(parquet_paths: List[Path]) -> List[str]:
    """
    Extract unique tokens from parquet files, preserving order of first occurrence.

    IMPORTANT: Uses Polars unique(maintain_order=True) to preserve the order
    in which tokens first appear in the data. This is critical because some
    tokens don't have timestamps (demographics, PREV_NARRATIVE_START, etc.)
    and rely on sequence_order for correct positioning.

    Args:
        parquet_paths: List of paths to parquet files

    Returns:
        List of unique tokens in order of first occurrence
    """
    all_tokens = []

    for parquet_path in parquet_paths:
        print(f"Loading {parquet_path}...")
        df = pl.read_parquet(parquet_path)

        # Verify the data is sorted by sequence_order within each hospitalization
        # This is critical for preserving the correct order of events
        print(f"  Total rows: {len(df):,}")
        print(f"  Unique hospitalizations: {df['hospitalization_id'].n_unique():,}")

        # Extract unique tokens from this file (preserving order)
        file_tokens = df['clif_sentence'].unique(maintain_order=True).to_list()
        print(f"  Unique tokens in this file: {len(file_tokens):,}")

        all_tokens.extend(file_tokens)

    # Get unique tokens across all files (preserving order of first occurrence)
    unique_tokens = []
    seen = set()
    for token in all_tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)

    print(f"\nTotal unique tokens across all files: {len(unique_tokens):,}")
    return unique_tokens


def build_vocabulary(
    train_val_path: Path,
    test_path: Path,
    include_time_tokens: bool = True
) -> Dict[str, int]:
    """
    Build complete vocabulary from parquet data.

    Vocabulary structure:
    1. Special tokens (5): [PAD], [UNK], [BOS], [EOS], [SEP]
    2. Clinical tokens: From parquet data (preserving order)
    3. Time tokens (55): day_1 to day_30, day_30+, hour_1 to hour_24

    Args:
        train_val_path: Path to train_val_sequences.parquet
        test_path: Path to test_sequences.parquet
        include_time_tokens: Whether to add day/hour tokens

    Returns:
        Dictionary mapping token -> token_id
    """
    # 1. Special tokens (always first)
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    current_id = len(special_tokens)

    print("=" * 70)
    print("BUILDING VOCABULARY FROM PARQUET DATA")
    print("=" * 70)
    print(f"\nSpecial tokens: {special_tokens}")
    print(f"Starting vocab size: {len(vocab)}")

    # 2. Extract clinical tokens from parquet data
    print("\n" + "=" * 70)
    print("EXTRACTING CLINICAL TOKENS FROM PARQUET FILES")
    print("=" * 70)

    clinical_tokens = extract_unique_tokens_ordered([train_val_path, test_path])

    # Filter out time tokens if they're already in the data
    # (we'll add them explicitly at the end)
    if include_time_tokens:
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

        print(f"\nFound {len(found_time_tokens)} time tokens in data: {sorted(found_time_tokens)[:10]}...")
        print(f"These will be added explicitly at the end to ensure all are present")
        clinical_tokens = clinical_tokens_filtered

    # Add clinical tokens
    for token in clinical_tokens:
        vocab[token] = current_id
        current_id += 1

    print(f"\nClinical tokens added: {len(clinical_tokens):,}")
    print(f"Current vocab size: {len(vocab):,}")

    # 3. Add time tokens explicitly (if requested)
    if include_time_tokens:
        print("\n" + "=" * 70)
        print("ADDING TIME TOKENS")
        print("=" * 70)

        # Day tokens (31 total: day_1 through day_30, plus day_30+)
        day_tokens_list = [f"day_{i}" for i in range(1, 31)] + ["day_30+"]
        for token in day_tokens_list:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1

        # Hour tokens (24 total: hour_1 through hour_24)
        hour_tokens_list = [f"hour_{i}" for i in range(1, 25)]
        for token in hour_tokens_list:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1

        print(f"Day tokens: {len(day_tokens_list)} (day_1 to day_30, day_30+)")
        print(f"Hour tokens: {len(hour_tokens_list)} (hour_1 to hour_24)")

    print("\n" + "=" * 70)
    print("VOCABULARY COMPLETE")
    print("=" * 70)
    print(f"Final vocabulary size: {len(vocab):,}")
    print(f"  - Special tokens: {len(special_tokens)}")
    print(f"  - Clinical tokens: {len(clinical_tokens):,}")
    if include_time_tokens:
        print(f"  - Time tokens: 55 (31 days + 24 hours)")

    return vocab


def save_vocabulary(vocab: Dict[str, int], output_paths: List[Path]):
    """
    Save vocabulary to multiple locations.

    Args:
        vocab: Dictionary mapping token -> token_id
        output_paths: List of paths to save vocabulary
    """
    print("\n" + "=" * 70)
    print("SAVING VOCABULARY")
    print("=" * 70)

    # Create parent directories if needed
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Save vocabulary
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

    for path in output_paths:
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"✓ Saved to: {path}")
        print(f"  Size: {path.stat().st_size / 1024:.1f} KB")

    # Print sample tokens
    print("\n" + "=" * 70)
    print("SAMPLE TOKENS FROM VOCABULARY")
    print("=" * 70)

    # Get some example tokens from different categories
    tokens_by_id = sorted(vocab.items(), key=lambda x: x[1])

    print("\nFirst 10 tokens (special + clinical):")
    for token, token_id in tokens_by_id[:10]:
        print(f"  {token_id:4d}: {token}")

    print("\nLast 10 tokens (time markers):")
    for token, token_id in tokens_by_id[-10:]:
        print(f"  {token_id:4d}: {token}")

    # Print some clinical token examples
    clinical_start = 5  # After special tokens
    print(f"\nClinical tokens (IDs {clinical_start}-{clinical_start+9}):")
    for token, token_id in tokens_by_id[clinical_start:clinical_start+10]:
        print(f"  {token_id:4d}: {token}")


def main():
    parser = argparse.ArgumentParser(
        description="Build vocabulary from parquet data (no token_registry)"
    )
    parser.add_argument(
        "--train-val-path",
        type=Path,
        default=Path("/home/vchaudha/CLIFATRON/OutputTokens/narratives/train_val_sequences.parquet"),
        help="Path to train_val_sequences.parquet"
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("/home/vchaudha/CLIFATRON/OutputTokens/narratives/test_sequences.parquet"),
        help="Path to test_sequences.parquet"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/vchaudha/CLIFATRON/models"),
        help="Root directory for saving vocab (saves to root/qwen2_vocab.json)"
    )
    parser.add_argument(
        "--output-lock",
        type=Path,
        default=Path("/home/vchaudha/CLIFATRON/AR/qwen2/vocab_lock.json"),
        help="Path for vocab_lock.json"
    )
    parser.add_argument(
        "--no-time-tokens",
        action="store_true",
        help="Don't add explicit time tokens (day/hour markers)"
    )

    args = parser.parse_args()

    # Validate input files exist
    if not args.train_val_path.exists():
        raise FileNotFoundError(f"Train/val parquet not found: {args.train_val_path}")
    if not args.test_path.exists():
        raise FileNotFoundError(f"Test parquet not found: {args.test_path}")

    # Build vocabulary
    vocab = build_vocabulary(
        train_val_path=args.train_val_path,
        test_path=args.test_path,
        include_time_tokens=not args.no_time_tokens
    )

    # Save to both locations
    output_paths = [
        args.output_root / "qwen2_vocab.json",
        args.output_lock
    ]
    save_vocabulary(vocab, output_paths)

    print("\n" + "=" * 70)
    print("VOCABULARY BUILD COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Create tokenizer that loads from vocab_lock.json")
    print("2. Verify vocabulary is consistent across all sites")
    print("3. Train model with this vocabulary")


if __name__ == "__main__":
    main()
