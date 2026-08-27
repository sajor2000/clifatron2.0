#!/usr/bin/env python3

"""
Step 2: Build vocabulary from CLIF tokens

This script:
1. Loads prepared sequences
2. Extracts all unique tokens
3. Builds vocabulary with special tokens
4. Saves vocabulary for training
"""

import argparse
import pathlib
from collections import Counter

import polars as pl

from config import Config
from utils import setup_logging
from vocabulary import Vocabulary

logger = setup_logging()


def load_prepared_data(data_dir: pathlib.Path) -> pl.LazyFrame:
    """Load prepared sequences with lazy evaluation for low memory usage"""
    input_path = data_dir / "prepared_sequences.parquet"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {input_path}\n"
            "Please run 01_prepare_data.py first"
        )

    logger.info(f"Loading prepared data from: {input_path}")
    logger.info("Using streaming mode for low memory usage...")

    # Use scan_parquet for lazy loading
    df = pl.scan_parquet(input_path)

    return df


def build_vocabulary(df: pl.LazyFrame, min_freq: int = 1) -> Vocabulary:
    """
    Build vocabulary from token sequences using streaming for low memory

    Args:
        df: LazyFrame with token sequences
        min_freq: Minimum frequency for a token to be included

    Returns:
        Vocabulary object
    """
    logger.info("Building vocabulary from tokens using streaming mode...")
    logger.info("Processing sequences (counting in chunks)...")

    # Use polars streaming to efficiently count tokens with low memory
    token_counts_df = (
        df.select("tokens")
        .explode("tokens")
        .group_by("tokens")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .collect(engine="streaming")  # Streaming mode for low memory
    )

    logger.info(f"Unique tokens: {len(token_counts_df)}")

    # Get total count
    total_count = token_counts_df["count"].sum()
    logger.info(f"Total token occurrences: {total_count:,}")

    # Convert to dict for compatibility (only materializing filtered results)
    token_counts = {
        row["tokens"]: row["count"]
        for row in token_counts_df.iter_rows(named=True)
    }

    # Filter by minimum frequency
    filtered_tokens = {
        token: count for token, count in token_counts.items() if count >= min_freq
    }

    logger.info(
        f"Tokens after filtering (min_freq={min_freq}): {len(filtered_tokens)}"
    )

    # Special tokens for Llama
    special_tokens = (
        "PAD",          # Padding token
        "TL_START",     # Timeline start / BOS
        "TL_END",       # Timeline end / EOS
        "UNK",          # Unknown token
        "TRUNC",        # Truncation marker
    )

    # Initialize vocabulary with special tokens
    vocab = Vocabulary(words=special_tokens, is_training=True)

    logger.info(f"Added {len(special_tokens)} special tokens")

    # Add tokens sorted by frequency (most frequent first)
    sorted_tokens = sorted(
        filtered_tokens.items(), key=lambda x: x[1], reverse=True
    )

    for token, count in sorted_tokens:
        if token not in special_tokens:  # Don't add special tokens again
            vocab(token)

    vocab.is_training = False  # Freeze vocabulary

    logger.info(f"Final vocabulary size: {len(vocab)}")

    # Log top tokens
    logger.info("\nTop 20 most frequent tokens:")
    for i, (token, count) in enumerate(sorted_tokens[:20], 1):
        logger.info(f"  {i:2d}. {token:30s} ({count:,} occurrences)")

    return vocab, token_counts


def save_vocabulary(
    vocab: Vocabulary, token_counts: dict, output_dir: pathlib.Path
):
    """
    Save vocabulary and statistics

    Args:
        vocab: Vocabulary object
        token_counts: Token frequency counts (dict)
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save vocabulary
    vocab_path = output_dir / "vocab.gzip"
    vocab.save(vocab_path)
    logger.info(f"Saved vocabulary to: {vocab_path}")

    # Save vocabulary as CSV for inspection
    vocab_df = vocab.get_frame()

    # Add frequency information
    freq_map = {vocab(token): count for token, count in token_counts.items()}
    vocab_df = vocab_df.with_columns([
        pl.col("token")
        .map_elements(lambda t: freq_map.get(t, 0), return_dtype=pl.Int64)
        .alias("frequency")
    ])

    # Sort by token ID
    vocab_df = vocab_df.sort("token")

    csv_path = output_dir / "vocabulary.csv"
    vocab_df.write_csv(csv_path)
    logger.info(f"Saved vocabulary CSV to: {csv_path}")

    # Save vocabulary statistics
    stats_path = output_dir / "vocab_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Vocabulary Statistics\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Total vocabulary size: {len(vocab)}\n")
        f.write(f"Unique tokens in data: {len(token_counts)}\n\n")

        f.write(f"Special tokens:\n")
        for token in ["PAD", "TL_START", "TL_END", "UNK", "TRUNC"]:
            f.write(f"  {token}: {vocab(token)}\n")

        f.write(f"\nTop 50 most frequent tokens:\n")
        sorted_tokens = sorted(
            token_counts.items(), key=lambda x: x[1], reverse=True
        )
        for i, (token, count) in enumerate(sorted_tokens[:50], 1):
            token_id = vocab.lookup.get(token, vocab("UNK"))
            f.write(f"  {i:3d}. ID={token_id:5d} {token:30s} ({count:,})\n")

    logger.info(f"Saved statistics to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary from CLIF tokens")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: from config)",
    )
    parser.add_argument(
        "--vocab-dir",
        type=str,
        default=None,
        help="Vocabulary output directory (default: from config)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum token frequency (default: 1)",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Set directories
    data_dir = pathlib.Path(args.data_dir) if args.data_dir else config.data.data_dir
    vocab_dir = pathlib.Path(args.vocab_dir) if args.vocab_dir else config.data.vocab_dir

    # Load prepared data
    df = load_prepared_data(data_dir)

    # Build vocabulary
    vocab, token_counts = build_vocabulary(df, min_freq=args.min_freq)

    # Save vocabulary
    save_vocabulary(vocab, token_counts, vocab_dir)

    logger.info("Vocabulary building complete!")


if __name__ == "__main__":
    main()
