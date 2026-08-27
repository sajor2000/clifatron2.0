#!/usr/bin/env python3

"""
Step 3: Create train/validation/test splits

This script supports two modes:
1. Standard mode: Loads prepared_sequences.parquet and creates 80/10/10 splits
2. Presplit mode: Loads separate train_val and test files, splits train_val into 90/10

Usage:
    # Standard mode
    python 03_create_splits.py --data-dir ./data --vocab-dir ./vocab

    # Presplit mode (for narrative data)
    python 03_create_splits.py --presplit \\
        --train-val ./gpt2_data/clif_sentences_train_val.parquet \\
        --test ./gpt2_data/clif_sentences_test.parquet \\
        --vocab-dir ./gpt2_data/vocab \\
        --max-length 4096
"""

import argparse
import pathlib

import polars as pl
import numpy as np

from config import Config
from utils import setup_logging
from vocabulary import Vocabulary

logger = setup_logging()


def load_data(data_dir: pathlib.Path, vocab_dir: pathlib.Path):
    """Load prepared sequences and vocabulary using streaming"""
    # Load sequences
    sequences_path = data_dir / "prepared_sequences.parquet"
    if not sequences_path.exists():
        raise FileNotFoundError(
            f"Prepared sequences not found: {sequences_path}\n"
            "Please run 01_prepare_data.py first"
        )

    logger.info(f"Loading sequences from: {sequences_path}")
    logger.info("Using streaming mode for low memory...")

    # Use scan_parquet for lazy loading
    df = pl.scan_parquet(sequences_path)

    # Load vocabulary - try vocab_lock.json first, then fall back to vocab.gzip
    vocab_lock_path = vocab_dir / "vocab_lock.json"
    vocab_gzip_path = vocab_dir / "vocab.gzip"

    if vocab_lock_path.exists():
        logger.info(f"Loading vocabulary from: {vocab_lock_path}")
        vocab = Vocabulary.from_vocab_lock(vocab_lock_path)
        logger.info(f"Vocabulary size: {len(vocab)}")
        logger.info(f"Vocabulary hash: {vocab.get_vocab_hash()[:16]}...")
    elif vocab_gzip_path.exists():
        logger.info(f"Loading vocabulary from: {vocab_gzip_path} (legacy format)")
        vocab = Vocabulary().load(vocab_gzip_path)
        vocab.is_training = False  # Freeze vocabulary
        logger.info(f"Vocabulary size: {len(vocab)}")
    else:
        raise FileNotFoundError(
            f"Vocabulary not found at {vocab_lock_path} or {vocab_gzip_path}\n"
            "Please run scripts/build_vocab_from_data.py first"
        )

    return df, vocab


def load_presplit_data(train_val_path: pathlib.Path, test_path: pathlib.Path, vocab_dir: pathlib.Path):
    """
    Load pre-split train_val and test files (for narrative data)

    Args:
        train_val_path: Path to train_val sentences parquet
        test_path: Path to test sentences parquet
        vocab_dir: Path to vocabulary directory

    Returns:
        Tuple of (train_val_df, test_df, vocab)
    """
    # Check files exist
    if not train_val_path.exists():
        raise FileNotFoundError(
            f"Train/val file not found: {train_val_path}\n"
            "Please run 00_convert_narrative_to_sentences.py first"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_path}\n"
            "Please run 00_convert_narrative_to_sentences.py first"
        )

    logger.info("Loading presplit data...")
    logger.info(f"Train/val: {train_val_path}")
    logger.info(f"Test: {test_path}")
    logger.info("Using streaming mode for low memory...")

    # Load using scan_parquet for lazy loading
    train_val_df = pl.scan_parquet(train_val_path)
    test_df = pl.scan_parquet(test_path)

    # Load vocabulary - try vocab_lock.json first, then fall back to vocab.gzip
    vocab_lock_path = vocab_dir / "vocab_lock.json"
    vocab_gzip_path = vocab_dir / "vocab.gzip"

    if vocab_lock_path.exists():
        logger.info(f"Loading vocabulary from: {vocab_lock_path}")
        vocab = Vocabulary.from_vocab_lock(vocab_lock_path)
        logger.info(f"Vocabulary size: {len(vocab)}")
        logger.info(f"Vocabulary hash: {vocab.get_vocab_hash()[:16]}...")
    elif vocab_gzip_path.exists():
        logger.info(f"Loading vocabulary from: {vocab_gzip_path} (legacy format)")
        vocab = Vocabulary().load(vocab_gzip_path)
        vocab.is_training = False  # Freeze vocabulary
        logger.info(f"Vocabulary size: {len(vocab)}")
    else:
        raise FileNotFoundError(
            f"Vocabulary not found at {vocab_lock_path} or {vocab_gzip_path}\n"
            "Please run scripts/build_vocab_from_data.py first"
        )

    return train_val_df, test_df, vocab


def tokenize_sequences(
    df: pl.LazyFrame, vocab: Vocabulary, max_length: int = 4096, has_clif_sentence: bool = False
) -> pl.LazyFrame:
    """
    Convert token strings to token IDs using vocabulary with streaming

    Args:
        df: LazyFrame with token sequences
        vocab: Vocabulary object
        max_length: Maximum sequence length
        has_clif_sentence: If True, expect clif_sentence column (space-separated string)
                          If False, expect tokens column (list of strings)

    Returns:
        LazyFrame with tokenized sequences
    """
    logger.info("Tokenizing sequences using streaming mode...")

    # Pre-build special token IDs (updated to gpt2_hf style)
    bos_id = vocab("[BOS]")
    eos_id = vocab("[EOS]")
    unk_id = vocab("[UNK]")

    logger.info(f"Creating vocabulary lookup with {len(vocab)} tokens...")

    # OPTIMIZATION: Use native Polars replace operation instead of map_elements or joins
    # This is much faster and stays in streaming mode
    # Create lists of tokens and their IDs for bulk replace
    token_list = list(vocab.lookup.keys())
    token_id_list = list(vocab.lookup.values())

    logger.info("Converting tokens to IDs using vectorized replace...")

    # If we have clif_sentence (space-separated string), split it first
    if has_clif_sentence:
        logger.info("Splitting clif_sentence into tokens...")
        df = df.with_columns([
            pl.col("clif_sentence").str.split(" ").alias("tokens")
        ])

    # Convert tokens to IDs using native replace operation within list.eval
    df = df.with_columns([
        pl.col("tokens").list.eval(
            pl.element().replace_strict(
                old=token_list,
                new=token_id_list,
                default=unk_id,  # Unknown tokens map to UNK
                return_dtype=pl.Int64
            )
        ).alias("token_ids")
    ])

    # Add [BOS] and [EOS] tokens
    df = df.with_columns([
        pl.concat_list([
            pl.lit([bos_id]),
            pl.col("token_ids"),
            pl.lit([eos_id])
        ]).alias("input_ids")
    ]).drop("token_ids", "tokens")  # Drop intermediate columns

    # Update sequence length
    df = df.with_columns([pl.col("input_ids").list.len().alias("seq_len")])

    # Filter sequences that are too long
    logger.info(f"Filtering sequences longer than {max_length} tokens...")
    original_count = df.select(pl.len()).collect().item()
    df = df.filter(pl.col("seq_len") <= max_length)
    filtered_count = df.select(pl.len()).collect().item()

    if filtered_count < original_count:
        logger.warning(f"Filtered out {original_count - filtered_count:,} sequences (>{max_length} tokens)")
        logger.warning(f"Kept {filtered_count:,} sequences ({filtered_count/original_count*100:.1f}%)")

    return df


def create_splits_lazy(
    df: pl.LazyFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int, output_dir: pathlib.Path
):
    """
    Split data into train/val/test sets using pure lazy operations
    Writes directly to parquet without loading full dataset into memory

    Args:
        df: LazyFrame to split
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed
        output_dir: Output directory for splits
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )

    logger.info(
        f"Creating splits: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}"
    )
    logger.info("Using pure lazy operations for memory efficiency...")

    # Add row index for deterministic splitting
    df = df.with_row_index("__split_idx__")

    # Create hash for shuffling based on seed
    # Use row index + seed to create deterministic shuffle
    df = df.with_columns([
        ((pl.col("__split_idx__") * 2654435761 + seed) % 2147483647).alias("__hash__")
    ])

    # Sort by hash for shuffling
    df = df.sort("__hash__")

    # Re-index after shuffle
    df = df.drop("__split_idx__", "__hash__").with_row_index("__idx__")

    # OPTIMIZATION: Get total count efficiently without materializing full dataset
    logger.info("Counting total sequences...")
    # Use fetch to get count from a single row instead of full collect
    sample_count = df.select(pl.col("__idx__").max()).collect()["__idx__"][0] + 1

    logger.info(f"Total sequences: {sample_count:,}")

    # Calculate split points
    train_end = int(sample_count * train_ratio)
    val_end = train_end + int(sample_count * val_ratio)

    logger.info(f"Split points: train=0-{train_end}, val={train_end}-{val_end}, test={val_end}-{sample_count}")

    # Create output directories
    for split_name in ["train", "val", "test"]:
        (output_dir / split_name).mkdir(parents=True, exist_ok=True)

    # Write train split
    logger.info("Writing train split...")
    train_df = df.filter(pl.col("__idx__") < train_end).drop("__idx__")
    train_df.sink_parquet(output_dir / "train" / "data.parquet")
    logger.info(f"Train set: {train_end} sequences")

    # Write val split
    logger.info("Writing validation split...")
    val_df = df.filter((pl.col("__idx__") >= train_end) & (pl.col("__idx__") < val_end)).drop("__idx__")
    val_df.sink_parquet(output_dir / "val" / "data.parquet")
    logger.info(f"Validation set: {val_end - train_end} sequences")

    # Write test split
    logger.info("Writing test split...")
    test_df = df.filter(pl.col("__idx__") >= val_end).drop("__idx__")
    test_df.sink_parquet(output_dir / "test" / "data.parquet")
    logger.info(f"Test set: {sample_count - val_end} sequences")

    return train_end, val_end - train_end, sample_count - val_end


def save_splits_summary(
    train_count: int,
    val_count: int,
    test_count: int,
    output_dir: pathlib.Path,
):
    """Save summary of splits"""
    summary_path = output_dir / "splits_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Dataset Splits Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Train Set:\n")
        f.write(f"  Sequences: {train_count}\n\n")

        f.write(f"Validation Set:\n")
        f.write(f"  Sequences: {val_count}\n\n")

        f.write(f"Test Set:\n")
        f.write(f"  Sequences: {test_count}\n\n")

        f.write(f"Total: {train_count + val_count + test_count}\n")

    logger.info(f"Saved summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits",
        epilog="""
Examples:
  # Standard mode (80/10/10 split)
  python 03_create_splits.py --data-dir ./data --vocab-dir ./vocab

  # Presplit mode (for narrative data, 90/10 train/val split)
  python 03_create_splits.py --presplit \\
      --train-val ./gpt2_data/clif_sentences_train_val.parquet \\
      --test ./gpt2_data/clif_sentences_test.parquet \\
      --vocab-dir ./gpt2_data/vocab \\
      --output-dir ./gpt2_data/splits \\
      --max-length 4096 \\
      --train-val-split 0.9
        """
    )

    # Mode selection
    parser.add_argument(
        "--presplit",
        action="store_true",
        help="Use presplit mode (separate train_val and test files)"
    )

    # Standard mode arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (standard mode only)",
    )

    # Presplit mode arguments
    parser.add_argument(
        "--train-val",
        type=str,
        default=None,
        help="Path to train_val parquet (presplit mode only)"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Path to test parquet (presplit mode only)"
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.9,
        help="Fraction for train (rest is val) in presplit mode (default: 0.9)"
    )

    # Common arguments
    parser.add_argument(
        "--vocab-dir",
        type=str,
        default=None,
        help="Vocabulary directory (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=None, help="Train ratio for standard mode (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=None, help="Validation ratio for standard mode (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=None, help="Test ratio for standard mode (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # OPTIMIZATION: Configure Polars for low memory streaming operations
    logger.info("Configuring Polars for memory-optimized streaming...")
    pl.Config.set_streaming_chunk_size(10_000)  # Process in smaller chunks
    pl.Config.set_verbose(True)  # Show streaming info

    # Load configuration
    config = Config()

    # Set common parameters
    vocab_dir = pathlib.Path(args.vocab_dir) if args.vocab_dir else config.data.vocab_dir
    max_length = args.max_length
    seed = args.seed

    if args.presplit:
        # ========================================================================
        # PRESPLIT MODE (for narrative data)
        # ========================================================================
        logger.info("=" * 60)
        logger.info("Running in PRESPLIT mode")
        logger.info("=" * 60)

        # Validate arguments
        if not args.train_val or not args.test:
            raise ValueError("Presplit mode requires --train-val and --test arguments")

        # Set output directory
        output_dir = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path("./splits")

        # Load pre-split data
        train_val_path = pathlib.Path(args.train_val)
        test_path = pathlib.Path(args.test)
        train_val_df, test_df, vocab = load_presplit_data(train_val_path, test_path, vocab_dir)

        # Tokenize train_val
        logger.info("Tokenizing train_val sequences...")
        train_val_df = tokenize_sequences(train_val_df, vocab, max_length, has_clif_sentence=True)

        # Tokenize test
        logger.info("Tokenizing test sequences...")
        test_df = tokenize_sequences(test_df, vocab, max_length, has_clif_sentence=True)

        # Split train_val into train and val
        logger.info(f"Splitting train_val into train ({args.train_val_split:.1%}) and val ({1-args.train_val_split:.1%})...")
        train_ratio = args.train_val_split
        val_ratio = 1.0 - args.train_val_split

        # Create splits for train_val
        train_count, val_count, _ = create_splits_lazy(
            train_val_df, train_ratio, val_ratio, 0.0, seed, output_dir
        )

        # Save test split separately
        logger.info("Writing test split...")
        (output_dir / "test").mkdir(parents=True, exist_ok=True)
        test_df = test_df.with_row_index("__idx__").drop("__idx__")  # Ensure no index column
        test_df.sink_parquet(output_dir / "test" / "data.parquet")
        test_count = test_df.select(pl.len()).collect().item()
        logger.info(f"Test set: {test_count} sequences")

    else:
        # ========================================================================
        # STANDARD MODE (80/10/10 split)
        # ========================================================================
        logger.info("=" * 60)
        logger.info("Running in STANDARD mode")
        logger.info("=" * 60)

        # Set parameters
        data_dir = pathlib.Path(args.data_dir) if args.data_dir else config.data.data_dir
        output_dir = pathlib.Path(args.output_dir) if args.output_dir else data_dir
        train_ratio = args.train_ratio or config.data.train_ratio
        val_ratio = args.val_ratio or config.data.val_ratio
        test_ratio = args.test_ratio or config.data.test_ratio

        # Load data
        df, vocab = load_data(data_dir, vocab_dir)

        # Tokenize sequences
        df = tokenize_sequences(df, vocab, max_length, has_clif_sentence=False)

        # Create splits and write directly (stays lazy throughout)
        train_count, val_count, test_count = create_splits_lazy(
            df, train_ratio, val_ratio, test_ratio, seed, output_dir
        )

    # Save summary
    save_splits_summary(train_count, val_count, test_count, output_dir)

    logger.info("=" * 60)
    logger.info("Data splitting complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  train/data.parquet: {train_count:,} sequences")
    logger.info(f"  val/data.parquet: {val_count:,} sequences")
    logger.info(f"  test/data.parquet: {test_count:,} sequences")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
