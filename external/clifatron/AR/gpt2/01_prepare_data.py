#!/usr/bin/env python3

"""
Step 1: Prepare CLIF sentences data from tokenization_example.py output

This script:
1. Loads the clif_sentences.parquet file
2. Extracts the tokenized sequences
3. Prepares the data for vocabulary building and model training
"""

import argparse
import pathlib
import sys

import polars as pl

from config import Config
from utils import setup_logging

logger = setup_logging()


def load_clif_sentences(input_path: pathlib.Path) -> pl.DataFrame:
    """
    Load CLIF sentences from parquet file

    Args:
        input_path: Path to clif_sentences.parquet

    Returns:
        DataFrame with CLIF sentences
    """
    logger.info(f"Loading CLIF sentences from: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pl.read_parquet(input_path)

    logger.info(f"Loaded {len(df)} CLIF sentence records")
    logger.info(f"Columns: {df.columns}")

    return df


def validate_data(df: pl.DataFrame) -> bool:
    """
    Validate the loaded data has required columns

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, raises error otherwise
    """
    required_columns = ["clif_sentence"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")

    # Check for null values in clif_sentence
    null_count = df["clif_sentence"].null_count()
    if null_count > 0:
        logger.warning(f"Found {null_count} null values in clif_sentence column")

    logger.info("Data validation passed")
    return True


def prepare_sequences(df: pl.DataFrame) -> pl.DataFrame:
    """
    Prepare sequences for vocabulary building and training

    Args:
        df: DataFrame with CLIF sentences

    Returns:
        Processed DataFrame
    """
    # Remove null sentences
    df = df.filter(pl.col("clif_sentence").is_not_null())

    # Split sentences into tokens (tokens are space-separated)
    df = df.with_columns([
        pl.col("clif_sentence")
        .str.split(" ")
        .alias("tokens")
    ])

    # Calculate sequence lengths
    df = df.with_columns([
        pl.col("tokens").list.len().alias("seq_length")
    ])

    # Log statistics
    stats = df["seq_length"].describe()
    logger.info(f"\nSequence length statistics:\n{stats}")

    return df


def save_prepared_data(df: pl.DataFrame, output_dir: pathlib.Path):
    """
    Save prepared data

    Args:
        df: Prepared DataFrame
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "prepared_sequences.parquet"
    df.write_parquet(output_path)

    logger.info(f"Saved prepared data to: {output_path}")

    # Save summary statistics
    summary_path = output_dir / "data_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total sequences: {len(df)}\n")
        f.write(f"\nSequence length statistics:\n")
        f.write(str(df["seq_length"].describe()))
        f.write(f"\n\nSample sentences:\n")
        for i, row in enumerate(df.head(5).iter_rows(named=True)):
            f.write(f"\n{i+1}. {row['clif_sentence'][:200]}...\n")

    logger.info(f"Saved summary to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare CLIF sentences data")
    parser.add_argument(
        "--input",
        type=str,
        default="clif_sentences.parquet",
        help="Path to clif_sentences.parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)",
    )

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Set output directory
    output_dir = (
        pathlib.Path(args.output_dir) if args.output_dir else config.data.data_dir
    )

    # Load data
    input_path = pathlib.Path(args.input)
    df = load_clif_sentences(input_path)

    # Validate data
    validate_data(df)

    # Prepare sequences
    df = prepare_sequences(df)

    # Save prepared data
    save_prepared_data(df, output_dir)

    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
