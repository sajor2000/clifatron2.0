#!/usr/bin/env python3
"""
00_convert_narrative_to_sentences.py - Convert Narrative Sequences to Sentence Format

Converts tokenETL narrative output (one token per row with metadata) into
GPT-2 training format (one hospitalization per row, space-separated tokens).

Input Format (from assemble_narratives.py):
    - train_val_sequences.parquet: Columns [hospitalization_id, event_time, clif_sentence, day, sequence_order]
    - test_sequences.parquet: Same schema
    - Each row is a single token with temporal metadata

Output Format (for GPT-2 pipeline):
    - clif_sentences_train_val.parquet: Columns [hospitalization_id, clif_sentence, seq_length]
    - clif_sentences_test.parquet: Same schema
    - clif_sentence is space-separated token string per hospitalization

Usage:
    uv run AR/gpt2/00_convert_narrative_to_sentences.py \\
        --train-val narratives/train_val_sequences.parquet \\
        --test narratives/test_sequences.parquet \\
        --output-dir ./gpt2_data

Author: Generated for CLIF GPT-2 Training Pipeline
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import polars as pl
from datetime import datetime


def setup_logging():
    """Setup simple logging to stdout."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def convert_narrative_to_sentences(
    input_path: str,
    output_path: str,
    split_name: str,
    logger
) -> pl.DataFrame:
    """
    Convert narrative sequences to sentence format.

    Args:
        input_path: Path to narrative sequences parquet
        output_path: Path for output sentences parquet
        split_name: Name for logging (e.g., 'train_val', 'test')
        logger: Logger instance

    Returns:
        Polars DataFrame with aggregated sentences
    """
    logger.info(f"=" * 60)
    logger.info(f"Converting {split_name} narratives to sentences")
    logger.info(f"=" * 60)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("")

    # Check input exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load narrative sequences
    logger.info("Loading narrative sequences...")
    start_time = datetime.now()

    # Use streaming mode for memory efficiency
    narrative_df = pl.scan_parquet(input_path)

    # Get row count (this requires a scan but is useful for progress)
    total_rows = narrative_df.select(pl.len()).collect().item()
    logger.info(f"  Loaded {total_rows:,} narrative events")

    # Group by hospitalization and concatenate tokens
    logger.info("Aggregating tokens by hospitalization...")
    logger.info("  - Sorting by event_time, sequence_order")
    logger.info("  - Concatenating clif_sentence tokens")

    sentences_df = (
        narrative_df
        # Sort within each group before aggregation
        .sort(['hospitalization_id', 'event_time', 'sequence_order'])
        # Group by hospitalization and concatenate tokens
        .group_by('hospitalization_id')
        .agg([
            pl.col('clif_sentence')
                .str.concat(delimiter=' ')
                .alias('clif_sentence')
        ])
        # Calculate sequence length (number of tokens)
        .with_columns([
            pl.col('clif_sentence')
                .str.split(' ')
                .list.len()
                .alias('seq_length')
        ])
        # Sort by hospitalization_id for determinism
        .sort('hospitalization_id')
        # Execute the lazy query
        .collect()
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"  ✓ Aggregation complete in {elapsed:.1f}s")
    logger.info("")

    # Statistics
    logger.info("Aggregated Sentence Statistics:")
    logger.info(f"  Total hospitalizations: {len(sentences_df):,}")
    logger.info(f"  Total tokens: {sentences_df['seq_length'].sum():,}")

    # Sequence length statistics
    seq_stats = sentences_df.select([
        pl.col('seq_length').min().alias('min_length'),
        pl.col('seq_length').quantile(0.25).alias('p25_length'),
        pl.col('seq_length').median().alias('median_length'),
        pl.col('seq_length').quantile(0.75).alias('p75_length'),
        pl.col('seq_length').quantile(0.95).alias('p95_length'),
        pl.col('seq_length').quantile(0.99).alias('p99_length'),
        pl.col('seq_length').max().alias('max_length'),
        pl.col('seq_length').mean().alias('mean_length'),
    ]).to_dicts()[0]

    logger.info(f"  Sequence length statistics:")
    logger.info(f"    Min:    {seq_stats['min_length']:,} tokens")
    logger.info(f"    25th:   {seq_stats['p25_length']:,.0f} tokens")
    logger.info(f"    Median: {seq_stats['median_length']:,.0f} tokens")
    logger.info(f"    75th:   {seq_stats['p75_length']:,.0f} tokens")
    logger.info(f"    95th:   {seq_stats['p95_length']:,.0f} tokens")
    logger.info(f"    99th:   {seq_stats['p99_length']:,.0f} tokens")
    logger.info(f"    Max:    {seq_stats['max_length']:,} tokens")
    logger.info(f"    Mean:   {seq_stats['mean_length']:,.1f} tokens")
    logger.info("")

    # Context length warnings
    context_4096_pct = (sentences_df['seq_length'] <= 4096).mean() * 100
    context_8192_pct = (sentences_df['seq_length'] <= 8192).mean() * 100

    logger.info(f"Context length coverage:")
    logger.info(f"  4096 tokens: {context_4096_pct:.1f}% of hospitalizations fit")
    logger.info(f"  8192 tokens: {context_8192_pct:.1f}% of hospitalizations fit")

    if context_4096_pct < 95:
        logger.warning(f"  ⚠ Only {context_4096_pct:.1f}% fit in 4096 context!")
        logger.warning(f"    Consider using 8192 context or sequence truncation")
    logger.info("")

    # Save to parquet
    logger.info(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sentences_df.write_parquet(output_path)

    file_size_mb = os.path.getsize(output_path) / (1024**2)
    logger.info(f"  ✓ Saved {file_size_mb:.2f} MB")
    logger.info("")

    return sentences_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Convert narrative sequences to sentence format for GPT-2 training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  uv run AR/gpt2/00_convert_narrative_to_sentences.py \\
      --train-val OutputTokens/narratives/train_val_sequences.parquet \\
      --test OutputTokens/narratives/test_sequences.parquet \\
      --output-dir ./gpt2_data

This will create:
  ./gpt2_data/clif_sentences_train_val.parquet
  ./gpt2_data/clif_sentences_test.parquet
        """
    )

    parser.add_argument(
        '--train-val',
        type=str,
        required=True,
        help='Path to train_val_sequences.parquet from assemble_narratives.py'
    )

    parser.add_argument(
        '--test',
        type=str,
        required=True,
        help='Path to test_sequences.parquet from assemble_narratives.py'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./gpt2_data',
        help='Output directory for sentence parquet files (default: ./gpt2_data)'
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Banner
    logger.info("=" * 60)
    logger.info("NARRATIVE TO SENTENCES CONVERTER")
    logger.info("Convert tokenETL narratives → GPT-2 training format")
    logger.info("=" * 60)
    logger.info("")

    # Validate inputs
    if not os.path.exists(args.train_val):
        logger.error(f"Train/val file not found: {args.train_val}")
        logger.error("Have you run assemble_narratives.py?")
        sys.exit(1)

    if not os.path.exists(args.test):
        logger.error(f"Test file not found: {args.test}")
        logger.error("Have you run assemble_narratives.py?")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output paths
    train_val_output = os.path.join(args.output_dir, 'clif_sentences_train_val.parquet')
    test_output = os.path.join(args.output_dir, 'clif_sentences_test.parquet')

    try:
        # Convert train_val
        train_val_df = convert_narrative_to_sentences(
            input_path=args.train_val,
            output_path=train_val_output,
            split_name='train_val (2018-2023)',
            logger=logger
        )

        # Convert test
        test_df = convert_narrative_to_sentences(
            input_path=args.test,
            output_path=test_output,
            split_name='test (2024)',
            logger=logger
        )

        # Summary
        logger.info("=" * 60)
        logger.info("CONVERSION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Train/Val: {len(train_val_df):,} hospitalizations")
        logger.info(f"Test:      {len(test_df):,} hospitalizations")
        logger.info(f"Total:     {len(train_val_df) + len(test_df):,} hospitalizations")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  {train_val_output}")
        logger.info(f"  {test_output}")
        logger.info("")
        logger.info("Next step:")
        logger.info("  uv run AR/gpt2/01a_build_vocab_from_registry.py \\")
        logger.info("      --token-registry token_registry.json \\")
        logger.info("      --output-dir ./gpt2_data/vocab")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
