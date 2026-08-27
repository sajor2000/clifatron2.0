#!/usr/bin/env python3
"""
01a_build_vocab_from_registry.py - Build Vocabulary from Token Registry

Builds complete vocabulary directly from token_registry.json, including all tokens
(even those with count=0) for maximum generalization capability.

Alternative to 02_build_vocab.py which builds vocabulary from actual data.

Input:
    - token_registry.json: Complete token registry with categories, counts, presence flags

Output:
    - vocab.gzip: Pickled Vocabulary object
    - vocabulary.csv: Human-readable token→ID mapping
    - vocab_stats.txt: Vocabulary statistics

Special Tokens (IDs 0-4):
    0: PAD       - Padding token
    1: TL_START  - Timeline start (BOS)
    2: TL_END    - Timeline end (EOS)
    3: UNK       - Unknown token
    4: TRUNC     - Truncation marker

Usage:
    uv run AR/gpt2/01a_build_vocab_from_registry.py \\
        --token-registry token_registry.json \\
        --output-dir ./gpt2_data/vocab

Author: Generated for CLIF GPT-2 Training Pipeline
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import polars as pl

from vocabulary import Vocabulary


def setup_logging():
    """Setup simple logging to stdout."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_token_registry(registry_path: str, logger) -> Dict:
    """
    Load token registry JSON file.

    Args:
        registry_path: Path to token_registry.json
        logger: Logger instance

    Returns:
        Dictionary with token registry data
    """
    logger.info(f"Loading token registry: {registry_path}")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Token registry not found: {registry_path}")

    with open(registry_path, 'r') as f:
        registry = json.load(f)

    logger.info(f"  ✓ Loaded {len(registry)} categories")
    return registry


def extract_all_tokens(registry: Dict, logger) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Extract all tokens from registry, including those with count=0.

    Args:
        registry: Token registry dictionary
        logger: Logger instance

    Returns:
        Tuple of (sorted_tokens, token_metadata)
        - sorted_tokens: List of tokens sorted by category then frequency
        - token_metadata: Dict mapping token → {category, count, present_in_data}
    """
    logger.info("=" * 60)
    logger.info("Extracting tokens from registry")
    logger.info("=" * 60)

    all_tokens = []
    token_metadata = {}

    # Category order (determines token ID ranges)
    # Special tokens come first (IDs 0-4)
    # Then clinical tokens in logical order
    category_order = [
        'cohort_adt',          # Demographics, transfers, disposition
        'elixhauser',          # Comorbidities
        'assessment',          # GCS, RASS
        'vitals',              # Vital signs
        'labs',                # Laboratory values
        'respiratory_support', # Respiratory devices and parameters
        'medications',         # Medication administration (renamed from medication_admin_continuous)
        'crrt_therapy',        # CRRT therapy
        'ecmo_mcs',            # ECMO/MCS therapy
    ]

    # Collect tokens by category
    tokens_by_category = {}
    total_tokens = 0
    present_tokens = 0

    for category in category_order:
        # Handle category name variations
        registry_category = category
        if category == 'medications' and category not in registry:
            registry_category = 'medication_admin_continuous'

        if registry_category not in registry:
            logger.warning(f"  ⚠ Category '{category}' not found in registry, skipping")
            continue

        category_tokens = registry[registry_category]
        logger.info(f"  {category:25s}: {len(category_tokens):5,} tokens")

        # Sort tokens by count (descending) for more logical ID assignment
        sorted_tokens = sorted(
            category_tokens.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        tokens_by_category[category] = []
        for token, metadata in sorted_tokens:
            tokens_by_category[category].append(token)
            token_metadata[token] = {
                'category': category,
                'count': metadata['count'],
                'present_in_data': metadata['present_in_data']
            }
            total_tokens += 1
            if metadata['present_in_data']:
                present_tokens += 1

    # Flatten tokens (preserving category order)
    sorted_tokens = []
    for category in category_order:
        if category in tokens_by_category:
            sorted_tokens.extend(tokens_by_category[category])

    logger.info("")
    logger.info(f"Total tokens extracted: {total_tokens:,}")
    logger.info(f"  Present in data: {present_tokens:,} ({present_tokens/total_tokens*100:.1f}%)")
    logger.info(f"  Missing from data: {total_tokens - present_tokens:,} ({(total_tokens-present_tokens)/total_tokens*100:.1f}%)")
    logger.info("")

    return sorted_tokens, token_metadata


def build_vocabulary_from_registry(
    registry_path: str,
    output_dir: str,
    logger
) -> Vocabulary:
    """
    Build vocabulary from token registry.

    Args:
        registry_path: Path to token_registry.json
        output_dir: Output directory for vocabulary files
        logger: Logger instance

    Returns:
        Vocabulary object
    """
    # Load registry
    registry = load_token_registry(registry_path, logger)

    # Extract all tokens
    clinical_tokens, token_metadata = extract_all_tokens(registry, logger)

    # Define special tokens (always IDs 0-4)
    special_tokens = ['PAD', 'TL_START', 'TL_END', 'UNK', 'TRUNC']

    logger.info("=" * 60)
    logger.info("Building vocabulary")
    logger.info("=" * 60)
    logger.info(f"Special tokens: {len(special_tokens)}")
    logger.info(f"  {', '.join(special_tokens)}")
    logger.info(f"Clinical tokens: {len(clinical_tokens):,}")
    logger.info(f"Total vocabulary size: {len(special_tokens) + len(clinical_tokens):,}")
    logger.info("")

    # Combine: special tokens first, then clinical tokens
    all_tokens = special_tokens + clinical_tokens

    # Check for duplicates
    if len(all_tokens) != len(set(all_tokens)):
        duplicates = [token for token in all_tokens if all_tokens.count(token) > 1]
        raise ValueError(f"Duplicate tokens found: {set(duplicates)}")

    # Create vocabulary
    logger.info("Creating Vocabulary object...")
    vocab = Vocabulary(words=tuple(all_tokens), is_training=True)

    # Add metadata to aux
    logger.info("Adding token metadata to vocabulary.aux...")
    for token, metadata in token_metadata.items():
        vocab.set_aux(token, metadata)

    # Mark special tokens in aux
    for special_token in special_tokens:
        vocab.set_aux(special_token, {
            'category': 'special',
            'count': 0,
            'present_in_data': True  # Always present by definition
        })

    # Freeze vocabulary
    vocab.is_training = False
    logger.info("  ✓ Vocabulary frozen (training mode disabled)")
    logger.info("")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save vocabulary
    logger.info("=" * 60)
    logger.info("Saving vocabulary files")
    logger.info("=" * 60)

    # 1. Save pickled vocabulary
    vocab_path = os.path.join(output_dir, 'vocab.gzip')
    logger.info(f"Saving vocab.gzip: {vocab_path}")
    vocab.save(vocab_path)
    vocab_size_mb = os.path.getsize(vocab_path) / (1024**2)
    logger.info(f"  ✓ Saved ({vocab_size_mb:.2f} MB)")

    # 2. Save CSV mapping
    vocab_csv_path = os.path.join(output_dir, 'vocabulary.csv')
    logger.info(f"Saving vocabulary.csv: {vocab_csv_path}")

    # Create DataFrame with token info
    vocab_data = []
    for token, token_id in vocab.lookup.items():
        metadata = vocab.aux.get(token, {})
        vocab_data.append({
            'token_id': token_id,
            'token': token,
            'category': metadata.get('category', 'unknown'),
            'count': metadata.get('count', 0),
            'present_in_data': metadata.get('present_in_data', False)
        })

    vocab_df = pl.DataFrame(vocab_data).sort('token_id')
    vocab_df.write_csv(vocab_csv_path)
    logger.info(f"  ✓ Saved ({len(vocab_df):,} tokens)")

    # 3. Save statistics
    stats_path = os.path.join(output_dir, 'vocab_stats.txt')
    logger.info(f"Saving vocab_stats.txt: {stats_path}")

    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("VOCABULARY STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Built: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {registry_path}\n")
        f.write("\n")

        f.write("Vocabulary Size:\n")
        f.write(f"  Total tokens: {len(vocab):,}\n")
        f.write(f"  Special tokens: {len(special_tokens)}\n")
        f.write(f"  Clinical tokens: {len(clinical_tokens):,}\n")
        f.write("\n")

        # Category breakdown
        f.write("Tokens by Category:\n")
        category_counts = vocab_df.group_by('category').agg([
            pl.len().alias('count'),
            pl.col('present_in_data').sum().alias('present_count')
        ]).sort('count', descending=True)

        for row in category_counts.iter_rows(named=True):
            category = row['category']
            count = row['count']
            present = row['present_count']
            pct = present / count * 100 if count > 0 else 0
            f.write(f"  {category:25s}: {count:5,} tokens ({present:5,} present, {pct:5.1f}%)\n")

        f.write("\n")

        # Present vs. missing
        present_total = vocab_df.filter(pl.col('present_in_data') == True).height
        missing_total = vocab_df.filter(pl.col('present_in_data') == False).height

        f.write("Data Coverage:\n")
        f.write(f"  Tokens present in data: {present_total:,} ({present_total/len(vocab)*100:.1f}%)\n")
        f.write(f"  Tokens missing from data: {missing_total:,} ({missing_total/len(vocab)*100:.1f}%)\n")
        f.write("\n")

        # Special tokens
        f.write("Special Tokens (IDs 0-4):\n")
        for token_id, token in enumerate(special_tokens):
            f.write(f"  ID {token_id}: {token}\n")
        f.write("\n")

        # Token ID ranges by category
        f.write("Token ID Ranges by Category:\n")
        for category in vocab_df['category'].unique().sort():
            category_tokens = vocab_df.filter(pl.col('category') == category)
            min_id = category_tokens['token_id'].min()
            max_id = category_tokens['token_id'].max()
            count = len(category_tokens)
            f.write(f"  {category:25s}: IDs {min_id:5,} - {max_id:5,} ({count:5,} tokens)\n")

    logger.info(f"  ✓ Saved")
    logger.info("")

    # 4. Generate and save vocabulary metadata (for vocabulary lock)
    logger.info("=" * 60)
    logger.info("VOCABULARY LOCK SYSTEM")
    logger.info("=" * 60)

    # Validate vocabulary size
    logger.info("Validating vocabulary...")
    vocab.validate_vocab_size(expected_size=1373)
    logger.info(f"  ✓ Vocabulary size: {len(vocab)} (expected: 1373)")

    # Generate vocabulary hash
    vocab_hash = vocab.get_vocab_hash()
    logger.info(f"  ✓ Vocabulary hash: {vocab_hash[:16]}...")

    # Save metadata
    metadata_path = os.path.join(output_dir, 'vocab_metadata.json')
    logger.info(f"Saving vocab_metadata.json: {metadata_path}")
    vocab.save_metadata(metadata_path)
    logger.info(f"  ✓ Saved")
    logger.info("")

    # Warning about vocabulary consistency
    logger.info("=" * 60)
    logger.info("⚠️  VOCABULARY LOCK: IMPORTANT!")
    logger.info("=" * 60)
    logger.info("This vocabulary MUST be used for ALL training/evaluation!")
    logger.info(f"Vocabulary hash: {vocab_hash}")
    logger.info("")
    logger.info("Do NOT rebuild vocabulary unless absolutely necessary.")
    logger.info("Rebuilding with different token registry will create")
    logger.info("incompatible models that cannot be compared.")
    logger.info("=" * 60)
    logger.info("")

    return vocab


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Build vocabulary from token_registry.json',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  uv run AR/gpt2/01a_build_vocab_from_registry.py \\
      --token-registry token_registry.json \\
      --output-dir ./gpt2_data/vocab

This will create:
  ./gpt2_data/vocab/vocab.gzip         (pickled Vocabulary)
  ./gpt2_data/vocab/vocabulary.csv      (human-readable mapping)
  ./gpt2_data/vocab/vocab_stats.txt     (statistics)

Vocabulary Structure:
  IDs 0-4:        Special tokens (PAD, TL_START, TL_END, UNK, TRUNC)
  IDs 5+:         Clinical tokens (sorted by category, then frequency)

All tokens from token_registry.json are included, even those with count=0.
This ensures the model can generalize to rare tokens not seen during training.
        """
    )

    parser.add_argument(
        '--token-registry',
        type=str,
        required=True,
        help='Path to token_registry.json from tokenETL pipeline'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./gpt2_data/vocab',
        help='Output directory for vocabulary files (default: ./gpt2_data/vocab)'
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Banner
    logger.info("=" * 60)
    logger.info("VOCABULARY BUILDER FROM REGISTRY")
    logger.info("Build complete vocabulary from token_registry.json")
    logger.info("=" * 60)
    logger.info("")

    # Validate input
    if not os.path.exists(args.token_registry):
        logger.error(f"Token registry not found: {args.token_registry}")
        logger.error("")
        logger.error("Expected file: token_registry.json")
        logger.error("Generated by: tokenETL/main.py")
        sys.exit(1)

    try:
        # Build vocabulary
        vocab = build_vocabulary_from_registry(
            registry_path=args.token_registry,
            output_dir=args.output_dir,
            logger=logger
        )

        # Summary
        logger.info("=" * 60)
        logger.info("VOCABULARY BUILD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Vocabulary size: {len(vocab):,} tokens")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info("")
        logger.info("Files created:")
        logger.info(f"  {os.path.join(args.output_dir, 'vocab.gzip')}")
        logger.info(f"  {os.path.join(args.output_dir, 'vocabulary.csv')}")
        logger.info(f"  {os.path.join(args.output_dir, 'vocab_stats.txt')}")
        logger.info("")
        logger.info("Next step:")
        logger.info("  uv run AR/gpt2/03_create_splits.py \\")
        logger.info("      --presplit \\")
        logger.info("      --train-val ./gpt2_data/clif_sentences_train_val.parquet \\")
        logger.info("      --test ./gpt2_data/clif_sentences_test.parquet \\")
        logger.info("      --vocab-dir ./gpt2_data/vocab \\")
        logger.info("      --output-dir ./gpt2_data/splits \\")
        logger.info("      --max-length 4096")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Vocabulary build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
