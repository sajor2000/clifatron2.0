#!/usr/bin/env python3
"""
01_preprocess_data.py - Data Preprocessing Script for GPT2 Training

Preprocesses and caches tokenized datasets to avoid reprocessing on every training run.

This script:
1. Loads/builds the clinical tokenizer
2. Loads narrative datasets for all splits (train/val/test)
3. Tokenizes all samples
4. Saves cached tensors to disk

Usage:
    uv run AR/gpt2_hf/01_preprocess_data.py --model-size small --split-mode temporal
    uv run AR/gpt2_hf/01_preprocess_data.py --model-size medium --split-mode random --output-dir /path/to/cache

Output:
    Cached datasets saved to: models/gpt2_hf/preprocessed/{model_size}_{split_mode}_len8192/
    - train_dataset.pt (tokenized train data)
    - val_dataset.pt (tokenized validation data)
    - test_dataset.pt (tokenized test data)
    - metadata.json (dataset statistics and config)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer.clinical_tokenizer import ClinicalTokenizer
from data.narrative_dataset import load_narrative_dataset
from utils.cache_utils import (
    get_cache_dir,
    save_cached_datasets,
    verify_cache,
    print_cache_info,
)


def load_clif_config(config_path: str) -> dict:
    """Load clif_config.json."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def tokenize_dataset(dataset, split_name: str) -> dict:
    """
    Tokenize an entire dataset and return tensor dictionary.

    Args:
        dataset: ClinicalNarrativeDataset instance
        split_name: Name of split ('train', 'val', 'test')

    Returns:
        Dictionary containing:
            - input_ids: List of tokenized sequences
            - attention_mask: List of attention masks
            - labels: List of label sequences
            - hospitalization_ids: List of hospitalization IDs
            - chunk_info: List of chunk metadata
    """
    print(f"\nTokenizing {split_name} dataset ({len(dataset)} samples)...")

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    hosp_ids_list = []
    chunk_info_list = []

    # Process each sample
    for idx in tqdm(range(len(dataset)), desc=f"Tokenizing {split_name}"):
        sample = dataset[idx]

        # Extract tensors
        input_ids_list.append(sample['input_ids'])
        attention_mask_list.append(sample['attention_mask'])
        labels_list.append(sample['labels'])

        # Extract metadata
        hosp_ids_list.append(sample.get('hospitalization_id', f'unknown_{idx}'))
        chunk_info_list.append(sample.get('chunk_info', {}))

    print(f"  ✓ Tokenized {len(input_ids_list)} samples")

    return {
        'input_ids': input_ids_list,
        'attention_mask': attention_mask_list,
        'labels': labels_list,
        'hospitalization_ids': hosp_ids_list,
        'chunk_info': chunk_info_list,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and cache tokenized datasets for Qwen2 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess data for small model with temporal splits
  uv run AR/gpt2_hf/01_preprocess_data.py --model-size small --split-mode temporal

  # Preprocess with custom output directory
  uv run AR/gpt2_hf/01_preprocess_data.py --model-size medium --output-dir /data/cache

  # Check existing cache
  uv run AR/gpt2_hf/01_preprocess_data.py --model-size small --check-only

Output:
  Cached datasets saved to models/gpt2_hf/preprocessed/{model_size}_{split_mode}_len8192/
        """
    )

    # Model configuration
    parser.add_argument(
        '--model-size',
        type=str,
        required=True,
        choices=['nano', 'micro', 'tiny', 'small', 'medium'],
        help='Model size (determines cache directory name)'
    )

    # Data configuration
    parser.add_argument(
        '--clif-config',
        type=str,
        default='clif_config.json',
        help='Path to clif_config.json (default: clif_config.json)'
    )

    parser.add_argument(
        '--split-mode',
        type=str,
        default='temporal',
        choices=['temporal', 'random'],
        help='Data split strategy: temporal (2018-2023 train/val, 2024 test) or random'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=8192,
        help='Maximum sequence length (default: 8192)'
    )

    parser.add_argument(
        '--val-fraction',
        type=float,
        default=0.1,
        help='Validation fraction for random split (default: 0.1)'
    )

    parser.add_argument(
        '--test-fraction',
        type=float,
        default=0.1,
        help='Test fraction for random split (default: 0.1)'
    )

    parser.add_argument(
        '--train-val-fraction',
        type=float,
        default=0.9,
        help='Fraction of train_val data for training in temporal mode (default: 0.9)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for data splitting (default: 42)'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Base output directory for cached datasets (default: models/gpt2_hf/preprocessed/)'
    )

    # Tokenizer configuration
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default=None,
        help='Path to tokenizer directory (default: AR/gpt2_hf/tokenizer/clinical_tokenizer/)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['primary', 'secondary'],
        help='Vocabulary mode: primary (can build if missing), secondary (must use existing vocab)'
    )

    # Utility options
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check if cache exists and print info (no processing)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if cache exists'
    )

    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip preprocessing test split (only do train/val)'
    )

    args = parser.parse_args()

    # Determine paths
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        # Data outputs go to models/gpt2_hf/preprocessed at root level
        root_dir = script_dir.parent.parent  # Go up to CLIFATRON root (AR/gpt2_hf -> AR -> CLIFATRON)
        args.output_dir = root_dir / "models" / "gpt2_hf" / "preprocessed"

    if args.tokenizer_path is None:
        script_dir = Path(__file__).parent
        args.tokenizer_path = script_dir / "tokenizer" / "clinical_tokenizer"

    # Get cache directory
    cache_dir = get_cache_dir(
        base_output_dir=args.output_dir,
        model_size=args.model_size,
        split_mode=args.split_mode,
        max_length=args.max_length,
    )

    print("=" * 80)
    print("GPT2 DATA PREPROCESSING")
    print("=" * 80)
    print(f"Model size: {args.model_size}")
    print(f"Split mode: {args.split_mode}")
    print(f"Max length: {args.max_length}")
    print(f"Cache directory: {cache_dir}")
    print("")

    # Check-only mode
    if args.check_only:
        if verify_cache(cache_dir, splits=['train', 'val', 'test'] if not args.skip_test else ['train', 'val']):
            print("✓ Cache exists and is valid")
            print_cache_info(cache_dir)
        else:
            print("✗ Cache not found or invalid")
        return

    # Check if cache already exists
    if not args.force and verify_cache(cache_dir, splits=['train', 'val', 'test'] if not args.skip_test else ['train', 'val']):
        print("✓ Cache already exists at", cache_dir)
        print("\nUse --force to reprocess or --check-only to view cache info")
        print_cache_info(cache_dir)
        return

    # Load configuration
    print("Loading configurations...")
    clif_config = load_clif_config(args.clif_config)
    print(f"  ✓ CLIF config: {args.clif_config}")

    # Load or build tokenizer (with mode-based protection)
    print(f"\nLoading tokenizer (mode: {args.mode})...")
    tokenizer_exists = Path(args.tokenizer_path).exists()

    if args.mode == 'secondary':
        # Secondary mode: MUST use existing tokenizer, cannot build
        if not tokenizer_exists:
            raise FileNotFoundError(
                f"SECONDARY MODE ERROR: Tokenizer not found at {args.tokenizer_path}\n"
                f"Secondary mode requires existing vocabulary from primary site.\n"
                f"Please ensure the tokenizer directory from the primary site is available."
            )
        tokenizer = ClinicalTokenizer.from_pretrained(args.tokenizer_path)
        print(f"  ✓ Loaded existing tokenizer from {args.tokenizer_path}")

    elif args.mode == 'primary':
        # Primary mode: Use existing if available, build only if missing
        if tokenizer_exists:
            tokenizer = ClinicalTokenizer.from_pretrained(args.tokenizer_path)
            print(f"  ✓ Loaded existing tokenizer from {args.tokenizer_path}")
        else:
            print(f"  Tokenizer not found at {args.tokenizer_path}")
            print("  Building tokenizer from vocab_lock.json...")

            # Get vocab_lock.json path (data-driven vocabulary)
            script_dir = Path(__file__).parent
            root_dir = script_dir.parent.parent  # Go up to CLIFATRON root
            vocab_lock_path = root_dir / 'models' / 'gpt2_hf' / 'vocab_lock.json'

            if not vocab_lock_path.exists():
                raise FileNotFoundError(
                    f"Vocabulary lock file not found at {vocab_lock_path}\n"
                    f"Please build vocabulary first:\n"
                    f"  uv run python AR/gpt2_hf/scripts/build_vocab_from_data.py"
                )

            # Build tokenizer from vocab_lock.json
            tokenizer = ClinicalTokenizer.from_vocab_lock(
                vocab_lock_path=str(vocab_lock_path),
            )

            # Save the newly built tokenizer to both locations
            # 1. Code reference location: AR/gpt2_hf/tokenizer/clinical_tokenizer/
            os.makedirs(args.tokenizer_path, exist_ok=True)
            tokenizer.save_pretrained(args.tokenizer_path)
            print(f"  ✓ Built and saved tokenizer to {args.tokenizer_path}")

            # 2. Model artifacts location: models/gpt2_hf/tokenizer/
            script_dir = Path(__file__).parent
            root_dir = script_dir.parent.parent  # Go up to CLIFATRON root
            models_tokenizer_path = root_dir / "models" / "gpt2_hf" / "tokenizer"
            os.makedirs(models_tokenizer_path, exist_ok=True)
            tokenizer.save_pretrained(models_tokenizer_path)
            print(f"  ✓ Also saved tokenizer to {models_tokenizer_path} (locked vocabulary)")

    # Validate vocabulary
    print(f"  ✓ Vocabulary size: {len(tokenizer)}")
    vocab_hash = tokenizer.get_vocab_hash()
    print(f"  ✓ Vocabulary hash: {vocab_hash[:16]}...")
    print(f"  ⚠ IMPORTANT: Use this vocabulary for all training and finetuning!")

    # Determine which splits to process
    splits_to_process = ['train', 'val']
    if not args.skip_test:
        splits_to_process.append('test')

    print(f"\nProcessing splits: {', '.join(splits_to_process)}")
    print("")

    # Load and tokenize datasets
    all_tensors = {}

    for split in splits_to_process:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING {split.upper()} SPLIT")
        print(f"{'=' * 80}")

        # Load dataset
        dataset = load_narrative_dataset(
            config_path=args.clif_config,
            tokenizer=tokenizer,
            split=split,
            split_mode=args.split_mode,
            max_length=args.max_length,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            train_val_fraction=args.train_val_fraction,
            seed=args.seed,
        )

        # Tokenize dataset
        tensors = tokenize_dataset(dataset, split)
        all_tensors[split] = tensors

    # Create metadata (including vocab hash for consistency validation)
    metadata = {
        'model_size': args.model_size,
        'split_mode': args.split_mode,
        'max_length': args.max_length,
        'vocab_size': len(tokenizer),
        'vocab_hash': tokenizer.get_vocab_hash(),
        'mode': args.mode,
        'val_fraction': args.val_fraction,
        'test_fraction': args.test_fraction,
        'train_val_fraction': args.train_val_fraction,
        'seed': args.seed,
        'train_samples': len(all_tensors['train']['input_ids']),
        'val_samples': len(all_tensors['val']['input_ids']),
    }

    if 'test' in all_tensors:
        metadata['test_samples'] = len(all_tensors['test']['input_ids'])

    # Build config for cache verification
    cache_config = {
        'split_mode': args.split_mode,
        'max_length': args.max_length,
        'val_fraction': args.val_fraction,
        'test_fraction': args.test_fraction,
        'train_val_fraction': args.train_val_fraction,
        'seed': args.seed,
    }

    # Save cached datasets
    print(f"\n{'=' * 80}")
    print("SAVING CACHED DATASETS")
    print(f"{'=' * 80}")

    save_cached_datasets(
        cache_dir=cache_dir,
        train_tensors=all_tensors['train'],
        val_tensors=all_tensors['val'],
        test_tensors=all_tensors.get('test', None),
        metadata=metadata,
        config=cache_config,
    )

    print(f"\n{'=' * 80}")
    print("PREPROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Cache directory: {cache_dir}")
    print(f"\nTo train using this cached data:")
    print(f"  uv run torchrun --nproc_per_node=auto AR/qwen2/02_train_qwen2.py \\")
    print(f"    --model-size {args.model_size} \\")
    print(f"    --preprocessed-dir {cache_dir}")
    print("")


if __name__ == '__main__':
    main()
