"""Utilities for caching preprocessed datasets."""

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import hashlib


def get_cache_dir(
    base_output_dir: str,
    model_size: str,
    split_mode: str,
    max_length: int = 8192,
) -> Path:
    """
    Generate cache directory path based on configuration.

    Args:
        base_output_dir: Base output directory (e.g., AR/qwen2/outputs/preprocessed)
        model_size: Model size (e.g., "0.5b", "1.5b", "7b")
        split_mode: Split mode ("temporal" or "random")
        max_length: Maximum sequence length

    Returns:
        Path to cache directory
    """
    cache_name = f"{model_size}_{split_mode}_len{max_length}"
    return Path(base_output_dir) / cache_name


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute hash of configuration to detect changes.

    Args:
        config: Configuration dictionary

    Returns:
        Hash string
    """
    # Sort keys for consistent hashing
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def save_cached_datasets(
    cache_dir: Path,
    train_tensors: Dict[str, torch.Tensor],
    val_tensors: Dict[str, torch.Tensor],
    test_tensors: Optional[Dict[str, torch.Tensor]],
    metadata: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Save preprocessed datasets to cache directory.

    Args:
        cache_dir: Directory to save cached data
        train_tensors: Dictionary of train tensors (input_ids, attention_mask, etc.)
        val_tensors: Dictionary of validation tensors
        test_tensors: Dictionary of test tensors (optional)
        metadata: Metadata about the datasets
        config: Dataset configuration
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save train dataset
    print(f"Saving train dataset ({len(train_tensors['input_ids'])} samples)...")
    torch.save(train_tensors, cache_dir / "train_dataset.pt")

    # Save val dataset
    print(f"Saving val dataset ({len(val_tensors['input_ids'])} samples)...")
    torch.save(val_tensors, cache_dir / "val_dataset.pt")

    # Save test dataset if provided
    if test_tensors is not None:
        print(f"Saving test dataset ({len(test_tensors['input_ids'])} samples)...")
        torch.save(test_tensors, cache_dir / "test_dataset.pt")

    # Add config hash and timestamp to metadata
    metadata['config'] = config
    metadata['config_hash'] = compute_config_hash(config)
    metadata['created_at'] = datetime.now().isoformat()
    metadata['cache_version'] = '1.0'

    # Save metadata
    with open(cache_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Cached datasets saved to: {cache_dir}")
    print(f"  - train_dataset.pt: {(cache_dir / 'train_dataset.pt').stat().st_size / 1e9:.2f} GB")
    print(f"  - val_dataset.pt: {(cache_dir / 'val_dataset.pt').stat().st_size / 1e9:.2f} GB")
    if test_tensors is not None:
        print(f"  - test_dataset.pt: {(cache_dir / 'test_dataset.pt').stat().st_size / 1e9:.2f} GB")


def load_cached_metadata(cache_dir: Path) -> Dict[str, Any]:
    """
    Load metadata from cache directory.

    Args:
        cache_dir: Cache directory path

    Returns:
        Metadata dictionary

    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    with open(metadata_path, 'r') as f:
        return json.load(f)


def verify_cache(
    cache_dir: Path,
    expected_config: Optional[Dict[str, Any]] = None,
    splits: Optional[List[str]] = None,
) -> bool:
    """
    Verify that cache exists and is valid.

    Args:
        cache_dir: Cache directory path
        expected_config: Expected dataset configuration (optional)
        splits: List of required splits (e.g., ['train', 'val', 'test'])

    Returns:
        True if cache is valid, False otherwise
    """
    # Check if cache directory exists
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return False

    # Check if metadata exists
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}")
        return False

    # Load metadata
    try:
        metadata = load_cached_metadata(cache_dir)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return False

    # Verify required splits exist
    if splits is None:
        splits = ['train', 'val']  # Default required splits

    for split in splits:
        split_path = cache_dir / f"{split}_dataset.pt"
        if not split_path.exists():
            print(f"Missing cached dataset: {split_path}")
            return False

    # Verify config matches if provided
    if expected_config is not None:
        expected_hash = compute_config_hash(expected_config)
        cached_hash = metadata.get('config_hash', '')

        if expected_hash != cached_hash:
            print("Cache config mismatch:")
            print(f"  Expected: {expected_config}")
            print(f"  Cached: {metadata.get('config', {})}")
            return False

    return True


def load_cached_dataset(
    cache_dir: Path,
    split: str = 'train',
) -> Dict[str, torch.Tensor]:
    """
    Load a cached dataset split.

    Args:
        cache_dir: Cache directory path
        split: Dataset split ('train', 'val', or 'test')

    Returns:
        Dictionary of tensors (input_ids, attention_mask, labels, etc.)

    Raises:
        FileNotFoundError: If cached dataset doesn't exist
    """
    dataset_path = cache_dir / f"{split}_dataset.pt"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Cached {split} dataset not found at {dataset_path}")

    print(f"Loading {split} dataset from cache...")
    tensors = torch.load(dataset_path)
    print(f"  ✓ Loaded {len(tensors['input_ids'])} samples")

    return tensors


def print_cache_info(cache_dir: Path) -> None:
    """
    Print information about cached datasets.

    Args:
        cache_dir: Cache directory path
    """
    if not cache_dir.exists():
        print(f"Cache not found: {cache_dir}")
        return

    print(f"\n{'='*80}")
    print(f"CACHED DATASET INFORMATION")
    print(f"{'='*80}")
    print(f"Cache directory: {cache_dir}")
    print()

    # Load and print metadata
    try:
        metadata = load_cached_metadata(cache_dir)

        print("Metadata:")
        print(f"  Model size: {metadata.get('model_size', 'unknown')}")
        print(f"  Split mode: {metadata.get('split_mode', 'unknown')}")
        print(f"  Vocab size: {metadata.get('vocab_size', 'unknown')}")
        if 'vocab_hash' in metadata:
            vocab_hash = metadata['vocab_hash']
            print(f"  Vocab hash: {vocab_hash[:16]}...")
        if 'mode' in metadata:
            print(f"  Mode: {metadata['mode']}")
        print(f"  Max length: {metadata.get('max_length', 'unknown')}")
        print(f"  Created: {metadata.get('created_at', 'unknown')}")
        print()

        print("Dataset sizes:")
        for split in ['train', 'val', 'test']:
            count_key = f'{split}_samples'
            if count_key in metadata:
                print(f"  {split.capitalize()}: {metadata[count_key]:,} samples")

        print()
        print("Files:")
        for split_file in ['train_dataset.pt', 'val_dataset.pt', 'test_dataset.pt']:
            split_path = cache_dir / split_file
            if split_path.exists():
                size_gb = split_path.stat().st_size / 1e9
                print(f"  {split_file}: {size_gb:.2f} GB")

    except Exception as e:
        print(f"Error reading cache: {e}")

    print(f"{'='*80}\n")
