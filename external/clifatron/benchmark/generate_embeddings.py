#!/usr/bin/env python3
"""
Generate Embeddings for Tasks 1 & 2 (Shared Data)

This script extracts embeddings from AR models for the shared disposition data.
The same embeddings can be used for both Task 1 (discharged home) and Task 2 (LTACH).

1. Loads shared data files (task1_task2_disposition_{train_val,test}.parquet)
2. Extracts hidden state embeddings using specified model type (gpt2, gpt2_hf, qwen2)
3. Saves embeddings to NPZ files for later use

Embeddings are cached to disk. Use --overwrite-embeddings to force re-extraction.

Usage:
    uv run benchmark/generate_embeddings.py --model-type gpt2_hf --batch-size 8
    uv run benchmark/generate_embeddings.py --model-type gpt2 --model-size small --batch-size 8
"""

import argparse
import logging
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from tqdm import tqdm

# Add benchmark directory to path for shared utils imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import load_benchmark_dataset
from utils.model_loader import ModelLoader, get_model_embeddings, load_vocabulary, get_default_vocab_path
from utils.gpu_utils import (
    setup_distributed,
    cleanup_distributed,
    get_device_strategy,
    get_rank,
    is_main_process,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Pad sequences to max length in batch
    sequences = [item["sequence"] for item in batch]

    # Handle labels - can be numeric or string
    raw_labels = [item["label"] for item in batch]
    if isinstance(raw_labels[0], str):
        # String labels (e.g., Task 3: "imv_on", "expired", "imv_off")
        # Keep as list of strings, don't convert to tensor
        labels = raw_labels
    else:
        # Numeric labels (e.g., Task 1, 2, 4)
        labels = torch.tensor(raw_labels, dtype=torch.long)

    dispositions = [item["disposition"] for item in batch]
    example_ids = [item["example_id"] for item in batch]
    lengths = [item["length"] for item in batch]

    # Pad sequences and create attention masks
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = torch.zeros((len(sequences), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded_sequences[i, :seq_len] = seq
        attention_mask[i, :seq_len] = 1  # 1 for real tokens, 0 for padding

    return {
        "sequence": padded_sequences,
        "attention_mask": attention_mask,
        "label": labels,
        "disposition": dispositions,
        "example_id": example_ids,
        "length": lengths,
    }


def extract_embeddings(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 16,
    layer: str = "last",
    use_distributed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from model for all examples.

    Args:
        model: The AR model
        dataset: Benchmark dataset
        device: Compute device
        batch_size: Batch size for processing
        layer: Which layer to extract from
        use_distributed: Whether to use distributed sampling

    Returns:
        Tuple of (embeddings, example_ids)
    """
    logger.info(f"Extracting embeddings from {len(dataset)} examples...")

    model.eval()
    all_embeddings = []
    all_example_ids = []

    # Create dataloader with optional distributed sampler
    sampler = None
    if use_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=False,
        )
        logger.info(f"Using distributed sampler (rank {get_rank()})")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler else False,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", disable=not is_main_process()):
            # Get input sequences and attention mask
            sequences = batch["sequence"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            example_ids = batch["example_id"]

            # Extract embeddings
            embeddings = get_model_embeddings(model, sequences, attention_mask=attention_mask, layer=layer)

            # Store
            all_embeddings.append(embeddings.cpu().numpy())
            all_example_ids.extend(example_ids)

    # Concatenate
    embeddings = np.vstack(all_embeddings)
    example_ids = np.array(all_example_ids)

    logger.info(f"Extracted embeddings shape: {embeddings.shape} (rank {get_rank()})")

    return embeddings, example_ids


def get_embedding_cache_path(
    data_dir: Path,
    model_name: str,
    layer: str,
    split: str,
    task: str = "task1_task2",
) -> Path:
    """
    Get path to cached embeddings file.

    Args:
        data_dir: Data directory (benchmark/data)
        model_name: Name of the model (extracted from checkpoint path, e.g., "qwen2optuna", "gpt2_hf")
        layer: Layer type (last, mean)
        split: Data split (train_val, test)
        task: Task identifier (task1_task2, task3_task4)

    Returns:
        Path to embeddings cache file in organized folder structure
    """
    # Determine task-specific subfolder
    embeddings_base = data_dir.parent / "embeddings"
    if task == "task1_task2":
        embeddings_dir = embeddings_base / "task1_task2_disposition"
    else:  # task3_task4
        embeddings_dir = embeddings_base / "task3_task4_respiratory"

    # Create directory if it doesn't exist
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    filename = f"embeddings_{model_name}_{layer}_{split}.npz"
    return embeddings_dir / filename


def save_embeddings(
    embeddings: np.ndarray,
    example_ids: np.ndarray,
    cache_path: Path,
):
    """
    Save embeddings and example IDs to cache file.

    Args:
        embeddings: Embeddings array
        example_ids: Example IDs array
        cache_path: Path to save cache file
    """
    logger.info(f"Saving embeddings to {cache_path}")
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        example_ids=example_ids,
    )
    logger.info(f"Embeddings saved successfully")


def load_embeddings(cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings and example IDs from cache file.

    Args:
        cache_path: Path to cache file

    Returns:
        Tuple of (embeddings, example_ids)
    """
    logger.info(f"Loading cached embeddings from {cache_path}")
    data = np.load(cache_path)
    embeddings = data["embeddings"]
    example_ids = data["example_ids"]
    logger.info(f"Loaded embeddings shape: {embeddings.shape}, example_ids shape: {example_ids.shape}")
    return embeddings, example_ids


def main():
    """Main function for embedding generation."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for Tasks 1 & 2 (shared data)"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["gpt2", "gpt2_hf", "qwen2", "qwen2optuna"],
        help="Type of AR model to use",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="task1_task2",
        choices=["task1_task2", "task3_task4"],
        help="Which task data to process (task1_task2: disposition data, task3_task4: respiratory data)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="benchmark/data",
        help="Directory containing shared benchmark data",
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to model checkpoint (optional, uses default if not specified)",
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        help="Model size (small, medium, etc.)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding extraction",
    )

    parser.add_argument(
        "--layer",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="Which layer to extract embeddings from",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation",
    )

    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Force re-extraction of embeddings even if cached embeddings exist",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    # Setup distributed training if available
    use_distributed = setup_distributed(backend="nccl" if args.device != "cpu" else "gloo")

    # Get device strategy
    device_strategy = get_device_strategy(auto_detect=True, fallback_cpu=True)

    # Convert to Path objects
    data_dir = Path(args.data_dir)

    # Extract model_name from checkpoint_path
    # e.g., /home/vchaudha/CLIFATRON/models/qwen2optuna/model_weights -> qwen2optuna
    # e.g., /home/vchaudha/CLIFATRON/models/gpt2_hf/checkpoints -> gpt2_hf
    model_name = None
    if args.checkpoint_path is not None:
        checkpoint_path_parts = Path(args.checkpoint_path).parts
        # Look for the part right after "models"
        for i, part in enumerate(checkpoint_path_parts):
            if part == "models" and i + 1 < len(checkpoint_path_parts):
                model_name = checkpoint_path_parts[i + 1]
                break

    # Fallback to model_type if model_name not found
    if model_name is None:
        model_name = args.model_type

    task_description = "Tasks 1 & 2 (Disposition)" if args.task == "task1_task2" else "Tasks 3 & 4 (Respiratory)"

    if is_main_process():
        logger.info("=" * 80)
        logger.info(f"Generate Embeddings for {task_description}")
        logger.info("=" * 80)
        logger.info(f"Task: {args.task}")
        logger.info(f"Model name: {model_name}")
        logger.info(f"Checkpoint path: {args.checkpoint_path or 'default'}")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Distributed: {use_distributed} (world_size={device_strategy.world_size})")
        logger.info(f"Device strategy: {device_strategy.device_type} (rank={device_strategy.rank})")

    # Check for cached embeddings
    train_cache_path = get_embedding_cache_path(
        data_dir, model_name, args.layer, "train_val", task=args.task
    )
    test_cache_path = get_embedding_cache_path(
        data_dir, model_name, args.layer, "test", task=args.task
    )

    use_cached_embeddings = (
        not args.overwrite_embeddings
        and train_cache_path.exists()
        and test_cache_path.exists()
    )

    if use_cached_embeddings and is_main_process():
        logger.info("=" * 80)
        logger.info("Found cached embeddings - loading from cache")
        logger.info(f"Train cache: {train_cache_path}")
        logger.info(f"Test cache: {test_cache_path}")
        logger.info("Use --overwrite-embeddings to force re-extraction")
        logger.info("=" * 80)

    # Load or extract embeddings
    if use_cached_embeddings and is_main_process():
        # Load cached embeddings (only on main process)
        try:
            train_embeddings, train_example_ids = load_embeddings(train_cache_path)
            test_embeddings, test_example_ids = load_embeddings(test_cache_path)

            logger.info("=" * 80)
            logger.info("Embeddings loaded successfully!")
            logger.info(f"Train embeddings: {train_embeddings.shape}")
            logger.info(f"Test embeddings: {test_embeddings.shape}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to load cached embeddings: {e}", exc_info=True)
            logger.info("Will re-extract embeddings...")
            use_cached_embeddings = False

    if not use_cached_embeddings:
        # Need to extract embeddings - load vocabulary, model, and datasets
        if is_main_process():
            logger.info("=" * 80)
            logger.info("Extracting embeddings from model")
            logger.info("=" * 80)

        # Load vocabulary
        if is_main_process():
            logger.info("Loading vocabulary...")
        vocab_path = get_default_vocab_path(args.model_type)
        token_to_id, id_to_token = load_vocabulary(vocab_path)
        logger.info(f"Loaded {len(token_to_id)} tokens from {vocab_path}")

        # Load model
        if is_main_process():
            logger.info("Loading model...")
        try:
            # Use device strategy's primary device
            model_loader = ModelLoader(
                model_type=args.model_type,
                device=str(device_strategy.primary_device),
            )

            model = model_loader.load_model(
                checkpoint_path=args.checkpoint_path,
                model_size=args.model_size,
            )

            device = device_strategy.primary_device

            # Wrap model with DDP if distributed
            if use_distributed and device_strategy.device_type == "cuda":
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[device_strategy.local_rank],
                )
                if is_main_process():
                    logger.info("Model wrapped with DistributedDataParallel")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            cleanup_distributed()
            sys.exit(1)

        # Load datasets based on task
        logger.info(f"Loading benchmark datasets for {args.task}...")
        try:
            if args.task == "task1_task2":
                train_file = data_dir / "task1_task2_disposition_train_val.parquet"
                test_file = data_dir / "task1_task2_disposition_test.parquet"
            else:  # task3_task4
                train_file = data_dir / "task3_task4_respiratory_train_val.parquet"
                test_file = data_dir / "task3_task4_respiratory_test.parquet"

            train_dataset = load_benchmark_dataset(
                train_file,
                vocab=token_to_id,
                max_length=args.max_length,
            )

            test_dataset = load_benchmark_dataset(
                test_file,
                vocab=token_to_id,
                max_length=args.max_length,
            )

            logger.info(f"Train dataset: {len(train_dataset)} examples")
            logger.info(f"Test dataset: {len(test_dataset)} examples")

        except Exception as e:
            logger.error(f"Failed to load datasets: {e}", exc_info=True)
            cleanup_distributed()
            sys.exit(1)

        # Extract embeddings
        try:
            train_embeddings, train_example_ids = extract_embeddings(
                model=model,
                dataset=train_dataset,
                device=device,
                batch_size=args.batch_size,
                layer=args.layer,
                use_distributed=use_distributed,
            )

            test_embeddings, test_example_ids = extract_embeddings(
                model=model,
                dataset=test_dataset,
                device=device,
                batch_size=args.batch_size,
                layer=args.layer,
                use_distributed=use_distributed,
            )

            # Gather embeddings from all processes if distributed
            if use_distributed:
                import torch.distributed as dist

                # Gather train embeddings and example_ids
                train_embeddings_list = [None] * device_strategy.world_size
                train_example_ids_list = [None] * device_strategy.world_size

                dist.all_gather_object(train_embeddings_list, train_embeddings)
                dist.all_gather_object(train_example_ids_list, train_example_ids)

                if is_main_process():
                    train_embeddings = np.vstack(train_embeddings_list)
                    train_example_ids = np.concatenate(train_example_ids_list)
                    logger.info(f"Gathered train embeddings shape: {train_embeddings.shape}")

                # Gather test embeddings and example_ids
                test_embeddings_list = [None] * device_strategy.world_size
                test_example_ids_list = [None] * device_strategy.world_size

                dist.all_gather_object(test_embeddings_list, test_embeddings)
                dist.all_gather_object(test_example_ids_list, test_example_ids)

                if is_main_process():
                    test_embeddings = np.vstack(test_embeddings_list)
                    test_example_ids = np.concatenate(test_example_ids_list)
                    logger.info(f"Gathered test embeddings shape: {test_embeddings.shape}")

            # Save embeddings to cache (only on main process)
            if is_main_process():
                try:
                    save_embeddings(train_embeddings, train_example_ids, train_cache_path)
                    save_embeddings(test_embeddings, test_example_ids, test_cache_path)

                    logger.info("=" * 80)
                    logger.info("Embeddings extracted and saved successfully!")
                    logger.info(f"Train embeddings: {train_embeddings.shape}")
                    logger.info(f"Test embeddings: {test_embeddings.shape}")
                    logger.info(f"Saved to: {data_dir}")
                    logger.info("=" * 80)
                    logger.info("These embeddings can now be used for both:")
                    logger.info("  - Task 1: Discharged Home prediction (label_home)")
                    logger.info("  - Task 2: LTACH prediction (label_ltach)")
                    logger.info("=" * 80)

                except Exception as e:
                    logger.warning(f"Failed to save embeddings to cache: {e}")

        except Exception as e:
            logger.error(f"Failed to extract embeddings: {e}", exc_info=True)
            cleanup_distributed()
            sys.exit(1)

    # Cleanup distributed
    cleanup_distributed()


if __name__ == "__main__":
    main()
