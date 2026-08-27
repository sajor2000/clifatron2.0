#!/usr/bin/env python3
"""
Method 1: Embedding-based Benchmark for Task 3: IMV Status - 72hr (Multiclass)

This script evaluates AR models using extracted embeddings:
1. Loads AR model (gpt2, gpt2_hf, or qwen2)
2. Extracts hidden state embeddings for each narrative
3. Trains XGBoost classifier on embeddings
4. Predicts multiclass outcome: IMV status (imv_off, expired, imv_on)
5. Computes and saves evaluation metrics

Embeddings are cached to disk for faster re-runs. Use --overwrite-embeddings to force re-extraction.

Usage:
    uv run run_embedding_benchmark.py --model-type gpt2_hf --batch-size 8
"""

import argparse
import logging
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
from tqdm import tqdm

# Add benchmark directory to path for shared utils imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.data_loader import load_benchmark_dataset
from utils.metrics import compute_metrics, save_results, print_metrics_summary
from utils.model_loader import ModelLoader, get_model_embeddings, load_vocabulary, get_default_vocab_path
from utils.gpu_utils import (
    setup_distributed,
    cleanup_distributed,
    get_device_strategy,
    get_rank,
    is_main_process,
)
from utils.results_writer import get_results_writer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_default_checkpoint_path(model_type: str, model_size: str = "small") -> Optional[Path]:
    """
    Auto-detect checkpoint path based on model type.

    Args:
        model_type: Model type (gpt2, gpt2_hf, qwen2)
        model_size: Model size (small, medium, etc.)

    Returns:
        Path to model checkpoint or None if not found
    """
    project_root = Path(__file__).parent.parent.parent.parent

    # Search common checkpoint locations
    search_paths = [
        project_root / "models" / model_type / "model_weights",
        project_root / "models" / model_type / "checkpoints",
        project_root / "models" / model_type,
    ]

    for path in search_paths:
        if path.exists():
            logger.info(f"Auto-detected checkpoint path: {path}")
            return path

    logger.warning(f"Could not auto-detect checkpoint path for model_type={model_type}")
    return None


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Pad sequences to max length in batch
    sequences = [item["sequence"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
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
        Tuple of (embeddings, labels)
    """
    logger.info(f"Extracting embeddings from {len(dataset)} examples...")

    model.eval()
    all_embeddings = []
    all_labels = []

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
            labels = batch["label"].numpy()

            # Extract embeddings
            embeddings = get_model_embeddings(model, sequences, attention_mask=attention_mask, layer=layer)

            # Store
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels)

    # Concatenate
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)

    logger.info(f"Extracted embeddings shape: {embeddings.shape} (rank {get_rank()})")

    return embeddings, labels


def train_classifier(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    config: Optional[dict] = None,
):
    """
    Train XGBoost multiclass classifier on embeddings.

    Args:
        train_embeddings: Training embeddings
        train_labels: Training labels (encoded as integers 0, 1, 2)
        config: Classifier configuration

    Returns:
        Trained XGBoost classifier
    """
    import xgboost as xgb

    if config is None:
        config = {}

    logger.info("Training XGBoost multiclass classifier...")

    clf_config = config.get("xgboost", {})

    # Log class distribution
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    logger.info(f"Class distribution:")
    for label, count in zip(unique_labels, counts):
        logger.info(f"  Class {label}: {count} examples")

    classifier = xgb.XGBClassifier(
        max_depth=clf_config.get("max_depth", 6),
        learning_rate=clf_config.get("learning_rate", 0.1),
        n_estimators=clf_config.get("n_estimators", 100),
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    classifier.fit(
        train_embeddings,
        train_labels,
        verbose=clf_config.get("verbose", False)
    )

    logger.info(f"XGBoost feature importance shape: {classifier.feature_importances_.shape}")
    logger.info("Classifier training complete")

    return classifier


def evaluate_classifier(
    classifier,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
):
    """
    Evaluate classifier on test set.

    Args:
        classifier: Trained classifier
        test_embeddings: Test embeddings
        test_labels: Test labels

    Returns:
        Dictionary with predictions and probabilities
    """
    logger.info("Evaluating classifier on test set...")

    # Get predictions
    predictions = classifier.predict(test_embeddings)

    # For multiclass, we don't use a single probability column
    # Store the full probability matrix for potential future use
    probabilities = None
    if hasattr(classifier, "predict_proba"):
        # For multiclass: shape (n_samples, n_classes)
        # We'll leave it as None since compute_metrics expects binary probs
        # The multiclass metrics will be computed differently
        probabilities = None

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "labels": test_labels,
    }


def get_embedding_cache_path(
    data_dir: Path,
    model_type: str,
    model_size: str,
    layer: str,
    split: str,
) -> Path:
    """
    Get path to cached embeddings file.

    Args:
        data_dir: Data directory
        model_type: Model type (gpt2, gpt2_hf, qwen2)
        model_size: Model size (small, medium, etc.)
        layer: Layer type (last, mean)
        split: Data split (train_val, test)

    Returns:
        Path to embeddings cache file
    """
    # Use the standard embeddings directory relative to project root
    project_root = Path(__file__).parent.parent.parent.parent
    embeddings_dir = project_root / "benchmark" / "embeddings" / "task3_task4_respiratory"
    filename = f"embeddings_{model_type}_{layer}_{split}.npz"
    return embeddings_dir / filename


def save_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cache_path: Path,
):
    """
    Save embeddings and labels to cache file.

    Args:
        embeddings: Embeddings array
        labels: Labels array
        cache_path: Path to save cache file
    """
    logger.info(f"Saving embeddings to {cache_path}")
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        labels=labels,
    )
    logger.info(f"Embeddings saved successfully")


def load_embeddings(cache_path: Path, data_path: Path, label_column: str = "task3_label") -> tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings from cache file and labels from data file.

    This function loads pre-generated embeddings from generate_embeddings.py
    and matches them with labels from the original data file.

    Args:
        cache_path: Path to cached embeddings file (from generate_embeddings.py)
        data_path: Path to data parquet file containing labels
        label_column: Name of label column in data file (default: "task3_label" for Task3)

    Returns:
        Tuple of (embeddings, encoded_labels)
    """
    import pandas as pd

    logger.info(f"Loading cached embeddings from {cache_path}")
    data = np.load(cache_path)
    embeddings = data["embeddings"]
    example_ids = data["example_ids"]
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")

    # Load labels from data file
    logger.info(f"Loading labels from {data_path}")
    df = pd.read_parquet(data_path)

    # Match labels to embeddings using hospitalization_id (example_id)
    df_indexed = df.set_index('hospitalization_id')
    labels = df_indexed.loc[example_ids][label_column].values

    # Encode string labels to integers for Task3 (multiclass IMV status)
    # imv_off -> 0, expired -> 1, imv_on -> 2
    label_mapping = {"imv_off": 0, "expired": 1, "imv_on": 2}
    encoded_labels = np.array([label_mapping[label] for label in labels])

    logger.info(f"Loaded {len(labels)} labels, shape: {labels.shape}")
    logger.info(f"Label encoding: {label_mapping}")
    return embeddings, encoded_labels


def main():
    """Main function for embedding-based benchmark."""
    parser = argparse.ArgumentParser(
        description="Method 1: Embedding-based benchmark for discharged home prediction"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["gpt2", "gpt2_hf", "qwen2"],
        help="Type of AR model to use",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="benchmark/data",
        help="Directory containing processed benchmark data",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (auto-constructed if not specified)",
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
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "config.yaml"),
        help="Path to configuration file",
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

    args = parser.parse_args()

    # Auto-detect checkpoint path if not provided
    if args.checkpoint_path is None:
        detected_checkpoint = get_default_checkpoint_path(args.model_type, args.model_size)
        if detected_checkpoint:
            args.checkpoint_path = str(detected_checkpoint)

    # Setup distributed training if available
    use_distributed = setup_distributed(backend="nccl" if args.device != "cpu" else "gloo")

    # Get device strategy
    device_strategy = get_device_strategy(auto_detect=True, fallback_cpu=True)

    # Auto-construct output directory using ResultsWriter if not specified
    use_results_writer = False
    results_writer = None
    if args.output_dir is None:
        results_writer = get_results_writer(
            task_name="task3-outcome-72hr",
            model_type=args.model_type,
            method_name="method1-embedding"
        )
        output_dir = results_writer.result_dir
        use_results_writer = True
    else:
        output_dir = Path(args.output_dir).resolve()

    # Convert to Path objects and resolve to absolute paths
    input_dir = Path(args.input_dir).resolve()
    config_path = Path(args.config)

    if is_main_process():
        logger.info("=" * 80)
        logger.info("Method 1: Embedding-based Benchmark (XGBoost Classifier)")
        logger.info("=" * 80)
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Model size: {args.model_size}")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Distributed: {use_distributed} (world_size={device_strategy.world_size})")
        logger.info(f"Device strategy: {device_strategy.device_type} (rank={device_strategy.rank})")

    # Load configuration
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Create output directory
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached embeddings
    train_cache_path = get_embedding_cache_path(
        input_dir, args.model_type, args.model_size, args.layer, "train_val"
    )
    test_cache_path = get_embedding_cache_path(
        input_dir, args.model_type, args.model_size, args.layer, "test"
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

    # Load cached embeddings (only on main process)
    # NOTE: Embeddings must be pre-generated using benchmark/generate_embeddings.py
    # This script does NOT extract embeddings from the model
    if is_main_process():
        if not use_cached_embeddings:
            logger.error("=" * 80)
            logger.error("CACHED EMBEDDINGS REQUIRED")
            logger.error("=" * 80)
            logger.error("This script requires pre-generated embeddings from generate_embeddings.py")
            logger.error("Cached embeddings not found at:")
            logger.error(f"  Train: {train_cache_path}")
            logger.error(f"  Test: {test_cache_path}")
            logger.error("")
            logger.error("Please run generate_embeddings.py first:")
            logger.error("  uv run benchmark/generate_embeddings.py \\")
            logger.error(f"    --model-type {args.model_type} \\")
            logger.error("    --task task3_task4 \\")
            logger.error(f"    --layer {args.layer}")
            logger.error("=" * 80)
            cleanup_distributed()
            sys.exit(1)

        try:
            # Define data file paths
            train_data_path = input_dir / "task3_task4_respiratory_train_val.parquet"
            test_data_path = input_dir / "task3_task4_respiratory_test.parquet"

            # Load embeddings and labels from cache + data files
            train_embeddings, train_labels = load_embeddings(
                train_cache_path,
                train_data_path,
                label_column="task3_label"  # Task1: Discharged Home
            )
            test_embeddings, test_labels = load_embeddings(
                test_cache_path,
                test_data_path,
                label_column="task3_label"  # Task1: Discharged Home
            )

            logger.info("=" * 80)
            logger.info("Successfully loaded embeddings and labels")
            logger.info(f"Train embeddings: {train_embeddings.shape}")
            logger.info(f"Train labels: {train_labels.shape}")
            logger.info(f"Test embeddings: {test_embeddings.shape}")
            logger.info(f"Test labels: {test_labels.shape}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to load embeddings and labels: {e}", exc_info=True)
            logger.error("=" * 80)
            logger.error("Make sure you have run generate_embeddings.py first!")
            logger.error("=" * 80)
            cleanup_distributed()
            sys.exit(1)

    # Train classifier and evaluate (only on main process)
    if is_main_process():
        try:
            classifier_config = config.get("method1_embedding", {}).get("classifier", {})

            classifier = train_classifier(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                config=classifier_config,
            )

        except Exception as e:
            logger.error(f"Failed to train classifier: {e}", exc_info=True)
            cleanup_distributed()
            sys.exit(1)

        # Evaluate
        try:
            eval_results = evaluate_classifier(
                classifier=classifier,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
            )

            # Compute multiclass metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

            y_true = eval_results["labels"]
            y_pred = eval_results["predictions"]

            accuracy = accuracy_score(y_true, y_pred)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(y_true, y_pred, average=None)
            cm = confusion_matrix(y_true, y_pred)

            metrics = {
                "accuracy": float(accuracy),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
                "precision_weighted": float(precision_weighted),
                "recall_weighted": float(recall_weighted),
                "f1_weighted": float(f1_weighted),
                "confusion_matrix": cm.tolist(),
                "per_class_metrics": {
                    "class_0_imv_off": {
                        "precision": float(precision_per_class[0]),
                        "recall": float(recall_per_class[0]),
                        "f1": float(f1_per_class[0]),
                        "support": int(support[0]),
                    },
                    "class_1_expired": {
                        "precision": float(precision_per_class[1]),
                        "recall": float(recall_per_class[1]),
                        "f1": float(f1_per_class[1]),
                        "support": int(support[1]),
                    },
                    "class_2_imv_on": {
                        "precision": float(precision_per_class[2]),
                        "recall": float(recall_per_class[2]),
                        "f1": float(f1_per_class[2]),
                        "support": int(support[2]),
                    },
                },
            }

            # Print multiclass metrics summary
            logger.info("=" * 60)
            logger.info("MULTICLASS METRICS SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Accuracy:            {metrics['accuracy']:.4f}")
            logger.info(f"Macro Precision:     {metrics['precision_macro']:.4f}")
            logger.info(f"Macro Recall:        {metrics['recall_macro']:.4f}")
            logger.info(f"Macro F1:            {metrics['f1_macro']:.4f}")
            logger.info(f"Weighted Precision:  {metrics['precision_weighted']:.4f}")
            logger.info(f"Weighted Recall:     {metrics['recall_weighted']:.4f}")
            logger.info(f"Weighted F1:         {metrics['f1_weighted']:.4f}")
            logger.info("")
            logger.info("Per-Class Metrics:")
            logger.info(f"  IMV Off (0):  P={metrics['per_class_metrics']['class_0_imv_off']['precision']:.4f}, R={metrics['per_class_metrics']['class_0_imv_off']['recall']:.4f}, F1={metrics['per_class_metrics']['class_0_imv_off']['f1']:.4f}, N={metrics['per_class_metrics']['class_0_imv_off']['support']}")
            logger.info(f"  Expired (1):  P={metrics['per_class_metrics']['class_1_expired']['precision']:.4f}, R={metrics['per_class_metrics']['class_1_expired']['recall']:.4f}, F1={metrics['per_class_metrics']['class_1_expired']['f1']:.4f}, N={metrics['per_class_metrics']['class_1_expired']['support']}")
            logger.info(f"  IMV On  (2):  P={metrics['per_class_metrics']['class_2_imv_on']['precision']:.4f}, R={metrics['per_class_metrics']['class_2_imv_on']['recall']:.4f}, F1={metrics['per_class_metrics']['class_2_imv_on']['f1']:.4f}, N={metrics['per_class_metrics']['class_2_imv_on']['support']}")
            logger.info("=" * 60)

            # Save results
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else "default",
                "num_train_examples": len(train_embeddings),
                "num_test_examples": len(test_embeddings),
                "embedding_dim": train_embeddings.shape[1],
                "distributed": use_distributed,
                "world_size": device_strategy.world_size,
            }

            # Save using ResultsWriter or legacy method
            if use_results_writer:
                results_writer.save_results(
                    metrics=metrics,
                    metadata=metadata,
                    method="embedding",
                    model_size=args.model_size,
                    classifier="xgboost",
                    layer=args.layer
                )
                logger.info("=" * 80)
                logger.info("Benchmark completed successfully!")
                logger.info(f"Results saved to: {results_writer.get_summary_path()}")
                logger.info("=" * 80)
            else:
                # Legacy save method for backwards compatibility
                results = {
                    "method": "embedding",
                    "model_type": args.model_type,
                    "model_size": args.model_size,
                    "classifier": "xgboost",
                    "layer": args.layer,
                    "metrics": metrics,
                    "metadata": metadata,
                }
                output_file = (
                    output_dir
                    / f"method1_embedding_{args.model_type}_{args.model_size}_results.json"
                )
                save_results(results, output_file)
                logger.info("=" * 80)
                logger.info("Benchmark completed successfully!")
                logger.info(f"Results saved to: {output_file}")
                logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to evaluate: {e}", exc_info=True)
            cleanup_distributed()
            sys.exit(1)

    # Cleanup distributed
    cleanup_distributed()


if __name__ == "__main__":
    main()
