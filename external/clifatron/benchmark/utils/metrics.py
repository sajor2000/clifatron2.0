"""
Metrics calculation utilities for benchmarking

Provides functions to compute classification metrics including
accuracy, precision, recall, F1, AUROC, AUPRC, and confusion matrix.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        roc_curve,
        precision_recall_curve,
        confusion_matrix as sklearn_confusion_matrix,
        classification_report,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("sklearn not available. Some metrics will be computed manually.")
    SKLEARN_AVAILABLE = False


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    specificity: Optional[float] = None
    npv: Optional[float] = None  # Negative predictive value

    # Raw counts
    tp: int = 0  # True positives
    tn: int = 0  # True negatives
    fp: int = 0  # False positives
    fn: int = 0  # False negatives

    # Total examples
    total: int = 0
    positive_examples: int = 0
    negative_examples: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCalculator:
    """Calculate classification metrics."""

    def __init__(self, use_sklearn: bool = True):
        """
        Initialize metrics calculator.

        Args:
            use_sklearn: Whether to use sklearn for metric calculation
        """
        self.use_sklearn = use_sklearn and SKLEARN_AVAILABLE

    def compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """
        Compute confusion matrix elements.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Tuple of (TP, TN, FP, FN)
        """
        if self.use_sklearn:
            cm = sklearn_confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
        else:
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

        return int(tp), int(tn), int(fp), int(fn)

    def compute_metrics(
        self,
        y_true: Union[List, np.ndarray],
        y_pred: Union[List, np.ndarray],
        y_prob: Optional[Union[List, np.ndarray]] = None,
    ) -> BenchmarkMetrics:
        """
        Compute all benchmark metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for AUROC/AUPRC)

        Returns:
            BenchmarkMetrics object
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_prob is not None:
            y_prob = np.array(y_prob)

        # Compute confusion matrix
        tp, tn, fp, fn = self.compute_confusion_matrix(y_true, y_pred)

        # Compute basic metrics
        if self.use_sklearn:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            # Manual calculation
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

        # Compute specificity and NPV
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Compute AUROC and AUPRC if probabilities provided
        auroc = None
        auprc = None
        if y_prob is not None and self.use_sklearn:
            try:
                # Check if we have both classes in y_true
                if len(np.unique(y_true)) > 1:
                    auroc = roc_auc_score(y_true, y_prob)
                    auprc = average_precision_score(y_true, y_prob)
                else:
                    logger.warning(
                        "Only one class present in y_true. Cannot compute AUROC/AUPRC."
                    )
            except Exception as e:
                logger.warning(f"Error computing AUROC/AUPRC: {e}")

        # Create metrics object
        metrics = BenchmarkMetrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            auroc=float(auroc) if auroc is not None else None,
            auprc=float(auprc) if auprc is not None else None,
            specificity=float(specificity),
            npv=float(npv),
            tp=tp,
            tn=tn,
            fp=fp,
            fn=fn,
            total=int(len(y_true)),
            positive_examples=int(np.sum(y_true == 1)),
            negative_examples=int(np.sum(y_true == 0)),
        )

        return metrics


def compute_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
) -> Dict:
    """
    Compute confusion matrix and return as dictionary.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with confusion matrix and derived metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if SKLEARN_AVAILABLE:
        cm = sklearn_confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

    return {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total": int(tp + tn + fp + fn),
    }


def compute_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_prob: Optional[Union[List, np.ndarray]] = None,
) -> Dict:
    """
    Compute all metrics and return as dictionary.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dictionary with all computed metrics
    """
    calculator = MetricsCalculator()
    metrics = calculator.compute_metrics(y_true, y_pred, y_prob)
    result = metrics.to_dict()

    # Add ROC and PR curve data if probabilities provided
    if y_prob is not None and SKLEARN_AVAILABLE:
        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)

        # Only compute curves if we have both classes
        if len(np.unique(y_true_arr)) > 1:
            try:
                # Compute ROC curve
                fpr, tpr, roc_thresholds = roc_curve(y_true_arr, y_prob_arr)
                result["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist()
                }

                # Compute Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
                result["pr_curve"] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": pr_thresholds.tolist()
                }
            except Exception as e:
                logger.warning(f"Error computing ROC/PR curves: {e}")

    return result


def save_results(
    results: Dict,
    output_path: Union[str, Path],
    include_metadata: bool = True,
    timestamp: bool = True,
) -> None:
    """
    Save benchmark results to JSON file.

    Args:
        results: Results dictionary
        output_path: Path to save results
        include_metadata: Whether to include metadata
        timestamp: Whether to add timestamp
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata if requested
    if include_metadata:
        if "metadata" not in results:
            results["metadata"] = {}

        if timestamp:
            results["metadata"]["timestamp"] = datetime.now().isoformat()

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def load_results(file_path: Union[str, Path]) -> Dict:
    """
    Load benchmark results from JSON file.

    Args:
        file_path: Path to results file

    Returns:
        Results dictionary
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {file_path}")

    with open(file_path, "r") as f:
        results = json.load(f)

    return results


def print_metrics_summary(metrics: Union[Dict, BenchmarkMetrics]) -> None:
    """
    Print a summary of metrics.

    Args:
        metrics: Metrics dictionary or BenchmarkMetrics object
    """
    if isinstance(metrics, BenchmarkMetrics):
        metrics = metrics.to_dict()

    print("\n" + "=" * 60)
    print("BENCHMARK METRICS SUMMARY")
    print("=" * 60)

    # Basic metrics
    print(f"\nAccuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1 Score:   {metrics['f1']:.4f}")

    if metrics.get("auroc") is not None:
        print(f"AUROC:      {metrics['auroc']:.4f}")
    if metrics.get("auprc") is not None:
        print(f"AUPRC:      {metrics['auprc']:.4f}")

    if metrics.get("specificity") is not None:
        print(f"Specificity: {metrics['specificity']:.4f}")
    if metrics.get("npv") is not None:
        print(f"NPV:        {metrics['npv']:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  True Negatives:  {metrics['tn']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")

    # Class distribution
    print("\nClass Distribution:")
    print(f"  Total Examples:     {metrics['total']}")
    print(f"  Positive Examples:  {metrics['positive_examples']}")
    print(f"  Negative Examples:  {metrics['negative_examples']}")

    print("=" * 60 + "\n")


def compare_results(
    results_list: List[Dict],
    model_names: List[str],
    metric_name: str = "f1",
) -> None:
    """
    Compare results from multiple models.

    Args:
        results_list: List of result dictionaries
        model_names: List of model names
        metric_name: Name of metric to compare
    """
    print("\n" + "=" * 60)
    print(f"MODEL COMPARISON - {metric_name.upper()}")
    print("=" * 60)

    for model_name, results in zip(model_names, results_list):
        metrics = results.get("metrics", results)
        metric_value = metrics.get(metric_name, "N/A")

        if isinstance(metric_value, float):
            print(f"{model_name:20s}: {metric_value:.4f}")
        else:
            print(f"{model_name:20s}: {metric_value}")

    print("=" * 60 + "\n")
