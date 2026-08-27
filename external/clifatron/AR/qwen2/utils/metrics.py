"""
metrics.py - Evaluation Metrics for Clinical Narrative Language Models

Provides comprehensive metrics for evaluating causal language models:
- Perplexity: Standard language modeling metric
- Token Accuracy: Next-token prediction accuracy
- Category-wise metrics: Metrics broken down by token categories (vitals, labs, meds, etc.)
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict
import torch.nn.functional as F


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value
    """
    try:
        perplexity = np.exp(loss)
        # Cap extremely high perplexity values
        if perplexity > 1e10:
            return float('inf')
        return float(perplexity)
    except OverflowError:
        return float('inf')


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Compute next-token prediction accuracy.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        ignore_index: Index to ignore (padding tokens)

    Returns:
        Token accuracy (fraction of correct predictions)
    """
    # Get predictions (argmax over vocabulary)
    predictions = torch.argmax(logits, dim=-1)

    # Create mask for non-ignored positions
    mask = (labels != ignore_index)

    # Count correct predictions
    correct = (predictions == labels) & mask
    total = mask.sum().item()

    if total == 0:
        return 0.0

    accuracy = correct.sum().item() / total
    return float(accuracy)


def compute_top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    ignore_index: int = -100
) -> float:
    """
    Compute top-k accuracy (correct token in top-k predictions).

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        k: Number of top predictions to consider
        ignore_index: Index to ignore (padding tokens)

    Returns:
        Top-k accuracy
    """
    # Get top-k predictions
    top_k_preds = torch.topk(logits, k, dim=-1).indices

    # Expand labels to match top-k shape
    labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)

    # Check if true label is in top-k
    mask = (labels != ignore_index)
    correct = (top_k_preds == labels_expanded).any(dim=-1) & mask
    total = mask.sum().item()

    if total == 0:
        return 0.0

    accuracy = correct.sum().item() / total
    return float(accuracy)


def compute_category_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: Any,
    ignore_index: int = -100
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by token categories.

    Categories:
    - vitals: vitals_*
    - labs: labs_*
    - medications: medications_*
    - respiratory_support: resp_*, respiratory_support_*
    - assessment: assessment_*
    - demographics: age_*, sex_*
    - location: transfer_to_*, disposition_*
    - elixhauser: elix_*, no_patient_history
    - special: PREV_NARRATIVE_START, PREV_NARRATIVE_END
    - time: day_*, hour_*

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target token IDs [batch_size, seq_len]
        tokenizer: Tokenizer with vocabulary
        ignore_index: Index to ignore

    Returns:
        Dictionary mapping category names to metrics (accuracy, count)
    """
    # Define category prefixes
    category_prefixes = {
        'vitals': ['vitals_'],
        'labs': ['labs_'],
        'medications': ['medications_'],
        'respiratory_support': ['resp_', 'respiratory_support_'],
        'assessment': ['assessment_'],
        'demographics': ['age_', 'sex_'],
        'location': ['transfer_to_', 'disposition_'],
        'elixhauser': ['elix_', 'no_patient_history'],
        'special': ['PREV_NARRATIVE_START', 'PREV_NARRATIVE_END', 'NARRATIVE_START', 'NARRATIVE_END'],
        'time': ['day_', 'hour_'],
    }

    # Create mapping from token ID to category
    id_to_category = {}
    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        for category, prefixes in category_prefixes.items():
            for prefix in prefixes:
                if token.startswith(prefix) or token == prefix:
                    id_to_category[token_id] = category
                    break

    # Get predictions
    predictions = torch.argmax(logits, dim=-1)

    # Create mask
    mask = (labels != ignore_index)

    # Initialize category stats
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Flatten tensors for easier processing
    labels_flat = labels[mask].cpu().numpy()
    predictions_flat = predictions[mask].cpu().numpy()

    # Compute per-category metrics
    for true_id, pred_id in zip(labels_flat, predictions_flat):
        category = id_to_category.get(true_id, 'other')
        category_stats[category]['total'] += 1
        if true_id == pred_id:
            category_stats[category]['correct'] += 1

    # Convert to accuracy metrics
    category_metrics = {}
    for category, stats in category_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            category_metrics[category] = {
                'accuracy': float(accuracy),
                'count': int(stats['total']),
                'correct': int(stats['correct'])
            }

    return category_metrics


def compute_metrics_for_trainer(eval_pred):
    """
    Compute metrics for HuggingFace Trainer.

    This function is called during evaluation in the Trainer loop.

    Args:
        eval_pred: EvalPrediction object with:
            - predictions: Model logits [batch_size, seq_len, vocab_size]
            - label_ids: Target labels [batch_size, seq_len]

    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred

    # Convert to tensors if needed
    if not isinstance(logits, torch.Tensor):
        logits = torch.from_numpy(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels)

    # Compute cross-entropy loss manually
    # Reshape for loss computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute loss (ignoring -100 labels)
    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=-100,
        reduction='mean'
    )
    loss_value = loss.item()

    # Compute perplexity
    perplexity = compute_perplexity(loss_value)

    # Compute token accuracy
    accuracy = compute_token_accuracy(logits, labels, ignore_index=-100)

    # Compute top-5 accuracy
    top5_accuracy = compute_top_k_accuracy(logits, labels, k=5, ignore_index=-100)

    return {
        'eval_loss': float(loss_value),
        'perplexity': float(perplexity),
        'token_accuracy': float(accuracy),
        'top5_accuracy': float(top5_accuracy),
    }


def compute_detailed_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: Any,
    ignore_index: int = -100
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics including category breakdown.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]
        tokenizer: Tokenizer with vocabulary
        ignore_index: Index to ignore

    Returns:
        Dictionary with overall metrics and category-wise breakdown
    """
    # Compute overall metrics
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    loss = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction='mean'
    )
    loss_value = loss.item()

    perplexity = compute_perplexity(loss_value)
    accuracy = compute_token_accuracy(logits, labels, ignore_index)
    top5_accuracy = compute_top_k_accuracy(logits, labels, k=5, ignore_index=ignore_index)

    # Compute category metrics
    category_metrics = compute_category_metrics(
        logits, labels, tokenizer, ignore_index
    )

    # Build comprehensive metrics dict
    metrics = {
        'loss': float(loss_value),
        'perplexity': float(perplexity),
        'token_accuracy': float(accuracy),
        'top5_accuracy': float(top5_accuracy),
        'category_metrics': category_metrics,
    }

    return metrics


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a formatted summary of metrics.

    Args:
        metrics: Dictionary of metrics from compute_detailed_metrics
    """
    print("=" * 70)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(f"\nOverall Metrics:")
    print(f"  Loss:            {metrics['loss']:.4f}")
    print(f"  Perplexity:      {metrics['perplexity']:.2f}")
    print(f"  Token Accuracy:  {metrics['token_accuracy']*100:.2f}%")
    print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']*100:.2f}%")

    if 'category_metrics' in metrics:
        print(f"\nCategory-wise Accuracy:")
        print(f"  {'Category':<25} {'Accuracy':>10} {'Count':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10}")

        # Sort by count (descending)
        sorted_categories = sorted(
            metrics['category_metrics'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        for category, stats in sorted_categories:
            acc_pct = stats['accuracy'] * 100
            count = stats['count']
            print(f"  {category:<25} {acc_pct:>9.2f}% {count:>10,}")

    print("=" * 70)


class MetricsTracker:
    """
    Track metrics across training steps for analysis.
    """

    def __init__(self):
        self.metrics_history = defaultdict(list)

    def add_metrics(self, metrics: Dict[str, float], step: int):
        """Add metrics for a given step."""
        for key, value in metrics.items():
            self.metrics_history[key].append((step, value))

    def get_metric_history(self, metric_name: str) -> List[tuple]:
        """Get history for a specific metric."""
        return self.metrics_history[metric_name]

    def get_latest_metrics(self) -> Dict[str, float]:
        """Get the most recent values for all metrics."""
        latest = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                latest[metric_name] = history[-1][1]
        return latest

    def reset(self):
        """Reset all tracked metrics."""
        self.metrics_history.clear()
