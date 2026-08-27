#!/usr/bin/env python3
"""
evaluate_packed.py - Evaluate model trained on packed sequences

Evaluates a trained model on the packed validation dataset using the same
1D document ID approach from training.

Usage:
    uv run AR/qwen2/evaluate_packed.py \
        --checkpoint /dev/shm/qwen2/clif-qwen2-sft-0.5b/final_model \
        --packed-dir models/qwen2/preprocessed/packed_temporal_len8192 \
        --batch-size 4
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM, Qwen2Config
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'AR' / 'qwen2'))

from tokenizer.clinical_tokenizer import ClinicalTokenizer
from data.packed_dataset import load_packed_dataset


def data_collator(features):
    """
    Collator that handles packed sequences with 1D attention masks.
    Document isolation is enforced by [SEP] tokens, not 2D masks.
    This matches the training approach for consistency.
    """
    # Convert lists to tensors
    # Use standard 1D attention mask - document isolation enforced by [SEP] tokens
    batch = {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
    }

    # Note: document_ids are available in features but not used
    # Document isolation is handled by [SEP] tokens between documents

    return batch


def evaluate_model(
    model: Qwen2ForCausalLM,
    dataloader: DataLoader,
    device: str = 'cuda',
) -> Dict:
    """
    Evaluate model on packed validation dataset.

    Args:
        model: Trained Qwen2 model
        dataloader: DataLoader with packed sequences
        device: Device to use

    Returns:
        Dictionary with metrics (loss, perplexity, tokens, top-1, top-5 accuracy)
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    # Accuracy tracking
    correct_top1 = 0
    correct_top5 = 0
    total_predictions = 0

    print(f"\nEvaluating on {len(dataloader)} batches...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Accumulate loss (weighted by number of non-masked tokens)
            loss = outputs.loss
            num_valid_tokens = (labels != -100).sum().item()

            total_loss += loss.item() * num_valid_tokens
            total_tokens += num_valid_tokens

            # Calculate top-1 and top-5 accuracy
            logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

            # Shift logits and labels for next-token prediction
            # logits[:, :-1, :] predicts labels[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Get predictions
            _, top1_pred = shift_logits.max(dim=-1)  # Top-1 predictions
            _, top5_pred = shift_logits.topk(k=5, dim=-1)  # Top-5 predictions

            # Mask for valid positions (not -100)
            valid_mask = shift_labels != -100

            # Top-1 accuracy
            correct_top1 += (top1_pred == shift_labels).masked_select(valid_mask).sum().item()

            # Top-5 accuracy
            # Check if true label is in top-5 predictions
            shift_labels_expanded = shift_labels.unsqueeze(-1).expand_as(top5_pred)
            in_top5 = (top5_pred == shift_labels_expanded).any(dim=-1)
            correct_top5 += in_top5.masked_select(valid_mask).sum().item()

            # Count valid predictions
            total_predictions += valid_mask.sum().item()

    # Compute metrics
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    top1_accuracy = (correct_top1 / total_predictions) * 100 if total_predictions > 0 else 0
    top5_accuracy = (correct_top5 / total_predictions) * 100 if total_predictions > 0 else 0

    return {
        'loss': float(avg_loss),
        'perplexity': float(perplexity),
        'num_tokens': int(total_tokens),
        'num_batches': len(dataloader),
        'top1_accuracy': float(top1_accuracy),
        'top5_accuracy': float(top5_accuracy),
        'total_predictions': int(total_predictions),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on packed validation set")

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--packed-dir',
        type=str,
        default='models/qwen2/preprocessed/packed_temporal_len8192',
        help='Directory with packed parquet files (default: models/qwen2/preprocessed/packed_temporal_len8192)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for evaluation (default: 4)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='eval_results_packed.json',
        help='Output file for results (default: eval_results_packed.json)'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate (default: test)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PACKED SEQUENCE EVALUATION")
    print("=" * 80)
    print()

    # Load model and tokenizer
    print(f"Loading model from {args.checkpoint}...")
    model = Qwen2ForCausalLM.from_pretrained(args.checkpoint)
    tokenizer = ClinicalTokenizer.from_pretrained(args.checkpoint)
    print(f"  ✓ Loaded model ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"  ✓ Loaded tokenizer (vocab size: {len(tokenizer)})")
    print()

    # Load packed dataset
    print(f"Loading packed {args.split} dataset...")
    dataset = load_packed_dataset(args.packed_dir, split=args.split)
    print(f"  ✓ Loaded {len(dataset)} packed sequences")
    print()

    # Create dataloader with simple collator (matches training approach)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,  # Uses 1D attention masks with [SEP] token isolation
        num_workers=0,  # Keep simple for evaluation
    )

    # Evaluate
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=args.device,
    )

    # Print results
    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Loss:              {metrics['loss']:.4f}")
    print(f"Perplexity:        {metrics['perplexity']:.2f}")
    print(f"Top-1 Accuracy:    {metrics['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy:    {metrics['top5_accuracy']:.2f}%")
    print(f"Tokens:            {metrics['num_tokens']:,}")
    print(f"Predictions:       {metrics['total_predictions']:,}")
    print(f"Batches:           {metrics['num_batches']:,}")
    print("=" * 80)
    print()

    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'packed_dir': args.packed_dir,
        'batch_size': args.batch_size,
        'metrics': metrics,
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
