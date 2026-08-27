#!/usr/bin/env python3
"""
05_evaluate_test.py - Test Set Evaluation for GPT2 Clinical Models

Evaluates trained GPT2 models on test data with memory-efficient processing.

Usage:
    python AR/gpt2/05_evaluate_test.py \\
        --model models/gpt2/trained_models/clif-small-124m-production/final_model \\
        --test-data models/gpt2/splits/test/data.parquet \\
        --vocab models/gpt2/vocab
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add AR/gpt2 to path
sys.path.insert(0, str(Path(__file__).parent))

from model import GPT, GPTConfig
from vocabulary import Vocabulary
from dataset import ClifDataset


def variable_length_collator(features, pad_token_id=0, max_length=4096):
    """Custom data collator for variable-length sequences with padding"""
    input_ids_list = []
    for f in features:
        # Handle both dict and direct tensor formats
        if isinstance(f, dict):
            if "input_ids" in f:
                ids = f["input_ids"]
            else:
                continue
        else:
            ids = f  # Assume it's already the tensor/list

        # Convert to tensor if needed
        if isinstance(ids, list):
            ids = torch.tensor(ids, dtype=torch.long)
        elif not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)

        # Truncate if too long
        if len(ids) > max_length:
            ids = ids[:max_length]

        input_ids_list.append(ids)

    if not input_ids_list:
        raise ValueError("No valid input_ids found in batch")

    # Find max length in batch
    max_len = max(len(ids) for ids in input_ids_list)

    # Pad sequences to max length in batch
    padded_input_ids = []
    for ids in input_ids_list:
        if len(ids) < max_len:
            padding = torch.full((max_len - len(ids),), pad_token_id, dtype=torch.long)
            ids = torch.cat([ids, padding])
        padded_input_ids.append(ids)

    batch = {
        "input_ids": torch.stack(padded_input_ids),
    }
    # For causal LM, labels are the same as input_ids (shifted internally by model)
    batch["labels"] = batch["input_ids"].clone()
    return batch


def evaluate_model(
    model,
    dataloader,
    device: str = 'cuda',
    num_examples: int = None,
) -> Dict[str, Any]:
    """
    Evaluate model on test dataset.

    Args:
        model: Trained model
        dataloader: Test data loader
        device: Device to use

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_top5_correct = 0
    total_tokens = 0
    num_batches = 0

    print("\nEvaluating model on test set...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, targets=labels)

            # Handle both tuple and object outputs
            if hasattr(outputs, 'loss'):
                # CausalLMOutputWithPast or similar
                loss = outputs.loss
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                # (loss, logits) tuple
                loss, logits = outputs[0], outputs[1] if len(outputs) > 1 else None
            else:
                # Just loss scalar
                loss = outputs
                logits = None

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Get logits if not already available
            if logits is None:
                logits = model(input_ids)  # This returns logits when no targets provided

            # Compute metrics for this batch
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for metrics computation
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            # Filter out padding (assuming -100 or if you don't have padding, skip this)
            # Since packed sequences don't have padding, we can use all tokens
            valid_mask = flat_labels != -100
            if valid_mask.sum() == 0:
                # If no valid tokens, use all tokens
                valid_mask = torch.ones_like(flat_labels, dtype=torch.bool)

            valid_logits = flat_logits[valid_mask]
            valid_labels = flat_labels[valid_mask]

            # Token accuracy
            predictions = valid_logits.argmax(dim=-1)
            total_correct += (predictions == valid_labels).sum().item()

            # Top-5 accuracy
            top5_preds = valid_logits.topk(5, dim=-1).indices
            total_top5_correct += (top5_preds == valid_labels.unsqueeze(-1)).any(dim=-1).sum().item()

            # Token count
            total_tokens += valid_mask.sum().item()

    # Compute aggregate metrics
    print("\nComputing aggregate metrics...")
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    top5_accuracy = total_top5_correct / total_tokens if total_tokens > 0 else 0.0

    metrics = {
        'loss': avg_loss,
        'perplexity': perplexity,
        'token_accuracy': token_accuracy,
        'top5_accuracy': top5_accuracy,
        'num_tokens': int(total_tokens),
        'num_examples': num_examples if num_examples is not None else 0,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GPT2 Clinical Model on Test Set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to data splits directory (containing train/val/test subdirs)'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Path to vocabulary directory (containing vocab_lock.json or vocab.gzip)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='gpt2_test_results.json',
        help='Output path for test results JSON'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for evaluation (reduce if OOM)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    print("=" * 80)
    print("GPT2 TEST SET EVALUATION")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Vocabulary: {args.vocab}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch_size}")
    print()

    # Load model
    print("Loading model...")

    # Load vocabulary - try vocab_lock.json first, then fall back to vocab.gzip
    vocab_dir = Path(args.vocab)
    vocab_lock_path = vocab_dir / "vocab_lock.json"
    vocab_gzip_path = vocab_dir / "vocab.gzip"

    if vocab_lock_path.exists():
        print(f"  Loading from: {vocab_lock_path}")
        vocab = Vocabulary.from_vocab_lock(vocab_lock_path)
        print(f"  ✓ Loaded vocabulary: {len(vocab)} tokens")
        print(f"  ✓ Vocabulary hash: {vocab.get_vocab_hash()[:16]}...")
    elif vocab_gzip_path.exists():
        print(f"  Loading from: {vocab_gzip_path} (legacy format)")
        vocab = Vocabulary().load(vocab_gzip_path)
        print(f"  ✓ Loaded vocabulary: {len(vocab)} tokens")
    else:
        raise FileNotFoundError(
            f"Vocabulary not found at {vocab_lock_path} or {vocab_gzip_path}\n"
            "Please ensure vocabulary files exist in the specified directory"
        )

    # Create config for GPT2-small (124M)
    # Note: block_size=8192 to match the trained checkpoint
    model_config = GPTConfig(
        vocab_size=len(vocab),
        block_size=8192,  # Match checkpoint training config
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,  # No dropout during eval
        bias=True,
    )

    # Initialize model with config
    model = GPT(model_config)

    # Load weights
    model_path = Path(args.model) / "pytorch_model.bin"
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)

    model = model.to(args.device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model loaded: {num_params:,} parameters ({num_params/1e6:.1f}M)")
    print()

    # Load test dataset using ClifDataset
    print("Loading test dataset...")
    clif_dataset = ClifDataset(
        data_dir=Path(args.data_dir),
        vocab_path=vocab_path,
        collation="packed",
        max_seq_length=model.config.block_size,
    )

    # Get test dataset
    test_dataset = clif_dataset.get_test_dataset()
    print(f"  ✓ Loaded test dataset: {clif_dataset.n_test:,} sequences")
    print()

    # Get PAD token ID for collator
    pad_token_id = vocab("PAD")

    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda features: variable_length_collator(
            features, pad_token_id=pad_token_id, max_length=model.config.block_size
        ),
        num_workers=0,
    )

    # Evaluate model
    metrics = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=args.device,
        num_examples=clif_dataset.n_test,
    )

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Loss:            {metrics['loss']:.4f}")
    print(f"Perplexity:      {metrics['perplexity']:.2f}")
    print(f"Token Accuracy:  {metrics['token_accuracy']*100:.2f}%")
    print(f"Top-5 Accuracy:  {metrics['top5_accuracy']*100:.2f}%")
    print(f"Total Tokens:    {metrics['num_tokens']:,}")
    print(f"Num Examples:    {metrics['num_examples']:,}")
    print("=" * 80)
    print()

    # Save results to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'model': str(args.model),
        'data_dir': str(args.data_dir),
        'metrics': metrics,
        'config': {
            'batch_size': args.batch_size,
            'device': args.device,
            'vocab_size': len(vocab),
            'block_size': model.config.block_size,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_path}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
