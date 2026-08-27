#!/usr/bin/env python3
"""
03_evaluate_test.py - Test Set Evaluation for GPT2 Clinical Models

Evaluates trained models on cached test data with memory-efficient processing.
Designed to run separately from training to avoid OOM errors.

Usage:
    python 03_evaluate_test.py \\
        --checkpoint output/gpt2_hf-small/final_model \\
        --preprocessed-dir AR/gpt2_hf/outputs/preprocessed/small_temporal_len8192
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader
from tokenizer.clinical_tokenizer import ClinicalTokenizer
from data.narrative_dataset import ClinicalNarrativeDataset
from data.data_collator import DataCollatorForClinicalCausalLM


def compute_metrics(all_logits, all_labels, ignore_index=-100):
    """
    Compute evaluation metrics from logits and labels.

    Args:
        all_logits: Tensor of shape (total_samples, seq_len, vocab_size)
        all_labels: Tensor of shape (total_samples, seq_len)
        ignore_index: Index to ignore in loss calculation

    Returns:
        Dictionary of metrics
    """
    # Filter out ignored indices
    mask = all_labels != ignore_index
    valid_logits = all_logits[mask]
    valid_labels = all_labels[mask]

    # Compute token accuracy (top-1)
    predictions = valid_logits.argmax(dim=-1)
    token_accuracy = (predictions == valid_labels).float().mean().item()

    # Compute top-5 accuracy
    top5_preds = valid_logits.topk(5, dim=-1).indices
    top5_accuracy = (top5_preds == valid_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()

    # Compute loss and perplexity
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(valid_logits, valid_labels).item()
    perplexity = torch.exp(torch.tensor(loss)).item()

    return {
        'loss': loss,
        'perplexity': perplexity,
        'token_accuracy': token_accuracy,
        'top5_accuracy': top5_accuracy,
        'num_tokens': int(mask.sum().item()),
    }


def evaluate_model(
    model,
    dataloader,
    device: str = 'cuda',
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
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Accumulate loss
            total_loss += outputs.loss.item()
            num_batches += 1

            # Compute metrics for this batch
            logits = outputs.logits
            mask = labels != -100
            valid_logits = logits[mask]
            valid_labels = labels[mask]

            # Token accuracy
            predictions = valid_logits.argmax(dim=-1)
            total_correct += (predictions == valid_labels).sum().item()

            # Top-5 accuracy
            top5_preds = valid_logits.topk(5, dim=-1).indices
            total_top5_correct += (top5_preds == valid_labels.unsqueeze(-1)).any(dim=-1).sum().item()

            # Token count
            total_tokens += mask.sum().item()

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
        'num_examples': len(dataloader.dataset),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GPT2 Clinical Model on Test Set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint directory'
    )
    parser.add_argument(
        '--preprocessed-dir',
        type=Path,
        required=True,
        help='Path to preprocessed data directory containing test_dataset.pt'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_results.json',
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
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['primary', 'secondary'],
        help='Vocabulary mode: primary (first site), secondary (other sites using primary vocab)'
    )

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    print("=" * 80)
    print("GPT2 TEST SET EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Preprocessed Dir: {args.preprocessed_dir}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch_size}")
    print()

    # Verify test data exists
    test_cache_path = args.preprocessed_dir / "test_dataset.pt"
    if not test_cache_path.exists():
        print(f"❌ Test dataset not found at: {test_cache_path}")
        print("\nTest data should be generated during preprocessing with temporal split mode.")
        print("Make sure you ran preprocessing with --split-mode temporal")
        sys.exit(1)

    # Load tokenizer (with validation)
    print(f"Loading tokenizer (mode: {args.mode})...")
    # Try to find tokenizer in checkpoint dir first, then fall back to default location
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = checkpoint_path.parent.parent / 'tokenizer' / 'clinical_tokenizer'
    if not tokenizer_path.exists():
        tokenizer_path = Path('AR/gpt2_hf/tokenizer/clinical_tokenizer')

    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found at: {tokenizer_path}")
        print("Please ensure tokenizer exists before evaluation.")
        sys.exit(1)

    tokenizer = ClinicalTokenizer.from_pretrained(str(tokenizer_path))
    print(f"  ✓ Loaded tokenizer from {tokenizer_path}")
    print(f"  ✓ Vocabulary size: {len(tokenizer)}")

    # Validate vocabulary (skipping hardcoded size check - use metadata validation instead)
    # tokenizer.validate_vocab_size(expected_size=1388)
    vocab_hash = tokenizer.get_vocab_hash()
    print(f"  ✓ Vocabulary hash: {vocab_hash[:16]}...")

    # Load metadata to validate against cached data
    metadata_path = args.preprocessed_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            cache_metadata = json.load(f)

        if 'vocab_hash' in cache_metadata:
            cached_vocab_hash = cache_metadata['vocab_hash']
            if vocab_hash != cached_vocab_hash:
                print(f"❌ VOCABULARY MISMATCH!")
                print(f"   Tokenizer vocab hash: {vocab_hash[:16]}...")
                print(f"   Test data vocab hash: {cached_vocab_hash[:16]}...")
                print(f"   The test data was preprocessed with a different vocabulary!")
                sys.exit(1)
            print(f"  ✓ Vocabulary matches test data")
    print()

    # Load model
    print("Loading model...")
    model = GPT2LMHeadModel.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if args.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32,
    )
    model = model.to(args.device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model loaded: {num_params:,} parameters ({num_params/1e9:.2f}B)")
    print()

    # Load cached test dataset
    print("Loading cached test dataset...")
    test_dataset = ClinicalNarrativeDataset.from_cached_tensors(
        cache_dir=str(args.preprocessed_dir),
        split='test',
        tokenizer=tokenizer,
    )
    print(f"  ✓ Loaded {len(test_dataset):,} test samples")
    print()

    # Create dataloader
    collator = DataCollatorForClinicalCausalLM(tokenizer=tokenizer)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues with cached tensors
    )

    # Evaluate model
    metrics = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=args.device,
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
        'checkpoint': str(args.checkpoint),
        'preprocessed_dir': str(args.preprocessed_dir),
        'metrics': metrics,
        'config': {
            'batch_size': args.batch_size,
            'device': args.device,
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_path}")
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
