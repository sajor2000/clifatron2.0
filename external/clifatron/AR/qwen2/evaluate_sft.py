#!/usr/bin/env python3
"""
evaluate_sft.py - Evaluate Trained SFT Models

Evaluate Qwen2 models trained with SFT on test set.

Usage:
    uv run AR/qwen2_sft/evaluate_sft.py \\
        --checkpoint models/qwen2_sft/checkpoints/clif-qwen2-sft-0.5b/final_model \\
        --clif-config clif_config.json \\
        --mode secondary \\
        --output test_results.json

Features:
    - Evaluates on test set (2024 data for temporal split)
    - Computes perplexity and loss
    - Generates sample predictions
    - Saves results to JSON
    - Optional W&B logging
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM
from tqdm import tqdm
import wandb

# Add parent directory to path for tokenizer import (shared for primary/secondary mode)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Tokenizer is shared between qwen2 and qwen2_sft (required for vocab locking)
from qwen2.tokenizer.clinical_tokenizer import ClinicalTokenizer

# Local imports (self-contained)
from data.hospitalization_dataset import load_hospitalization_dataset


def load_model_and_tokenizer(checkpoint_path: str):
    """
    Load trained model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {checkpoint_path}...")

    # Load tokenizer
    tokenizer = ClinicalTokenizer.from_pretrained(checkpoint_path)
    print(f"  ✓ Loaded tokenizer (vocab size: {len(tokenizer)})")

    # Load model
    model = Qwen2ForCausalLM.from_pretrained(checkpoint_path)
    print(f"  ✓ Loaded model ({sum(p.numel() for p in model.parameters()):,} parameters)")

    return model, tokenizer


def evaluate_on_dataset(
    model: Qwen2ForCausalLM,
    tokenizer: ClinicalTokenizer,
    dataset,
    device: str = 'cuda',
    batch_size: int = 4,
    max_samples: int = None,
) -> Dict:
    """
    Evaluate model on dataset.

    Args:
        model: Qwen2 model
        tokenizer: Clinical tokenizer
        dataset: HospitalizationTextDataset
        device: Device to use ('cuda' or 'cpu')
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate (None = all)

    Returns:
        Dictionary with evaluation metrics
    """
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    print(f"\nEvaluating on {num_samples:,} samples...")

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating"):
            batch_end = min(i + batch_size, num_samples)
            batch_texts = [dataset[j]["text"] for j in range(i, batch_end)]

            # Tokenize batch
            encodings = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=8192,
                add_special_tokens=True,
            )

            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # For causal LM, labels = input_ids
            )

            # Accumulate loss
            loss = outputs.loss
            num_tokens_in_batch = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens_in_batch
            total_tokens += num_tokens_in_batch

    # Compute metrics
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_samples": num_samples,
        "num_tokens": total_tokens,
    }

    return metrics


def generate_sample_predictions(
    model: Qwen2ForCausalLM,
    tokenizer: ClinicalTokenizer,
    dataset,
    num_samples: int = 5,
    max_new_tokens: int = 50,
    device: str = 'cuda',
) -> List[Dict]:
    """
    Generate sample predictions for inspection.

    Args:
        model: Qwen2 model
        tokenizer: Clinical tokenizer
        dataset: HospitalizationTextDataset
        num_samples: Number of samples to generate
        max_new_tokens: Maximum tokens to generate
        device: Device to use

    Returns:
        List of dictionaries with input and generated text
    """
    model.to(device)
    model.eval()

    samples = []

    print(f"\nGenerating {num_samples} sample predictions...")

    for i in range(min(num_samples, len(dataset))):
        text = dataset[i]["text"]

        # Take first 100 tokens as context
        context_tokens = text.split()[:100]
        context = " ".join(context_tokens)

        # Tokenize context
        input_ids = tokenizer(
            context,
            return_tensors='pt',
            add_special_tokens=True,
        )['input_ids'].to(device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        context_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

        samples.append({
            "sample_id": i,
            "context": context_text,
            "generated": generated_text,
            "context_length": len(input_ids[0]),
        })

        print(f"\nSample {i+1}:")
        print(f"  Context: {context_text[:100]}...")
        print(f"  Generated: {generated_text[:100]}...")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained SFT model on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model and config
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint directory'
    )

    parser.add_argument(
        '--clif-config',
        type=str,
        default='clif_config.json',
        help='Path to clif_config.json (default: clif_config.json)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['primary', 'secondary'],
        help='Vocabulary mode (for consistency checking)'
    )

    # Evaluation settings
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for evaluation (default: 4)'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all)'
    )

    parser.add_argument(
        '--num-generation-samples',
        type=int,
        default=5,
        help='Number of sample generations to produce (default: 5)'
    )

    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=50,
        help='Maximum new tokens to generate per sample (default: 50)'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='test_results.json',
        help='Output file for results (default: test_results.json)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available, else cpu)'
    )

    # Logging
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )

    parser.add_argument(
        '--wandb-project',
        type=str,
        default='CLIFATRON',
        help='W&B project name (default: CLIFATRON)'
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    # Validate vocabulary
    tokenizer.validate_vocab_size(expected_size=1380)
    vocab_hash = tokenizer.get_vocab_hash()
    print(f"  ✓ Vocabulary hash: {vocab_hash[:16]}...")
    print()

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_hospitalization_dataset(
        config_path=args.clif_config,
        split='test',
        split_mode='temporal',
        seed=42,
    )
    print()

    # Initialize W&B if enabled
    if not args.no_wandb:
        # Load clif_config for site name
        with open(args.clif_config, 'r') as f:
            clif_config = json.load(f)
        site = clif_config.get('site', 'unknown')

        wandb.init(
            project=args.wandb_project,
            name=f"{site}-sft-evaluation",
            config={
                "checkpoint": args.checkpoint,
                "mode": args.mode,
                "batch_size": args.batch_size,
                "max_samples": args.max_samples,
            }
        )

    # Evaluate
    metrics = evaluate_on_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Samples: {metrics['num_samples']:,}")
    print(f"Tokens: {metrics['num_tokens']:,}")
    print("=" * 80)
    print()

    # Generate sample predictions
    samples = generate_sample_predictions(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        num_samples=args.num_generation_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    # Combine results
    results = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "metrics": metrics,
        "sample_generations": samples,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Log to W&B
    if not args.no_wandb:
        wandb.log(metrics)
        wandb.finish()

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
