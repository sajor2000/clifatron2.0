#!/usr/bin/env python3
"""
train_only.py - Fast Training Script for GPT2 Using Cached Data

Trains GPT2 models using pre-processed cached datasets, avoiding the
15-30 minute data loading overhead on every run.

Prerequisites:
    Run data_prep.py first to create cached datasets:
    uv run AR/gpt2_hf/data_prep.py --model-size small --split-mode temporal

Usage:
    uv run torchrun --nproc_per_node=auto AR/gpt2_hf/train_only.py \\
        --model-size small \\
        --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192

    # Resume from checkpoint
    uv run torchrun --nproc_per_node=auto AR/gpt2_hf/train_only.py \\
        --model-size small \\
        --preprocessed-dir models/gpt2_hf/preprocessed/small_temporal_len8192 \\
        --resume-from checkpoint-1000

Features:
    - Loads pre-tokenized datasets in <2 minutes
    - Train from scratch with custom vocabulary
    - Causal language modeling objective
    - Mixed precision training (bf16)
    - Multi-GPU support with DeepSpeed
    - Weights & Biases logging
    - Automatic checkpointing
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
import wandb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer.clinical_tokenizer import ClinicalTokenizer
from data.narrative_dataset import ClinicalNarrativeDataset
from data.data_collator import create_data_collator
from models.gpt2_configs import get_gpt2_config, print_model_info
from utils.gpu_detector import auto_configure
from utils.metrics import compute_metrics_for_trainer
from utils.cache_utils import verify_cache, load_cached_metadata


def load_clif_config(config_path: str) -> dict:
    """Load clif_config.json."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def load_training_config(config_path: str) -> dict:
    """Load training_config.yaml."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_wandb(args, training_config, gpu_config=None, clif_config=None, cache_metadata=None):
    """Setup Weights & Biases logging via environment variables (GPT2-style)."""
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return None

    # Login to W&B using API key from clif_config
    if clif_config and 'wandb_api_key' in clif_config:
        try:
            wandb.login(key=clif_config['wandb_api_key'])
            print("  ✓ Logged into Weights & Biases")
        except Exception as e:
            print(f"  ⚠️  W&B login failed: {e}")
            print("  Continuing without W&B logging...")
            os.environ["WANDB_DISABLED"] = "true"
            return None

    # Extract site name from clif_config
    site = clif_config.get('site', 'unknown') if clif_config else 'unknown'

    # Use custom run name if provided, otherwise auto-generate with site-CLIFATRON-model format
    if args.run_name:
        run_name = args.run_name
    else:
        # Format: site-CLIFATRON-model_name (e.g., rush-CLIFATRON-gpt2-small)
        run_name = f"{site}-CLIFATRON-gpt2-{args.model_size}"

    # Setup wandb via environment variables (GPT2-style)
    os.environ["WANDB_PROJECT"] = training_config.get('wandb_project', 'gpt2_hf-clinical')
    os.environ["WANDB_RUN_NAME"] = run_name

    print(f"  ✓ WandB project: {os.environ['WANDB_PROJECT']}")
    print(f"  ✓ WandB run name: {run_name}")
    print()

    return run_name


def main():
    parser = argparse.ArgumentParser(
        description="Train GPT2 models using pre-processed cached datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train 0.5B model with cached data
  uv run torchrun --nproc_per_node=auto AR/gpt2_hf/train_only.py \\
      --model-size small \\
      --preprocessed-dir AR/gpt2_hf/outputs/preprocessed/small_temporal_len8192

  # Train with custom config
  uv run torchrun --nproc_per_node=auto AR/gpt2_hf/train_only.py \\
      --model-size medium \\
      --preprocessed-dir models/gpt2_hf/preprocessed/medium_temporal_len8192 \\
      --train-config custom.yaml

  # Resume from checkpoint
  uv run torchrun --nproc_per_node=auto AR/gpt2_hf/train_only.py \\
      --model-size small \\
      --preprocessed-dir AR/gpt2_hf/outputs/preprocessed/small_temporal_len8192 \\
      --resume-from checkpoint-1000

Output:
  - Checkpoints saved to specified output directory
  - Logs and metrics tracked in W&B (if enabled)

Prerequisites:
  Run data_prep.py first to create cached datasets
        """
    )

    # Model configuration
    parser.add_argument(
        '--model-size',
        type=str,
        required=True,
        choices=['nano', 'micro', 'tiny', 'small', 'medium'],
        help='Model size: nano, micro, tiny, small, or medium'
    )

    # Cached data configuration (REQUIRED)
    parser.add_argument(
        '--preprocessed-dir',
        type=str,
        required=True,
        help='Path to directory with preprocessed cached datasets (from data_prep.py)'
    )

    # Configuration files
    parser.add_argument(
        '--clif-config',
        type=str,
        default='clif_config.json',
        help='Path to clif_config.json (default: clif_config.json)'
    )

    parser.add_argument(
        '--train-config',
        type=str,
        default=None,
        help='Path to training config YAML (default: AR/gpt2_hf/config/training_config.yaml)'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for checkpoints (default: models/gpt2_hf/checkpoints/clif-gpt2_hf-{model_size}/)'
    )

    # Training configuration
    parser.add_argument(
        '--deepspeed',
        type=str,
        default=None,
        help='Path to DeepSpeed config JSON (auto-detected if not specified)'
    )

    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    # Logging configuration
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )

    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Custom name for the training run (default: auto-generated)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['primary', 'secondary'],
        help='Vocabulary mode: primary (first site), secondary (other sites using primary vocab)'
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Resolve paths
    script_dir = Path(__file__).parent

    if args.train_config is None:
        args.train_config = script_dir / "config" / "training_config.yaml"

    if args.output_dir is None:
        # Checkpoints go to models/gpt2_hf/checkpoints at root level
        root_dir = script_dir.parent.parent  # Go up to CLIFATRON root (AR/gpt2_hf -> AR -> CLIFATRON)
        args.output_dir = root_dir / "models" / "gpt2_hf" / "checkpoints" / f"clif-gpt2_hf-{args.model_size}"

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.preprocessed_dir = Path(args.preprocessed_dir)

    # Print header
    print("=" * 80)
    print("GPT2 CLINICAL LANGUAGE MODEL TRAINING (CACHED DATA)")
    print("=" * 80)
    print(f"Model Size: GPT2-{args.model_size}")
    print(f"Preprocessed Data: {args.preprocessed_dir}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    print()

    # Verify cached data exists
    print("Verifying cached data...")
    if not verify_cache(args.preprocessed_dir, splits=['train', 'val']):
        print(f"\n❌ Cached data not found or invalid at {args.preprocessed_dir}")
        print("\nPlease run data_prep.py first to create cached datasets:")
        print(f"  uv run AR/gpt2_hf/data_prep.py --model-size {args.model_size}")
        sys.exit(1)

    # Load cache metadata
    cache_metadata = load_cached_metadata(args.preprocessed_dir)
    print(f"  ✓ Cache validated")
    print(f"  ✓ Split mode: {cache_metadata.get('split_mode', 'unknown')}")
    print(f"  ✓ Train samples: {cache_metadata.get('train_samples', 'unknown'):,}")
    print(f"  ✓ Val samples: {cache_metadata.get('val_samples', 'unknown'):,}")
    print(f"  ✓ Created: {cache_metadata.get('created_at', 'unknown')}")
    print()

    # Load configurations
    print("Loading configurations...")
    clif_config = load_clif_config(args.clif_config)
    print(f"  ✓ CLIF config: {args.clif_config}")

    training_config = load_training_config(args.train_config)
    print(f"  ✓ Training config: {args.train_config}")
    print()

    # Auto-configure GPU settings
    print("Detecting GPU configuration...")
    gpu_config = auto_configure(
        config_dir=script_dir / "config",
        model_size=args.model_size,
        verbose=True
    )
    print()

    # Extract GPU settings
    use_bf16 = gpu_config.use_bf16 if hasattr(gpu_config, 'use_bf16') else False
    use_fp16 = gpu_config.use_fp16 if hasattr(gpu_config, 'use_fp16') else False

    # Use detected DeepSpeed config if not specified
    if args.deepspeed is None and hasattr(gpu_config, 'deepspeed_config_path'):
        args.deepspeed = gpu_config.deepspeed_config_path

    # Setup W&B
    try:
        setup_wandb(args, training_config, gpu_config=gpu_config, clif_config=clif_config, cache_metadata=cache_metadata)
    except:
        setup_wandb(args, training_config, gpu_config=None, clif_config=clif_config, cache_metadata=cache_metadata)

    # Load tokenizer (with validation)
    print(f"Loading tokenizer (mode: {args.mode})...")
    tokenizer_path = script_dir / "tokenizer" / "clinical_tokenizer"

    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        print("Please build tokenizer first using data_prep.py:")
        print("  uv run AR/gpt2_hf/01_preprocess_data.py --mode primary ...")
        sys.exit(1)

    tokenizer = ClinicalTokenizer.from_pretrained(str(tokenizer_path))
    print(f"  ✓ Loaded tokenizer from {tokenizer_path}")
    print(f"  ✓ Vocabulary size: {len(tokenizer)}")

    # Get vocabulary hash for consistency validation
    vocab_hash = tokenizer.get_vocab_hash()
    print(f"  ✓ Vocabulary hash: {vocab_hash[:16]}...")
    print(f"  ⚠ IMPORTANT: Use this vocabulary for all training and finetuning!")

    # Validate against cached data if available
    if cache_metadata and 'vocab_hash' in cache_metadata:
        cached_vocab_hash = cache_metadata['vocab_hash']
        if vocab_hash != cached_vocab_hash:
            print(f"❌ VOCABULARY MISMATCH!")
            print(f"   Tokenizer vocab hash: {vocab_hash[:16]}...")
            print(f"   Cached data vocab hash: {cached_vocab_hash[:16]}...")
            print(f"   The preprocessed data was created with a different vocabulary!")
            print(f"   Please rerun preprocessing with the correct tokenizer.")
            sys.exit(1)
        print(f"  ✓ Vocabulary matches preprocessed data")
    print()

    # Create model configuration
    print(f"Creating GPT2-{args.model_size} configuration...")
    model_config = get_gpt2_config(args.model_size, vocab_size=len(tokenizer))
    print_model_info(args.model_size)

    # Initialize model from scratch
    print("Initializing model from scratch (random weights)...")
    model = GPT2LMHeadModel(model_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  ✓ Trainable parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
    print()

    # Load cached datasets
    print("Loading cached datasets...")
    train_dataset = ClinicalNarrativeDataset.from_cached_tensors(
        cache_dir=str(args.preprocessed_dir),
        split='train',
        tokenizer=tokenizer,
    )
    print()

    eval_dataset = ClinicalNarrativeDataset.from_cached_tensors(
        cache_dir=str(args.preprocessed_dir),
        split='val',
        tokenizer=tokenizer,
    )
    print()

    print(f"  ✓ Train samples: {len(train_dataset):,}")
    print(f"  ✓ Validation samples: {len(eval_dataset):,}")
    print()

    # Create data collator with packing enabled
    data_collator = create_data_collator(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
        pad_to_multiple_of=8,  # For efficiency
        enable_packing=training_config.get('enable_packing', False),
        pack_to_max_length=training_config.get('pack_to_max_length', 8192),
        repeat_short_sequences=training_config.get('repeat_short_sequences', True)
    )

    # Get training hyperparameters
    model_training_config = training_config['models'][args.model_size]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,

        # Training
        num_train_epochs=model_training_config['num_epochs'],
        per_device_train_batch_size=model_training_config['batch_size'],
        per_device_eval_batch_size=model_training_config.get('per_device_eval_batch_size', model_training_config['batch_size']),  # Use separate eval batch size if specified
        gradient_accumulation_steps=model_training_config['gradient_accumulation_steps'],

        # Optimization
        learning_rate=model_training_config['learning_rate'],
        weight_decay=model_training_config['weight_decay'],
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,

        # Learning rate schedule
        lr_scheduler_type=model_training_config['lr_scheduler'],
        warmup_steps=model_training_config['warmup_steps'],

        # Mixed precision
        bf16=use_bf16,
        fp16=use_fp16,

        # Memory optimization
        gradient_checkpointing=True,

        # Logging
        logging_dir=str(args.output_dir / "logs"),
        logging_steps=training_config['logging_steps'],
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.run_name if args.run_name else f"{clif_config.get('site', 'unknown')}-CLIFATRON-gpt2-{args.model_size}",

        # Evaluation
        eval_strategy="steps",
        eval_steps=training_config['eval_steps'],
        prediction_loss_only=training_config.get('prediction_loss_only', True),  # Only compute loss, don't store predictions (saves memory)

        # Checkpointing
        save_strategy="steps",
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # DeepSpeed
        deepspeed=args.deepspeed,

        # Other
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # Initialize trainer
    print("Initializing trainer...")
    # CRITICAL: If prediction_loss_only=True, do NOT set compute_metrics
    # Setting compute_metrics forces Trainer to collect all predictions in memory (causes OOM)
    prediction_loss_only = training_config.get('prediction_loss_only', True)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=None if prediction_loss_only else compute_metrics_for_trainer,
    )
    print("  ✓ Trainer initialized")
    print()

    # Print training info
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model: GPT2-{args.model_size}")
    print(f"Epochs: {model_training_config['num_epochs']}")
    print(f"Batch size: {model_training_config['batch_size']}")
    print(f"Gradient accumulation: {model_training_config['gradient_accumulation_steps']}")
    effective_batch_size = (
        model_training_config['batch_size'] *
        model_training_config['gradient_accumulation_steps'] *
        (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    )
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Learning rate: {model_training_config['learning_rate']}")
    print(f"Warmup steps: {model_training_config['warmup_steps']}")
    print(f"Weight decay: {model_training_config['weight_decay']}")
    print(f"LR scheduler: {model_training_config['lr_scheduler']}")
    print(f"Mixed precision: BF16={use_bf16}, FP16={use_fp16}")
    print(f"DeepSpeed: {args.deepspeed is not None}")
    print("=" * 80)
    print()

    # WandB Configuration Verification
    print("=" * 80)
    print("WANDB CONFIGURATION VERIFICATION")
    print("=" * 80)
    print(f"WANDB_PROJECT: {os.environ.get('WANDB_PROJECT', 'NOT SET')}")
    print(f"WANDB_RUN_NAME: {os.environ.get('WANDB_RUN_NAME', 'NOT SET')}")
    print(f"WANDB_API_KEY: {'SET' if os.environ.get('WANDB_API_KEY') else 'NOT SET'}")
    print(f"report_to: {training_args.report_to}")
    print(f"run_name: {training_args.run_name}")
    print(f"logging_steps: {training_args.logging_steps}")
    print(f"eval_steps: {training_args.eval_steps}")
    print(f"compute_metrics: {'DISABLED (prediction_loss_only=True)' if prediction_loss_only else 'ENABLED'}")
    print("=" * 80)
    print()

    # Auto-detect and resume from latest checkpoint
    resume_from_checkpoint = None
    if args.resume_from:
        # Manual checkpoint specification takes priority
        resume_from_checkpoint = args.resume_from
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        print()
    else:
        # Auto-detect checkpoints in output_dir
        checkpoint_dirs = list(args.output_dir.glob("checkpoint-*"))
        if checkpoint_dirs:
            # Find the latest checkpoint by step number
            latest_checkpoint = max(checkpoint_dirs, key=lambda p: int(p.name.split("-")[1]))
            resume_from_checkpoint = str(latest_checkpoint)
            step_num = latest_checkpoint.name.split('-')[1]
            print("=" * 80)
            print("AUTO-RESUMING FROM CHECKPOINT")
            print("=" * 80)
            print(f"Found checkpoint: {latest_checkpoint.name}")
            print(f"Resuming training from step {step_num}")
            print()
        else:
            print("No checkpoint found. Starting training from scratch.")
            print()

    # Train
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        print()
        print("=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print()

        # Save final model
        final_model_path = args.output_dir / "final_model"
        print(f"Saving final model to {final_model_path}...")
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        print("  ✓ Model saved")
        print()

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("TRAINING INTERRUPTED")
        print("=" * 80)
        print("\nYou can resume training with:")
        print(f"  uv run torchrun --nproc_per_node=auto AR/gpt2_hf/train_only.py \\")
        print(f"    --model-size {args.model_size} \\")
        print(f"    --preprocessed-dir {args.preprocessed_dir} \\")
        print(f"    --resume-from {args.output_dir}/checkpoint-XXXX")
        print()
        sys.exit(0)

    except Exception as e:
        print("\n\n" + "=" * 80)
        print("ERROR DURING TRAINING")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
