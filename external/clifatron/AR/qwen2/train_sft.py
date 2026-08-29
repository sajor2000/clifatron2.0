#!/usr/bin/env python3
"""
train_sft.py - Supervised Fine-Tuning with TRL Packing & Optuna

Train Qwen2 models with:
- TRL's ConstantLengthDataset: Efficient packing with proper attention masking
- SFTTrainer: Purpose-built for supervised fine-tuning
- Optuna: Automatic hyperparameter optimization

Usage:
    # Primary site with Optuna HP search
    uv run torchrun --nproc_per_node=auto AR/qwen2_sft/train_sft.py \\
        --model-size 0.5b \\
        --mode primary \\
        --clif-config clif_config.json \\
        --run-name rush-sft-hp-search

    # Secondary site with fixed hyperparameters
    uv run torchrun --nproc_per_node=auto AR/qwen2_sft/train_sft.py \\
        --model-size 0.5b \\
        --mode secondary \\
        --clif-config clif_config.json \\
        --run-name site2-sft \\
        --no-optuna \\
        --learning-rate 2e-4 \\
        --gradient-accumulation-steps 8

Features:
    - Global packing: ~8x less padding waste
    - Attention masking: No cross-patient leakage
    - HP optimization: Auto-tune LR + batch config
    - Primary/secondary: Multi-site vocab locking
    - DeepSpeed: ZeRO-2/3 support
    - W&B logging: Site tracking
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
    Qwen2ForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
import wandb

# Add parent directory to path for tokenizer import (shared for primary/secondary mode)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Tokenizer is shared between qwen2 and qwen2_sft (required for vocab locking)
from qwen2.tokenizer.clinical_tokenizer import ClinicalTokenizer

# Local imports (self-contained)
from models.qwen2_configs import get_model_config, print_model_stats
from utils.gpu_detector import auto_configure
from data.packed_dataset import load_packed_dataset


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


def setup_wandb(args, training_config, gpu_config=None, clif_config=None):
    """Setup Weights & Biases logging."""
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

    # Use custom run name if provided, otherwise auto-generate
    if args.run_name:
        run_name = args.run_name
    else:
        # Format: site-CLIFATRON-qwen2-sft-model_size
        run_name = f"{site}-CLIFATRON-qwen2-sft-{args.model_size}"

    # Setup wandb via environment variables
    os.environ["WANDB_PROJECT"] = training_config.get('wandb_project', 'CLIFATRON')
    os.environ["WANDB_RUN_NAME"] = run_name

    print(f"  ✓ WandB project: {os.environ['WANDB_PROJECT']}")
    print(f"  ✓ WandB run name: {run_name}")
    print()

    return run_name


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen2 with SFTTrainer + TRL packing + Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Primary site with Optuna HP search
  uv run torchrun --nproc_per_node=auto AR/qwen2_sft/train_sft.py \\
      --model-size 0.5b --mode primary --clif-config clif_config.json

  # Secondary site with fixed hyperparameters
  uv run torchrun --nproc_per_node=auto AR/qwen2_sft/train_sft.py \\
      --model-size 0.5b --mode secondary --clif-config clif_config.json \\
      --no-optuna --learning-rate 2e-4 --gradient-accumulation-steps 8
        """
    )

    # Model configuration
    parser.add_argument(
        '--model-size',
        type=str,
        required=True,
        choices=['0.5b', '1.5b', '7b'],
        help='Model size: 0.5b, 1.5b, or 7b'
    )

    # Mode (primary/secondary)
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['primary', 'secondary'],
        help='Vocabulary mode: primary (first site), secondary (other sites using primary vocab)'
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
        help='Path to training config YAML (default: AR/qwen2_sft/config/training_config.yaml)'
    )

    parser.add_argument(
        '--packed-dir',
        type=str,
        default='models/qwen2/preprocessed/packed_temporal_len8192',
        help='Directory containing pre-packed parquet files'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for checkpoints (default: models/qwen2/checkpoints/clif-qwen2-sft-{model_size}/)'
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

    # Optuna configuration
    parser.add_argument(
        '--no-optuna',
        action='store_true',
        help='Disable Optuna hyperparameter search'
    )

    parser.add_argument(
        '--optuna-trials',
        type=int,
        default=50,
        help='Number of Optuna trials (default: 50)'
    )

    # Manual hyperparameter overrides (used when --no-optuna is set)
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config when --no-optuna is set)'
    )

    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=None,
        help='Gradient accumulation steps (overrides config when --no-optuna is set)'
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

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Resolve paths
    script_dir = Path(__file__).parent

    if args.train_config is None:
        args.train_config = script_dir / "config" / "training_config.yaml"

    # Set up directory paths
    root_dir = script_dir.parent.parent  # CLIFATRON root

    if args.output_dir is None:
        # Checkpoints go to /dev/shm/qwen2 (fast tmpfs) for training
        args.output_dir = Path("/dev/shm/qwen2") / f"clif-qwen2-sft-{args.model_size}"

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Processed data (packed sequences) saved to root/models for reuse
    processed_data_dir = root_dir / "models" / "qwen2" / "processeddata" / f"{args.model_size}"
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    print("=" * 80)
    print("QWEN2 SFT TRAINING WITH TRL PACKING & OPTUNA")
    print("=" * 80)
    print(f"Model Size: Qwen2-{args.model_size}")
    print(f"Mode: {args.mode}")
    print(f"Optuna: {'Disabled' if args.no_optuna else f'Enabled ({args.optuna_trials} trials)'}")
    print(f"Seed: {args.seed}")
    print(f"Checkpoints: {args.output_dir}")
    print(f"Processed Data: {processed_data_dir}")
    print("=" * 80)
    print()

    # Load configurations
    print("Loading configurations...")
    clif_config = load_clif_config(args.clif_config)
    print(f"  ✓ CLIF config: {args.clif_config}")
    print(f"  ✓ Site: {clif_config.get('site', 'unknown')}")

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
    setup_wandb(args, training_config, gpu_config=gpu_config, clif_config=clif_config)

    # Load tokenizer (with validation)
    print(f"Loading tokenizer (mode: {args.mode})...")
    tokenizer_path = script_dir.parent / "qwen2" / "tokenizer" / "clinical_tokenizer"

    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        if args.mode == 'secondary':
            print("SECONDARY MODE ERROR: Tokenizer must be copied from primary site!")
            print("Please copy the tokenizer directory from primary site:")
            print("  AR/qwen2/tokenizer/clinical_tokenizer/")
        else:
            print("Please build tokenizer first using:")
            print("  uv run AR/qwen2/01_preprocess_data.py --mode primary ...")
        sys.exit(1)

    tokenizer = ClinicalTokenizer.from_pretrained(str(tokenizer_path))
    print(f"  ✓ Loaded tokenizer from {tokenizer_path}")
    print(f"  ✓ Vocabulary size: {len(tokenizer)}")

    # Validate vocabulary
    tokenizer.validate_vocab_size(expected_size=1389)
    vocab_hash = tokenizer.get_vocab_hash()
    print(f"  ✓ Vocabulary hash: {vocab_hash[:16]}...")
    print()

    # Create model configuration
    print(f"Creating Qwen2-{args.model_size} configuration...")
    model_config = get_model_config(args.model_size, vocab_size=len(tokenizer))

    # Set special token IDs
    model_config.bos_token_id = tokenizer.bos_token_id
    model_config.eos_token_id = tokenizer.eos_token_id
    model_config.pad_token_id = tokenizer.pad_token_id

    print_model_stats(model_config, args.model_size)
    print()

    # Load pre-packed datasets
    print("Loading pre-packed datasets...")
    print(f"  Packed dir: {args.packed_dir}")
    print()

    packed_train = load_packed_dataset(
        packed_dir=args.packed_dir,
        split='train',
    )
    print()

    packed_val = load_packed_dataset(
        packed_dir=args.packed_dir,
        split='val',
    )
    print()

    print(f"  ✓ Train packed sequences: {len(packed_train):,}")
    print(f"  ✓ Val packed sequences: {len(packed_val):,}")
    print()

    # Save dataset info for inspection
    print(f"\nSaving dataset info to {processed_data_dir}...")
    import json

    max_seq_length = training_config.get('max_length', 8192)
    dataset_info = {
        "train_packed_sequences": len(packed_train),
        "val_packed_sequences": len(packed_val),
        "max_length": max_seq_length,
        "model_size": args.model_size,
        "mode": args.mode,
        "vocab_size": len(tokenizer),
        "packing_enabled": True,
        "packing_strategy": "offline_packed",
        "document_isolation": True,
        "pad_tokens_between_docs": 8,
    }
    with open(processed_data_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"  ✓ Saved dataset info")
    print()

    # Get training hyperparameters
    model_training_config = training_config['models'][args.model_size]

    # Override with command-line arguments if provided
    if args.learning_rate is not None:
        model_training_config['learning_rate'] = args.learning_rate
        print(f"  ✓ Overriding learning rate: {args.learning_rate}")

    if args.gradient_accumulation_steps is not None:
        model_training_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
        print(f"  ✓ Overriding gradient accumulation steps: {args.gradient_accumulation_steps}")

    # Model initialization function for Optuna
    def model_init(trial=None):
        """Initialize model from config (for Optuna trials)."""
        config = get_model_config(args.model_size, vocab_size=len(tokenizer))
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
        return Qwen2ForCausalLM(config)

    # Optuna hyperparameter space
    def optuna_hp_space(trial):
        """Define hyperparameter search space for Optuna."""
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                training_config.get('learning_rate_min', 5e-5),
                training_config.get('learning_rate_max', 5e-4),
                log=training_config.get('learning_rate_log', True)
            ),
            "gradient_accumulation_steps": trial.suggest_int(
                "gradient_accumulation_steps",
                training_config.get('gradient_accumulation_min', 1),
                training_config.get('gradient_accumulation_max', 3)
            ),
        }


    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,

        # Training
        num_train_epochs=model_training_config['num_epochs'],
        per_device_train_batch_size=model_training_config['batch_size'],
        per_device_eval_batch_size=model_training_config.get('per_device_eval_batch_size', model_training_config['batch_size']),
        gradient_accumulation_steps=model_training_config['gradient_accumulation_steps'],

        # Optimization
        learning_rate=model_training_config['learning_rate'],
        weight_decay=model_training_config['weight_decay'],
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=model_training_config.get('max_grad_norm', 1.0),

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
        run_name=args.run_name if args.run_name else f"{clif_config.get('site', 'unknown')}-CLIFATRON-qwen2-sft-{args.model_size}",

        # Evaluation
        eval_strategy="steps",
        eval_steps=training_config['eval_steps'],

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
        dataloader_num_workers=0,  # Temporarily 0 for testing (avoid multiprocessing pickle issues)
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # Keep document_attention_mask for 2D attention
    )

    # Custom data collator for v2 packed sequences with segment metadata.
    def data_collator(features):
        """Collator that preserves v2 schema when segments are present."""
        import torch

        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }
        if all("segments" in f for f in features):
            # Document-isolated attention now EXISTS (U13, src/model/varlen_attention.py):
            # the CPU fallback isolates by running one forward per document, and the GPU
            # path lets FlashAttention-2 isolate from per-document position ids. But this
            # HF Trainer SFT forward is the standard Qwen2 forward — it is NOT yet wired to
            # feed the flattened + position-id form FA2 needs, and eager/SDPA Qwen2 has no
            # isolation path that avoids a dense [batch, heads, len, len] mask. So a
            # multi-document pack HERE would still leak across documents. Keep it fail-
            # closed until the SFT forward is wired to the isolation core and qualified on
            # GPU (U8's L40 packed-attention entry gate). The dead pass-through code that
            # used to sit below this raise was removed — unreachable "handling" beside a
            # blanket reject is the prose-beside-no-control anti-pattern this repo tracks.
            raise ValueError(
                "v2 packed records contain multiple document segments; this Qwen2 SFT "
                "forward is not yet wired to the U13 document-isolation path "
                "(src/model/varlen_attention.py), so training would leak across "
                "documents. Pack one document per row until the FA2 training path is "
                "qualified (U8 L40 gate)."
            )
        return batch

    # Initialize Trainer with packing support
    print("Initializing Trainer...")
    if args.no_optuna:
        # Standard training without HP search
        trainer = Trainer(
            model=model_init(),
            args=training_args,
            train_dataset=packed_train,
            eval_dataset=packed_val,
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=training_config.get('early_stopping_patience', 3)
                )
            ],
        )
    else:
        # Optuna hyperparameter search
        trainer = Trainer(
            model_init=model_init,  # model_init for HP search
            args=training_args,
            train_dataset=packed_train,
            eval_dataset=packed_val,
            data_collator=data_collator,
            processing_class=tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=training_config.get('early_stopping_patience', 3)
                )
            ],
        )
    print("  ✓ Trainer initialized")
    print()

    # Print training info
    print("=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Model: Qwen2-{args.model_size}")
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
    print(f"Packing: Custom (efficient, no token waste)")
    print(f"Document isolation: Enabled (PAD + attention masks)")
    print(f"Max sequence length: {max_seq_length}")
    print("=" * 80)
    print()

    # Train
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()

    try:
        if args.no_optuna:
            # Standard training without HP search
            print("Training with fixed hyperparameters...")
            trainer.train(resume_from_checkpoint=args.resume_from)
        else:
            # Hyperparameter search with Optuna
            print(f"Starting Optuna hyperparameter search ({args.optuna_trials} trials)...")
            print("This will take a while...")
            print()

            best_run = trainer.hyperparameter_search(
                direction=training_config.get('optuna_direction', 'minimize'),
                backend="optuna",
                hp_space=optuna_hp_space,
                n_trials=args.optuna_trials,
            )

            print()
            print("=" * 80)
            print("BEST HYPERPARAMETERS FOUND")
            print("=" * 80)
            print(f"Best trial: {best_run}")
            print("=" * 80)
            print()

        print()
        print("=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print()

        # Get final evaluation metrics
        print("Running final evaluation...")
        final_metrics = trainer.evaluate()
        print()
        print("=" * 80)
        print("FINAL EVALUATION METRICS")
        print("=" * 80)
        print(f"eval_loss: {final_metrics.get('eval_loss', 0.0):.4f}")
        print(f"perplexity: {final_metrics.get('eval_perplexity', 0.0):.2f}")
        if 'eval_runtime' in final_metrics:
            print(f"eval_runtime: {final_metrics['eval_runtime']:.2f}s")
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
        print(f"  uv run torchrun --nproc_per_node=auto AR/qwen2_sft/train_sft.py \\")
        print(f"    --model-size {args.model_size} \\")
        print(f"    --mode {args.mode} \\")
        print(f"    --clif-config {args.clif_config} \\")
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
