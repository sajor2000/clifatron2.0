#!/usr/bin/env python3

"""
Step 4: Train GPT2 model from scratch on CLIF domain-specific vocabulary

This script:
1. Loads the vocabulary and datasets
2. Initializes a GPT2 model from scratch with custom vocabulary
3. Trains the model using causal language modeling
4. Automatically detects and uses GPU if available, falls back to CPU
5. Supports different model sizes: small, medium, large, xl
"""

import argparse
import json
import os
import pathlib

import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import Config, GPT2_CONFIGS
from dataset import ClifDataset
from model import GPT, GPTConfig  # Import custom GPT model
from utils import (
    setup_logging,
    detect_device,
    print_model_info,
    get_device_info,
)
from vocabulary import Vocabulary

# Import GPU-specific config profiles
try:
    from configs.l40_config import get_l40_config
    from configs.a100_config import get_a100_config
    CONFIG_PROFILES_AVAILABLE = True
except ImportError:
    CONFIG_PROFILES_AVAILABLE = False

logger = setup_logging()


def load_wandb_config():
    """Load W&B API key from clif_config.json if it exists"""
    config_path = pathlib.Path("clif_config.json")
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                if "wandb_api_key" in config and config["wandb_api_key"] != "your_wandb_api_key_here":
                    os.environ["WANDB_API_KEY"] = config["wandb_api_key"]
                    logger.info("Loaded W&B API key from clif_config.json")
                    return True
        except Exception as e:
            logger.warning(f"Failed to load W&B config from clif_config.json: {e}")
    return False


def packed_data_collator(features):
    """Custom data collator for packed sequences (no padding needed)"""
    # For packed sequences, all sequences are already max_length
    # Convert to tensors if needed (from streaming datasets)
    input_ids_list = []
    for f in features:
        # Handle both dict and direct tensor formats
        if isinstance(f, dict):
            if "input_ids" in f:
                ids = f["input_ids"]
            else:
                # Debug: print what keys are available
                print(f"Warning: Expected 'input_ids' key, got keys: {f.keys() if hasattr(f, 'keys') else type(f)}")
                continue
        else:
            ids = f  # Assume it's already the tensor/list

        # Convert to tensor if needed
        if isinstance(ids, list):
            ids = torch.tensor(ids, dtype=torch.long)
        elif not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        input_ids_list.append(ids)

    if not input_ids_list:
        raise ValueError("No valid input_ids found in batch")

    batch = {
        "input_ids": torch.stack(input_ids_list),
    }
    # For causal LM, labels are the same as input_ids (shifted internally by model)
    batch["labels"] = batch["input_ids"].clone()
    return batch


def initialize_model(
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    config: Config,
):
    """
    Initialize custom GPT model from scratch with flash attention support

    Args:
        vocab_size: Size of vocabulary
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        config: Model configuration

    Returns:
        Initialized model
    """
    logger.info(f"Initializing custom GPT model ({config.model.model_size}) from scratch...")

    # Create model configuration using our custom GPTConfig
    model_config = GPTConfig(
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        n_embd=config.model.n_embd,
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        block_size=config.model.block_size,  # Use block_size instead of n_positions
        dropout=config.model.dropout,
        bias=config.model.bias,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )

    # Log model configuration
    gpt2_params = GPT2_CONFIGS[config.model.model_size]
    logger.info(f"Model size: {config.model.model_size} ({gpt2_params['n_params']} parameters)")
    logger.info(f"Model configuration:")
    logger.info(f"  Embedding dimension: {config.model.n_embd}")
    logger.info(f"  Number of layers: {config.model.n_layer}")
    logger.info(f"  Number of heads: {config.model.n_head}")
    logger.info(f"  Context size (block_size): {config.model.block_size}")
    logger.info(f"  Vocabulary size: {vocab_size}")
    logger.info(f"  Dropout: {config.model.dropout}")
    logger.info(f"  Bias in Linear/LayerNorm: {config.model.bias}")
    logger.info(f"  Gradient checkpointing: {'✓ Enabled' if config.training.gradient_checkpointing else '✗ Disabled'}")

    # Check for Flash Attention support
    flash_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    logger.info(f"  Flash Attention: {'✓ Available' if flash_available else '✗ Not available (PyTorch >= 2.0 required)'}")

    # Initialize custom GPT model
    model = GPT(model_config)

    logger.info("Model initialized successfully")
    print_model_info(model)

    return model


def setup_training(
    model,
    train_dataset,
    val_dataset,
    vocab,
    config: Config,
    device,
    use_fp16: bool,
    use_bf16: bool,
):
    """
    Setup training configuration and trainer

    Args:
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        vocab: Vocabulary
        config: Configuration
        device: Device to use
        use_fp16: Whether to use fp16
        use_bf16: Whether to use bf16

    Returns:
        Trainer object
    """
    logger.info("Setting up training configuration...")

    # Create output directory
    output_dir = config.data.model_dir / config.training.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of devices
    if device.type == "cuda":
        n_devices = torch.cuda.device_count()
    else:
        n_devices = 1

    # Calculate training steps
    if hasattr(train_dataset, 'num_rows'):
        # For map-style datasets
        n_train_samples = train_dataset.num_rows
    else:
        # For iterable datasets, estimate from config
        n_train_samples = getattr(train_dataset, 'n_train', 10000)

    steps_per_epoch = (
        n_train_samples
        // config.training.per_device_train_batch_size
        // config.training.gradient_accumulation_steps
        // n_devices
    )
    max_steps = steps_per_epoch * config.training.num_train_epochs

    # Log batch size info
    effective_batch_size = (
        config.training.per_device_train_batch_size
        * config.training.gradient_accumulation_steps
        * n_devices
    )

    logger.info("Training batch configuration:")
    logger.info(f"  Per-device batch size: {config.training.per_device_train_batch_size}")
    logger.info(f"  Number of devices: {n_devices}")
    logger.info(f"  Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"Estimated steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {max_steps}")

    # Setup wandb if requested
    if config.training.report_to == "wandb":
        os.environ["WANDB_PROJECT"] = config.training.wandb_project
        os.environ["WANDB_RUN_NAME"] = config.training.run_name

    # Training arguments
    training_args_dict = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.training.num_train_epochs,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_ratio": config.training.warmup_ratio,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "max_grad_norm": config.training.max_grad_norm,
        "logging_steps": config.training.logging_steps,
        "eval_strategy": config.training.eval_strategy,
        "eval_steps": config.training.eval_steps,
        "save_strategy": config.training.save_strategy,
        "save_steps": config.training.save_steps,
        "save_total_limit": config.training.save_total_limit,
        "load_best_model_at_end": config.training.load_best_model_at_end,
        "metric_for_best_model": config.training.metric_for_best_model,
        "greater_is_better": config.training.greater_is_better,
        "fp16": use_fp16 and device.type == "cuda",
        "bf16": use_bf16 and device.type == "cuda",
        "ddp_find_unused_parameters": config.training.ddp_find_unused_parameters,
        "report_to": config.training.report_to,
        "run_name": config.training.run_name,
        "max_steps": max_steps,
        "save_safetensors": False,  # Disable safetensors due to weight tying in model
    }

    # Add gradient checkpointing if enabled
    if config.training.gradient_checkpointing:
        training_args_dict["gradient_checkpointing"] = True
        if hasattr(config.training, 'gradient_checkpointing_kwargs') and config.training.gradient_checkpointing_kwargs:
            training_args_dict["gradient_checkpointing_kwargs"] = config.training.gradient_checkpointing_kwargs

    training_args = TrainingArguments(**training_args_dict)

    # Use custom data collator for packed sequences
    data_collator = packed_data_collator

    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience
            )
        ],
    )

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train GPT2 model on CLIF data")
    parser.add_argument(
        "--config-profile",
        type=str,
        choices=["l40", "a100-40gb", "a100-80gb"],
        default=None,
        help="GPU config profile (l40: 2x L40, a100-40gb: 8x A100-40GB, a100-80gb: 8x A100-80GB). "
             "Overrides other settings with optimized defaults for hardware.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory containing train/val/test splits",
    )
    parser.add_argument(
        "--vocab-dir",
        type=str,
        default=None,
        help="Vocabulary directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large", "xl"],
        default="medium",
        help="GPT2 model size (default: medium)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device batch size (default: from config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: from config)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: 2, increase for larger effective batch)",
    )
    parser.add_argument(
        "--collation",
        type=str,
        choices=["padded", "packed"],
        default=None,
        help="Collation strategy (default: from config)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Context size / block size (default: 4096 tokens). "
             "Use 1024/2048 for smaller memory, 8192 for longer context.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage (recommended for context >= 4096)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability (default: 0.0 for pretraining, 0.1+ for finetuning)",
    )
    parser.add_argument(
        "--no-bias",
        action="store_true",
        help="Disable bias in Linear/LayerNorm layers (slightly faster and more efficient)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for logging (default: from config)",
    )

    args = parser.parse_args()

    # Load W&B API key from clif_config.json if available
    load_wandb_config()

    # Load configuration from profile or default
    if args.config_profile:
        if not CONFIG_PROFILES_AVAILABLE:
            logger.error("Config profiles not available. Please check configs/l40_config.py and configs/a100_config.py exist.")
            return

        logger.info(f"Loading config profile: {args.config_profile}")

        # Parse profile
        if args.config_profile == "l40":
            # Only pass non-None values to preserve profile defaults
            profile_kwargs = {
                "model_size": args.model_size,
                "num_epochs": args.epochs or 10,
                "output_dir": args.output_dir or "./gpt2_output_l40",
                "wandb": args.wandb,
            }
            if args.batch_size is not None:
                profile_kwargs["batch_size"] = args.batch_size
            if args.gradient_accumulation_steps is not None:
                profile_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            if args.learning_rate is not None:
                profile_kwargs["learning_rate"] = args.learning_rate
            if args.run_name is not None:
                profile_kwargs["run_name"] = args.run_name

            config = get_l40_config(**profile_kwargs)

        elif args.config_profile.startswith("a100"):
            gpu_memory = "80gb" if args.config_profile == "a100-80gb" else "40gb"

            # Only pass non-None values to preserve profile defaults
            profile_kwargs = {
                "model_size": args.model_size,
                "gpu_memory": gpu_memory,
                "num_epochs": args.epochs or 10,
                "output_dir": args.output_dir or f"./gpt2_output_a100_{gpu_memory}",
                "wandb": args.wandb,
            }
            if args.batch_size is not None:
                profile_kwargs["batch_size"] = args.batch_size
            if args.gradient_accumulation_steps is not None:
                profile_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
            if args.learning_rate is not None:
                profile_kwargs["learning_rate"] = args.learning_rate
            if args.run_name is not None:
                profile_kwargs["run_name"] = args.run_name

            config = get_a100_config(**profile_kwargs)

        logger.info(f"  ✓ Loaded {args.config_profile} profile with {args.model_size} model")
    else:
        # Load default configuration
        config = Config()

        # Set model size first (before __post_init__ runs)
        config.model.model_size = args.model_size
        config.model.__post_init__()

    # Override config with command-line arguments
    if args.data_dir:
        config.data.data_dir = pathlib.Path(args.data_dir)
    if args.vocab_dir:
        config.data.vocab_dir = pathlib.Path(args.vocab_dir)
    if args.output_dir:
        config.data.model_dir = pathlib.Path(args.output_dir)
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.gradient_accumulation_steps:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.collation:
        config.training.collation = args.collation
    if args.context_size:
        config.model.block_size = args.context_size
        config.data.max_seq_length = args.context_size  # Update data processing too
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.no_bias:
        config.model.bias = False
    if args.gradient_checkpointing:
        config.training.gradient_checkpointing = True
    if args.wandb:
        config.training.report_to = "wandb"
    if args.run_name:
        config.training.run_name = args.run_name
    else:
        # Update default run name to include model size and context size
        context_suffix = f"-ctx{config.model.block_size}" if config.model.block_size != 4096 else ""
        config.training.run_name = f"clif-gpt2-{args.model_size}{context_suffix}"

    # Print system information
    logger.info("=" * 60)
    logger.info(f"CLIF GPT2 Training ({args.model_size})")
    logger.info("=" * 60)

    device_info = get_device_info()
    logger.info("\nSystem Information:")
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")

    # Detect device
    device, use_fp16, use_bf16 = detect_device()
    logger.info(f"\nUsing device: {device}")

    # Load vocabulary - try vocab_lock.json first, then fall back to vocab.gzip
    vocab_lock_path = config.data.vocab_dir / "vocab_lock.json"
    vocab_gzip_path = config.data.vocab_dir / "vocab.gzip"

    if vocab_lock_path.exists():
        logger.info(f"\nLoading vocabulary from: {vocab_lock_path}")
        vocab = Vocabulary.from_vocab_lock(vocab_lock_path)
        logger.info(f"Vocabulary size: {len(vocab)}")
        logger.info(f"Vocabulary hash: {vocab.get_vocab_hash()[:16]}...")
    elif vocab_gzip_path.exists():
        logger.info(f"\nLoading vocabulary from: {vocab_gzip_path} (legacy format)")
        vocab = Vocabulary().load(vocab_gzip_path)
        logger.info(f"Vocabulary size: {len(vocab)}")
    else:
        raise FileNotFoundError(
            f"Vocabulary not found at {vocab_lock_path} or {vocab_gzip_path}\n"
            "Please run scripts/build_vocab_from_data.py first"
        )

    # Get special token IDs (updated to gpt2_hf style)
    bos_token_id = vocab("[BOS]")
    eos_token_id = vocab("[EOS]")
    pad_token_id = vocab("[PAD]")

    logger.info(f"Special tokens:")
    logger.info(f"  BOS: {bos_token_id}")
    logger.info(f"  EOS: {eos_token_id}")
    logger.info(f"  PAD: {pad_token_id}")

    # Load datasets
    logger.info(f"\nLoading datasets from: {config.data.data_dir}")
    # Use the vocab path that actually exists (prefer vocab_lock.json)
    active_vocab_path = vocab_lock_path if vocab_lock_path.exists() else vocab_gzip_path
    dataset = ClifDataset(
        data_dir=config.data.data_dir,
        vocab_path=active_vocab_path,
        collation=config.training.collation,
        max_seq_length=config.data.max_seq_length,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
    )

    logger.info(f"Train samples: {dataset.n_train}")
    logger.info(f"Validation samples: {dataset.n_val}")
    logger.info(f"Test samples: {dataset.n_test}")

    # Initialize model
    model = initialize_model(
        vocab_size=len(vocab),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        config=config,
    )

    # Get datasets
    train_dataset = dataset.get_train_dataset(n_epochs=config.training.num_train_epochs)
    val_dataset = dataset.get_val_dataset()

    # Setup training
    trainer = setup_training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        vocab=vocab,
        config=config,
        device=device,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
    )

    # Train model
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60 + "\n")

    trainer.train()

    # Save final model
    final_model_path = config.data.model_dir / config.training.run_name / "final_model"
    logger.info(f"\nSaving final model to: {final_model_path}")
    trainer.save_model(str(final_model_path))

    # Save vocabulary with model (both formats for compatibility)
    vocab.save(final_model_path / "vocab.gzip")
    vocab.save_to_vocab_lock(final_model_path / "vocab_lock.json")
    logger.info("Saved vocabulary with model (vocab.gzip + vocab_lock.json)")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
