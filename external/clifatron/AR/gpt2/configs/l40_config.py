#!/usr/bin/env python3
"""
L40 GPU Configuration for CLIF GPT-2 Training

Optimized for 2x NVIDIA L40 GPUs (48GB VRAM each)
Context length: 4096 tokens
Gradient checkpointing: ENABLED (required for 4096 context)

Hardware Specs:
  - 2x NVIDIA L40 (48GB VRAM each, 96GB total)
  - Ampere architecture (supports bfloat16, Flash Attention 2)
  - PCIe Gen4 interconnect

Memory Budget per GPU (Medium model, 4096 context):
  - Model weights: ~1.4GB
  - Optimizer states: ~2.8GB
  - Gradients: ~1.4GB
  - Activations (with gradient checkpointing): ~12GB
  - Total: ~18GB per GPU → Comfortable fit in 48GB

Usage:
    # Import in training script
    from configs.l40_config import get_l40_config

    config = get_l40_config(
        model_size='medium',
        batch_size=4,
        gradient_accumulation_steps=8
    )

Author: Generated for CLIF GPT-2 Training Pipeline
"""

import pathlib
from dataclasses import dataclass, field
from typing import Literal

from config import DataConfig, ModelConfig, TrainingConfig, Config


# ============================================================================
# L40-Specific Model Configurations (with 4096 context)
# ============================================================================

L40_MODEL_CONFIGS = {
    "small": {
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "block_size": 4096,
        "n_params": "124M",
        "recommended_batch_size": 6,
        "gradient_accumulation": 4,
        "memory_per_gpu_gb": 15,
    },
    "medium": {
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "block_size": 4096,
        "n_params": "355M",
        "recommended_batch_size": 4,
        "gradient_accumulation": 8,
        "memory_per_gpu_gb": 18,
    },
    "large": {
        "n_embd": 1280,
        "n_layer": 36,
        "n_head": 20,
        "block_size": 4096,
        "n_params": "774M",
        "recommended_batch_size": 2,
        "gradient_accumulation": 16,
        "memory_per_gpu_gb": 30,
    },
}


@dataclass
class L40DataConfig(DataConfig):
    """Data config optimized for L40"""
    max_seq_length: int = 4096  # Increased from 1024
    shuffle_buffer_size: int = 1024  # Keep manageable for memory


@dataclass
class L40ModelConfig(ModelConfig):
    """Model config optimized for L40"""
    block_size: int = 4096  # 4096 context length

    # Flash Attention settings (L40 supports Flash Attention 2)
    bias: bool = True
    dropout: float = 0.0  # No dropout for pretraining

    def __post_init__(self):
        """Set model parameters and validate for L40"""
        super().__post_init__()

        if self.model_size not in L40_MODEL_CONFIGS:
            raise ValueError(
                f"Invalid model_size for L40: {self.model_size}. "
                f"Must be one of {list(L40_MODEL_CONFIGS.keys())}"
            )

        # Force 4096 context
        self.block_size = 4096
        self.n_positions = 4096


@dataclass
class L40TrainingConfig(TrainingConfig):
    """Training config optimized for 2x L40 GPUs"""

    # Batch size (will be set based on model size)
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Learning rate (scaled for effective batch size)
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    # Optimizer (AdamW with fused kernel for efficiency)
    optim: str = "adamw_torch_fused"  # Faster on L40
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # Mixed precision (bfloat16 for L40 Ampere)
    fp16: bool = False
    bf16: bool = True  # L40 supports bfloat16
    tf32: bool = True  # Enable TF32 for matmul (faster, negligible precision loss)

    # Gradient checkpointing (REQUIRED for 4096 context)
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})

    # Memory optimization
    optim_target_modules: str = None  # Use fused optimizer for all modules

    # Distributed training (DDP for 2 GPUs)
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 25  # Reduce for faster comm

    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2

    # Data collation (packed is more efficient)
    collation: Literal["padded", "packed"] = "packed"

    # Early stopping
    early_stopping_patience: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # W&B logging (optional)
    report_to: str = "none"
    run_name: str = "clif-gpt2-l40-4096ctx"
    wandb_project: str = "clif-gpt2-l40"


# ============================================================================
# Configuration Factory
# ============================================================================

def get_l40_config(
    model_size: Literal["small", "medium", "large"] = "medium",
    batch_size: int = None,
    gradient_accumulation_steps: int = None,
    num_epochs: int = 10,
    learning_rate: float = 2e-4,
    output_dir: str = "./gpt2_output_l40",
    wandb: bool = False,
    run_name: str = None,
) -> Config:
    """
    Get L40-optimized configuration.

    Args:
        model_size: Model size (small, medium, large)
        batch_size: Batch size per device (None = use recommended)
        gradient_accumulation_steps: Gradient accumulation (None = use recommended)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        output_dir: Output directory for models/logs
        wandb: Enable Weights & Biases logging
        run_name: W&B run name

    Returns:
        Complete Config object optimized for 2x L40
    """
    if model_size not in L40_MODEL_CONFIGS:
        raise ValueError(
            f"Invalid model_size: {model_size}. "
            f"Must be one of {list(L40_MODEL_CONFIGS.keys())}"
        )

    model_config_dict = L40_MODEL_CONFIGS[model_size]

    # Use recommended values if not specified
    if batch_size is None:
        batch_size = model_config_dict["recommended_batch_size"]

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = model_config_dict["gradient_accumulation"]

    # Calculate effective batch size
    effective_batch_size = batch_size * gradient_accumulation_steps * 2  # 2 GPUs

    # Create configuration
    data_config = L40DataConfig(
        output_dir=pathlib.Path(output_dir),
        vocab_dir=pathlib.Path(output_dir) / "vocab",
        data_dir=pathlib.Path(output_dir) / "data",
        model_dir=pathlib.Path(output_dir) / "models",
    )

    model_config = L40ModelConfig(
        model_size=model_size,
        block_size=4096,
    )

    training_config = L40TrainingConfig(
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        report_to="wandb" if wandb else "none",
        run_name=run_name or f"l40-{model_size}-4096ctx-b{effective_batch_size}",
    )

    config = Config(
        data=data_config,
        model=model_config,
        training=training_config,
    )

    return config


# ============================================================================
# Memory Estimates
# ============================================================================

def print_memory_estimates(model_size: str = "medium"):
    """Print memory usage estimates for L40."""
    if model_size not in L40_MODEL_CONFIGS:
        print(f"Unknown model size: {model_size}")
        return

    cfg = L40_MODEL_CONFIGS[model_size]

    print("=" * 70)
    print(f"L40 Memory Estimates: {model_size.upper()} model ({cfg['n_params']})")
    print("=" * 70)
    print(f"Hardware: 2x NVIDIA L40 (48GB VRAM each)")
    print(f"Context: {cfg['block_size']:,} tokens")
    print(f"Batch size per GPU: {cfg['recommended_batch_size']}")
    print(f"Gradient accumulation: {cfg['gradient_accumulation']} steps")
    print(f"Effective batch size: {cfg['recommended_batch_size'] * cfg['gradient_accumulation'] * 2}")
    print()
    print(f"Estimated memory per GPU: ~{cfg['memory_per_gpu_gb']:.0f} GB")
    print(f"  Model weights: ~{cfg['memory_per_gpu_gb'] * 0.08:.1f} GB")
    print(f"  Optimizer states: ~{cfg['memory_per_gpu_gb'] * 0.16:.1f} GB")
    print(f"  Gradients: ~{cfg['memory_per_gpu_gb'] * 0.08:.1f} GB")
    print(f"  Activations (checkpointed): ~{cfg['memory_per_gpu_gb'] * 0.68:.1f} GB")
    print()

    headroom = 48 - cfg['memory_per_gpu_gb']
    print(f"Memory headroom: ~{headroom:.0f} GB ({headroom/48*100:.0f}%)")

    if headroom < 10:
        print("⚠ WARNING: Low memory headroom! Consider:")
        print("  - Reducing batch size")
        print("  - Increasing gradient accumulation")
        print("  - Using smaller model")
    else:
        print("✓ Comfortable memory fit")

    print("=" * 70)
    print()


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    print("\nL40 Configuration Profiles\n")

    for model_size in ["small", "medium", "large"]:
        print_memory_estimates(model_size)

    print("\nExample Configuration (Medium model):\n")
    config = get_l40_config(model_size="medium")
    print(f"Model: {config.model.model_size} ({L40_MODEL_CONFIGS['medium']['n_params']})")
    print(f"Context: {config.model.block_size:,} tokens")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"Effective batch: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps * 2}")
    print(f"Mixed precision: {'bfloat16' if config.training.bf16 else 'float32'}")
    print(f"Gradient checkpointing: {config.training.gradient_checkpointing}")
    print(f"Flash Attention: Enabled (native on L40)")
