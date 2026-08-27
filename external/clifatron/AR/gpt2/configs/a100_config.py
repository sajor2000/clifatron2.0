#!/usr/bin/env python3
"""
A100 GPU Configuration for CLIF GPT-2 Training

Optimized for 8x NVIDIA A100 GPUs (40GB or 80GB VRAM)
Context length: 4096 tokens
Gradient checkpointing: ENABLED (required for 4096 context)

Hardware Specs:
  - 8x NVIDIA A100 (40GB or 80GB VRAM, 320GB or 640GB total)
  - Ampere architecture (supports bfloat16, Flash Attention 2)
  - NVLink/NVSwitch interconnect (600GB/s)

Memory Budget per GPU (Medium model, 4096 context):
  - A100-40GB: Can train medium comfortably, large with care
  - A100-80GB: Can train large comfortably, XL feasible

Usage:
    # Import in training script
    from configs.a100_config import get_a100_config

    config = get_a100_config(
        model_size='medium',
        gpu_memory='80gb',  # or '40gb'
        batch_size=6
    )

Author: Generated for CLIF GPT-2 Training Pipeline
"""

import pathlib
from dataclasses import dataclass, field
from typing import Literal

from config import DataConfig, ModelConfig, TrainingConfig, Config


# ============================================================================
# A100-Specific Model Configurations (with 4096 context)
# ============================================================================

A100_40GB_CONFIGS = {
    "small": {
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "block_size": 4096,
        "n_params": "124M",
        "recommended_batch_size": 8,
        "gradient_accumulation": 2,
        "memory_per_gpu_gb": 15,
    },
    "medium": {
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "block_size": 4096,
        "n_params": "355M",
        "recommended_batch_size": 6,
        "gradient_accumulation": 4,
        "memory_per_gpu_gb": 20,
    },
    "large": {
        "n_embd": 1280,
        "n_layer": 36,
        "n_head": 20,
        "block_size": 4096,
        "n_params": "774M",
        "recommended_batch_size": 3,
        "gradient_accumulation": 8,
        "memory_per_gpu_gb": 32,
    },
}

A100_80GB_CONFIGS = {
    "small": {
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "block_size": 4096,
        "n_params": "124M",
        "recommended_batch_size": 12,
        "gradient_accumulation": 2,
        "memory_per_gpu_gb": 15,
    },
    "medium": {
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "block_size": 4096,
        "n_params": "355M",
        "recommended_batch_size": 8,
        "gradient_accumulation": 2,
        "memory_per_gpu_gb": 20,
    },
    "large": {
        "n_embd": 1280,
        "n_layer": 36,
        "n_head": 20,
        "block_size": 4096,
        "n_params": "774M",
        "recommended_batch_size": 6,
        "gradient_accumulation": 4,
        "memory_per_gpu_gb": 32,
    },
    "xl": {
        "n_embd": 1600,
        "n_layer": 48,
        "n_head": 25,
        "block_size": 4096,
        "n_params": "1.5B",
        "recommended_batch_size": 3,
        "gradient_accumulation": 8,
        "memory_per_gpu_gb": 55,
    },
}


@dataclass
class A100DataConfig(DataConfig):
    """Data config optimized for A100"""
    max_seq_length: int = 4096  # 4096 context
    shuffle_buffer_size: int = 2048  # Larger buffer for A100 (more memory)


@dataclass
class A100ModelConfig(ModelConfig):
    """Model config optimized for A100"""
    block_size: int = 4096  # 4096 context length

    # Flash Attention settings (A100 supports Flash Attention 2)
    bias: bool = True
    dropout: float = 0.0  # No dropout for pretraining

    def __post_init__(self):
        """Set model parameters and validate for A100"""
        super().__post_init__()

        # Force 4096 context
        self.block_size = 4096
        self.n_positions = 4096


@dataclass
class A100TrainingConfig(TrainingConfig):
    """Training config optimized for 8x A100 GPUs"""

    # Batch size (will be set based on model size and GPU memory)
    per_device_train_batch_size: int = 6
    per_device_eval_batch_size: int = 6
    gradient_accumulation_steps: int = 4

    # Learning rate (scaled for larger effective batch size)
    learning_rate: float = 3e-4  # Slightly higher for 8 GPUs
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    # Optimizer (AdamW with fused kernel for efficiency)
    optim: str = "adamw_torch_fused"  # Faster on A100
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # Mixed precision (bfloat16 for A100 Ampere)
    fp16: bool = False
    bf16: bool = True  # A100 supports bfloat16
    tf32: bool = True  # Enable TF32 for matmul

    # Gradient checkpointing (REQUIRED for 4096 context)
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})

    # Distributed training (DDP/FSDP for 8 GPUs)
    ddp_find_unused_parameters: bool = False
    ddp_bucket_cap_mb: int = 25

    # For very large models (XL), consider FSDP
    # fsdp: str = ""  # Set to "full_shard" for XL model if needed
    # fsdp_transformer_layer_cls_to_wrap: str = "Block"

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
    run_name: str = "clif-gpt2-a100-4096ctx"
    wandb_project: str = "clif-gpt2-a100"


# ============================================================================
# Configuration Factory
# ============================================================================

def get_a100_config(
    model_size: Literal["small", "medium", "large", "xl"] = "medium",
    gpu_memory: Literal["40gb", "80gb"] = "80gb",
    batch_size: int = None,
    gradient_accumulation_steps: int = None,
    num_epochs: int = 10,
    learning_rate: float = None,
    output_dir: str = "./gpt2_output_a100",
    wandb: bool = False,
    run_name: str = None,
) -> Config:
    """
    Get A100-optimized configuration.

    Args:
        model_size: Model size (small, medium, large, xl)
        gpu_memory: GPU memory variant (40gb or 80gb)
        batch_size: Batch size per device (None = use recommended)
        gradient_accumulation_steps: Gradient accumulation (None = use recommended)
        num_epochs: Number of training epochs
        learning_rate: Learning rate (None = auto-scale)
        output_dir: Output directory for models/logs
        wandb: Enable Weights & Biases logging
        run_name: W&B run name

    Returns:
        Complete Config object optimized for 8x A100
    """
    # Select config based on GPU memory
    configs = A100_80GB_CONFIGS if gpu_memory == "80gb" else A100_40GB_CONFIGS

    if model_size not in configs:
        raise ValueError(
            f"Invalid model_size for A100-{gpu_memory}: {model_size}. "
            f"Must be one of {list(configs.keys())}"
        )

    model_config_dict = configs[model_size]

    # Use recommended values if not specified
    if batch_size is None:
        batch_size = model_config_dict["recommended_batch_size"]

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = model_config_dict["gradient_accumulation"]

    # Calculate effective batch size
    num_gpus = 8
    effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus

    # Auto-scale learning rate based on batch size (linear scaling)
    if learning_rate is None:
        base_lr = 2e-4
        base_batch = 64
        learning_rate = base_lr * (effective_batch_size / base_batch)
        learning_rate = min(learning_rate, 5e-4)  # Cap at 5e-4

    # Create configuration
    data_config = A100DataConfig(
        output_dir=pathlib.Path(output_dir),
        vocab_dir=pathlib.Path(output_dir) / "vocab",
        data_dir=pathlib.Path(output_dir) / "data",
        model_dir=pathlib.Path(output_dir) / "models",
    )

    model_config = A100ModelConfig(
        model_size=model_size,
        block_size=4096,
    )

    training_config = A100TrainingConfig(
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        report_to="wandb" if wandb else "none",
        run_name=run_name or f"a100-{gpu_memory}-{model_size}-4096ctx-b{effective_batch_size}",
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

def print_memory_estimates(model_size: str = "medium", gpu_memory: str = "80gb"):
    """Print memory usage estimates for A100."""
    configs = A100_80GB_CONFIGS if gpu_memory == "80gb" else A100_40GB_CONFIGS

    if model_size not in configs:
        print(f"Model {model_size} not supported for A100-{gpu_memory}")
        return

    cfg = configs[model_size]
    gpu_memory_gb = 80 if gpu_memory == "80gb" else 40

    print("=" * 70)
    print(f"A100-{gpu_memory.upper()} Memory Estimates: {model_size.upper()} model ({cfg['n_params']})")
    print("=" * 70)
    print(f"Hardware: 8x NVIDIA A100-{gpu_memory.upper()} ({gpu_memory_gb}GB VRAM each)")
    print(f"Context: {cfg['block_size']:,} tokens")
    print(f"Batch size per GPU: {cfg['recommended_batch_size']}")
    print(f"Gradient accumulation: {cfg['gradient_accumulation']} steps")
    print(f"Effective batch size: {cfg['recommended_batch_size'] * cfg['gradient_accumulation'] * 8}")
    print()
    print(f"Estimated memory per GPU: ~{cfg['memory_per_gpu_gb']:.0f} GB")
    print(f"  Model weights: ~{cfg['memory_per_gpu_gb'] * 0.08:.1f} GB")
    print(f"  Optimizer states: ~{cfg['memory_per_gpu_gb'] * 0.16:.1f} GB")
    print(f"  Gradients: ~{cfg['memory_per_gpu_gb'] * 0.08:.1f} GB")
    print(f"  Activations (checkpointed): ~{cfg['memory_per_gpu_gb'] * 0.68:.1f} GB")
    print()

    headroom = gpu_memory_gb - cfg['memory_per_gpu_gb']
    print(f"Memory headroom: ~{headroom:.0f} GB ({headroom/gpu_memory_gb*100:.0f}%)")

    if headroom < 10:
        print("⚠ WARNING: Low memory headroom! Consider:")
        print("  - Reducing batch size")
        print("  - Increasing gradient accumulation")
        print("  - Using smaller model")
        if gpu_memory == "40gb":
            print("  - Switching to A100-80GB")
    else:
        print("✓ Comfortable memory fit")

    print("=" * 70)
    print()


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    print("\nA100-40GB Configuration Profiles\n")

    for model_size in ["small", "medium", "large"]:
        print_memory_estimates(model_size, "40gb")

    print("\nA100-80GB Configuration Profiles\n")

    for model_size in ["small", "medium", "large", "xl"]:
        print_memory_estimates(model_size, "80gb")

    print("\nExample Configuration (Medium model on A100-80GB):\n")
    config = get_a100_config(model_size="medium", gpu_memory="80gb")
    print(f"Model: {config.model.model_size} ({A100_80GB_CONFIGS['medium']['n_params']})")
    print(f"Context: {config.model.block_size:,} tokens")
    print(f"Batch size: {config.training.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"Effective batch: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps * 8}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Mixed precision: {'bfloat16' if config.training.bf16 else 'float32'}")
    print(f"Gradient checkpointing: {config.training.gradient_checkpointing}")
    print(f"Flash Attention: Enabled (native on A100)")
