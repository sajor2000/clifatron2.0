#!/usr/bin/env python3

"""
Configuration parameters for CLIF GPT2 training
"""

import pathlib
from dataclasses import dataclass, field
from typing import Literal


# GPT2 model size configurations
# Note: block_size (context length) can be overridden at runtime
GPT2_CONFIGS = {
    "small": {
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "block_size": 1024,  # context size (can be increased to 2048, 4096, etc.)
        "n_params": "124M",
    },
    "medium": {
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "block_size": 1024,  # context size (can be increased to 2048, 4096, etc.)
        "n_params": "355M",
    },
    "large": {
        "n_embd": 1280,
        "n_layer": 36,
        "n_head": 20,
        "block_size": 1024,  # context size (can be increased to 2048, 4096, etc.)
        "n_params": "774M",
    },
    "xl": {
        "n_embd": 1600,
        "n_layer": 48,
        "n_head": 25,
        "block_size": 1024,  # context size (can be increased to 2048, 4096, etc.)
        "n_params": "1.5B",
    },
}


@dataclass
class DataConfig:
    """Data-related configuration"""
    # Input data from tokenization_example.py
    clif_sentences_path: str = "clif_sentences.parquet"

    # Output directories
    output_dir: pathlib.Path = pathlib.Path("./gpt2_output")
    vocab_dir: pathlib.Path = pathlib.Path("./gpt2_output/vocab")
    data_dir: pathlib.Path = pathlib.Path("./gpt2_output/data")
    model_dir: pathlib.Path = pathlib.Path("./gpt2_output/models")

    # Train/val/test split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Random seed for reproducibility
    random_seed: int = 42

    # Sequence parameters
    max_seq_length: int = 1024
    shuffle_buffer_size: int = 1024


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Model size: small, medium, large, xl
    model_size: Literal["small", "medium", "large", "xl"] = "small"

    # Base model architecture
    model_name: str = "gpt2"

    # Model architecture parameters (will be set based on model_size)
    n_embd: int = None  # Embedding dimension
    n_layer: int = None  # Number of transformer layers
    n_head: int = None  # Number of attention heads
    block_size: int = 1024  # Maximum sequence length / context size

    # Keep n_positions for backward compatibility (will be set to block_size)
    n_positions: int = None

    # Flash Attention and efficiency options
    bias: bool = True  # True: use bias in Linear/LayerNorm. False: more efficient
    dropout: float = 0.0  # Dropout (0.0 for pretraining, 0.1+ for finetuning)

    # Additional GPT2 parameters (for HuggingFace compatibility if needed)
    resid_pdrop: float = None  # Will be set to dropout
    embd_pdrop: float = None  # Will be set to dropout
    attn_pdrop: float = None  # Will be set to dropout
    layer_norm_epsilon: float = 1e-5

    # Activation function
    activation_function: str = "gelu_new"

    # Vocab will be set dynamically from data
    vocab_size: int = None  # Will be set after vocabulary is built

    # Special tokens (will be set from vocabulary)
    bos_token_id: int = None
    eos_token_id: int = None
    pad_token_id: int = None

    def __post_init__(self):
        """Set model parameters based on model_size"""
        if self.model_size not in GPT2_CONFIGS:
            raise ValueError(
                f"Invalid model_size: {self.model_size}. "
                f"Must be one of {list(GPT2_CONFIGS.keys())}"
            )

        config = GPT2_CONFIGS[self.model_size]
        self.n_embd = config["n_embd"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]

        # Set block_size from config if not already set
        if self.block_size == 1024:  # If using default
            self.block_size = config["block_size"]

        # Set n_positions to match block_size for backward compatibility
        self.n_positions = self.block_size

        # Set dropout values for HuggingFace compatibility
        self.resid_pdrop = self.dropout
        self.embd_pdrop = self.dropout
        self.attn_pdrop = self.dropout

        self.model_name = f"gpt2-{self.model_size}"


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Training parameters
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03

    # Optimization
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # Logging and checkpointing
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2

    # Evaluation and early stopping
    eval_strategy: Literal["steps", "epoch"] = "steps"
    save_strategy: Literal["steps", "epoch"] = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    early_stopping_patience: int = 3

    # Data collation strategy
    collation: Literal["padded", "packed"] = "packed"

    # Mixed precision training
    fp16: bool = False  # Will be set based on GPU availability
    bf16: bool = False  # Will be set based on GPU availability

    # Distributed training
    ddp_find_unused_parameters: bool = False

    # Weights & Biases
    report_to: str = "none"  # Change to "wandb" if you want to use W&B
    run_name: str = "clif-gpt2"
    wandb_project: str = "clif-gpt2-training"


@dataclass
class Config:
    """Complete configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Create output directories"""
        for dir_path in [
            self.data.output_dir,
            self.data.vocab_dir,
            self.data.data_dir,
            self.data.model_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
        }


# Default configuration instance
default_config = Config()
