"""
Model loading utilities for benchmarking

Provides universal model loader for all AR models (gpt2, gpt2_hf, qwen2).
"""

import os
import torch
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging

# Import GPU utilities
from .gpu_utils import try_load_model_on_device

logger = logging.getLogger(__name__)


def get_default_vocab_path(
    model_type: str,
    root_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Get default vocabulary path for a given model type.

    Args:
        model_type: Model type (gpt2, gpt2_hf, qwen2)
        root_dir: Root directory of CLIFATRON project (optional, auto-detected if not provided)

    Returns:
        Path to vocabulary file
    """
    # Auto-detect root directory if not provided
    if root_dir is None:
        # Try environment variable first
        root_dir = os.getenv("CLIFATRON_ROOT")
        if root_dir is None:
            # Fallback: detect from script location (go up from benchmark/utils)
            root_dir = Path(__file__).parent.parent.parent

    root_dir = Path(root_dir)
    model_type = model_type.lower()

    # Check in models/ folder (new reorganized structure)
    vocab_paths = []

    # All models now have vocab files in their model directories
    if model_type == "gpt2":
        vocab_paths = [
            root_dir / "models" / "gpt2" / "vocab.json",
            root_dir / "models" / "gpt2" / "vocab.gzip",
        ]
    elif model_type == "gpt2_hf":
        vocab_paths = [
            root_dir / "models" / "gpt2_hf" / "vocab.json",
        ]
    elif model_type in ["qwen2", "qwen2optuna"]:
        # qwen2 and qwen2optuna use the same vocab structure
        model_folder = "qwen2" if model_type == "qwen2" else "qwen2optuna"
        vocab_paths = [
            root_dir / "models" / model_folder / "vocab.json",
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: gpt2, gpt2_hf, qwen2, qwen2optuna")

    # Return first existing path
    for path in vocab_paths:
        if path.exists():
            return path

    # If none exist, return the first preferred path with a warning
    logger.warning(f"Vocab file not found in expected locations for {model_type}. Returning default path: {vocab_paths[0]}")
    return vocab_paths[0]


def load_vocabulary(vocab_path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load vocabulary from token registry JSON.

    Args:
        vocab_path: Path to vocab.json

    Returns:
        Tuple of (token_to_id, id_to_token) dictionaries
    """
    vocab_path = Path(vocab_path)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    logger.info(f"Loading vocabulary from {vocab_path}")

    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    # Create token_to_id and id_to_token mappings
    token_to_id = {}
    id_to_token = {}

    # Handle "vocab" wrapper (Qwen2 format)
    if "vocab" in vocab_data and isinstance(vocab_data["vocab"], dict):
        vocab_data = vocab_data["vocab"]

    # Check if vocab has direct token->id mapping or nested structure
    first_value = next(iter(vocab_data.values()))

    if isinstance(first_value, dict):
        # Nested structure: {"token": {"id": x}}
        for token_name, token_info in vocab_data.items():
            token_id = token_info["id"]
            token_to_id[token_name] = token_id
            id_to_token[token_id] = token_name
    else:
        # Direct mapping: {"token": id}
        token_to_id = vocab_data
        id_to_token = {v: k for k, v in vocab_data.items()}

    logger.info(f"Loaded vocabulary with {len(token_to_id)} tokens")

    return token_to_id, id_to_token


class ModelLoader:
    """Universal model loader for AR models."""

    def __init__(
        self,
        model_type: str,
        root_dir: Union[str, Path] = "/home/vchaudha/CLIFATRON",
        device: str = "auto",
    ):
        """
        Initialize model loader.

        Args:
            model_type: Type of model (gpt2, gpt2_hf, qwen2)
            root_dir: Root directory of CLIFATRON project
            device: Device to load model on ('auto', 'cuda', 'cpu')
        """
        self.model_type = model_type.lower()
        self.root_dir = Path(root_dir)
        self.device = self._setup_device(device)

        # Validate model type
        if self.model_type not in ["gpt2", "gpt2_hf", "qwen2", "qwen2optuna"]:
            raise ValueError(
                f"Invalid model type: {model_type}. Must be one of: gpt2, gpt2_hf, qwen2, qwen2optuna"
            )

        # Setup paths
        self.model_dir = self.root_dir / "AR" / self.model_type
        # Use get_default_vocab_path to get the correct vocab path based on model_type
        self.vocab_path = get_default_vocab_path(self.model_type, self.root_dir)

        # Add model directory to path for imports
        if str(self.model_dir) not in sys.path:
            sys.path.insert(0, str(self.model_dir))

    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        logger.info(f"Using device: {device}")
        return torch.device(device)

    def load_vocabulary(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Load vocabulary."""
        return load_vocabulary(self.vocab_path)

    def load_model(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_size: str = "small",
    ) -> torch.nn.Module:
        """
        Load AR model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            model_size: Size of model (small, medium, etc.)

        Returns:
            Loaded model
        """
        logger.info(f"Loading {self.model_type} model")

        if self.model_type == "gpt2_hf":
            return self._load_gpt2_hf(checkpoint_path, model_size)
        elif self.model_type in ["qwen2", "qwen2optuna"]:
            return self._load_qwen2(checkpoint_path, model_size)
        elif self.model_type == "gpt2":
            return self._load_gpt2(checkpoint_path, model_size)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _load_gpt2_hf(
        self,
        checkpoint_path: Optional[Union[str, Path]],
        model_size: str,
    ) -> torch.nn.Module:
        """Load GPT2-HF model with GPU fallback."""
        from transformers import GPT2LMHeadModel

        if checkpoint_path is None:
            # Use default checkpoint path (new reorganized structure)
            checkpoint_path = self.root_dir / "models" / "gpt2_hf" / "model_weights"

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading from checkpoint: {checkpoint_path}")

        # Load model (initially on CPU to avoid OOM during loading)
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        model.eval()

        # Try to move to target device with fallback
        model, actual_device = try_load_model_on_device(model, self.device, fallback_cpu=True)

        # Update device if fallback occurred
        self.device = actual_device

        logger.info(f"Model loaded successfully on {actual_device}")

        return model

    def _load_qwen2(
        self,
        checkpoint_path: Optional[Union[str, Path]],
        model_size: str,
    ) -> torch.nn.Module:
        """Load Qwen2 model with GPU fallback."""
        from transformers import Qwen2ForCausalLM

        if checkpoint_path is None:
            # Use default checkpoint path based on model type
            model_folder = "qwen2" if self.model_type == "qwen2" else "qwen2optuna"
            checkpoint_path = self.root_dir / "models" / model_folder / "model_weights"

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading from checkpoint: {checkpoint_path}")

        # Load model (initially on CPU to avoid OOM during loading)
        model = Qwen2ForCausalLM.from_pretrained(checkpoint_path)
        model.eval()

        # Try to move to target device with fallback
        model, actual_device = try_load_model_on_device(model, self.device, fallback_cpu=True)

        # Update device if fallback occurred
        self.device = actual_device

        logger.info(f"Model loaded successfully on {actual_device}")

        return model

    def _load_gpt2(
        self,
        checkpoint_path: Optional[Union[str, Path]],
        model_size: str,
    ) -> torch.nn.Module:
        """Load custom GPT2 model with GPU fallback."""
        # Import custom GPT2 model
        try:
            from model import GPT
            from config import ModelConfig
        except ImportError:
            logger.error(
                f"Could not import GPT model from {self.model_dir}. "
                "Make sure the model files exist."
            )
            raise

        if checkpoint_path is None:
            # Use default checkpoint path (new reorganized structure)
            checkpoint_path = self.root_dir / "models" / "gpt2" / "model_weights" / "pytorch_model.bin"

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading from checkpoint: {checkpoint_path}")

        # Load checkpoint (initially to CPU)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Create model config with vocab_size and block_size
        # Load vocabulary to get vocab size
        token_to_id, _ = self.load_vocabulary()
        # The checkpoint was trained with block_size=8192
        config = ModelConfig(model_size=model_size, vocab_size=len(token_to_id), block_size=8192)

        # Create model
        model = GPT(config)

        # Load state dict (directly from checkpoint, not nested under 'model')
        model.load_state_dict(checkpoint)
        model.eval()

        # Try to move to target device with fallback
        model, actual_device = try_load_model_on_device(model, self.device, fallback_cpu=True)

        # Update device if fallback occurred
        self.device = actual_device

        logger.info(f"Model loaded successfully on {actual_device}")

        return model


def load_model(
    model_type: str,
    checkpoint_path: Optional[Union[str, Path]] = None,
    model_size: str = "small",
    root_dir: Union[str, Path] = "/home/vchaudha/CLIFATRON",
    device: str = "auto",
) -> Tuple[torch.nn.Module, Dict[str, int], Dict[int, str]]:
    """
    Load model and vocabulary.

    Args:
        model_type: Type of model (gpt2, gpt2_hf, qwen2)
        checkpoint_path: Path to checkpoint (optional)
        model_size: Size of model
        root_dir: Root directory
        device: Device to use

    Returns:
        Tuple of (model, token_to_id, id_to_token)
    """
    loader = ModelLoader(model_type, root_dir, device)

    # Load vocabulary
    token_to_id, id_to_token = loader.load_vocabulary()

    # Load model
    model = loader.load_model(checkpoint_path, model_size)

    return model, token_to_id, id_to_token


def get_model_embeddings(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    layer: str = "mean",
) -> torch.Tensor:
    """
    Extract embeddings from model.

    Args:
        model: The model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len] (optional)
        layer: Which layer to extract from ('last', 'mean')

    Returns:
        Embeddings tensor [batch_size, hidden_dim]
    """
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Get hidden states
        hidden_states = outputs.hidden_states

        if layer == "last":
            # Use last layer, last token (or last non-padding token if attention_mask provided)
            if attention_mask is not None:
                # Get the last non-padding token for each sequence
                sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                batch_size = input_ids.shape[0]
                embeddings = hidden_states[-1][torch.arange(batch_size), sequence_lengths]
            else:
                # Use last token
                embeddings = hidden_states[-1][:, -1, :]
        elif layer == "mean":
            # Use last layer, mean pooling (masked if attention_mask provided)
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states[-1].size()).float()
                sum_embeddings = torch.sum(hidden_states[-1] * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            else:
                # Regular mean pooling
                embeddings = hidden_states[-1].mean(dim=1)
        else:
            raise ValueError(f"Unsupported layer type: {layer}")

    return embeddings
