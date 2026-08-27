#!/usr/bin/env python3

"""
Utility functions for CLIF Llama training
"""

import logging
import platform
import sys
from typing import Tuple

import torch


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def detect_device() -> Tuple[torch.device, bool, bool]:
    """
    Detect available device (GPU or CPU) and return device info

    Returns:
        device: torch.device object
        use_fp16: whether to use fp16 precision
        use_bf16: whether to use bf16 precision
    """
    logger = logging.getLogger(__name__)

    # Check for CUDA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()

        logger.info(f"🚀 CUDA GPUs detected: {gpu_count}")

        # Log info for each GPU
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        logger.info(f"CUDA version: {torch.version.cuda}")

        # Check for bfloat16 support (Ampere and newer)
        use_bf16 = torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16  # Use fp16 if bf16 not supported

        if use_bf16:
            logger.info("✓ Using bfloat16 mixed precision (optimal for Ampere+ GPUs)")
        else:
            logger.info("✓ Using float16 mixed precision")

        # Multi-GPU info
        if gpu_count > 1:
            logger.info(f"✓ Multi-GPU training enabled: {gpu_count} GPUs with DDP")
            logger.info(f"  Training will distribute batches across all GPUs")

        return device, use_fp16, use_bf16

    # Check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple Silicon GPU (MPS) detected")
        logger.info("Mixed precision not supported on MPS, using float32")
        return device, False, False

    # Fallback to CPU
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected, using CPU")
        logger.info(f"CPU: {platform.processor()}")
        logger.info("Note: Training on CPU will be significantly slower")
        return device, False, False


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model

    Returns:
        total_params: total number of parameters
        trainable_params: number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_size(num: int) -> str:
    """Format number of parameters in human-readable format"""
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.2f}{unit}"
        num /= 1000.0
    return f"{num:.2f}P"


def print_model_info(model: torch.nn.Module):
    """Print model information"""
    logger = logging.getLogger(__name__)
    total_params, trainable_params = count_parameters(model)

    logger.info(f"Total parameters: {format_size(total_params)} ({total_params:,})")
    logger.info(f"Trainable parameters: {format_size(trainable_params)} ({trainable_params:,})")


def get_device_info() -> dict:
    """Get comprehensive device information"""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        })

    if torch.backends.mps.is_available():
        info["mps_available"] = True

    return info


def rt_padding_to_left(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Convert right-padded sequences to left-padded sequences
    Useful for causal language modeling with decoder-only models
    """
    # Count padding tokens at the end
    is_pad = input_ids == pad_token_id

    # For each sequence, find the number of padding tokens
    if len(input_ids.shape) == 1:
        # Single sequence
        pad_count = is_pad.sum()
        if pad_count > 0:
            return torch.cat([
                torch.full((pad_count,), pad_token_id, dtype=input_ids.dtype),
                input_ids[:-pad_count]
            ])
        return input_ids
    else:
        # Batch of sequences
        result = []
        for seq in input_ids:
            pad_count = (seq == pad_token_id).sum()
            if pad_count > 0:
                # Find where padding starts
                non_pad_mask = seq != pad_token_id
                if non_pad_mask.any():
                    non_pad_tokens = seq[non_pad_mask]
                    new_seq = torch.cat([
                        torch.full((pad_count,), pad_token_id, dtype=seq.dtype),
                        non_pad_tokens
                    ])
                else:
                    new_seq = seq
            else:
                new_seq = seq
            result.append(new_seq)
        return torch.stack(result)
