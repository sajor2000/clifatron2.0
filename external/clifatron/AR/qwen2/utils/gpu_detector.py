"""
GPU Detection and Configuration System for Qwen2 Training

Automatically detects available GPUs and configures optimal training settings
for multi-GPU distributed training with DeepSpeed.

Supported configurations:
- 2x NVIDIA L40 (48GB) - ZeRO-2
- 8x NVIDIA A100 (40GB/80GB) - ZeRO-2
- Mixed GPU configurations (uses minimum VRAM)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import yaml

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a single GPU"""
    index: int
    name: str
    memory_total_gb: float
    compute_capability: Tuple[int, int]
    is_bf16_supported: bool


@dataclass
class GPUConfiguration:
    """Complete GPU configuration for training"""
    num_gpus: int
    gpu_infos: List[GPUInfo]
    min_memory_gb: float
    max_memory_gb: float
    gpu_model: str  # "L40", "A100", "V100", "Unknown"
    recommended_deepspeed_stage: str  # "zero2", "zero3"
    deepspeed_config_path: Optional[str]
    use_bf16: bool
    use_fp16: bool
    warnings: List[str]


class GPUDetector:
    """Detects GPU configuration and recommends optimal training settings"""

    def __init__(self, config_dir: Path):
        """
        Initialize GPU detector

        Args:
            config_dir: Path to directory containing gpu_profiles.yaml and DeepSpeed configs
        """
        self.config_dir = Path(config_dir)
        self.gpu_profiles = self._load_gpu_profiles()

    def _load_gpu_profiles(self) -> Dict:
        """Load GPU profiles from YAML configuration"""
        profile_path = self.config_dir / "gpu_profiles.yaml"
        if not profile_path.exists():
            logger.warning(f"GPU profiles not found at {profile_path}, using defaults")
            return self._get_default_profiles()

        with open(profile_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_default_profiles(self) -> Dict:
        """Default GPU profiles if config file doesn't exist"""
        return {
            "gpu_models": {
                "L40": {
                    "expected_vram_gb": 48,
                    "compute_capability": [8, 9],
                    "bf16_supported": True,
                    "zero_stage": "zero2"
                },
                "A100": {
                    "expected_vram_gb": [40, 80],
                    "compute_capability": [8, 0],
                    "bf16_supported": True,
                    "zero_stage": "zero2"
                }
            },
            "model_requirements": {
                "0.5b": {"min_vram_per_gpu": 8},
                "1.5b": {"min_vram_per_gpu": 12},
                "7b": {"min_vram_per_gpu": 24}
            }
        }

    def detect_gpus(self) -> List[GPUInfo]:
        """
        Detect all available CUDA GPUs

        Returns:
            List of GPUInfo objects, one per GPU
        """
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. No GPUs detected.")
            return []

        num_gpus = torch.cuda.device_count()
        gpu_infos = []

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)

            # Convert bytes to GB
            memory_gb = props.total_memory / (1024**3)

            # Get compute capability
            compute_cap = (props.major, props.minor)

            # Check BF16 support (requires compute capability >= 8.0)
            bf16_supported = compute_cap >= (8, 0)

            gpu_info = GPUInfo(
                index=i,
                name=props.name,
                memory_total_gb=round(memory_gb, 2),
                compute_capability=compute_cap,
                is_bf16_supported=bf16_supported
            )

            gpu_infos.append(gpu_info)
            logger.info(
                f"GPU {i}: {gpu_info.name} | "
                f"VRAM: {gpu_info.memory_total_gb}GB | "
                f"Compute: {compute_cap[0]}.{compute_cap[1]} | "
                f"BF16: {bf16_supported}"
            )

        return gpu_infos

    def _identify_gpu_model(self, gpu_name: str) -> str:
        """
        Identify GPU model from name string

        Args:
            gpu_name: GPU name from CUDA properties

        Returns:
            Model identifier: "L40", "A100", "V100", "H100", "Unknown"
        """
        name_upper = gpu_name.upper()

        if "L40" in name_upper:
            return "L40"
        elif "A100" in name_upper:
            return "A100"
        elif "V100" in name_upper:
            return "V100"
        elif "H100" in name_upper:
            return "H100"
        elif "A6000" in name_upper or "RTX" in name_upper:
            return "RTX"
        else:
            return "Unknown"

    def _select_deepspeed_stage(
        self,
        num_gpus: int,
        gpu_model: str,
        min_memory_gb: float
    ) -> str:
        """
        Select optimal DeepSpeed ZeRO stage

        Args:
            num_gpus: Number of available GPUs
            gpu_model: Identified GPU model
            min_memory_gb: Minimum VRAM across all GPUs

        Returns:
            DeepSpeed stage: "zero2" or "zero3"
        """
        # Based on user preferences:
        # - 2x L40: ZeRO-2
        # - 8x A100: ZeRO-2

        if gpu_model in ["L40", "A100"]:
            # Use ZeRO-2 for both L40 and A100 as per requirements
            return "zero2"
        elif min_memory_gb >= 40:
            # High VRAM GPUs - ZeRO-2 is usually sufficient
            return "zero2"
        elif num_gpus >= 4:
            # Multiple GPUs can distribute memory better with ZeRO-2
            return "zero2"
        else:
            # Lower memory or fewer GPUs - use ZeRO-3 for safety
            return "zero3"

    def _validate_memory_requirements(
        self,
        model_size: str,
        min_memory_gb: float,
        num_gpus: int
    ) -> List[str]:
        """
        Validate that GPUs have sufficient memory for model

        Args:
            model_size: Model size ("0.5b", "1.5b", "7b")
            min_memory_gb: Minimum VRAM per GPU
            num_gpus: Number of GPUs

        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []

        requirements = self.gpu_profiles.get("model_requirements", {})
        if model_size not in requirements:
            warnings.append(
                f"Unknown model size '{model_size}'. Cannot validate memory requirements."
            )
            return warnings

        required_vram = requirements[model_size]["min_vram_per_gpu"]

        if min_memory_gb < required_vram:
            warnings.append(
                f"WARNING: Model {model_size} requires {required_vram}GB per GPU, "
                f"but minimum available is {min_memory_gb}GB. "
                f"Training may fail with OOM errors."
            )

        # Additional check for mixed GPU configurations
        if num_gpus > 1:
            warnings.append(
                f"Using {num_gpus} GPUs. Batch will be distributed across devices. "
                f"Ensure network bandwidth (NVLink/InfiniBand) is sufficient."
            )

        return warnings

    def configure(
        self,
        model_size: Optional[str] = None,
        force_deepspeed_stage: Optional[str] = None
    ) -> GPUConfiguration:
        """
        Detect GPUs and generate optimal configuration

        Args:
            model_size: Model size for validation ("0.5b", "1.5b", "7b")
            force_deepspeed_stage: Override automatic stage selection

        Returns:
            GPUConfiguration object with all settings
        """
        # Detect all GPUs
        gpu_infos = self.detect_gpus()

        if not gpu_infos:
            raise RuntimeError(
                "No CUDA GPUs detected. This training script requires at least one GPU."
            )

        num_gpus = len(gpu_infos)

        # Get memory range (handle mixed GPU configurations)
        memory_values = [gpu.memory_total_gb for gpu in gpu_infos]
        min_memory_gb = min(memory_values)
        max_memory_gb = max(memory_values)

        # Check for mixed GPU configuration
        warnings = []
        if len(set(memory_values)) > 1:
            warnings.append(
                f"MIXED GPU CONFIGURATION DETECTED: VRAM ranges from "
                f"{min_memory_gb}GB to {max_memory_gb}GB. "
                f"Training will use settings for minimum VRAM ({min_memory_gb}GB)."
            )

        # Identify GPU model (use first GPU's name)
        gpu_model = self._identify_gpu_model(gpu_infos[0].name)

        # Check if all GPUs are the same model
        all_models = [self._identify_gpu_model(gpu.name) for gpu in gpu_infos]
        if len(set(all_models)) > 1:
            warnings.append(
                f"HETEROGENEOUS GPU TYPES DETECTED: {', '.join(set(all_models))}. "
                f"This may lead to suboptimal performance."
            )

        # Select DeepSpeed stage
        if force_deepspeed_stage:
            deepspeed_stage = force_deepspeed_stage
            logger.info(f"Using forced DeepSpeed stage: {deepspeed_stage}")
        else:
            deepspeed_stage = self._select_deepspeed_stage(
                num_gpus, gpu_model, min_memory_gb
            )

        # Get DeepSpeed config path
        deepspeed_config_path = self.config_dir / f"ds_config_{deepspeed_stage}.json"
        if not deepspeed_config_path.exists():
            warnings.append(
                f"DeepSpeed config not found: {deepspeed_config_path}. "
                f"Training may fail."
            )
            deepspeed_config_path = None
        else:
            deepspeed_config_path = str(deepspeed_config_path)

        # Determine precision support
        use_bf16 = all(gpu.is_bf16_supported for gpu in gpu_infos)
        use_fp16 = not use_bf16  # Fall back to FP16 if BF16 not supported

        if not use_bf16:
            warnings.append(
                "BF16 not supported on all GPUs. Falling back to FP16. "
                "This may affect training stability."
            )

        # Validate memory requirements
        if model_size:
            memory_warnings = self._validate_memory_requirements(
                model_size, min_memory_gb, num_gpus
            )
            warnings.extend(memory_warnings)

        # Create configuration
        config = GPUConfiguration(
            num_gpus=num_gpus,
            gpu_infos=gpu_infos,
            min_memory_gb=min_memory_gb,
            max_memory_gb=max_memory_gb,
            gpu_model=gpu_model,
            recommended_deepspeed_stage=deepspeed_stage,
            deepspeed_config_path=deepspeed_config_path,
            use_bf16=use_bf16,
            use_fp16=use_fp16,
            warnings=warnings
        )

        return config

    def print_configuration_summary(self, config: GPUConfiguration):
        """Print a formatted summary of the GPU configuration"""
        print("\n" + "="*70)
        print("GPU CONFIGURATION DETECTED")
        print("="*70)

        print(f"\nNumber of GPUs: {config.num_gpus}")
        print(f"GPU Model: {config.gpu_model}")
        print(f"Memory Range: {config.min_memory_gb}GB - {config.max_memory_gb}GB")
        print(f"Mixed Precision: BF16={config.use_bf16}, FP16={config.use_fp16}")
        print(f"Recommended DeepSpeed: {config.recommended_deepspeed_stage.upper()}")

        if config.deepspeed_config_path:
            print(f"DeepSpeed Config: {Path(config.deepspeed_config_path).name}")

        print("\nPer-GPU Details:")
        for gpu in config.gpu_infos:
            print(
                f"  GPU {gpu.index}: {gpu.name} | "
                f"{gpu.memory_total_gb}GB | "
                f"Compute {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
            )

        if config.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in config.warnings:
                print(f"  - {warning}")

        print("="*70 + "\n")


def auto_configure(
    config_dir: Path,
    model_size: Optional[str] = None,
    force_deepspeed_stage: Optional[str] = None,
    verbose: bool = True
) -> GPUConfiguration:
    """
    Convenience function for automatic GPU configuration

    Args:
        config_dir: Path to config directory
        model_size: Model size for validation
        force_deepspeed_stage: Override automatic stage selection
        verbose: Print configuration summary

    Returns:
        GPUConfiguration object
    """
    detector = GPUDetector(config_dir)
    config = detector.configure(model_size, force_deepspeed_stage)

    if verbose:
        detector.print_configuration_summary(config)

    return config


if __name__ == "__main__":
    # Test the detector
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get config directory
    script_dir = Path(__file__).parent.parent
    config_dir = script_dir / "config"

    # Run auto-configuration
    try:
        config = auto_configure(config_dir, model_size="7b", verbose=True)
        print("\n✓ GPU detection successful!")
        print(f"  Use: --deepspeed {config.deepspeed_config_path}")
        print(f"  Precision: {'BF16' if config.use_bf16 else 'FP16'}")
    except Exception as e:
        print(f"\n✗ GPU detection failed: {e}")
        sys.exit(1)
