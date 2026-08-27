"""
Utilities for Qwen2 SFT training.

Self-contained utilities copied from AR/qwen2/utils for isolation.
"""

# Import from local files (self-contained)
from .gpu_detector import GPUDetector, auto_configure
from .metrics import compute_metrics_for_trainer

__all__ = [
    'GPUDetector',
    'auto_configure',
    'compute_metrics_for_trainer',
]
