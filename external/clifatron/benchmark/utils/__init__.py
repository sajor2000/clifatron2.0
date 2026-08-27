"""
Utilities for Task 1: Discharged Home Prediction Benchmark

This package provides utilities for benchmarking AR models on the binary classification
task of predicting whether a patient will be discharged home.
"""

from .data_loader import (
    load_narratives,
    create_binary_labels,
    BenchmarkDataset,
    prepare_benchmark_data,
)

from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    save_results,
    MetricsCalculator,
)

from .model_loader import (
    load_model,
    load_vocabulary,
    ModelLoader,
    get_default_vocab_path,
)

from .gpu_utils import (
    detect_gpus,
    setup_distributed,
    cleanup_distributed,
    get_device_strategy,
    get_world_size,
    get_rank,
    is_main_process,
    gather_objects,
    DeviceStrategy,
)

from .results_writer import (
    ResultsWriter,
    get_results_writer,
    load_summary_results,
    load_detailed_results,
    list_available_results,
)

__all__ = [
    # Data loading
    "load_narratives",
    "create_binary_labels",
    "BenchmarkDataset",
    "prepare_benchmark_data",
    # Metrics
    "compute_metrics",
    "compute_confusion_matrix",
    "save_results",
    "MetricsCalculator",
    # Model loading
    "load_model",
    "load_vocabulary",
    "ModelLoader",
    # GPU and distributed
    "detect_gpus",
    "setup_distributed",
    "cleanup_distributed",
    "get_device_strategy",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "gather_objects",
    "DeviceStrategy",
    # Results writing
    "ResultsWriter",
    "get_results_writer",
    "load_summary_results",
    "load_detailed_results",
    "list_available_results",
]

__version__ = "0.1.0"
