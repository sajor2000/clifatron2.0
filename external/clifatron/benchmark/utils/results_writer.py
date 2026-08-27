"""
Standardized results writer for benchmark tasks.

This module provides utilities for saving benchmark results in a consistent
structure across all tasks, models, and methods.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil


class ResultsWriter:
    """
    Standardized writer for benchmark results.

    Organizes results into:
    benchmark/results/
        {task_name}/
            {model_type}/
                method{N}-{method_name}/
                    summary_metrics.json      # Core metrics only
                    detailed_data.json        # Full curves, predictions, etc.
                    plots/                    # All visualizations
                    logs/                     # Execution logs
    """

    def __init__(
        self,
        task_name: str,
        model_type: str,
        method_name: str,
        results_root: str = "benchmark/results"
    ):
        """
        Initialize results writer.

        Args:
            task_name: Task name (e.g., "task1-discharged-home")
            model_type: Model type (e.g., "qwen2", "gpt2")
            method_name: Method name (e.g., "method1-embedding", "method2-montecarlo")
            results_root: Root directory for results
        """
        self.task_name = task_name
        self.model_type = model_type
        self.method_name = method_name
        self.results_root = Path(results_root)

        # Create result directory structure
        self.result_dir = self.results_root / task_name / model_type / method_name
        self.plots_dir = self.result_dir / "plots"
        self.logs_dir = self.result_dir / "logs"

        # Create directories
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    def save_results(
        self,
        metrics: Dict[str, Any],
        metadata: Dict[str, Any],
        method: str = "embedding",
        model_size: str = "small",
        classifier: str = "xgboost",
        layer: str = "mean",
        detailed_data: Optional[Dict[str, Any]] = None
    ):
        """
        Save benchmark results.

        Args:
            metrics: Dictionary of metrics (accuracy, precision, etc.)
            metadata: Dictionary of metadata (timestamp, checkpoint, etc.)
            method: Method type ("embedding", "montecarlo", etc.)
            model_size: Model size ("small", "medium", etc.)
            classifier: Classifier type ("xgboost", "random_forest", etc.)
            layer: Layer used for embeddings ("last", "mean", etc.)
            detailed_data: Optional detailed data (ROC curves, predictions, etc.)
        """
        # Prepare summary results (compact, human-readable)
        summary = {
            "method": method,
            "model_type": self.model_type,
            "model_size": model_size,
            "classifier": classifier,
            "layer": layer,
            "metrics": self._simplify_metrics(metrics),
            "metadata": metadata
        }

        # Save summary
        summary_path = self.result_dir / "summary_metrics.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary metrics to: {summary_path}")

        # Save detailed data if provided
        if detailed_data:
            detailed = {
                "method": method,
                "model_type": self.model_type,
                "model_size": model_size,
                "detailed_curves": detailed_data,
                "metadata": metadata
            }

            detailed_path = self.result_dir / "detailed_data.json"
            with open(detailed_path, 'w') as f:
                json.dump(detailed, f, indent=2)

            print(f"Saved detailed data to: {detailed_path}")

    def _simplify_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplify metrics by removing large arrays/curves.

        Args:
            metrics: Full metrics dictionary

        Returns:
            Simplified metrics (scalars and small arrays only)
        """
        simplified = {}

        for key, value in metrics.items():
            # Include ROC/PR curves in summary for analysis
            # (previously they were skipped and supposed to go in detailed_data)

            # Keep scalars, small lists, and confusion matrices
            if isinstance(value, (int, float, str, bool)):
                simplified[key] = value
            elif isinstance(value, list):
                if len(value) < 100:  # Keep small arrays like confusion matrices
                    simplified[key] = value
                else:
                    simplified[key + '_shape'] = f"Array of length {len(value)}"
            elif isinstance(value, dict):
                simplified[key] = value  # Keep nested dicts (e.g., ROC/PR curves, per_class_metrics)
            else:
                simplified[key] = str(value)

        return simplified

    def save_plot(self, figure_or_path, plot_name: str):
        """
        Save a plot to the plots directory.

        Args:
            figure_or_path: Either a matplotlib figure or path to existing plot
            plot_name: Name for the plot file (e.g., "roc_curve.png")
        """
        dest_path = self.plots_dir / plot_name

        if isinstance(figure_or_path, (str, Path)):
            # Copy existing file
            shutil.copy(figure_or_path, dest_path)
        else:
            # Save matplotlib figure
            figure_or_path.savefig(dest_path, dpi=300, bbox_inches='tight')

        print(f"Saved plot to: {dest_path}")
        return dest_path

    def save_log(self, log_content: str, log_name: Optional[str] = None):
        """
        Save execution log.

        Args:
            log_content: Log content as string
            log_name: Optional log name. If None, uses timestamp.
        """
        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"run_{timestamp}.log"

        log_path = self.logs_dir / log_name
        with open(log_path, 'w') as f:
            f.write(log_content)

        print(f"Saved log to: {log_path}")
        return log_path

    def get_summary_path(self) -> Path:
        """Get path to summary metrics file."""
        return self.result_dir / "summary_metrics.json"

    def get_detailed_path(self) -> Path:
        """Get path to detailed data file."""
        return self.result_dir / "detailed_data.json"

    def exists(self) -> bool:
        """Check if results already exist."""
        return self.get_summary_path().exists()


def get_results_writer(
    task_name: str,
    model_type: str,
    method_name: str,
    results_root: str = "benchmark/results"
) -> ResultsWriter:
    """
    Factory function to create a ResultsWriter.

    Args:
        task_name: Task name (e.g., "task1-discharged-home")
        model_type: Model type (e.g., "qwen2", "gpt2")
        method_name: Method name (e.g., "method1-embedding")
        results_root: Root directory for results

    Returns:
        ResultsWriter instance
    """
    return ResultsWriter(task_name, model_type, method_name, results_root)


def load_summary_results(
    task_name: str,
    model_type: str,
    method_name: str,
    results_root: str = "benchmark/results"
) -> Optional[Dict[str, Any]]:
    """
    Load summary results for a given task/model/method.

    Args:
        task_name: Task name
        model_type: Model type
        method_name: Method name
        results_root: Root directory for results

    Returns:
        Summary results dictionary or None if not found
    """
    path = Path(results_root) / task_name / model_type / method_name / "summary_metrics.json"

    if not path.exists():
        return None

    with open(path, 'r') as f:
        return json.load(f)


def load_detailed_results(
    task_name: str,
    model_type: str,
    method_name: str,
    results_root: str = "benchmark/results"
) -> Optional[Dict[str, Any]]:
    """
    Load detailed results for a given task/model/method.

    Args:
        task_name: Task name
        model_type: Model type
        method_name: Method name
        results_root: Root directory for results

    Returns:
        Detailed results dictionary or None if not found
    """
    path = Path(results_root) / task_name / model_type / method_name / "detailed_data.json"

    if not path.exists():
        return None

    with open(path, 'r') as f:
        return json.load(f)


def list_available_results(
    results_root: str = "benchmark/results"
) -> List[Dict[str, str]]:
    """
    List all available results.

    Args:
        results_root: Root directory for results

    Returns:
        List of dicts with task_name, model_type, method_name
    """
    results_root = Path(results_root)
    available = []

    if not results_root.exists():
        return available

    for task_dir in results_root.iterdir():
        if not task_dir.is_dir():
            continue

        for model_dir in task_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for method_dir in model_dir.iterdir():
                if not method_dir.is_dir():
                    continue

                summary_path = method_dir / "summary_metrics.json"
                if summary_path.exists():
                    available.append({
                        "task_name": task_dir.name,
                        "model_type": model_dir.name,
                        "method_name": method_dir.name,
                        "path": str(summary_path)
                    })

    return available
