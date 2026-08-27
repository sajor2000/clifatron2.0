#!/usr/bin/env python3
"""
Enhanced visualization for binary classification tasks (Task 1 & 2)
with 95% confidence intervals using dot plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse


# Set modern style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.8


def compute_bootstrap_ci(metric_value, n, n_bootstrap=10000, confidence=0.95):
    """
    Compute bootstrap confidence interval for a metric

    Args:
        metric_value: The point estimate (e.g., accuracy, precision)
        n: Sample size
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound)
    """
    # Simulate the underlying binary outcomes based on the metric value
    # For a proportion-based metric, we can bootstrap from a binomial distribution
    np.random.seed(42)  # For reproducibility

    # Approximate number of successes
    n_successes = int(metric_value * n)

    # Generate bootstrap samples
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.binomial(1, metric_value, n)
        bootstrap_samples.append(np.mean(sample))

    bootstrap_samples = np.array(bootstrap_samples)

    # Compute percentile-based confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_samples, lower_percentile)
    upper_bound = np.percentile(bootstrap_samples, upper_percentile)

    return (max(0, lower_bound), min(1, upper_bound))


def load_task_results(results_dir, task_name):
    """Load results for a specific task across all models"""
    results_path = Path(results_dir) / "results" / task_name

    if not results_path.exists():
        return {}

    task_results = {}
    for model_dir in results_path.iterdir():
        if model_dir.is_dir():
            summary_path = model_dir / "method1-embedding" / "summary_metrics.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    task_results[model_dir.name] = json.load(f)

    return task_results


def create_dotplot_comparison(task1_results, task2_results, output_path):
    """
    Create enhanced dot plot with bootstrap 95% CI for binary classification tasks
    """
    # Define metrics to compare
    metrics_info = [
        ('accuracy', 'Accuracy'),
        ('auroc', 'AUROC'),
        ('f1', 'F1 Score'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('auprc', 'AUPRC'),
    ]

    # Get all models
    all_models = sorted(set(list(task1_results.keys()) + list(task2_results.keys())))

    # Define modern color palette (colorblind-friendly)
    # Using a professional palette inspired by scientific visualization best practices
    model_colors = {
        'gpt2_hf': '#0077BB',  # Vibrant Blue
        'qwen2': '#EE7733',     # Vivid Orange
        'llama': '#009988',     # Teal (for future models)
        'bert': '#CC3311',      # Red (for future models)
    }

    task_markers = {
        'task1': 'o',  # Circle
        'task2': 'D',  # Diamond (more distinctive than square)
    }

    # Create figure - simpler layout with just side-by-side tasks
    fig = plt.figure(figsize=(18, 8))

    # Main title
    fig.suptitle('Binary Classification Tasks: Model Comparison with Bootstrap 95% CI',
                 fontsize=18, fontweight='bold', y=0.96)

    # Side-by-side comparison by task
    for task_col, (task_name, task_results, task_label) in enumerate([
        ('task1', task1_results, 'Task 1: Discharged Home'),
        ('task2', task2_results, 'Task 2: Discharged to LTACH')
    ]):
        ax = plt.subplot(1, 2, task_col + 1)

        x_positions = np.arange(len(metrics_info))
        width = 0.25

        for model_idx, model in enumerate(all_models):
            if model not in task_results:
                continue

            metrics = task_results[model]['metrics']
            n_samples = task_results[model]['metadata']['num_test_examples']

            offset = (model_idx - 0.5) * width
            x_pos = x_positions + offset

            y_values = []
            y_errors_lower = []
            y_errors_upper = []

            print(f"Computing bootstrap CI for {model.upper()} - {task_name}...")
            for metric_key, metric_label in metrics_info:
                value = metrics.get(metric_key, 0)
                lower, upper = compute_bootstrap_ci(value, n_samples)

                y_values.append(value)
                y_errors_lower.append(value - lower)
                y_errors_upper.append(upper - value)

            # Plot with thicker error bars
            ax.errorbar(x_pos, y_values,
                       yerr=[y_errors_lower, y_errors_upper],
                       fmt='o',
                       color=model_colors.get(model, '#95a5a6'),
                       label=model.upper(),
                       markersize=12,
                       capsize=6,
                       capthick=2.5,
                       elinewidth=2.5,
                       alpha=0.9,
                       markeredgewidth=1.5,
                       markeredgecolor='white')

            # Add value labels on top of error bars
            for x, y, err_u in zip(x_pos, y_values, y_errors_upper):
                ax.text(x, y + err_u + 0.02, f'{y:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       color=model_colors.get(model, '#95a5a6'))

        ax.set_xticks(x_positions)
        ax.set_xticklabels([info[1] for info in metrics_info], fontsize=12)
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title(task_label, fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='gray')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.25, axis='y', linestyle='-', linewidth=0.8)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)

        # Add subtle background for better readability
        ax.set_facecolor('#FAFAFA')

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved enhanced dot plot to: {output_path}")

    return fig


def print_comparison_table_with_ci(task1_results, task2_results):
    """Print comparison table with bootstrap confidence intervals"""
    print("\n" + "="*120)
    print("BINARY CLASSIFICATION TASKS: Model Comparison with Bootstrap 95% Confidence Intervals")
    print("="*120)

    metrics_info = [
        ('accuracy', 'Accuracy'),
        ('auroc', 'AUROC'),
        ('f1', 'F1'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
    ]

    for task_name, task_results, task_label in [
        ('task1', task1_results, 'Task 1: Discharged Home Prediction'),
        ('task2', task2_results, 'Task 2: Discharged to LTACH Prediction')
    ]:
        print(f"\n📊 {task_label}")
        print("-" * 120)

        for model in sorted(task_results.keys()):
            metrics = task_results[model]['metrics']
            n_samples = task_results[model]['metadata']['num_test_examples']

            print(f"\n{model.upper()} (n={n_samples}):")
            print("-" * 120)

            for metric_key, metric_label in metrics_info:
                value = metrics.get(metric_key, 0)
                lower, upper = compute_bootstrap_ci(value, n_samples)

                print(f"  {metric_label:12s}: {value:.4f}  [95% CI: {lower:.4f} - {upper:.4f}]  (±{(upper-lower)/2:.4f})")

    print("\n" + "="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Create enhanced dot plot comparison for binary classification tasks'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='benchmark',
        help='Directory containing benchmark results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the visualization'
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = Path(args.results_dir) / "binary_tasks_dotplot_comparison.png"

    print(f"Loading results from {args.results_dir}...")

    # Load results
    task1_results = load_task_results(args.results_dir, "task1-discharged-home")
    task2_results = load_task_results(args.results_dir, "task2-discharged-ltach")

    if not task1_results and not task2_results:
        print("Error: No results found for tasks 1 or 2!")
        return

    print(f"Found results for Task 1: {list(task1_results.keys())}")
    print(f"Found results for Task 2: {list(task2_results.keys())}")

    # Create visualization
    print("\nCreating enhanced dot plot comparison...")
    fig = create_dotplot_comparison(task1_results, task2_results, args.output)

    # Print comparison table
    print_comparison_table_with_ci(task1_results, task2_results)

    print(f"\n✅ Visualization saved to: {args.output}")


if __name__ == "__main__":
    main()
