#!/usr/bin/env python3
"""
Aggregated Visualization Script for All 4 Tasks with Model Comparison
Auto-detects available models and creates comparison plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10


def detect_available_models(results_dir):
    """Auto-detect available models from results directory"""
    results_path = Path(results_dir) / "results"
    if not results_path.exists():
        return []

    models = set()
    for task_dir in results_path.iterdir():
        if task_dir.is_dir() and task_dir.name.startswith("task"):
            for model_dir in task_dir.iterdir():
                if model_dir.is_dir() and (model_dir / "method1-embedding" / "summary_metrics.json").exists():
                    models.add(model_dir.name)

    return sorted(list(models))


def load_results_new_structure(results_dir, models=None):
    """Load results from new directory structure for all available models"""
    results_path = Path(results_dir) / "results"

    # Auto-detect models if not provided
    if models is None:
        models = detect_available_models(results_dir)

    if not models:
        raise ValueError(f"No models found in {results_path}")

    print(f"Found models: {', '.join(models)}")

    all_results = {}

    for model_type in models:
        model_results = {}

        # Task 1: Discharged Home
        task1_path = results_path / "task1-discharged-home" / model_type / "method1-embedding" / "summary_metrics.json"
        if task1_path.exists():
            with open(task1_path, 'r') as f:
                model_results['task1'] = json.load(f)

        # Task 2: Discharged LTACH
        task2_path = results_path / "task2-discharged-ltach" / model_type / "method1-embedding" / "summary_metrics.json"
        if task2_path.exists():
            with open(task2_path, 'r') as f:
                model_results['task2'] = json.load(f)

        # Task 3: IMV Status at 72hr
        task3_path = results_path / "task3-outcome-72hr" / model_type / "method1-embedding" / "summary_metrics.json"
        if task3_path.exists():
            with open(task3_path, 'r') as f:
                model_results['task3'] = json.load(f)

        # Task 4: Hypoxic Proportion
        task4_path = results_path / "task4-hypoxic-proportion" / model_type / "method1-embedding" / "summary_metrics.json"
        if task4_path.exists():
            with open(task4_path, 'r') as f:
                model_results['task4'] = json.load(f)

        if model_results:
            all_results[model_type] = model_results

    return all_results


def plot_task1_comparison(ax, all_results, task_key='task1'):
    """Plot Task 1 (Binary Classification) comparison across models"""
    models = list(all_results.keys())
    metrics_to_plot = ['accuracy', 'auroc', 'f1', 'precision', 'recall', 'auprc']

    # Prepare data
    data = {metric: [] for metric in metrics_to_plot}
    for model in models:
        if task_key in all_results[model]:
            m = all_results[model][task_key]['metrics']
            for metric in metrics_to_plot:
                data[metric].append(m.get(metric, 0))
        else:
            for metric in metrics_to_plot:
                data[metric].append(0)

    # Create grouped bar chart with modern color scheme
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(models)

    # Modern orange, blue, black color palette
    color_palette = ['#FF6B35', '#004E89', '#2C2C2C', '#7D8491']  # Orange, Blue, Black, Gray
    colors = [color_palette[i % len(color_palette)] for i in range(len(models))]

    for i, model in enumerate(models):
        values = [data[metric][i] for metric in metrics_to_plot]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model.upper(),
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Task 1: Discharged Home Prediction', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot], rotation=0)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')


def plot_task2_comparison(ax, all_results, task_key='task2'):
    """Plot Task 2 (Binary Classification) comparison across models"""
    plot_task1_comparison(ax, all_results, task_key)
    ax.set_title('Task 2: Discharged to LTACH Prediction', fontsize=13, fontweight='bold', pad=10)


def plot_task3_comparison(ax, all_results):
    """Plot Task 3 (Multiclass) comparison across models"""
    models = list(all_results.keys())

    # Overall metrics comparison
    overall_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
    data = {metric: [] for metric in overall_metrics}

    for model in models:
        if 'task3' in all_results[model]:
            m = all_results[model]['task3']['metrics']
            for metric in overall_metrics:
                data[metric].append(m.get(metric, 0))
        else:
            for metric in overall_metrics:
                data[metric].append(0)

    # Create grouped bar chart with modern color scheme
    x = np.arange(len(overall_metrics))
    width = 0.8 / len(models)

    # Modern orange, blue, black color palette
    color_palette = ['#FF6B35', '#004E89', '#2C2C2C', '#7D8491']  # Orange, Blue, Black, Gray
    colors = [color_palette[i % len(color_palette)] for i in range(len(models))]

    for i, model in enumerate(models):
        values = [data[metric][i] for metric in overall_metrics]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model.upper(),
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Task 3: IMV Status at 72hr (Multiclass)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Macro F1', 'Weighted F1'], rotation=0)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')


def plot_task4_comparison(ax, all_results):
    """Plot Task 4 (Regression) comparison across models"""
    models = list(all_results.keys())

    # Regression metrics - note: lower is better for MSE, RMSE, MAE
    # higher is better for R2 and Pearson
    metrics_to_plot = ['r2_score', 'pearson_correlation', 'mae', 'rmse']
    metric_labels = ['R² Score', 'Pearson Corr', 'MAE', 'RMSE']

    data = {metric: [] for metric in metrics_to_plot}
    for model in models:
        if 'task4' in all_results[model]:
            m = all_results[model]['task4']['metrics']
            for metric in metrics_to_plot:
                data[metric].append(m.get(metric, 0))
        else:
            for metric in metrics_to_plot:
                data[metric].append(0)

    # Create grouped bar chart with modern color scheme
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(models)

    # Modern orange, blue, black color palette
    color_palette = ['#FF6B35', '#004E89', '#2C2C2C', '#7D8491']  # Orange, Blue, Black, Gray
    colors = [color_palette[i % len(color_palette)] for i in range(len(models))]

    for i, model in enumerate(models):
        values = [data[metric][i] for metric in metrics_to_plot]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model.upper(),
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax.set_title('Task 4: Hypoxic Proportion (Regression)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=0)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim([0, max([max(data[m]) for m in metrics_to_plot]) * 1.15])
    ax.grid(True, alpha=0.3, axis='y')

    # Add note about error metrics
    ax.text(0.5, -0.15, 'Note: Lower is better for MAE and RMSE; Higher is better for R² and Pearson',
            transform=ax.transAxes, fontsize=8, ha='center', style='italic')


def create_comparison_plot(all_results, output_path):
    """Create aggregated comparison visualization across all models"""
    fig = plt.figure(figsize=(20, 12))

    # Add main title
    models_str = ', '.join([m.upper() for m in all_results.keys()])
    fig.suptitle(f'Model Performance Comparison: {models_str}\nXGBoost + Embeddings (Method 1)',
                 fontsize=18, fontweight='bold', y=0.98)

    # Create 2x2 subplot grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Task 1: Binary Classification Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    plot_task1_comparison(ax1, all_results)

    # Task 2: Binary Classification Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    plot_task2_comparison(ax2, all_results)

    # Task 3: Multiclass Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    plot_task3_comparison(ax3, all_results)

    # Task 4: Regression Comparison
    ax4 = fig.add_subplot(gs[1, 1])
    plot_task4_comparison(ax4, all_results)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved aggregated comparison visualization to: {output_path}")

    return fig


def print_comparison_table(all_results):
    """Print comparison table of all models and tasks"""
    print("\n" + "="*100)
    print("MODEL PERFORMANCE COMPARISON: XGBoost + Embeddings (Method 1)")
    print("="*100)

    models = list(all_results.keys())

    # Task 1
    print("\n📊 Task 1: Discharged Home Prediction")
    print("-" * 100)
    print(f"{'Model':<15} {'Accuracy':<12} {'AUROC':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 100)
    for model in models:
        if 'task1' in all_results[model]:
            m = all_results[model]['task1']['metrics']
            print(f"{model.upper():<15} {m['accuracy']:<12.4f} {m['auroc']:<12.4f} "
                  f"{m['f1']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f}")

    # Task 2
    print("\n📊 Task 2: Discharged to LTACH Prediction")
    print("-" * 100)
    print(f"{'Model':<15} {'Accuracy':<12} {'AUROC':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 100)
    for model in models:
        if 'task2' in all_results[model]:
            m = all_results[model]['task2']['metrics']
            print(f"{model.upper():<15} {m['accuracy']:<12.4f} {m['auroc']:<12.4f} "
                  f"{m['f1']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f}")

    # Task 3
    print("\n📊 Task 3: IMV Status at 72hr (Multiclass)")
    print("-" * 100)
    print(f"{'Model':<15} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-" * 100)
    for model in models:
        if 'task3' in all_results[model]:
            m = all_results[model]['task3']['metrics']
            print(f"{model.upper():<15} {m['accuracy']:<12.4f} {m['f1_macro']:<12.4f} "
                  f"{m['f1_weighted']:<12.4f}")

    # Task 4
    print("\n📊 Task 4: Hypoxic Proportion (Regression)")
    print("-" * 100)
    print(f"{'Model':<15} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'Pearson':<12}")
    print("-" * 100)
    for model in models:
        if 'task4' in all_results[model]:
            m = all_results[model]['task4']['metrics']
            print(f"{model.upper():<15} {m['mse']:<12.4f} {m['rmse']:<12.4f} "
                  f"{m['mae']:<12.4f} {m['r2_score']:<12.4f} {m['pearson_correlation']:<12.4f}")

    print("\n" + "="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare performance of all models across all 4 benchmark tasks'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='benchmark',
        help='Directory containing benchmark results'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to compare (default: auto-detect all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the visualization'
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        args.output = Path(args.results_dir) / "aggregated_model_comparison.png"

    print(f"Loading results from {args.results_dir}...")
    all_results = load_results_new_structure(args.results_dir, args.models)

    if not all_results:
        print("Error: No results found!")
        return

    print(f"\nLoaded results for {len(all_results)} model(s)")

    print("\nCreating comparison visualization...")
    fig = create_comparison_plot(all_results, args.output)

    print_comparison_table(all_results)

    print(f"\n✅ Visualization saved to: {args.output}")


if __name__ == "__main__":
    main()
