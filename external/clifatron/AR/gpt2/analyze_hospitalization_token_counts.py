#!/usr/bin/env python3

"""
Hospitalization Token Count Analysis for CLIF Data

Analyzes total token counts per hospitalization and determines how many
hospitalizations fit within different context window sizes.

Usage:
    uv run gpt2/analyze_hospitalization_token_counts.py --input clif_output_rush/sentences/clif_sentences.parquet
"""

import argparse
import pathlib
from datetime import datetime
from typing import Dict, List

import polars as pl
import numpy as np

from utils import setup_logging

logger = setup_logging()


# Model configurations for GPT-2 variants
MODEL_CONFIGS = {
    "124M": {"params": 124_000_000, "hidden": 768, "layers": 12, "heads": 12},
    "355M": {"params": 355_000_000, "hidden": 1024, "layers": 24, "heads": 16},
    "774M": {"params": 774_000_000, "hidden": 1280, "layers": 36, "heads": 20},
    "1.5B": {"params": 1_500_000_000, "hidden": 1600, "layers": 48, "heads": 25},
}


def estimate_vram_requirements(
    model_size: str,
    context_length: int,
    batch_size: int,
    precision: str = "bf16",
    use_flash_attn: bool = True,
    gradient_checkpointing: bool = False
) -> Dict[str, float]:
    """
    Estimate VRAM requirements for training

    Args:
        model_size: Model size (124M, 355M, 774M, 1.5B)
        context_length: Context window size
        batch_size: Batch size
        precision: Training precision (fp32, fp16, bf16)
        use_flash_attn: Whether to use flash attention
        gradient_checkpointing: Whether to use gradient checkpointing

    Returns:
        Dictionary with memory breakdown in GB
    """
    config = MODEL_CONFIGS[model_size]
    params = config["params"]
    hidden = config["hidden"]
    layers = config["layers"]

    # Precision bytes
    precision_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}[precision]

    # 1. Model weights
    model_memory = params * precision_bytes / 1e9  # GB

    # 2. Gradients (same as model)
    gradient_memory = model_memory

    # 3. Optimizer states (Adam: momentum + variance, stored in fp32)
    optimizer_memory = params * 8 / 1e9  # GB (2 states × 4 bytes each)

    # 4. Activations
    # Rough estimate: batch_size × seq_len × hidden × layers × factor
    # Factor accounts for multiple activations per layer (attention QKV, MLP, etc.)
    if use_flash_attn:
        activation_factor = 16  # Flash attention reduces memory significantly
    else:
        activation_factor = 34  # Standard attention stores full matrices

    if gradient_checkpointing:
        activation_factor *= 0.5  # Gradient checkpointing trades compute for memory

    activation_memory = (batch_size * context_length * hidden * layers *
                        activation_factor * precision_bytes) / 1e9

    # 5. Framework overhead (PyTorch, CUDA kernels, etc.)
    overhead = 2.0  # GB

    total = model_memory + gradient_memory + optimizer_memory + activation_memory + overhead

    return {
        "model": model_memory,
        "gradients": gradient_memory,
        "optimizer": optimizer_memory,
        "activations": activation_memory,
        "overhead": overhead,
        "total": total
    }


def estimate_training_time(
    total_tokens: int,
    context_length: int,
    model_size: str,
    hardware: str = "2xL40",
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Estimate training time for one epoch

    Args:
        total_tokens: Total tokens in dataset
        context_length: Context window size
        model_size: Model size (124M, 355M, 774M, 1.5B)
        hardware: Hardware type (A100, 2xL40, H100, 4xA100)
        batch_size: Batch size (affects throughput slightly)

    Returns:
        Dictionary with time estimates and throughput
    """
    # Throughput baselines (tokens/sec) for different setups
    # These are empirical estimates based on typical training runs with flash attention
    # Format: {hardware: {model_size: base_throughput}}
    # Base throughput is for context_length=2048

    throughput_baselines = {
        "A100": {
            "124M": 35000,
            "355M": 20000,
            "774M": 12000,
            "1.5B": 7000,
        },
        "2xL40": {
            "124M": 28000,
            "355M": 16000,
            "774M": 9000,
            "1.5B": 5000,
        },
        "H100": {
            "124M": 65000,
            "355M": 38000,
            "774M": 22000,
            "1.5B": 13000,
        },
        "4xA100": {
            "124M": 120000,
            "355M": 70000,
            "774M": 42000,
            "1.5B": 24000,
        }
    }

    # Get baseline throughput
    hw_throughput = throughput_baselines.get(hardware, throughput_baselines["2xL40"])
    base_throughput = hw_throughput.get(model_size, hw_throughput["124M"])

    # Adjust for context length (throughput scales roughly as sqrt(context_length))
    # Longer contexts are less efficient due to attention computation
    context_scale_factor = (2048 / context_length) ** 0.5
    tokens_per_sec = base_throughput * context_scale_factor

    # Adjust for batch size (slight efficiency gain with larger batches)
    if batch_size < 8:
        tokens_per_sec *= (batch_size / 8) * 0.9  # Less efficient with smaller batches
    elif batch_size > 8:
        tokens_per_sec *= 1 + (batch_size - 8) * 0.02  # Slight gain with larger batches (diminishing returns)

    # Calculate time
    total_seconds = total_tokens / tokens_per_sec
    hours = total_seconds / 3600
    days = hours / 24

    return {
        "tokens_per_sec": tokens_per_sec,
        "total_seconds": total_seconds,
        "hours": hours,
        "days": days
    }


def load_clif_data(input_path: pathlib.Path) -> pl.DataFrame:
    """
    Load CLIF sentences data

    Args:
        input_path: Path to clif_sentences.parquet

    Returns:
        DataFrame with hospitalization_id and clif_sentence columns
    """
    logger.info(f"Loading CLIF data from: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pl.read_parquet(input_path)

    logger.info(f"Loaded {len(df):,} event lines")
    logger.info(f"Columns: {df.columns}")

    # Check required columns
    required_cols = ['hospitalization_id', 'clif_sentence']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def calculate_tokens_per_hospitalization(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate total tokens per hospitalization

    Args:
        df: DataFrame with hospitalization_id and clif_sentence columns

    Returns:
        DataFrame with hospitalization_id and total_tokens columns
    """
    logger.info("Calculating tokens per sentence...")

    # Count tokens (space-separated) per sentence
    df = df.with_columns([
        pl.col("clif_sentence")
        .str.split(" ")
        .list.len()
        .alias("token_count")
    ])

    logger.info("Aggregating tokens per hospitalization...")

    # Group by hospitalization and sum tokens
    hosp_tokens = df.group_by("hospitalization_id").agg([
        pl.col("token_count").sum().alias("total_tokens"),
        pl.col("token_count").count().alias("num_sentences"),
    ])

    logger.info(f"Analyzed {len(hosp_tokens):,} unique hospitalizations")

    return hosp_tokens


def compute_statistics(hosp_tokens: pl.DataFrame) -> Dict[str, float]:
    """
    Compute summary statistics for token counts

    Args:
        hosp_tokens: DataFrame with total_tokens column

    Returns:
        Dictionary of statistics
    """
    logger.info("Computing summary statistics...")

    tokens = hosp_tokens["total_tokens"]

    stats = {
        "count": len(hosp_tokens),
        "total_tokens": tokens.sum(),
        "mean": tokens.mean(),
        "median": tokens.median(),
        "min": tokens.min(),
        "max": tokens.max(),
        "std": tokens.std(),
        "p25": tokens.quantile(0.25),
        "p75": tokens.quantile(0.75),
        "p90": tokens.quantile(0.90),
        "p95": tokens.quantile(0.95),
        "p99": tokens.quantile(0.99),
    }

    logger.info(f"Total hospitalizations: {stats['count']:,}")
    logger.info(f"Mean tokens per hospitalization: {stats['mean']:.1f}")
    logger.info(f"Median tokens per hospitalization: {stats['median']:.1f}")

    return stats


def analyze_context_windows(
    hosp_tokens: pl.DataFrame,
    context_sizes: List[int]
) -> Dict[int, Dict[str, float]]:
    """
    Analyze how many hospitalizations fit within different context windows

    Args:
        hosp_tokens: DataFrame with total_tokens column
        context_sizes: List of context window sizes to analyze

    Returns:
        Dictionary mapping context_size -> metrics
    """
    logger.info("Analyzing context window coverage...")

    results = {}
    total_hosps = len(hosp_tokens)

    for context_size in context_sizes:
        # Count hospitalizations that fit completely
        fits_completely = (hosp_tokens["total_tokens"] <= context_size).sum()
        pct_fits = (fits_completely / total_hosps) * 100

        # Calculate what percentage of tokens are covered on average
        hosp_with_coverage = hosp_tokens.with_columns([
            (pl.col("total_tokens").clip(0, context_size) / pl.col("total_tokens") * 100)
            .alias("pct_covered")
        ])

        avg_coverage = hosp_with_coverage["pct_covered"].mean()

        results[context_size] = {
            "num_fits": fits_completely,
            "pct_fits": pct_fits,
            "avg_coverage": avg_coverage,
        }

        logger.info(f"Context {context_size}: {fits_completely:,} hospitalizations ({pct_fits:.1f}%) fit completely")

    return results


def analyze_gpu_requirements(
    context_sizes: List[int],
    model_sizes: List[str],
    batch_size: int,
    precision: str,
    use_flash_attn: bool,
    gradient_checkpointing: bool
) -> Dict:
    """
    Analyze GPU memory requirements for different configurations

    Args:
        context_sizes: List of context window sizes
        model_sizes: List of model sizes to analyze
        batch_size: Batch size
        precision: Training precision
        use_flash_attn: Whether to use flash attention
        gradient_checkpointing: Whether to use gradient checkpointing

    Returns:
        Dictionary mapping (context_size, model_size) -> memory requirements
    """
    logger.info("Analyzing GPU memory requirements...")

    results = {}

    for context_size in context_sizes:
        for model_size in model_sizes:
            mem_req = estimate_vram_requirements(
                model_size=model_size,
                context_length=context_size,
                batch_size=batch_size,
                precision=precision,
                use_flash_attn=use_flash_attn,
                gradient_checkpointing=gradient_checkpointing
            )

            results[(context_size, model_size)] = mem_req

            logger.info(f"  {model_size} @ {context_size} tokens: {mem_req['total']:.1f} GB VRAM")

    return results


def analyze_training_time(
    context_sizes: List[int],
    model_sizes: List[str],
    total_tokens: int,
    hardware: str,
    batch_size: int
) -> Dict:
    """
    Analyze training time for different configurations

    Args:
        context_sizes: List of context window sizes
        model_sizes: List of model sizes
        total_tokens: Total tokens in dataset
        hardware: Hardware type
        batch_size: Batch size

    Returns:
        Dictionary mapping (context_size, model_size) -> time estimates
    """
    logger.info("Analyzing training time estimates...")

    results = {}

    for context_size in context_sizes:
        for model_size in model_sizes:
            time_est = estimate_training_time(
                total_tokens=total_tokens,
                context_length=context_size,
                model_size=model_size,
                hardware=hardware,
                batch_size=batch_size
            )

            results[(context_size, model_size)] = time_est

            if time_est['days'] >= 1:
                logger.info(f"  {model_size} @ {context_size} tokens: {time_est['days']:.2f} days/epoch ({time_est['tokens_per_sec']:,.0f} tok/s)")
            else:
                logger.info(f"  {model_size} @ {context_size} tokens: {time_est['hours']:.2f} hours/epoch ({time_est['tokens_per_sec']:,.0f} tok/s)")

    return results


def generate_report(
    hosp_tokens: pl.DataFrame,
    stats: Dict[str, float],
    context_results: Dict[int, Dict[str, float]],
    gpu_results: Dict,
    time_results: Dict,
    model_sizes: List[str],
    hardware: str,
    batch_size: int,
    precision: str,
    use_flash_attn: bool,
    gradient_checkpointing: bool,
    output_dir: pathlib.Path
):
    """
    Generate comprehensive report

    Args:
        hosp_tokens: Per-hospitalization token counts
        stats: Summary statistics
        context_results: Context window analysis results
        gpu_results: GPU memory requirements
        time_results: Training time estimates
        model_sizes: Model sizes analyzed
        hardware: Hardware type
        batch_size: Batch size
        precision: Training precision
        use_flash_attn: Whether flash attention is used
        gradient_checkpointing: Whether gradient checkpointing is used
        output_dir: Output directory
    """
    logger.info("Generating report...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save per-hospitalization token counts
    hosp_path = output_dir / "hospitalization_token_counts.parquet"
    hosp_tokens.sort("total_tokens", descending=True).write_parquet(hosp_path)
    logger.info(f"Saved hospitalization token counts to: {hosp_path}")

    # Also save as CSV for easy viewing
    csv_path = output_dir / "hospitalization_token_counts.csv"
    hosp_tokens.sort("total_tokens", descending=True).write_csv(csv_path)
    logger.info(f"Saved hospitalization token counts CSV to: {csv_path}")

    # 2. Save context window analysis as CSV
    context_rows = []
    for context_size, metrics in sorted(context_results.items()):
        context_rows.append({
            "context_size": context_size,
            "num_hospitalizations_fit": metrics["num_fits"],
            "pct_hospitalizations_fit": round(metrics["pct_fits"], 2),
            "avg_pct_coverage": round(metrics["avg_coverage"], 2),
        })

    context_df = pl.DataFrame(context_rows)
    context_csv = output_dir / "context_window_analysis.csv"
    context_df.write_csv(context_csv)
    logger.info(f"Saved context window analysis to: {context_csv}")

    # 3. Save GPU memory requirements as CSV
    gpu_rows = []
    for (context_size, model_size), mem_req in sorted(gpu_results.items()):
        gpu_rows.append({
            "context_size": context_size,
            "model_size": model_size,
            "total_vram_gb": round(mem_req["total"], 2),
            "model_gb": round(mem_req["model"], 2),
            "gradients_gb": round(mem_req["gradients"], 2),
            "optimizer_gb": round(mem_req["optimizer"], 2),
            "activations_gb": round(mem_req["activations"], 2),
            "overhead_gb": round(mem_req["overhead"], 2),
        })

    gpu_df = pl.DataFrame(gpu_rows)
    gpu_csv = output_dir / "gpu_memory_requirements.csv"
    gpu_df.write_csv(gpu_csv)
    logger.info(f"Saved GPU memory requirements to: {gpu_csv}")

    # 4. Save training time estimates as CSV
    time_rows = []
    for (context_size, model_size), time_est in sorted(time_results.items()):
        time_rows.append({
            "context_size": context_size,
            "model_size": model_size,
            "tokens_per_sec": round(time_est["tokens_per_sec"], 0),
            "hours_per_epoch": round(time_est["hours"], 2),
            "days_per_epoch": round(time_est["days"], 3),
        })

    time_df = pl.DataFrame(time_rows)
    time_csv = output_dir / "training_time_estimates.csv"
    time_df.write_csv(time_csv)
    logger.info(f"Saved training time estimates to: {time_csv}")

    # 5. Generate detailed text report
    report_path = output_dir / "token_count_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CLIF Hospitalization Token Count Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary statistics
        f.write("Summary Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total hospitalizations: {stats['count']:,}\n")
        f.write(f"Total tokens: {stats['total_tokens']:,}\n\n")

        f.write(f"Tokens per hospitalization:\n")
        f.write(f"  Mean:   {stats['mean']:>10,.1f}\n")
        f.write(f"  Median: {stats['median']:>10,.1f}\n")
        f.write(f"  Min:    {stats['min']:>10,}\n")
        f.write(f"  Max:    {stats['max']:>10,}\n")
        f.write(f"  Std:    {stats['std']:>10,.1f}\n\n")

        f.write(f"Percentiles:\n")
        f.write(f"  P25:    {stats['p25']:>10,.1f}\n")
        f.write(f"  P75:    {stats['p75']:>10,.1f}\n")
        f.write(f"  P90:    {stats['p90']:>10,.1f}\n")
        f.write(f"  P95:    {stats['p95']:>10,.1f}\n")
        f.write(f"  P99:    {stats['p99']:>10,.1f}\n\n")

        # Context window analysis
        f.write("=" * 80 + "\n")
        f.write("Context Window Coverage Analysis\n")
        f.write("=" * 80 + "\n\n")

        for context_size, metrics in sorted(context_results.items()):
            f.write(f"Context Size: {context_size:,} tokens\n")
            f.write("-" * 80 + "\n")
            f.write(f"Hospitalizations that fit completely: {metrics['num_fits']:,} ({metrics['pct_fits']:.2f}%)\n")
            f.write(f"Average coverage across all hospitalizations: {metrics['avg_coverage']:.2f}%\n\n")

            # Recommendations
            f.write("Recommendations:\n")
            if metrics['pct_fits'] >= 90:
                f.write(f"  ✓ Excellent - covers nearly all hospitalizations completely\n")
            elif metrics['pct_fits'] >= 75:
                f.write(f"  ✓ Good - covers most hospitalizations completely\n")
            elif metrics['pct_fits'] >= 50:
                f.write(f"  ⚠ Moderate - may need truncation strategies for larger cases\n")
            else:
                f.write(f"  ⚠ Limited - will require truncation for many hospitalizations\n")

            if context_size <= 2048:
                f.write(f"  ✓ Efficient training on 2x L40 GPUs\n")
            elif context_size <= 4096:
                f.write(f"  ⚠ Requires reduced batch size on 2x L40\n")
            else:
                f.write(f"  ⚠ May cause OOM on 2x L40 (48GB VRAM)\n")

            f.write("\n")

        # Final recommendations
        f.write("=" * 80 + "\n")
        f.write("Summary Recommendations\n")
        f.write("=" * 80 + "\n\n")

        # Find best context size for different goals
        best_for_coverage = max(context_results.items(), key=lambda x: x[1]['pct_fits'])
        best_for_efficiency = min([cs for cs in context_results.keys() if context_results[cs]['pct_fits'] >= 75], default=2048)

        f.write(f"For maximum coverage:\n")
        f.write(f"  → {best_for_coverage[0]:,} tokens ({best_for_coverage[1]['pct_fits']:.1f}% fully covered)\n\n")

        if best_for_efficiency in context_results:
            f.write(f"For training efficiency on 2x L40:\n")
            metrics = context_results[best_for_efficiency]
            f.write(f"  → {best_for_efficiency:,} tokens\n")
            f.write(f"  → {metrics['pct_fits']:.1f}% fully covered\n")
            f.write(f"  → Good balance of coverage and GPU memory\n\n")

        # Truncation strategy recommendations
        f.write(f"Truncation strategies:\n")
        f.write(f"  • First N tokens: Simple, preserves admission data\n")
        f.write(f"  • Last N tokens: Preserves recent/discharge data\n")
        f.write(f"  • Sliding window: Create multiple samples per hospitalization\n")
        f.write(f"  • Smart sampling: Sample critical time periods\n\n")

        # GPU Memory Requirements
        f.write("=" * 80 + "\n")
        f.write("GPU Memory Requirements\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training configuration:\n")
        f.write(f"  Hardware: {hardware}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Precision: {precision}\n")
        f.write(f"  Flash attention: {use_flash_attn}\n")
        f.write(f"  Gradient checkpointing: {gradient_checkpointing}\n\n")

        for context_size in sorted(set(cs for cs, _ in gpu_results.keys())):
            f.write(f"Context Size: {context_size:,} tokens\n")
            f.write("-" * 80 + "\n")

            for model_size in model_sizes:
                mem_req = gpu_results.get((context_size, model_size))
                if mem_req:
                    f.write(f"\n{model_size}:\n")
                    f.write(f"  Total VRAM:      {mem_req['total']:>6.1f} GB\n")
                    f.write(f"    Model weights: {mem_req['model']:>6.1f} GB\n")
                    f.write(f"    Gradients:     {mem_req['gradients']:>6.1f} GB\n")
                    f.write(f"    Optimizer:     {mem_req['optimizer']:>6.1f} GB\n")
                    f.write(f"    Activations:   {mem_req['activations']:>6.1f} GB\n")
                    f.write(f"    Overhead:      {mem_req['overhead']:>6.1f} GB\n")

                    # Memory recommendations
                    total_vram = mem_req['total']
                    if total_vram <= 24:
                        f.write(f"  ✓ Fits on single RTX 4090 / L40 (24GB)\n")
                    elif total_vram <= 40:
                        f.write(f"  ✓ Fits on single A100 (40GB)\n")
                    elif total_vram <= 48:
                        f.write(f"  ⚠ Requires A100 80GB or multi-GPU\n")
                    elif total_vram <= 80:
                        f.write(f"  ⚠ Requires A100 80GB\n")
                    else:
                        f.write(f"  ⚠ Requires multi-GPU setup\n")

            f.write("\n")

        # Training Time Estimates
        f.write("=" * 80 + "\n")
        f.write(f"Training Time Estimates (1 Epoch on {hardware})\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total tokens in dataset: {stats['total_tokens']:,}\n\n")

        for context_size in sorted(set(cs for cs, _ in time_results.keys())):
            f.write(f"Context Size: {context_size:,} tokens\n")
            f.write("-" * 80 + "\n")

            for model_size in model_sizes:
                time_est = time_results.get((context_size, model_size))
                if time_est:
                    f.write(f"\n{model_size}:\n")
                    f.write(f"  Throughput:   {time_est['tokens_per_sec']:>10,.0f} tokens/sec\n")

                    if time_est['days'] >= 1:
                        f.write(f"  Time/epoch:   {time_est['days']:>10,.2f} days ({time_est['hours']:.1f} hours)\n")
                    else:
                        f.write(f"  Time/epoch:   {time_est['hours']:>10,.2f} hours\n")

                    # Time recommendations
                    if time_est['hours'] <= 4:
                        f.write(f"  ✓ Very fast - ideal for experimentation\n")
                    elif time_est['hours'] <= 24:
                        f.write(f"  ✓ Fast - reasonable for iterative development\n")
                    elif time_est['days'] <= 3:
                        f.write(f"  ⚠ Moderate - plan training runs carefully\n")
                    else:
                        f.write(f"  ⚠ Slow - consider smaller model or shorter context\n")

            f.write("\n")

    logger.info(f"Saved detailed report to: {report_path}")

    # Print summary tables to console
    print("\n" + "=" * 80)
    print("TOKEN COUNT SUMMARY")
    print("=" * 80)
    print(f"\nTotal hospitalizations: {stats['count']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Mean tokens: {stats['mean']:,.1f}")
    print(f"Median tokens: {stats['median']:,.1f}")
    print(f"Max tokens: {stats['max']:,}\n")

    print("CONTEXT WINDOW COVERAGE")
    print("-" * 80)
    print(context_df)

    print("\n" + "=" * 80)
    print(f"GPU MEMORY REQUIREMENTS ({precision.upper()}, batch={batch_size})")
    print("=" * 80)
    # Create a pivot table for GPU memory
    gpu_pivot_rows = []
    for context_size in sorted(set(cs for cs, _ in gpu_results.keys())):
        row = {"context_size": context_size}
        for model_size in model_sizes:
            mem_req = gpu_results.get((context_size, model_size))
            if mem_req:
                row[f"{model_size}_vram_gb"] = round(mem_req['total'], 1)
        gpu_pivot_rows.append(row)

    if gpu_pivot_rows:
        gpu_pivot_df = pl.DataFrame(gpu_pivot_rows)
        print(gpu_pivot_df)

    print("\n" + "=" * 80)
    print(f"TRAINING TIME (1 EPOCH on {hardware})")
    print("=" * 80)
    # Create a pivot table for training time
    time_pivot_rows = []
    for context_size in sorted(set(cs for cs, _ in time_results.keys())):
        row = {"context_size": context_size}
        for model_size in model_sizes:
            time_est = time_results.get((context_size, model_size))
            if time_est:
                if time_est['days'] >= 1:
                    row[f"{model_size}_time"] = f"{time_est['days']:.2f}d"
                else:
                    row[f"{model_size}_time"] = f"{time_est['hours']:.2f}h"
        time_pivot_rows.append(row)

    if time_pivot_rows:
        time_pivot_df = pl.DataFrame(time_pivot_rows)
        print(time_pivot_df)

    print(f"\n{'=' * 80}")
    print("OUTPUT FILES")
    print(f"{'=' * 80}")
    print(f"Token counts:      {csv_path}")
    print(f"Context analysis:  {context_csv}")
    print(f"GPU requirements:  {gpu_csv}")
    print(f"Training time:     {time_csv}")
    print(f"Detailed report:   {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token counts per hospitalization with GPU and training time estimates"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="clif_output_rush/sentences/clif_sentences.parquet",
        help="Path to clif_sentences.parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gpt2_output/token_analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--context-sizes",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
        help="Context window sizes to analyze",
    )
    parser.add_argument(
        "--model-sizes",
        type=str,
        nargs="+",
        default=["124M", "355M", "774M"],
        choices=["124M", "355M", "774M", "1.5B"],
        help="Model sizes to analyze for GPU/time estimates",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default="2xL40",
        choices=["A100", "2xL40", "H100", "4xA100"],
        help="Hardware type for training time estimates",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for GPU memory and time estimates",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=True,
        help="Use flash attention (reduces memory)",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_false",
        dest="flash_attn",
        help="Disable flash attention",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action="store_true",
        default=False,
        help="Use gradient checkpointing (reduces memory, increases compute)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N random hospitalizations for faster analysis",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CLIF Hospitalization Token Count Analysis")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Model sizes: {', '.join(args.model_sizes)}")
    logger.info(f"  Hardware: {args.hardware}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Precision: {args.precision}")
    logger.info(f"  Flash attention: {args.flash_attn}")
    logger.info(f"  Gradient checkpointing: {args.grad_checkpoint}")

    # Load data
    input_path = pathlib.Path(args.input)
    df = load_clif_data(input_path)

    # Sample if requested
    if args.sample:
        logger.info(f"Sampling {args.sample} random hospitalizations...")
        unique_hosps = df["hospitalization_id"].unique()
        if len(unique_hosps) > args.sample:
            sampled_hosps = unique_hosps.sample(args.sample)
            df = df.filter(pl.col("hospitalization_id").is_in(sampled_hosps))
            logger.info(f"Sampled to {len(df):,} event lines")

    # Calculate tokens per hospitalization
    hosp_tokens = calculate_tokens_per_hospitalization(df)

    # Compute statistics
    stats = compute_statistics(hosp_tokens)

    # Analyze context windows
    context_results = analyze_context_windows(hosp_tokens, args.context_sizes)

    # Analyze GPU requirements
    gpu_results = analyze_gpu_requirements(
        context_sizes=args.context_sizes,
        model_sizes=args.model_sizes,
        batch_size=args.batch_size,
        precision=args.precision,
        use_flash_attn=args.flash_attn,
        gradient_checkpointing=args.grad_checkpoint
    )

    # Analyze training time
    time_results = analyze_training_time(
        context_sizes=args.context_sizes,
        model_sizes=args.model_sizes,
        total_tokens=stats['total_tokens'],
        hardware=args.hardware,
        batch_size=args.batch_size
    )

    # Generate report
    output_dir = pathlib.Path(args.output_dir)
    generate_report(
        hosp_tokens=hosp_tokens,
        stats=stats,
        context_results=context_results,
        gpu_results=gpu_results,
        time_results=time_results,
        model_sizes=args.model_sizes,
        hardware=args.hardware,
        batch_size=args.batch_size,
        precision=args.precision,
        use_flash_attn=args.flash_attn,
        gradient_checkpointing=args.grad_checkpoint,
        output_dir=output_dir
    )

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
