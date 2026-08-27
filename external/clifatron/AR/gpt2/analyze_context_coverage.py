#!/usr/bin/env python3

"""
Context Coverage Analysis for CLIF Data

Analyzes how many days of patient data can fit in different context lengths.
Helps determine optimal context size for training and inference.

Usage:
    uv run gpt2/analyze_context_coverage.py --input clif_output/sentences/clif_sentences.parquet
"""

import argparse
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple

import polars as pl
import numpy as np

from utils import setup_logging

logger = setup_logging()


def load_clif_data(input_path: pathlib.Path) -> pl.DataFrame:
    """
    Load CLIF sentences with event time information

    Args:
        input_path: Path to clif_sentences.parquet

    Returns:
        DataFrame with hospitalization_id, event_time, clif_sentence, line_number
    """
    logger.info(f"Loading CLIF data from: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pl.read_parquet(input_path)

    logger.info(f"Loaded {len(df):,} event lines")
    logger.info(f"Columns: {df.columns}")

    # Check required columns
    required_cols = ['hospitalization_id', 'event_time', 'clif_sentence']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def calculate_token_counts(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate token count for each event line

    Args:
        df: DataFrame with clif_sentence column

    Returns:
        DataFrame with added token_count column
    """
    logger.info("Calculating token counts...")

    # Count tokens (space-separated)
    df = df.with_columns([
        pl.col("clif_sentence")
        .str.split(" ")
        .list.len()
        .alias("token_count")
    ])

    return df


def analyze_per_hospitalization(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate metrics per hospitalization

    Args:
        df: DataFrame with hospitalization_id, event_time, token_count

    Returns:
        DataFrame with per-hospitalization metrics
    """
    logger.info("Analyzing per-hospitalization metrics...")

    # Convert event_time to datetime if needed
    if df["event_time"].dtype != pl.Datetime:
        df = df.with_columns([
            pl.col("event_time").cast(pl.Datetime)
        ])

    # Group by hospitalization and calculate metrics
    hosp_metrics = df.group_by("hospitalization_id").agg([
        # Time span
        pl.col("event_time").min().alias("first_event"),
        pl.col("event_time").max().alias("last_event"),

        # Event counts
        pl.col("event_time").count().alias("total_lines"),

        # Token stats
        pl.col("token_count").sum().alias("total_tokens"),
        pl.col("token_count").mean().alias("avg_tokens_per_line"),
        pl.col("token_count").median().alias("median_tokens_per_line"),
    ])

    # Calculate time span in days
    hosp_metrics = hosp_metrics.with_columns([
        ((pl.col("last_event") - pl.col("first_event")).dt.total_seconds() / 86400)
        .alias("duration_days")
    ])

    # Avoid division by zero - set minimum duration to 0.1 days (2.4 hours)
    hosp_metrics = hosp_metrics.with_columns([
        pl.when(pl.col("duration_days") < 0.1)
        .then(0.1)
        .otherwise(pl.col("duration_days"))
        .alias("duration_days")
    ])

    # Calculate density metrics
    hosp_metrics = hosp_metrics.with_columns([
        (pl.col("total_lines") / pl.col("duration_days")).alias("lines_per_day"),
        (pl.col("total_tokens") / pl.col("duration_days")).alias("tokens_per_day"),
    ])

    logger.info(f"Analyzed {len(hosp_metrics):,} unique hospitalizations")

    return hosp_metrics


def analyze_context_coverage(
    hosp_metrics: pl.DataFrame,
    context_sizes: List[int]
) -> Dict[int, Dict[str, float]]:
    """
    Analyze coverage for different context sizes

    Args:
        hosp_metrics: Per-hospitalization metrics
        context_sizes: List of context sizes to analyze

    Returns:
        Dictionary mapping context_size -> metrics
    """
    logger.info("Analyzing context coverage for different sizes...")

    results = {}

    for context_size in context_sizes:
        logger.info(f"\nAnalyzing context size: {context_size}")

        # Calculate days of coverage for each hospitalization
        hosp_with_coverage = hosp_metrics.with_columns([
            (context_size / pl.col("tokens_per_day")).alias("days_coverage"),
            (context_size / pl.col("total_tokens") * 100).alias("pct_coverage"),
        ])

        # Calculate statistics
        days_coverage = hosp_with_coverage["days_coverage"]
        pct_coverage = hosp_with_coverage["pct_coverage"]

        # Filter out extreme outliers for percentiles (cap at 100%)
        pct_coverage_capped = pct_coverage.clip(0, 100)

        results[context_size] = {
            # Days coverage
            "mean_days": days_coverage.mean(),
            "median_days": days_coverage.median(),
            "p25_days": days_coverage.quantile(0.25),
            "p75_days": days_coverage.quantile(0.75),

            # Percentage coverage
            "mean_pct": pct_coverage_capped.mean(),
            "median_pct": pct_coverage_capped.median(),
            "pct_fully_covered": (pct_coverage >= 100).mean() * 100,

            # Additional stats
            "pct_50_covered": (pct_coverage >= 50).mean() * 100,
            "pct_75_covered": (pct_coverage >= 75).mean() * 100,
        }

        logger.info(f"  Mean days coverage: {results[context_size]['mean_days']:.2f}")
        logger.info(f"  Median days coverage: {results[context_size]['median_days']:.2f}")
        logger.info(f"  % hospitalizations fully covered: {results[context_size]['pct_fully_covered']:.1f}%")

    return results


def generate_report(
    hosp_metrics: pl.DataFrame,
    coverage_results: Dict[int, Dict[str, float]],
    output_dir: pathlib.Path
):
    """
    Generate comprehensive coverage report

    Args:
        hosp_metrics: Per-hospitalization metrics
        coverage_results: Coverage analysis results
        output_dir: Output directory
    """
    logger.info("Generating coverage report...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save detailed hospitalization metrics
    metrics_path = output_dir / "hospitalization_metrics.parquet"
    hosp_metrics.write_parquet(metrics_path)
    logger.info(f"Saved detailed metrics to: {metrics_path}")

    # 2. Save summary CSV
    summary_rows = []
    for context_size, metrics in sorted(coverage_results.items()):
        summary_rows.append({
            "context_size": context_size,
            "mean_days_coverage": round(metrics["mean_days"], 2),
            "median_days_coverage": round(metrics["median_days"], 2),
            "p25_days": round(metrics["p25_days"], 2),
            "p75_days": round(metrics["p75_days"], 2),
            "pct_fully_covered": round(metrics["pct_fully_covered"], 1),
            "pct_75_covered": round(metrics["pct_75_covered"], 1),
            "pct_50_covered": round(metrics["pct_50_covered"], 1),
        })

    summary_df = pl.DataFrame(summary_rows)
    summary_path = output_dir / "context_analysis_summary.csv"
    summary_df.write_csv(summary_path)
    logger.info(f"Saved summary to: {summary_path}")

    # 3. Generate detailed text report
    report_path = output_dir / "context_coverage_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CLIF Context Coverage Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Overall statistics
        f.write("Overall Dataset Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total hospitalizations: {len(hosp_metrics):,}\n")
        f.write(f"Total event lines: {hosp_metrics['total_lines'].sum():,}\n")
        f.write(f"Total tokens: {hosp_metrics['total_tokens'].sum():,}\n\n")

        f.write(f"Average per hospitalization:\n")
        f.write(f"  Duration: {hosp_metrics['duration_days'].mean():.2f} days (median: {hosp_metrics['duration_days'].median():.2f})\n")
        f.write(f"  Event lines: {hosp_metrics['total_lines'].mean():.1f} (median: {hosp_metrics['total_lines'].median():.0f})\n")
        f.write(f"  Total tokens: {hosp_metrics['total_tokens'].mean():.1f} (median: {hosp_metrics['total_tokens'].median():.0f})\n")
        f.write(f"  Lines per day: {hosp_metrics['lines_per_day'].mean():.1f} (median: {hosp_metrics['lines_per_day'].median():.1f})\n")
        f.write(f"  Tokens per day: {hosp_metrics['tokens_per_day'].mean():.1f} (median: {hosp_metrics['tokens_per_day'].median():.1f})\n\n")

        # Context size analysis
        f.write("=" * 80 + "\n")
        f.write("Context Size Coverage Analysis\n")
        f.write("=" * 80 + "\n\n")

        for context_size, metrics in sorted(coverage_results.items()):
            f.write(f"Context Size: {context_size} tokens\n")
            f.write("-" * 80 + "\n")
            f.write(f"Days of coverage:\n")
            f.write(f"  Mean: {metrics['mean_days']:.2f} days\n")
            f.write(f"  Median: {metrics['median_days']:.2f} days\n")
            f.write(f"  25th percentile: {metrics['p25_days']:.2f} days\n")
            f.write(f"  75th percentile: {metrics['p75_days']:.2f} days\n\n")

            f.write(f"Coverage percentages:\n")
            f.write(f"  Fully covered (100%): {metrics['pct_fully_covered']:.1f}% of hospitalizations\n")
            f.write(f"  ≥75% covered: {metrics['pct_75_covered']:.1f}% of hospitalizations\n")
            f.write(f"  ≥50% covered: {metrics['pct_50_covered']:.1f}% of hospitalizations\n\n")

            # Recommendations
            f.write("Recommendations:\n")
            if metrics['pct_fully_covered'] >= 80:
                f.write(f"  ✓ Excellent for complete timeline modeling\n")
            elif metrics['pct_fully_covered'] >= 50:
                f.write(f"  ✓ Good for most use cases\n")
            else:
                f.write(f"  ⚠ May need sequence truncation strategies\n")

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

        # Find best context size
        best_for_coverage = max(coverage_results.items(), key=lambda x: x[1]['pct_fully_covered'])
        best_for_efficiency = 2048  # Sweet spot for L40s

        f.write(f"For maximum coverage:\n")
        f.write(f"  → {best_for_coverage[0]} tokens ({best_for_coverage[1]['pct_fully_covered']:.1f}% fully covered)\n\n")

        f.write(f"For 2x L40 training efficiency:\n")
        if best_for_efficiency in coverage_results:
            metrics = coverage_results[best_for_efficiency]
            f.write(f"  → {best_for_efficiency} tokens\n")
            f.write(f"  → Covers {metrics['median_days']:.2f} days (median)\n")
            f.write(f"  → {metrics['pct_fully_covered']:.1f}% fully covered\n")
            f.write(f"  → Good balance of coverage and GPU memory\n\n")

        f.write(f"For early prediction tasks (<3 days):\n")
        f.write(f"  → 1024 tokens sufficient for most cases\n")
        f.write(f"  → Faster inference\n\n")

        f.write(f"For full timeline modeling:\n")
        best_full = [cs for cs, m in coverage_results.items() if m['pct_fully_covered'] >= 75]
        if best_full:
            f.write(f"  → {min(best_full)} tokens minimum\n")
            f.write(f"  → Consider {max(best_full)} tokens for comprehensive coverage\n\n")

    logger.info(f"Saved detailed report to: {report_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("CONTEXT COVERAGE SUMMARY")
    print("=" * 80)
    print(summary_df)
    print("\nSee detailed report at:", report_path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze context coverage for CLIF data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="clif_output/sentences/clif_sentences.parquet",
        help="Path to clif_sentences.parquet file (default: clif_output/sentences/clif_sentences.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gpt2_output/context_analysis",
        help="Output directory for analysis results (default: gpt2_output/context_analysis)",
    )
    parser.add_argument(
        "--context-sizes",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Context sizes to analyze (default: 1024 2048 4096 8192)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N random hospitalizations for faster analysis (default: use all)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CLIF Context Coverage Analysis")
    logger.info("=" * 80)

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

    # Calculate token counts
    df = calculate_token_counts(df)

    # Analyze per hospitalization
    hosp_metrics = analyze_per_hospitalization(df)

    # Analyze context coverage
    coverage_results = analyze_context_coverage(hosp_metrics, args.context_sizes)

    # Generate report
    output_dir = pathlib.Path(args.output_dir)
    generate_report(hosp_metrics, coverage_results, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
