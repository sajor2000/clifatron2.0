#!/usr/bin/env python3
"""
Medication Quantile and ECDF Analysis

Analyzes medication dose distributions from medication_admin_continuous.parquet:
- Calculates deciles (10 quantiles) per medication-unit combination
- Computes empirical cumulative distribution functions (ECDF)
- Generates interactive plotly overlay plots (histogram + ECDF together)
- Saves quantile and ECDF data as CSV files
- Creates HTML plots with interactive hover and zoom

Usage:
    uv run tokenETL/medication_quantile_analysis.py
"""

import os
import logging
from pathlib import Path
from typing import Tuple
import polars as pl
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


# ============================================================================
# Configuration
# ============================================================================

# Hardcoded paths - adjust as needed
INPUT_DIR = "output_tokens"  # Directory containing medication_admin_continuous.parquet
OUTPUT_DIR = "output_tokens/medication_analysis"  # Where to save results
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
OVERLAY_PLOTS_DIR = os.path.join(PLOTS_DIR, "overlay")
COMBINED_PLOTS_DIR = os.path.join(PLOTS_DIR, "combined")

# Quantile levels (deciles: 10 bins)
QUANTILE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger() -> logging.Logger:
    """Set up logger for the analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_path = os.path.join(OUTPUT_DIR, 'medication_quantile_analysis.log')

    logger = logging.getLogger('medication_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("Medication Quantile & ECDF Analysis")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_path}")

    return logger


# ============================================================================
# Data Loading
# ============================================================================

def load_medication_data(logger: logging.Logger) -> pl.DataFrame:
    """
    Load medication_admin_continuous.parquet and filter to successful conversions.

    Args:
        logger: Logger instance

    Returns:
        Polars DataFrame with successful medication conversions
    """
    logger.info("=" * 60)
    logger.info("LOADING MEDICATION DATA")
    logger.info("=" * 60)

    input_path = os.path.join(INPUT_DIR, "medication_admin_continuous.parquet")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Medication data not found: {input_path}")

    logger.info(f"Reading: {input_path}")

    # Load with polars
    df = pl.read_parquet(input_path)

    logger.info(f"  ✓ Loaded {len(df):,} total records")
    logger.info(f"  ✓ Columns: {df.columns}")

    # Filter to successful conversions only
    logger.info("")
    logger.info("Filtering to successful conversions...")

    initial_count = len(df)
    df = df.filter(pl.col("_convert_status") == "success")
    final_count = len(df)

    logger.info(f"  ✓ Successful conversions: {final_count:,} / {initial_count:,} ({final_count/initial_count*100:.1f}%)")

    # Select relevant columns
    df = df.select([
        "hospitalization_id",
        "admin_dttm",
        "med_category",
        "med_dose_converted",
        "med_dose_unit_converted"
    ])

    # Summary stats
    logger.info("")
    logger.info("Data summary:")
    n_med_categories = df.select("med_category").n_unique()
    n_units = df.select("med_dose_unit_converted").n_unique()
    n_combinations = df.select(["med_category", "med_dose_unit_converted"]).unique().height

    logger.info(f"  - Unique medications: {n_med_categories}")
    logger.info(f"  - Unique units: {n_units}")
    logger.info(f"  - Unique med-unit combinations: {n_combinations}")

    logger.info("=" * 60)

    return df


# ============================================================================
# Quantile Calculation
# ============================================================================

def calculate_quantiles(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """
    Calculate decile bins for each med_category + med_dose_unit_converted combination.

    Creates 10 bins based on quantiles and counts administrations in each bin.

    Args:
        df: Medication DataFrame
        logger: Logger instance

    Returns:
        Polars DataFrame with bin information (min_bin, max_bin, interval, n_in_bin, etc.)
    """
    logger.info("=" * 60)
    logger.info("CALCULATING QUANTILE BINS")
    logger.info("=" * 60)

    logger.info(f"Quantile levels: {QUANTILE_LEVELS}")
    logger.info(f"Creating {len(QUANTILE_LEVELS) - 1} bins per medication-unit combination")
    logger.info("")

    # Group by medication-unit combination and calculate quantile bins
    quantiles_list = []

    # Get unique combinations
    combinations = df.select(["med_category", "med_dose_unit_converted"]).unique()

    logger.info(f"Processing {len(combinations):,} medication-unit combinations...")

    for row in combinations.iter_rows(named=True):
        med_cat = row["med_category"]
        unit = row["med_dose_unit_converted"]

        # Filter to this combination
        subset = df.filter(
            (pl.col("med_category") == med_cat) &
            (pl.col("med_dose_unit_converted") == unit)
        )

        n_obs = len(subset)
        doses = subset.select("med_dose_converted").to_series()
        doses_array = doses.to_numpy()

        # Calculate quantile values
        quantile_values = [doses.quantile(q) for q in QUANTILE_LEVELS]

        # Create bins from consecutive quantiles
        for i in range(len(QUANTILE_LEVELS) - 1):
            q_level = QUANTILE_LEVELS[i]
            min_bin = quantile_values[i]
            max_bin = quantile_values[i + 1]

            # Count doses in this bin [min_bin, max_bin)
            # For the last bin, include max_bin
            if i == len(QUANTILE_LEVELS) - 2:  # Last bin
                n_in_bin = ((doses_array >= min_bin) & (doses_array <= max_bin)).sum()
            else:
                n_in_bin = ((doses_array >= min_bin) & (doses_array < max_bin)).sum()

            # Create interval string
            interval = f"[{min_bin:.4g}, {max_bin:.4g})"
            if i == len(QUANTILE_LEVELS) - 2:  # Last bin is inclusive
                interval = f"[{min_bin:.4g}, {max_bin:.4g}]"

            quantiles_list.append({
                "med_category": med_cat,
                "med_dose_unit_converted": unit,
                "quantile_level": q_level,
                "dose_value": min_bin,
                "min_bin": min_bin,
                "max_bin": max_bin,
                "interval": interval,
                "n_in_bin": int(n_in_bin),
                "n_observations": n_obs
            })

    # Create DataFrame
    quantiles_df = pl.DataFrame(quantiles_list)

    logger.info(f"  ✓ Created bins for {len(combinations):,} medication-unit combinations")
    logger.info(f"  ✓ Total bins: {len(quantiles_df):,} ({len(QUANTILE_LEVELS) - 1} bins per combination)")

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, "medication_quantiles.csv")
    quantiles_df.write_csv(output_path)

    file_size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"  ✓ Saved: {output_path} ({file_size_kb:.2f} KB)")

    logger.info("=" * 60)

    return quantiles_df


# ============================================================================
# ECDF Calculation
# ============================================================================

def calculate_ecdf(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    """
    Calculate empirical cumulative distribution function for each medication-unit combination.

    Args:
        df: Medication DataFrame
        logger: Logger instance

    Returns:
        Polars DataFrame with ECDF values
    """
    logger.info("=" * 60)
    logger.info("CALCULATING ECDF")
    logger.info("=" * 60)

    ecdf_list = []

    # Get unique combinations
    combinations = df.select(["med_category", "med_dose_unit_converted"]).unique()

    logger.info(f"Processing {len(combinations):,} medication-unit combinations...")

    for row in combinations.iter_rows(named=True):
        med_cat = row["med_category"]
        unit = row["med_dose_unit_converted"]

        # Filter to this combination
        subset = df.filter(
            (pl.col("med_category") == med_cat) &
            (pl.col("med_dose_unit_converted") == unit)
        )

        # Get sorted doses
        doses = subset.select("med_dose_converted").to_series().sort()
        n = len(doses)

        # Calculate ECDF values
        unique_doses = doses.unique().sort()

        for dose in unique_doses:
            ecdf_val = (doses <= dose).sum() / n

            ecdf_list.append({
                "med_category": med_cat,
                "med_dose_unit_converted": unit,
                "dose_value": float(dose),
                "ecdf_value": float(ecdf_val)
            })

    # Create DataFrame
    ecdf_df = pl.DataFrame(ecdf_list)

    logger.info(f"  ✓ Calculated ECDF for {len(combinations):,} combinations")
    logger.info(f"  ✓ Total ECDF points: {len(ecdf_df):,}")

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, "medication_ecdf.csv")
    ecdf_df.write_csv(output_path)

    file_size_kb = os.path.getsize(output_path) / 1024
    logger.info(f"  ✓ Saved: {output_path} ({file_size_kb:.2f} KB)")

    logger.info("=" * 60)

    return ecdf_df


# ============================================================================
# Overlay Plots (Histogram + ECDF together)
# ============================================================================

def create_overlay_plots(
    df: pl.DataFrame,
    quantiles_df: pl.DataFrame,
    ecdf_df: pl.DataFrame,
    logger: logging.Logger
):
    """
    Create plotly overlay plots with histogram + ECDF on same plot for each medication-unit combination.

    Args:
        df: Medication DataFrame
        quantiles_df: Quantiles DataFrame
        ecdf_df: ECDF DataFrame
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("CREATING OVERLAY PLOTS (Histogram + ECDF)")
    logger.info("=" * 60)

    # Create output directory
    os.makedirs(OVERLAY_PLOTS_DIR, exist_ok=True)

    # Get unique combinations
    combinations = df.select(["med_category", "med_dose_unit_converted"]).unique()

    logger.info(f"Generating interactive plotly plots for {len(combinations):,} combinations...")

    for idx, row in enumerate(combinations.iter_rows(named=True), 1):
        med_cat = row["med_category"]
        unit = row["med_dose_unit_converted"]

        # Filter data
        subset = df.filter(
            (pl.col("med_category") == med_cat) &
            (pl.col("med_dose_unit_converted") == unit)
        )

        doses = subset.select("med_dose_converted").to_series().to_numpy()
        n_obs = len(doses)

        # Get quantile bins for this combination
        quant_subset = quantiles_df.filter(
            (pl.col("med_category") == med_cat) &
            (pl.col("med_dose_unit_converted") == unit)
        ).sort("quantile_level")

        bins = quant_subset.select("max_bin").to_series().to_numpy()
        bins = np.concatenate([[quant_subset.select("min_bin").to_series().to_numpy()[0]], bins])

        # Get ECDF for this combination
        ecdf_subset = ecdf_df.filter(
            (pl.col("med_category") == med_cat) &
            (pl.col("med_dose_unit_converted") == unit)
        ).sort("dose_value")

        ecdf_doses = ecdf_subset.select("dose_value").to_series().to_numpy()
        ecdf_vals = ecdf_subset.select("ecdf_value").to_series().to_numpy()

        # Calculate quartiles for reference lines
        q25, q50, q75 = np.percentile(doses, [25, 50, 75])

        # Create plotly figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add histogram (primary y-axis)
        hist_counts, _ = np.histogram(doses, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist_counts,
                name="Frequency",
                marker=dict(color='steelblue', opacity=0.7, line=dict(color='black', width=1)),
                hovertemplate='Dose: %{x}<br>Count: %{y}<extra></extra>'
            ),
            secondary_y=False
        )

        # Add ECDF line (secondary y-axis)
        fig.add_trace(
            go.Scatter(
                x=ecdf_doses,
                y=ecdf_vals,
                name="ECDF",
                line=dict(color='darkred', width=3),
                hovertemplate='Dose: %{x}<br>ECDF: %{y:.3f}<extra></extra>'
            ),
            secondary_y=True
        )

        # Add quartile reference lines
        for q_val, q_name, color in [(q25, 'Q1 (25%)', 'orange'),
                                       (q50, 'Median', 'red'),
                                       (q75, 'Q3 (75%)', 'purple')]:
            fig.add_vline(
                x=q_val,
                line_dash="dash",
                line_color=color,
                opacity=0.6,
                annotation_text=q_name,
                annotation_position="top"
            )

        # Update axes
        fig.update_xaxes(title_text=f"Dose ({unit})")
        fig.update_yaxes(title_text="Frequency", secondary_y=False)
        fig.update_yaxes(title_text="ECDF", secondary_y=True, range=[-0.05, 1.05])

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{med_cat}<br><sub>n = {n_obs:,} administrations</sub>",
                x=0.5,
                xanchor='center'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(x=1.15, y=1)
        )

        # Save as HTML
        safe_filename = f"{med_cat}_{unit}".replace("/", "_").replace(" ", "_")
        output_path = os.path.join(OVERLAY_PLOTS_DIR, f"{safe_filename}.html")
        fig.write_html(output_path)

        if idx % 10 == 0:
            logger.info(f"  Progress: {idx}/{len(combinations)}")

    logger.info(f"  ✓ Created {len(combinations):,} overlay plots")
    logger.info(f"  ✓ Saved to: {OVERLAY_PLOTS_DIR}")
    logger.info("=" * 60)


# ============================================================================
# Combined Plots
# ============================================================================

def create_combined_plots(
    df: pl.DataFrame,
    quantiles_df: pl.DataFrame,
    ecdf_df: pl.DataFrame,
    logger: logging.Logger
):
    """
    Create combined multi-panel plotly plot with histogram + ECDF overlays for all medications.

    Args:
        df: Medication DataFrame
        quantiles_df: Quantiles DataFrame
        ecdf_df: ECDF DataFrame
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("CREATING COMBINED OVERLAY PLOTS")
    logger.info("=" * 60)

    # Create output directory
    os.makedirs(COMBINED_PLOTS_DIR, exist_ok=True)

    # Get unique combinations sorted by observation count
    combinations = (
        df.group_by(["med_category", "med_dose_unit_converted"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )

    n_combinations = len(combinations)
    logger.info(f"Creating combined multi-panel plot for {n_combinations} combinations...")

    # Determine grid layout
    n_cols = 3
    n_rows = int(np.ceil(n_combinations / n_cols))

    # Create subplot titles
    subplot_titles = []
    for row in combinations.iter_rows(named=True):
        med_cat = row["med_category"]
        count = row["count"]
        subplot_titles.append(f"{med_cat}<br><sub>n={count:,}</sub>")

    # Create subplots with secondary y-axes
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": True} for _ in range(n_cols)] for _ in range(n_rows)],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # Add traces for each medication
    for idx, comb_row in enumerate(combinations.iter_rows(named=True)):
        med_cat = comb_row["med_category"]
        unit = comb_row["med_dose_unit_converted"]

        row_idx = idx // n_cols + 1
        col_idx = idx % n_cols + 1

        # Filter data
        subset = df.filter(
            (pl.col("med_category") == med_cat) &
            (pl.col("med_dose_unit_converted") == unit)
        )

        doses = subset.select("med_dose_converted").to_series().to_numpy()

        # Get quantile bins
        quant_subset = quantiles_df.filter(
            (pl.col("med_category") == med_cat) &
            (pl.col("med_dose_unit_converted") == unit)
        ).sort("quantile_level")

        bins = quant_subset.select("max_bin").to_series().to_numpy()
        bins = np.concatenate([[quant_subset.select("min_bin").to_series().to_numpy()[0]], bins])

        # Get ECDF
        ecdf_subset = ecdf_df.filter(
            (pl.col("med_category") == med_cat) &
            (pl.col("med_dose_unit_converted") == unit)
        ).sort("dose_value")

        ecdf_doses = ecdf_subset.select("dose_value").to_series().to_numpy()
        ecdf_vals = ecdf_subset.select("ecdf_value").to_series().to_numpy()

        # Add histogram
        hist_counts, _ = np.histogram(doses, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist_counts,
                name="Frequency" if idx == 0 else None,
                marker=dict(color='steelblue', opacity=0.7, line=dict(color='black', width=0.5)),
                showlegend=(idx == 0),
                hovertemplate=f'Dose: %{{x}}<br>Count: %{{y}}<extra></extra>'
            ),
            row=row_idx,
            col=col_idx,
            secondary_y=False
        )

        # Add ECDF line
        fig.add_trace(
            go.Scatter(
                x=ecdf_doses,
                y=ecdf_vals,
                name="ECDF" if idx == 0 else None,
                line=dict(color='darkred', width=2),
                showlegend=(idx == 0),
                hovertemplate=f'Dose: %{{x}}<br>ECDF: %{{y:.3f}}<extra></extra>'
            ),
            row=row_idx,
            col=col_idx,
            secondary_y=True
        )

        # Update axes for this subplot
        fig.update_xaxes(title_text=f"Dose ({unit})", row=row_idx, col=col_idx, title_font=dict(size=10))
        fig.update_yaxes(title_text="Frequency", row=row_idx, col=col_idx, secondary_y=False, title_font=dict(size=10))
        fig.update_yaxes(title_text="ECDF", row=row_idx, col=col_idx, secondary_y=True, range=[-0.05, 1.05], title_font=dict(size=10))

    # Update overall layout
    fig.update_layout(
        title=dict(
            text="Medication Dose Distributions - Histogram + ECDF Overlay",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=400 * n_rows,
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
    )

    # Save as HTML
    output_path = os.path.join(COMBINED_PLOTS_DIR, "all_medications_overlay.html")
    fig.write_html(output_path)

    logger.info(f"  ✓ Saved: {output_path}")
    logger.info("=" * 60)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main analysis pipeline."""
    # Setup
    logger = setup_logger()

    start_time = datetime.now()
    logger.info(f"Analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    try:
        # Step 1: Load data
        df = load_medication_data(logger)

        # Step 2: Calculate quantiles
        quantiles_df = calculate_quantiles(df, logger)

        # Step 3: Calculate ECDF
        ecdf_df = calculate_ecdf(df, logger)

        # Step 4: Create overlay plots
        create_overlay_plots(df, quantiles_df, ecdf_df, logger)

        # Step 5: Create combined plots
        create_combined_plots(df, quantiles_df, ecdf_df, logger)

        # Completion
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info("")
        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info("")
        logger.info("Generated files:")
        logger.info(f"  - medication_quantiles.csv")
        logger.info(f"  - medication_ecdf.csv")
        logger.info(f"  - plots/overlay/*.html (interactive plots)")
        logger.info(f"  - plots/combined/all_medications_overlay.html")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
