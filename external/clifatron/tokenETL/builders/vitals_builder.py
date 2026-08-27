"""
Vitals Builder Module

Loads vitals data for cohort and creates vital tokens based on interval-aware numeric binning.
Bins are loaded from config/critical_illness_tokenization_final_with_intervals.csv.
Uses polars for 10-100x faster binning operations with mathematical interval notation support.
"""

import os
import pandas as pd
import logging
from typing import Dict, Any, Tuple
from clifpy.tables import Vitals
from utils.polars_utils import read_numeric_ranges_polars, bin_numeric_values_with_intervals_by_category, strip_all_datetime_timezones


def load_numeric_ranges(category: str, script_dir: str) -> pd.DataFrame:
    """
    Load numeric ranges CSV and filter to specific category using polars (15x faster).

    Args:
        category: Category to filter (e.g., 'labs', 'vitals', 'respiratory_support', 'medications')
        script_dir: Directory containing config folder

    Returns:
        Filtered DataFrame with numeric ranges
    """
    ranges_path = os.path.join(script_dir, 'config', 'critical_illness_tokenization_final_with_intervals.csv')

    # Use polars for faster CSV reading
    return read_numeric_ranges_polars(ranges_path, category)


def assign_token_from_bins(value: float, bins_df: pd.DataFrame) -> str:
    """
    Assign token based on numeric value and bins.

    Args:
        value: Numeric value to bin
        bins_df: DataFrame with min_value, max_value, token columns

    Returns:
        Token string or None if no match
    """
    if pd.isna(value):
        return None

    # Find matching bin (left-inclusive, right-exclusive: min_value <= value < max_value)
    # But handle edge case where value equals max of last bin
    matches = bins_df[
        (bins_df['min_value'] <= value) &
        ((bins_df['max_value'] > value) |
         ((bins_df['max_value'] == value) & (bins_df['max_value'] == bins_df['max_value'].max())))
    ]

    if len(matches) > 0:
        return matches.iloc[0]['token']
    else:
        return None


def build_vitals_tokens(
    cohort_df: pd.DataFrame,
    config_path: str,
    token_config: Dict[str, Any],
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load vitals for cohort and create vital tokens based on interval-aware numeric binning.

    Process:
    1. Load critical_illness_tokenization_final_with_intervals.csv and filter to vitals category
    2. Get cohort hospitalization_ids
    3. Load vitals data filtered to cohort
    4. Bin vital_value using interval-aware binning (supports [, (, ], ) notation)
    5. Create vitals_token column
    6. Filter out unmapped values
    7. Count token occurrences

    Args:
        cohort_df: Cohort DataFrame with hospitalization_id column
        config_path: Path to clif_config.json
        token_config: Loaded token configuration
        logger: Logger instance

    Returns:
        Tuple of (DataFrame with hospitalization_id, recorded_dttm, vital_category,
                  vital_value, vitals_token, token_counts dict)
    """
    if logger:
        logger.info("=" * 60)
        logger.info("PHASE 8: VITALS")
        logger.info("=" * 60)

    # Get script directory for loading numeric_ranges.csv
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load numeric ranges for vitals
    if logger:
        logger.info("Loading numeric ranges for vitals...")

    ranges_df = load_numeric_ranges('vitals', script_dir)

    if logger:
        logger.info(f"  ✓ Loaded {len(ranges_df)} vital bins")

    # Get unique vital categories available in bins
    available_categories = ranges_df['measurement'].unique()

    if logger:
        logger.info(f"  ✓ Vital categories with bins: {len(available_categories)}")
        logger.info(f"    {sorted(available_categories)}")

    # Get cohort hospitalization_ids
    cohort_hosp_ids = cohort_df['hospitalization_id'].unique()

    if logger:
        logger.info("")
        logger.info(f"Cohort hospitalizations: {len(cohort_hosp_ids):,}")
        logger.info("Loading vitals for cohort hospitalizations...")

    # Load vitals filtered to cohort
    vitals_table = Vitals.from_file(
        config_path=config_path,
        columns=['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        filters={'hospitalization_id': list(cohort_hosp_ids)}
    )
    vitals_df = vitals_table.df.copy()

    # Strip timezone from ALL datetime columns using utility function
    vitals_df = strip_all_datetime_timezones(vitals_df)

    if logger:
        logger.info(f"  ✓ Loaded {len(vitals_df):,} vital records")

    # Filter to only vital categories with bins (should be all of them)
    if logger:
        logger.info("")
        logger.info(f"Filtering to vital categories with bins in critical_illness_tokenization_final_with_intervals.csv...")

    initial_count = len(vitals_df)
    vitals_df = vitals_df[vitals_df['vital_category'].isin(available_categories)].copy()
    removed = initial_count - len(vitals_df)

    if logger:
        pct_kept = (len(vitals_df) / initial_count * 100) if initial_count > 0 else 0
        logger.info(f"  ✓ Kept {len(vitals_df):,} records ({pct_kept:.1f}%)")
        if removed > 0:
            logger.info(f"  ✓ Removed {removed:,} records (categories not in bins)")

    # Tokenize each vital category using interval-aware binning (10-100x faster than pandas apply)
    if logger:
        logger.info("")
        logger.info("Tokenizing vital values using interval-aware binning (category-based processing)...")

    # Use interval-aware category-based binning with streaming to prevent memory exhaustion
    # Processes each vital_category separately with early deduplication in polars
    vitals_with_tokens = bin_numeric_values_with_intervals_by_category(
        df=vitals_df,
        value_col='vital_value',
        category_col='vital_category',
        bins_df=ranges_df,
        measurement_col='measurement',
        logger=logger
    )

    # Extract the token column (deduplication now handled within polars streaming pipeline)
    vitals_df['vitals_token'] = vitals_with_tokens.set_index(
        ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value']
    )['token'].reindex(
        vitals_df.set_index(['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value']).index
    ).values

    # Log binning results per category
    if logger:
        logger.info("")
        for category in sorted(vitals_df['vital_category'].unique()):
            category_df = vitals_df[vitals_df['vital_category'] == category]
            total = len(category_df)
            tokenized = category_df['vitals_token'].notna().sum()
            pct = (tokenized / total * 100) if total > 0 else 0
            logger.info(f"  {category}: {tokenized:,}/{total:,} ({pct:.1f}%) tokenized")

    # Filter out unmapped values
    initial_count = len(vitals_df)
    vitals_df = vitals_df[vitals_df['vitals_token'].notna()].copy()
    filtered_out = initial_count - len(vitals_df)

    if logger:
        logger.info("")
        logger.info(f"Total records: {initial_count:,}")
        logger.info(f"  ✓ Successfully tokenized: {len(vitals_df):,} ({len(vitals_df)/initial_count*100:.2f}%)")
        if filtered_out > 0:
            logger.info(f"  ✓ Filtered out (unmapped): {filtered_out:,} ({filtered_out/initial_count*100:.2f}%)")

    # Count vital tokens
    token_counts = {}
    vital_token_counts = vitals_df['vitals_token'].value_counts()
    for token, count in vital_token_counts.items():
        if pd.notna(token):
            token_counts[token] = count

    if logger:
        logger.info("")
        logger.info(f"Vital token counts:")
        logger.info(f"  Total unique tokens: {len(token_counts)}")
        logger.info(f"  Total token occurrences: {sum(token_counts.values()):,}")
        logger.info("=" * 60)

    # Return final columns and token counts
    return vitals_df[['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value', 'vitals_token']], token_counts
