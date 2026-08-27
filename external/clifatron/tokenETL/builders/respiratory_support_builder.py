"""
Respiratory Support Builder Module

Loads respiratory support data for cohort and replaces numeric values with tokens.
Bins are loaded from config/critical_illness_tokenization_final_with_intervals.csv.
Uses optimized interval-aware binning with mathematical interval notation support.

Performance:
- Uses vectorized polars operations (10-100x faster than pandas apply)
- Processes data category-by-category for memory efficiency
- Transforms from wide to long format for processing, then back to wide
"""

import os
import pandas as pd
import logging
from typing import Dict, Any, Tuple
from clifpy.tables import RespiratorySupport
from utils.polars_utils import read_numeric_ranges_polars, strip_all_datetime_timezones, bin_numeric_values_with_intervals_by_category


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


def build_respiratory_support_tokens(
    cohort_df: pd.DataFrame,
    config_path: str,
    token_config: Dict[str, Any],
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load respiratory support for cohort and replace numeric values with interval-aware tokens.

    Process:
    1. Load critical_illness_tokenization_final_with_intervals.csv and filter to respiratory_support category
    2. Get cohort hospitalization_ids
    3. Load respiratory_support data filtered to cohort
    4. Transform numeric columns from wide to long format for efficient processing
    5. Apply vectorized interval-aware tokenization (10-100x faster than pandas apply)
    6. Transform back to wide format with tokenized values
    7. For tracheostomy: 1 -> "tracheostomy_present", 0/NA -> "NA"
    8. For device_category and mode_category, apply categorical tokenization
    9. Count all token occurrences

    Args:
        cohort_df: Cohort DataFrame with hospitalization_id column
        config_path: Path to clif_config.json
        token_config: Loaded token configuration
        logger: Logger instance

    Returns:
        Tuple of (DataFrame with all columns tokenized, token_counts dict)
    """
    if logger:
        logger.info("=" * 60)
        logger.info("PHASE 9: RESPIRATORY SUPPORT")
        logger.info("=" * 60)

    # Get script directory for loading numeric_ranges.csv
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load numeric ranges for respiratory support
    if logger:
        logger.info("Loading numeric ranges for respiratory support...")

    ranges_df = load_numeric_ranges('respiratory_support', script_dir)

    if logger:
        logger.info(f"  ✓ Loaded {len(ranges_df)} respiratory support bins")

    # Get unique measurements (column names) available in bins
    available_measurements = ranges_df['measurement'].unique()

    if logger:
        logger.info(f"  ✓ Measurements with bins: {len(available_measurements)}")
        logger.info(f"    {sorted(available_measurements)}")

    # Get cohort hospitalization_ids
    cohort_hosp_ids = cohort_df['hospitalization_id'].unique()

    if logger:
        logger.info("")
        logger.info(f"Cohort hospitalizations: {len(cohort_hosp_ids):,}")
        logger.info("Loading respiratory support for cohort hospitalizations...")

    # Define columns to load
    columns_to_load = [
        'hospitalization_id',
        'recorded_dttm',
        'device_category',
        'mode_category',
        'tracheostomy',
        'fio2_set',
        'lpm_set',
        'tidal_volume_set',
        'resp_rate_set',
        'pressure_control_set',
        'pressure_support_set',
        'flow_rate_set',
        'peak_inspiratory_pressure_set',
        'inspiratory_time_set',
        'peep_set',
        'tidal_volume_obs',
        'resp_rate_obs',
        'plateau_pressure_obs',
        'peak_inspiratory_pressure_obs',
        'peep_obs',
        'minute_vent_obs',
        'mean_airway_pressure_obs'
    ]

    # Load respiratory support filtered to cohort
    resp_table = RespiratorySupport.from_file(
        config_path=config_path,
        columns=columns_to_load,
        filters={'hospitalization_id': list(cohort_hosp_ids)}
    )
    resp_df = resp_table.df.copy()

    # Strip timezone from ALL datetime columns using utility function
    resp_df = strip_all_datetime_timezones(resp_df)

    if logger:
        logger.info(f"  ✓ Loaded {len(resp_df):,} respiratory support records")

    # Tokenize numeric columns using optimized vectorized approach
    if logger:
        logger.info("")
        logger.info("Tokenizing numeric values using interval-aware binning (category-based processing)...")

    numeric_columns = [
        'fio2_set', 'lpm_set', 'tidal_volume_set', 'resp_rate_set',
        'pressure_control_set', 'pressure_support_set', 'flow_rate_set',
        'peak_inspiratory_pressure_set', 'inspiratory_time_set', 'peep_set',
        'tidal_volume_obs', 'resp_rate_obs', 'plateau_pressure_obs',
        'peak_inspiratory_pressure_obs', 'peep_obs', 'minute_vent_obs',
        'mean_airway_pressure_obs'
    ]

    # Keep only numeric columns that exist in the dataframe and have bins
    numeric_columns_present = [col for col in numeric_columns if col in resp_df.columns and col in available_measurements]

    if logger:
        logger.info(f"  Processing {len(numeric_columns_present)} numeric measurements...")

    # Transform from wide to long format for vectorized processing
    # Keep ID columns for later merging
    id_cols = ['hospitalization_id', 'recorded_dttm']
    categorical_cols = ['device_category', 'mode_category', 'tracheostomy']

    # Separate categorical columns for later
    categorical_data = resp_df[id_cols + [col for col in categorical_cols if col in resp_df.columns]].copy()

    # Melt numeric columns into long format
    resp_long = resp_df[id_cols + numeric_columns_present].melt(
        id_vars=id_cols,
        value_vars=numeric_columns_present,
        var_name='measurement',
        value_name='value'
    )

    # Remove rows with null values (can't be tokenized)
    initial_long_count = len(resp_long)
    resp_long = resp_long[resp_long['value'].notna()].copy()
    if logger:
        logger.info(f"  Removed {initial_long_count - len(resp_long):,} null values from {initial_long_count:,} total measurement records")

    # Use optimized interval-aware binning (10-100x faster than pandas apply)
    resp_with_tokens = bin_numeric_values_with_intervals_by_category(
        df=resp_long,
        value_col='value',
        category_col='measurement',
        bins_df=ranges_df,
        measurement_col='measurement',
        logger=logger
    )

    # Keep only unique rows (in case multiple bins matched, take first)
    resp_with_tokens = resp_with_tokens.drop_duplicates(
        subset=['hospitalization_id', 'recorded_dttm', 'measurement', 'value'],
        keep='first'
    )

    # Log tokenization results per measurement
    if logger:
        logger.info("")
        for measurement in sorted(resp_long['measurement'].unique()):
            measurement_df = resp_long[resp_long['measurement'] == measurement]
            tokenized_df = resp_with_tokens[resp_with_tokens['measurement'] == measurement]
            total = len(measurement_df)
            tokenized = len(tokenized_df)
            pct = (tokenized / total * 100) if total > 0 else 0
            logger.info(f"  {measurement}: {tokenized:,}/{total:,} ({pct:.1f}%) tokenized")

    # Transform back to wide format with tokenized values
    # Pivot tokens back to wide format
    resp_tokens_wide = resp_with_tokens.pivot_table(
        index=['hospitalization_id', 'recorded_dttm'],
        columns='measurement',
        values='token',
        aggfunc='first'  # Take first token if duplicates exist
    ).reset_index()

    # Merge tokenized data back with categorical columns
    resp_df = categorical_data.merge(
        resp_tokens_wide,
        on=['hospitalization_id', 'recorded_dttm'],
        how='left'
    )

    # Count all tokens
    token_counts = {}
    for col in numeric_columns_present:
        if col in resp_df.columns:
            col_token_counts = resp_df[col].value_counts()
            for token, count in col_token_counts.items():
                if pd.notna(token):
                    token_counts[token] = token_counts.get(token, 0) + count

    # Load respiratory support config for tokenization
    resp_config = token_config['tables']['respiratory_support']['tokenization']

    # Handle tracheostomy column using config mapping
    if logger:
        logger.info("")
        logger.info("Tokenizing tracheostomy column...")

    if 'tracheostomy' in resp_df.columns:
        trach_config = resp_config['tracheostomy']
        prefix = trach_config.get('prefix', '')
        mapping = trach_config.get('mapping', {})

        def tokenize_tracheostomy(value):
            if pd.isna(value):
                return None
            # Convert to int for mapping lookup
            value_int = int(value)
            token = mapping.get(value_int)
            if token:
                return f"{prefix}{token}" if prefix else token
            else:
                return None  # For value 0 or unmapped values

        resp_df['tracheostomy'] = resp_df['tracheostomy'].apply(tokenize_tracheostomy)

        # Count tracheostomy tokens
        trach_counts = resp_df['tracheostomy'].value_counts()
        for token, count in trach_counts.items():
            if pd.notna(token):
                token_counts[token] = token_counts.get(token, 0) + count

        if logger:
            present_count = trach_counts.get('tracheostomy_present', 0)
            logger.info(f"  ✓ Tracheostomy present: {present_count:,}")

    # Tokenize categorical columns (device_category, mode_category)
    if logger:
        logger.info("")
        logger.info("Tokenizing categorical columns...")

    # Device category
    if 'device_category' in resp_df.columns:
        device_config = resp_config['device_category']
        prefix = device_config.get('prefix', '')
        mapping = device_config.get('mapping', {})
        map_unmapped = device_config.get('map_unmapped_to_other', False)

        def tokenize_device(value):
            if pd.isna(value):
                return None
            token = mapping.get(value)
            if token:
                return f"{prefix}{token}"
            elif map_unmapped:
                return f"{prefix}other"
            else:
                return None

        resp_df['device_category'] = resp_df['device_category'].apply(tokenize_device)

        # Count device tokens
        device_counts = resp_df['device_category'].value_counts()
        for token, count in device_counts.items():
            if pd.notna(token):
                token_counts[token] = token_counts.get(token, 0) + count

        if logger:
            logger.info(f"  ✓ Device category: {len(device_counts)} unique tokens")

    # Mode category
    if 'mode_category' in resp_df.columns:
        mode_config = resp_config['mode_category']
        prefix = mode_config.get('prefix', '')
        mapping = mode_config.get('mapping', {})
        map_unmapped = mode_config.get('map_unmapped_to_other', False)

        def tokenize_mode(value):
            if pd.isna(value):
                return None
            token = mapping.get(value)
            if token:
                return f"{prefix}{token}"
            elif map_unmapped:
                return f"{prefix}other"
            else:
                return None

        resp_df['mode_category'] = resp_df['mode_category'].apply(tokenize_mode)

        # Count mode tokens
        mode_counts = resp_df['mode_category'].value_counts()
        for token, count in mode_counts.items():
            if pd.notna(token):
                token_counts[token] = token_counts.get(token, 0) + count

        if logger:
            logger.info(f"  ✓ Mode category: {len(mode_counts)} unique tokens")

    if logger:
        logger.info("")
        logger.info(f"Respiratory support token counts:")
        logger.info(f"  Total unique tokens: {len(token_counts)}")
        logger.info(f"  Total token occurrences: {sum(token_counts.values()):,}")
        logger.info("=" * 60)

    # Return final DataFrame and token counts
    return resp_df, token_counts
