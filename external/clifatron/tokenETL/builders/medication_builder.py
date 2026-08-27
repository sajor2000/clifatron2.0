"""
Medication Builder Module

Loads medication_admin_continuous data and converts units to standardized formats.
Uses clifpy's unit_converter utility for dose unit standardization.
Tokenizes converted medication doses using interval-aware numeric bins from
config/critical_illness_tokenization_final_with_intervals.csv.
"""

import os
import pandas as pd
import logging
from typing import Dict, Any, Tuple
from clifpy.tables import MedicationAdminContinuous, Vitals
from clifpy.utils.unit_converter import convert_dose_units_by_med_category
from utils.polars_utils import strip_all_datetime_timezones, read_numeric_ranges_polars, bin_numeric_values_with_intervals_by_category


def build_medication_data(
    cohort_df: pd.DataFrame,
    config_path: str,
    token_config: Dict[str, Any],
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load medication_admin_continuous data, convert units, tokenize doses, and return cleaned DataFrame.

    Process:
    1. Get cohort hospitalization_ids
    2. Load medication_admin_continuous filtered to those IDs
    3. Clean data: remove rows with null/NaN med_dose or med_dose_unit
    4. Load patient weights from vitals (needed for weight-based conversions)
    5. Convert medication doses to standardized units using clifpy's unit_converter
    6. Filter to only successful conversions (_convert_status == 'success')
    7. Load medication bins from config/critical_illness_tokenization_final_with_intervals.csv
    8. Tokenize med_dose_converted values using interval-aware binning (supports [, (, ], ) notation)
    9. Count token occurrences
    10. Return cleaned DataFrame with medication_dose_token column and token counts

    Args:
        cohort_df: Cohort DataFrame with hospitalization_id column
        config_path: Path to clif_config.json
        token_config: Loaded token configuration
        logger: Logger instance

    Returns:
        Tuple of (DataFrame with medication_dose_token column, token_counts dict)
    """
    if logger:
        logger.info("=" * 60)
        logger.info("PHASE 6: MEDICATION ADMIN CONTINUOUS")
        logger.info("=" * 60)

    # Get cohort hospitalization_ids
    cohort_hosp_ids = cohort_df['hospitalization_id'].unique()

    if logger:
        logger.info(f"Cohort hospitalizations: {len(cohort_hosp_ids):,}")
        logger.info("Loading medication_admin_continuous for cohort hospitalizations...")

    # Load medication_admin_continuous filtered to cohort
    try:
        med_table = MedicationAdminContinuous.from_file(
            config_path=config_path,
            columns=['hospitalization_id', 'admin_dttm', 'med_category', 'med_dose', 'med_dose_unit'],
            filters={'hospitalization_id': list(cohort_hosp_ids)}
        )
        med_df = med_table.df.copy()

        # Strip timezone from ALL datetime columns using utility function
        med_df = strip_all_datetime_timezones(med_df)
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠ Failed to load medication_admin_continuous: {e}")
            logger.info("  Skipping medication processing")
            logger.info("=" * 60)
        # Return empty DataFrame and counts
        return pd.DataFrame(), {}

    if logger:
        logger.info(f"  ✓ Loaded {len(med_df):,} medication records")

    # Clean data: remove null/NaN values
    if logger:
        logger.info("")
        logger.info("Cleaning medication data...")

    initial_count = len(med_df)

    # Remove null med_dose
    med_df = med_df[med_df['med_dose'].notna()]

    # Remove null med_dose_unit
    med_df = med_df[med_df['med_dose_unit'].notna()]

    # Remove 'nan' string values
    med_df = med_df[~med_df['med_dose_unit'].astype(str).str.lower().isin(['nan', 'none', ''])]

    final_count = len(med_df)

    if logger:
        removed_count = initial_count - final_count
        logger.info(f"  ✓ Cleaned: {initial_count:,} → {final_count:,} records ({removed_count:,} removed)")

    if final_count == 0:
        if logger:
            logger.info("  ⚠ No medication records remain after cleaning")
            logger.info("=" * 60)
        return pd.DataFrame(), {}

    # Get preferred units from config (needed for filtering)
    med_config = token_config.get('tables', {}).get('medication_admin_continuous', {})
    unit_config = med_config.get('unit_conversion', {})
    preferred_units = unit_config.get('preferred_units', {})
    override = unit_config.get('override', True)

    # Filter to only configured medications
    if logger:
        logger.info("")
        logger.info("Filtering to configured medications only...")

    pre_filter_count = len(med_df)
    configured_meds = set(preferred_units.keys())
    med_df = med_df[med_df['med_category'].isin(configured_meds)].copy()
    post_filter_count = len(med_df)
    filtered_out = pre_filter_count - post_filter_count

    if logger:
        logger.info(f"  ✓ Kept {len(configured_meds)} configured medications: {post_filter_count:,} records")
        if filtered_out > 0:
            pct_filtered = (filtered_out / pre_filter_count * 100) if pre_filter_count > 0 else 0
            logger.info(f"  ✓ Filtered out unconfigured medications: {filtered_out:,} records ({pct_filtered:.1f}%)")

    if post_filter_count == 0:
        if logger:
            logger.info("  ⚠ No medication records remain after filtering")
            logger.info("=" * 60)
        return pd.DataFrame(), {}

    # Load patient weights from vitals
    if logger:
        logger.info("")
        logger.info("Loading patient weights from vitals...")

    try:
        vitals_table = Vitals.from_file(
            config_path=config_path,
            columns=['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
            filters={'hospitalization_id': list(cohort_hosp_ids), 'vital_category': ['weight_kg']}
        )
        vitals_df = vitals_table.df.copy()

        # Strip timezone from ALL datetime columns using utility function
        vitals_df = strip_all_datetime_timezones(vitals_df)

        if logger:
            logger.info(f"  ✓ Loaded {len(vitals_df):,} weight measurements")
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠ Failed to load vitals: {e}")
            logger.info("  Continuing without weight data (weight-based conversions may fail)")
        vitals_df = None

    if logger:
        logger.info("")
        logger.info(f"Converting medication units using {len(preferred_units)} preferred unit mappings...")

    # Convert medication units
    try:
        med_df_converted, conversion_counts = convert_dose_units_by_med_category(
            med_df=med_df,
            vitals_df=vitals_df,
            preferred_units=preferred_units,
            show_intermediate=False,
            override=override
        )

        if logger:
            logger.info(f"  ✓ Unit conversion complete")
    except Exception as e:
        if logger:
            logger.error(f"  ✗ Unit conversion failed: {e}")
            logger.info("=" * 60)
        return pd.DataFrame(), {}

    # Filter to only successful conversions
    if logger:
        logger.info("")
        logger.info("Filtering to successful conversions...")

    pre_filter_count = len(med_df_converted)
    med_df_converted = med_df_converted[med_df_converted['_convert_status'] == 'success'].copy()
    post_filter_count = len(med_df_converted)

    if logger:
        success_rate = (post_filter_count / pre_filter_count * 100) if pre_filter_count > 0 else 0
        logger.info(f"  ✓ Successful conversions: {post_filter_count:,} / {pre_filter_count:,} ({success_rate:.1f}%)")

    # Log conversion summary by status
    if logger:
        logger.info("")
        logger.info("Conversion status summary:")
        status_counts = conversion_counts.groupby('_convert_status')['count'].sum().sort_values(ascending=False)
        for status, count in status_counts.items():
            pct = (count / conversion_counts['count'].sum() * 100) if conversion_counts['count'].sum() > 0 else 0
            logger.info(f"  - {status}: {count:,} ({pct:.1f}%)")

    # Check for medications configured but not in data
    if logger:
        logger.info("")
        logger.info("Checking for configured medications not found in data...")

        configured_meds = set(preferred_units.keys())
        actual_meds = set(conversion_counts['med_category'].unique())
        missing_meds = configured_meds - actual_meds

        if missing_meds:
            logger.info(f"  ⚠ {len(missing_meds)} medication(s) configured but NOT found in data:")
            for med in sorted(missing_meds):
                logger.info(f"    - {med}")
        else:
            logger.info(f"  ✓ All {len(configured_meds)} configured medications found in data")

    # Check if converted units match configured preferred units
    if logger:
        logger.info("")
        logger.info("Checking if converted units match configured preferred units...")

        # Get successful conversions only
        success_counts = conversion_counts[conversion_counts['_convert_status'] == 'success'].copy()

        # For each medication, check if converted unit matches preferred unit
        mismatch_summary = []

        for med_category in sorted(actual_meds):
            med_data = success_counts[success_counts['med_category'] == med_category]

            if len(med_data) == 0:
                continue

            preferred_unit = preferred_units.get(med_category)

            # Check for mismatches where converted unit != preferred unit
            mismatches = med_data[med_data['med_dose_unit_converted'] != preferred_unit]

            if len(mismatches) > 0:
                total_med_records = med_data['count'].sum()
                mismatch_records = mismatches['count'].sum()
                pct = (mismatch_records / total_med_records * 100) if total_med_records > 0 else 0

                # Get breakdown of what units they converted to
                unit_breakdown = mismatches.groupby('med_dose_unit_converted')['count'].sum().sort_values(ascending=False)

                mismatch_summary.append({
                    'med_category': med_category,
                    'preferred_unit': preferred_unit,
                    'mismatch_count': mismatch_records,
                    'total_count': total_med_records,
                    'pct': pct,
                    'unit_breakdown': unit_breakdown
                })

        if mismatch_summary:
            logger.info(f"  ⚠ {len(mismatch_summary)} medication(s) have records that didn't convert to preferred unit:")
            for item in mismatch_summary:
                logger.info(f"    - {item['med_category']} (preferred: {item['preferred_unit']}): "
                           f"{item['mismatch_count']:,} / {item['total_count']:,} records ({item['pct']:.1f}%) failed")
                # Show top 3 actual units
                for unit, count in list(item['unit_breakdown'].items())[:3]:
                    logger.info(f"      → Actual: {unit} ({count:,} records)")
        else:
            logger.info(f"  ✓ All medications: successfully converted to preferred units")

    # ============================================================================
    # MEDICATION DOSE TOKENIZATION
    # ============================================================================

    if logger:
        logger.info("")
        logger.info("Loading medication bins from critical_illness_tokenization_final_with_intervals.csv...")

    # Get script directory for loading interval CSV
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ranges_path = os.path.join(script_dir, 'config', 'critical_illness_tokenization_final_with_intervals.csv')

    # Load medication bins
    med_ranges_df = read_numeric_ranges_polars(ranges_path, 'medications')

    if logger:
        logger.info(f"  ✓ Loaded {len(med_ranges_df)} medication bins")

    # Get unique medications available in bins
    available_meds = med_ranges_df['measurement'].unique()

    if logger:
        logger.info(f"  ✓ Medications with bins: {len(available_meds)}")

    # Create measurement column in med_df_converted: med_category + unit (with / replaced by _)
    med_df_converted['_measurement'] = (
        med_df_converted['med_category'] + '_' +
        med_df_converted['med_dose_unit_converted'].str.replace('/', '_')
    )

    if logger:
        logger.info("")
        logger.info("Tokenizing medication doses using interval-aware binning (category-based processing)...")

    # Filter to only measurements with bins
    initial_count = len(med_df_converted)
    med_df_converted = med_df_converted[med_df_converted['_measurement'].isin(available_meds)].copy()
    removed = initial_count - len(med_df_converted)

    if logger:
        pct_kept = (len(med_df_converted) / initial_count * 100) if initial_count > 0 else 0
        logger.info(f"  ✓ Kept {len(med_df_converted):,} records ({pct_kept:.1f}%)")
        if removed > 0:
            logger.info(f"  ✓ Removed {removed:,} records (measurements not in bins)")

    # Use interval-aware category-based binning with streaming to prevent memory exhaustion
    # Processes each medication separately with early deduplication in polars
    med_with_tokens = bin_numeric_values_with_intervals_by_category(
        df=med_df_converted,
        value_col='med_dose_converted',
        category_col='_measurement',
        bins_df=med_ranges_df,
        measurement_col='measurement',
        logger=logger
    )

    # Extract the token column (deduplication now handled within polars streaming pipeline)
    med_df_converted['medication_dose_token'] = med_with_tokens.set_index(
        ['hospitalization_id', 'admin_dttm', 'med_category', 'med_dose_converted']
    )['token'].reindex(
        med_df_converted.set_index(['hospitalization_id', 'admin_dttm', 'med_category', 'med_dose_converted']).index
    ).values

    # Count all medication tokens
    token_counts = {}
    medication_token_counts = med_df_converted['medication_dose_token'].value_counts()
    for token, count in medication_token_counts.items():
        if pd.notna(token):
            token_counts[token] = count

    tokenized_count = med_df_converted['medication_dose_token'].notna().sum()
    total_count = len(med_df_converted)

    if logger:
        logger.info("")
        logger.info(f"Medication dose tokenization complete:")
        logger.info(f"  ✓ Successfully tokenized: {tokenized_count:,} / {total_count:,} ({tokenized_count/total_count*100:.1f}%)")
        logger.info(f"  ✓ Unique tokens: {len(token_counts)}")
        logger.info(f"  ✓ Total token occurrences: {sum(token_counts.values()):,}")

    # Drop temporary _measurement column
    med_df_converted = med_df_converted.drop(columns=['_measurement'])

    if logger:
        logger.info("=" * 60)

    # Return cleaned, converted, and tokenized medication data
    return med_df_converted, token_counts
