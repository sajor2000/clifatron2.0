"""
Cohort Builder Module

Contains all logic for creating the cohort table from loaded CLIF tables.
Includes filtering, merging, and calculating derived fields.
"""

import os
import pandas as pd
import logging
from typing import Dict, Tuple


def merge_patient_hospitalization(
    patient_df: pd.DataFrame,
    hosp_df: pd.DataFrame,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Merge patient and hospitalization tables.

    Args:
        patient_df: Patient DataFrame
        hosp_df: Hospitalization DataFrame
        logger: Logger instance

    Returns:
        Merged DataFrame
    """
    if logger:
        logger.info("Merging patient and hospitalization tables...")

    merged = pd.merge(
        hosp_df,
        patient_df,
        on='patient_id',
        how='left'
    )

    if logger:
        logger.info(f"  ✓ Merged: {len(merged):,} rows")

    return merged


def filter_null_dates(df: pd.DataFrame, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Filter out hospitalizations with null admission_dttm or discharge_dttm.

    Args:
        df: Input DataFrame with admission_dttm and discharge_dttm columns
        logger: Logger instance

    Returns:
        Filtered DataFrame without null dates
    """
    if logger:
        logger.info("Filtering out null admission/discharge dates...")

    initial_count = len(df)

    # Remove rows with null admission_dttm or discharge_dttm
    df_filtered = df[
        df['admission_dttm'].notna() &
        df['discharge_dttm'].notna()
    ].copy()

    if logger:
        removed = initial_count - len(df_filtered)
        logger.info(f"  ✓ Removed {removed:,} rows (null admission or discharge dates)")
        logger.info(f"  ✓ Remaining: {len(df_filtered):,} rows")

    return df_filtered


def filter_time_period(
    df: pd.DataFrame,
    start_date: str = '2018-01-01',
    end_date: str = '2024-12-31',
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Filter for hospitalizations within a specific time period.

    Both admission_dttm and discharge_dttm must be within the specified date range.

    Args:
        df: Input DataFrame with admission_dttm and discharge_dttm
        start_date: Start date (inclusive) as string 'YYYY-MM-DD'
        end_date: End date (inclusive) as string 'YYYY-MM-DD'
        logger: Logger instance

    Returns:
        Filtered DataFrame containing only hospitalizations within the time period
    """
    if logger:
        logger.info(f"Filtering for time period {start_date} to {end_date}...")

    initial_count = len(df)

    # Convert date strings to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Ensure datetime columns
    df['admission_dttm'] = pd.to_datetime(df['admission_dttm'])
    df['discharge_dttm'] = pd.to_datetime(df['discharge_dttm'])

    # Filter: both admission and discharge must be within the time period
    df_filtered = df[
        (df['admission_dttm'] >= start_dt) &
        (df['admission_dttm'] <= end_dt) &
        (df['discharge_dttm'] >= start_dt) &
        (df['discharge_dttm'] <= end_dt)
    ].copy()

    if logger:
        removed = initial_count - len(df_filtered)
        logger.info(f"  ✓ Removed {removed:,} rows (outside time period)")
        logger.info(f"  ✓ Remaining: {len(df_filtered):,} rows")

    return df_filtered


def filter_adults(df: pd.DataFrame, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Filter for adults only (age >= 18).

    Args:
        df: Input DataFrame
        logger: Logger instance

    Returns:
        Filtered DataFrame
    """
    if logger:
        logger.info("Filtering for adults (age >= 18)...")

    initial_count = len(df)
    df_filtered = df[df['age_at_admission'] >= 18].copy()

    if logger:
        removed = initial_count - len(df_filtered)
        logger.info(f"  ✓ Removed {removed:,} rows (age < 18)")
        logger.info(f"  ✓ Remaining: {len(df_filtered):,} rows")

    return df_filtered


def calculate_los(df: pd.DataFrame, logger: logging.Logger = None) -> pd.DataFrame:
    """
    Calculate hospitalization length of stay and filter for LOS > 0.

    Args:
        df: Input DataFrame with admission_dttm and discharge_dttm
        logger: Logger instance

    Returns:
        DataFrame with hospitalization_los column, filtered for LOS > 0
    """
    if logger:
        logger.info("Calculating hospitalization length of stay...")

    # Ensure datetime columns
    df['admission_dttm'] = pd.to_datetime(df['admission_dttm'])
    df['discharge_dttm'] = pd.to_datetime(df['discharge_dttm'])

    # Calculate LOS in days
    df['hospitalization_los'] = (df['discharge_dttm'] - df['admission_dttm']).dt.total_seconds() / (24 * 3600)

    initial_count = len(df)
    df_filtered = df[df['hospitalization_los'] > 0].copy()

    if logger:
        removed = initial_count - len(df_filtered)
        logger.info(f"  ✓ Removed {removed:,} rows (LOS <= 0)")
        logger.info(f"  ✓ Remaining: {len(df_filtered):,} rows")

    return df_filtered


def filter_icu_only(
    cohort_df: pd.DataFrame,
    adt_df: pd.DataFrame,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Keep only hospitalizations with at least one ICU stay.

    Filters cohort to include only hospitalizations that have at least one
    ADT event with location_category = 'icu' (case-insensitive).

    Args:
        cohort_df: Cohort DataFrame with hospitalization_id
        adt_df: ADT DataFrame with hospitalization_id and location_category
        logger: Logger instance

    Returns:
        Filtered cohort DataFrame containing only ICU hospitalizations
    """
    if logger:
        logger.info("Filtering to ICU-only hospitalizations...")

    # Lowercase location_category for case-insensitive comparison
    adt_df['location_category_lower'] = adt_df['location_category'].str.lower()

    # Find all hospitalizations with at least one ICU stay
    icu_hosp_ids = adt_df[adt_df['location_category_lower'] == 'icu']['hospitalization_id'].unique()

    # Filter cohort to only ICU hospitalizations
    initial_count = len(cohort_df)
    filtered_cohort = cohort_df[cohort_df['hospitalization_id'].isin(icu_hosp_ids)].copy()

    # Clean up temporary column
    if 'location_category_lower' in adt_df.columns:
        adt_df.drop(columns=['location_category_lower'], inplace=True)

    if logger:
        kept = len(filtered_cohort)
        removed = initial_count - kept
        logger.info(f"  ✓ Found {kept:,} hospitalizations with at least one ICU stay")
        logger.info(f"  ✓ Removed {removed:,} non-ICU hospitalizations")
        logger.info(f"  ✓ Remaining: {len(filtered_cohort):,} rows")

    return filtered_cohort


def calculate_icu_stays(
    cohort_df: pd.DataFrame,
    adt_df: pd.DataFrame,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Calculate ICU stay timing metrics for each hospitalization.

    Groups consecutive ICU ADT events into discrete ICU stays and calculates:
    - first_icu_start_time: Time of first ICU admission
    - first_icu_end_time: Time of first ICU discharge
    - first_icu_24hr_completion_time: First ICU admission + 24hrs (NULL if stay < 24hrs)
    - second_icu_start_time: Time of second ICU admission (readmissions only, not transfers)

    An ICU readmission is defined as: ICU → non-ICU location → ICU
    An ICU-to-ICU transfer is NOT counted as a readmission.

    Args:
        cohort_df: Cohort DataFrame with hospitalization_id, discharge_dttm
        adt_df: ADT DataFrame with hospitalization_id, location_category, in_dttm
        logger: Logger instance

    Returns:
        Cohort DataFrame with added ICU timing columns
    """
    if logger:
        logger.info("Calculating ICU stay timing metrics...")

    # Get cohort hospitalization IDs
    cohort_hosp_ids = cohort_df['hospitalization_id'].unique()

    # Filter ADT to cohort hospitalizations and sort
    adt_cohort = adt_df[adt_df['hospitalization_id'].isin(cohort_hosp_ids)].copy()
    adt_cohort = adt_cohort.sort_values(['hospitalization_id', 'in_dttm']).reset_index(drop=True)

    # Add lowercase location category for case-insensitive comparison
    adt_cohort['is_icu'] = adt_cohort['location_category'].str.lower() == 'icu'

    # Calculate the end time for each location (next location's in_dttm)
    adt_cohort['next_in_dttm'] = adt_cohort.groupby('hospitalization_id')['in_dttm'].shift(-1)

    # For the last location, use discharge_dttm from cohort
    discharge_map = cohort_df.set_index('hospitalization_id')['discharge_dttm'].to_dict()
    adt_cohort['discharge_dttm'] = adt_cohort['hospitalization_id'].map(discharge_map)
    adt_cohort['out_dttm'] = adt_cohort['next_in_dttm'].fillna(adt_cohort['discharge_dttm'])

    # Group consecutive ICU events into discrete ICU stays
    # A new ICU stay starts when:
    # 1. Previous location was NOT ICU (readmission), or
    # 2. It's the first event for this hospitalization
    adt_cohort['prev_is_icu'] = adt_cohort.groupby('hospitalization_id')['is_icu'].shift(1)
    # Fill NaN with False for first events in each hospitalization
    adt_cohort['prev_is_icu'] = adt_cohort['prev_is_icu'].fillna(False)
    adt_cohort['is_new_icu_stay'] = (adt_cohort['is_icu']) & (~adt_cohort['prev_is_icu'])

    # Number ICU stays within each hospitalization (cumulative count of new stays)
    adt_cohort['icu_stay_number'] = adt_cohort.groupby('hospitalization_id')['is_new_icu_stay'].cumsum()
    # Set to 0 for non-ICU events
    adt_cohort.loc[~adt_cohort['is_icu'], 'icu_stay_number'] = 0

    # Filter to ICU events only for stay calculations
    icu_events = adt_cohort[adt_cohort['is_icu']].copy()

    # Calculate metrics per ICU stay
    icu_stays = icu_events.groupby(['hospitalization_id', 'icu_stay_number']).agg({
        'in_dttm': 'min',  # Start of ICU stay
        'out_dttm': 'max'  # End of ICU stay
    }).reset_index()

    icu_stays.rename(columns={
        'in_dttm': 'icu_start_time',
        'out_dttm': 'icu_end_time'
    }, inplace=True)

    # Calculate ICU stay duration
    icu_stays['icu_los_hours'] = (icu_stays['icu_end_time'] - icu_stays['icu_start_time']).dt.total_seconds() / 3600

    # Extract first ICU stay
    first_icu = icu_stays[icu_stays['icu_stay_number'] == 1].copy()
    first_icu['first_icu_start_time'] = first_icu['icu_start_time']
    first_icu['first_icu_end_time'] = first_icu['icu_end_time']

    # Calculate 24hr completion time (only if first ICU stay >= 24 hours)
    first_icu['first_icu_24hr_completion_time'] = first_icu.apply(
        lambda row: row['icu_start_time'] + pd.Timedelta(hours=24) if row['icu_los_hours'] >= 24 else pd.NaT,
        axis=1
    )

    first_icu = first_icu[['hospitalization_id', 'first_icu_start_time', 'first_icu_end_time', 'first_icu_24hr_completion_time']]

    # Extract second ICU stay (readmissions only)
    second_icu = icu_stays[icu_stays['icu_stay_number'] == 2].copy()
    second_icu['second_icu_start_time'] = second_icu['icu_start_time']
    second_icu = second_icu[['hospitalization_id', 'second_icu_start_time']]

    # Merge ICU metrics back to cohort
    cohort_df = cohort_df.merge(first_icu, on='hospitalization_id', how='left')
    cohort_df = cohort_df.merge(second_icu, on='hospitalization_id', how='left')

    # Log statistics
    if logger:
        has_first_icu = cohort_df['first_icu_start_time'].notna().sum()
        has_24hr_completion = cohort_df['first_icu_24hr_completion_time'].notna().sum()
        has_second_icu = cohort_df['second_icu_start_time'].notna().sum()

        logger.info(f"  ✓ Hospitalizations with first ICU stay: {has_first_icu:,} / {len(cohort_df):,}")
        logger.info(f"  ✓ First ICU stays >= 24 hours: {has_24hr_completion:,} ({has_24hr_completion/has_first_icu*100:.1f}%)")
        logger.info(f"  ✓ Hospitalizations with ICU readmission (2nd ICU stay): {has_second_icu:,} ({has_second_icu/has_first_icu*100:.1f}%)")

    return cohort_df


def calculate_previous_hospitalization(
    df: pd.DataFrame,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Calculate previous_hospitalization_id for each patient.

    Sorts by patient_id and admission_dttm, then uses shift to get the previous
    hospitalization_id for the same patient.

    Args:
        df: Input DataFrame with patient_id, hospitalization_id, admission_dttm
        logger: Logger instance

    Returns:
        DataFrame with previous_hospitalization_id column
    """
    if logger:
        logger.info("Calculating previous hospitalization IDs...")

    # Sort by patient_id and admission_dttm
    df = df.sort_values(['patient_id', 'admission_dttm']).copy()

    # Calculate previous hospitalization_id per patient
    df['previous_hospitalization_id'] = df.groupby('patient_id')['hospitalization_id'].shift(1)

    # Count how many have a previous hospitalization
    has_previous = df['previous_hospitalization_id'].notna().sum()

    if logger:
        logger.info(f"  ✓ {has_previous:,} hospitalizations have a previous hospitalization")
        logger.info(f"  ✓ {len(df) - has_previous:,} are first-time hospitalizations")

    return df


def create_cohort(
    tables: Dict[str, pd.DataFrame],
    site: str,
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main cohort creation function.

    Creates cohort table by:
    1. Merging hospitalization + patient
    2. Filtering out null admission/discharge dates
    3. Filtering for time period (2018-2024) - skipped if site is "mimic"
    4. Filtering for adults (age >= 18)
    5. Calculating hospitalization LOS and filtering LOS > 0
    6. Filtering to ICU-only hospitalizations (at least one ICU stay)
    6.5. Calculating ICU stay timing metrics (first/second ICU stays, 24hr completion)
    7. Calculating previous_hospitalization_id

    Args:
        tables: Dictionary of loaded tables (patient, hospitalization, adt)
        site: Site identifier (time filter skipped if "mimic")
        logger: Logger instance

    Returns:
        Tuple of:
        - Final cohort DataFrame with columns:
          - hospitalization_id, patient_id, admission_dttm, discharge_dttm,
          - age_at_admission, discharge_category, sex_category, race_category,
          - ethnicity_category, hospitalization_los, previous_hospitalization_id,
          - first_icu_start_time, first_icu_end_time, first_icu_24hr_completion_time,
          - second_icu_start_time
        - Exclusion statistics dictionary with counts at each step
    """
    if logger:
        logger.info("=" * 60)
        logger.info("CREATING COHORT")
        logger.info("=" * 60)

    exclusion_stats = {}

    # Step 1: Merge patient and hospitalization
    cohort_df = merge_patient_hospitalization(
        patient_df=tables['patient'],
        hosp_df=tables['hospitalization'],
        logger=logger
    )
    exclusion_stats['initial'] = len(cohort_df)

    # Step 2: Filter out null dates
    cohort_df = filter_null_dates(cohort_df, logger)
    exclusion_stats['after_null_filter'] = len(cohort_df)
    exclusion_stats['excluded_null_dates'] = exclusion_stats['initial'] - exclusion_stats['after_null_filter']

    # Step 3: Filter for time period (2018-2024) - skip for MIMIC
    if site.lower() == 'mimic':
        if logger:
            logger.info("Skipping time period filter (site is MIMIC)...")
        exclusion_stats['after_time_filter'] = len(cohort_df)
        exclusion_stats['excluded_time_period'] = 0
        exclusion_stats['time_filter_applied'] = False
    else:
        cohort_df = filter_time_period(cohort_df, logger=logger)
        exclusion_stats['after_time_filter'] = len(cohort_df)
        exclusion_stats['excluded_time_period'] = exclusion_stats['after_null_filter'] - exclusion_stats['after_time_filter']
        exclusion_stats['time_filter_applied'] = True

    # Step 4: Filter adults
    cohort_df = filter_adults(cohort_df, logger)
    exclusion_stats['after_age_filter'] = len(cohort_df)
    exclusion_stats['excluded_age'] = exclusion_stats['after_time_filter'] - exclusion_stats['after_age_filter']

    # Step 5: Calculate LOS and filter
    cohort_df = calculate_los(cohort_df, logger)
    exclusion_stats['after_los_filter'] = len(cohort_df)
    exclusion_stats['excluded_los'] = exclusion_stats['after_age_filter'] - exclusion_stats['after_los_filter']

    # Step 6: Filter to ICU-only hospitalizations
    cohort_df = filter_icu_only(
        cohort_df=cohort_df,
        adt_df=tables['adt'],
        logger=logger
    )
    exclusion_stats['after_icu_filter'] = len(cohort_df)
    exclusion_stats['excluded_no_icu'] = exclusion_stats['after_los_filter'] - exclusion_stats['after_icu_filter']

    # Step 6.5: Calculate ICU stay timing metrics
    cohort_df = calculate_icu_stays(
        cohort_df=cohort_df,
        adt_df=tables['adt'],
        logger=logger
    )

    # Step 7: Calculate previous hospitalization
    cohort_df = calculate_previous_hospitalization(cohort_df, logger)
    exclusion_stats['final'] = len(cohort_df)

    # Reorder columns
    final_columns = [
        'hospitalization_id',
        'patient_id',
        'admission_dttm',
        'discharge_dttm',
        'age_at_admission',
        'discharge_category',
        'sex_category',
        'race_category',
        'ethnicity_category',
        'hospitalization_los',
        'previous_hospitalization_id',
        'first_icu_start_time',
        'first_icu_end_time',
        'first_icu_24hr_completion_time',
        'second_icu_start_time'
    ]

    cohort_df = cohort_df[final_columns]

    if logger:
        logger.info("=" * 60)
        logger.info("COHORT CREATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final cohort size: {len(cohort_df):,} rows")
        logger.info(f"Columns: {len(cohort_df.columns)}")
        memory_mb = cohort_df.memory_usage(deep=True).sum() / (1024**2)
        logger.info(f"Memory: {memory_mb:.2f} MB")
        logger.info("")
        logger.info("ICU Timing Summary:")
        logger.info(f"  - Hospitalizations with first ICU stay: {cohort_df['first_icu_start_time'].notna().sum():,}")
        logger.info(f"  - First ICU stays >= 24 hours: {cohort_df['first_icu_24hr_completion_time'].notna().sum():,}")
        logger.info(f"  - Hospitalizations with ICU readmission: {cohort_df['second_icu_start_time'].notna().sum():,}")
        logger.info("=" * 60)

    return cohort_df, exclusion_stats


def filter_adt_to_cohort(
    adt_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Filter ADT table to only include hospitalizations present in the cohort.

    Args:
        adt_df: Full ADT DataFrame
        cohort_df: Final cohort DataFrame with hospitalization_id
        logger: Logger instance

    Returns:
        Filtered ADT DataFrame containing only cohort hospitalizations
    """
    if logger:
        logger.info("Filtering ADT to cohort hospitalizations...")

    initial_adt_count = len(adt_df)
    cohort_hosp_ids = cohort_df['hospitalization_id'].unique()

    # Filter ADT to only cohort hospitalizations
    adt_filtered = adt_df[adt_df['hospitalization_id'].isin(cohort_hosp_ids)].copy()

    if logger:
        logger.info(f"  ✓ ADT filtered: {initial_adt_count:,} → {len(adt_filtered):,} rows")
        logger.info(f"  ✓ Removed {initial_adt_count - len(adt_filtered):,} ADT events")

    return adt_filtered


def create_consort_diagram(
    exclusion_stats: Dict,
    output_dir: str,
    logger: logging.Logger = None
):
    """
    Create and save a CONSORT diagram showing cohort exclusions.

    Args:
        exclusion_stats: Dictionary with exclusion counts at each step
        output_dir: Directory to save the diagram
        logger: Logger instance
    """
    if logger:
        logger.info("=" * 60)
        logger.info("CONSORT DIAGRAM")
        logger.info("=" * 60)

    # Build CONSORT diagram text
    diagram = []
    diagram.append("=" * 80)
    diagram.append("CONSORT DIAGRAM - Cohort Selection")
    diagram.append("=" * 80)
    diagram.append("")
    diagram.append(f"Initial hospitalizations (merged):           {exclusion_stats['initial']:>10,}")
    diagram.append("")
    diagram.append(f"  Excluded: Null admission/discharge dates   -{exclusion_stats['excluded_null_dates']:>10,}")
    diagram.append("  " + "-" * 76)
    diagram.append(f"After null date filter:                       {exclusion_stats['after_null_filter']:>10,}")
    diagram.append("")

    # Conditionally show time period filter
    if exclusion_stats.get('time_filter_applied', False):
        diagram.append(f"  Excluded: Outside 2018-2024 time period    -{exclusion_stats['excluded_time_period']:>10,}")
        diagram.append("  " + "-" * 76)
        diagram.append(f"After time period filter:                     {exclusion_stats['after_time_filter']:>10,}")
    else:
        diagram.append("  Time period filter: SKIPPED (MIMIC site)")
        diagram.append(f"After time period filter:                     {exclusion_stats['after_time_filter']:>10,}")

    diagram.append("")
    diagram.append(f"  Excluded: Age < 18                         -{exclusion_stats['excluded_age']:>10,}")
    diagram.append("  " + "-" * 76)
    diagram.append(f"After age filter:                             {exclusion_stats['after_age_filter']:>10,}")
    diagram.append("")
    diagram.append(f"  Excluded: LOS <= 0 days                    -{exclusion_stats['excluded_los']:>10,}")
    diagram.append("  " + "-" * 76)
    diagram.append(f"After LOS filter:                             {exclusion_stats['after_los_filter']:>10,}")
    diagram.append("")
    diagram.append(f"  Excluded: No ICU stay                      -{exclusion_stats['excluded_no_icu']:>10,}")
    diagram.append("  (no ADT event with location_category = 'icu')")
    diagram.append("  " + "-" * 76)
    diagram.append(f"FINAL COHORT (ICU-only):                      {exclusion_stats['final']:>10,}")
    diagram.append("")
    diagram.append("=" * 80)
    diagram.append("")
    diagram.append("EXCLUSION SUMMARY:")
    diagram.append(f"  Total excluded:  {exclusion_stats['initial'] - exclusion_stats['final']:,}")
    diagram.append(f"  Retention rate:  {(exclusion_stats['final'] / exclusion_stats['initial'] * 100):.1f}%")
    diagram.append("=" * 80)

    # Save to file
    consort_path = os.path.join(output_dir, 'consort_diagram.txt')
    with open(consort_path, 'w') as f:
        f.write('\n'.join(diagram))

    # Log the diagram
    if logger:
        for line in diagram:
            logger.info(line)
        logger.info(f"✓ CONSORT diagram saved to: {consort_path}")
