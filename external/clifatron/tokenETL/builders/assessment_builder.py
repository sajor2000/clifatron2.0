"""
Assessment Builder Module

Loads patient assessments for cohort and creates assessment tokens.
Similar pattern to elixhauser_builder.py.
"""

import pandas as pd
import logging
from typing import Dict, Any, Tuple
from clifpy.tables import PatientAssessments
from utils.polars_utils import strip_all_datetime_timezones


def build_assessment_tokens(
    cohort_df: pd.DataFrame,
    config_path: str,
    token_config: Dict[str, Any],
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load patient assessments for cohort and create assessment tokens.

    Process:
    1. Get cohort hospitalization_ids
    2. Load patient_assessments filtered to those IDs
    3. Lowercase assessment_category
    4. Filter to configured assessments
    5. Map numeric values using explicit config mappings
    6. Log matched/unmapped values with counts
    7. Create assessment_token column
    8. Filter out unmapped values
    9. Count token occurrences

    Args:
        cohort_df: Cohort DataFrame with hospitalization_id column
        config_path: Path to clif_config.json
        token_config: Loaded token configuration
        logger: Logger instance

    Returns:
        Tuple of (DataFrame with hospitalization_id, recorded_dttm, assessment_token, token_counts dict)
    """
    if logger:
        logger.info("=" * 60)
        logger.info("PHASE 4: PATIENT ASSESSMENTS")
        logger.info("=" * 60)

    # Get cohort hospitalization_ids
    cohort_hosp_ids = cohort_df['hospitalization_id'].unique()

    if logger:
        logger.info(f"Cohort hospitalizations: {len(cohort_hosp_ids):,}")
        logger.info("Loading patient_assessments for cohort hospitalizations...")

    # Load patient_assessments filtered to cohort
    pa_table = PatientAssessments.from_file(
        config_path=config_path,
        columns=['hospitalization_id', 'recorded_dttm', 'assessment_category', 'numerical_value'],
        filters={'hospitalization_id': list(cohort_hosp_ids)}
    )
    pa_df = pa_table.df.copy()

    # Strip timezone from ALL datetime columns using utility function
    pa_df = strip_all_datetime_timezones(pa_df)

    if logger:
        logger.info(f"  ✓ Loaded {len(pa_df):,} assessment records")

    # Lowercase assessment_category
    if logger:
        logger.info("")
        logger.info("Lowercasing assessment_category column...")
        # Count uppercase values for logging
        uppercase_count = pa_df['assessment_category'].str.match(r'^[A-Z]+$').sum()

    pa_df['assessment_category'] = pa_df['assessment_category'].str.lower()

    if logger and uppercase_count > 0:
        logger.info(f"  ✓ Lowercased {uppercase_count:,} values (e.g., RASS → rass)")

    # Get configured assessments
    assess_config = token_config['tables']['patient_assessments']['tokenization']
    prefix = assess_config.get('prefix', 'assessment_')
    assessments = assess_config['assessments']
    categories_to_keep = list(assessments.keys())

    if logger:
        logger.info("")
        logger.info(f"Filtering to configured assessments: {categories_to_keep}")

    initial_count = len(pa_df)
    pa_df = pa_df[pa_df['assessment_category'].isin(categories_to_keep)].copy()
    removed = initial_count - len(pa_df)

    if logger:
        pct_kept = (len(pa_df) / initial_count * 100) if initial_count > 0 else 0
        logger.info(f"  ✓ Kept {len(pa_df):,} records ({pct_kept:.1f}%)")
        if removed > 0:
            logger.info(f"  ✓ Removed {removed:,} records")

    # Tokenize each assessment category
    if logger:
        logger.info("")

    for category, config in assessments.items():
        if logger:
            logger.info(f"Tokenizing assessment_category: {category}...")

        # Get subset for this category
        category_df = pa_df[pa_df['assessment_category'] == category].copy()

        if len(category_df) == 0:
            if logger:
                logger.info(f"  ⚠ No records found for {category}")
                logger.info("")
            continue

        # Get mapping and counts
        mapping = config.get('mapping', {})
        value_counts = category_df['numerical_value'].value_counts().sort_index()

        # Track matched and unmapped
        matched = {}
        unmapped = []

        for value in value_counts.index:
            if pd.isna(value):
                continue

            # Try both float and int lookup
            token = mapping.get(value)
            if not token and value == int(value):
                token = mapping.get(int(value))

            if token:
                matched[value] = (token, value_counts[value])
            else:
                unmapped.append((value, value_counts[value]))

        # Log results
        if logger:
            logger.info(f"  ✓ Matched {len(matched)} unique values:")
            for value, (token, count) in sorted(matched.items()):
                logger.info(f"    - {value} ({count:,}) → {prefix}{token}")

            if unmapped:
                logger.info(f"  ✗ Unmapped: {len(unmapped)} values")
                for value, count in unmapped:
                    logger.info(f"    - {value} ({count:,}) [NO MATCH - FILTERED OUT]")
            else:
                logger.info(f"  ✓ Unmapped: 0 values")
            logger.info("")

    # Create assessment_token column
    if logger:
        logger.info("Creating assessment_token column...")

    def create_assessment_token(row):
        category = row['assessment_category']
        value = row['numerical_value']

        if pd.isna(value):
            return None

        # Get mapping for this category
        category_config = assessments.get(category, {})
        mapping = category_config.get('mapping', {})

        # Try lookup (handle both float and int)
        token = mapping.get(value)
        if not token and value == int(value):
            token = mapping.get(int(value))

        if token:
            return f"{prefix}{token}"
        else:
            return None

    pa_df['assessment_token'] = pa_df.apply(create_assessment_token, axis=1)

    # Filter out unmapped values
    initial_count = len(pa_df)
    pa_df = pa_df[pa_df['assessment_token'].notna()].copy()
    filtered_out = initial_count - len(pa_df)

    if logger:
        logger.info(f"  ✓ Total records: {initial_count:,}")
        logger.info(f"  ✓ Successfully tokenized: {len(pa_df):,} ({len(pa_df)/initial_count*100:.2f}%)")
        if filtered_out > 0:
            logger.info(f"  ✓ Filtered out (unmapped): {filtered_out:,} ({filtered_out/initial_count*100:.2f}%)")

    # Count assessment tokens
    token_counts = {}
    assessment_token_counts = pa_df['assessment_token'].value_counts()
    for token, count in assessment_token_counts.items():
        if pd.notna(token):
            token_counts[token] = count

    if logger:
        logger.info("")
        logger.info(f"Assessment token counts:")
        logger.info(f"  Total unique tokens: {len(token_counts)}")
        logger.info(f"  Total token occurrences: {sum(token_counts.values()):,}")
        logger.info("=" * 60)

    # Return final columns and token counts
    return pa_df[['hospitalization_id', 'recorded_dttm', 'assessment_token']], token_counts
