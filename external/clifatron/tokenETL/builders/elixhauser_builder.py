"""
Elixhauser Comorbidity Builder Module

Calculates Elixhauser comorbidities for previous hospitalizations and adds them to cohort.
Uses clifpy.utils.comorbidity for Elixhauser calculation.
"""

import os
import json
import pandas as pd
import logging
from typing import Dict, Any, Tuple
from clifpy.utils.comorbidity import calculate_elix
from clifpy.tables import HospitalDiagnosis


def calculate_previous_elix(
    cohort_df: pd.DataFrame,
    config_path: str,
    token_config: Dict[str, Any],
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Calculate Elixhauser comorbidities for previous hospitalizations.

    This function:
    1. Extracts unique previous_hospitalization_id values from cohort
    2. Loads hospital_diagnosis filtered to only those IDs
    3. Calculates Elixhauser comorbidity flags using clifpy
    4. Creates pipe-separated token string containing only comorbidities where flag=1
    5. Applies configured prefix to all comorbidity tokens
    6. Counts all tokens (comorbidity tokens and fill_no_history token)
    7. Fills empty strings with configured fill_no_history token
    8. Adds prev_hosp_comorbidities column to cohort (pipe-separated string)

    Token format: {prefix}comorbidity_name
    Example: "elix_congestive_heart_failure|elix_diabetes_uncomplicated"
    No history: "no_patient_history" (for patients with no previous hospitalization)

    Args:
        cohort_df: Cohort DataFrame with previous_hospitalization_id column
        config_path: Path to clif_config.json
        token_config: Loaded token configuration
        logger: Logger instance

    Returns:
        Tuple of (cohort_df with prev_hosp_comorbidities column, token_counts dict)
    """
    if logger:
        logger.info("=" * 60)
        logger.info("PHASE 2: PREVIOUS HOSPITALIZATION ELIXHAUSER")
        logger.info("=" * 60)

    # Get unique previous hospitalization IDs (excluding NaN)
    prev_hosp_ids = cohort_df['previous_hospitalization_id'].dropna().unique()

    if logger:
        total_rows = len(cohort_df)
        with_previous = cohort_df['previous_hospitalization_id'].notna().sum()
        logger.info(f"Total hospitalizations in cohort: {total_rows:,}")
        logger.info(f"Hospitalizations with previous admission: {with_previous:,}")
        logger.info(f"Unique previous hospitalization IDs: {len(prev_hosp_ids):,}")

    # If no previous hospitalizations, skip this step
    if len(prev_hosp_ids) == 0:
        if logger:
            logger.info("  No previous hospitalizations found - skipping Elixhauser calculation")
            logger.info("=" * 60)
        # Return empty token_counts
        return cohort_df, {}

    # Load hospital_diagnosis filtered to previous hospitalization IDs only
    if logger:
        logger.info("Loading hospital_diagnosis for previous hospitalizations...")

    hosp_dx = HospitalDiagnosis.from_file(
        config_path=config_path,
        filters={'hospitalization_id': list(prev_hosp_ids)}
    )

    if logger:
        logger.info(f"  ✓ Loaded {len(hosp_dx.df):,} diagnosis codes")

    # Calculate Elixhauser comorbidity scores
    if logger:
        logger.info("Calculating Elixhauser comorbidities...")

    elix_results = calculate_elix(hosp_dx.df, hierarchy=True)

    # Get configured comorbidities from token_config
    elix_config = token_config.get('tables', {}).get('hospital_diagnosis', {}).get('tokenization', {}).get('elixhauser', {})
    comorbidities = elix_config.get('comorbidities', [])
    prefix = elix_config.get('prefix', '')
    fill_no_history = elix_config.get('fill_no_history', 'no_patient_history')

    # Initialize token counts dictionary
    token_counts = {}

    if logger:
        logger.info(f"  ✓ Configured comorbidities from token_config ({len(comorbidities)}): {comorbidities}")

    # Select hospitalization_id + comorbidity columns
    # elix_results has hospitalization_id as a column (not index)
    columns_to_select = ['hospitalization_id'] + comorbidities
    elix_subset = elix_results[columns_to_select].copy()

    # Create token string: convert binary flags to pipe-separated string where value=1
    # Instead of 31 columns, create single column with pipe-separated token string
    if logger:
        logger.info("Creating comorbidity token strings...")

    def create_token_string(row):
        """Convert binary comorbidity flags to pipe-separated token string (only tokens where value=1)"""
        tokens = []
        for comorbidity in comorbidities:
            if row[comorbidity] == 1:
                tokens.append(f"{prefix}{comorbidity}")
        return '|'.join(tokens)

    elix_subset['prev_hosp_comorbidities'] = elix_subset.apply(create_token_string, axis=1)

    # Keep only hospitalization_id and token string column for merge
    elix_merge = elix_subset[['hospitalization_id', 'prev_hosp_comorbidities']].copy()

    # Convert hospitalization_id to string to match cohort dtype
    elix_merge['hospitalization_id'] = elix_merge['hospitalization_id'].astype(str)

    # Merge Elixhauser results to cohort on previous_hospitalization_id
    if logger:
        logger.info("Merging Elixhauser results to cohort...")

    cohort_df = cohort_df.merge(
        elix_merge,
        left_on='previous_hospitalization_id',
        right_on='hospitalization_id',
        how='left',
        suffixes=('', '_DROP')
    )

    # Clean up merge artifacts (drop duplicate hospitalization_id column)
    drop_cols = [col for col in cohort_df.columns if col.endswith('_DROP')]
    if drop_cols:
        cohort_df = cohort_df.drop(columns=drop_cols)

    # Fill NaN/empty strings with configured fill_no_history token
    cohort_df['prev_hosp_comorbidities'] = cohort_df['prev_hosp_comorbidities'].apply(
        lambda x: x if (isinstance(x, str) and len(x) > 0) else fill_no_history
    )

    # Count all tokens (comorbidity tokens and fill_no_history token)
    if logger:
        logger.info("Counting Elixhauser tokens...")

    for token_string in cohort_df['prev_hosp_comorbidities']:
        if pd.notna(token_string) and len(token_string) > 0:
            # Split pipe-separated string into individual tokens
            tokens = token_string.split('|')
            for token in tokens:
                token = token.strip()
                if token:
                    token_counts[token] = token_counts.get(token, 0) + 1

    if logger:
        logger.info(f"  ✓ Unique Elixhauser tokens: {len(token_counts)}")
        logger.info(f"  ✓ Total token occurrences: {sum(token_counts.values()):,}")

    # Log statistics
    if logger:
        total_with_prev = cohort_df['previous_hospitalization_id'].notna().sum()
        # Count non-empty strings (those with at least one comorbidity)
        non_empty_strings = cohort_df['prev_hosp_comorbidities'].str.len().gt(0).sum()
        # Calculate average tokens by splitting on pipe
        avg_tokens = cohort_df['prev_hosp_comorbidities'].apply(
            lambda x: len(x.split('|')) if x else 0
        ).mean()

        logger.info(f"  ✓ Created token strings for {total_with_prev:,} previous hospitalizations")
        logger.info(f"  ✓ {non_empty_strings:,} have at least one comorbidity")
        logger.info(f"  ✓ Average tokens per hospitalization: {avg_tokens:.2f}")
        logger.info("=" * 60)

    return cohort_df, token_counts
