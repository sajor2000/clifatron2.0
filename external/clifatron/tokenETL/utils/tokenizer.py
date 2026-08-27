"""
Tokenizer Module

Contains logic for tokenizing categorical and numeric columns in cohort and ADT tables.
Includes normalization, mapping, and binning functionality with detailed logging.
"""

import re
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any


def normalize_string(s: str) -> str:
    """
    Normalize string for matching by converting to lowercase and removing
    spaces and special characters.

    Args:
        s: Input string

    Returns:
        Normalized string (lowercase, no spaces, no special chars)
    """
    if pd.isna(s):
        return ""

    # Convert to string and lowercase
    normalized = str(s).lower()

    # Remove special characters: . / \ { } [ ] ( ) and spaces
    normalized = re.sub(r'[./\\{}\[\]()\s]', '', normalized)

    return normalized


def tokenize_mapping(
    df: pd.DataFrame,
    column: str,
    mapping: Dict[str, str],
    prefix: str = "",
    map_unmapped_to_other: bool = False,
    token_counts: Dict[str, int] = None,
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Tokenize a column using a mapping dictionary with normalization.

    Creates a new {column}_token column while preserving the original column.
    Normalizes both data values and mapping keys for robust matching.
    Logs matched and unmapped values for data quality checking.
    Tracks token counts across all tokenized values.

    Args:
        df: Input DataFrame
        column: Column name to tokenize
        mapping: Dictionary mapping original values to tokens
        prefix: Prefix to add to all tokens (e.g., "age_", "disposition_")
        map_unmapped_to_other: If True, map unmapped values to "{prefix}other"
        token_counts: Dictionary tracking token counts (updated in place)
        logger: Logger instance

    Returns:
        Tuple of (DataFrame with new {column}_token column, updated token_counts dict)
    """
    if logger:
        logger.info(f"Tokenizing {column} (mapping)...")

    df = df.copy()

    # Initialize token_counts if not provided
    if token_counts is None:
        token_counts = {}

    # Create normalized mapping: {normalized_key: token_value}
    # Apply prefix to token values
    normalized_mapping = {}
    for original_key, token_value in mapping.items():
        norm_key = normalize_string(original_key)
        prefixed_token = f"{prefix}{token_value}"
        normalized_mapping[norm_key] = prefixed_token

    # Get unique values in column
    unique_values = df[column].dropna().unique()

    # Track matches and mismatches
    matched = {}
    unmapped = []
    unmapped_to_other = []

    for value in unique_values:
        norm_value = normalize_string(value)

        if norm_value in normalized_mapping:
            token = normalized_mapping[norm_value]
            matched[value] = token
        else:
            unmapped.append(value)

    # Apply tokenization - create NEW column instead of overwriting
    value_counts = df[column].value_counts()
    token_column = f"{column}_token"

    def map_value(val):
        if pd.isna(val):
            return val
        norm_val = normalize_string(val)

        if norm_val in normalized_mapping:
            return normalized_mapping[norm_val]
        elif map_unmapped_to_other:
            return f"{prefix}other"
        else:
            return None  # Return None for unmapped values

    df[token_column] = df[column].apply(map_value)

    # Count tokens
    token_value_counts = df[token_column].value_counts()
    for token, count in token_value_counts.items():
        if pd.notna(token):
            token_counts[token] = token_counts.get(token, 0) + count

    # Log results
    if logger:
        logger.info(f"  ✓ Matched {len(matched)} unique values:")
        for original, token in sorted(matched.items()):
            count = value_counts.get(original, 0)
            logger.info(f"    - {original} ({count:,}) → {token}")

        if unmapped:
            if map_unmapped_to_other:
                logger.info(f"  ✓ Unmapped → Other: {len(unmapped)} values")
                for value in sorted(unmapped):
                    count = value_counts.get(value, 0)
                    logger.info(f"    - {value} ({count:,}) → {prefix}other [UNMAPPED → OTHER]")
            else:
                logger.info(f"  ✗ Unmapped: {len(unmapped)} values")
                for value in sorted(unmapped):
                    count = value_counts.get(value, 0)
                    logger.info(f"    - {value} ({count:,}) [NO MATCH]")
        else:
            logger.info(f"  ✓ Unmapped: 0 values")

    return df, token_counts


def tokenize_bins(
    df: pd.DataFrame,
    column: str,
    bins: List[Dict[str, Any]],
    prefix: str = "",
    token_counts: Dict[str, int] = None,
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Tokenize a numeric column using bins.

    Creates a new {column}_token column while preserving the original column.
    Tracks token counts across all tokenized values.

    Args:
        df: Input DataFrame
        column: Column name to tokenize
        bins: List of bin definitions with 'name', 'min', 'max'
        prefix: Prefix to add to all bin tokens (e.g., "age_")
        token_counts: Dictionary tracking token counts (updated in place)
        logger: Logger instance

    Returns:
        Tuple of (DataFrame with new {column}_token column, updated token_counts dict)
    """
    if logger:
        logger.info(f"Tokenizing {column} (bins)...")

    df = df.copy()

    # Initialize token_counts if not provided
    if token_counts is None:
        token_counts = {}

    # Apply binning logic
    def assign_bin(value):
        if pd.isna(value):
            return None

        for bin_def in bins:
            bin_name = bin_def['name']
            bin_min = bin_def['min']
            bin_max = bin_def['max']

            # Handle null max (for 85+)
            if bin_max is None or bin_max == 999:
                if value >= bin_min:
                    return f"{prefix}{bin_name}"
            else:
                if bin_min <= value <= bin_max:
                    return f"{prefix}{bin_name}"

        # If no bin matches, return None
        return None

    # Store original for counting
    original_col = df[column].copy()
    token_column = f"{column}_token"

    # Apply binning - create NEW column instead of overwriting
    df[token_column] = df[column].apply(assign_bin)

    # Count tokens
    token_value_counts = df[token_column].value_counts()
    for token, count in token_value_counts.items():
        if pd.notna(token):
            token_counts[token] = token_counts.get(token, 0) + count

    # Log distribution
    if logger:
        bin_counts = df[token_column].value_counts().sort_index()
        logger.info(f"  ✓ Distribution across {len(bin_counts)} bins:")
        for bin_def in bins:
            bin_token = f"{prefix}{bin_def['name']}"
            count = bin_counts.get(bin_token, 0)
            logger.info(f"    - {bin_token}: {count:,}")

        # Check for unmapped values
        unmapped_count = df[token_column].isna().sum() - original_col.isna().sum()
        if unmapped_count > 0:
            logger.info(f"  ✗ {unmapped_count:,} values did not fall into any bin")

    return df, token_counts


def tokenize_tables(
    cohort_df: pd.DataFrame,
    adt_df: pd.DataFrame,
    token_config: Dict[str, Any],
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Tokenize cohort and ADT tables based on token configuration.

    Creates new _token columns for each tokenized field while preserving originals.
    Tracks token counts across all tokenized columns.

    Applies tokenization to:
    - cohort: sex_category, age_at_admission, discharge_category
    - adt: location_category

    Args:
        cohort_df: Cohort DataFrame
        adt_df: ADT DataFrame
        token_config: Loaded token configuration
        logger: Logger instance

    Returns:
        Tuple of (tokenized_cohort, tokenized_adt, token_counts)
    """
    if logger:
        logger.info("=" * 60)
        logger.info("TOKENIZING TABLES")
        logger.info("=" * 60)

    cohort_df = cohort_df.copy()
    adt_df = adt_df.copy()

    # Initialize token counts dictionary
    token_counts = {}

    tables_config = token_config.get('tables', {})

    # Tokenize patient columns in cohort
    patient_config = tables_config.get('patient', {}).get('tokenization', {})

    if 'sex_category' in cohort_df.columns:
        sex_config = patient_config.get('sex_category', {})
        if sex_config.get('enabled') and sex_config.get('method') == 'mapping':
            cohort_df, token_counts = tokenize_mapping(
                cohort_df,
                'sex_category',
                sex_config.get('mapping', {}),
                prefix=sex_config.get('prefix', ''),
                map_unmapped_to_other=sex_config.get('map_unmapped_to_other', False),
                token_counts=token_counts,
                logger=logger
            )

    # Tokenize hospitalization columns in cohort
    hosp_config = tables_config.get('hospitalization', {}).get('tokenization', {})

    if 'age_at_admission' in cohort_df.columns:
        age_config = hosp_config.get('age_at_admission', {})
        if age_config.get('enabled') and age_config.get('method') == 'bins':
            cohort_df, token_counts = tokenize_bins(
                cohort_df,
                'age_at_admission',
                age_config.get('bins', []),
                prefix=age_config.get('prefix', ''),
                token_counts=token_counts,
                logger=logger
            )

    if 'discharge_category' in cohort_df.columns:
        discharge_config = hosp_config.get('discharge_category', {})
        if discharge_config.get('enabled') and discharge_config.get('method') == 'mapping':
            cohort_df, token_counts = tokenize_mapping(
                cohort_df,
                'discharge_category',
                discharge_config.get('mapping', {}),
                prefix=discharge_config.get('prefix', ''),
                map_unmapped_to_other=discharge_config.get('map_unmapped_to_other', False),
                token_counts=token_counts,
                logger=logger
            )

    # Tokenize ADT columns
    adt_config = tables_config.get('adt', {}).get('tokenization', {})

    if 'location_category' in adt_df.columns:
        location_config = adt_config.get('location_category', {})
        if location_config.get('enabled') and location_config.get('method') == 'mapping':
            adt_df, token_counts = tokenize_mapping(
                adt_df,
                'location_category',
                location_config.get('mapping', {}),
                prefix=location_config.get('prefix', ''),
                map_unmapped_to_other=location_config.get('map_unmapped_to_other', False),
                token_counts=token_counts,
                logger=logger
            )

    if logger:
        logger.info("=" * 60)
        logger.info("TOKENIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total unique tokens: {len(token_counts)}")
        logger.info(f"Total token occurrences: {sum(token_counts.values()):,}")
        logger.info("=" * 60)

    return cohort_df, adt_df, token_counts
