"""
Therapy Builder Module

Loads CRRT and ECMO/MCS therapy tables and creates presence-based tokens.
Each row represents therapy occurring, so we add a constant token.
"""

import pandas as pd
import logging
from typing import Dict, Any, Tuple
from clifpy.tables import CrrtTherapy, EcmoMcs
from utils.polars_utils import strip_all_datetime_timezones


def build_crrt_tokens(
    cohort_df: pd.DataFrame,
    config_path: str,
    token_config: Dict[str, Any],
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load CRRT therapy records for cohort and create presence tokens.

    Each row in crrt_therapy represents CRRT occurring, so we add a constant
    token "crrt_occurring" to every record.

    Process:
    1. Get cohort hospitalization_ids
    2. Load crrt_therapy filtered to those IDs
    3. Add crrt_token column with constant "crrt_occurring"
    4. Keep only: hospitalization_id, recorded_dttm, crrt_token
    5. Count token occurrences

    Args:
        cohort_df: Cohort DataFrame with hospitalization_id column
        config_path: Path to clif_config.json
        token_config: Loaded token configuration
        logger: Logger instance

    Returns:
        Tuple of (DataFrame with hospitalization_id, recorded_dttm, crrt_token, token_counts dict)
    """
    if logger:
        logger.info("=" * 60)
        logger.info("LOADING CRRT THERAPY")
        logger.info("=" * 60)

    # Get cohort hospitalization_ids
    cohort_hosp_ids = cohort_df['hospitalization_id'].unique()

    if logger:
        logger.info(f"Cohort hospitalizations: {len(cohort_hosp_ids):,}")
        logger.info("Loading crrt_therapy for cohort hospitalizations...")

    # Load crrt_therapy filtered to cohort
    try:
        crrt_table = CrrtTherapy.from_file(
            config_path=config_path,
            columns=['hospitalization_id', 'recorded_dttm'],
            filters={'hospitalization_id': list(cohort_hosp_ids)}
        )
        crrt_df = crrt_table.df.copy()

        # Strip timezone from ALL datetime columns using utility function
        crrt_df = strip_all_datetime_timezones(crrt_df)
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠ Failed to load crrt_therapy: {e}")
            logger.info("  Skipping CRRT tokenization")
            logger.info("=" * 60)
        # Return empty DataFrame and counts
        return pd.DataFrame(columns=['hospitalization_id', 'recorded_dttm', 'crrt_token']), {}

    if logger:
        logger.info(f"  ✓ Loaded {len(crrt_df):,} CRRT records")

    # Get configured token name from config
    crrt_config = token_config.get('tables', {}).get('crrt_therapy', {}).get('tokenization', {})
    token_name = crrt_config.get('add_token', 'crrt_occurring')

    # Add constant token column
    if logger:
        logger.info("")
        logger.info(f"Adding crrt_token column with value '{token_name}'...")

    crrt_df['crrt_token'] = token_name

    # Count tokens
    token_counts = {token_name: len(crrt_df)}

    if logger:
        logger.info(f"  ✓ Created {len(crrt_df):,} CRRT tokens")
        logger.info(f"  ✓ Token: {token_name}")
        logger.info("=" * 60)

    # Return final columns and token counts
    return crrt_df[['hospitalization_id', 'recorded_dttm', 'crrt_token']], token_counts


def build_ecmo_tokens(
    cohort_df: pd.DataFrame,
    config_path: str,
    token_config: Dict[str, Any],
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load ECMO/MCS records for cohort and create presence tokens.

    Each row in ecmo_mcs represents ECMO/MCS occurring, so we add a constant
    token "ecmo_occurring" to every record.

    Process:
    1. Get cohort hospitalization_ids
    2. Load ecmo_mcs filtered to those IDs
    3. Add ecmo_token column with constant "ecmo_occurring"
    4. Keep only: hospitalization_id, recorded_dttm, ecmo_token
    5. Count token occurrences

    Args:
        cohort_df: Cohort DataFrame with hospitalization_id column
        config_path: Path to clif_config.json
        token_config: Loaded token configuration
        logger: Logger instance

    Returns:
        Tuple of (DataFrame with hospitalization_id, recorded_dttm, ecmo_token, token_counts dict)
    """
    if logger:
        logger.info("=" * 60)
        logger.info("LOADING ECMO/MCS")
        logger.info("=" * 60)

    # Get cohort hospitalization_ids
    cohort_hosp_ids = cohort_df['hospitalization_id'].unique()

    if logger:
        logger.info(f"Cohort hospitalizations: {len(cohort_hosp_ids):,}")
        logger.info("Loading ecmo_mcs for cohort hospitalizations...")

    # Load ecmo_mcs filtered to cohort
    try:
        ecmo_table = EcmoMcs.from_file(
            config_path=config_path,
            columns=['hospitalization_id', 'recorded_dttm'],
            filters={'hospitalization_id': list(cohort_hosp_ids)}
        )
        ecmo_df = ecmo_table.df.copy()

        # Strip timezone from ALL datetime columns using utility function
        ecmo_df = strip_all_datetime_timezones(ecmo_df)
    except Exception as e:
        if logger:
            logger.warning(f"  ⚠ Failed to load ecmo_mcs: {e}")
            logger.info("  Skipping ECMO tokenization")
            logger.info("=" * 60)
        # Return empty DataFrame and counts
        return pd.DataFrame(columns=['hospitalization_id', 'recorded_dttm', 'ecmo_token']), {}

    if logger:
        logger.info(f"  ✓ Loaded {len(ecmo_df):,} ECMO/MCS records")

    # Get configured token name from config
    ecmo_config = token_config.get('tables', {}).get('ecmo_mcs', {}).get('tokenization', {})
    token_name = ecmo_config.get('add_token', 'ecmo_occurring')

    # Add constant token column
    if logger:
        logger.info("")
        logger.info(f"Adding ecmo_token column with value '{token_name}'...")

    ecmo_df['ecmo_token'] = token_name

    # Count tokens
    token_counts = {token_name: len(ecmo_df)}

    if logger:
        logger.info(f"  ✓ Created {len(ecmo_df):,} ECMO/MCS tokens")
        logger.info(f"  ✓ Token: {token_name}")
        logger.info("=" * 60)

    # Return final columns and token counts
    return ecmo_df[['hospitalization_id', 'recorded_dttm', 'ecmo_token']], token_counts
