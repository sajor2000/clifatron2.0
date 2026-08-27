#!/usr/bin/env python3
"""
tokenETL - Standalone CLIF Data ETL Pipeline (Cohort Loading)

Step 1: Load intermediate tables using clifpy table objects for tokenization.

Usage:
    uv run tokenETL/main.py
    uv run tokenETL/main.py --config path/to/clif_config.json

Default: Looks for clif_config.json in repository root
"""

import os
import json
import yaml
import argparse
import logging
import gc
from datetime import datetime
from typing import Dict, Any
import pandas as pd
from clifpy.tables import Patient, Hospitalization, Adt, HospitalDiagnosis, PatientAssessments, RespiratorySupport, CrrtTherapy, EcmoMcs, MedicationAdminContinuous, Labs, Vitals

from builders.cohort_builder import create_cohort, filter_adt_to_cohort, create_consort_diagram
from utils.tokenizer import tokenize_tables
from builders.elixhauser_builder import calculate_previous_elix
from builders.assessment_builder import build_assessment_tokens
from builders.therapy_builder import build_crrt_tokens, build_ecmo_tokens
from builders.medication_builder import build_medication_data
from builders.labs_builder import build_labs_tokens
from builders.vitals_builder import build_vitals_tokens
from builders.respiratory_support_builder import build_respiratory_support_tokens


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logger(output_dir: str, log_filename: str = 'tokenETL.log') -> logging.Logger:
    """
    Set up logger that writes to output directory.

    Args:
        output_dir: Directory to write log files
        log_filename: Name of the log file

    Returns:
        Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, log_filename)

    # Create logger
    logger = logging.getLogger('tokenETL')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
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

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("tokenETL Pipeline - Cohort Loading")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_path}")

    return logger


# ============================================================================
# Configuration Loading Functions
# ============================================================================

def load_clif_config(config_path: str) -> Dict[str, Any]:
    """
    Load clifpy configuration file (clif_config.json).

    The clifpy config contains:
    - site: Site identifier
    - data_directory: Path to CLIF data
    - filetype: Data file format (parquet, csv, etc.)
    - timezone: Timezone for datetime columns
    - output_dir: Output directory for results and logs

    Args:
        config_path: Path to clif_config.json file

    Returns:
        Loaded clifpy configuration

    Raises:
        FileNotFoundError: If config file not found
        json.JSONDecodeError: If JSON is invalid
        ValueError: If required keys are missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"clifpy configuration not found: {config_path}\n"
            "Expected clif_config.json with keys: site, data_directory, filetype, timezone, output_dir"
        )

    try:
        with open(config_path, 'r') as f:
            clif_config = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error parsing JSON config {config_path}: {e}")

    # Validate required keys
    required_keys = ['site', 'data_directory', 'filetype', 'timezone', 'output_dir']
    missing_keys = [key for key in required_keys if key not in clif_config]

    if missing_keys:
        raise ValueError(
            f"Required keys missing from clifpy config: {missing_keys}\n"
            f"Expected keys: {required_keys}"
        )

    return clif_config


def load_token_config(token_config_path: str) -> Dict[str, Any]:
    """
    Load token configuration from YAML file.

    The token configuration defines:
    - Tables to load
    - Columns for each table

    Args:
        token_config_path: Path to token_config.yaml file

    Returns:
        Loaded token configuration data

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML is invalid
    """
    if not os.path.exists(token_config_path):
        raise FileNotFoundError(
            f"Token configuration not found: {token_config_path}\n"
            "Expected token_config.yaml in tokenETL/config directory"
        )

    try:
        with open(token_config_path, 'r') as f:
            token_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config {token_config_path}: {e}")

    if 'tables' not in token_config:
        raise ValueError(f"'tables' key missing from token config: {token_config_path}")

    return token_config


# ============================================================================
# Table Loading Functions
# ============================================================================

# Map table names to clifpy table classes
TABLE_MAP = {
    'patient': Patient,
    'hospitalization': Hospitalization,
    'adt': Adt,
    'patient_assessments': PatientAssessments,
    'hospital_diagnosis': HospitalDiagnosis,
    'respiratory_support': RespiratorySupport,
    'crrt_therapy': CrrtTherapy,
    'ecmo_mcs': EcmoMcs,
    'medication_admin_continuous': MedicationAdminContinuous,
    'labs': Labs,
    'vitals': Vitals
}


def load_table(
    table_name: str,
    config_path: str,
    columns: list = None,
    filters: dict = None,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Load a single table using clifpy table objects.

    Args:
        table_name: Name of the table (e.g., 'patient', 'adt')
        config_path: Path to clif_config.json
        columns: List of columns to load (optional)
        filters: Dictionary of filters (optional)
        logger: Logger instance

    Returns:
        Loaded pandas DataFrame
    """
    if table_name not in TABLE_MAP:
        raise ValueError(f"Unknown table: {table_name}. Available: {list(TABLE_MAP.keys())}")

    TableClass = TABLE_MAP[table_name]

    if logger:
        logger.info(f"Loading {table_name} table...")
        if columns:
            logger.info(f"  Columns: {columns}")
        if filters:
            logger.info(f"  Filters: {filters}")

    # Load table using clifpy
    table_obj = TableClass.from_file(
        config_path=config_path,
        columns=columns,
        filters=filters
    )

    # Get pandas DataFrame
    df = table_obj.df.copy()

    # Strip timezone info from all datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)

    # Log shape
    rows, cols = df.shape
    if logger:
        logger.info(f"  ✓ Loaded {table_name}: {rows:,} rows × {cols} columns")

    return df


def load_all_tables(
    config_path: str,
    token_config: Dict[str, Any],
    logger: logging.Logger = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all tables defined in token configuration.

    Args:
        config_path: Path to clif_config.json
        token_config: Loaded token configuration
        logger: Logger instance

    Returns:
        Dictionary mapping table names to pandas DataFrames
    """
    if logger:
        logger.info("=" * 60)
        logger.info("LOADING TABLES")
        logger.info("=" * 60)

    tables = {}
    table_configs = token_config.get('tables', {})

    if logger:
        logger.info(f"Tables to load: {list(table_configs.keys())}")
        logger.info("")

    for table_name, table_config in table_configs.items():
        # Check if table is enabled
        if not table_config.get('enabled', True):
            if logger:
                logger.info(f"Skipping {table_name} (disabled in config)")
            continue

        # Skip patient_assessments - loaded separately in Phase 4 with cohort filters
        if table_name == 'patient_assessments':
            if logger:
                logger.info(f"Skipping {table_name} (loaded in Phase 4 with cohort filters)")
            continue

        # Skip medication_admin_continuous - loaded separately in Phase 6 with unit conversion
        if table_name == 'medication_admin_continuous':
            if logger:
                logger.info(f"Skipping {table_name} (loaded in Phase 6 with unit conversion)")
            continue

        # Skip labs - loaded separately in Phase 7 with cohort filters
        if table_name == 'labs':
            if logger:
                logger.info(f"Skipping {table_name} (loaded in Phase 7 with cohort filters)")
            continue

        # Skip vitals - loaded separately in Phase 8 with cohort filters
        if table_name == 'vitals':
            if logger:
                logger.info(f"Skipping {table_name} (loaded in Phase 8 with cohort filters)")
            continue

        # Skip respiratory_support - loaded separately in Phase 9 with cohort filters
        if table_name == 'respiratory_support':
            if logger:
                logger.info(f"Skipping {table_name} (loaded in Phase 9 with cohort filters)")
            continue

        try:
            columns = table_config.get('columns')
            filters = table_config.get('filters')

            df = load_table(
                table_name=table_name,
                config_path=config_path,
                columns=columns,
                filters=filters,
                logger=logger
            )

            tables[table_name] = df

        except Exception as e:
            if logger:
                logger.error(f"Failed to load {table_name}: {e}")
            raise

    if logger:
        logger.info("=" * 60)
        logger.info("LOADING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Successfully loaded {len(tables)} tables")
        for table_name, df in tables.items():
            rows, cols = df.shape
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            logger.info(f"  {table_name}: {rows:,} rows × {cols} cols ({memory_mb:.2f} MB)")
        logger.info("=" * 60)

    return tables


# ============================================================================
# Table Saving Utilities
# ============================================================================

def standardize_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize datetime columns for consistent parquet storage.

    This ensures all parquet files have compatible datetime schemas:
    1. Strips timezone information (doesn't convert, just removes tz)
    2. Casts all datetime columns to microsecond precision (datetime64[us])

    This prevents polars concat errors like:
    "type Datetime('ns') is incompatible with expected type Datetime('μs')"

    Args:
        df: Input pandas DataFrame

    Returns:
        DataFrame with standardized datetime columns
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            # Strip timezone if present (don't convert, just remove)
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)

            # Cast to microsecond precision for consistency
            df[col] = df[col].astype('datetime64[us]')

    return df


def save_cohort_and_adt(
    cohort_df: pd.DataFrame,
    adt_df: pd.DataFrame,
    output_dir: str,
    logger: logging.Logger = None
):
    """
    Save cohort and ADT tables to tokentables subdirectory.

    Args:
        cohort_df: Cohort DataFrame
        adt_df: ADT DataFrame
        output_dir: Output directory path
        logger: Logger instance
    """
    # Create tokentables subdirectory
    tokentables_dir = os.path.join(output_dir, 'tokentables')
    os.makedirs(tokentables_dir, exist_ok=True)

    if logger:
        logger.info("=" * 60)
        logger.info("SAVING FINAL TABLES")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Token tables directory: {tokentables_dir}")
        logger.info("")

    # Save cohort
    try:
        cohort_path = os.path.join(tokentables_dir, 'cohort.parquet')
        cohort_standardized = standardize_datetime_columns(cohort_df)
        cohort_standardized.to_parquet(cohort_path, index=False)
        file_size_mb = os.path.getsize(cohort_path) / (1024**2)
        if logger:
            logger.info(f"  ✓ Saved tokentables/cohort.parquet ({file_size_mb:.2f} MB)")
    except Exception as e:
        if logger:
            logger.error(f"  ✗ Failed to save cohort: {e}")

    # Save ADT
    try:
        adt_path = os.path.join(tokentables_dir, 'adt.parquet')
        adt_standardized = standardize_datetime_columns(adt_df)
        adt_standardized.to_parquet(adt_path, index=False)
        file_size_mb = os.path.getsize(adt_path) / (1024**2)
        if logger:
            logger.info(f"  ✓ Saved tokentables/adt.parquet ({file_size_mb:.2f} MB)")
    except Exception as e:
        if logger:
            logger.error(f"  ✗ Failed to save adt: {e}")

    if logger:
        logger.info("=" * 60)


# ============================================================================
# Main Pipeline
# ============================================================================

def run_cohort_loading(config_path: str) -> pd.DataFrame:
    """
    Main cohort loading pipeline.

    PHASE 1: COHORT CREATION
    - Loads all tables (patient, hospitalization, adt, hospital_diagnosis)
    - Creates cohort (merge, filter, calculate previous_hospitalization_id)

    MEMORY CLEANUP
    - Frees patient, hospitalization, hospital_diagnosis tables
    - Keeps ADT for Phase 3

    PHASE 2: PREVIOUS HOSPITALIZATION ELIXHAUSER
    - Calculates Elixhauser comorbidities for previous hospitalizations
    - Adds prev_hosp_comorbidities column to cohort (pipe-separated string)

    PHASE 3: TOKENIZATION
    - Tokenizes categorical columns in cohort and ADT
    - Filters ADT to cohort hospitalizations
    - Creates CONSORT diagram
    - Saves cohort.parquet and adt.parquet

    PHASE 4: PATIENT ASSESSMENTS
    - Loads patient_assessments for cohort hospitalizations
    - Creates assessment tokens (e.g., assessment_gcs_total_15, assessment_rass_0)
    - Saves assessment.parquet

    PHASE 5: THERAPY TABLES
    - Loads crrt_therapy and ecmo_mcs for cohort hospitalizations
    - Creates presence tokens (crrt_occurring, ecmo_occurring)
    - Saves crrt_therapy.parquet and ecmo_mcs.parquet

    Args:
        config_path: Path to clif_config.json

    Returns:
        Final cohort DataFrame with prev_hosp_comorbidities column
    """
    # Step 1: Load clif config
    clif_config = load_clif_config(config_path)

    # Extract configuration
    site = clif_config['site']
    data_directory = clif_config['data_directory']
    filetype = clif_config['filetype']
    timezone = clif_config['timezone']
    output_dir = clif_config['output_dir']

    # Create tokentables subdirectory for organized output
    tokentables_dir = os.path.join(output_dir, 'tokentables')
    os.makedirs(tokentables_dir, exist_ok=True)

    # Step 2: Setup logger
    logger = setup_logger(output_dir)

    logger.info("STARTING tokenETL - COHORT LOADING")

    logger.info(f"Site: {site}")
    logger.info(f"Data directory: {data_directory}")
    logger.info(f"File type: {filetype}")
    logger.info(f"Timezone: {timezone}")
    logger.info(f"Output directory: {output_dir}")

    # Step 3: Load token config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    token_config_path = os.path.join(script_dir, 'config', 'token_config.yaml')

    logger.info(f"Loading token configuration: {token_config_path}")
    token_config = load_token_config(token_config_path)
    logger.info(f"Tables configured: {list(token_config['tables'].keys())}")

    # ========================================================================
    # PHASE 1: COHORT CREATION
    # ========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 1: COHORT CREATION")
    logger.info("=" * 60)

    # Step 4: Load all tables
    tables = load_all_tables(
        config_path=config_path,
        token_config=token_config,
        logger=logger
    )

    # Step 5: Create cohort
    cohort_df, exclusion_stats = create_cohort(tables, site, logger)

    # ========================================================================
    # MEMORY CLEANUP
    # ========================================================================
    logger.info("=" * 60)
    logger.info("MEMORY CLEANUP")
    logger.info("=" * 60)

    # Keep ADT for later tokenization, delete everything else
    adt_df = tables['adt'].copy()
    del tables
    gc.collect()

    logger.info("  ✓ Freed patient, hospitalization, hospital_diagnosis from memory")
    logger.info("  ✓ Kept ADT for tokenization (Phase 3)")

    # ========================================================================
    # PHASE 2: PREVIOUS HOSPITALIZATION ELIXHAUSER
    # ========================================================================
    cohort_df, elix_token_counts = calculate_previous_elix(cohort_df, config_path, token_config, logger)

    # ========================================================================
    # PHASE 3: TOKENIZATION
    # ========================================================================
    logger.info("=" * 60)
    logger.info("PHASE 3: TOKENIZATION")
    logger.info("=" * 60)

    # Step 6: Tokenize cohort and ADT tables (ADT already in memory)
    cohort_df, adt_tokenized, token_counts = tokenize_tables(cohort_df, adt_df, token_config, logger)

    # Build comprehensive token list with sources
    # This will track which table each token came from
    if logger:
        logger.info("")
        logger.info("Building token list with source tracking...")

    token_list = []

    # Add cohort/ADT tokens (source: cohort_adt)
    for token, count in token_counts.items():
        token_list.append({'token': token, 'count': count, 'source': 'cohort_adt'})

    # Add Elixhauser tokens (source: elixhauser)
    for token, count in elix_token_counts.items():
        token_list.append({'token': token, 'count': count, 'source': 'elixhauser'})

    if logger:
        logger.info(f"  ✓ Added cohort/ADT tokens: {len(token_counts)} unique")
        logger.info(f"  ✓ Added Elixhauser tokens: {len(elix_token_counts)} unique")

    # Step 7: Filter ADT to only cohort hospitalizations
    adt_filtered = filter_adt_to_cohort(adt_tokenized, cohort_df, logger)

    # Step 8: Create CONSORT diagram
    create_consort_diagram(exclusion_stats, output_dir, logger)

    # ========================================================================
    # PHASE 4: PATIENT ASSESSMENTS
    # ========================================================================
    assessments_df, assessment_token_counts = build_assessment_tokens(cohort_df, config_path, token_config, logger)

    # Save patient assessments
    if logger:
        logger.info("")
        logger.info("Saving patient assessments...")

    try:
        assessment_path = os.path.join(tokentables_dir, 'assessment.parquet')
        assessments_standardized = standardize_datetime_columns(assessments_df)
        assessments_standardized.to_parquet(assessment_path, index=False)
        file_size_mb = os.path.getsize(assessment_path) / (1024**2)
        if logger:
            logger.info(f"  ✓ Saved tokentables/assessment.parquet ({file_size_mb:.2f} MB)")
            logger.info(f"  ✓ Records: {len(assessments_df):,}")
    except Exception as e:
        if logger:
            logger.error(f"  ✗ Failed to save assessment.parquet: {e}")

    # ========================================================================
    # PHASE 5: THERAPY TABLES (CRRT, ECMO/MCS)
    # ========================================================================
    # Build CRRT tokens
    crrt_df, crrt_token_counts = build_crrt_tokens(cohort_df, config_path, token_config, logger)

    # Save CRRT therapy table
    if len(crrt_df) > 0:
        if logger:
            logger.info("")
            logger.info("Saving CRRT therapy...")

        try:
            crrt_path = os.path.join(tokentables_dir, 'crrt_therapy.parquet')
            crrt_standardized = standardize_datetime_columns(crrt_df)
            crrt_standardized.to_parquet(crrt_path, index=False)
            file_size_mb = os.path.getsize(crrt_path) / (1024**2)
            if logger:
                logger.info(f"  ✓ Saved tokentables/crrt_therapy.parquet ({file_size_mb:.2f} MB)")
                logger.info(f"  ✓ Records: {len(crrt_df):,}")
        except Exception as e:
            if logger:
                logger.error(f"  ✗ Failed to save crrt_therapy.parquet: {e}")

    # Build ECMO/MCS tokens
    ecmo_df, ecmo_token_counts = build_ecmo_tokens(cohort_df, config_path, token_config, logger)

    # Save ECMO/MCS table
    if len(ecmo_df) > 0:
        if logger:
            logger.info("")
            logger.info("Saving ECMO/MCS...")

        try:
            ecmo_path = os.path.join(tokentables_dir, 'ecmo_mcs.parquet')
            ecmo_standardized = standardize_datetime_columns(ecmo_df)
            ecmo_standardized.to_parquet(ecmo_path, index=False)
            file_size_mb = os.path.getsize(ecmo_path) / (1024**2)
            if logger:
                logger.info(f"  ✓ Saved tokentables/ecmo_mcs.parquet ({file_size_mb:.2f} MB)")
                logger.info(f"  ✓ Records: {len(ecmo_df):,}")
        except Exception as e:
            if logger:
                logger.error(f"  ✗ Failed to save ecmo_mcs.parquet: {e}")

    # ========================================================================
    # PHASE 6: MEDICATION ADMIN CONTINUOUS
    # ========================================================================
    # Build medication data with unit conversion and tokenization
    medication_df, medication_token_counts = build_medication_data(cohort_df, config_path, token_config, logger)

    # Save medication table
    if len(medication_df) > 0:
        if logger:
            logger.info("")
            logger.info("Saving medication data...")

        try:
            medication_path = os.path.join(tokentables_dir, 'medication_admin_continuous.parquet')
            medication_standardized = standardize_datetime_columns(medication_df)
            medication_standardized.to_parquet(medication_path, index=False)
            file_size_mb = os.path.getsize(medication_path) / (1024**2)
            if logger:
                logger.info(f"  ✓ Saved tokentables/medication_admin_continuous.parquet ({file_size_mb:.2f} MB)")
                logger.info(f"  ✓ Records: {len(medication_df):,}")
        except Exception as e:
            if logger:
                logger.error(f"  ✗ Failed to save medication_admin_continuous.parquet: {e}")

    # ========================================================================
    # PHASE 7: LABS
    # ========================================================================
    # Build labs tokens
    labs_df, labs_token_counts = build_labs_tokens(cohort_df, config_path, token_config, logger)

    # Save labs table
    if len(labs_df) > 0:
        if logger:
            logger.info("")
            logger.info("Saving labs data...")

        try:
            labs_path = os.path.join(tokentables_dir, 'labs.parquet')
            labs_standardized = standardize_datetime_columns(labs_df)
            labs_standardized.to_parquet(labs_path, index=False)
            file_size_mb = os.path.getsize(labs_path) / (1024**2)
            if logger:
                logger.info(f"  ✓ Saved tokentables/labs.parquet ({file_size_mb:.2f} MB)")
                logger.info(f"  ✓ Records: {len(labs_df):,}")
        except Exception as e:
            if logger:
                logger.error(f"  ✗ Failed to save labs.parquet: {e}")

    # ========================================================================
    # PHASE 8: VITALS
    # ========================================================================
    # Build vitals tokens
    vitals_df, vitals_token_counts = build_vitals_tokens(cohort_df, config_path, token_config, logger)

    # Save vitals table
    if len(vitals_df) > 0:
        if logger:
            logger.info("")
            logger.info("Saving vitals data...")

        try:
            vitals_path = os.path.join(tokentables_dir, 'vitals.parquet')
            vitals_standardized = standardize_datetime_columns(vitals_df)
            vitals_standardized.to_parquet(vitals_path, index=False)
            file_size_mb = os.path.getsize(vitals_path) / (1024**2)
            if logger:
                logger.info(f"  ✓ Saved tokentables/vitals.parquet ({file_size_mb:.2f} MB)")
                logger.info(f"  ✓ Records: {len(vitals_df):,}")
        except Exception as e:
            if logger:
                logger.error(f"  ✗ Failed to save vitals.parquet: {e}")

    # ========================================================================
    # PHASE 9: RESPIRATORY SUPPORT
    # ========================================================================
    # Build respiratory support tokens
    resp_support_df, resp_support_token_counts = build_respiratory_support_tokens(cohort_df, config_path, token_config, logger)

    # Save respiratory support table
    if len(resp_support_df) > 0:
        if logger:
            logger.info("")
            logger.info("Saving respiratory support data...")

        try:
            resp_support_path = os.path.join(tokentables_dir, 'respiratory_support.parquet')
            resp_support_standardized = standardize_datetime_columns(resp_support_df)
            resp_support_standardized.to_parquet(resp_support_path, index=False)
            file_size_mb = os.path.getsize(resp_support_path) / (1024**2)
            if logger:
                logger.info(f"  ✓ Saved tokentables/respiratory_support.parquet ({file_size_mb:.2f} MB)")
                logger.info(f"  ✓ Records: {len(resp_support_df):,}")
        except Exception as e:
            if logger:
                logger.error(f"  ✗ Failed to save respiratory_support.parquet: {e}")

    # Merge all token counts with source tracking
    if logger:
        logger.info("")
        logger.info("Merging token counts from all sources...")

    # Add assessment tokens (source: assessment)
    for token, count in assessment_token_counts.items():
        token_list.append({'token': token, 'count': count, 'source': 'assessment'})

    # Add CRRT tokens (source: crrt_therapy)
    for token, count in crrt_token_counts.items():
        token_list.append({'token': token, 'count': count, 'source': 'crrt_therapy'})

    # Add ECMO/MCS tokens (source: ecmo_mcs)
    for token, count in ecmo_token_counts.items():
        token_list.append({'token': token, 'count': count, 'source': 'ecmo_mcs'})

    # Add Medication tokens (source: medication_admin_continuous)
    for token, count in medication_token_counts.items():
        token_list.append({'token': token, 'count': count, 'source': 'medication_admin_continuous'})

    # Add Labs tokens (source: labs)
    for token, count in labs_token_counts.items():
        token_list.append({'token': token, 'count': count, 'source': 'labs'})

    # Add Vitals tokens (source: vitals)
    for token, count in vitals_token_counts.items():
        token_list.append({'token': token, 'count': count, 'source': 'vitals'})

    # Add Respiratory Support tokens (source: respiratory_support)
    for token, count in resp_support_token_counts.items():
        token_list.append({'token': token, 'count': count, 'source': 'respiratory_support'})

    if logger:
        total_tokens = len(token_list)
        total_occurrences = sum(item['count'] for item in token_list)
        logger.info(f"  ✓ Total token entries (with sources): {total_tokens:,}")
        logger.info(f"  ✓ Total token occurrences: {total_occurrences:,}")

    # Save complete token registry with zeros for missing tokens (site auditing)
    if logger:
        logger.info("")
        logger.info("Generating complete token registry from configuration...")

    try:
        # Convert to DataFrame for merging with registry
        token_counts_df = pd.DataFrame(token_list)

        from utils.polars_utils import load_master_token_registry

        # Load master registry from CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))
        registry_path = os.path.join(script_dir, 'config', 'critical_illness_tokenization_final_with_intervals.csv')
        master_registry = load_master_token_registry(registry_path)

        # Extract all possible tokens and categorize by source
        all_tokens = []

        # From critical_illness_tokenization_final_with_intervals.csv
        for category in ['labs', 'medications', 'vitals', 'respiratory_support']:
            category_tokens = master_registry[master_registry['category'] == category]['token'].unique()

            # Map category to source name
            source_map = {
                'labs': 'labs',
                'medications': 'medication_admin_continuous',
                'vitals': 'vitals',
                'respiratory_support': 'respiratory_support'
            }
            source = source_map[category]

            for token in category_tokens:
                if pd.notna(token):
                    all_tokens.append({'token': token, 'source': source})

        # Add categorical tokens from token_config.yaml
        # ADT tokens (cohort_adt)
        adt_config = token_config['tables']['adt']['tokenization']['location_category']
        for original, mapped in adt_config['mapping'].items():
            token = f"{adt_config['prefix']}{mapped}"
            all_tokens.append({'token': token, 'source': 'cohort_adt'})

        # Disposition tokens (demographics - but tracked as cohort_adt in token_counts)
        hosp_config = token_config['tables']['hospitalization']['tokenization']
        for original, mapped in hosp_config['discharge_category']['mapping'].items():
            token = f"{hosp_config['discharge_category']['prefix']}{mapped}"
            all_tokens.append({'token': token, 'source': 'cohort_adt'})

        # Age bins (cohort_adt)
        for bin_info in hosp_config['age_at_admission']['bins']:
            token = f"{hosp_config['age_at_admission']['prefix']}{bin_info['name']}"
            all_tokens.append({'token': token, 'source': 'cohort_adt'})

        # Sex tokens (cohort_adt)
        patient_config = token_config['tables']['patient']['tokenization']
        for original, mapped in patient_config['sex_category']['mapping'].items():
            token = mapped
            all_tokens.append({'token': token, 'source': 'cohort_adt'})

        # Elixhauser tokens (from hospital_diagnosis config)
        diag_config = token_config['tables']['hospital_diagnosis']['tokenization']
        elix_config = diag_config['elixhauser']
        for comorbidity in elix_config['comorbidities']:
            token = f"{elix_config['prefix']}{comorbidity}"
            all_tokens.append({'token': token, 'source': 'elixhauser'})
        # Add no_patient_history token
        all_tokens.append({'token': elix_config['fill_no_history'], 'source': 'elixhauser'})

        # Assessment tokens
        assessment_config = token_config['tables']['patient_assessments']['tokenization']
        for assessment_name, assessment_info in assessment_config['assessments'].items():
            for original, mapped in assessment_info['mapping'].items():
                token = f"{assessment_config['prefix']}{mapped}"
                all_tokens.append({'token': token, 'source': 'assessment'})

        # CRRT token
        crrt_config = token_config['tables']['crrt_therapy']['tokenization']
        all_tokens.append({'token': crrt_config['add_token'], 'source': 'crrt_therapy'})

        # ECMO token
        ecmo_config = token_config['tables']['ecmo_mcs']['tokenization']
        all_tokens.append({'token': ecmo_config['add_token'], 'source': 'ecmo_mcs'})

        # Respiratory support categorical tokens
        resp_config = token_config['tables']['respiratory_support']['tokenization']

        # Device category
        device_config = resp_config['device_category']
        for original, mapped in device_config['mapping'].items():
            token = f"{device_config['prefix']}{mapped}"
            all_tokens.append({'token': token, 'source': 'respiratory_support'})

        # Mode category
        mode_config = resp_config['mode_category']
        for original, mapped in mode_config['mapping'].items():
            token = f"{mode_config['prefix']}{mapped}"
            all_tokens.append({'token': token, 'source': 'respiratory_support'})

        # Tracheostomy tokens (from config mapping)
        trach_config = resp_config['tracheostomy']
        for original, mapped in trach_config['mapping'].items():
            token = f"{trach_config['prefix']}{mapped}"
            all_tokens.append({'token': token, 'source': 'respiratory_support'})

        # Convert to DataFrame
        all_tokens_df = pd.DataFrame(all_tokens).drop_duplicates(subset=['token', 'source'])

        # Merge with actual counts
        complete_token_counts = all_tokens_df.merge(
            token_counts_df[['token', 'count']],
            on='token',
            how='left'
        )

        # Fill missing counts with 0
        complete_token_counts['count'] = complete_token_counts['count'].fillna(0).astype(int)

        # Add present_in_data column
        complete_token_counts['present_in_data'] = complete_token_counts['count'] > 0

        # Sort by source, then by count descending
        complete_token_counts = complete_token_counts.sort_values(
            ['source', 'count'],
            ascending=[True, False]
        )

        # Save complete token counts (CSV)
        complete_path = os.path.join(output_dir, 'token_counts_complete.csv')
        complete_token_counts.to_csv(complete_path, index=False)
        file_size_kb = os.path.getsize(complete_path) / 1024

        if logger:
            logger.info(f"  ✓ Saved token_counts_complete.csv ({file_size_kb:.2f} KB)")
            logger.info(f"  ✓ Total tokens in registry: {len(complete_token_counts):,}")
            logger.info(f"  ✓ Tokens present in data: {complete_token_counts['present_in_data'].sum():,}")
            logger.info(f"  ✓ Tokens missing from data: {(~complete_token_counts['present_in_data']).sum():,}")
            logger.info(f"  ✓ Format: token, source, count, present_in_data")

        # Generate JSON output (hierarchical structure by source)
        import json

        token_registry = {}
        for source in complete_token_counts['source'].unique():
            source_tokens = complete_token_counts[complete_token_counts['source'] == source]
            token_registry[source] = {}

            for _, row in source_tokens.iterrows():
                token_registry[source][row['token']] = {
                    'count': int(row['count']),
                    'present_in_data': bool(row['present_in_data'])
                }

        # Save JSON output
        json_path = os.path.join(output_dir, 'token_registry.json')
        with open(json_path, 'w') as f:
            json.dump(token_registry, f, indent=2)

        json_size_kb = os.path.getsize(json_path) / 1024

        if logger:
            logger.info(f"  ✓ Saved token_registry.json ({json_size_kb:.2f} KB)")
            logger.info(f"  ✓ Hierarchical structure: {{source: {{token: {{count, present_in_data}}}}}}")
            logger.info(f"  ✓ Sources: {len(token_registry)}")
            logger.info(f"  ✓ Use these files for site auditing to identify missing tokens")

    except Exception as e:
        if logger:
            logger.error(f"  ✗ Failed to save token counts: {e}")
            import traceback
            traceback.print_exc()

    # Step 9: Save final tables (cohort and filtered ADT)
    save_cohort_and_adt(cohort_df, adt_filtered, output_dir, logger)

    # Completion
    logger.info("=" * 60)
    logger.info("TOKENIZATION PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Cohort: {len(cohort_df):,} rows, {len(cohort_df.columns)} columns")
    logger.info(f"ADT (filtered): {len(adt_filtered):,} rows, {len(adt_filtered.columns)} columns")
    logger.info(f"Assessments: {len(assessments_df):,} rows, {len(assessments_df.columns)} columns")
    logger.info(f"CRRT Therapy: {len(crrt_df):,} rows, {len(crrt_df.columns)} columns")
    logger.info(f"ECMO/MCS: {len(ecmo_df):,} rows, {len(ecmo_df.columns)} columns")
    logger.info(f"Medication Admin Continuous: {len(medication_df):,} rows, {len(medication_df.columns)} columns")
    logger.info(f"Labs: {len(labs_df):,} rows, {len(labs_df.columns)} columns")
    logger.info(f"Vitals: {len(vitals_df):,} rows, {len(vitals_df.columns)} columns")
    logger.info(f"Respiratory Support: {len(resp_support_df):,} rows, {len(resp_support_df.columns)} columns")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("Next Step: Assemble Narratives")
    logger.info("=" * 60)
    logger.info("Run narrative assembly to create chronological sequences:")
    logger.info("  uv run tokenETL/assemble_narratives.py --config clif_config.json")
    logger.info("=" * 60)

    return cohort_df


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point with argument parsing."""
    # Default config path: repo root / clif_config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)  # Go up one level from tokenETL/
    default_config = os.path.join(repo_root, 'clif_config.json')

    parser = argparse.ArgumentParser(
        description='tokenETL - Cohort Loading Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  uv run tokenETL/main.py
  uv run tokenETL/main.py --config /path/to/custom_config.json

Config file should contain:
  - site: Site identifier
  - data_directory: Path to CLIF data
  - filetype: File format (parquet, csv, etc.)
  - timezone: Timezone for datetime columns
  - output_dir: Output directory for results and logs

Default: Looks for clif_config.json in repository root
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default=default_config,
        help=f'Path to clif_config.json (default: {default_config})'
    )

    args = parser.parse_args()

    try:
        # Run the cohort loading pipeline
        cohort_df = run_cohort_loading(config_path=args.config)

        print(f"\nPipeline completed successfully!")
        print(f"Cohort shape: {cohort_df.shape}")

    except Exception as e:
        print(f"\nError running tokenETL pipeline: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        # Explicitly clear polars thread pool to avoid semaphore leaks on shutdown
        try:
            import polars as pl
            pl.clear_thread_pool()
            import gc
            gc.collect()
        except (ImportError, AttributeError):
            # Handle cases where polars is not installed or version is too old
            pass


if __name__ == "__main__":
    main()
