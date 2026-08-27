#!/usr/bin/env python3
"""
assemble_narratives.py - CLIF Narrative Assembly Pipeline (Step 2)

Assembles chronological clinical narratives from tokenized parquet files.
Run this AFTER tokenization (main.py) completes.

Usage:
    uv run tokenETL/assemble_narratives.py
    uv run tokenETL/assemble_narratives.py --config path/to/clif_config.json

Default: Looks for clif_config.json in repository root
"""

import os
import json
import argparse
import logging
import polars as pl
from datetime import datetime
from builders.narrative_assembler import build_narrative_sequences


def setup_logger(output_dir: str, log_filename: str = 'narrative_assembly.log') -> logging.Logger:
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
    logger = logging.getLogger('narrative_assembly')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter('%(message)s')

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> dict:
    """
    Load clif_config.json and validate required fields.

    Args:
        config_path: Path to clif_config.json

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If required fields missing
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Validate required fields
    required_keys = ['output_dir']
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ValueError(f"Required keys missing from clif_config: {missing_keys}")

    return config


def format_narrative_with_timestamps(narrative_df: pl.DataFrame) -> str:
    """
    Format a single hospitalization narrative with left-aligned timestamps and hierarchical indentation.

    Args:
        narrative_df: Polars DataFrame for ONE hospitalization (filtered), sorted by event_time, sequence_order

    Returns:
        Formatted string with timestamps and indented tokens
    """
    import pandas as pd

    # Convert to pandas for easier iteration
    df_pd = narrative_df.to_pandas()

    output_lines = []
    current_day = None
    previous_was_day_marker = False

    for idx, row in df_pd.iterrows():
        token = row['clif_sentence']
        event_time = row['event_time']
        day = row['day']
        sequence_order = row['sequence_order']

        # Format timestamp (left-aligned, 25 chars wide)
        if pd.notna(event_time):
            timestamp_str = event_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = ''
        timestamp_col = f"{timestamp_str:<25}"

        # Determine indentation and formatting based on token type
        if token == 'PREV_NARRATIVE_START':
            output_lines.append(f"{timestamp_col}PREV_NARRATIVE_START")

        elif token == 'PREV_NARRATIVE_END':
            output_lines.append(f"{timestamp_col}PREV_NARRATIVE_END")

        elif token.startswith('elix_'):
            # Elixhauser comorbidities (within PREV_NARRATIVE)
            output_lines.append(f"{timestamp_col}    {token}")

        elif sequence_order == 4:
            # Demographics tokens (sex, age)
            output_lines.append(f"{timestamp_col}  {token}")

        elif token.startswith('day_'):
            # Day marker - add blank line before if not first
            if current_day is not None:
                output_lines.append("")
            current_day = day
            output_lines.append(f"{timestamp_col}  [{token}]")
            previous_was_day_marker = True

        elif token.startswith('hour_'):
            # Hour marker
            output_lines.append(f"{timestamp_col}    [{token}]")
            previous_was_day_marker = False

        elif sequence_order == 8:
            # Discharge token (disposition)
            if not previous_was_day_marker:
                output_lines.append("")  # Blank line before discharge
            output_lines.append(f"{timestamp_col}    {token}")
            previous_was_day_marker = False

        else:
            # Clinical event tokens (labs, vitals, meds, etc.)
            output_lines.append(f"{timestamp_col}      {token}")
            previous_was_day_marker = False

    return "\n".join(output_lines)


def generate_narrative_examples(
    narrative_df: pl.DataFrame,
    output_dir: str,
    logger: logging.Logger = None
):
    """
    Generate 6 representative narrative examples for comprehensive auditing.

    Creates separate text files for different clinical scenarios:
    1. No patient history
    2. With Elixhausers (comorbidities)
    3. ICU stay discharged home
    4. Expired (death)
    5. Complex case (ICU + therapies + multiple transfers)
    6. Norepinephrine (vasopressor support)

    Args:
        narrative_df: Polars DataFrame with narrative sequences
        output_dir: Output directory path
        logger: Logger instance
    """
    import pandas as pd

    # Create examples directory at root level
    examples_dir = os.path.join(output_dir, 'examples')
    os.makedirs(examples_dir, exist_ok=True)

    if logger:
        logger.info("=" * 60)
        logger.info("GENERATING NARRATIVE EXAMPLES FOR AUDIT")
        logger.info("=" * 60)
        logger.info(f"Examples directory: {examples_dir}")
        logger.info("")

    examples_generated = 0

    # Example 1: No patient history
    if logger:
        logger.info("Example 1: No patient history...")

    no_history_hosps = narrative_df.filter(
        pl.col('clif_sentence') == 'no_patient_history'
    ).select('hospitalization_id').unique()

    if len(no_history_hosps) > 0:
        hosp_id = no_history_hosps['hospitalization_id'][0]
        example_narrative = narrative_df.filter(
            pl.col('hospitalization_id') == hosp_id
        ).sort(['event_time', 'sequence_order', 'clif_sentence'])

        example_path = os.path.join(examples_dir, 'example_1_no_patient_history.txt')

        with open(example_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NARRATIVE EXAMPLE 1: NO PATIENT HISTORY\n")
            f.write("=" * 80 + "\n")
            f.write("Selection criteria: Has 'no_patient_history' token\n")
            f.write("Description: Patient with no previous hospitalization history\n")
            f.write(f"Total events: {len(example_narrative)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(format_narrative_with_timestamps(example_narrative))
            f.write("\n\n" + "=" * 80 + "\n")

        examples_generated += 1
        if logger:
            logger.info(f"  ✓ Generated example_1_no_patient_history.txt ({len(example_narrative)} events)")
    else:
        if logger:
            logger.warning(f"  ⚠ No hospitalizations found with no_patient_history token")

    # Example 2: With Elixhausers (comorbidities)
    if logger:
        logger.info("Example 2: With Elixhausers...")

    elix_hosps = narrative_df.filter(
        pl.col('clif_sentence').str.starts_with('elix_')
    ).select('hospitalization_id').unique()

    if len(elix_hosps) > 0:
        hosp_id = elix_hosps['hospitalization_id'][0]
        example_narrative = narrative_df.filter(
            pl.col('hospitalization_id') == hosp_id
        ).sort(['event_time', 'sequence_order', 'clif_sentence'])

        # Get list of elixhausers for this patient
        elix_tokens = narrative_df.filter(
            (pl.col('hospitalization_id') == hosp_id) &
            (pl.col('clif_sentence').str.starts_with('elix_'))
        ).select('clif_sentence').unique()['clif_sentence'].to_list()

        example_path = os.path.join(examples_dir, 'example_2_with_elixhausers.txt')

        with open(example_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NARRATIVE EXAMPLE 2: WITH ELIXHAUSER COMORBIDITIES\n")
            f.write("=" * 80 + "\n")
            f.write("Selection criteria: Has Elixhauser comorbidity tokens (elix_*)\n")
            f.write("Description: Patient with documented comorbidities from previous hospitalizations\n")
            f.write(f"Comorbidities ({len(elix_tokens)}): {', '.join(elix_tokens)}\n")
            f.write(f"Total events: {len(example_narrative)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(format_narrative_with_timestamps(example_narrative))
            f.write("\n\n" + "=" * 80 + "\n")

        examples_generated += 1
        if logger:
            logger.info(f"  ✓ Generated example_2_with_elixhausers.txt ({len(example_narrative)} events, {len(elix_tokens)} comorbidities)")
    else:
        if logger:
            logger.warning(f"  ⚠ No hospitalizations found with Elixhauser tokens")

    # Example 3: ICU stay + discharged home
    if logger:
        logger.info("Example 3: ICU stay discharged home...")

    icu_hosps = narrative_df.filter(
        pl.col('clif_sentence') == 'transfer_to_icu'
    ).select('hospitalization_id').unique()

    home_hosps = narrative_df.filter(
        pl.col('clif_sentence') == 'disposition_home'
    ).select('hospitalization_id').unique()

    icu_and_home = icu_hosps.join(home_hosps, on='hospitalization_id', how='inner')

    if len(icu_and_home) > 0:
        hosp_id = icu_and_home['hospitalization_id'][0]
        example_narrative = narrative_df.filter(
            pl.col('hospitalization_id') == hosp_id
        ).sort(['event_time', 'sequence_order', 'clif_sentence'])

        example_path = os.path.join(examples_dir, 'example_3_icu_discharged_home.txt')

        with open(example_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NARRATIVE EXAMPLE 3: ICU STAY DISCHARGED HOME\n")
            f.write("=" * 80 + "\n")
            f.write("Selection criteria: Has 'transfer_to_icu' AND 'disposition_home' tokens\n")
            f.write("Description: Patient admitted to ICU who recovered and was discharged home\n")
            f.write(f"Total events: {len(example_narrative)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(format_narrative_with_timestamps(example_narrative))
            f.write("\n\n" + "=" * 80 + "\n")

        examples_generated += 1
        if logger:
            logger.info(f"  ✓ Generated example_3_icu_discharged_home.txt ({len(example_narrative)} events)")
    else:
        if logger:
            logger.warning(f"  ⚠ No hospitalizations found with ICU stay + discharged home")

    # Example 4: Expired (death)
    if logger:
        logger.info("Example 4: Expired...")

    expired_hosps = narrative_df.filter(
        pl.col('clif_sentence') == 'disposition_expired'
    ).select('hospitalization_id').unique()

    if len(expired_hosps) > 0:
        hosp_id = expired_hosps['hospitalization_id'][0]
        example_narrative = narrative_df.filter(
            pl.col('hospitalization_id') == hosp_id
        ).sort(['event_time', 'sequence_order', 'clif_sentence'])

        example_path = os.path.join(examples_dir, 'example_4_expired.txt')

        with open(example_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NARRATIVE EXAMPLE 4: EXPIRED (DEATH)\n")
            f.write("=" * 80 + "\n")
            f.write("Selection criteria: Has 'disposition_expired' token\n")
            f.write("Description: Patient who died during hospitalization\n")
            f.write(f"Total events: {len(example_narrative)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(format_narrative_with_timestamps(example_narrative))
            f.write("\n\n" + "=" * 80 + "\n")

        examples_generated += 1
        if logger:
            logger.info(f"  ✓ Generated example_4_expired.txt ({len(example_narrative)} events)")
    else:
        if logger:
            logger.warning(f"  ⚠ No hospitalizations found with expired disposition")

    # Example 5: Complex case (ICU + therapies + multiple transfers)
    if logger:
        logger.info("Example 5: Complex case...")

    # Find hospitalizations with ICU AND (CRRT OR ECMO)
    icu_hosps = narrative_df.filter(
        pl.col('clif_sentence') == 'transfer_to_icu'
    ).select('hospitalization_id').unique()

    crrt_hosps = narrative_df.filter(
        pl.col('clif_sentence') == 'crrt_occurring'
    ).select('hospitalization_id').unique()

    ecmo_hosps = narrative_df.filter(
        pl.col('clif_sentence') == 'ecmo_occurring'
    ).select('hospitalization_id').unique()

    # Union CRRT and ECMO
    therapy_hosps = pl.concat([crrt_hosps, ecmo_hosps]).unique()

    # Intersection: ICU + therapy
    complex_hosps = icu_hosps.join(therapy_hosps, on='hospitalization_id', how='inner')

    if len(complex_hosps) > 0:
        # Find the one with most transfer events (most complex)
        best_hosp_id = None
        max_transfers = 0

        for hosp_id in complex_hosps['hospitalization_id'].to_list()[:10]:  # Check first 10
            transfer_count = narrative_df.filter(
                (pl.col('hospitalization_id') == hosp_id) &
                (pl.col('clif_sentence').str.starts_with('transfer_to_'))
            ).shape[0]

            if transfer_count > max_transfers:
                max_transfers = transfer_count
                best_hosp_id = hosp_id

        if best_hosp_id is not None:
            example_narrative = narrative_df.filter(
                pl.col('hospitalization_id') == best_hosp_id
            ).sort(['event_time', 'sequence_order', 'clif_sentence'])

            # Get therapy info
            has_crrt = 'crrt_occurring' in example_narrative['clif_sentence'].to_list()
            has_ecmo = 'ecmo_occurring' in example_narrative['clif_sentence'].to_list()

            example_path = os.path.join(examples_dir, 'example_5_complex_case.txt')

            with open(example_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("NARRATIVE EXAMPLE 5: COMPLEX CASE\n")
                f.write("=" * 80 + "\n")
                f.write("Selection criteria: Has ICU + (CRRT OR ECMO) + multiple transfers\n")
                f.write("Description: Complex critically ill patient with advanced therapies\n")
                therapies = []
                if has_crrt:
                    therapies.append("CRRT")
                if has_ecmo:
                    therapies.append("ECMO")
                f.write(f"Therapies: {', '.join(therapies)}\n")
                f.write(f"Number of transfers: {max_transfers}\n")
                f.write(f"Total events: {len(example_narrative)}\n")
                f.write("=" * 80 + "\n\n")
                f.write(format_narrative_with_timestamps(example_narrative))
                f.write("\n\n" + "=" * 80 + "\n")

            examples_generated += 1
            if logger:
                logger.info(f"  ✓ Generated example_5_complex_case.txt ({len(example_narrative)} events, {max_transfers} transfers)")
    else:
        if logger:
            logger.warning(f"  ⚠ No hospitalizations found with ICU + therapies")

    # Example 6: Norepinephrine (vasopressor support)
    if logger:
        logger.info("Example 6: Norepinephrine (vasopressor support)...")

    norepi_hosps = narrative_df.filter(
        pl.col('clif_sentence').str.starts_with('medications_norepinephrine_mcg_kg_min_')
    ).select('hospitalization_id').unique()

    if len(norepi_hosps) > 0:
        hosp_id = norepi_hosps['hospitalization_id'][0]
        example_narrative = narrative_df.filter(
            pl.col('hospitalization_id') == hosp_id
        ).sort(['event_time', 'sequence_order', 'clif_sentence'])

        # Get all norepinephrine doses for this hospitalization
        norepi_tokens = narrative_df.filter(
            (pl.col('hospitalization_id') == hosp_id) &
            (pl.col('clif_sentence').str.starts_with('medications_norepinephrine_'))
        ).select('clif_sentence').unique().sort('clif_sentence')['clif_sentence'].to_list()

        example_path = os.path.join(examples_dir, 'example_6_norepinephrine.txt')

        with open(example_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NARRATIVE EXAMPLE 6: NOREPINEPHRINE (VASOPRESSOR)\n")
            f.write("=" * 80 + "\n")
            f.write("Selection criteria: Has norepinephrine tokens (medications_norepinephrine_*)\n")
            f.write("Description: Patient requiring vasopressor support\n")
            f.write(f"Number of norepinephrine observations: {len(norepi_tokens)}\n")
            f.write(f"Total events: {len(example_narrative)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(format_narrative_with_timestamps(example_narrative))
            f.write("\n\n" + "=" * 80 + "\n")

        examples_generated += 1
        if logger:
            logger.info(f"  ✓ Generated example_6_norepinephrine.txt ({len(example_narrative)} events, {len(norepi_tokens)} norepi observations)")
    else:
        if logger:
            logger.warning(f"  ⚠ No hospitalizations found with norepinephrine")

    # Summary
    if logger:
        logger.info("")
        logger.info(f"✓ Generated {examples_generated}/6 narrative examples")
        if examples_generated < 6:
            logger.warning(f"⚠ {6 - examples_generated} examples could not be generated due to missing data")
        logger.info("")


def split_train_val_test(
    narrative_df: pl.DataFrame,
    cohort_path: str,
    narratives_dir: str,
    logger: logging.Logger = None
):
    """
    Split narrative sequences into train/val and test sets based on admission year.

    Train/Val: 2018-2023 data (combined in one parquet)
    Test: 2024 data (separate parquet)

    Args:
        narrative_df: Complete narrative sequences DataFrame
        cohort_path: Path to cohort.parquet for admission dates
        narratives_dir: Directory to write split parquet files
        logger: Logger instance
    """
    if logger:
        logger.info("=" * 60)
        logger.info("Splitting data into train/val and test sets...")
        logger.info("=" * 60)

    # Read cohort to get admission dates
    cohort_df = pl.read_parquet(cohort_path).select(['hospitalization_id', 'admission_dttm'])

    if logger:
        logger.info(f"Loaded cohort: {len(cohort_df):,} hospitalizations")

    # Extract year from admission_dttm
    cohort_df = cohort_df.with_columns(
        pl.col('admission_dttm').dt.year().alias('admission_year')
    )

    # Split by year
    test_hosp_ids = cohort_df.filter(pl.col('admission_year') == 2024).select('hospitalization_id')
    train_val_hosp_ids = cohort_df.filter(pl.col('admission_year') < 2024).select('hospitalization_id')

    if logger:
        logger.info(f"  Train/Val hospitalizations (2018-2023): {len(train_val_hosp_ids):,}")
        logger.info(f"  Test hospitalizations (2024): {len(test_hosp_ids):,}")

    # Filter narrative sequences
    train_val_df = narrative_df.join(train_val_hosp_ids, on='hospitalization_id', how='inner')
    test_df = narrative_df.join(test_hosp_ids, on='hospitalization_id', how='inner')

    if logger:
        logger.info(f"  Train/Val narrative events: {len(train_val_df):,}")
        logger.info(f"  Test narrative events: {len(test_df):,}")

    # Validation: check no overlap
    train_val_ids = set(train_val_df['hospitalization_id'].unique().to_list())
    test_ids = set(test_df['hospitalization_id'].unique().to_list())
    overlap = train_val_ids & test_ids

    if len(overlap) > 0:
        if logger:
            logger.error(f"  ✗ ERROR: {len(overlap)} hospitalizations overlap between train/val and test!")
        raise ValueError(f"Data leakage detected: {len(overlap)} hospitalizations in both sets")

    if logger:
        logger.info(f"  ✓ Validation passed: No overlap between train/val and test sets")

    # Write parquet files
    train_val_path = os.path.join(narratives_dir, 'train_val_sequences.parquet')
    test_path = os.path.join(narratives_dir, 'test_sequences.parquet')

    train_val_df.write_parquet(train_val_path)
    test_df.write_parquet(test_path)

    train_val_size_mb = os.path.getsize(train_val_path) / (1024**2)
    test_size_mb = os.path.getsize(test_path) / (1024**2)

    if logger:
        logger.info("")
        logger.info(f"  ✓ Saved train_val_sequences.parquet ({train_val_size_mb:.2f} MB)")
        logger.info(f"  ✓ Saved test_sequences.parquet ({test_size_mb:.2f} MB)")
        logger.info("")


def run_narrative_assembly(config_path: str) -> pl.DataFrame:
    """
    Main narrative assembly pipeline.

    Reads tokenized parquets from output_dir and assembles chronological narratives.

    Args:
        config_path: Path to clif_config.json

    Returns:
        Polars DataFrame with narrative sequences
    """
    # Load configuration
    config = load_config(config_path)
    output_dir = config['output_dir']

    # Setup logger
    logger = setup_logger(output_dir)

    # Print banner
    logger.info("🚀" * 30)
    logger.info("NARRATIVE ASSEMBLY PIPELINE - STEP 2")
    logger.info("🚀" * 30)
    logger.info("")
    logger.info("=" * 60)
    logger.info("Configuration")
    logger.info("=" * 60)
    logger.info(f"Config file: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {os.path.join(output_dir, 'narrative_assembly.log')}")
    logger.info("")

    # Verify tokenized parquets exist in tokentables subdirectory
    tokentables_dir = os.path.join(output_dir, 'tokentables')

    logger.info("=" * 60)
    logger.info("Checking for tokenized parquets...")
    logger.info("=" * 60)
    logger.info(f"Token tables directory: {tokentables_dir}")
    logger.info("")

    required_files = [
        'cohort.parquet',
        'adt.parquet',
        'assessment.parquet',
        'crrt_therapy.parquet',
        'ecmo_mcs.parquet',
        'medication_admin_continuous.parquet',
        'labs.parquet',
        'vitals.parquet',
        'respiratory_support.parquet'
    ]

    missing_files = []
    for filename in required_files:
        filepath = os.path.join(tokentables_dir, filename)
        if os.path.exists(filepath):
            file_size_mb = os.path.getsize(filepath) / (1024**2)
            logger.info(f"  ✓ {filename} ({file_size_mb:.1f} MB)")
        else:
            missing_files.append(filename)
            logger.error(f"  ✗ {filename} NOT FOUND")

    if missing_files:
        logger.error("")
        logger.error("=" * 60)
        logger.error("ERROR: Missing tokenized parquet files!")
        logger.error("=" * 60)
        logger.error("Please run tokenization first:")
        logger.error("  uv run tokenETL/main.py")
        logger.error("")
        logger.error(f"Missing files in {tokentables_dir}: {missing_files}")
        raise FileNotFoundError(f"Missing tokenized parquets: {missing_files}")

    logger.info("")
    logger.info("✓ All required tokenized parquets found")
    logger.info("")

    # Build narrative sequences from tokentables
    cohort_path = os.path.join(tokentables_dir, 'cohort.parquet')

    narrative_df, narrative_token_counts = build_narrative_sequences(
        cohort_path=cohort_path,
        parquet_dir=tokentables_dir,
        logger=logger
    )

    # Save token counts
    logger.info("")
    logger.info("=" * 60)
    logger.info("Saving token counts...")
    logger.info("=" * 60)

    # Save special tokens (special, time_marker, demographics)
    special_tokens = narrative_token_counts[
        narrative_token_counts['source'].isin(['special', 'time_marker', 'demographics'])
    ].copy()

    special_path = os.path.join(output_dir, 'special_token_counts.csv')
    special_tokens.to_csv(special_path, index=False)
    logger.info(f"  ✓ Saved special_token_counts.csv")
    logger.info(f"    - Special tokens: {len(special_tokens):,}")
    logger.info(f"    - Occurrences: {special_tokens['count'].sum():,}")

    # Save comprehensive narrative token counts (all tokens)
    narrative_counts_path = os.path.join(output_dir, 'narrative_token_counts.csv')
    narrative_token_counts.to_csv(narrative_counts_path, index=False)
    logger.info(f"  ✓ Saved narrative_token_counts.csv")
    logger.info(f"    - Total unique tokens: {len(narrative_token_counts):,}")
    logger.info(f"    - Total occurrences: {narrative_token_counts['count'].sum():,}")
    logger.info("")

    # Save narrative sequences to narratives subdirectory
    logger.info("")
    logger.info("=" * 60)
    logger.info("Saving narrative sequences...")
    logger.info("=" * 60)

    # Create narratives subdirectory
    narratives_dir = os.path.join(output_dir, 'narratives')
    os.makedirs(narratives_dir, exist_ok=True)
    logger.info(f"Narratives directory: {narratives_dir}")
    logger.info("")

    narrative_path = os.path.join(narratives_dir, 'narrative_sequences.parquet')
    narrative_df.write_parquet(narrative_path)
    file_size_mb = os.path.getsize(narrative_path) / (1024**2)

    logger.info(f"  ✓ Saved narratives/narrative_sequences.parquet ({file_size_mb:.2f} MB)")
    logger.info(f"  ✓ Total narrative rows: {len(narrative_df):,}")
    logger.info("")

    # Generate 6 narrative examples for comprehensive auditing
    generate_narrative_examples(
        narrative_df=narrative_df,
        output_dir=output_dir,
        logger=logger
    )

    # Split data into train/val and test sets
    split_train_val_test(
        narrative_df=narrative_df,
        cohort_path=cohort_path,
        narratives_dir=narratives_dir,
        logger=logger
    )

    # Pipeline complete
    logger.info("=" * 60)
    logger.info("NARRATIVE ASSEMBLY COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Narrative sequences: {narrative_path}")
    logger.info(f"Total rows: {len(narrative_df):,}")
    logger.info("")

    # Count statistics
    total_hosps = narrative_df.select('hospitalization_id').n_unique()
    avg_rows = len(narrative_df) / total_hosps if total_hosps > 0 else 0

    logger.info(f"Statistics:")
    logger.info(f"  Hospitalizations: {total_hosps:,}")
    logger.info(f"  Average events per hospitalization: {avg_rows:.1f}")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info("")
    logger.info("=" * 60)
    logger.info("Output files:")
    logger.info(f"  - narratives/narrative_sequences.parquet (main output)")
    logger.info(f"  - narratives/train_val_sequences.parquet (2018-2023 data)")
    logger.info(f"  - narratives/test_sequences.parquet (2024 data)")
    logger.info(f"  - examples/ (6 comprehensive examples for audit)")
    logger.info(f"  - token_summary_statistics.csv (token distribution statistics)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Review examples/ folder for diverse clinical scenarios")
    logger.info("  - Review token_summary_statistics.csv for distribution analysis")
    logger.info("  - Validate narrative quality and token coverage")
    logger.info("  - Use train_val_sequences.parquet and test_sequences.parquet for model training")
    logger.info("=" * 60)

    return narrative_df


def main():
    """CLI entry point with argument parsing."""
    # Default config path: repo root / clif_config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)  # Go up one level from tokenETL/
    default_config = os.path.join(repo_root, 'clif_config.json')

    parser = argparse.ArgumentParser(
        description='Narrative Assembly Pipeline - Step 2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  uv run tokenETL/assemble_narratives.py
  uv run tokenETL/assemble_narratives.py --config /path/to/custom_config.json

This script should be run AFTER tokenization (main.py) completes.

It reads tokenized parquets from the output_dir specified in clif_config.json
and assembles chronological clinical narratives.

Outputs:
  - narrative_sequences.parquet: Main output file
  - narrative_example.txt: Sample narrative for review
  - narrative_assembly.log: Execution log

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
        # Run narrative assembly
        narrative_df = run_narrative_assembly(config_path=args.config)

        print(f"\n✅ Narrative assembly completed successfully!")
        print(f"Total narrative rows: {len(narrative_df):,}")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure tokenization has been completed first:")
        print("  uv run tokenETL/main.py")
        exit(1)

    except Exception as e:
        print(f"\n❌ Error running narrative assembly: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()
