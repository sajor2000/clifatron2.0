"""
Narrative Assembler Module

Assembles chronological clinical narratives from tokenized parquet files.
Uses polars for high-performance data processing (10-100x faster than pandas).

Creates narrative sequences with:
- Special tokens: PREV_NARRATIVE_START, PREV_NARRATIVE_END
- Demographics tokens
- Day/hour markers (day_1, hour_1, etc.)
- Chronological clinical events sorted by datetime
"""

import os
import polars as pl
import pandas as pd
import logging
from typing import Optional, Tuple


def load_and_standardize_table(
    parquet_path: str,
    dttm_col: str,
    token_cols: list,
    hosp_ids: pl.Series,
    logger: logging.Logger = None
) -> pl.DataFrame:
    """
    Load parquet with polars, filter to cohort, standardize to
    (hospitalization_id, event_time, clif_sentence).

    For multi-token tables (respiratory), explode token columns to rows.

    Args:
        parquet_path: Path to parquet file
        dttm_col: Name of datetime column
        token_cols: List of token column names
        hosp_ids: Series of hospitalization IDs to filter to
        logger: Logger instance

    Returns:
        Standardized polars DataFrame with columns:
        - hospitalization_id
        - event_time
        - clif_sentence
    """
    if not os.path.exists(parquet_path):
        if logger:
            logger.warning(f"  ⚠ File not found: {parquet_path}, skipping")
        return pl.DataFrame({
            'hospitalization_id': [],
            'event_time': [],
            'clif_sentence': []
        })

    if logger:
        table_name = os.path.basename(parquet_path)
        logger.info(f"  Loading {table_name}...")

    # Load parquet
    df = pl.read_parquet(parquet_path)

    # Filter to cohort hospitalizations
    df = df.filter(pl.col('hospitalization_id').is_in(hosp_ids))

    initial_rows = len(df)

    # Handle single vs multi-token tables
    if len(token_cols) == 1:
        # Single token column (narrow tables: labs, vitals, etc.)
        # Filter: remove rows where token is null, 'NA', or 'NULL'
        result = df.select([
            'hospitalization_id',
            pl.col(dttm_col).alias('event_time'),
            pl.col(token_cols[0]).alias('clif_sentence')
        ]).filter(
            pl.col('clif_sentence').is_not_null() &
            (pl.col('clif_sentence') != 'NA') &
            (pl.col('clif_sentence') != 'NULL')
        )
    else:
        # Multi-token columns (wide tables: respiratory_support with 17+ columns)
        # Step 1: Filter out rows where timestamp is null BEFORE melting
        # This prevents row explosion from melting rows with no valid timestamp
        df = df.filter(pl.col(dttm_col).is_not_null())

        # Step 2: Melt from wide to long format
        # Each row becomes N rows (one per token column)
        result = df.melt(
            id_vars=['hospitalization_id', dttm_col],
            value_vars=token_cols,
            value_name='clif_sentence'
        ).filter(
            # Step 3: Filter out null/NA/NULL token values after melting
            pl.col('clif_sentence').is_not_null() &
            (pl.col('clif_sentence') != 'NA') &
            (pl.col('clif_sentence') != 'NULL')
        ).select([
            'hospitalization_id',
            pl.col(dttm_col).alias('event_time'),
            'clif_sentence'
        ])

    if logger:
        logger.info(f"    ✓ Loaded {initial_rows:,} rows → {len(result):,} events after filtering nulls and NA")

    return result


def calculate_day_hour(
    events_df: pl.DataFrame,
    cohort_df: pl.DataFrame,
    logger: logging.Logger = None
) -> pl.DataFrame:
    """
    Calculate day and hour for each event relative to first event in hospitalization.

    Day: (event_time - first_event_time).days + 1, capped at '30+'
    Hour: event_time.hour + 1 (1-24 instead of 0-23)

    Uses the FIRST token's event_time as day 1 baseline (not admission_dttm).
    This allows pre-admission events to be included correctly.

    Args:
        events_df: Events DataFrame with hospitalization_id, event_time, clif_sentence
        cohort_df: Cohort DataFrame (not used, kept for backward compatibility)
        logger: Logger instance

    Returns:
        Events DataFrame with added day and hour columns
    """
    if logger:
        logger.info("  Calculating day and hour for events...")

    # Calculate first event time per hospitalization
    first_event_times = events_df.group_by('hospitalization_id').agg([
        pl.col('event_time').min().alias('first_event_time')
    ])

    # Join events with first event times
    events_with_baseline = events_df.join(
        first_event_times,
        on='hospitalization_id',
        how='left'
    )

    # Calculate day and hour
    result = events_with_baseline.with_columns([
        # Day: calendar days since first event + 1, capped at 30
        (
            (pl.col('event_time').dt.date() - pl.col('first_event_time').dt.date()).dt.total_days() + 1
        ).clip(1, 30).alias('day_num')
    ]).with_columns([
        # Convert day to string, replace 30 with '30+'
        pl.when(pl.col('day_num') >= 30)
            .then(pl.lit('30+'))
            .otherwise(pl.col('day_num').cast(pl.Utf8))
            .alias('day'),
        # Hour: 1-24 (not 0-23)
        (pl.col('event_time').dt.hour() + 1).cast(pl.Utf8).alias('hour')
    ]).select([
        'hospitalization_id',
        'event_time',
        'day',
        'hour',
        'clif_sentence'
    ])

    if logger:
        logger.info(f"    ✓ Calculated day/hour for {len(result):,} events")

    return result


def add_day_hour_markers(
    events_df: pl.DataFrame,
    logger: logging.Logger = None
) -> pl.DataFrame:
    """
    Insert day_X and hour_Y marker tokens before events in each time period.

    Day markers appear ONCE per day (at first event of that day).
    Hour markers appear ONCE per hour (at first event of that hour).

    Args:
        events_df: Events DataFrame with day and hour columns
        logger: Logger instance

    Returns:
        Events DataFrame with day/hour marker rows inserted
    """
    if logger:
        logger.info("  Adding day and hour marker tokens...")

    # Create day markers (one per day, at first event of that day)
    # Group by (hospitalization_id, day) and get minimum event_time
    day_markers = events_df.group_by(['hospitalization_id', 'day']).agg([
        pl.col('event_time').min().alias('event_time')
    ]).with_columns([
        # Recalculate hour from the min event_time (1-24 format)
        (pl.col('event_time').dt.hour() + 1).cast(pl.Utf8).alias('hour'),
        pl.lit(5, dtype=pl.Int64).alias('sequence_order'),
        (pl.lit('day_') + pl.col('day')).alias('clif_sentence')
    ]).select([
        'hospitalization_id',
        'event_time',
        'day',
        'hour',
        'sequence_order',
        'clif_sentence'
    ])

    # Create hour markers (one per hour, at first event of that hour)
    # Group by (hospitalization_id, day, hour) and get minimum event_time
    hour_markers = events_df.group_by(['hospitalization_id', 'day', 'hour']).agg([
        pl.col('event_time').min().alias('event_time')
    ]).with_columns([
        pl.lit(6, dtype=pl.Int64).alias('sequence_order'),
        (pl.lit('hour_') + pl.col('hour')).alias('clif_sentence')
    ]).select([
        'hospitalization_id',
        'event_time',
        'day',
        'hour',
        'sequence_order',
        'clif_sentence'
    ])

    # Add sequence_order = 7 to events (clinical events)
    # Reorder columns to match day_markers and hour_markers
    events_with_order = events_df.with_columns([
        pl.lit(7, dtype=pl.Int64).alias('sequence_order')
    ]).select([
        'hospitalization_id',
        'event_time',
        'day',
        'hour',
        'sequence_order',
        'clif_sentence'
    ])

    # Concatenate markers with events (all have same column order now)
    result = pl.concat([day_markers, hour_markers, events_with_order]).sort([
        'hospitalization_id',
        'event_time',
        'sequence_order',
        'clif_sentence'
    ])

    if logger:
        day_count = len(day_markers)
        hour_count = len(hour_markers)
        logger.info(f"    ✓ Added {day_count:,} day markers and {hour_count:,} hour markers")

    return result


def add_special_tokens(
    events_df: pl.DataFrame,
    cohort_df: pl.DataFrame,
    logger: logging.Logger = None
) -> pl.DataFrame:
    """
    Add special tokens for each hospitalization:
    - PREV_NARRATIVE_START
    - Elixhauser comorbidity tokens (split from prev_hosp_comorbidities)
    - PREV_NARRATIVE_END
    - Demographics tokens (sex_category_token, age_at_admission_token)
    - [chronological events]
    - discharge_category_token (with event_time = discharge_dttm)

    Args:
        events_df: Events DataFrame with all clinical events
        cohort_df: Cohort DataFrame with demographics and tokens
        logger: Logger instance

    Returns:
        Complete narrative DataFrame with all special tokens
    """
    if logger:
        logger.info("  Adding special tokens (demographics, previous narrative, etc.)...")

    special_rows = []

    # Process each hospitalization
    for row in cohort_df.iter_rows(named=True):
        hosp_id = row['hospitalization_id']

        # 1. PREV_NARRATIVE_START (sequence_order = 1)
        special_rows.append({
            'hospitalization_id': hosp_id,
            'event_time': None,
            'day': None,
            'hour': None,
            'sequence_order': 1,
            'clif_sentence': 'PREV_NARRATIVE_START'
        })

        # 2. Elixhauser comorbidity tokens (split on |) (sequence_order = 2)
        if row.get('prev_hosp_comorbidities'):
            elix_tokens = str(row['prev_hosp_comorbidities']).split('|')
            for token in elix_tokens:
                if token.strip() and token.strip() != 'NULL':  # Skip empty strings and 'NULL'
                    special_rows.append({
                        'hospitalization_id': hosp_id,
                        'event_time': None,
                        'day': None,
                        'hour': None,
                        'sequence_order': 2,
                        'clif_sentence': token.strip()
                    })

        # 3. PREV_NARRATIVE_END (sequence_order = 3)
        special_rows.append({
            'hospitalization_id': hosp_id,
            'event_time': None,
            'day': None,
            'hour': None,
            'sequence_order': 3,
            'clif_sentence': 'PREV_NARRATIVE_END'
        })

        # 4. Demographics tokens (sequence_order = 4)
        if row.get('sex_category_token') and row['sex_category_token'] not in ['NULL', None]:
            special_rows.append({
                'hospitalization_id': hosp_id,
                'event_time': None,
                'day': None,
                'hour': None,
                'sequence_order': 4,
                'clif_sentence': row['sex_category_token']
            })

        if row.get('age_at_admission_token') and row['age_at_admission_token'] not in ['NULL', None]:
            special_rows.append({
                'hospitalization_id': hosp_id,
                'event_time': None,
                'day': None,
                'hour': None,
                'sequence_order': 4,
                'clif_sentence': row['age_at_admission_token']
            })

        # 5. discharge_category_token (with event_time = discharge_dttm) (sequence_order = 8)
        # Note: day/hour will be calculated later when this gets merged with events
        # We set day/hour to None here and let the natural flow calculate it
        if row.get('discharge_category_token') and row['discharge_category_token'] not in ['NULL', None] and row.get('discharge_dttm'):
            special_rows.append({
                'hospitalization_id': hosp_id,
                'event_time': row['discharge_dttm'],
                'day': None,  # Will be calculated based on first_event_time
                'hour': None,  # Will be calculated based on first_event_time
                'sequence_order': 8,
                'clif_sentence': row['discharge_category_token']
            })

    # Convert special rows to polars DataFrame
    special_df = pl.DataFrame(special_rows)

    # Cast day and hour to String to match events_df schema
    special_df = special_df.with_columns([
        pl.col('day').cast(pl.Utf8),
        pl.col('hour').cast(pl.Utf8)
    ])

    # Concatenate with events and sort
    # Sort order:
    # 1. hospitalization_id (group by hospitalization)
    # 2. event_time (chronological, with nulls first for special tokens)
    # 3. sequence_order (1-8: PREV_NARRATIVE → demographics → events → discharge)
    # 4. clif_sentence (alphabetical within same sequence_order)
    result = pl.concat([special_df, events_df]).sort([
        'hospitalization_id',
        pl.col('event_time').fill_null(pl.datetime(1900, 1, 1)),  # Put nulls first
        'sequence_order',
        'clif_sentence'
    ])

    # Recalculate day/hour for discharge token using first_event_time baseline
    # This ensures they use the same baseline as all other events
    if logger:
        logger.info("  Recalculating day/hour for discharge token...")

    # Get first event time per hospitalization from events that have event_time
    first_event_times = events_df.filter(
        pl.col('event_time').is_not_null()
    ).group_by('hospitalization_id').agg([
        pl.col('event_time').min().alias('first_event_time')
    ])

    # Join result with first_event_times to recalculate day/hour for tokens with None
    result_with_baseline = result.join(
        first_event_times,
        on='hospitalization_id',
        how='left'
    )

    # Recalculate day and hour for rows where they are None and event_time is not None
    result = result_with_baseline.with_columns([
        # Calculate day_num for rows with None day
        pl.when(pl.col('day').is_null() & pl.col('event_time').is_not_null() & pl.col('first_event_time').is_not_null())
            .then(
                (pl.col('event_time').dt.date() - pl.col('first_event_time').dt.date()).dt.total_days() + 1
            )
            .otherwise(
                pl.when(pl.col('day') == '30+').then(pl.lit(30))
                .otherwise(pl.col('day').cast(pl.Int64, strict=False))
            )
            .clip(1, 30)
            .alias('day_num')
    ]).with_columns([
        # Update day string
        pl.when(pl.col('day_num') >= 30)
            .then(pl.lit('30+'))
            .otherwise(pl.col('day_num').cast(pl.Utf8))
            .alias('day'),
        # Update hour for rows with None hour
        pl.when(pl.col('hour').is_null() & pl.col('event_time').is_not_null())
            .then((pl.col('event_time').dt.hour() + 1).cast(pl.Utf8))
            .otherwise(pl.col('hour'))
            .alias('hour')
    ]).select([
        'hospitalization_id',
        'event_time',
        'day',
        'hour',
        'sequence_order',
        'clif_sentence'
    ])

    # Re-sort after recalculation
    result = result.sort([
        'hospitalization_id',
        pl.col('event_time').fill_null(pl.datetime(1900, 1, 1)),
        'sequence_order',
        'clif_sentence'
    ])

    if logger:
        logger.info(f"    ✓ Added {len(special_df):,} special token rows")
        logger.info(f"    ✓ Total narrative rows: {len(result):,}")

    return result


def generate_token_statistics(
    final_narrative: pl.DataFrame,
    output_path: str,
    logger: logging.Logger = None
) -> None:
    """
    Generate token count statistics per hospitalization without PHI.

    Creates distribution table with token count ranges and summary statistics.
    NO hospitalization IDs are included (PHI protection).

    Args:
        final_narrative: Complete narrative DataFrame
        output_path: Path to save CSV file
        logger: Logger instance
    """
    if logger:
        logger.info("")
        logger.info("Generating token summary statistics...")

    # Count tokens per hospitalization
    tokens_per_hosp = final_narrative.group_by('hospitalization_id').agg([
        pl.count().alias('token_count')
    ])

    # Calculate summary statistics
    total_hosps = len(tokens_per_hosp)
    avg_tokens = tokens_per_hosp['token_count'].mean()
    min_tokens = tokens_per_hosp['token_count'].min()
    max_tokens = tokens_per_hosp['token_count'].max()
    median_tokens = tokens_per_hosp['token_count'].median()

    # Create distribution bins
    distribution = tokens_per_hosp.with_columns([
        pl.when(pl.col('token_count') <= 100).then(pl.lit('0-100'))
        .when(pl.col('token_count') <= 500).then(pl.lit('101-500'))
        .when(pl.col('token_count') <= 1000).then(pl.lit('501-1000'))
        .when(pl.col('token_count') <= 2000).then(pl.lit('1001-2000'))
        .when(pl.col('token_count') <= 5000).then(pl.lit('2001-5000'))
        .otherwise(pl.lit('5000+')).alias('token_range')
    ]).group_by('token_range').agg([
        pl.count().alias('num_hospitalizations')
    ]).with_columns([
        (pl.col('num_hospitalizations') / total_hosps * 100).round(2).alias('percentage')
    ])

    # Sort by token range order
    range_order = {'0-100': 0, '101-500': 1, '501-1000': 2, '1001-2000': 3, '2001-5000': 4, '5000+': 5}
    distribution = distribution.sort(
        by=pl.col('token_range').map_elements(lambda x: range_order.get(x, 999), return_dtype=pl.Int32)
    )

    # Convert to pandas for CSV writing
    dist_pd = distribution.to_pandas()

    # Write to CSV
    with open(output_path, 'w') as f:
        f.write("TOKEN SUMMARY STATISTICS\n")
        f.write(f"Total Hospitalizations: {total_hosps:,}\n")
        f.write(f"Average Tokens per Hospitalization: {avg_tokens:.1f}\n")
        f.write(f"Minimum Tokens: {min_tokens:,}\n")
        f.write(f"Maximum Tokens: {max_tokens:,}\n")
        f.write(f"Median Tokens: {median_tokens:.1f}\n")
        f.write("\n")
        f.write("DISTRIBUTION TABLE\n")
        dist_pd.to_csv(f, index=False)

    if logger:
        logger.info(f"  ✓ Token statistics saved to: {output_path}")
        logger.info(f"    Total hospitalizations: {total_hosps:,}")
        logger.info(f"    Avg tokens/hosp: {avg_tokens:.1f}")
        logger.info(f"    Range: {min_tokens:,} - {max_tokens:,}")


def build_narrative_sequences(
    cohort_path: str,
    parquet_dir: str,
    logger: logging.Logger = None
) -> Tuple[pl.DataFrame, pd.DataFrame]:
    """
    Build chronological narrative sequences from all tokenized parquet files.

    Process:
    1. Load cohort
    2. Load and standardize all event tables (labs, vitals, respiratory, etc.)
    3. Concatenate all events
    4. Calculate day/hour for each event
    5. Add day/hour marker tokens
    6. Add special tokens (HOSP_START, demographics, etc.)
    7. Sort chronologically
    8. Count all tokens with source categorization

    Args:
        cohort_path: Path to cohort.parquet
        parquet_dir: Directory containing all parquet files
        logger: Logger instance

    Returns:
        Tuple of:
        - Polars DataFrame with narrative sequences (columns:
          - hospitalization_id
          - event_time (nullable datetime)
          - day (nullable string: '1', '2', ..., '30+', or null)
          - hour (nullable string: '1'-'24', or null)
          - clif_sentence (string token))
        - Pandas DataFrame with token counts (columns:
          - token (string)
          - count (int)
          - source (string: special, time_marker, demographics, vitals, labs, etc.))
    """
    if logger:
        logger.info("=" * 60)
        logger.info("PHASE 10: NARRATIVE ASSEMBLY")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Building chronological narrative sequences from tokenized data...")

    # Load cohort
    if logger:
        logger.info("")
        logger.info("Loading cohort...")

    cohort_df = pl.read_parquet(cohort_path)
    hosp_ids = cohort_df.select('hospitalization_id')['hospitalization_id']

    if logger:
        logger.info(f"  ✓ Loaded cohort: {len(cohort_df):,} hospitalizations")

    # Define table configurations
    # (parquet_file, datetime_column, token_columns)
    tables_config = [
        ('labs.parquet', 'lab_result_dttm', ['labs_token']),
        ('vitals.parquet', 'recorded_dttm', ['vitals_token']),
        ('assessment.parquet', 'recorded_dttm', ['assessment_token']),
        ('medication_admin_continuous.parquet', 'admin_dttm', ['medication_dose_token']),
        ('adt.parquet', 'in_dttm', ['location_category_token']),
        ('crrt_therapy.parquet', 'recorded_dttm', ['crrt_token']),
        ('ecmo_mcs.parquet', 'recorded_dttm', ['ecmo_token']),
        ('respiratory_support.parquet', 'recorded_dttm', [
            'device_category', 'mode_category', 'tracheostomy',
            'fio2_set', 'lpm_set', 'tidal_volume_set', 'resp_rate_set',
            'pressure_control_set', 'pressure_support_set', 'flow_rate_set',
            'peak_inspiratory_pressure_set', 'inspiratory_time_set', 'peep_set',
            'tidal_volume_obs', 'resp_rate_obs', 'plateau_pressure_obs',
            'peak_inspiratory_pressure_obs', 'peep_obs', 'minute_vent_obs',
            'mean_airway_pressure_obs'
        ])
    ]

    # Load and standardize all tables
    if logger:
        logger.info("")
        logger.info("Loading and standardizing event tables...")

    all_events = []
    for parquet_file, dttm_col, token_cols in tables_config:
        parquet_path = os.path.join(parquet_dir, parquet_file)
        events = load_and_standardize_table(
            parquet_path, dttm_col, token_cols, hosp_ids, logger
        )
        if len(events) > 0:
            all_events.append(events)

    # Concatenate all events
    if logger:
        logger.info("")
        logger.info("Concatenating all events...")

    combined_events = pl.concat(all_events)

    if logger:
        logger.info(f"  ✓ Total events: {len(combined_events):,}")

    # Calculate day and hour
    if logger:
        logger.info("")

    events_with_time = calculate_day_hour(combined_events, cohort_df, logger)

    # Add day/hour markers
    if logger:
        logger.info("")

    events_with_markers = add_day_hour_markers(events_with_time, logger)

    # Add special tokens
    if logger:
        logger.info("")

    final_narrative = add_special_tokens(events_with_markers, cohort_df, logger)

    # Count all tokens in narratives (including special tokens)
    if logger:
        logger.info("")
        logger.info("Counting all tokens in narratives...")

    token_counts = final_narrative.group_by('clif_sentence').agg([
        pl.count().alias('count')
    ]).to_pandas()

    # Categorize tokens by source
    def categorize_token(token):
        """Determine source category for a token."""
        # Handle null/None tokens
        if token is None or pd.isna(token):
            return 'null'

        # Special control tokens
        if token in ['PREV_NARRATIVE_START', 'PREV_NARRATIVE_END']:
            return 'special'

        # Time markers
        elif token.startswith('day_'):
            return 'time_marker'
        elif token.startswith('hour_'):
            return 'time_marker'

        # Demographics (sex, age, disposition from hospitalization/patient tables)
        elif token.startswith('sex_'):  # sex_male, sex_female
            return 'demographics'
        elif token.startswith('age_'):  # age_18_25, age_66_75, age_86_plus
            return 'demographics'
        elif token.startswith('disposition_'):  # disposition_home, disposition_expired, etc.
            return 'demographics'
        elif token == 'no_patient_history':  # Special demographics token
            return 'demographics'

        # Elixhauser comorbidities
        elif token.startswith('elix_'):
            return 'elixhauser'

        # ADT/Location tokens (transfers)
        elif token.startswith('transfer_to_'):  # transfer_to_icu, transfer_to_ward, etc.
            return 'cohort_adt'
        elif token.startswith('location_'):  # location tokens (if any)
            return 'cohort_adt'

        # Clinical event tokens - infer from prefix
        elif token.startswith('vitals_') or token.startswith('vital_'):
            return 'vitals'
        elif token.startswith('labs_') or token.startswith('lab_'):
            return 'labs'
        elif token.startswith('assessment_'):
            return 'assessment'
        elif token.startswith('medications_') or token.startswith('medication_') or token.startswith('med_'):
            return 'medication_admin_continuous'
        elif token.startswith('respiratory_support_'):  # New interval-based tokens: respiratory_support_fio2_set_...
            return 'respiratory_support'
        elif token.startswith('resp_') or token.startswith('respiratory_'):  # Legacy tokens: resp_device_imv, respiratory_...
            return 'respiratory_support'
        elif token.startswith('tracheostomy_'):  # tracheostomy_present
            return 'respiratory_support'
        elif token.startswith('crrt_'):
            return 'crrt_therapy'
        elif token.startswith('ecmo_'):
            return 'ecmo_mcs'
        else:
            return 'other'

    token_counts['source'] = token_counts['clif_sentence'].apply(categorize_token)
    token_counts = token_counts.rename(columns={'clif_sentence': 'token'})
    token_counts = token_counts[['token', 'count', 'source']].sort_values('count', ascending=False)

    if logger:
        logger.info(f"  ✓ Counted {len(token_counts):,} unique tokens")
        logger.info(f"  ✓ Total token occurrences: {token_counts['count'].sum():,}")
        logger.info("")
        logger.info("Token breakdown by source:")
        source_summary = token_counts.groupby('source')['count'].agg(['count', 'sum'])
        for source, row in source_summary.iterrows():
            logger.info(f"    {source}: {int(row['count'])} unique tokens, {int(row['sum']):,} occurrences")

    if logger:
        logger.info("")
        logger.info("Narrative assembly complete!")
        logger.info(f"  Total narrative rows: {len(final_narrative):,}")
        logger.info(f"  Hospitalizations: {len(cohort_df):,}")
        logger.info(f"  Avg events per hospitalization: {len(final_narrative) / len(cohort_df):.1f}")
        logger.info("=" * 60)

    # Generate token summary statistics
    statistics_output_path = os.path.join(
        os.path.dirname(parquet_dir),
        'token_summary_statistics.csv'
    )
    generate_token_statistics(final_narrative, statistics_output_path, logger)

    return final_narrative, token_counts
