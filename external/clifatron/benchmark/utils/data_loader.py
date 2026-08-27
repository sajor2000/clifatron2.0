"""
Data loading utilities for Task 1: Discharged Home Prediction

Provides functions to load narrative sequences, create binary labels,
and prepare data for benchmarking.
"""

import polars as pl
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkExample:
    """Single benchmark example with narrative and label."""

    sequence: List[int]  # Token IDs
    label: int  # Binary label: 1 = discharged home, 0 = other
    disposition: str  # Original disposition token
    example_id: int  # Unique identifier


class BenchmarkDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for benchmark data."""

    def __init__(
        self,
        sequences: List[List[int]],
        labels: List[int],
        dispositions: List[str],
        example_ids: List[int],
        max_length: int = 8192,
    ):
        """
        Initialize benchmark dataset.

        Args:
            sequences: List of token ID sequences
            labels: Binary labels (1 = home, 0 = other)
            dispositions: Original disposition tokens
            example_ids: Unique identifiers
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.dispositions = dispositions
        self.example_ids = example_ids
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example."""
        sequence = self.sequences[idx]

        # Truncate if needed
        if len(sequence) > self.max_length:
            sequence = sequence[-self.max_length :]

        # Handle labels - can be numeric or string
        label = self.labels[idx]
        if isinstance(label, str):
            # String labels (e.g., Task 3: "imv_on", "expired", "imv_off")
            # Don't convert to tensor, keep as string
            label_tensor = label
        else:
            # Numeric labels (e.g., Task 1, 2, 4)
            label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "sequence": torch.tensor(sequence, dtype=torch.long),
            "label": label_tensor,
            "disposition": self.dispositions[idx],
            "example_id": self.example_ids[idx],
            "length": len(sequence),
        }


def load_narratives(
    file_path: Union[str, Path],
    min_length: int = 10,
    max_length: int = 8192,
) -> pl.DataFrame:
    """
    Load narrative sequences from parquet file.

    Args:
        file_path: Path to parquet file
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        Polars DataFrame with narrative sequences
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Narrative file not found: {file_path}")

    logger.info(f"Loading narratives from {file_path}")

    # Load parquet file
    df = pl.read_parquet(file_path)

    logger.info(f"Loaded {len(df)} sequences")

    # Filter by length
    if "sequence_length" in df.columns:
        df = df.filter(
            (pl.col("sequence_length") >= min_length)
            & (pl.col("sequence_length") <= max_length)
        )
        logger.info(f"After length filtering: {len(df)} sequences")

    return df


def load_cohort(cohort_path: Union[str, Path]) -> pl.DataFrame:
    """
    Load cohort data with ICU timing information.

    Args:
        cohort_path: Path to cohort.parquet file

    Returns:
        Polars DataFrame with cohort data
    """
    cohort_path = Path(cohort_path)

    if not cohort_path.exists():
        raise FileNotFoundError(f"Cohort file not found: {cohort_path}")

    logger.info(f"Loading cohort from {cohort_path}")

    # Load cohort
    cohort_df = pl.read_parquet(cohort_path)

    logger.info(f"Loaded {len(cohort_df)} hospitalizations")

    # Check for required columns
    required_cols = ["hospitalization_id", "first_icu_24hr_completion_time"]
    missing_cols = [col for col in required_cols if col not in cohort_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns in cohort: {missing_cols}")

    return cohort_df


def truncate_sequences_at_24hr(
    narratives_df: pl.DataFrame,
    cohort_df: pl.DataFrame,
    vocab: Optional[Dict[int, str]] = None,
) -> pl.DataFrame:
    """
    Filter narratives to only include sequences from first 24hr of ICU stay.

    OPTIMIZATION: Filter cohort FIRST, select only minimal columns, then join,
    then filter sequences by event_time.

    Args:
        narratives_df: DataFrame with narrative sequences
        cohort_df: DataFrame with cohort data including first_icu_24hr_completion_time
        vocab: Vocabulary mapping (id_to_token) - OPTIONAL, not currently used

    Returns:
        DataFrame filtered to first 24hr of ICU stays
    """
    logger.info("Filtering to hospitalizations with >= 24hr ICU stay...")

    # OPTIMIZATION 1: Filter cohort FIRST before join
    cohort_filtered = cohort_df.filter(
        pl.col("first_icu_24hr_completion_time").is_not_null()
    ).select(
        ["hospitalization_id", "first_icu_24hr_completion_time"]  # Only needed columns
    )

    logger.info(
        f"Cohort: {len(cohort_filtered)} hospitalizations have ICU stay >= 24 hours"
    )

    # Check narratives have hospitalization_id
    if "hospitalization_id" not in narratives_df.columns:
        logger.warning(
            "No hospitalization_id in narratives. Cannot filter by ICU stay."
        )
        return narratives_df

    # OPTIMIZATION 2: Inner join with filtered cohort (much smaller)
    df_joined = narratives_df.join(
        cohort_filtered,
        on="hospitalization_id",
        how="inner",
    )

    logger.info(f"After join: {len(df_joined)} sequences from {len(cohort_filtered)} hospitalizations")

    # OPTIMIZATION 3: Filter sequences by event_time (only keep sequences within first 24hr)
    # IMPORTANT: Include sequences with NULL event_time (pre-hospitalization data: age, sex, history, etc.)
    if "event_time" in df_joined.columns:
        logger.info("Filtering sequences to first 24hr using event_time (including pre-hospitalization)...")
        df_filtered = df_joined.filter(
            pl.col("event_time").is_null() |
            (pl.col("event_time") <= pl.col("first_icu_24hr_completion_time"))
        )
        logger.info(f"After 24hr filtering: {len(df_filtered)} sequences")
        return df_filtered
    else:
        logger.warning("No event_time column found. Skipping sequence-level 24hr filtering.")
        return df_joined


def truncate_sequence_tokens(
    sequence: List[int],
    vocab: Dict[int, str],
    max_hours: int = 24,
) -> List[int]:
    """
    Truncate a token sequence to include only the first N hours of ICU time.

    Looks for time marker tokens (day_X, hour_Y) and truncates after the
    specified number of hours.

    Args:
        sequence: List of token IDs
        vocab: ID to token mapping
        max_hours: Maximum hours to include (default: 24)

    Returns:
        Truncated sequence
    """
    truncated = []
    current_day = 0
    current_hour = 0
    total_hours = 0

    for token_id in sequence:
        token = vocab.get(token_id, "")

        # Check for day marker
        if token.startswith("day_"):
            try:
                current_day = int(token.split("_")[1])
            except (ValueError, IndexError):
                pass

        # Check for hour marker
        if token.startswith("hour_"):
            try:
                hour = int(token.split("_")[1])
                # Calculate total hours from start
                total_hours = (current_day - 1) * 24 + hour

                # Stop if we've reached the max hours
                if total_hours > max_hours:
                    logger.debug(f"Truncated at day {current_day}, hour {hour}")
                    break

            except (ValueError, IndexError):
                pass

        truncated.append(token_id)

    return truncated


def extract_task3_label_from_sequence(
    sequence: List[int],
    vocab: Dict[int, str],
    start_hour: int = 24,
    end_hour: int = 72,
) -> str:
    """
    Extract Task 3 multi-class label from hours 24-72 window.

    Priority order:
    1. disposition_expired (death - highest priority)
    2. imv_off (if last resp_device_* is NOT imv)
    3. imv_on (if last resp_device_* IS imv)

    Args:
        sequence: Full token sequence (NOT truncated)
        vocab: ID to token mapping
        start_hour: Start of label window (default: 24)
        end_hour: End of label window (default: 72)

    Returns:
        Label string: "expired", "imv_off", or "imv_on"
    """
    # Track current time position
    current_day = 0
    current_hour = 0
    total_hours = 0

    # Track indices within the target window
    window_start_idx = None
    window_end_idx = None

    # Parse sequence to find time window
    for idx, token_id in enumerate(sequence):
        token = vocab.get(token_id, "")

        # Update day tracking
        if token.startswith("day_"):
            try:
                current_day = int(token.split("_")[1])
            except (ValueError, IndexError):
                pass

        # Update hour tracking
        if token.startswith("hour_"):
            try:
                hour = int(token.split("_")[1])
                total_hours = (current_day - 1) * 24 + hour

                # Mark window start
                if window_start_idx is None and total_hours >= start_hour:
                    window_start_idx = idx
                    logger.debug(f"Task3 window starts at idx {idx}, hour {total_hours}")

                # Mark window end
                if total_hours > end_hour:
                    window_end_idx = idx
                    logger.debug(f"Task3 window ends at idx {idx}, hour {total_hours}")
                    break

            except (ValueError, IndexError):
                pass

    # If no window found, return default
    if window_start_idx is None:
        logger.warning("No time markers found in sequence for Task 3 window")
        return "unknown"

    # Use end of sequence if window_end_idx not set
    if window_end_idx is None:
        window_end_idx = len(sequence)

    # Extract tokens from window
    window_tokens = [vocab.get(tid, "") for tid in sequence[window_start_idx:window_end_idx]]

    # Priority 1: Check for disposition_expired
    if "disposition_expired" in window_tokens:
        return "expired"

    # Priority 2/3: Find last resp_device_* token
    last_resp_device = None
    for token in reversed(window_tokens):
        if token.startswith("resp_device_"):
            last_resp_device = token
            break

    # Classify based on last respiratory device
    if last_resp_device is None:
        logger.warning(f"No resp_device found in hours {start_hour}-{end_hour} window")
        return "unknown"

    # Check if it's IMV
    if "imv" in last_resp_device.lower() and not "imv_off" in last_resp_device.lower():
        return "imv_on"
    else:
        return "imv_off"


def create_binary_labels(
    df: pl.DataFrame,
    positive_class: str = "disposition_home",
    disposition_tokens: Optional[List[str]] = None,
) -> Tuple[pl.DataFrame, Dict]:
    """
    Create binary labels for discharge home prediction.

    Args:
        df: DataFrame with narrative sequences
        positive_class: Token representing positive class (default: disposition_home)
        disposition_tokens: List of all disposition tokens to filter for

    Returns:
        Tuple of (DataFrame with labels, label statistics dict)
    """
    if disposition_tokens is None:
        disposition_tokens = [
            "disposition_home",
            "disposition_expired",
            "disposition_rehab",
            "disposition_snf",
            "disposition_hospice",
            "disposition_ltach",
            "disposition_other",
        ]

    logger.info("Creating binary labels for discharge home prediction")

    # Assuming the last token in each sequence is the disposition
    # We need to extract it from the sequence

    def extract_disposition(sequence: List[int], vocab_id_to_token: Dict[int, str]) -> str:
        """Extract disposition token from sequence."""
        # Look for disposition tokens at the end of the sequence
        for token_id in reversed(sequence):
            token = vocab_id_to_token.get(token_id, "")
            if any(disp in token for disp in disposition_tokens):
                return token
        return "unknown"

    # For now, we'll work with the assumption that the disposition is stored
    # in a separate column. If not, we'll need to modify this.

    # Check if disposition column exists
    if "disposition" not in df.columns:
        logger.warning(
            "No 'disposition' column found. Will need to extract from sequences."
        )
        # Will be handled by caller with vocabulary
        return df, {}

    # Create binary label
    df = df.with_columns(
        pl.when(pl.col("disposition") == positive_class)
        .then(1)
        .otherwise(0)
        .alias("label")
    )

    # Compute statistics
    stats = {
        "total_examples": len(df),
        "positive_examples": df.filter(pl.col("label") == 1).shape[0],
        "negative_examples": df.filter(pl.col("label") == 0).shape[0],
        "positive_ratio": df.filter(pl.col("label") == 1).shape[0] / len(df),
    }

    logger.info(f"Label statistics: {stats}")

    return df, stats


def aggregate_sequences_per_hospitalization(
    df: pl.DataFrame,
    vocab: Dict[int, str],
    disposition_tokens: List[str],
    sequence_col: str = "clif_sentence",
) -> pl.DataFrame:
    """
    Aggregate all sequences per hospitalization into single row.

    MAJOR OPTIMIZATION: Converts ~66M rows (one per sequence) to ~35K rows (one per hospitalization).
    Uses Polars vectorized operations.

    For each hospitalization:
    1. Concatenate all sequences into space-separated text (already token strings, not IDs)
    2. Extract disposition from last sequence
    3. Create one row: [hospitalization_id, clif_text, disposition]

    Args:
        df: DataFrame with sequences (one row per sequence, already filtered to 24hr)
        vocab: Vocabulary mapping token ID to token string (not used, data is already strings)
        disposition_tokens: List of disposition tokens to search for
        sequence_col: Name of column containing token strings

    Returns:
        DataFrame with one row per hospitalization
    """
    logger.info(f"Aggregating {len(df)} sequences into per-hospitalization rows...")

    # Group by hospitalization and aggregate using Polars expressions
    logger.info("Grouping sequences by hospitalization with Polars expressions...")
    df_aggregated = df.group_by("hospitalization_id").agg([
        pl.col(sequence_col).str.concat(delimiter=" ").alias("clif_text"),
        pl.col(sequence_col).last().alias("disposition"),
    ])

    logger.info(f"Grouped into {len(df_aggregated)} hospitalizations")

    # Check disposition distribution before creating labels
    disp_counts = df_aggregated["disposition"].value_counts()
    logger.info(f"Disposition distribution:\n{disp_counts}")

    return df_aggregated


def extract_disposition_from_sequences(
    sequences: List[List[int]],
    vocab: Dict[int, str],
    disposition_tokens: List[str],
) -> List[str]:
    """
    Extract disposition tokens from sequences.

    Args:
        sequences: List of token ID sequences
        vocab: Vocabulary mapping token ID to token string
        disposition_tokens: List of disposition token strings

    Returns:
        List of disposition tokens (one per sequence)
    """
    dispositions = []

    for sequence in sequences:
        # Look for disposition token at end of sequence
        found_disposition = "unknown"

        # Check last 10 tokens for disposition
        for token_id in reversed(sequence[-10:]):
            token = vocab.get(token_id, "")
            for disp_token in disposition_tokens:
                if disp_token in token:
                    found_disposition = token
                    break
            if found_disposition != "unknown":
                break

        dispositions.append(found_disposition)

    return dispositions


def calculate_hypoxic_proportion(sequences_list: List[str], icu_start_hours: float) -> float:
    """
    Calculate proportion of time in severe hypoxic respiratory failure (PEEP>8 OR FiO2>50%)
    during hours 24-72 after ICU admission.

    Logic:
    1. Clip to 24-72hr window after ICU admission
    2. Track day/hour/FiO2/PEEP tokens
    3. Carry forward last seen FiO2 and PEEP values when no charting
    4. Count hours where PEEP>8 OR FiO2>50%
    5. Return proportion = hours_in_severe_failure / total_hours_in_window

    Args:
        sequences_list: List of token strings for a hospitalization
        icu_start_hours: Hours from hospitalization start to ICU admission

    Returns:
        Proportion (0.0-1.0) of time in severe hypoxic failure, or 0.0 if unavailable
    """
    if icu_start_hours is None or icu_start_hours != icu_start_hours:  # Check for NaN
        return 0.0

    # Define window boundaries
    window_start = icu_start_hours + 24
    window_end = icu_start_hours + 72

    # Track time and respiratory values
    current_day = 0
    total_hours = 0

    # Carry-forward state (last seen values before/at 24hr)
    last_fio2 = None
    last_peep = None

    # Hour-by-hour tracking in 24-72 window
    hours_in_severe_failure = 0
    hours_tracked = set()  # Track unique hours to avoid double-counting

    for token in sequences_list:
        if token is None:
            continue

        # Update day tracking
        if token.startswith("day_"):
            try:
                current_day = int(token.split("_")[1])
            except (ValueError, IndexError):
                pass

        # Update hour tracking
        if token.startswith("hour_"):
            try:
                hour = int(token.split("_")[1])
                total_hours = (current_day - 1) * 24 + hour

                # Stop after window ends
                if total_hours > window_end:
                    break

                # In 24-72 window: check if current state is severe hypoxic failure
                if window_start <= total_hours <= window_end:
                    if total_hours not in hours_tracked:
                        # Check if in severe hypoxic failure based on carried-forward values
                        in_severe_failure = False

                        # PEEP > 8 (check both observed and set)
                        if last_peep is not None:
                            # Parse PEEP value from token
                            try:
                                if "(" in last_peep and "," in last_peep:
                                    # Extract upper bound: (X,Y] means >X to ≤Y
                                    peep_str = last_peep.split("(")[1].split("]")[0].split(",")[1]
                                    peep_val = float(peep_str)
                                    if peep_val > 8.0:
                                        in_severe_failure = True
                                elif "[" in last_peep and "," in last_peep:
                                    # Extract upper bound: [X,Y] means ≥X to ≤Y
                                    peep_str = last_peep.split("[")[1].split("]")[0].split(",")[1]
                                    peep_val = float(peep_str)
                                    if peep_val > 8.0:
                                        in_severe_failure = True
                            except (ValueError, IndexError):
                                pass

                        # FiO2 > 0.5 (50%)
                        if last_fio2 is not None:
                            try:
                                if "(" in last_fio2 and "," in last_fio2:
                                    # Extract upper bound
                                    fio2_str = last_fio2.split("(")[1].split("]")[0].split(",")[1]
                                    fio2_val = float(fio2_str)
                                    if fio2_val > 0.5:
                                        in_severe_failure = True
                                elif "[" in last_fio2 and "," in last_fio2:
                                    fio2_str = last_fio2.split("[")[1].split("]")[0].split(",")[1]
                                    fio2_val = float(fio2_str)
                                    if fio2_val > 0.5:
                                        in_severe_failure = True
                            except (ValueError, IndexError):
                                pass

                        if in_severe_failure:
                            hours_in_severe_failure += 1

                        hours_tracked.add(total_hours)

            except (ValueError, IndexError):
                pass

        # Track FiO2 tokens (set, not obs since obs doesn't exist)
        if token.startswith("respiratory_support_fio2_set"):
            # Update carry-forward value
            last_fio2 = token

            # If we're below threshold, stop carrying forward
            try:
                if "(" in token and "," in token:
                    fio2_str = token.split("(")[1].split("]")[0].split(",")[1]
                    fio2_val = float(fio2_str)
                    if fio2_val <= 0.5:
                        last_fio2 = None  # Below threshold, stop carrying forward
                elif "[" in token and "," in token:
                    fio2_str = token.split("[")[1].split("]")[0].split(",")[1]
                    fio2_val = float(fio2_str)
                    if fio2_val <= 0.5:
                        last_fio2 = None
            except (ValueError, IndexError):
                pass

        # Track PEEP tokens (both obs and set)
        if token.startswith("respiratory_support_peep_obs") or token.startswith("respiratory_support_peep_set"):
            # Update carry-forward value
            last_peep = token

            # If we're below threshold, stop carrying forward
            try:
                if "(" in token and "," in token:
                    peep_str = token.split("(")[1].split("]")[0].split(",")[1]
                    peep_val = float(peep_str)
                    if peep_val <= 8.0:
                        last_peep = None  # Below threshold, stop carrying forward
                elif "[" in token and "," in token:
                    peep_str = token.split("[")[1].split("]")[0].split(",")[1]
                    peep_val = float(peep_str)
                    if peep_val <= 8.0:
                        last_peep = None
            except (ValueError, IndexError):
                pass

    # Calculate proportion
    total_hours_in_window = len(hours_tracked)
    if total_hours_in_window == 0:
        return 0.0

    proportion = hours_in_severe_failure / total_hours_in_window
    return min(1.0, max(0.0, proportion))  # Clamp to [0, 1]


def check_imv_in_icu_0_24hr(sequences_list: List[str], icu_start_hours: float) -> bool:
    """
    Check if hospitalization had at least one IMV event during first 24hr of ICU.

    Args:
        sequences_list: List of token strings for a hospitalization
        icu_start_hours: Hours from hospitalization start to ICU admission

    Returns:
        True if at least one IMV event found in first 24hr of ICU, False otherwise
    """
    if icu_start_hours is None:
        return False

    # Check for NaN using the fact that NaN != NaN
    if icu_start_hours != icu_start_hours:
        return False

    icu_end_hours = icu_start_hours + 24
    current_day = 0
    total_hours = 0

    for token in sequences_list:
        if token is None:
            continue

        # Update day tracking
        if token.startswith("day_"):
            try:
                current_day = int(token.split("_")[1])
            except (ValueError, IndexError):
                pass

        # Update hour tracking
        if token.startswith("hour_"):
            try:
                hour = int(token.split("_")[1])
                total_hours = (current_day - 1) * 24 + hour

                # Stop checking after ICU 24hr window ends
                if total_hours > icu_end_hours:
                    break

            except (ValueError, IndexError):
                pass

        # Check for IMV event in ICU 0-24hr window
        if icu_start_hours <= total_hours <= icu_end_hours and token.startswith("resp_device_"):
            if "imv" in token.lower() and "imv_off" not in token.lower():
                return True

    return False


def prepare_benchmark_data(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    vocab: Optional[Dict[int, str]] = None,
    cohort_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict] = None,
) -> Dict:
    """
    Prepare benchmark dataset from raw narratives with 24hr ICU truncation.

    Generates 2 output files:
    1. task1_task2_disposition.parquet - Shared file for disposition tasks (Tasks 1 & 2)
    2. task3_task4_respiratory.parquet - Shared file for respiratory tasks (Tasks 3 & 4)
       - Task 3: IMV status classification (expired/imv_off/imv_on)
       - Task 4: Hypoxic failure proportion (0.0-1.0 regression target)

    Args:
        input_dir: Directory containing narrative parquet files
        output_dir: Directory to save processed data
        vocab: Vocabulary mapping - OPTIONAL, not currently needed for benchmark generation
        cohort_path: Path to cohort.parquet file (optional, required for 24hr truncation)
        config: Optional configuration dict

    Returns:
        Dictionary with data preparation statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = {}

    # Default configuration
    positive_class = config.get("positive_class", "disposition_home")
    disposition_tokens = config.get(
        "disposition_tokens",
        [
            "disposition_home",
            "disposition_expired",
            "disposition_rehab",
            "disposition_snf",
            "disposition_hospice",
            "disposition_ltach",
            "disposition_other",
        ],
    )
    min_length = config.get("min_sequence_length", 10)
    max_length = config.get("max_sequence_length", 8192)
    truncate_24hr = config.get("truncate_at_24hr_icu", True)

    stats = {}

    # Load cohort if truncation requested
    cohort_df = None
    if truncate_24hr and cohort_path:
        cohort_df = load_cohort(cohort_path)

    # Process train/val data
    train_val_file = input_dir / "train_val_sequences.parquet"
    if train_val_file.exists():
        logger.info("Processing train/val data")
        df_train_val = load_narratives(train_val_file, min_length, max_length)

        # Determine sequence column name
        sequence_col_name = None
        if "sequence" in df_train_val.columns:
            sequence_col_name = "sequence"
        elif "token_ids" in df_train_val.columns:
            sequence_col_name = "token_ids"
        elif "clif_sentence" in df_train_val.columns:
            sequence_col_name = "clif_sentence"
        else:
            raise ValueError("No sequence column found in dataframe")

        # CRITICAL: Extract disposition AND Task 3 labels BEFORE 24hr filtering
        logger.info("Extracting disposition from full sequences (before 24hr filtering)...")
        # Filter to only disposition rows FIRST, then take last
        disposition_map = (df_train_val
            .filter(pl.col(sequence_col_name).str.starts_with("disposition_"))
            .group_by("hospitalization_id")
            .agg([
                pl.col(sequence_col_name).last().alias("disposition")
            ])
        )
        logger.info(f"Extracted dispositions for {len(disposition_map)} hospitalizations with valid disposition tokens")

        # Extract Task 3 labels and Task 4 proportions from hours 24-72 window (before truncation)
        logger.info("Extracting Task 3 labels and Task 4 proportions from hours 24-72 window (before 24hr truncation)...")
        # For Task 3 & 4, we need to work with full sequences per hospitalization
        # Group all sequences per hospitalization to reconstruct full timeline
        df_full_sequences = df_train_val.group_by("hospitalization_id").agg([
            pl.col(sequence_col_name).alias("sequences_list"),
        ])

        # Extract Task 3 labels AND Task 4 proportions by processing full sequences
        # Note: This is done with token strings, not IDs
        def extract_task3_and_task4_from_tokens(sequences_list: List[str], icu_start_hours: float) -> tuple[str, float]:
            """
            Extract Task 3 label and Task 4 proportion from list of token strings.

            Task 3 Logic:
            1. Check for disposition_expired in 24-72hr window → "expired"
            2. Collect all resp_device tokens from 0-72hr
            3. Deduplicate consecutive identical tokens
            4. Check if patient was EVER on IMV during 24-72hr specifically
            5. Classify:
               - If ever on IMV in 24-72hr:
                 * Last token (after dedup) is NOT imv → "imv_off"
                 * Last token (after dedup) IS imv → "imv_on"
               - If NEVER on IMV in 24-72hr → "other_o2"

            Task 4 Logic:
            Calculate proportion of time in severe hypoxic respiratory failure (PEEP>8 OR FiO2>50%)
            during hours 24-72 after ICU admission.

            Returns:
                Tuple of (task3_label, task4_proportion)
            """
            all_tokens = sequences_list  # List of token strings

            # Track time windows
            tokens_24_72 = []  # For checking expired and IMV presence
            all_resp_devices = []  # All resp_device tokens from 0-72hr
            resp_devices_24_72 = []  # resp_device tokens specifically from 24-72hr
            current_day = 0
            total_hours = 0
            in_window_24_72 = False

            for token in all_tokens:
                # Skip None values
                if token is None:
                    continue

                # Update day tracking
                if token.startswith("day_"):
                    try:
                        current_day = int(token.split("_")[1])
                    except (ValueError, IndexError):
                        pass

                # Update hour tracking
                if token.startswith("hour_"):
                    try:
                        hour = int(token.split("_")[1])
                        total_hours = (current_day - 1) * 24 + hour

                        # Update window status
                        in_window_24_72 = 24 <= total_hours <= 72

                        # Stop collecting after 72 hours
                        if total_hours > 72:
                            break

                    except (ValueError, IndexError):
                        pass

                # Collect tokens for 24-72hr window
                if in_window_24_72:
                    tokens_24_72.append(token)

                # Collect all resp_device tokens from 0-72hr
                if token.startswith("resp_device_") and total_hours <= 72:
                    all_resp_devices.append(token)
                    # Also track which are from 24-72hr
                    if in_window_24_72:
                        resp_devices_24_72.append(token)

            # Priority 1: Check for disposition_expired in 24-72hr window
            if "disposition_expired" in tokens_24_72:
                task3_label = "expired"
            # If no resp_device tokens at all, classify as other_o2
            # (will be filtered out later when we keep only IMV patients)
            elif len(all_resp_devices) == 0:
                task3_label = "other_o2"
            else:
                # Deduplicate consecutive identical resp_device tokens
                deduped_resp_devices = []
                for token in all_resp_devices:
                    if not deduped_resp_devices or deduped_resp_devices[-1] != token:
                        deduped_resp_devices.append(token)

                # Get last resp_device token (after deduplication)
                last_resp_device = deduped_resp_devices[-1]

                # For Task 3: Check if last token is IMV or not
                # Since Task 3 only includes patients with IMV in 0-24hr ICU,
                # there are only 3 classes: expired, imv_off, imv_on
                if "imv" in last_resp_device.lower() and "imv_off" not in last_resp_device.lower():
                    task3_label = "imv_on"  # Last respiratory device is IMV
                else:
                    task3_label = "imv_off"  # Last respiratory device is NOT IMV

            # Calculate Task 4 proportion
            task4_proportion = calculate_hypoxic_proportion(sequences_list, icu_start_hours)

            return task3_label, task4_proportion

        # Calculate IMV presence in first 24hr of ICU for Task 3 filtering
        # Do this BEFORE extraction to have timing data available
        if cohort_df is not None:
            logger.info("Getting ICU timing data for Task 3 & 4 extraction (train/val)...")

            # Get cohort timing data
            cohort_timing = cohort_df.select([
                "hospitalization_id",
                "admission_dttm",
                "first_icu_start_time"
            ])

            # Calculate ICU start hours offset from hospitalization admission
            cohort_timing = cohort_timing.with_columns([
                ((pl.col("first_icu_start_time") - pl.col("admission_dttm")).dt.total_seconds() / 3600.0)
                .alias("icu_start_hours")
            ])

            # Join timing data to full sequences
            df_full_sequences_with_timing = df_full_sequences.join(
                cohort_timing.select(["hospitalization_id", "icu_start_hours"]),
                on="hospitalization_id",
                how="left"
            )
        else:
            # Default to 0 if no cohort data
            df_full_sequences_with_timing = df_full_sequences.with_columns([
                pl.lit(0.0).alias("icu_start_hours")
            ])

        # Apply extraction (this will be slow, but necessary for Task 3 & 4)
        task3_labels = []
        task4_proportions = []
        for idx, row in enumerate(df_full_sequences_with_timing.iter_rows(named=True)):
            hosp_id = row["hospitalization_id"]
            sequences_list = row["sequences_list"]
            icu_start_hours = row.get("icu_start_hours", 0.0)

            task3_label, task4_proportion = extract_task3_and_task4_from_tokens(sequences_list, icu_start_hours)
            task3_labels.append(task3_label)
            task4_proportions.append(task4_proportion)

        # Create Task 3 & 4 label/proportion map
        task3_task4_map = pl.DataFrame({
            "hospitalization_id": df_full_sequences["hospitalization_id"].to_list(),
            "task3_label": task3_labels,
            "task4_proportion": task4_proportions,
        })

        logger.info(f"Extracted Task 3 labels and Task 4 proportions for {len(task3_task4_map)} hospitalizations")
        logger.info(f"Task 3 label distribution:\n{task3_task4_map['task3_label'].value_counts()}")
        logger.info(f"Task 4 proportion stats: mean={sum(task4_proportions)/len(task4_proportions):.3f}, min={min(task4_proportions):.3f}, max={max(task4_proportions):.3f}")

        # Check for IMV presence in first 24hr of ICU for Task 3 filtering
        if cohort_df is not None:
            logger.info("Checking for IMV presence in first 24hr of ICU (train/val)...")

            # Check for IMV in first 24hr of ICU using Polars map_elements
            df_full_sequences_with_timing = df_full_sequences_with_timing.with_columns([
                pl.struct(["sequences_list", "icu_start_hours"])
                .map_elements(
                    lambda row: check_imv_in_icu_0_24hr(row["sequences_list"], row["icu_start_hours"]),
                    return_dtype=pl.Boolean
                )
                .alias("has_imv_in_first_24hr_icu")
            ])

            # Create IMV indicator map
            imv_map = df_full_sequences_with_timing.select([
                "hospitalization_id",
                "has_imv_in_first_24hr_icu"
            ])

            task3_task4_map = task3_task4_map.join(imv_map, on="hospitalization_id", how="left")

            imv_count = imv_map.filter(pl.col("has_imv_in_first_24hr_icu") == True).shape[0]
            logger.info(f"Found {imv_count} hospitalizations with IMV in first 24hr of ICU (train/val)")

            # Filter to 24hr ICU if requested
            df_train_val = truncate_sequences_at_24hr(df_train_val, cohort_df, vocab)
        else:
            # No cohort data, can't filter by ICU timing
            task3_task4_map = task3_task4_map.with_columns([
                pl.lit(True).alias("has_imv_in_first_24hr_icu")
            ])

        # OPTIMIZATION: Aggregate sequences per hospitalization (66M rows → ~35K rows)
        # Note: This only includes sequences from first 24hr, no disposition info
        df_train_val = df_train_val.group_by("hospitalization_id").agg([
            pl.col(sequence_col_name).str.concat(delimiter=" ").alias("clif_text"),
        ])

        # Join disposition and Task 3 & 4 labels/proportions back from full dataset
        df_train_val = df_train_val.join(disposition_map, on="hospitalization_id", how="left")
        df_train_val = df_train_val.join(task3_task4_map, on="hospitalization_id", how="left")

        # Check disposition distribution
        disp_counts = df_train_val["disposition"].value_counts()
        logger.info(f"Disposition distribution:\n{disp_counts}")

        # Create label_home and label_ltach columns for Tasks 1 & 2
        df_train_val = df_train_val.with_columns([
            pl.when(pl.col("disposition") == "disposition_home")
            .then(1)
            .otherwise(0)
            .alias("label_home"),
            pl.when(pl.col("disposition") == "disposition_ltach")
            .then(1)
            .otherwise(0)
            .alias("label_ltach")
        ])

        # Compute statistics for both tasks
        stats_home = {
            "total_examples": len(df_train_val),
            "positive_examples": df_train_val.filter(pl.col("label_home") == 1).shape[0],
            "negative_examples": df_train_val.filter(pl.col("label_home") == 0).shape[0],
            "positive_ratio": df_train_val.filter(pl.col("label_home") == 1).shape[0] / len(df_train_val),
        }

        stats_ltach = {
            "total_examples": len(df_train_val),
            "positive_examples": df_train_val.filter(pl.col("label_ltach") == 1).shape[0],
            "negative_examples": df_train_val.filter(pl.col("label_ltach") == 0).shape[0],
            "positive_ratio": df_train_val.filter(pl.col("label_ltach") == 1).shape[0] / len(df_train_val),
        }

        # Save File 1: Task 1 & 2 shared disposition file
        df_disposition = df_train_val.select(["hospitalization_id", "clif_text", "label_home", "label_ltach", "disposition"])
        output_file_disp = output_dir / "task1_task2_disposition_train_val.parquet"
        df_disposition.write_parquet(output_file_disp)
        logger.info(f"Saved Task 1 & 2 train/val data ({len(df_disposition)} rows) to {output_file_disp}")

        # Filter for Task 3 & 4: Only hospitalizations with IMV in 0-24hr of ICU
        # Also exclude "other_o2" edge cases (patients with IMV but no resp_device tokens)
        logger.info("Filtering Task 3 & 4 cohort: hospitalizations with IMV in 0-24hr of ICU...")
        task3_task4_train_val = df_train_val.filter(
            (pl.col("has_imv_in_first_24hr_icu") == True) & (pl.col("task3_label") != "other_o2")
        )

        # Save File 2: Task 3 & 4 respiratory status file
        df_task3_task4 = task3_task4_train_val.select(["hospitalization_id", "clif_text", "task3_label", "task4_proportion"])
        output_file_respiratory = output_dir / "task3_task4_respiratory_train_val.parquet"
        df_task3_task4.write_parquet(output_file_respiratory)
        logger.info(f"Saved Task 3 & 4 train/val data ({len(df_task3_task4)} rows) to {output_file_respiratory}")

        # Task 3 statistics
        task3_counts = task3_task4_train_val["task3_label"].value_counts()
        logger.info(f"Task 3 label distribution:\n{task3_counts}")

        stats_task3 = {
            "total_examples": len(task3_task4_train_val),
            "label_counts": dict(zip(task3_counts["task3_label"].to_list(), task3_counts["count"].to_list()))
        }

        # Task 4 statistics
        task4_proportions_train = task3_task4_train_val["task4_proportion"].to_list()
        stats_task4 = {
            "total_examples": len(task3_task4_train_val),
            "mean_proportion": sum(task4_proportions_train) / len(task4_proportions_train),
            "min_proportion": min(task4_proportions_train),
            "max_proportion": max(task4_proportions_train),
        }
        logger.info(f"Task 4 proportion stats: mean={stats_task4['mean_proportion']:.3f}, min={stats_task4['min_proportion']:.3f}, max={stats_task4['max_proportion']:.3f}")

        stats["train_val"] = {
            "task1_home": stats_home,
            "task2_ltach": stats_ltach,
            "task3_imv": stats_task3,
            "task4_hypoxic": stats_task4
        }

    # Process test data
    test_file = input_dir / "test_sequences.parquet"
    if test_file.exists():
        logger.info("Processing test data")
        df_test = load_narratives(test_file, min_length, max_length)

        # Determine sequence column name
        sequence_col_name = None
        if "sequence" in df_test.columns:
            sequence_col_name = "sequence"
        elif "token_ids" in df_test.columns:
            sequence_col_name = "token_ids"
        elif "clif_sentence" in df_test.columns:
            sequence_col_name = "clif_sentence"
        else:
            raise ValueError("No sequence column found in dataframe")

        # CRITICAL: Extract disposition AND Task 3 labels BEFORE 24hr filtering
        logger.info("Extracting disposition from full sequences (before 24hr filtering)...")
        # Filter to only disposition rows FIRST, then take last
        disposition_map = (df_test
            .filter(pl.col(sequence_col_name).str.starts_with("disposition_"))
            .group_by("hospitalization_id")
            .agg([
                pl.col(sequence_col_name).last().alias("disposition")
            ])
        )
        logger.info(f"Extracted dispositions for {len(disposition_map)} hospitalizations with valid disposition tokens")

        # Extract Task 3 labels and Task 4 proportions from hours 24-72 window (before truncation)
        logger.info("Extracting Task 3 labels and Task 4 proportions from hours 24-72 window (before 24hr truncation)...")
        # For Task 3 & 4, we need to work with full sequences per hospitalization
        # Group all sequences per hospitalization to reconstruct full timeline
        df_full_sequences = df_test.group_by("hospitalization_id").agg([
            pl.col(sequence_col_name).alias("sequences_list"),
        ])

        # Extract Task 3 labels AND Task 4 proportions by processing full sequences
        # Note: This is done with token strings, not IDs
        def extract_task3_and_task4_from_tokens(sequences_list: List[str], icu_start_hours: float) -> tuple[str, float]:
            """
            Extract Task 3 label and Task 4 proportion from list of token strings.

            Task 3 Logic:
            1. Check for disposition_expired in 24-72hr window → "expired"
            2. Collect all resp_device tokens from 0-72hr
            3. Deduplicate consecutive identical tokens
            4. Check if patient was EVER on IMV during 24-72hr specifically
            5. Classify:
               - If ever on IMV in 24-72hr:
                 * Last token (after dedup) is NOT imv → "imv_off"
                 * Last token (after dedup) IS imv → "imv_on"
               - If NEVER on IMV in 24-72hr → "other_o2"

            Task 4 Logic:
            Calculate proportion of time in severe hypoxic respiratory failure (PEEP>8 OR FiO2>50%)
            during hours 24-72 after ICU admission.

            Returns:
                Tuple of (task3_label, task4_proportion)
            """
            all_tokens = sequences_list  # List of token strings

            # Track time windows
            tokens_24_72 = []  # For checking expired and IMV presence
            all_resp_devices = []  # All resp_device tokens from 0-72hr
            resp_devices_24_72 = []  # resp_device tokens specifically from 24-72hr
            current_day = 0
            total_hours = 0
            in_window_24_72 = False

            for token in all_tokens:
                # Skip None values
                if token is None:
                    continue

                # Update day tracking
                if token.startswith("day_"):
                    try:
                        current_day = int(token.split("_")[1])
                    except (ValueError, IndexError):
                        pass

                # Update hour tracking
                if token.startswith("hour_"):
                    try:
                        hour = int(token.split("_")[1])
                        total_hours = (current_day - 1) * 24 + hour

                        # Update window status
                        in_window_24_72 = 24 <= total_hours <= 72

                        # Stop collecting after 72 hours
                        if total_hours > 72:
                            break

                    except (ValueError, IndexError):
                        pass

                # Collect tokens for 24-72hr window
                if in_window_24_72:
                    tokens_24_72.append(token)

                # Collect all resp_device tokens from 0-72hr
                if token.startswith("resp_device_") and total_hours <= 72:
                    all_resp_devices.append(token)
                    # Also track which are from 24-72hr
                    if in_window_24_72:
                        resp_devices_24_72.append(token)

            # Priority 1: Check for disposition_expired in 24-72hr window
            if "disposition_expired" in tokens_24_72:
                task3_label = "expired"
            # If no resp_device tokens at all, classify as other_o2
            # (will be filtered out later when we keep only IMV patients)
            elif len(all_resp_devices) == 0:
                task3_label = "other_o2"
            else:
                # Deduplicate consecutive identical resp_device tokens
                deduped_resp_devices = []
                for token in all_resp_devices:
                    if not deduped_resp_devices or deduped_resp_devices[-1] != token:
                        deduped_resp_devices.append(token)

                # Get last resp_device token (after deduplication)
                last_resp_device = deduped_resp_devices[-1]

                # For Task 3: Check if last token is IMV or not
                # Since Task 3 only includes patients with IMV in 0-24hr ICU,
                # there are only 3 classes: expired, imv_off, imv_on
                if "imv" in last_resp_device.lower() and "imv_off" not in last_resp_device.lower():
                    task3_label = "imv_on"  # Last respiratory device is IMV
                else:
                    task3_label = "imv_off"  # Last respiratory device is NOT IMV

            # Calculate Task 4 proportion
            task4_proportion = calculate_hypoxic_proportion(sequences_list, icu_start_hours)

            return task3_label, task4_proportion

        # Calculate IMV presence in first 24hr of ICU for Task 3 filtering
        # Do this BEFORE extraction to have timing data available
        if cohort_df is not None:
            logger.info("Getting ICU timing data for Task 3 & 4 extraction (test)...")

            # Get cohort timing data
            cohort_timing = cohort_df.select([
                "hospitalization_id",
                "admission_dttm",
                "first_icu_start_time"
            ])

            # Calculate ICU start hours offset from hospitalization admission
            cohort_timing = cohort_timing.with_columns([
                ((pl.col("first_icu_start_time") - pl.col("admission_dttm")).dt.total_seconds() / 3600.0)
                .alias("icu_start_hours")
            ])

            # Join timing data to full sequences
            df_full_sequences_with_timing = df_full_sequences.join(
                cohort_timing.select(["hospitalization_id", "icu_start_hours"]),
                on="hospitalization_id",
                how="left"
            )
        else:
            # Default to 0 if no cohort data
            df_full_sequences_with_timing = df_full_sequences.with_columns([
                pl.lit(0.0).alias("icu_start_hours")
            ])

        # Apply extraction (this will be slow, but necessary for Task 3 & 4)
        task3_labels = []
        task4_proportions = []
        for idx, row in enumerate(df_full_sequences_with_timing.iter_rows(named=True)):
            hosp_id = row["hospitalization_id"]
            sequences_list = row["sequences_list"]
            icu_start_hours = row.get("icu_start_hours", 0.0)

            task3_label, task4_proportion = extract_task3_and_task4_from_tokens(sequences_list, icu_start_hours)
            task3_labels.append(task3_label)
            task4_proportions.append(task4_proportion)

        # Create Task 3 & 4 label/proportion map
        task3_task4_map = pl.DataFrame({
            "hospitalization_id": df_full_sequences["hospitalization_id"].to_list(),
            "task3_label": task3_labels,
            "task4_proportion": task4_proportions,
        })

        logger.info(f"Extracted Task 3 labels and Task 4 proportions for {len(task3_task4_map)} hospitalizations")
        logger.info(f"Task 3 label distribution:\n{task3_task4_map['task3_label'].value_counts()}")
        logger.info(f"Task 4 proportion stats: mean={sum(task4_proportions)/len(task4_proportions):.3f}, min={min(task4_proportions):.3f}, max={max(task4_proportions):.3f}")

        # Check for IMV presence in first 24hr of ICU for Task 3 filtering
        if cohort_df is not None:
            logger.info("Checking for IMV presence in first 24hr of ICU (test)...")

            # Check for IMV in first 24hr of ICU using Polars map_elements
            df_full_sequences_with_timing = df_full_sequences_with_timing.with_columns([
                pl.struct(["sequences_list", "icu_start_hours"])
                .map_elements(
                    lambda row: check_imv_in_icu_0_24hr(row["sequences_list"], row["icu_start_hours"]),
                    return_dtype=pl.Boolean
                )
                .alias("has_imv_in_first_24hr_icu")
            ])

            # Create IMV indicator map
            imv_map = df_full_sequences_with_timing.select([
                "hospitalization_id",
                "has_imv_in_first_24hr_icu"
            ])

            task3_task4_map = task3_task4_map.join(imv_map, on="hospitalization_id", how="left")

            imv_count = imv_map.filter(pl.col("has_imv_in_first_24hr_icu") == True).shape[0]
            logger.info(f"Found {imv_count} hospitalizations with IMV in first 24hr of ICU (test)")

        # Filter to 24hr ICU if requested
        if cohort_df is not None:
            df_test = truncate_sequences_at_24hr(df_test, cohort_df, vocab)

        # OPTIMIZATION: Aggregate sequences per hospitalization (10M rows → ~7K rows)
        # Note: This only includes sequences from first 24hr, no disposition info
        df_test = df_test.group_by("hospitalization_id").agg([
            pl.col(sequence_col_name).str.concat(delimiter=" ").alias("clif_text"),
        ])

        # Join disposition and Task 3 & 4 labels/proportions back from full dataset
        df_test = df_test.join(disposition_map, on="hospitalization_id", how="left")
        df_test = df_test.join(task3_task4_map, on="hospitalization_id", how="left")

        # Check disposition distribution
        disp_counts = df_test["disposition"].value_counts()
        logger.info(f"Disposition distribution:\n{disp_counts}")

        # Create label_home and label_ltach columns for Tasks 1 & 2
        df_test = df_test.with_columns([
            pl.when(pl.col("disposition") == "disposition_home")
            .then(1)
            .otherwise(0)
            .alias("label_home"),
            pl.when(pl.col("disposition") == "disposition_ltach")
            .then(1)
            .otherwise(0)
            .alias("label_ltach")
        ])

        # Compute statistics for both tasks
        stats_home = {
            "total_examples": len(df_test),
            "positive_examples": df_test.filter(pl.col("label_home") == 1).shape[0],
            "negative_examples": df_test.filter(pl.col("label_home") == 0).shape[0],
            "positive_ratio": df_test.filter(pl.col("label_home") == 1).shape[0] / len(df_test),
        }

        stats_ltach = {
            "total_examples": len(df_test),
            "positive_examples": df_test.filter(pl.col("label_ltach") == 1).shape[0],
            "negative_examples": df_test.filter(pl.col("label_ltach") == 0).shape[0],
            "positive_ratio": df_test.filter(pl.col("label_ltach") == 1).shape[0] / len(df_test),
        }

        # Save File 1: Task 1 & 2 shared disposition file
        df_disposition = df_test.select(["hospitalization_id", "clif_text", "label_home", "label_ltach", "disposition"])
        output_file_disp = output_dir / "task1_task2_disposition_test.parquet"
        df_disposition.write_parquet(output_file_disp)
        logger.info(f"Saved Task 1 & 2 test data ({len(df_disposition)} rows) to {output_file_disp}")

        # Filter for Task 3 & 4: Only hospitalizations with IMV in 0-24hr of ICU
        # Also exclude "other_o2" edge cases (patients with IMV but no resp_device tokens)
        logger.info("Filtering Task 3 & 4 cohort: hospitalizations with IMV in 0-24hr of ICU...")
        task3_task4_test = df_test.filter(
            (pl.col("has_imv_in_first_24hr_icu") == True) & (pl.col("task3_label") != "other_o2")
        )

        # Save File 2: Task 3 & 4 respiratory status file
        df_task3_task4 = task3_task4_test.select(["hospitalization_id", "clif_text", "task3_label", "task4_proportion"])
        output_file_respiratory = output_dir / "task3_task4_respiratory_test.parquet"
        df_task3_task4.write_parquet(output_file_respiratory)
        logger.info(f"Saved Task 3 & 4 test data ({len(df_task3_task4)} rows) to {output_file_respiratory}")

        # Task 3 statistics
        task3_counts = task3_task4_test["task3_label"].value_counts()
        logger.info(f"Task 3 label distribution:\n{task3_counts}")

        stats_task3 = {
            "total_examples": len(task3_task4_test),
            "label_counts": dict(zip(task3_counts["task3_label"].to_list(), task3_counts["count"].to_list()))
        }

        # Task 4 statistics
        task4_proportions_test = task3_task4_test["task4_proportion"].to_list()
        stats_task4 = {
            "total_examples": len(task3_task4_test),
            "mean_proportion": sum(task4_proportions_test) / len(task4_proportions_test),
            "min_proportion": min(task4_proportions_test),
            "max_proportion": max(task4_proportions_test),
        }
        logger.info(f"Task 4 proportion stats: mean={stats_task4['mean_proportion']:.3f}, min={stats_task4['min_proportion']:.3f}, max={stats_task4['max_proportion']:.3f}")

        stats["test"] = {
            "task1_home": stats_home,
            "task2_ltach": stats_ltach,
            "task3_imv": stats_task3,
            "task4_hypoxic": stats_task4
        }

    return stats


def load_benchmark_dataset(
    data_file: Union[str, Path],
    vocab: Optional[Dict[str, int]] = None,
    max_length: int = 8192,
) -> BenchmarkDataset:
    """
    Load processed benchmark dataset.

    Args:
        data_file: Path to processed parquet file
        vocab: Token to ID mapping (required if data contains token strings)
        max_length: Maximum sequence length

    Returns:
        BenchmarkDataset instance
    """
    data_file = Path(data_file)

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    logger.info(f"Loading benchmark dataset from {data_file}")

    # Load parquet
    df = pl.read_parquet(data_file)

    # Extract data - handle both token IDs and token strings
    if "sequence" in df.columns:
        sequences = df["sequence"].to_list()
    elif "token_ids" in df.columns:
        sequences = df["token_ids"].to_list()
    elif "clif_text" in df.columns:
        # clif_text contains space-separated token strings, need to convert to IDs
        if vocab is None:
            raise ValueError("vocab required to convert token strings to IDs")

        logger.info("Converting token strings to IDs...")
        clif_texts = df["clif_text"].to_list()
        sequences = []
        # Get UNK token ID - try both [UNK] and <unk> formats
        unk_id = vocab.get("[UNK]", vocab.get("<unk>", 1))
        logger.info(f"Using UNK token ID: {unk_id}")

        unknown_tokens_count = 0
        for text in clif_texts:
            tokens = text.split()
            token_ids = []
            for token in tokens:
                if token in vocab:
                    token_ids.append(vocab[token])
                else:
                    token_ids.append(unk_id)
                    unknown_tokens_count += 1
            sequences.append(token_ids)

        if unknown_tokens_count > 0:
            logger.warning(f"Found {unknown_tokens_count} unknown tokens in dataset, mapped to UNK (ID={unk_id})")
    elif "clif_sentence" in df.columns:
        sequences = df["clif_sentence"].to_list()
    else:
        raise ValueError("No sequence column found (expected: sequence, token_ids, clif_text, or clif_sentence)")

    # Handle labels - multi-task datasets have different label columns
    # For embedding extraction, we don't need labels, so use dummy labels if not found
    if "label" in df.columns:
        labels = df["label"].to_list()
    elif "label_home" in df.columns:
        # Use label_home as default for task1_task2 datasets
        labels = df["label_home"].to_list()
    elif "task3_label" in df.columns:
        # Use task3_label for task3_task4 datasets
        labels = df["task3_label"].to_list()
    else:
        # No label column found, use dummy labels (all zeros)
        labels = [0] * len(sequences)

    # Handle dispositions
    if "disposition" in df.columns:
        dispositions = df["disposition"].to_list()
    else:
        # Use empty string as default
        dispositions = [""] * len(sequences)

    # Create example IDs - use hospitalization_id if available
    if "hospitalization_id" in df.columns:
        example_ids = df["hospitalization_id"].to_list()
    elif "example_id" in df.columns:
        example_ids = df["example_id"].to_list()
    else:
        example_ids = list(range(len(sequences)))

    # Create dataset
    dataset = BenchmarkDataset(
        sequences=sequences,
        labels=labels,
        dispositions=dispositions,
        example_ids=example_ids,
        max_length=max_length,
    )

    logger.info(f"Loaded {len(dataset)} examples")

    return dataset
