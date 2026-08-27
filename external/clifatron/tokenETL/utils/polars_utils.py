"""
Polars Utilities for tokenETL Pipeline

Provides optimized functions using polars for faster data processing:
- Numeric binning (10-100x faster than pandas apply)
- Timezone stripping
- Value mapping operations
"""

import pandas as pd
import polars as pl
import gc
from typing import Dict


def strip_all_datetime_timezones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove timezone from ALL datetime columns in a pandas DataFrame.

    Args:
        df: Input pandas DataFrame

    Returns:
        DataFrame with all datetime columns timezone-free
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
    return df


def bin_numeric_values_polars(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
    bins_df: pd.DataFrame,
    measurement_col: str = 'measurement'
) -> pd.DataFrame:
    """
    Bin numeric values using polars (10-100x faster than pandas apply).

    Uses vectorized join operations instead of row-by-row apply.

    Process:
    1. Convert to polars for speed
    2. Join data with bins on category
    3. Filter to matching bins (min_value <= value < max_value)
    4. Select token for each value
    5. Convert back to pandas

    Args:
        df: Input DataFrame with columns to bin
        value_col: Name of column containing numeric values
        category_col: Name of column containing category (e.g., 'lab_category')
        bins_df: DataFrame with bins (columns: measurement, min_value, max_value, token)
        measurement_col: Name of measurement column in bins_df (default: 'measurement')

    Returns:
        DataFrame with token column added
    """
    try:
        # Convert to polars lazy frames for efficiency
        data_pl = pl.from_pandas(df).lazy()
        bins_pl = pl.from_pandas(bins_df).lazy()

        # Join data with bins on category/measurement
        # Then filter to matching bins where value falls within range
        joined = data_pl.join(
            bins_pl,
            left_on=category_col,
            right_on=measurement_col,
            how='left'
        ).filter(
            (pl.col(value_col) >= pl.col('min_value')) &
            (
                (pl.col(value_col) < pl.col('max_value')) |
                # Handle edge case: value equals max of last bin
                (
                    (pl.col(value_col) == pl.col('max_value')) &
                    (pl.col('max_value') == pl.col('max_value').max().over(category_col))
                )
            )
        )

        # Collect and convert back to pandas
        result_df = joined.collect().to_pandas()

        return result_df

    finally:
        # Explicitly delete polars objects to release semaphores (prevents leaks in Python 3.13+)
        try:
            del joined
        except NameError:
            pass
        try:
            del bins_pl
        except NameError:
            pass
        try:
            del data_pl
        except NameError:
            pass
        # Force garbage collection to clean up polars thread pool resources
        gc.collect()


def bin_numeric_values_by_category(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
    bins_df: pd.DataFrame,
    measurement_col: str = 'measurement',
    logger=None
) -> pd.DataFrame:
    """
    Bin numeric values by processing each category separately (memory efficient for large datasets).

    Instead of joining all records at once, this function processes each unique category
    separately, dramatically reducing memory usage and preventing hangs on large datasets.

    For example, with 47M vital records:
    - Old approach: Join 47M records × all bins = billions of intermediate rows
    - New approach: Join 2M heart_rate records × 10 heart_rate bins = 20M intermediate rows per category

    Args:
        df: Input DataFrame with columns to bin
        value_col: Name of column containing numeric values
        category_col: Name of column containing category (e.g., 'vital_category')
        bins_df: DataFrame with bins (columns: measurement, min_value, max_value, token)
        measurement_col: Name of measurement column in bins_df (default: 'measurement')
        logger: Optional logger for progress updates

    Returns:
        DataFrame with token column added
    """
    if logger:
        logger.info(f"  Processing by {category_col} for memory efficiency...")

    results = []
    categories = df[category_col].unique()
    total_categories = len(categories)

    for idx, category in enumerate(categories, 1):
        if logger:
            logger.info(f"    [{idx}/{total_categories}] Processing {category}...")

        # Filter data to this category only
        category_data = df[df[category_col] == category].copy()
        category_bins = bins_df[bins_df[measurement_col] == category].copy()

        if len(category_bins) == 0:
            if logger:
                logger.info(f"      ⚠ No bins found for {category}, skipping")
            continue

        try:
            # Convert to polars for this category only
            data_pl = pl.from_pandas(category_data).lazy()
            bins_pl = pl.from_pandas(category_bins).lazy()

            # Join and filter (much smaller intermediate result per category)
            joined = data_pl.join(
                bins_pl,
                left_on=category_col,
                right_on=measurement_col,
                how='left'
            ).filter(
                (pl.col(value_col) >= pl.col('min_value')) &
                (
                    (pl.col(value_col) < pl.col('max_value')) |
                    # Handle edge case: value equals max of last bin
                    (
                        (pl.col(value_col) == pl.col('max_value')) &
                        (pl.col('max_value') == pl.col('max_value').max())
                    )
                )
            )

            # Collect this category's results
            category_result = joined.collect().to_pandas()

            if len(category_result) > 0:
                results.append(category_result)

                if logger:
                    records_in = len(category_data)
                    records_out = len(category_result)
                    pct = (records_out / records_in * 100) if records_in > 0 else 0
                    logger.info(f"      ✓ Tokenized {records_out:,}/{records_in:,} ({pct:.1f}%) records")

        except Exception as e:
            if logger:
                logger.error(f"      ✗ Error processing {category}: {e}")
            raise

        finally:
            # Explicitly delete polars objects to release semaphores (prevents leaks in Python 3.13+)
            try:
                del category_result
            except NameError:
                pass
            try:
                del joined
            except NameError:
                pass
            try:
                del bins_pl
            except NameError:
                pass
            try:
                del data_pl
            except NameError:
                pass
            # Force garbage collection to clean up polars thread pool resources
            gc.collect()

    # Concatenate all category results
    if len(results) == 0:
        if logger:
            logger.warning("  No records were successfully tokenized")
        # Return empty DataFrame with same structure as input
        return df.head(0).copy()

    if logger:
        logger.info(f"  Combining results from {len(results)} categories...")

    final_df = pd.concat(results, ignore_index=True)

    if logger:
        logger.info(f"  ✓ Total records after tokenization: {len(final_df):,}")

    return final_df


def bin_numeric_values_with_intervals_by_category(
    df: pd.DataFrame,
    value_col: str,
    category_col: str,
    bins_df: pd.DataFrame,
    measurement_col: str = 'measurement',
    logger=None
) -> pd.DataFrame:
    """
    Bin numeric values using interval notation (memory efficient, category-by-category).

    Supports mathematical interval notation:
    - [a,b] means a <= value <= b (both inclusive)
    - (a,b] means a < value <= b (left exclusive, right inclusive)
    - [a,b) means a <= value < b (left inclusive, right exclusive)
    - (a,b) means a < value < b (both exclusive)

    Uses Polars streaming mode to avoid materializing large joins in memory.
    Deduplicates within the query pipeline before collection to prevent duplicate tokens.

    Args:
        df: Input DataFrame with columns to bin
        value_col: Name of column containing numeric values
        category_col: Name of column containing category (e.g., 'lab_category', 'vital_category')
        bins_df: DataFrame with bins, must include columns:
            - measurement: measurement name
            - min_interval: '[' for inclusive or '(' for exclusive
            - min_value: numeric lower bound
            - max_value: numeric upper bound
            - max_interval: ']' for inclusive or ')' for exclusive
            - token: token name
            - exact_dose_token: 1 for exact match, 0 for range match
        measurement_col: Name of measurement column in bins_df (default: 'measurement')
        logger: Optional logger for progress updates

    Returns:
        DataFrame with token column added
    """
    # Limit Polars to 8 threads to prevent semaphore leaks (Python 3.13+)
    import os
    original_threads = os.environ.get('POLARS_MAX_THREADS', None)
    os.environ['POLARS_MAX_THREADS'] = '8'

    try:
        if logger:
            logger.info(f"  Processing by {category_col} for memory efficiency (interval-aware streaming)...")

        results = []
        categories = df[category_col].unique()
        total_categories = len(categories)

        # Get original column names for deduplication
        original_cols = list(df.columns)

        for idx, category in enumerate(categories, 1):
            if logger:
                logger.info(f"    [{idx}/{total_categories}] Processing {category}...")

            # Filter data to this category only
            category_data = df[df[category_col] == category].copy()
            category_bins = bins_df[bins_df[measurement_col] == category].copy()

            if len(category_bins) == 0:
                if logger:
                    logger.info(f"      ⚠ No bins found for {category}, skipping")
                continue

            try:
                # Convert to polars for this category only
                data_pl = pl.from_pandas(category_data).lazy()
                bins_pl = pl.from_pandas(category_bins).lazy()

                # Join data with bins
                joined = data_pl.join(
                    bins_pl,
                    left_on=category_col,
                    right_on=measurement_col,
                    how='left'
                )

                # Apply interval logic
                # Check if exact_dose_token column exists
                if 'exact_dose_token' in category_bins.columns:
                    # For exact dose tokens (exact_dose_token == 1), require exact match
                    # For range tokens (exact_dose_token == 0), use interval logic
                    filtered = joined.filter(
                        (
                            # Exact match case
                            (pl.col('exact_dose_token') == 1) &
                            (pl.col(value_col) == pl.col('min_value'))
                        ) |
                        (
                            # Range match case
                            (pl.col('exact_dose_token') == 0) &
                            # Min bound check
                            (
                                ((pl.col('min_interval') == '[') & (pl.col(value_col) >= pl.col('min_value'))) |
                                ((pl.col('min_interval') == '(') & (pl.col(value_col) > pl.col('min_value')))
                            ) &
                            # Max bound check
                            (
                                ((pl.col('max_interval') == ']') & (pl.col(value_col) <= pl.col('max_value'))) |
                                ((pl.col('max_interval') == ')') & (pl.col(value_col) < pl.col('max_value')))
                            )
                        )
                    )
                else:
                    # No exact_dose_token column, use interval logic only
                    filtered = joined.filter(
                        # Min bound check
                        (
                            ((pl.col('min_interval') == '[') & (pl.col(value_col) >= pl.col('min_value'))) |
                            ((pl.col('min_interval') == '(') & (pl.col(value_col) > pl.col('min_value')))
                        ) &
                        # Max bound check
                        (
                            ((pl.col('max_interval') == ']') & (pl.col(value_col) <= pl.col('max_value'))) |
                            ((pl.col('max_interval') == ')') & (pl.col(value_col) < pl.col('max_value')))
                        )
                    )

                # Deduplicate within polars query (before collection)
                # Keep only first matching bin for each unique combination of original data
                # This prevents duplicate token assignment when multiple bins match the same value
                deduplicated = filtered.unique(subset=original_cols, keep='first')

                # Collect using streaming mode to avoid materializing full join in memory
                category_result = deduplicated.collect(streaming=True).to_pandas()

                if len(category_result) > 0:
                    results.append(category_result)

                    if logger:
                        records_in = len(category_data)
                        records_out = len(category_result)
                        pct = (records_out / records_in * 100) if records_in > 0 else 0
                        logger.info(f"      ✓ Tokenized {records_out:,}/{records_in:,} ({pct:.1f}%) records")

            except Exception as e:
                if logger:
                    logger.error(f"      ✗ Error processing {category}: {e}")
                raise

            finally:
                # Explicitly delete polars objects to release semaphores (prevents leaks in Python 3.13+)
                try:
                    del category_result
                except NameError:
                    pass
                try:
                    del deduplicated
                except NameError:
                    pass
                try:
                    del filtered
                except NameError:
                    pass
                try:
                    del joined
                except NameError:
                    pass
                try:
                    del bins_pl
                except NameError:
                    pass
                try:
                    del data_pl
                except NameError:
                    pass
                # Force garbage collection to clean up polars thread pool resources
                gc.collect()

        # Concatenate all category results
        if len(results) == 0:
            if logger:
                logger.warning("  No records were successfully tokenized")
            # Return empty DataFrame with same structure as input
            return df.head(0).copy()

        if logger:
            logger.info(f"  Combining results from {len(results)} categories...")

        final_df = pd.concat(results, ignore_index=True)

        if logger:
            logger.info(f"  ✓ Total records after tokenization: {len(final_df):,}")

        return final_df
    finally:
        # Restore original thread setting
        if original_threads is not None:
            os.environ['POLARS_MAX_THREADS'] = original_threads
        else:
            os.environ.pop('POLARS_MAX_THREADS', None)


def read_numeric_ranges_polars(csv_path: str, category: str) -> pd.DataFrame:
    """
    Read numeric ranges CSV using polars (15x faster than pandas).

    The CSV contains a pre-generated token column with interval notation:
    Format: {category}_{measurement}_{min_interval}{min_value},{max_value}{max_interval}
    Example: labs_albumin_[0.4,1.4] or vitals_heart_rate_(120.0,122.0]

    Args:
        csv_path: Path to critical_illness_tokenization_final_with_intervals.csv
        category: Category to filter ('labs', 'vitals', 'respiratory_support', 'medications')

    Returns:
        Filtered pandas DataFrame with token column
    """
    try:
        # Read with polars (much faster than pandas for large CSVs)
        ranges_pl = pl.read_csv(csv_path).lazy()

        # Filter to category and collect
        filtered = ranges_pl.filter(pl.col('category') == category)

        # Convert to pandas for compatibility
        return filtered.collect().to_pandas()

    finally:
        # Explicitly delete polars objects to release semaphores (prevents leaks in Python 3.13+)
        try:
            del filtered
        except NameError:
            pass
        try:
            del ranges_pl
        except NameError:
            pass
        # Force garbage collection to clean up polars resources
        gc.collect()


def load_master_token_registry(csv_path: str) -> pd.DataFrame:
    """
    Load complete token registry from critical_illness_tokenization CSV.

    This loads ALL possible tokens from the tokenization specification,
    enabling zero-count tracking for site auditing.

    The CSV contains a pre-generated token column with interval notation:
    Format: {category}_{measurement}_{min_interval}{min_value},{max_value}{max_interval}

    Args:
        csv_path: Path to critical_illness_tokenization_final_with_intervals.csv

    Returns:
        DataFrame with all token definitions (includes token column)
    """
    try:
        # Read entire CSV with polars
        registry_pl = pl.read_csv(csv_path).lazy()

        # Convert to pandas
        return registry_pl.collect().to_pandas()

    finally:
        # Explicitly delete polars objects to release semaphores (prevents leaks in Python 3.13+)
        try:
            del registry_pl
        except NameError:
            pass
        # Force garbage collection to clean up polars resources
        gc.collect()


def replace_values_with_tokens_polars(
    df: pd.DataFrame,
    value_col: str,
    bins_df: pd.DataFrame,
    measurement_name: str
) -> pd.Series:
    """
    Replace numeric values with tokens using polars vectorized operations.

    Faster than pandas apply for large datasets.

    Args:
        df: Input DataFrame
        value_col: Column name containing values to replace
        bins_df: DataFrame with bins for this measurement
        measurement_name: Name of the measurement (for filtering bins)

    Returns:
        Pandas Series with tokens (or None for unmapped values)
    """
    if len(bins_df) == 0:
        # No bins available, return None series
        return pd.Series([None] * len(df), index=df.index)

    try:
        # Create temp df with just value column and index
        temp_df = df[[value_col]].copy()
        temp_df['__index'] = range(len(temp_df))

        # Convert to polars
        data_pl = pl.from_pandas(temp_df).lazy()
        bins_pl = pl.from_pandas(bins_df).lazy()

        # Join and filter to matching bins
        matched = data_pl.join(
            bins_pl,
            how='cross'  # Cross join, then filter
        ).filter(
            (pl.col(value_col) >= pl.col('min_value')) &
            (
                (pl.col(value_col) < pl.col('max_value')) |
                ((pl.col(value_col) == pl.col('max_value')) &
                 (pl.col('max_value') == pl.col('max_value').max()))
            )
        ).group_by('__index').agg(
            pl.col('token').first()  # Take first matching token
        )

        # Convert back to pandas and align with original index
        result_df = matched.collect().to_pandas()
        result_series = pd.Series([None] * len(df), index=df.index)

        if len(result_df) > 0:
            result_series.iloc[result_df['__index'].values] = result_df['token'].values

        return result_series

    finally:
        # Explicitly delete polars objects to release semaphores (prevents leaks in Python 3.13+)
        try:
            del matched
        except NameError:
            pass
        try:
            del bins_pl
        except NameError:
            pass
        try:
            del data_pl
        except NameError:
            pass
        # Force garbage collection to clean up polars resources
        gc.collect()
