"""Time-series aggregation for crash count prediction.

Aggregates individual crash records into crash counts per time bucket and zone.
Supports hourly, daily, and weekly granularities.
"""

from __future__ import annotations

import logging

TimeGranularity = Literal["hourly", "daily", "weekly"]


@dataclass
class TimeSeriesConfig:
    """Configuration for time series aggregation.

    Attributes:
        granularity: Time bucket size ('hourly', 'daily', 'weekly').
        crash_date_col: Name of crash datetime column.
        zone_col: Name of zone/cluster column (optional).
        fill_gaps: Whether to fill missing time periods with zero counts.
    """

    granularity: TimeGranularity = "hourly"
    crash_date_col: str = "CRASH_DATE"
    zone_col: str | None = "LOCATION_CLUSTER"
    fill_gaps: bool = True

    @property
    def freq(self) -> str:
        """Pandas frequency string for this granularity."""
        return {
            "hourly": "h",
            "daily": "D",
            "weekly": "W",
        }[self.granularity]


def aggregate_crashes_by_time(
    df: pd.DataFrame,
    config: TimeSeriesConfig | None = None,
) -> pd.DataFrame:
    """Aggregate crash records into counts per time bucket.

    Groups crashes by time period (and optionally by zone), counting
    the number of crashes in each bucket. Optionally fills gaps in
    the time series with zero counts.

    Args:
        df: DataFrame with individual crash records.
        config: Aggregation configuration. Uses defaults if None.

    Returns:
        DataFrame with columns:
        - timestamp: Start of the time bucket
        - zone_id: Zone identifier (if zone_col specified)
        - crash_count: Number of crashes in this bucket
    """
    if config is None:
        config = TimeSeriesConfig()

    # Make a copy to avoid modifying original
    df = df.copy()

    # Parse and validate datetime column
    if config.crash_date_col not in df.columns:
        raise ValueError(f"Column '{config.crash_date_col}' not found in DataFrame")

    df["_timestamp"] = pd.to_datetime(df[config.crash_date_col], errors="coerce")

    # Drop rows with invalid timestamps
    n_invalid = df["_timestamp"].isna().sum()
    if n_invalid > 0:
        logger.warning(f"Dropping {n_invalid} rows with invalid timestamps")
        df = df.dropna(subset=["_timestamp"])

    # Floor timestamp to bucket start
    df["_bucket"] = df["_timestamp"].dt.floor(config.freq)

    # Group by time bucket (and zone if specified)
    if config.zone_col and config.zone_col in df.columns:
        group_cols = ["_bucket", config.zone_col]
        result = (
            df.groupby(group_cols)
            .size()
            .reset_index(name="crash_count")
        )
        result = result.rename(columns={
            "_bucket": "timestamp",
            config.zone_col: "zone_id",
        })
    else:
        result = (
            df.groupby("_bucket")
            .size()
            .reset_index(name="crash_count")
        )
        result = result.rename(columns={"_bucket": "timestamp"})

    # Fill gaps with zero counts if requested
    if config.fill_gaps:
        result = _fill_time_gaps(result, config)

    # Sort by time (and zone)
    sort_cols = ["timestamp"]
    if "zone_id" in result.columns:
        sort_cols.append("zone_id")
    result = result.sort_values(sort_cols).reset_index(drop=True)

    return result


def _fill_time_gaps(
    df: pd.DataFrame,
    config: TimeSeriesConfig,
) -> pd.DataFrame:
    """Fill missing time periods with zero crash counts.

    Creates a complete time series from min to max timestamp,
    filling any gaps with zero counts.

    Args:
        df: Aggregated DataFrame with timestamp and crash_count.
        config: Aggregation configuration.

    Returns:
        DataFrame with no gaps in the time series.
    """
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()

    # Create complete date range
    full_range = pd.date_range(start=min_time, end=max_time, freq=config.freq)

    if "zone_id" in df.columns:
        # Create all combinations of time × zone
        zones = df["zone_id"].unique()
        full_index = pd.MultiIndex.from_product(
            [full_range, zones],
            names=["timestamp", "zone_id"],
        )
        full_df = pd.DataFrame(index=full_index).reset_index()

        # Merge with actual counts
        result = full_df.merge(
            df,
            on=["timestamp", "zone_id"],
            how="left",
        )
    else:
        # Simple single time series
        full_df = pd.DataFrame({"timestamp": full_range})
        result = full_df.merge(df, on="timestamp", how="left")

    # Fill NaN counts with 0
    result["crash_count"] = result["crash_count"].fillna(0).astype(int)

    return result


def aggregate_with_zones(
    df: pd.DataFrame,
    zone_labels: np.ndarray,
    config: TimeSeriesConfig | None = None,
) -> pd.DataFrame:
    """Aggregate crashes with pre-computed zone labels.

    Convenience function that assigns zone labels to the DataFrame
    before aggregation.

    Args:
        df: DataFrame with individual crash records.
        zone_labels: Array of zone IDs for each crash (same length as df).
        config: Aggregation configuration.

    Returns:
        Aggregated DataFrame with crash counts per time bucket and zone.
    """
    if config is None:
        config = TimeSeriesConfig()

    df = df.copy()
    df[config.zone_col or "LOCATION_CLUSTER"] = zone_labels

    return aggregate_crashes_by_time(df, config)


def get_zone_time_matrix(
    df: pd.DataFrame,
    config: TimeSeriesConfig | None = None,
) -> tuple[pd.DataFrame, list[str], list[int]]:
    """Convert aggregated data to zone × time matrix format.

    Pivots the aggregated data into a matrix where rows are time buckets
    and columns are zones. Useful for multi-zone modeling.

    Args:
        df: Aggregated DataFrame from aggregate_crashes_by_time.
        config: Configuration (unused, for consistency).

    Returns:
        Tuple of:
        - DataFrame with timestamps as index, zones as columns, counts as values
        - List of timestamp strings
        - List of zone IDs
    """
    if "zone_id" not in df.columns:
        raise ValueError("DataFrame must have 'zone_id' column for matrix format")

    # Pivot to matrix
    matrix = df.pivot(
        index="timestamp",
        columns="zone_id",
        values="crash_count",
    ).fillna(0).astype(int)

    timestamps = matrix.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
    zones = matrix.columns.tolist()

    return matrix, timestamps, zones


def compute_global_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute city-wide crash counts from zone-level data.

    Aggregates across all zones to get total counts per time bucket.
    Useful for the global model in the ensemble.

    Args:
        df: Zone-level aggregated DataFrame with zone_id column.

    Returns:
        DataFrame with timestamp and total crash_count (no zone_id).
    """
    if "zone_id" not in df.columns:
        # Already global
        return df

    return (
        df.groupby("timestamp")["crash_count"]
        .sum()
        .reset_index()
    )
