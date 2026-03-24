"""Aggregation utilities for hourly crash count prediction.

This module provides functions to:
1. Aggregate crash data by hour (counting crashes per hour)
2. Generate weather lag features
3. Add cyclical time encodings
4. Prepare data for hourly crash count regression
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_preparation.time_features import (
    DEFAULT_TIME_CONFIG,
    TimeFeatureConfig,
    add_time_features,
)


# Weather columns suitable for lag features (using CSV column names with spaces)
DEFAULT_WEATHER_LAG_COLUMNS = [
    "Air Temperature",
    "Humidity",
    "Rain Intensity",
    "Wind Speed",
]

# All available weather columns (for reference)
ALL_WEATHER_COLUMNS = [
    "Air Temperature",
    "Wet Bulb Temperature",
    "Humidity",
    "Rain Intensity",
    "Interval Rain",
    "Total Rain",
    "Precipitation Type",
    "Wind Direction",
    "Wind Speed",
    "Maximum Wind Speed",
    "Barometric Pressure",
    "Solar Radiation",
]

# US Federal Holidays (month, day) - used for is_holiday feature
US_HOLIDAYS = [
    (1, 1),    # New Year's Day
    (1, 15),   # MLK Day (approx)
    (2, 19),   # Presidents Day (approx)
    (5, 27),   # Memorial Day (approx)
    (7, 4),    # Independence Day
    (9, 2),    # Labor Day (approx)
    (10, 14),  # Columbus Day (approx)
    (11, 11),  # Veterans Day
    (11, 28),  # Thanksgiving (approx)
    (12, 25),  # Christmas
    (12, 31),  # New Year's Eve
]

# Lighting conditions indicating darkness
DARKNESS_CONDITIONS = ["DARKNESS", "DARKNESS, LIGHTED ROAD"]

# Road surface conditions indicating wet/hazardous
WET_ROAD_CONDITIONS = ["WET", "SNOW OR SLUSH", "ICE", "SAND, MUD, DIRT"]

# Weather conditions indicating bad weather
BAD_WEATHER_CONDITIONS = ["RAIN", "SNOW", "SLEET/HAIL", "FOG/SMOKE/HAZE", "FREEZING RAIN/DRIZZLE"]


def add_holiday_features(
    df: pd.DataFrame,
    timestamp_col: str = "hour_timestamp",
) -> pd.DataFrame:
    """Add holiday indicator feature.

    Args:
        df: DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame with added 'is_holiday' column (0 or 1).
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Create (month, day) tuples for each row
    month_day = list(zip(df[timestamp_col].dt.month, df[timestamp_col].dt.day))
    
    # Check if each date is a holiday
    holiday_set = set(US_HOLIDAYS)
    df["is_holiday"] = [1 if md in holiday_set else 0 for md in month_day]
    
    return df


def aggregate_crash_conditions(
    crash_df: pd.DataFrame,
    timestamp_col: str = "CRASH_DATE",
) -> pd.DataFrame:
    """Aggregate crash conditions by hour of day.

    Computes historical percentages for each hour (0-23):
    - pct_darkness: % of crashes that occurred in darkness
    - pct_wet_road: % of crashes on wet/icy roads
    - pct_bad_weather: % of crashes in rain/snow/fog

    Args:
        crash_df: DataFrame containing crash records with condition columns.
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame with hour (0-23) and condition percentages.
    """
    df = crash_df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    df["_hour"] = df[timestamp_col].dt.hour
    
    # Calculate condition flags
    if "LIGHTING_CONDITION" in df.columns:
        df["_is_darkness"] = df["LIGHTING_CONDITION"].isin(DARKNESS_CONDITIONS).astype(int)
    else:
        df["_is_darkness"] = 0
        
    if "ROADWAY_SURFACE_COND" in df.columns:
        df["_is_wet_road"] = df["ROADWAY_SURFACE_COND"].isin(WET_ROAD_CONDITIONS).astype(int)
    else:
        df["_is_wet_road"] = 0
        
    if "WEATHER_CONDITION" in df.columns:
        df["_is_bad_weather"] = df["WEATHER_CONDITION"].isin(BAD_WEATHER_CONDITIONS).astype(int)
    else:
        df["_is_bad_weather"] = 0
    
    # Aggregate by hour
    hourly_conditions = df.groupby("_hour").agg(
        pct_darkness=("_is_darkness", "mean"),
        pct_wet_road=("_is_wet_road", "mean"),
        pct_bad_weather=("_is_bad_weather", "mean"),
    ).reset_index()
    
    hourly_conditions = hourly_conditions.rename(columns={"_hour": "hour"})
    
    return hourly_conditions


def aggregate_crashes_by_hour(
    crash_df: pd.DataFrame,
    timestamp_col: str = "CRASH_DATE",
    include_zero_hours: bool = True,
) -> pd.DataFrame:
    """Aggregate crash data to hourly counts.

    Groups crashes by hour and counts occurrences. Optionally generates
    a complete hourly range and fills missing hours with zero counts.

    Args:
        crash_df: DataFrame containing crash records with timestamps.
        timestamp_col: Name of the timestamp column.
        include_zero_hours: If True, generates full hourly range and fills
            missing hours with 0 counts. If False, only includes hours
            with at least one crash.

    Returns:
        DataFrame with columns:
            - hour_timestamp: Start of each hour (datetime)
            - crash_count: Number of crashes in that hour (int)

    Example:
        >>> crashes = pd.DataFrame({
        ...     "CRASH_DATE": ["2024-01-01 10:15", "2024-01-01 10:45", "2024-01-01 12:30"]
        ... })
        >>> result = aggregate_crashes_by_hour(crashes)
        >>> result["crash_count"].tolist()
        [2, 0, 1]  # 10:00 has 2, 11:00 has 0, 12:00 has 1
    """
    df = crash_df.copy()

    # Parse timestamp if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Floor to hour
    df["hour_timestamp"] = df[timestamp_col].dt.floor("h")

    # Count crashes per hour
    hourly_counts = (
        df.groupby("hour_timestamp")
        .size()
        .reset_index(name="crash_count")
    )

    if include_zero_hours and len(hourly_counts) > 0:
        # Generate complete hourly range
        min_hour = hourly_counts["hour_timestamp"].min()
        max_hour = hourly_counts["hour_timestamp"].max()
        full_range = pd.date_range(start=min_hour, end=max_hour, freq="h")

        # Create DataFrame with full range
        full_df = pd.DataFrame({"hour_timestamp": full_range})

        # Merge and fill missing with 0
        hourly_counts = full_df.merge(hourly_counts, on="hour_timestamp", how="left")
        hourly_counts["crash_count"] = hourly_counts["crash_count"].fillna(0).astype(int)

    return hourly_counts


def add_weather_lag_features(
    df: pd.DataFrame,
    weather_columns: list[str] | None = None,
    lag_hours: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged weather features to the DataFrame.

    Creates new columns for each combination of weather column and lag hour.
    For example, with weather_columns=["Rain_Intensity"] and lag_hours=[1, 2],
    creates columns "Rain_Intensity_lag_1h" and "Rain_Intensity_lag_2h".

    The lagged value at time T represents the weather value from T - lag_hours.

    Args:
        df: DataFrame with weather columns. Must be sorted by time.
        weather_columns: List of weather column names to create lags for.
            Defaults to DEFAULT_WEATHER_LAG_COLUMNS.
        lag_hours: List of lag hours to create. Defaults to [1, 2, 3].

    Returns:
        DataFrame with additional lag feature columns.
        First N rows will have NaN for lag features where N = max(lag_hours).

    Example:
        >>> df = pd.DataFrame({
        ...     "hour_timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        ...     "Rain_Intensity": [0, 1, 2, 0, 0]
        ... })
        >>> result = add_weather_lag_features(df, ["Rain_Intensity"], [1, 2])
        >>> result["Rain_Intensity_lag_1h"].tolist()
        [nan, 0, 1, 2, 0]
    """
    if weather_columns is None:
        weather_columns = DEFAULT_WEATHER_LAG_COLUMNS

    if lag_hours is None:
        lag_hours = [1, 2, 3]

    df = df.copy()

    for col in weather_columns:
        if col not in df.columns:
            continue

        for lag in lag_hours:
            lag_col_name = f"{col}_lag_{lag}h"
            df[lag_col_name] = df[col].shift(lag)

    return df


def add_cyclical_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "hour_timestamp",
    time_config: TimeFeatureConfig | None = None,
) -> pd.DataFrame:
    """Add cyclical (sin/cos) encodings for temporal features.

    Encodes hour of day, day of week, and month as sin/cos pairs
    to capture the cyclical nature of time (e.g., hour 23 is close to hour 0).

    Note:
        This function delegates to `add_time_features` from the time_features module.
        For full control over time features (including binary features), use
        `add_time_features` directly with a TimeFeatureConfig.

    Args:
        df: DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column.
        time_config: Configuration for time features. If None, uses DEFAULT_TIME_CONFIG
            which enables all cyclical features (hour, day_of_week, month) but no
            binary features.

    Returns:
        DataFrame with additional time feature columns based on config.
    """
    if time_config is None:
        time_config = DEFAULT_TIME_CONFIG

    return add_time_features(df, timestamp_col=timestamp_col, config=time_config)


def merge_hourly_crashes_with_weather(
    crash_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    crash_timestamp_col: str = "CRASH_DATE",
    weather_timestamp_col: str = "Measurement Timestamp",
    include_zero_hours: bool = True,
) -> pd.DataFrame:
    """Merge crash counts with weather data at hourly granularity.

    This function:
    1. Aggregates crashes to hourly counts
    2. Aligns weather data to hourly timestamps
    3. Merges on the hour

    Args:
        crash_df: DataFrame containing crash records.
        weather_df: DataFrame containing weather measurements.
        crash_timestamp_col: Timestamp column name in crash data.
        weather_timestamp_col: Timestamp column name in weather data.
        include_zero_hours: Whether to include hours with zero crashes.

    Returns:
        DataFrame with hourly crash counts and weather features.
    """
    # Aggregate crashes by hour
    hourly_crashes = aggregate_crashes_by_hour(
        crash_df,
        timestamp_col=crash_timestamp_col,
        include_zero_hours=include_zero_hours,
    )

    # Prepare weather data - floor to hour
    weather = weather_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(weather[weather_timestamp_col]):
        weather[weather_timestamp_col] = pd.to_datetime(weather[weather_timestamp_col])

    weather["hour_timestamp"] = weather[weather_timestamp_col].dt.floor("h")

    # Drop duplicates (keep first measurement per hour)
    weather = weather.drop_duplicates(subset=["hour_timestamp"], keep="first")

    # Select weather columns (exclude timestamp and merge key)
    weather_cols = [c for c in weather.columns if c not in [weather_timestamp_col, "hour_timestamp"]]
    weather_subset = weather[["hour_timestamp"] + weather_cols]

    # Merge
    merged = hourly_crashes.merge(weather_subset, on="hour_timestamp", how="left")

    return merged


def prepare_hourly_data(
    crash_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
    weather_lag_columns: list[str] | None = None,
    lag_hours: list[int] | None = None,
    include_zero_hours: bool = True,
    time_config: TimeFeatureConfig | None = None,
    include_holidays: bool = False,
    include_conditions: bool = False,
) -> pd.DataFrame:
    """Full pipeline to prepare hourly crash count data for ML.

    This orchestrator function:
    1. Loads data if not provided
    2. Merges crashes with weather at hourly granularity
    3. Adds weather lag features
    4. Adds time features (cyclical and/or binary based on config)
    5. Optionally adds holiday indicators
    6. Optionally adds historical crash condition percentages
    7. Drops rows with NaN (from lag features at start)

    Args:
        crash_df: DataFrame of crash records. If None, loads from default path.
        weather_df: DataFrame of weather data. If None, loads from default path.
        weather_lag_columns: Weather columns to create lag features for.
            Defaults to DEFAULT_WEATHER_LAG_COLUMNS.
        lag_hours: Lag hours to create. Defaults to [1, 2, 3].
        include_zero_hours: Whether to include hours with zero crashes.
        time_config: Configuration for time features (cyclical and binary).
            If None, uses DEFAULT_TIME_CONFIG (all cyclical, no binary).
        include_holidays: Whether to add is_holiday binary feature.
        include_conditions: Whether to add historical crash condition
            percentages (pct_darkness, pct_wet_road, pct_bad_weather).

    Returns:
        DataFrame ready for tensor preparation with columns:
            - hour_timestamp: Timestamp (for time-based splitting)
            - crash_count: Target variable
            - Weather features (current values)
            - Weather lag features (e.g., Rain_Intensity_lag_1h)
            - Time features based on config (e.g., hour_sin, is_night)
            - Optionally: is_holiday
            - Optionally: pct_darkness, pct_wet_road, pct_bad_weather
    """
    # Load data if not provided (import at runtime to avoid circular deps)
    if crash_df is None:
        from data_preparation.helpers.csv_loaders import get_traffic_crashes
        crash_df = get_traffic_crashes()
    if weather_df is None:
        from data_preparation.helpers.csv_loaders import get_weather_stations
        weather_df = get_weather_stations()

    if weather_lag_columns is None:
        weather_lag_columns = DEFAULT_WEATHER_LAG_COLUMNS
    if lag_hours is None:
        lag_hours = [1, 2, 3]

    # Pre-compute condition percentages if needed (before merging)
    condition_lookup = None
    if include_conditions:
        condition_lookup = aggregate_crash_conditions(crash_df)

    # Step 1: Merge crashes with weather at hourly level
    merged = merge_hourly_crashes_with_weather(
        crash_df=crash_df,
        weather_df=weather_df,
        include_zero_hours=include_zero_hours,
    )

    # Step 2: Sort by time (required for lag features and time-based splitting)
    merged = merged.sort_values("hour_timestamp").reset_index(drop=True)

    # Step 3: Add weather lag features
    merged = add_weather_lag_features(
        merged,
        weather_columns=weather_lag_columns,
        lag_hours=lag_hours,
    )

    # Step 4: Add time features (cyclical and/or binary based on config)
    merged = add_cyclical_time_features(
        merged, timestamp_col="hour_timestamp", time_config=time_config
    )

    # Step 5: Add holiday features if requested
    if include_holidays:
        merged = add_holiday_features(merged, timestamp_col="hour_timestamp")

    # Step 6: Add condition features if requested (join on hour-of-day)
    if include_conditions and condition_lookup is not None:
        merged["_hour"] = pd.to_datetime(merged["hour_timestamp"]).dt.hour
        merged = merged.merge(condition_lookup, left_on="_hour", right_on="hour", how="left")
        merged = merged.drop(columns=["_hour", "hour"])

    # Step 7: Drop rows with NaN from lag features (first max(lag_hours) rows)
    max_lag = max(lag_hours)
    merged = merged.iloc[max_lag:].reset_index(drop=True)

    return merged


def get_hourly_feature_columns(
    weather_lag_columns: list[str] | None = None,
    lag_hours: list[int] | None = None,
    include_current_weather: bool = True,
    time_config: TimeFeatureConfig | None = None,
    include_holidays: bool = False,
    include_conditions: bool = False,
) -> tuple[list[str], list[str]]:
    """Get lists of categorical and numerical feature columns for hourly config.

    Args:
        weather_lag_columns: Weather columns with lag features.
        lag_hours: Lag hours used.
        include_current_weather: Whether to include current weather values.
        time_config: Configuration for time features. If None, uses
            DEFAULT_TIME_CONFIG.
        include_holidays: Whether is_holiday feature is included.
        include_conditions: Whether crash condition percentages are included.

    Returns:
        Tuple of (categorical_columns, numerical_columns).
        Currently no categorical columns for hourly aggregation.
    """
    if weather_lag_columns is None:
        weather_lag_columns = DEFAULT_WEATHER_LAG_COLUMNS
    if lag_hours is None:
        lag_hours = [1, 2, 3]
    if time_config is None:
        time_config = DEFAULT_TIME_CONFIG

    categorical_cols: list[str] = []

    numerical_cols: list[str] = []

    # Time features from config
    numerical_cols.extend(time_config.get_feature_columns())

    # Current weather values
    if include_current_weather:
        numerical_cols.extend(weather_lag_columns)

    # Lag features
    for col in weather_lag_columns:
        for lag in lag_hours:
            numerical_cols.append(f"{col}_lag_{lag}h")

    # Holiday feature
    if include_holidays:
        numerical_cols.append("is_holiday")

    # Condition features
    if include_conditions:
        numerical_cols.extend(["pct_darkness", "pct_wet_road", "pct_bad_weather"])

    return categorical_cols, numerical_cols
