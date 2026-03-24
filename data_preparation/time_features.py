"""Configurable time feature generation for crash prediction.

This module provides:
- TimeFeatureConfig: Dataclass for configuring which time features to generate
- Feature generation functions for cyclical (sin/cos) and binary time patterns

Example:
    >>> config = TimeFeatureConfig(
    ...     include_hour_cycle=True,
    ...     include_is_night=True,
    ...     night_start=22,
    ...     night_end=6,
    ... )
    >>> df = add_time_features(df, config)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TimeFeatureConfig:
    """Configuration for time-based feature generation.

    Controls which temporal features are added to the dataset and their parameters.

    Attributes:
        include_hour_cycle: Add hour_sin, hour_cos (24-hour cycle).
        include_day_of_week_cycle: Add day_of_week_sin, day_of_week_cos (7-day cycle).
        include_month_cycle: Add month_sin, month_cos (12-month cycle).
        include_is_night: Add binary is_night feature.
        include_is_rush_hour: Add binary is_rush_hour feature.
        include_is_weekend: Add binary is_weekend feature.
        night_start: Hour when night begins (inclusive, 0-23).
        night_end: Hour when night ends (exclusive, 0-23).
        rush_morning_start: Morning rush hour start (inclusive).
        rush_morning_end: Morning rush hour end (exclusive).
        rush_evening_start: Evening rush hour start (inclusive).
        rush_evening_end: Evening rush hour end (exclusive).

    Example:
        >>> config = TimeFeatureConfig(
        ...     include_hour_cycle=True,
        ...     include_is_night=True,
        ...     night_start=23,  # 11 PM
        ...     night_end=5,     # 5 AM
        ... )
        >>> config.get_feature_columns()
        ['hour_sin', 'hour_cos', 'is_night']
    """

    # Cyclical feature toggles
    include_hour_cycle: bool = True
    include_day_of_week_cycle: bool = True
    include_month_cycle: bool = True

    # Binary feature toggles
    include_is_night: bool = False
    include_is_rush_hour: bool = False
    include_is_weekend: bool = False

    # Night thresholds (default: 10 PM to 6 AM)
    night_start: int = 22
    night_end: int = 6

    # Rush hour thresholds (default: 7-9 AM and 4-7 PM)
    rush_morning_start: int = 7
    rush_morning_end: int = 9
    rush_evening_start: int = 16
    rush_evening_end: int = 19

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate hour ranges
        for attr in [
            "night_start",
            "night_end",
            "rush_morning_start",
            "rush_morning_end",
            "rush_evening_start",
            "rush_evening_end",
        ]:
            value = getattr(self, attr)
            if not 0 <= value <= 23:
                raise ValueError(f"{attr} must be between 0 and 23, got {value}")

        # Validate rush hour ordering
        if self.rush_morning_start >= self.rush_morning_end:
            raise ValueError(
                f"rush_morning_start ({self.rush_morning_start}) must be less than "
                f"rush_morning_end ({self.rush_morning_end})"
            )
        if self.rush_evening_start >= self.rush_evening_end:
            raise ValueError(
                f"rush_evening_start ({self.rush_evening_start}) must be less than "
                f"rush_evening_end ({self.rush_evening_end})"
            )

    def get_feature_columns(self) -> list[str]:
        """Return list of feature column names that will be generated.

        Returns:
            List of column names based on enabled features.

        Example:
            >>> config = TimeFeatureConfig(
            ...     include_hour_cycle=True,
            ...     include_day_of_week_cycle=False,
            ...     include_month_cycle=False,
            ...     include_is_weekend=True,
            ... )
            >>> config.get_feature_columns()
            ['hour_sin', 'hour_cos', 'is_weekend']
        """
        columns: list[str] = []

        if self.include_hour_cycle:
            columns.extend(["hour_sin", "hour_cos"])

        if self.include_day_of_week_cycle:
            columns.extend(["day_of_week_sin", "day_of_week_cos"])

        if self.include_month_cycle:
            columns.extend(["month_sin", "month_cos"])

        if self.include_is_night:
            columns.append("is_night")

        if self.include_is_rush_hour:
            columns.append("is_rush_hour")

        if self.include_is_weekend:
            columns.append("is_weekend")

        return columns


# Default configuration with all cyclical features enabled (backward compatible)
DEFAULT_TIME_CONFIG = TimeFeatureConfig(
    include_hour_cycle=True,
    include_day_of_week_cycle=True,
    include_month_cycle=True,
    include_is_night=False,
    include_is_rush_hour=False,
    include_is_weekend=False,
)
"""Default config matching original behavior: only cyclical features, no binary."""


# Enhanced configuration with all features enabled
FULL_TIME_CONFIG = TimeFeatureConfig(
    include_hour_cycle=True,
    include_day_of_week_cycle=True,
    include_month_cycle=True,
    include_is_night=True,
    include_is_rush_hour=True,
    include_is_weekend=True,
)
"""Full config with all time features enabled."""


def add_cyclical_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "hour_timestamp",
    config: TimeFeatureConfig | None = None,
) -> pd.DataFrame:
    """Add cyclical (sin/cos) encodings for temporal features.

    Encodes hour of day, day of week, and/or month as sin/cos pairs
    to capture the cyclical nature of time (e.g., hour 23 is close to hour 0).

    Args:
        df: DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column.
        config: Configuration controlling which cyclical features to add.
            If None, uses DEFAULT_TIME_CONFIG.

    Returns:
        DataFrame with additional cyclical time columns based on config:
            - hour_sin, hour_cos (if include_hour_cycle)
            - day_of_week_sin, day_of_week_cos (if include_day_of_week_cycle)
            - month_sin, month_cos (if include_month_cycle)

    Example:
        >>> df = pd.DataFrame({
        ...     "hour_timestamp": pd.date_range("2024-01-01 22:00", periods=4, freq="h")
        ... })
        >>> config = TimeFeatureConfig(include_hour_cycle=True, include_month_cycle=False)
        >>> result = add_cyclical_time_features(df, config=config)
        >>> "hour_sin" in result.columns
        True
        >>> "month_sin" in result.columns
        False
    """
    if config is None:
        config = DEFAULT_TIME_CONFIG

    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    if config.include_hour_cycle:
        # Hour of day (0-23) -> 24-hour cycle
        hour = df[timestamp_col].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    if config.include_day_of_week_cycle:
        # Day of week (0-6 in pandas, Monday=0) -> 7-day cycle
        day_of_week = df[timestamp_col].dt.dayofweek
        df["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    if config.include_month_cycle:
        # Month (1-12) -> 12-month cycle
        month = df[timestamp_col].dt.month
        df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    return df


def add_binary_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "hour_timestamp",
    config: TimeFeatureConfig | None = None,
) -> pd.DataFrame:
    """Add binary time features based on configuration.

    Creates interpretable binary indicators for common temporal patterns:
    - is_night: Whether the hour falls within nighttime
    - is_rush_hour: Whether the hour is during morning or evening rush
    - is_weekend: Whether the day is Saturday or Sunday

    Args:
        df: DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column.
        config: Configuration controlling thresholds and which features to add.
            If None, uses DEFAULT_TIME_CONFIG (no binary features).

    Returns:
        DataFrame with additional binary columns based on config:
            - is_night (int, 0 or 1)
            - is_rush_hour (int, 0 or 1)
            - is_weekend (int, 0 or 1)

    Example:
        >>> df = pd.DataFrame({
        ...     "hour_timestamp": pd.to_datetime([
        ...         "2024-01-01 02:00",  # Night, Monday
        ...         "2024-01-06 08:00",  # Rush hour, Saturday
        ...         "2024-01-01 14:00",  # Afternoon, Monday
        ...     ])
        ... })
        >>> config = TimeFeatureConfig(
        ...     include_is_night=True,
        ...     include_is_rush_hour=True,
        ...     include_is_weekend=True,
        ... )
        >>> result = add_binary_time_features(df, config=config)
        >>> result["is_night"].tolist()
        [1, 0, 0]
        >>> result["is_weekend"].tolist()
        [0, 1, 0]
    """
    if config is None:
        config = DEFAULT_TIME_CONFIG

    df = df.copy()

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    hour = df[timestamp_col].dt.hour

    if config.include_is_night:
        # Handle wrap-around (e.g., 22:00 to 06:00)
        if config.night_start > config.night_end:
            # Night spans midnight: e.g., 22 <= hour OR hour < 6
            is_night = (hour >= config.night_start) | (hour < config.night_end)
        else:
            # Night within same day: e.g., 0 <= hour < 6 (unusual but supported)
            is_night = (hour >= config.night_start) & (hour < config.night_end)
        df["is_night"] = is_night.astype(int)

    if config.include_is_rush_hour:
        is_morning_rush = (hour >= config.rush_morning_start) & (
            hour < config.rush_morning_end
        )
        is_evening_rush = (hour >= config.rush_evening_start) & (
            hour < config.rush_evening_end
        )
        df["is_rush_hour"] = (is_morning_rush | is_evening_rush).astype(int)

    if config.include_is_weekend:
        day_of_week = df[timestamp_col].dt.dayofweek
        # Saturday=5, Sunday=6
        df["is_weekend"] = (day_of_week >= 5).astype(int)

    return df


def add_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "hour_timestamp",
    config: TimeFeatureConfig | None = None,
) -> pd.DataFrame:
    """Add all configured time features to a DataFrame.

    This is the main entry point that applies both cyclical and binary
    time features based on the configuration.

    Args:
        df: DataFrame with a timestamp column.
        timestamp_col: Name of the timestamp column.
        config: Configuration controlling which features to add.
            If None, uses DEFAULT_TIME_CONFIG.

    Returns:
        DataFrame with additional time feature columns.

    Example:
        >>> config = TimeFeatureConfig(
        ...     include_hour_cycle=True,
        ...     include_day_of_week_cycle=False,
        ...     include_month_cycle=False,
        ...     include_is_night=True,
        ...     include_is_weekend=True,
        ... )
        >>> df = pd.DataFrame({
        ...     "hour_timestamp": pd.date_range("2024-01-06 23:00", periods=3, freq="h")
        ... })
        >>> result = add_time_features(df, config=config)
        >>> sorted(result.columns.tolist())
        ['hour_cos', 'hour_sin', 'hour_timestamp', 'is_night', 'is_weekend']
    """
    if config is None:
        config = DEFAULT_TIME_CONFIG

    df = add_cyclical_time_features(df, timestamp_col=timestamp_col, config=config)
    df = add_binary_time_features(df, timestamp_col=timestamp_col, config=config)

    return df
