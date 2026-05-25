"""Time-series feature engineering for crash count prediction.

Creates lag features, rolling statistics, trend indicators, and cyclical
time encodings for temporal regression models.
"""

from __future__ import annotations

<<<<<<< HEAD
import logging
=======
>>>>>>> origin/dev
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from data_preparation.aggregate_time_series import TimeGranularity

<<<<<<< HEAD
logger = logging.getLogger(__name__)

=======
>>>>>>> origin/dev

@dataclass
class FeatureConfig:
    """Configuration for time-series feature engineering.

    Attributes:
        granularity: Time bucket size (affects default lag/window sizes).
        lag_periods: List of lag periods to create (e.g., [1, 2, 3] for t-1, t-2, t-3).
        rolling_windows: List of rolling window sizes for statistics.
        include_trends: Whether to include momentum/change features.
        include_cyclical: Whether to include sin/cos time encodings.
        target_col: Name of the target column to create features for.
    """

    granularity: TimeGranularity = "hourly"
    lag_periods: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24])
    rolling_windows: list[int] = field(default_factory=lambda: [6, 12, 24, 48])
    include_trends: bool = True
    include_cyclical: bool = True
    target_col: str = "crash_count"

    @classmethod
    def for_granularity(cls, granularity: TimeGranularity) -> "FeatureConfig":
        """Create config with appropriate defaults for the granularity.

        Different granularities need different lag/window sizes:
        - Hourly: More lags to capture daily patterns
        - Daily: Fewer lags, weekly seasonality
        - Weekly: Even fewer, capture monthly patterns
        """
        if granularity == "hourly":
            return cls(
                granularity=granularity,
                lag_periods=[1, 2, 3, 6, 12, 24, 48, 168],  # up to 1 week
                rolling_windows=[6, 12, 24, 48, 168],
            )
        elif granularity == "daily":
            return cls(
                granularity=granularity,
                lag_periods=[1, 2, 3, 7, 14, 30],  # up to 1 month
                rolling_windows=[3, 7, 14, 30],
            )
        else:  # weekly
            return cls(
                granularity=granularity,
                lag_periods=[1, 2, 4, 8, 12],  # up to 3 months
                rolling_windows=[2, 4, 8, 12],
            )


def create_lag_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Create lagged versions of the target variable.

    Shifts the target column by N periods to create lag_1, lag_2, etc.
    These capture auto-regressive patterns.

    Args:
        df: DataFrame with target column, sorted by time.
        config: Feature configuration.
        group_col: Column to group by (e.g., 'zone_id') for per-group lags.

    Returns:
        DataFrame with lag_N columns added.
    """
    if config is None:
        config = FeatureConfig()

    df = df.copy()
    target = config.target_col

    for lag in config.lag_periods:
        col_name = f"lag_{lag}"
        if group_col and group_col in df.columns:
            df[col_name] = df.groupby(group_col)[target].shift(lag)
        else:
            df[col_name] = df[target].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Create rolling window statistics.

    Computes rolling mean, std, min, max over different window sizes.
    Uses shift(1) to prevent data leakage from current period.

    Args:
        df: DataFrame with target column, sorted by time.
        config: Feature configuration.
        group_col: Column to group by for per-group statistics.

    Returns:
        DataFrame with rolling statistic columns added.
    """
    if config is None:
        config = FeatureConfig()

    df = df.copy()
    target = config.target_col

    for window in config.rolling_windows:
        prefix = f"roll_{window}"

        if group_col and group_col in df.columns:
            # Per-group rolling statistics
            rolling = df.groupby(group_col)[target].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1)
            )
            # Since transform doesn't support multiple operations, we do them separately
            df[f"{prefix}_mean"] = df.groupby(group_col)[target].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            df[f"{prefix}_std"] = df.groupby(group_col)[target].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
            df[f"{prefix}_min"] = df.groupby(group_col)[target].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
            )
            df[f"{prefix}_max"] = df.groupby(group_col)[target].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
            )
        else:
            # Global rolling statistics (shift by 1 to avoid leakage)
            shifted = df[target].shift(1)
            rolling = shifted.rolling(window=window, min_periods=1)
            df[f"{prefix}_mean"] = rolling.mean()
            df[f"{prefix}_std"] = rolling.std()
            df[f"{prefix}_min"] = rolling.min()
            df[f"{prefix}_max"] = rolling.max()

    # Fill NaN in std columns (happens when window has single value)
    std_cols = [c for c in df.columns if c.endswith("_std")]
    df[std_cols] = df[std_cols].fillna(0)

    return df


def add_trend_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add trend/momentum features.

    Computes:
    - diff_1: Change from previous period (momentum)
    - pct_change_1: Percent change from previous period
    - diff_from_mean: Deviation from rolling mean

    Args:
        df: DataFrame with target and rolling features.
        config: Feature configuration.
        group_col: Column to group by for per-group trends.

    Returns:
        DataFrame with trend columns added.
    """
    if config is None:
        config = FeatureConfig()

    if not config.include_trends:
        return df

    df = df.copy()
    target = config.target_col

    if group_col and group_col in df.columns:
        df["diff_1"] = df.groupby(group_col)[target].diff()
        df["pct_change_1"] = df.groupby(group_col)[target].pct_change()
    else:
        df["diff_1"] = df[target].diff()
        df["pct_change_1"] = df[target].pct_change()

    # Deviation from rolling mean (if available)
    if "roll_24_mean" in df.columns:
        df["diff_from_roll_mean"] = df[target] - df["roll_24_mean"]
    elif "roll_7_mean" in df.columns:
        df["diff_from_roll_mean"] = df[target] - df["roll_7_mean"]

    # Handle inf values from pct_change (division by zero)
    df["pct_change_1"] = df["pct_change_1"].replace([np.inf, -np.inf], 0)

    return df


def add_cyclical_time_encoding(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Add cyclical sin/cos encodings for time features.

    Encodes periodic features (hour, day-of-week, month) as sin/cos pairs
    to avoid artificial discontinuities (e.g., 23 -> 0 for hours).

    Args:
        df: DataFrame with timestamp column.
        timestamp_col: Name of the datetime column.
        config: Feature configuration.

    Returns:
        DataFrame with cyclical time columns added.
    """
    if config is None:
        config = FeatureConfig()

    if not config.include_cyclical:
        return df

    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    ts = df[timestamp_col]

    # Hour of day (0-23) -> sin/cos with period 24
    if config.granularity == "hourly":
        hour = ts.dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day of week (0-6) -> sin/cos with period 7
    dow = ts.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Month (1-12) -> sin/cos with period 12
    month = ts.dt.month
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # Day of month (1-31) -> sin/cos with period ~30
    if config.granularity in ["hourly", "daily"]:
        day = ts.dt.day
        df["day_sin"] = np.sin(2 * np.pi * (day - 1) / 30)
        df["day_cos"] = np.cos(2 * np.pi * (day - 1) / 30)

    return df


def add_calendar_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Add non-cyclical calendar features.

    Adds binary/categorical features like is_weekend, is_holiday, etc.

    Args:
        df: DataFrame with timestamp column.
        timestamp_col: Name of the datetime column.

    Returns:
        DataFrame with calendar feature columns added.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    ts = df[timestamp_col]

    # Weekend indicator
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Rush hour indicator (morning 7-9, evening 16-19)
    hour = ts.dt.hour
    df["is_rush_hour"] = ((hour >= 7) & (hour <= 9) | (hour >= 16) & (hour <= 19)).astype(int)

    # Night indicator (22-6)
    df["is_night"] = ((hour >= 22) | (hour <= 6)).astype(int)

    return df


def engineer_time_series_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    group_col: str | None = "zone_id",
    timestamp_col: str = "timestamp",
    drop_na: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply all time-series feature engineering.

    Main entry point that applies lag features, rolling statistics,
    trend features, and cyclical time encodings.

    Args:
        df: Aggregated DataFrame with timestamp, zone_id, crash_count.
        config: Feature configuration.
        group_col: Column to group by for per-zone features.
        timestamp_col: Name of timestamp column.
        drop_na: Whether to drop rows with NaN from lag/rolling operations.

    Returns:
        Tuple of:
        - DataFrame with all features added
        - List of feature column names (excluding target and metadata)
    """
    if config is None:
        config = FeatureConfig()

    # Sort by time (and zone if applicable)
    sort_cols = [timestamp_col]
    if group_col and group_col in df.columns:
        sort_cols.append(group_col)
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Apply feature engineering steps
    df = create_lag_features(df, config, group_col)
    df = create_rolling_features(df, config, group_col)
    df = add_trend_features(df, config, group_col)
    df = add_cyclical_time_encoding(df, timestamp_col, config)
    df = add_calendar_features(df, timestamp_col)

    # Determine feature columns (exclude target, timestamp, zone_id)
    exclude_cols = {timestamp_col, config.target_col, "zone_id"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Drop rows with NaN (from lag/rolling operations)
    if drop_na:
        n_before = len(df)
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
<<<<<<< HEAD
            logger.info(f"Dropped {n_dropped} rows with NaN from lag/rolling features")
=======
            print(f"Dropped {n_dropped} rows with NaN from lag/rolling features")
>>>>>>> origin/dev

    return df, feature_cols


def create_prediction_features(
    recent_history: pd.DataFrame,
    config: FeatureConfig,
    prediction_time: pd.Timestamp,
    zone_id: int | None = None,
) -> dict[str, float]:
    """Create feature vector for a single prediction.

    Used at inference time to construct features for predicting
    the crash count at a future time point.

    Args:
        recent_history: Recent crash counts (enough to fill all lags/windows).
        config: Feature configuration.
        prediction_time: Timestamp to predict for.
        zone_id: Zone ID if doing per-zone prediction.

    Returns:
        Dictionary of feature name -> value for the prediction point.
    """
    features = {}
    target = config.target_col

    # Get the most recent counts as a series
    if zone_id is not None and "zone_id" in recent_history.columns:
        history = recent_history[recent_history["zone_id"] == zone_id].copy()
    else:
        history = recent_history.copy()

    history = history.sort_values("timestamp")
    counts = history[target].values

    # Lag features (relative to prediction time, so lag_1 = most recent actual)
    for lag in config.lag_periods:
        if lag <= len(counts):
            features[f"lag_{lag}"] = counts[-lag]
        else:
            features[f"lag_{lag}"] = 0

    # Rolling features (over available history)
    for window in config.rolling_windows:
        window_data = counts[-window:] if len(counts) >= window else counts
        features[f"roll_{window}_mean"] = np.mean(window_data) if len(window_data) > 0 else 0
        features[f"roll_{window}_std"] = np.std(window_data) if len(window_data) > 1 else 0
        features[f"roll_{window}_min"] = np.min(window_data) if len(window_data) > 0 else 0
        features[f"roll_{window}_max"] = np.max(window_data) if len(window_data) > 0 else 0

    # Trend features
    if config.include_trends and len(counts) >= 2:
        features["diff_1"] = counts[-1] - counts[-2]
        features["pct_change_1"] = (
            (counts[-1] - counts[-2]) / counts[-2] if counts[-2] != 0 else 0
        )
        roll_mean = features.get("roll_24_mean", features.get("roll_7_mean", np.mean(counts)))
        features["diff_from_roll_mean"] = counts[-1] - roll_mean

    # Cyclical time encoding
    if config.include_cyclical:
        if config.granularity == "hourly":
            features["hour_sin"] = np.sin(2 * np.pi * prediction_time.hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * prediction_time.hour / 24)

        features["dow_sin"] = np.sin(2 * np.pi * prediction_time.dayofweek / 7)
        features["dow_cos"] = np.cos(2 * np.pi * prediction_time.dayofweek / 7)
        features["month_sin"] = np.sin(2 * np.pi * (prediction_time.month - 1) / 12)
        features["month_cos"] = np.cos(2 * np.pi * (prediction_time.month - 1) / 12)

        if config.granularity in ["hourly", "daily"]:
            features["day_sin"] = np.sin(2 * np.pi * (prediction_time.day - 1) / 30)
            features["day_cos"] = np.cos(2 * np.pi * (prediction_time.day - 1) / 30)

    # Calendar features
    features["is_weekend"] = int(prediction_time.dayofweek >= 5)
    features["is_rush_hour"] = int(
        (7 <= prediction_time.hour <= 9) or (16 <= prediction_time.hour <= 19)
    )
    features["is_night"] = int(prediction_time.hour >= 22 or prediction_time.hour <= 6)

    return features
