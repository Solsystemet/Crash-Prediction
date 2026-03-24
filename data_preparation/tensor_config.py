"""Configuration for tensor data preparation pipeline.

This module defines the TensorConfig dataclass for configuring how
DataFrame data is converted to PyTorch tensors for training.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TensorConfig:
    """Configuration for converting DataFrame to PyTorch tensors.

    Attributes:
        target_column: Column name to use as prediction target.
        feature_columns: List of column names to use as input features.
            If None, all columns except target_column are used.
        categorical_columns: Columns to apply label encoding to.
        numerical_columns: Columns to apply standard scaling to.
        task_type: Type of ML task - determines label tensor dtype.
            "classification" -> LongTensor, "regression" -> FloatTensor.
        train_ratio: Proportion of data for training set.
        val_ratio: Proportion of data for validation set.
        test_ratio: Proportion of data for test set.
        random_seed: Seed for reproducible train/val/test splits.
        fill_categorical_na: Value to fill missing categorical values.
        fill_numerical_na: Strategy for missing numerical values ("median" or "mean").
        split_by_time: If True, split data temporally (train < val < test in time).
            If False, use random shuffled split.
    """

    target_column: str
    feature_columns: list[str] | None = None
    categorical_columns: list[str] = field(default_factory=list)
    numerical_columns: list[str] = field(default_factory=list)
    task_type: Literal["classification", "regression"] = "classification"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    fill_categorical_na: str = "UNKNOWN"
    fill_numerical_na: Literal["median", "mean"] = "median"
    split_by_time: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )

        if self.train_ratio <= 0 or self.val_ratio < 0 or self.test_ratio <= 0:
            raise ValueError("train_ratio and test_ratio must be > 0, val_ratio must be >= 0")


# =============================================================================
# Preset Configurations
# =============================================================================

# Columns from FilteredTrafficCrashesSchema
_CATEGORICAL_COLS = [
    "TRAFFIC_CONTROL_DEVICE",
    "DEVICE_CONDITION",
    "WEATHER_CONDITION",
    "LIGHTING_CONDITION",
    "ROADWAY_SURFACE_COND",
    "ROAD_DEFECT",
]

_NUMERICAL_COLS = [
    "POSTED_SPEED_LIMIT",
    "LANE_CNT",
    "CRASH_HOUR",
    "CRASH_DAY_OF_WEEK",
    "CRASH_MONTH",
    "LATITUDE",
    "LONGITUDE",
]

_INJURY_COLS = [
    "INJURIES_FATAL",
    "INJURIES_INCAPACITATING",
    "INJURIES_NON_INCAPACITATING",
    "INJURIES_REPORTED_NOT_EVIDENT",
    "INJURIES_NO_INDICATION",
    "INJURIES_UNKNOWN",
]


SEVERITY_PREDICTION_CONFIG = TensorConfig(
    target_column="MOST_SEVERE_INJURY",
    feature_columns=_CATEGORICAL_COLS + _NUMERICAL_COLS,
    categorical_columns=_CATEGORICAL_COLS,
    numerical_columns=_NUMERICAL_COLS,
    task_type="classification",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)
"""Predict crash severity (MOST_SEVERE_INJURY) from road/weather/time conditions."""


FATAL_CRASH_CONFIG = TensorConfig(
    target_column="INJURIES_FATAL",
    feature_columns=_CATEGORICAL_COLS + _NUMERICAL_COLS,
    categorical_columns=_CATEGORICAL_COLS,
    numerical_columns=_NUMERICAL_COLS + ["INJURIES_FATAL"],  # Target needs scaling too
    task_type="regression",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)
"""Predict number of fatal injuries (regression task)."""


# Example of a minimal config for quick testing
MINIMAL_TEST_CONFIG = TensorConfig(
    target_column="MOST_SEVERE_INJURY",
    feature_columns=["WEATHER_CONDITION", "POSTED_SPEED_LIMIT", "CRASH_HOUR"],
    categorical_columns=["WEATHER_CONDITION"],
    numerical_columns=["POSTED_SPEED_LIMIT", "CRASH_HOUR"],
    task_type="classification",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)
"""Minimal config with just 3 features for quick testing."""


# =============================================================================
# Hourly Crash Count Configuration
# =============================================================================

# Feature columns for hourly crash count prediction
# These match the output of prepare_hourly_data() with default settings
_HOURLY_CYCLICAL_FEATURES = [
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
]

_HOURLY_WEATHER_FEATURES = [
    "Air Temperature",
    "Humidity",
    "Rain Intensity",
    "Wind Speed",
]

_HOURLY_LAG_FEATURES = [
    f"{col}_lag_{lag}h"
    for col in _HOURLY_WEATHER_FEATURES
    for lag in [1, 2, 3]
]

_HOURLY_ALL_FEATURES = (
    _HOURLY_CYCLICAL_FEATURES + _HOURLY_WEATHER_FEATURES + _HOURLY_LAG_FEATURES
)

HOURLY_CRASH_COUNT_CONFIG = TensorConfig(
    target_column="crash_count",
    feature_columns=_HOURLY_ALL_FEATURES,
    categorical_columns=[],
    numerical_columns=_HOURLY_ALL_FEATURES,
    task_type="regression",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    split_by_time=True,
)
"""Config for hourly crash count prediction using weather and time features."""
