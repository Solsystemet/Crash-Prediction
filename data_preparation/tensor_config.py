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
        exclude_target_values: List of target values to exclude from training.
            Rows with these target values are filtered out before processing.
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
    exclude_target_values: list[str] = field(default_factory=list)

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
    "FIRST_CRASH_TYPE",
    "TRAFFICWAY_TYPE",
    "ALIGNMENT",
    "PRIM_CONTRIBUTORY_CAUSE",
    "DAMAGE",
    "INTERSECTION_RELATED_I",
]

_NUMERICAL_COLS = [
    "POSTED_SPEED_LIMIT",
    "LANE_CNT",
    "NUM_UNITS",
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
    exclude_target_values=["NO INDICATION OF INJURY"],
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
# People-Enhanced Fatality Prediction
# =============================================================================

# New features derived from people data aggregation
_PEOPLE_CATEGORICAL_COLS = [
    "has_unbelted",
    "has_alcohol",
    "has_elderly",
    "has_young_driver",
    "has_cellphone_use",
]

_PEOPLE_NUMERICAL_COLS = [
    "num_occupants",
    "num_pedestrians",
    "num_cyclists",
    "num_drivers",
    "age_min",
    "age_max",
    "age_mean",
]

# Combined feature lists for fatality prediction
_FATALITY_CATEGORICAL_COLS = _CATEGORICAL_COLS + _PEOPLE_CATEGORICAL_COLS
_FATALITY_NUMERICAL_COLS = _NUMERICAL_COLS + _PEOPLE_NUMERICAL_COLS


FATALITY_PREDICTION_CONFIG = TensorConfig(
    target_column="has_fatality",
    feature_columns=_FATALITY_CATEGORICAL_COLS + _FATALITY_NUMERICAL_COLS,
    categorical_columns=_FATALITY_CATEGORICAL_COLS,
    numerical_columns=_FATALITY_NUMERICAL_COLS,
    task_type="classification",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    fill_categorical_na="UNKNOWN",
    fill_numerical_na="median",
)
"""Binary fatality prediction using crash + people-derived features.

Uses merged crash/people data from merge_crash_with_people.py.
Target: has_fatality (1 if any person died, 0 otherwise).

New features from people data:
- Risk indicators: has_unbelted, has_alcohol, has_elderly, has_young_driver
- Demographics: num_occupants, num_pedestrians, age statistics
"""
