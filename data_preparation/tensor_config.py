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
# Enhanced Configuration (Research-Based)
# =============================================================================

# Original crash columns (from filtered schema)
_BASE_CATEGORICAL_COLS = [
    "TRAFFIC_CONTROL_DEVICE",
    "DEVICE_CONDITION",
    "WEATHER_CONDITION",
    "LIGHTING_CONDITION",
    "ROADWAY_SURFACE_COND",
    "ROAD_DEFECT",
]

_BASE_NUMERICAL_COLS = [
    "POSTED_SPEED_LIMIT",
    "LANE_CNT",
    "CRASH_HOUR",
    "CRASH_DAY_OF_WEEK",
    "CRASH_MONTH",
    "LATITUDE",
    "LONGITUDE",
]

# Engineered temporal features
_ENGINEERED_TEMPORAL_CATEGORICAL = [
    "TIME_OF_DAY",      # Morning/Afternoon/Evening/Night
    "SEASON",           # Winter/Spring/Summer/Fall
]

_ENGINEERED_TEMPORAL_NUMERICAL = [
    "IS_PEAK_HOUR",     # Rush hour indicator
    "IS_WEEKEND",       # Weekend indicator
    "IS_NIGHT",         # Night indicator
]

# Vehicle features (from triple merge aggregation)
_VEHICLE_CATEGORICAL = [
    "VEHICLE_AGE_CATEGORY",  # New/Mid/Old/Very Old
]

_VEHICLE_NUMERICAL = [
    "VEHICLE_COUNT",         # Number of vehicles in crash
    "VEHICLE_AGE",           # Age of oldest vehicle
    "OLD_VEHICLE_FLAG",      # Boolean: vehicle > 10 years
    "ANY_SPEED_VIOLATION",   # Boolean: any vehicle exceeded limit
]

# People features (from triple merge aggregation)
_PEOPLE_NUMERICAL = [
    "PERSON_COUNT",          # Total people involved
    "DRIVER_COUNT",          # Number of drivers
    "AVG_AGE",               # Average age of people
    "MAX_BAC",               # Maximum blood alcohol content
    "ANY_BAC_POSITIVE",      # Boolean: any positive BAC
    "SEATBELT_USAGE_RATE",   # Proportion using seatbelts
    "ANY_CELL_PHONE_USE",    # Boolean: distraction
]

# Interaction/derived features
_INTERACTION_NUMERICAL = [
    "ADVERSE_CONDITIONS_COUNT",  # Count of adverse conditions
    "NIGHT_POOR_LIGHTING",       # Night + dark interaction
    "WET_ROAD",                  # Wet road surface
    "IMPAIRED_DRIVER",           # BAC positive flag
    "HIGH_SPEED_AREA",           # Speed limit >= 40
    "MULTI_VEHICLE",             # > 2 vehicles
]

# Weather features (from weather merge)
_WEATHER_NUMERICAL = [
    "Air Temperature",
    "Humidity",
    "Rain Intensity",
    "Wind Speed",
]

# Spatial cluster feature
_CLUSTER_CATEGORICAL = [
    "LOCATION_CLUSTER",      # K-means cluster assignment
]


# Combined enhanced feature lists
_ENHANCED_CATEGORICAL_COLS = (
    _BASE_CATEGORICAL_COLS +
    _ENGINEERED_TEMPORAL_CATEGORICAL +
    _VEHICLE_CATEGORICAL +
    _CLUSTER_CATEGORICAL
)

_ENHANCED_NUMERICAL_COLS = (
    _BASE_NUMERICAL_COLS +
    _ENGINEERED_TEMPORAL_NUMERICAL +
    _VEHICLE_NUMERICAL +
    _PEOPLE_NUMERICAL +
    _INTERACTION_NUMERICAL +
    _WEATHER_NUMERICAL
)


ENHANCED_SEVERITY_CONFIG = TensorConfig(
    target_column="MOST_SEVERE_INJURY",
    feature_columns=_ENHANCED_CATEGORICAL_COLS + _ENHANCED_NUMERICAL_COLS,
    categorical_columns=_ENHANCED_CATEGORICAL_COLS,
    numerical_columns=_ENHANCED_NUMERICAL_COLS,
    task_type="classification",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)
"""
Enhanced severity prediction config using research-based features:
- Triple merge (crash + vehicle + people + weather)
- Temporal features (peak hour, weekend, season)
- Vehicle age features
- People aggregation (BAC, seatbelt usage)
- Interaction features
- Spatial clustering
"""


# Moderate config - uses triple merge but not all engineered features
MODERATE_SEVERITY_CONFIG = TensorConfig(
    target_column="MOST_SEVERE_INJURY",
    feature_columns=(
        _BASE_CATEGORICAL_COLS +
        _BASE_NUMERICAL_COLS +
        _ENGINEERED_TEMPORAL_NUMERICAL +
        ["VEHICLE_COUNT", "AVG_AGE", "ANY_BAC_POSITIVE", "SEATBELT_USAGE_RATE"]
    ),
    categorical_columns=_BASE_CATEGORICAL_COLS,
    numerical_columns=(
        _BASE_NUMERICAL_COLS +
        _ENGINEERED_TEMPORAL_NUMERICAL +
        ["VEHICLE_COUNT", "AVG_AGE", "ANY_BAC_POSITIVE", "SEATBELT_USAGE_RATE"]
    ),
    task_type="classification",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)
"""Moderate config with triple merge data but fewer engineered features."""
