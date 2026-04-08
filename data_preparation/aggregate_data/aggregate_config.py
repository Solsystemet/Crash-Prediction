"""Configuration for aggregate hourly injury prediction.

This module defines the configuration for aggregating crash data by hour
and preparing it for neural network training.
"""

from dataclasses import dataclass, field
from typing import Literal


# Default target columns - the 5 injury types to predict
DEFAULT_TARGET_COLUMNS = [
    "INJURIES_FATAL",
    "INJURIES_INCAPACITATING",
    "INJURIES_NON_INCAPACITATING",
    "INJURIES_REPORTED_NOT_EVIDENT",
    "INJURIES_NO_INDICATION",
]

# Time-based feature columns
TIME_FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "month",
    "year",
]

# Weather feature columns (from weather stations data)
WEATHER_FEATURE_COLUMNS = [
    "Air Temperature",
    "Humidity",
    "Rain Intensity",
    "Total Rain",
    "Precipitation Type",
]


@dataclass
class AggregateConfig:
    """Configuration for aggregate hourly injury prediction.

    Attributes:
        target_columns: List of injury columns to predict (summed per hour).
        time_features: Time-based feature columns to include.
        weather_features: Weather feature columns to include.
        use_clusters: Whether to aggregate per geographic cluster instead of city-wide.
        n_clusters: Number of K-means clusters if use_clusters is True.
        train_ratio: Proportion of data for training set.
        val_ratio: Proportion of data for validation set.
        test_ratio: Proportion of data for test set.
        random_seed: Seed for reproducibility.
        fill_numerical_na: Strategy for missing numerical values.
        log_transform_targets: Whether to apply log(1+x) transform to targets.
    """

    target_columns: list[str] = field(
        default_factory=lambda: DEFAULT_TARGET_COLUMNS.copy()
    )
    time_features: list[str] = field(
        default_factory=lambda: TIME_FEATURE_COLUMNS.copy()
    )
    weather_features: list[str] = field(
        default_factory=lambda: WEATHER_FEATURE_COLUMNS.copy()
    )
    use_clusters: bool = False
    n_clusters: int | None = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    fill_numerical_na: Literal["median", "mean", "zero"] = "zero"
    log_transform_targets: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio})"
            )

        if self.train_ratio <= 0 or self.val_ratio < 0 or self.test_ratio <= 0:
            raise ValueError(
                "train_ratio and test_ratio must be > 0, val_ratio must be >= 0"
            )

        if self.use_clusters and self.n_clusters is None:
            raise ValueError("n_clusters must be specified when use_clusters is True")

    @property
    def feature_columns(self) -> list[str]:
        """All feature columns (time + weather)."""
        features = self.time_features.copy()
        features.extend(self.weather_features)
        if self.use_clusters:
            features.append("cluster")
        return features

    @property
    def num_features(self) -> int:
        """Number of input features."""
        return len(self.feature_columns)

    @property
    def num_targets(self) -> int:
        """Number of target columns to predict."""
        return len(self.target_columns)


# =============================================================================
# Preset Configurations
# =============================================================================

CITYWIDE_HOURLY_CONFIG = AggregateConfig(
    target_columns=DEFAULT_TARGET_COLUMNS.copy(),
    time_features=TIME_FEATURE_COLUMNS.copy(),
    weather_features=WEATHER_FEATURE_COLUMNS.copy(),
    use_clusters=False,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)
"""Predict total hourly injuries across entire Chicago."""


def create_cluster_config(
    n_clusters: int = 10,
    log_transform_targets: bool = False,
) -> AggregateConfig:
    """Create a configuration for per-cluster hourly prediction.

    Args:
        n_clusters: Number of geographic clusters to use.
        log_transform_targets: Whether to apply log(1+x) transform to targets.

    Returns:
        AggregateConfig configured for per-cluster prediction.
    """
    return AggregateConfig(
        target_columns=DEFAULT_TARGET_COLUMNS.copy(),
        time_features=TIME_FEATURE_COLUMNS.copy(),
        weather_features=WEATHER_FEATURE_COLUMNS.copy(),
        use_clusters=True,
        n_clusters=n_clusters,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        log_transform_targets=log_transform_targets,
    )
