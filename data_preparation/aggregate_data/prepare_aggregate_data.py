"""Prepare aggregated hourly injury data for neural network training.

This module handles:
1. Loading and merging crash data with weather data
2. Aggregating injuries by hour (city-wide or per-cluster)
3. Creating train/val/test splits (chronological)
4. Converting to PyTorch tensors
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data_preparation.helpers.csv_loaders import (
    get_traffic_crashes,
    get_weather_stations,
)
from data_preparation.aggregate_data.aggregate_config import (
    AggregateConfig,
    CITYWIDE_HOURLY_CONFIG,
)
from splice.k_means import k_means, group_by_k_means


class AggregateDataset(Dataset):
    """PyTorch Dataset for aggregated hourly injury prediction.

    Attributes:
        features: FloatTensor of shape (num_samples, num_features).
        targets: FloatTensor of shape (num_samples, num_targets).
    """

    def __init__(self, features: Tensor, targets: Tensor) -> None:
        """Initialize the dataset.

        Args:
            features: Feature tensor.
            targets: Target tensor (multiple injury counts).
        """
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.targets[idx]


@dataclass
class AggregateDataResult:
    """Result of the aggregate data preparation pipeline.

    Attributes:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        feature_columns: List of feature column names.
        target_columns: List of target column names.
        feature_means: Mean values for each feature (for normalization).
        feature_stds: Std values for each feature (for normalization).
        log_transform_targets: Whether targets are log-transformed.
        loss_function: Loss function type used for training ("mse", "poisson", "negbin").
        config: Configuration used.
    """

    train_dataset: AggregateDataset
    val_dataset: AggregateDataset
    test_dataset: AggregateDataset
    feature_columns: list[str]
    target_columns: list[str]
    feature_means: Tensor
    feature_stds: Tensor
    log_transform_targets: bool
    loss_function: str
    config: AggregateConfig

    @property
    def num_features(self) -> int:
        """Number of input features."""
        return len(self.feature_columns)

    @property
    def num_targets(self) -> int:
        """Number of target columns."""
        return len(self.target_columns)

    def save(self, path: Path | str) -> None:
        """Save the result to disk.

        Args:
            path: Path to save the result (.pt file).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_features": self.train_dataset.features,
                "train_targets": self.train_dataset.targets,
                "val_features": self.val_dataset.features,
                "val_targets": self.val_dataset.targets,
                "test_features": self.test_dataset.features,
                "test_targets": self.test_dataset.targets,
                "feature_columns": self.feature_columns,
                "target_columns": self.target_columns,
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds,
                "log_transform_targets": self.log_transform_targets,
                "loss_function": self.loss_function,
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "AggregateDataResult":
        """Load a saved result from disk.

        Args:
            path: Path to the saved result (.pt file).

        Returns:
            Loaded AggregateDataResult.
        """
        path = Path(path)
        data = torch.load(path, weights_only=False)
        return cls(
            train_dataset=AggregateDataset(
                data["train_features"], data["train_targets"]
            ),
            val_dataset=AggregateDataset(data["val_features"], data["val_targets"]),
            test_dataset=AggregateDataset(data["test_features"], data["test_targets"]),
            feature_columns=data["feature_columns"],
            target_columns=data["target_columns"],
            feature_means=data["feature_means"],
            feature_stds=data["feature_stds"],
            log_transform_targets=data.get("log_transform_targets", False),
            loss_function=data.get("loss_function", "mse"),
            config=data["config"],
        )


def _merge_crash_with_weather(
    crash_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    crash_timestamp_col: str = "CRASH_DATE",
    weather_timestamp_col: str = "Measurement Timestamp",
) -> pd.DataFrame:
    """Merge crash data with weather data by rounding crash time to nearest hour.

    Args:
        crash_df: Crash data DataFrame.
        weather_df: Weather data DataFrame.
        crash_timestamp_col: Column name for crash timestamp.
        weather_timestamp_col: Column name for weather timestamp.

    Returns:
        Merged DataFrame with weather data joined.
    """
    crash_df = crash_df.copy()
    weather_df = weather_df.copy()

    # Parse timestamps
    crash_df[crash_timestamp_col] = pd.to_datetime(crash_df[crash_timestamp_col])
    weather_df[weather_timestamp_col] = pd.to_datetime(
        weather_df[weather_timestamp_col]
    )

    # Round crash time to nearest hour
    crash_df["_merge_hour"] = (
        crash_df[crash_timestamp_col] + pd.Timedelta(seconds=1)
    ).dt.round("h")

    # Create merge key for weather
    weather_df["_merge_hour"] = weather_df[weather_timestamp_col].dt.floor("h")

    # Drop duplicate weather entries for same hour (keep first)
    weather_df = weather_df.drop_duplicates(subset=["_merge_hour"], keep="first")

    # Merge on the rounded hour
    merged_df = crash_df.merge(
        weather_df, on="_merge_hour", how="left", suffixes=("", "_weather")
    )

    # Clean up merge column
    merged_df = merged_df.drop(columns=["_merge_hour"])

    return merged_df


def _aggregate_hourly(
    df: pd.DataFrame,
    target_columns: list[str],
    weather_features: list[str],
    use_clusters: bool = False,
) -> pd.DataFrame:
    """Aggregate crash data by hour, summing injury counts.

    Args:
        df: Merged crash+weather DataFrame.
        target_columns: Columns to sum (injury counts).
        weather_features: Weather columns to include (take mean per hour).
        use_clusters: Whether to aggregate per cluster.

    Returns:
        Aggregated DataFrame with one row per hour (or per hour+cluster).
    """
    # Create hourly timestamp
    df = df.copy()
    df["hour_timestamp"] = df["CRASH_DATE"].dt.floor("h")

    # Extract time features
    df["hour_of_day"] = df["hour_timestamp"].dt.hour
    df["day_of_week"] = df["hour_timestamp"].dt.dayofweek
    df["month"] = df["hour_timestamp"].dt.month
    df["year"] = df["hour_timestamp"].dt.year

    # Define groupby columns
    group_cols = ["hour_timestamp"]
    if use_clusters:
        group_cols.append("cluster")

    # Aggregation dict: sum injuries, mean weather, first time features
    agg_dict = {}

    # Sum injury columns
    for col in target_columns:
        if col in df.columns:
            agg_dict[col] = "sum"

    # Mean weather features
    for col in weather_features:
        if col in df.columns:
            agg_dict[col] = "mean"

    # Time features (same for all rows in group)
    for col in ["hour_of_day", "day_of_week", "month", "year"]:
        agg_dict[col] = "first"

    # If using clusters, we want to keep cluster as a feature
    if use_clusters and "cluster" in df.columns:
        # cluster is already in group_cols, so we don't aggregate it
        pass

    # Perform aggregation
    aggregated = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Sort by timestamp for chronological splitting
    aggregated = aggregated.sort_values("hour_timestamp").reset_index(drop=True)

    return aggregated


def _apply_clustering(
    df: pd.DataFrame, n_clusters: int, num_iterations: int = 100
) -> pd.DataFrame:
    """Apply K-means clustering to the crash data.

    Args:
        df: DataFrame with LATITUDE and LONGITUDE columns.
        n_clusters: Number of clusters.
        num_iterations: Number of K-means iterations.

    Returns:
        DataFrame with added 'cluster' column.
    """
    # Filter out rows with missing coordinates
    valid_coords = df["LATITUDE"].notna() & df["LONGITUDE"].notna()
    df_with_coords = df[valid_coords].copy()

    if len(df_with_coords) == 0:
        raise ValueError("No valid coordinates found for clustering")

    # Run K-means
    centroids, _ = k_means(df_with_coords, n_clusters, num_iterations)

    # Assign clusters to all rows with valid coordinates
    df_clustered = group_by_k_means(centroids, df_with_coords)

    return df_clustered


def _fill_missing_values(
    df: pd.DataFrame,
    columns: list[str],
    strategy: str = "zero",
) -> pd.DataFrame:
    """Fill missing values in specified columns.

    Args:
        df: DataFrame to fill.
        columns: Columns to fill.
        strategy: Fill strategy ("zero", "mean", "median").

    Returns:
        DataFrame with filled values.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        if strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
    return df


def _chronological_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically (no shuffling to avoid data leakage).

    Args:
        df: DataFrame sorted by time.
        train_ratio: Proportion for training.
        val_ratio: Proportion for validation.
        test_ratio: Proportion for testing.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def prepare_aggregate_data(
    config: AggregateConfig | None = None,
    verbose: bool = True,
) -> AggregateDataResult:
    """Prepare aggregated hourly data for training.

    This function:
    1. Loads crash and weather data
    2. Merges them by hour
    3. Optionally applies K-means clustering
    4. Aggregates injuries by hour (and optionally cluster)
    5. Creates chronological train/val/test splits
    6. Normalizes features
    7. Converts to PyTorch tensors

    Args:
        config: Configuration for data preparation. Uses default if None.
        verbose: Whether to print progress messages.

    Returns:
        AggregateDataResult containing datasets and metadata.
    """
    if config is None:
        config = CITYWIDE_HOURLY_CONFIG

    if verbose:
        print("Loading crash data...")
    crash_df = get_traffic_crashes()

    if verbose:
        print("Loading weather data...")
    weather_df = get_weather_stations()

    if verbose:
        print(f"Merging crash data ({len(crash_df)} rows) with weather data...")
    merged_df = _merge_crash_with_weather(crash_df, weather_df)

    # Apply clustering if requested
    if config.use_clusters and config.n_clusters is not None:
        if verbose:
            print(f"Applying K-means clustering with {config.n_clusters} clusters...")
        merged_df = _apply_clustering(merged_df, config.n_clusters)

    if verbose:
        print("Aggregating by hour...")
    aggregated_df = _aggregate_hourly(
        merged_df,
        target_columns=config.target_columns,
        weather_features=config.weather_features,
        use_clusters=config.use_clusters,
    )

    if verbose:
        print(f"Aggregated to {len(aggregated_df)} hourly records")

    # Determine actual feature columns (only those present in data)
    feature_columns = []
    for col in config.time_features:
        if col in aggregated_df.columns:
            feature_columns.append(col)
    for col in config.weather_features:
        if col in aggregated_df.columns:
            feature_columns.append(col)
    if config.use_clusters and "cluster" in aggregated_df.columns:
        feature_columns.append("cluster")

    # Determine actual target columns
    target_columns = [
        col for col in config.target_columns if col in aggregated_df.columns
    ]

    if verbose:
        print(f"Features: {feature_columns}")
        print(f"Targets: {target_columns}")

    # Fill missing values
    all_columns = feature_columns + target_columns
    aggregated_df = _fill_missing_values(
        aggregated_df, all_columns, config.fill_numerical_na
    )

    # Chronological split
    if verbose:
        print("Splitting data chronologically...")
    train_df, val_df, test_df = _chronological_split(
        aggregated_df,
        config.train_ratio,
        config.val_ratio,
        config.test_ratio,
    )

    if verbose:
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Extract feature and target arrays
    train_features = train_df[feature_columns].values.astype(np.float32)
    train_targets = train_df[target_columns].values.astype(np.float32)

    val_features = val_df[feature_columns].values.astype(np.float32)
    val_targets = val_df[target_columns].values.astype(np.float32)

    test_features = test_df[feature_columns].values.astype(np.float32)
    test_targets = test_df[target_columns].values.astype(np.float32)

    # Normalize features using training data statistics
    feature_means = torch.tensor(train_features.mean(axis=0), dtype=torch.float32)
    feature_stds = torch.tensor(train_features.std(axis=0), dtype=torch.float32)
    # Avoid division by zero
    feature_stds = torch.where(
        feature_stds == 0, torch.ones_like(feature_stds), feature_stds
    )

    def normalize(arr: np.ndarray) -> Tensor:
        tensor = torch.tensor(arr, dtype=torch.float32)
        return (tensor - feature_means) / feature_stds

    train_features_norm = normalize(train_features)
    val_features_norm = normalize(val_features)
    test_features_norm = normalize(test_features)

    # Convert targets to tensors (with optional log transform)
    if config.log_transform_targets:
        if verbose:
            print("Applying log transform to targets: log(1 + x)...")
        # Log transform: log(1 + x) to handle zeros
        train_targets_tensor = torch.log1p(
            torch.tensor(train_targets, dtype=torch.float32)
        )
        val_targets_tensor = torch.log1p(torch.tensor(val_targets, dtype=torch.float32))
        test_targets_tensor = torch.log1p(
            torch.tensor(test_targets, dtype=torch.float32)
        )
    else:
        train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
        val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)
        test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)

    # Create datasets
    train_dataset = AggregateDataset(train_features_norm, train_targets_tensor)
    val_dataset = AggregateDataset(val_features_norm, val_targets_tensor)
    test_dataset = AggregateDataset(test_features_norm, test_targets_tensor)

    if verbose:
        print("Data preparation complete!")

    return AggregateDataResult(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        feature_columns=feature_columns,
        target_columns=target_columns,
        feature_means=feature_means,
        feature_stds=feature_stds,
        log_transform_targets=config.log_transform_targets,
        loss_function="mse",  # Default, will be updated by training script
        config=config,
    )
