"""PyTorch datasets for time-series crash count data.

Provides Dataset classes for:
- Standard tabular data (TimeSeriesDataset)
- Sequence data for LSTM models (SequenceDataset)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time-series crash count data.

    Stores features and targets as tensors, with optional zone IDs.
    Suitable for MLP models that treat each timestep independently.
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        zone_ids: np.ndarray | torch.Tensor | None = None,
        timestamps: np.ndarray | list | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            features: Feature matrix of shape (n_samples, n_features).
            targets: Target values of shape (n_samples,).
            zone_ids: Optional zone IDs of shape (n_samples,).
            timestamps: Optional timestamps for reference.
        """
        # Convert to tensors
        if isinstance(features, np.ndarray):
            self.features = torch.from_numpy(features).float()
        else:
            self.features = features.float()

        if isinstance(targets, np.ndarray):
            self.targets = torch.from_numpy(targets).float()
        else:
            self.targets = targets.float()

        if zone_ids is not None:
            if isinstance(zone_ids, np.ndarray):
                self.zone_ids = torch.from_numpy(zone_ids).long()
            else:
                self.zone_ids = zone_ids.long()
        else:
            self.zone_ids = None

        self.timestamps = timestamps

        # Validate shapes
        if len(self.features) != len(self.targets):
            raise ValueError(
                f"Features and targets must have same length: "
                f"{len(self.features)} vs {len(self.targets)}"
            )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Get sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features, target, zone_id) where zone_id may be None.
        """
        features = self.features[idx]
        target = self.targets[idx]
        zone_id = self.zone_ids[idx] if self.zone_ids is not None else None

        return features, target, zone_id

    def to(self, device: torch.device) -> "TimeSeriesDataset":
        """Move tensors to device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        if self.zone_ids is not None:
            self.zone_ids = self.zone_ids.to(device)
        return self

    def get_zone_subset(self, zone_id: int) -> "TimeSeriesDataset":
        """Get subset of data for a specific zone.

        Args:
            zone_id: Zone to filter by.

        Returns:
            New dataset with only samples from the specified zone.
        """
        if self.zone_ids is None:
            raise ValueError("Dataset has no zone IDs")

        mask = self.zone_ids == zone_id
        return TimeSeriesDataset(
            features=self.features[mask],
            targets=self.targets[mask],
            zone_ids=self.zone_ids[mask],
            timestamps=self.timestamps[mask.numpy()] if self.timestamps is not None else None,
        )

    @property
    def num_features(self) -> int:
        """Number of features per sample."""
        return self.features.shape[1]

    @property
    def num_zones(self) -> int:
        """Number of unique zones."""
        if self.zone_ids is None:
            return 0
        return len(torch.unique(self.zone_ids))

    def target_statistics(self) -> dict[str, float]:
        """Compute statistics of target variable.

        Returns:
            Dictionary with mean, std, min, max of targets.
        """
        targets_np = self.targets.numpy()
        return {
            "mean": float(np.mean(targets_np)),
            "std": float(np.std(targets_np)),
            "min": float(np.min(targets_np)),
            "max": float(np.max(targets_np)),
            "median": float(np.median(targets_np)),
        }


class SequenceDataset(Dataset):
    """Dataset for sequential data (LSTM models).

    Creates overlapping sequences from time series data.
    Each sample is a sequence of `seq_len` timesteps.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 24,
        stride: int = 1,
    ) -> None:
        """Initialize sequence dataset.

        Args:
            features: Feature matrix of shape (n_timesteps, n_features).
            targets: Target values of shape (n_timesteps,).
            seq_len: Length of each sequence.
            stride: Step between sequences.
        """
        self.seq_len = seq_len
        self.stride = stride

        # Create sequences
        n_timesteps = len(features)
        self.n_sequences = (n_timesteps - seq_len) // stride

        if self.n_sequences <= 0:
            raise ValueError(
                f"Not enough timesteps ({n_timesteps}) for sequence length {seq_len}"
            )

        # Pre-compute sequences
        self.sequences = []
        self.sequence_targets = []

        for i in range(self.n_sequences):
            start_idx = i * stride
            end_idx = start_idx + seq_len
            self.sequences.append(features[start_idx:end_idx])
            self.sequence_targets.append(targets[end_idx])  # Predict next value

        self.sequences = torch.from_numpy(np.array(self.sequences)).float()
        self.sequence_targets = torch.from_numpy(np.array(self.sequence_targets)).float()

    def __len__(self) -> int:
        """Return number of sequences."""
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sequence by index.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of (sequence, target) where sequence has shape (seq_len, n_features).
        """
        return self.sequences[idx], self.sequence_targets[idx]


def create_temporal_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally (not randomly) into train/val/test.

    Critical for time series to avoid data leakage.

    Args:
        df: DataFrame sorted by time.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        timestamp_col: Column with timestamps.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Ensure sorted by time
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def create_per_zone_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    timestamp_col: str = "timestamp",
    zone_col: str = "zone_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally within each zone.

    Ensures each zone has proper temporal splits.

    Args:
        df: DataFrame with zone_id and timestamp columns.
        train_ratio: Fraction for training per zone.
        val_ratio: Fraction for validation per zone.
        timestamp_col: Column with timestamps.
        zone_col: Column with zone IDs.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_parts = []
    val_parts = []
    test_parts = []

    for zone_id in df[zone_col].unique():
        zone_df = df[df[zone_col] == zone_id].copy()
        zone_train, zone_val, zone_test = create_temporal_splits(
            zone_df, train_ratio, val_ratio, timestamp_col
        )
        train_parts.append(zone_train)
        val_parts.append(zone_val)
        test_parts.append(zone_test)

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, val_df, test_df


def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "crash_count",
    zone_col: str | None = "zone_id",
) -> tuple[TimeSeriesDataset, TimeSeriesDataset, TimeSeriesDataset]:
    """Create PyTorch datasets from DataFrames.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        feature_cols: List of feature column names.
        target_col: Target column name.
        zone_col: Zone ID column name (optional).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """

    def df_to_dataset(df: pd.DataFrame) -> TimeSeriesDataset:
        features = df[feature_cols].values
        targets = df[target_col].values
        zone_ids = df[zone_col].values if zone_col and zone_col in df.columns else None
        timestamps = df["timestamp"].values if "timestamp" in df.columns else None

        return TimeSeriesDataset(
            features=features,
            targets=targets,
            zone_ids=zone_ids,
            timestamps=timestamps,
        )

    return (
        df_to_dataset(train_df),
        df_to_dataset(val_df),
        df_to_dataset(test_df),
    )
