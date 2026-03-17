"""PyTorch Dataset implementation for crash data.

This module provides a Dataset class that wraps tensor data for use
with PyTorch DataLoaders.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset


class CrashTensorDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch Dataset for crash prediction data.

    Stores features and labels as tensors and provides indexed access
    for use with DataLoader.

    Attributes:
        features: Float tensor of shape (num_samples, num_features).
        labels: Tensor of shape (num_samples,). LongTensor for classification,
            FloatTensor for regression.
        task_type: "classification" or "regression".
    """

    def __init__(
        self,
        features: Tensor,
        labels: Tensor,
        task_type: str = "classification",
    ) -> None:
        """Initialize the dataset.

        Args:
            features: Float tensor of input features.
            labels: Tensor of target labels.
            task_type: Type of ML task.
        """
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Features and labels must have same number of samples. "
                f"Got features: {features.shape[0]}, labels: {labels.shape[0]}"
            )

        self.features = features
        self.labels = labels
        self.task_type = task_type

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return the feature and label at the given index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (features, label) tensors for the sample.
        """
        return self.features[idx], self.labels[idx]

    @property
    def num_features(self) -> int:
        """Return the number of input features."""
        return self.features.shape[1]

    @property
    def num_samples(self) -> int:
        """Return the number of samples."""
        return self.features.shape[0]

    def get_label_distribution(self) -> dict[int, int]:
        """Return distribution of labels (for classification).

        Returns:
            Dict mapping label value to count.
        """
        if self.task_type != "classification":
            raise RuntimeError("Label distribution only available for classification tasks")

        unique, counts = torch.unique(self.labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts, strict=True)}

    def to(self, device: torch.device | str) -> CrashTensorDataset:
        """Move tensors to the specified device.

        Args:
            device: Target device (e.g., "cuda", "cpu").

        Returns:
            New dataset with tensors on the specified device.
        """
        return CrashTensorDataset(
            features=self.features.to(device),
            labels=self.labels.to(device),
            task_type=self.task_type,
        )
