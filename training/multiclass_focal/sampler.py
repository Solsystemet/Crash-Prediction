"""Class-balanced sampling strategies for imbalanced datasets.

This module provides samplers that ensure each batch contains examples from
all classes, without duplicating data like SMOTE/ADASYN.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from numpy.typing import NDArray


class ClassBalancedSampler(Sampler[int]):
    """Sampler that ensures each batch contains samples from all classes.

    Unlike oversampling methods (SMOTE, ADASYN), this sampler doesn't create
    synthetic data. Instead, it strategically samples from each class to
    construct balanced batches.

    For a batch of size B with C classes, each batch will contain approximately
    B/C samples from each class. Minority classes may repeat within an epoch,
    but no synthetic data is created.

    Args:
        labels: Array of class labels for each sample
        samples_per_class: Number of samples to draw from each class per batch.
            Total batch size = samples_per_class * num_classes
        num_batches: Number of batches per epoch. If None, computed from
            (total_samples / batch_size)
        replacement: Whether to sample with replacement within a class
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        labels: NDArray[np.int64],
        samples_per_class: int = 8,
        num_batches: int | None = None,
        replacement: bool = True,
        seed: int = 42,
    ) -> None:
        self.labels = np.asarray(labels)
        self.samples_per_class = samples_per_class
        self.replacement = replacement
        self.rng = np.random.default_rng(seed)

        # Find unique classes and indices
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)
        self.class_indices: dict[int, NDArray[np.int64]] = {
            cls: np.where(self.labels == cls)[0] for cls in self.classes
        }

        # Batch size
        self.batch_size = samples_per_class * self.num_classes

        # Number of batches
        if num_batches is None:
            # Default: roughly one epoch worth
            self.num_batches = max(1, len(self.labels) // self.batch_size)
        else:
            self.num_batches = num_batches

    def __iter__(self) -> Iterator[int]:
        """Generate indices for all batches."""
        for _ in range(self.num_batches):
            batch_indices = []
            for cls in self.classes:
                cls_indices = self.class_indices[cls]
                if self.replacement or len(cls_indices) >= self.samples_per_class:
                    selected = self.rng.choice(
                        cls_indices,
                        size=self.samples_per_class,
                        replace=self.replacement,
                    )
                else:
                    # Not enough samples, take all and repeat
                    repeats = (self.samples_per_class // len(cls_indices)) + 1
                    selected = np.tile(cls_indices, repeats)[: self.samples_per_class]
                    self.rng.shuffle(selected)
                batch_indices.extend(selected)

            # Shuffle within batch to avoid class ordering bias
            self.rng.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self) -> int:
        """Total number of samples per epoch."""
        return self.num_batches * self.batch_size


class ProgressiveBalancedSampler(Sampler[int]):
    """Sampler that progressively increases minority representation.

    Starts with natural class distribution and gradually shifts toward
    balanced sampling over training. This allows the model to first learn
    from the natural distribution before focusing on minorities.

    Args:
        labels: Array of class labels
        samples_per_class_final: Target samples per class when fully balanced
        num_batches: Number of batches per epoch
        total_epochs: Total training epochs (for scheduling)
        warmup_epochs: Epochs before starting progressive balancing
        seed: Random seed
    """

    def __init__(
        self,
        labels: NDArray[np.int64],
        samples_per_class_final: int = 8,
        num_batches: int = 100,
        total_epochs: int = 50,
        warmup_epochs: int = 5,
        seed: int = 42,
    ) -> None:
        self.labels = np.asarray(labels)
        self.samples_per_class_final = samples_per_class_final
        self.num_batches = num_batches
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.rng = np.random.default_rng(seed)
        self.current_epoch = 0

        # Class info
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)
        self.class_indices = {
            cls: np.where(self.labels == cls)[0] for cls in self.classes
        }

        # Natural class distribution
        counts = np.array([len(self.class_indices[c]) for c in self.classes])
        self.natural_probs = counts / counts.sum()

        # Balanced distribution
        self.balanced_probs = np.ones(self.num_classes) / self.num_classes

        self.batch_size = samples_per_class_final * self.num_classes

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for progressive scheduling."""
        self.current_epoch = epoch

    def _get_current_probs(self) -> NDArray[np.float64]:
        """Get current class sampling probabilities."""
        if self.current_epoch < self.warmup_epochs:
            return self.natural_probs

        # Linear interpolation from natural to balanced
        progress = (self.current_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        progress = min(1.0, progress)

        return (1 - progress) * self.natural_probs + progress * self.balanced_probs

    def __iter__(self) -> Iterator[int]:
        """Generate indices for all batches."""
        probs = self._get_current_probs()

        for _ in range(self.num_batches):
            batch_indices = []

            # Sample classes according to current distribution
            for _ in range(self.batch_size):
                cls = self.rng.choice(self.classes, p=probs)
                idx = self.rng.choice(self.class_indices[cls])
                batch_indices.append(idx)

            yield from batch_indices

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


class SquareRootSampler(Sampler[int]):
    """Sampler using square-root class frequency weighting.

    Instead of fully balancing classes (which can hurt majority class),
    this sampler uses sqrt(1/freq) weighting, which provides a middle
    ground between natural and balanced distributions.

    This is particularly effective for extreme imbalance where minorities
    are < 1% of the data.

    Args:
        labels: Array of class labels
        batch_size: Samples per batch
        num_batches: Number of batches per epoch
        seed: Random seed
    """

    def __init__(
        self,
        labels: NDArray[np.int64],
        batch_size: int = 64,
        num_batches: int | None = None,
        seed: int = 42,
    ) -> None:
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # Class info
        self.classes = np.unique(self.labels)
        self.class_indices = {
            cls: np.where(self.labels == cls)[0] for cls in self.classes
        }

        # Compute sqrt-balanced probabilities
        counts = np.array([len(self.class_indices[c]) for c in self.classes])
        freq = counts / counts.sum()
        weights = np.sqrt(1.0 / (freq + 1e-8))
        self.probs = weights / weights.sum()

        # Per-sample weights for weighted random sampling
        self.sample_weights = np.zeros(len(labels))
        for cls, prob in zip(self.classes, self.probs):
            cls_size = len(self.class_indices[cls])
            self.sample_weights[self.class_indices[cls]] = prob / cls_size

        if num_batches is None:
            self.num_batches = max(1, len(labels) // batch_size)
        else:
            self.num_batches = num_batches

    def __iter__(self) -> Iterator[int]:
        """Generate indices for all batches."""
        all_indices = np.arange(len(self.labels))

        for _ in range(self.num_batches):
            batch_indices = self.rng.choice(
                all_indices,
                size=self.batch_size,
                replace=True,
                p=self.sample_weights,
            )
            yield from batch_indices

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


def get_sampler(
    labels: NDArray[np.int64],
    strategy: str = "class_balanced",
    batch_size: int = 64,
    **kwargs,
) -> Sampler[int]:
    """Factory function to create a sampler.

    Args:
        labels: Array of class labels
        strategy: Sampling strategy
            - 'class_balanced': Equal samples from each class
            - 'sqrt': Square-root frequency weighting
            - 'progressive': Progressive balancing over training
        batch_size: Samples per batch
        **kwargs: Additional arguments for specific samplers

    Returns:
        Configured sampler.
    """
    n_classes = len(np.unique(labels))

    if strategy == "class_balanced":
        samples_per_class = max(1, batch_size // n_classes)
        return ClassBalancedSampler(
            labels,
            samples_per_class=samples_per_class,
            **kwargs,
        )
    elif strategy == "sqrt":
        return SquareRootSampler(labels, batch_size=batch_size, **kwargs)
    elif strategy == "progressive":
        samples_per_class = max(1, batch_size // n_classes)
        return ProgressiveBalancedSampler(
            labels,
            samples_per_class_final=samples_per_class,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
