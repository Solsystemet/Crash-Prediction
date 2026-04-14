"""Configuration for crash count regression models.

Provides dataclasses for configuring neural network architecture,
training parameters, and ensemble settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


TimeGranularity = Literal["hourly", "daily", "weekly"]


@dataclass
class RegressionConfig:
    """Configuration for crash count regression.

    Attributes:
        time_granularity: Time bucket size ('hourly', 'daily', 'weekly').
        n_zones: Number of geographic zones for clustering.
        hidden_sizes: Tuple of hidden layer sizes for main MLP.
        zone_hidden_sizes: Tuple of hidden layer sizes for zone adjustment MLP.
        dropout: Dropout probability.
        learning_rate: Adam optimizer learning rate.
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        patience: Early stopping patience (epochs without improvement).
        min_delta: Minimum improvement for early stopping.
        loss_type: Loss function ('mse', 'huber', 'poisson').
        huber_delta: Delta parameter for Huber loss.
        train_ratio: Fraction of data for training (temporal split).
        val_ratio: Fraction of data for validation.
        random_state: Random seed for reproducibility.
        device: Device to use ('auto', 'cpu', 'cuda').
    """

    # Time-series parameters
    time_granularity: TimeGranularity = "hourly"
    n_zones: int = 10

    # Model architecture
    hidden_sizes: tuple[int, ...] = (128, 64, 32)
    zone_hidden_sizes: tuple[int, ...] = (32, 16)
    dropout: float = 0.2

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4

    # Loss configuration
    loss_type: Literal["mse", "huber", "poisson"] = "huber"
    huber_delta: float = 1.0

    # Data split (temporal)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    # test_ratio is implicitly 1 - train_ratio - val_ratio

    # Reproducibility
    random_state: int = 42

    # Device
    device: str = "auto"

    def get_device(self) -> str:
        """Get the device to use for training.

        Returns:
            'cuda' if available and device is 'auto' or 'cuda', else 'cpu'.
        """
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @property
    def test_ratio(self) -> float:
        """Fraction of data for testing."""
        return 1.0 - self.train_ratio - self.val_ratio

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameters are invalid.
        """
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")
        if self.n_zones < 1:
            raise ValueError("n_zones must be >= 1")
        if any(h < 1 for h in self.hidden_sizes):
            raise ValueError("All hidden sizes must be >= 1")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def __post_init__(self) -> None:
        """Validate after initialization."""
        self.validate()


@dataclass
class ZoneConfig:
    """Configuration for a single zone model.

    Tracks zone-specific metadata for the ensemble.

    Attributes:
        zone_id: Unique identifier for the zone.
        centroid_lat: Latitude of zone centroid.
        centroid_lon: Longitude of zone centroid.
        n_samples: Number of training samples in this zone.
        avg_crash_count: Average crash count per period in this zone.
    """

    zone_id: int
    centroid_lat: float
    centroid_lon: float
    n_samples: int = 0
    avg_crash_count: float = 0.0


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble of global + zone models.

    Attributes:
        base_config: Base regression configuration.
        zone_configs: List of zone-specific configurations.
        global_weight: Weight for global model in ensemble (0-1).
        zone_weight: Weight for zone adjustment (1 - global_weight).
        train_zone_models: Whether to train zone adjustment models.
    """

    base_config: RegressionConfig = field(default_factory=RegressionConfig)
    zone_configs: list[ZoneConfig] = field(default_factory=list)
    global_weight: float = 0.7
    train_zone_models: bool = True

    @property
    def zone_weight(self) -> float:
        """Weight for zone-specific adjustments."""
        return 1.0 - self.global_weight
