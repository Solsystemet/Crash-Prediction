"""Neural network model definitions for crash prediction.

This module defines the MLP architectures for classification and regression tasks.
"""

import torch.nn as nn
from torch import Tensor


class CrashPredictionMLP(nn.Module):
    """Multi-layer perceptron for crash severity prediction.

    Architecture: input → 128 → ReLU → Dropout → 64 → ReLU → Dropout → num_classes

    Attributes:
        num_features: Number of input features.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_sizes: tuple[int, int] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        """Initialize the MLP.

        Args:
            num_features: Number of input features.
            num_classes: Number of output classes.
            hidden_sizes: Tuple of hidden layer sizes.
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout

        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, num_features).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.network(x)


class AggregateInjuryMLP(nn.Module):
    """Multi-layer perceptron for predicting aggregated hourly injury counts.

    This model predicts multiple injury count targets simultaneously using
    a shared representation. Designed for regression tasks.

    Architecture: Configurable depth with BatchNorm and residual-like connections.

    Attributes:
        num_features: Number of input features.
        num_targets: Number of output targets (injury types to predict).
        hidden_sizes: Tuple of hidden layer sizes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_features: int,
        num_targets: int = 5,
        hidden_sizes: tuple[int, ...] = (
            1024,
            1024,
            1024,
            1024,
            512,
            512,
            512,
            512,
            512,
            512,
            256,
            256,
            256,
            256,
            256,
            256,
            128,
            128,
            128,
            128,
        ),
        dropout: float = 0.3,
    ) -> None:
        """Initialize the MLP.

        Args:
            num_features: Number of input features.
            num_targets: Number of output targets (default 5 for injury types).
            hidden_sizes: Tuple of hidden layer sizes (variable length).
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        self.num_features = num_features
        self.num_targets = num_targets
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        # Build layers dynamically
        layers: list[nn.Module] = []
        in_features = num_features

        for i, out_features in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = out_features

        # Output layer
        layers.append(nn.Linear(in_features, num_targets))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, num_features).

        Returns:
            Predictions tensor of shape (batch_size, num_targets).
        """
        return self.network(x)
