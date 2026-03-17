"""Neural network model definitions for crash prediction.

This module defines the MLP architecture for classification tasks.
"""

import torch
import torch.nn as nn
from torch import Tensor


class CrashPredictionMLP(nn.Module):
    """Multi-layer perceptron for crash severity prediction.

    Architecture: input → [Linear → BatchNorm → ReLU → Dropout] × N → num_classes

    Features:
        - BatchNorm for stable training and regularization
        - Kaiming initialization for better gradient flow
        - Configurable depth via hidden_sizes tuple

    Attributes:
        num_features: Number of input features.
        num_classes: Number of output classes.
        dropout: Dropout probability.
        hidden_sizes: Tuple of hidden layer sizes.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_sizes: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.2,
    ) -> None:
        """Initialize the MLP.

        Args:
            num_features: Number of input features.
            num_classes: Number of output classes.
            hidden_sizes: Tuple of hidden layer sizes (variable length).
            dropout: Dropout probability for regularization.
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.hidden_sizes = hidden_sizes

        # Build network with BatchNorm
        layers: list[nn.Module] = []
        in_size = num_features

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_size = hidden_size

        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization for ReLU networks."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, num_features).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.network(x)
