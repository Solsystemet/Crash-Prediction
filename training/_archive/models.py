"""Neural network model definitions for crash prediction.

This module defines the MLP architecture for classification tasks.
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
