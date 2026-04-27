"""Neural network architecture for multiclass classification.

This module defines the MulticlassMLP model for direct 5-class
crash severity prediction.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class MulticlassMLP(nn.Module):
    """Multiclass MLP for crash severity prediction.

    Architecture: input → [hidden → activation → (batch_norm) → dropout] × N → output

    Attributes:
        num_features: Number of input features.
        num_classes: Number of output classes.
        hidden_sizes: Sizes of hidden layers.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 5,
        hidden_sizes: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        activation: Literal["relu", "gelu", "leaky_relu"] = "relu",
    ) -> None:
        """Initialize the multiclass MLP.

        Args:
            num_features: Number of input features.
            num_classes: Number of output classes (default 5).
            hidden_sizes: Tuple of hidden layer sizes.
            dropout: Dropout probability between layers.
            use_batch_norm: Whether to use batch normalization.
            activation: Activation function type.
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_sizes = hidden_sizes

        # Build activation function
        activation_fn = self._get_activation(activation)

        # Build network layers
        layers: list[nn.Module] = []
        in_features = num_features

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            layers.append(activation_fn)
            layers.append(nn.Dropout(dropout))

            in_features = hidden_size

        # Output layer (no activation - raw logits for CrossEntropyLoss)
        layers.append(nn.Linear(in_features, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(
        self, activation: Literal["relu", "gelu", "leaky_relu"]
    ) -> nn.Module:
        """Get activation function module."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape (batch_size, num_features).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities using softmax.

        Args:
            x: Input tensor of shape (batch_size, num_features).

        Returns:
            Probability tensor of shape (batch_size, num_classes).
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels.

        Args:
            x: Input tensor of shape (batch_size, num_features).

        Returns:
            Class labels tensor of shape (batch_size,).
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Get a summary of the model architecture."""
        lines = [
            f"MulticlassMLP(",
            f"  num_features={self.num_features},",
            f"  num_classes={self.num_classes},",
            f"  hidden_sizes={self.hidden_sizes},",
            f"  num_parameters={self.get_num_parameters():,}",
            ")",
        ]
        return "\n".join(lines)
