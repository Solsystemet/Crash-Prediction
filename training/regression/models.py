"""Neural network models for crash count regression.

Provides MLP architectures for:
- Global crash count prediction (CrashCountMLP)
- Zone-specific residual adjustments (ZoneAdjustmentMLP)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrashCountMLP(nn.Module):
    """Multi-layer perceptron for crash count regression.

    Architecture: input → [hidden layers with ReLU + Dropout] → output (1)

    The output has no activation since crash counts are non-negative
    but unbounded. We apply ReLU to predictions at inference time.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
    ) -> None:
        """Initialize crash count MLP.

        Args:
            num_features: Number of input features.
            hidden_sizes: Tuple of hidden layer sizes.
            dropout: Dropout probability between layers.
        """
        super().__init__()

        layers: list[nn.Module] = []

        # Input layer
        prev_size = num_features
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        # Output layer (single neuron, no activation)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, num_features).

        Returns:
            Output tensor of shape (batch_size,) with predicted counts.
        """
        return self.network(x).squeeze(-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Prediction with non-negativity constraint.

        Args:
            x: Input tensor.

        Returns:
            Non-negative predicted crash counts.
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
            # Ensure non-negative predictions
            return torch.clamp(preds, min=0)


class ZoneAdjustmentMLP(nn.Module):
    """Smaller MLP for learning zone-specific residual adjustments.

    Learns the residual: actual - global_prediction for a specific zone.
    This captures local patterns not explained by the global model.
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: tuple[int, ...] = (32, 16),
        dropout: float = 0.1,
    ) -> None:
        """Initialize zone adjustment MLP.

        Args:
            num_features: Number of input features.
            hidden_sizes: Tuple of hidden layer sizes (smaller than main model).
            dropout: Dropout probability.
        """
        super().__init__()

        layers: list[nn.Module] = []

        prev_size = num_features
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        # Output: residual adjustment (can be positive or negative)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with small weights to start near zero adjustment."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning residual adjustment.

        Args:
            x: Input tensor.

        Returns:
            Residual adjustment (can be positive or negative).
        """
        return self.network(x).squeeze(-1)


class CrashCountLSTM(nn.Module):
    """LSTM model for sequential crash count prediction.

    Alternative to MLP that explicitly models temporal dependencies.
    Expects input sequences of shape (batch, seq_len, features).
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        """Initialize LSTM model.

        Args:
            num_features: Number of input features per timestep.
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout between LSTM layers.
            bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output projection
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, features).
            return_sequence: If True, return predictions for all timesteps.

        Returns:
            Predictions of shape (batch,) or (batch, seq_len).
        """
        lstm_out, _ = self.lstm(x)

        if return_sequence:
            # Predict for each timestep
            return self.fc(lstm_out).squeeze(-1)
        else:
            # Predict for last timestep only
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden).squeeze(-1)


def create_model(
    model_type: str,
    num_features: int,
    **kwargs,
) -> nn.Module:
    """Factory function to create regression models.

    Args:
        model_type: Type of model ('mlp', 'lstm', 'zone_adj').
        num_features: Number of input features.
        **kwargs: Additional arguments passed to model constructor.

    Returns:
        Initialized model.

    Raises:
        ValueError: If model_type is unknown.
    """
    model_classes = {
        "mlp": CrashCountMLP,
        "lstm": CrashCountLSTM,
        "zone_adj": ZoneAdjustmentMLP,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_classes.keys())}")

    return model_classes[model_type](num_features, **kwargs)
