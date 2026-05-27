"""Neural network architecture for crash severity prediction.

This module implements an MLP architecture designed for tabular data with
proper regularization for handling class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SeverityMLPConfig:
    """Configuration for the severity MLP model.

    Attributes:
        input_dim: Number of input features
        num_classes: Number of output classes (default 5)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout probability between layers
        batch_norm: Whether to use batch normalization
        activation: Activation function ('relu', 'gelu', 'silu')
    """

    input_dim: int
    num_classes: int = 5
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 128])
    dropout_rate: float = 0.3
    batch_norm: bool = True
    activation: Literal["relu", "gelu", "silu"] = "relu"


class SeverityMLP(nn.Module):
    """MLP for crash severity classification.

    Architecture:
        Input → BatchNorm → [Linear → Activation → Dropout → BatchNorm] × N → Linear → Output

    The model outputs raw logits; softmax should be applied externally for
    probabilities.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: SeverityMLPConfig) -> None:
        super().__init__()
        self.config = config

        # Activation function
        if config.activation == "relu":
            self.activation_fn = nn.ReLU
        elif config.activation == "gelu":
            self.activation_fn = nn.GELU
        elif config.activation == "silu":
            self.activation_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {config.activation}")

        # Build layers
        layers: list[nn.Module] = []

        # Input batch normalization
        if config.batch_norm:
            layers.append(nn.BatchNorm1d(config.input_dim))

        # Hidden layers
        prev_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation_fn())
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Output layer
        self.classifier = nn.Linear(prev_dim, config.num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class SeverityMLPWithEmbeddings(nn.Module):
    """MLP with categorical embeddings for crash severity.

    This variant learns embeddings for categorical features rather than
    using one-hot encoding, which can improve generalization for high-cardinality
    categoricals.

    Args:
        num_numerical: Number of numerical features
        categorical_cardinalities: Dict mapping categorical feature names to
            their number of unique values
        embedding_dim: Dimension of categorical embeddings (or 'auto')
        mlp_config: Configuration for the MLP backbone
    """

    def __init__(
        self,
        num_numerical: int,
        categorical_cardinalities: dict[str, int],
        embedding_dim: int | Literal["auto"] = "auto",
        num_classes: int = 5,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.num_numerical = num_numerical
        self.categorical_names = list(categorical_cardinalities.keys())

        # Create embeddings for each categorical
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0

        for name, cardinality in categorical_cardinalities.items():
            # Auto-compute embedding dimension (heuristic: min(50, (cardinality+1)//2))
            if embedding_dim == "auto":
                emb_dim = min(50, (cardinality + 1) // 2)
            else:
                emb_dim = embedding_dim

            # Add 1 for unknown/missing category
            self.embeddings[name] = nn.Embedding(cardinality + 1, emb_dim)
            total_embedding_dim += emb_dim

        # MLP backbone
        input_dim = num_numerical + total_embedding_dim
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        config = SeverityMLPConfig(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )
        self.mlp = SeverityMLP(config)

    def forward(
        self,
        numerical: torch.Tensor,
        categoricals: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            numerical: Numerical features of shape (batch_size, num_numerical)
            categoricals: Dict of categorical tensors, each of shape (batch_size,)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Get embeddings
        embedded = []
        for name in self.categorical_names:
            emb = self.embeddings[name](categoricals[name])
            embedded.append(emb)

        # Concatenate all features
        all_features = torch.cat([numerical] + embedded, dim=1)

        return self.mlp(all_features)


class MCDropoutWrapper(nn.Module):
    """Wrapper for MC Dropout uncertainty estimation.

    Enables dropout at inference time to estimate prediction uncertainty
    via Monte Carlo sampling.

    Args:
        model: Base model with dropout layers
        num_samples: Number of forward passes for uncertainty estimation
    """

    def __init__(self, model: nn.Module, num_samples: int = 10) -> None:
        super().__init__()
        self.model = model
        self.num_samples = num_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (dropout disabled for eval)."""
        return self.model(x)

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation.

        Args:
            x: Input features

        Returns:
            Tuple of (mean_probs, std_probs) each of shape (batch_size, num_classes)
        """
        # Enable dropout
        self.model.train()

        # Multiple forward passes
        probs_list = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs)

        # Stack and compute statistics
        probs_stack = torch.stack(probs_list, dim=0)
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)

        # Return to eval mode
        self.model.eval()

        return mean_probs, std_probs
