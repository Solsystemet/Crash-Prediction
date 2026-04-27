"""Configuration for multiclass neural network classifier.

This module defines the configuration dataclass for the multiclass
neural network model, including architecture, training, and
class weighting parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class MulticlassNeuralConfig:
    """Configuration for multiclass neural network classifier.

    Attributes:
        num_classes: Number of output classes (default 5 for severity).
        hidden_sizes: Tuple of hidden layer sizes.
        dropout: Dropout probability between layers.
        use_batch_norm: Whether to use batch normalization.
        activation: Activation function ("relu", "gelu", "leaky_relu").

        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization coefficient.
        gradient_clip_norm: Max gradient norm for clipping (None to disable).

        loss_type: Loss function type ("cross_entropy", "focal").
        focal_alpha: Focal loss alpha (class weights). None for uniform.
        focal_gamma: Focal loss gamma (focusing parameter).
        class_weight_strategy: How to compute class weights.
            "inverse_frequency": Weight inversely proportional to class count.
            "effective_samples": Use effective number of samples formula.
            "custom": Use custom_class_weights.
            None: No class weighting.
        custom_class_weights: Custom weights when class_weight_strategy="custom".

        early_stopping_patience: Epochs without improvement before stopping.
        early_stopping_min_delta: Minimum improvement to reset patience.
        lr_scheduler: Learning rate scheduler type.
            "plateau": ReduceLROnPlateau.
            "cosine": CosineAnnealingLR.
            None: No scheduler.
        lr_scheduler_patience: Patience for plateau scheduler.
        lr_scheduler_factor: Factor for plateau scheduler.

        device: Device for training ("cuda", "mps", "cpu", or "auto").
        random_seed: Random seed for reproducibility.
        verbose: Whether to print training progress.
    """

    # Architecture
    num_classes: int = 5
    hidden_sizes: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    use_batch_norm: bool = True
    activation: Literal["relu", "gelu", "leaky_relu"] = "relu"

    # Training
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float | None = 1.0

    # Loss and class weighting
    loss_type: Literal["cross_entropy", "focal"] = "focal"
    focal_alpha: list[float] | None = None
    focal_gamma: float = 2.0
    class_weight_strategy: Literal[
        "inverse_frequency", "effective_samples", "custom"
    ] | None = "inverse_frequency"
    custom_class_weights: list[float] | None = None
    effective_samples_beta: float = 0.9999

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # Learning rate scheduling
    lr_scheduler: Literal["plateau", "cosine"] | None = "plateau"
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5

    # Device and reproducibility
    device: Literal["cuda", "mps", "cpu", "auto"] = "auto"
    random_seed: int = 42
    verbose: bool = True

    # Label smoothing
    label_smoothing: float = 0.0

    def get_device(self) -> str:
        """Get the actual device string, resolving 'auto'."""
        if self.device != "auto":
            return self.device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")

        if not self.hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")

        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be >= 0, got {self.focal_gamma}")

        if self.class_weight_strategy == "custom" and self.custom_class_weights is None:
            raise ValueError(
                "custom_class_weights must be provided when "
                "class_weight_strategy='custom'"
            )

        if self.custom_class_weights is not None:
            if len(self.custom_class_weights) != self.num_classes:
                raise ValueError(
                    f"custom_class_weights length ({len(self.custom_class_weights)}) "
                    f"must match num_classes ({self.num_classes})"
                )


# Preset configurations for common use cases

MULTICLASS_DEFAULT_CONFIG = MulticlassNeuralConfig()
"""Default configuration with focal loss and inverse frequency weighting."""


MULTICLASS_LIGHTWEIGHT_CONFIG = MulticlassNeuralConfig(
    hidden_sizes=(128, 64),
    epochs=50,
    batch_size=512,
    early_stopping_patience=10,
)
"""Lightweight configuration for faster training and iteration."""


MULTICLASS_DEEP_CONFIG = MulticlassNeuralConfig(
    hidden_sizes=(512, 256, 128, 64),
    dropout=0.4,
    epochs=150,
    learning_rate=5e-4,
    early_stopping_patience=20,
)
"""Deeper architecture for potentially better representation learning."""
