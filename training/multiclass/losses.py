"""Loss functions for multiclass classification with class imbalance.

This module provides focal loss and weighted cross-entropy implementations
to handle the severe class imbalance in crash severity prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Focal Loss for multiclass classification with class imbalance.

    Focal loss down-weights well-classified examples and focuses on
    hard, misclassified examples. This helps with class imbalance by
    preventing the majority class from dominating the loss.

    Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t is the probability of the true class.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Attributes:
        alpha: Per-class weights. Shape (num_classes,) or scalar.
        gamma: Focusing parameter. Higher values focus more on hard examples.
        reduction: How to reduce the loss ("mean", "sum", "none").
        label_smoothing: Label smoothing factor (0 = no smoothing).
    """

    def __init__(
        self,
        alpha: Tensor | list[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        num_classes: int | None = None,
    ) -> None:
        """Initialize focal loss.

        Args:
            alpha: Per-class weights. If None, all classes have weight 1.
                Can be a tensor, list, or scalar.
            gamma: Focusing parameter (default 2.0). gamma=0 is equivalent
                to weighted cross-entropy.
            reduction: Loss reduction method ("mean", "sum", "none").
            label_smoothing: Label smoothing factor in [0, 1).
            num_classes: Number of classes (required if alpha is None).
        """
        super().__init__()

        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size,).

        Returns:
            Focal loss value (scalar if reduction != "none").
        """
        num_classes = logits.shape[-1]

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probability of true class
        # Shape: (batch_size,)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets_one_hot = (
                targets_one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )

        # p_t: probability of the true class
        p_t = (probs * targets_one_hot).sum(dim=-1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy: -log(p_t)
        ce_loss = -torch.log(p_t.clamp(min=1e-8))

        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Move alpha to same device as logits
            alpha = self.alpha.to(logits.device)
            # Get alpha for each sample's true class
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def weighted_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    class_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute weighted cross-entropy loss.

    Args:
        logits: Raw model outputs of shape (batch_size, num_classes).
        targets: Ground truth labels of shape (batch_size,).
        class_weights: Per-class weights of shape (num_classes,).
        label_smoothing: Label smoothing factor.

    Returns:
        Weighted cross-entropy loss (scalar).
    """
    return F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


def compute_class_weights(
    class_counts: dict[int, int] | list[int] | Tensor,
    strategy: str = "inverse_frequency",
    beta: float = 0.9999,
    normalize: bool = True,
) -> Tensor:
    """Compute class weights for handling imbalance.

    Args:
        class_counts: Number of samples per class. Can be dict, list, or tensor.
        strategy: Weighting strategy:
            - "inverse_frequency": weight = 1 / count
            - "inverse_sqrt": weight = 1 / sqrt(count)
            - "effective_samples": weight based on effective number of samples
                (see "Class-Balanced Loss Based on Effective Number of Samples")
        beta: Beta parameter for effective_samples strategy.
        normalize: Whether to normalize weights to sum to num_classes.

    Returns:
        Class weights tensor of shape (num_classes,).
    """
    # Convert to tensor
    if isinstance(class_counts, dict):
        num_classes = max(class_counts.keys()) + 1
        counts = torch.zeros(num_classes, dtype=torch.float32)
        for cls, count in class_counts.items():
            counts[cls] = count
    elif isinstance(class_counts, list):
        counts = torch.tensor(class_counts, dtype=torch.float32)
    else:
        counts = class_counts.float()

    # Avoid division by zero
    counts = counts.clamp(min=1)

    if strategy == "inverse_frequency":
        weights = 1.0 / counts
    elif strategy == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(counts)
    elif strategy == "effective_samples":
        # Effective number of samples: (1 - beta^n) / (1 - beta)
        effective_num = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
        weights = 1.0 / effective_num
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Normalize so weights sum to num_classes
    if normalize:
        weights = weights * len(weights) / weights.sum()

    return weights
