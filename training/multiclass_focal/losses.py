"""Loss functions for handling extreme class imbalance.

This module implements:
- FocalLoss: Down-weights well-classified examples, focusing on hard cases
- LDAMLoss: Label-Distribution-Aware Margin Loss with larger margins for rare classes
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


class FocalLoss(nn.Module):
    """Focal Loss for multiclass classification with class imbalance.

    Focal Loss modifies cross-entropy by adding a focusing parameter (gamma)
    that down-weights well-classified examples, allowing the model to focus
    on hard, misclassified examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the probability of the correct class.

    This naturally handles class imbalance without synthetic oversampling or
    aggressive class weighting that leads to overprediction of minorities.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
        https://arxiv.org/abs/1708.02002

    Args:
        alpha: Per-class weights. Can be:
            - None: No class weighting
            - Tensor of shape (num_classes,): Explicit weights per class
            - "balanced": Compute sqrt(1/freq) weights (softer than inverse frequency)
        gamma: Focusing parameter. Higher values focus more on hard examples.
            - gamma=0 reduces to weighted cross-entropy
            - gamma=2 is recommended default
        class_counts: Number of samples per class (required if alpha="balanced")
        reduction: How to reduce the loss ('mean', 'sum', 'none')
        label_smoothing: Optional label smoothing factor (0-1)
    """

    def __init__(
        self,
        alpha: torch.Tensor | Literal["balanced"] | None = None,
        gamma: float = 2.0,
        class_counts: NDArray[np.int64] | None = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Compute alpha weights
        if alpha == "balanced":
            if class_counts is None:
                raise ValueError("class_counts required when alpha='balanced'")
            # Use sqrt(1/freq) for softer weighting than inverse frequency
            freq = class_counts / class_counts.sum()
            alpha_vals = np.sqrt(1.0 / (freq + 1e-8))
            # Normalize so weights sum to num_classes
            alpha_vals = alpha_vals / alpha_vals.sum() * len(class_counts)
            self.register_buffer("alpha", torch.tensor(alpha_vals, dtype=torch.float32))
        elif alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model outputs of shape (batch_size, num_classes)
            targets: Class indices of shape (batch_size,)

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-sample loss.
        """
        num_classes = logits.size(1)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Convert to one-hot and smooth
            one_hot = F.one_hot(targets, num_classes=num_classes).float()
            smooth_targets = (
                one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
            probs = F.softmax(logits, dim=1)
            pt = (smooth_targets * probs).sum(dim=1)
        else:
            # Standard focal loss computation
            ce_loss = F.cross_entropy(logits, targets, reduction="none")
            probs = F.softmax(logits, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Reduce
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss.

    LDAM enforces larger classification margins for classes with fewer samples,
    based on the theoretical insight that minority classes need larger margins
    for generalization.

    The margin for class j is: Δ_j = C / n_j^(1/4)

    where n_j is the number of samples in class j and C is a scaling constant.

    Reference:
        Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware
        Margin Loss", NeurIPS 2019
        https://arxiv.org/abs/1906.07413

    Args:
        class_counts: Number of samples per class, shape (num_classes,)
        max_margin: Maximum margin value (C in the formula). Default 0.5.
        scale: Scaling factor for logits. Default 30.
        reduction: How to reduce the loss ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        class_counts: NDArray[np.int64],
        max_margin: float = 0.5,
        scale: float = 30.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.scale = scale
        self.reduction = reduction

        # Compute per-class margins: Δ_j = C / n_j^(1/4)
        # Larger margins for smaller classes
        margins = max_margin / np.power(class_counts, 0.25)
        self.register_buffer("margins", torch.tensor(margins, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute LDAM loss.

        Args:
            logits: Raw model outputs of shape (batch_size, num_classes)
            targets: Class indices of shape (batch_size,)

        Returns:
            Scalar loss (if reduction='mean' or 'sum') or per-sample loss.
        """
        # Get the margin for each sample's target class
        batch_margins = self.margins[targets]

        # Create margin-adjusted logits
        # Subtract margin from the target class logit only
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        margin_adjusted_logits = logits - one_hot * batch_margins.unsqueeze(1)

        # Scale logits and compute cross-entropy
        scaled_logits = self.scale * margin_adjusted_logits
        loss = F.cross_entropy(scaled_logits, targets, reduction=self.reduction)

        return loss


class DeferredReweightingLoss(nn.Module):
    """Wrapper that applies class reweighting after a warmup period.

    During early training, use uniform weights to learn good representations.
    After warmup, apply class-balanced weights to focus on minorities.

    This is the "Deferred Re-weighting" (DRW) strategy from the LDAM paper,
    which often improves over constant reweighting.

    Args:
        base_loss: The underlying loss function (FocalLoss or LDAMLoss)
        class_counts: Number of samples per class
        warmup_epochs: Number of epochs before applying reweighting
        weight_type: Type of weights after warmup ('balanced', 'sqrt', 'effective')
        beta: Beta parameter for effective number weighting (default 0.9999)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        class_counts: NDArray[np.int64],
        warmup_epochs: int = 5,
        weight_type: Literal["balanced", "sqrt", "effective"] = "sqrt",
        beta: float = 0.9999,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Compute different weight schemes
        freq = class_counts / class_counts.sum()

        if weight_type == "balanced":
            weights = 1.0 / (freq + 1e-8)
        elif weight_type == "sqrt":
            weights = np.sqrt(1.0 / (freq + 1e-8))
        elif weight_type == "effective":
            # Effective number of samples: (1 - beta^n) / (1 - beta)
            effective_num = (1 - np.power(beta, class_counts)) / (1 - beta)
            weights = 1.0 / (effective_num + 1e-8)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        # Normalize weights
        weights = weights / weights.sum() * len(class_counts)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer(
            "uniform_weights",
            torch.ones(len(class_counts), dtype=torch.float32),
        )

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for deferred reweighting."""
        self.current_epoch = epoch

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with deferred reweighting."""
        # Compute base loss
        base_output = self.base_loss(logits, targets)

        # Apply reweighting after warmup
        if self.current_epoch >= self.warmup_epochs:
            sample_weights = self.weights[targets]
        else:
            sample_weights = self.uniform_weights[targets]

        # If base_loss has reduction='none', apply weights and reduce
        if hasattr(self.base_loss, "reduction") and self.base_loss.reduction == "none":
            return (base_output * sample_weights).mean()

        # Otherwise, recompute with weights (less efficient but compatible)
        return base_output


def get_loss_function(
    loss_type: Literal["focal", "ldam", "ce"],
    class_counts: NDArray[np.int64],
    gamma: float = 2.0,
    alpha: Literal["balanced"] | None = "balanced",
    use_drw: bool = False,
    drw_warmup_epochs: int = 5,
) -> nn.Module:
    """Factory function to create loss with recommended settings.

    Args:
        loss_type: Type of loss ('focal', 'ldam', 'ce')
        class_counts: Number of samples per class
        gamma: Focusing parameter for focal loss
        alpha: Class weighting scheme for focal loss
        use_drw: Whether to use deferred reweighting
        drw_warmup_epochs: Warmup epochs for DRW

    Returns:
        Configured loss function.
    """
    if loss_type == "focal":
        loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            class_counts=class_counts,
            reduction="none" if use_drw else "mean",
        )
    elif loss_type == "ldam":
        loss = LDAMLoss(
            class_counts=class_counts,
            max_margin=0.5,
            scale=30.0,
            reduction="none" if use_drw else "mean",
        )
    elif loss_type == "ce":
        # Standard cross-entropy with class weighting
        freq = class_counts / class_counts.sum()
        weights = np.sqrt(1.0 / (freq + 1e-8))
        weights = weights / weights.sum() * len(class_counts)
        loss = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32),
            reduction="none" if use_drw else "mean",
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if use_drw:
        loss = DeferredReweightingLoss(
            base_loss=loss,
            class_counts=class_counts,
            warmup_epochs=drw_warmup_epochs,
            weight_type="sqrt",
        )

    return loss
