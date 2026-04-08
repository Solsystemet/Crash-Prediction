"""Custom loss functions for count data prediction.

This module provides loss functions appropriate for count/rate data:
- PoissonLoss: For count data where variance ≈ mean
- NegativeBinomialLoss: For overdispersed count data where variance > mean

Both losses expect the model to output log-rates (log(λ)) for numerical stability.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PoissonLoss(nn.Module):
    """Poisson Negative Log-Likelihood Loss for count data.

    This loss is appropriate when:
    - Targets are non-negative counts (0, 1, 2, ...)
    - Variance is approximately equal to the mean

    The model should output log-rates (log(λ)), not raw rates.
    During inference, apply exp() to get the predicted rate/count.

    Loss formula: L = exp(log_λ) - y * log_λ = λ - y * log(λ)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize PoissonLoss.

        Args:
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        # PyTorch's PoissonNLLLoss with log_input=True expects log(lambda)
        self._loss = nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")

    def forward(self, log_rate: Tensor, target: Tensor) -> Tensor:
        """Compute Poisson NLL loss.

        Args:
            log_rate: Predicted log-rates from model, shape (batch, num_targets).
            target: Actual counts, shape (batch, num_targets).

        Returns:
            Scalar loss value.
        """
        # Ensure targets are non-negative
        target = torch.clamp(target, min=0.0)
        return self._loss(log_rate, target)


class NegativeBinomialLoss(nn.Module):
    """Negative Binomial Negative Log-Likelihood Loss for overdispersed count data.

    This loss is appropriate when:
    - Targets are non-negative counts (0, 1, 2, ...)
    - Variance is greater than the mean (overdispersion)

    The Negative Binomial distribution has two parameters:
    - μ (mu): mean/rate (predicted by the model as log(μ))
    - α (alpha): dispersion parameter (learned per target)

    Variance = μ + α * μ²

    When α → 0, NegBin → Poisson.
    When α > 0, variance exceeds mean (overdispersion).

    The model should output log-rates (log(μ)), not raw rates.
    During inference, apply exp() to get the predicted rate/count.
    """

    def __init__(
        self,
        num_targets: int,
        init_alpha: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        """Initialize NegativeBinomialLoss.

        Args:
            num_targets: Number of target columns (for per-target dispersion).
            init_alpha: Initial value for dispersion parameters.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.num_targets = num_targets

        # Learnable log-dispersion parameters (one per target)
        # Using log(alpha) ensures alpha > 0 after exp()
        self._log_alpha = nn.Parameter(
            torch.full((num_targets,), fill_value=float(torch.tensor(init_alpha).log()))
        )

    @property
    def alpha(self) -> Tensor:
        """Get dispersion parameters (always positive)."""
        return torch.exp(self._log_alpha)

    def forward(self, log_rate: Tensor, target: Tensor) -> Tensor:
        """Compute Negative Binomial NLL loss.

        Args:
            log_rate: Predicted log-rates from model, shape (batch, num_targets).
            target: Actual counts, shape (batch, num_targets).

        Returns:
            Scalar loss value.
        """
        # Ensure targets are non-negative
        target = torch.clamp(target, min=0.0)

        # Get rate μ = exp(log_rate)
        mu = torch.exp(log_rate)

        # Get dispersion α (broadcast to batch size)
        alpha = self.alpha  # shape: (num_targets,)

        # Compute r = 1/α (shape parameter of NegBin)
        # As α → 0, r → ∞, and NegBin → Poisson
        r = 1.0 / (alpha + self.eps)

        # Negative Binomial NLL:
        # -log P(y|μ,r) = -log(Γ(y+r)/(Γ(r)*y!)) - r*log(r/(r+μ)) - y*log(μ/(r+μ))
        #
        # Simplify using:
        # log(Γ(y+r)) - log(Γ(r)) - log(y!) = lgamma(y+r) - lgamma(r) - lgamma(y+1)

        # Term 1: log(Γ(y+r)) - log(Γ(r)) - log(y!)
        term1 = torch.lgamma(target + r) - torch.lgamma(r) - torch.lgamma(target + 1)

        # Term 2: r * log(r / (r + μ))
        term2 = r * torch.log(r / (r + mu + self.eps) + self.eps)

        # Term 3: y * log(μ / (r + μ))
        term3 = target * torch.log(mu / (r + mu + self.eps) + self.eps)

        # NLL = -(term1 + term2 + term3)
        nll = -(term1 + term2 + term3)

        # Return mean loss
        return nll.mean()

    def get_dispersion_summary(self) -> dict[str, float]:
        """Get summary of learned dispersion parameters.

        Returns:
            Dictionary with dispersion values per target index.
        """
        alpha = self.alpha.detach().cpu().numpy()
        return {f"target_{i}": float(alpha[i]) for i in range(self.num_targets)}


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss that gives more importance to rare targets.

    Each target column is weighted inversely by its mean value, so rare
    events (like fatal injuries) contribute more to the loss than common
    events (like no injury indication).

    Weight formula: w_i = 1 / (mean_i + eps), then normalized to sum to num_targets
    """

    target_weights: Tensor

    def __init__(self, target_weights: Tensor) -> None:
        """Initialize WeightedMSELoss.

        Args:
            target_weights: Pre-computed weights for each target, shape (num_targets,).
        """
        super().__init__()
        self.register_buffer("target_weights", target_weights)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute weighted MSE loss.

        Args:
            predictions: Predicted values, shape (batch, num_targets).
            targets: Actual values, shape (batch, num_targets).

        Returns:
            Scalar loss value.
        """
        # Compute squared errors per target
        squared_errors = (predictions - targets) ** 2  # (batch, num_targets)

        # Weight each target
        weighted_errors = squared_errors * self.target_weights  # (batch, num_targets)

        # Return mean
        return weighted_errors.mean()


class WeightedPoissonLoss(nn.Module):
    """Weighted Poisson loss that gives more importance to rare targets.

    Combines Poisson NLL with target weighting for imbalanced count data.
    """

    target_weights: Tensor

    def __init__(self, target_weights: Tensor, eps: float = 1e-8) -> None:
        """Initialize WeightedPoissonLoss.

        Args:
            target_weights: Pre-computed weights for each target, shape (num_targets,).
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.register_buffer("target_weights", target_weights)

    def forward(self, log_rate: Tensor, target: Tensor) -> Tensor:
        """Compute weighted Poisson NLL loss.

        Args:
            log_rate: Predicted log-rates from model, shape (batch, num_targets).
            target: Actual counts, shape (batch, num_targets).

        Returns:
            Scalar loss value.
        """
        # Ensure targets are non-negative
        target = torch.clamp(target, min=0.0)

        # Poisson NLL: lambda - y * log(lambda) = exp(log_rate) - y * log_rate
        rate = torch.exp(log_rate)
        nll = rate - target * log_rate  # (batch, num_targets)

        # Weight each target
        weighted_nll = nll * self.target_weights  # (batch, num_targets)

        return weighted_nll.mean()


def compute_target_weights(
    target_means: Tensor,
    num_targets: int,
    eps: float = 1e-8,
) -> Tensor:
    """Compute inverse-frequency weights for targets.

    Weights are computed as: w_i = 1 / (mean_i + eps)
    Then normalized so they sum to num_targets (preserves loss scale).

    Args:
        target_means: Mean value of each target in training data, shape (num_targets,).
        num_targets: Number of targets.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized weights, shape (num_targets,).
    """
    # Inverse of mean (higher weight for rarer targets)
    raw_weights = 1.0 / (target_means + eps)

    # Normalize so weights sum to num_targets (preserves loss scale)
    normalized_weights = raw_weights * (num_targets / raw_weights.sum())

    return normalized_weights


def get_loss_function(
    loss_type: str,
    num_targets: int = 5,
    target_weights: Tensor | None = None,
) -> nn.Module:
    """Factory function to create loss functions.

    Args:
        loss_type: Type of loss - "mse", "weighted_mse", "poisson", "weighted_poisson", or "negbin".
        num_targets: Number of targets (needed for NegBin dispersion params).
        target_weights: Weights for each target (required for weighted losses).

    Returns:
        Loss module.

    Raises:
        ValueError: If loss_type is not recognized or weights missing for weighted loss.
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "weighted_mse":
        if target_weights is None:
            raise ValueError("target_weights required for weighted_mse loss")
        return WeightedMSELoss(target_weights)
    elif loss_type == "poisson":
        return PoissonLoss()
    elif loss_type == "weighted_poisson":
        if target_weights is None:
            raise ValueError("target_weights required for weighted_poisson loss")
        return WeightedPoissonLoss(target_weights)
    elif loss_type == "negbin":
        return NegativeBinomialLoss(num_targets=num_targets)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            "Choose from 'mse', 'weighted_mse', 'poisson', 'weighted_poisson', 'negbin'."
        )


def is_count_loss(loss_type: str) -> bool:
    """Check if loss type expects log-rate outputs.

    Args:
        loss_type: Type of loss.

    Returns:
        True if the loss expects log-rate outputs (need exp() for inference).
    """
    return loss_type in ("poisson", "weighted_poisson", "negbin")
