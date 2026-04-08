"""Training utilities for the aggregate injury prediction model.

This module provides functions for training and evaluating the AggregateInjuryMLP model
for multi-output regression tasks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_preparation.aggregate_data import AggregateDataset
from training.models import AggregateInjuryMLP
from training.device_utils import get_device_for_config
from training.losses import (
    get_loss_function,
    is_count_loss,
    compute_target_weights,
    NegativeBinomialLoss,
    WeightedMSELoss,
    WeightedPoissonLoss,
)


@dataclass
class AggregateTrainingConfig:
    """Configuration for aggregate model training.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for Adam optimizer.
        early_stopping_patience: Stop if val loss doesn't improve for this many epochs.
        device: Device to train on ("cuda" or "cpu").
        hidden_sizes: Tuple of hidden layer sizes for the MLP.
        dropout: Dropout probability.
        loss_function: Loss function type - "mse", "weighted_mse", "poisson", "weighted_poisson", or "negbin".
    """

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    device: str = field(default_factory=get_device_for_config)
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
    )
    dropout: float = 0.3
    loss_function: Literal[
        "mse", "weighted_mse", "poisson", "weighted_poisson", "negbin"
    ] = "mse"


@dataclass
class AggregateTrainingHistory:
    """Training history containing metrics per epoch.

    Attributes:
        train_losses: Training MSE loss per epoch.
        val_losses: Validation MSE loss per epoch.
        train_mae: Training MAE per epoch.
        val_mae: Validation MAE per epoch.
        best_epoch: Epoch with best validation loss.
        target_names: Names of target columns.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_mae: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    best_epoch: int = 0
    target_names: list[str] = field(default_factory=list)


@dataclass
class AggregateEvaluationMetrics:
    """Evaluation metrics for aggregate model.

    Attributes:
        mse: Mean Squared Error (overall).
        mae: Mean Absolute Error (overall).
        rmse: Root Mean Squared Error (overall).
        r2: R-squared score (overall).
        per_target_mse: MSE for each target.
        per_target_mae: MAE for each target.
        per_target_r2: R-squared for each target.
        target_names: Names of target columns.
    """

    mse: float
    mae: float
    rmse: float
    r2: float
    per_target_mse: dict[str, float]
    per_target_mae: dict[str, float]
    per_target_r2: dict[str, float]
    target_names: list[str]

    def __str__(self) -> str:
        lines = [
            "Aggregate Model Evaluation Metrics:",
            f"  Overall MSE:  {self.mse:.4f}",
            f"  Overall MAE:  {self.mae:.4f}",
            f"  Overall RMSE: {self.rmse:.4f}",
            f"  Overall R²:   {self.r2:.4f}",
            "",
            "  Per-Target Metrics:",
        ]
        for name in self.target_names:
            lines.append(
                f"    {name}: MSE={self.per_target_mse[name]:.4f}, "
                f"MAE={self.per_target_mae[name]:.4f}, "
                f"R²={self.per_target_r2[name]:.4f}"
            )
        return "\n".join(lines)


def train_aggregate_model(
    train_dataset: AggregateDataset,
    val_dataset: AggregateDataset,
    num_features: int,
    num_targets: int,
    target_names: list[str],
    config: AggregateTrainingConfig | None = None,
    verbose: bool = True,
    log_transform_targets: bool = False,
) -> tuple[AggregateInjuryMLP, AggregateTrainingHistory]:
    """Train the aggregate injury prediction model.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        num_features: Number of input features.
        num_targets: Number of output targets.
        target_names: Names of the target columns.
        config: Training configuration. Uses defaults if None.
        verbose: Whether to print progress.
        log_transform_targets: Whether targets are log-transformed.

    Returns:
        Tuple of (trained model, training history).
    """
    if config is None:
        config = AggregateTrainingConfig()

    device = torch.device(config.device)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Validate: log_transform and count losses are mutually exclusive
    if log_transform_targets and is_count_loss(config.loss_function):
        raise ValueError(
            f"Cannot use log_transform_targets=True with loss_function='{config.loss_function}'. "
            "Poisson and Negative Binomial losses expect raw counts, not log-transformed targets."
        )

    # Initialize model
    model = AggregateInjuryMLP(
        num_features=num_features,
        num_targets=num_targets,
        hidden_sizes=config.hidden_sizes,
        dropout=config.dropout,
    ).to(device)

    # Compute target weights for weighted losses
    target_weights = None
    if config.loss_function in ("weighted_mse", "weighted_poisson"):
        # Compute mean of each target from training data
        all_targets = train_dataset.targets  # (num_samples, num_targets)
        target_means = all_targets.mean(dim=0)  # (num_targets,)
        target_weights = compute_target_weights(target_means, num_targets)
        target_weights = target_weights.to(device)
        if verbose:
            print(f"Target means: {target_means.tolist()}")
            print(f"Target weights: {target_weights.tolist()}")

    # Create loss function
    criterion = get_loss_function(
        config.loss_function,
        num_targets=num_targets,
        target_weights=target_weights,
    )
    if isinstance(
        criterion, (NegativeBinomialLoss, WeightedMSELoss, WeightedPoissonLoss)
    ):
        # Move to device
        criterion = criterion.to(device)

    # For count losses, include dispersion params in optimizer if applicable
    params_to_optimize = list(model.parameters())
    if isinstance(criterion, NegativeBinomialLoss):
        params_to_optimize.extend(criterion.parameters())

    optimizer = Adam(params_to_optimize, lr=config.learning_rate)

    # Track if using count loss (outputs are log-rates)
    using_count_loss = is_count_loss(config.loss_function)

    # Training history
    history = AggregateTrainingHistory(target_names=target_names)

    # Early stopping state
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Create short names for display (abbreviate injury types)
    name_map = {
        "INJURIES_FATAL": "Fatal",
        "INJURIES_INCAPACITATING": "Incapac.",
        "INJURIES_NON_INCAPACITATING": "Non-Incap.",
        "INJURIES_REPORTED_NOT_EVIDENT": "Not Evid.",
        "INJURIES_NO_INDICATION": "No Indic.",
    }
    short_names = [name_map.get(name, name[:10]) for name in target_names]

    if verbose:
        print(f"\nTraining Aggregate Injury Model on {device}")
        print(f"Targets: {', '.join(target_names)}")
        print(f"Loss function: {config.loss_function.upper()}")
        if log_transform_targets:
            print("Target transformation: log(1+x)")
        if using_count_loss:
            print("Note: Model outputs log-rates, metrics computed after exp()")
        print()
        print("Per-target columns show Validation MAE for each injury type:")
        print()
        # Adjust header based on loss type
        loss_col_name = "Tr.Loss" if using_count_loss else "Tr.MSE"
        val_col_name = "Va.Loss" if using_count_loss else "Va.MSE"
        header = f"{'Epoch':<6} {loss_col_name:<8} {val_col_name:<8} "
        header += " ".join([f"{name:<10}" for name in short_names])
        print(header)
        print("-" * len(header))

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_total = 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = features.size(0)
            train_loss += loss.item() * batch_size

            # For count losses, compute MAE on actual rates (exp(outputs))
            if using_count_loss:
                pred_rates = torch.exp(outputs)
                train_mae += torch.abs(pred_rates - targets).sum().item()
            else:
                train_mae += torch.abs(outputs - targets).sum().item()
            train_total += batch_size * num_targets

        train_loss /= len(train_dataset)
        train_mae /= train_total

        # Validation phase - track per-target metrics
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_total = 0
        per_target_mae = torch.zeros(num_targets)
        per_target_count = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)

                batch_size = features.size(0)
                val_loss += loss.item() * batch_size

                # For count losses, compute MAE on actual rates (exp(outputs))
                if using_count_loss:
                    pred_rates = torch.exp(outputs)
                    val_mae += torch.abs(pred_rates - targets).sum().item()
                    per_target_mae += torch.abs(pred_rates - targets).sum(dim=0).cpu()
                else:
                    val_mae += torch.abs(outputs - targets).sum().item()
                    per_target_mae += torch.abs(outputs - targets).sum(dim=0).cpu()
                val_total += batch_size * num_targets
                per_target_count += batch_size

        val_loss /= len(val_dataset)
        val_mae /= val_total
        per_target_mae /= per_target_count

        # Display metrics
        per_target_mae_display = per_target_mae
        train_loss_display = train_loss
        val_loss_display = val_loss

        # Record history (in original normalized scale for consistency)
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.train_mae.append(train_mae)
        history.val_mae.append(val_mae)

        if verbose:
            line = (
                f"{epoch + 1:<6} {train_loss_display:<8.2f} {val_loss_display:<8.2f} "
            )
            line += " ".join(
                [f"{mae:<10.2f}" for mae in per_target_mae_display.tolist()]
            )
            print(line)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            history.best_epoch = epoch
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if verbose:
        print(f"\nBest epoch: {history.best_epoch + 1} (val_loss: {best_val_loss:.4f})")

    return model, history


def evaluate_aggregate_model(
    model: AggregateInjuryMLP,
    dataset: AggregateDataset,
    target_names: list[str],
    device: str = "cpu",
    log_transform_targets: bool = False,
    loss_function: str = "mse",
) -> AggregateEvaluationMetrics:
    """Evaluate the aggregate model on a dataset.

    Args:
        model: Trained model.
        dataset: Dataset to evaluate on.
        target_names: Names of the target columns.
        device: Device to run evaluation on.
        log_transform_targets: Whether targets were log-transformed (needs inverse).
        loss_function: Loss function used during training ("mse", "poisson", "negbin").

    Returns:
        AggregateEvaluationMetrics with all metrics.
    """
    model.eval()
    device_obj = torch.device(device)
    model = model.to(device_obj)

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device_obj)
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    # Transform predictions/targets back to original scale
    if is_count_loss(loss_function):
        # Model outputs log-rates, convert to rates
        predictions = np.exp(predictions)
    elif log_transform_targets:
        # Inverse of log(1 + x) is exp(x) - 1
        predictions = np.expm1(predictions)
        targets = np.expm1(targets)

    # Overall metrics
    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(mse))

    # R-squared (coefficient of determination)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Per-target metrics
    per_target_mse = {}
    per_target_mae = {}
    per_target_r2 = {}

    for i, name in enumerate(target_names):
        pred_col = predictions[:, i]
        tgt_col = targets[:, i]

        per_target_mse[name] = float(np.mean((pred_col - tgt_col) ** 2))
        per_target_mae[name] = float(np.mean(np.abs(pred_col - tgt_col)))

        ss_res_col = np.sum((tgt_col - pred_col) ** 2)
        ss_tot_col = np.sum((tgt_col - np.mean(tgt_col)) ** 2)
        per_target_r2[name] = (
            float(1 - (ss_res_col / ss_tot_col)) if ss_tot_col > 0 else 0.0
        )

    return AggregateEvaluationMetrics(
        mse=mse,
        mae=mae,
        rmse=rmse,
        r2=r2,
        per_target_mse=per_target_mse,
        per_target_mae=per_target_mae,
        per_target_r2=per_target_r2,
        target_names=target_names,
    )


def save_aggregate_model(model: AggregateInjuryMLP, path: Path | str) -> None:
    """Save the aggregate model to disk.

    Args:
        model: Trained model to save.
        path: Path to save the model (.pt file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_features": model.num_features,
            "num_targets": model.num_targets,
            "hidden_sizes": model.hidden_sizes,
            "dropout": model.dropout,
        },
        path,
    )


def load_aggregate_model(path: Path | str, device: str = "cpu") -> AggregateInjuryMLP:
    """Load an aggregate model from disk.

    Args:
        path: Path to the saved model (.pt file).
        device: Device to load the model to.

    Returns:
        Loaded model.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = AggregateInjuryMLP(
        num_features=checkpoint["num_features"],
        num_targets=checkpoint["num_targets"],
        hidden_sizes=checkpoint["hidden_sizes"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model
