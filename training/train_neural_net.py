"""Neural network training utilities.

This module provides functions for training and evaluating the MLP model.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_preparation.tensor_dataset import CrashTensorDataset
from training.models import CrashPredictionMLP
from training.device_utils import get_device_for_config


@dataclass
class TrainingConfig:
    """Configuration for neural network training.

    Attributes:
        epochs: Maximum number of training epochs.
        learning_rate: Learning rate for Adam optimizer.
        early_stopping_patience: Stop if val loss doesn't improve for this many epochs.
        device: Device to train on ("cuda" or "cpu").
        use_class_weights: Whether to use class weights to handle imbalance.
    """

    epochs: int = 30
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5
    device: str = field(default_factory=get_device_for_config)
    use_class_weights: bool = True


@dataclass
class TrainingHistory:
    """Training history containing metrics per epoch.

    Attributes:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_accuracies: Training accuracy per epoch.
        val_accuracies: Validation accuracy per epoch.
        best_epoch: Epoch with best validation loss.
    """

    train_losses: list[float]
    val_losses: list[float]
    train_accuracies: list[float]
    val_accuracies: list[float]
    best_epoch: int


def compute_class_weights(
    dataset: CrashTensorDataset, num_classes: int
) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequency.

    Args:
        dataset: Dataset to compute weights from.
        num_classes: Total number of classes (from encoder).

    Returns:
        Tensor of class weights.
    """
    labels = dataset.labels.numpy()
    classes, counts = np.unique(labels, return_counts=True)

    # Inverse frequency weighting
    weights = 1.0 / counts
    # Normalize so weights sum to num_classes present in data
    weights = weights / weights.sum() * len(classes)

    # Create full weight tensor for all classes (some may not be in training data)
    # Use weight of 1.0 for missing classes
    full_weights = np.ones(num_classes)
    for cls, weight in zip(classes, weights):
        if cls < num_classes:
            full_weights[cls] = weight

    return torch.FloatTensor(full_weights)


def train_neural_network(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_features: int,
    num_classes: int,
    config: TrainingConfig | None = None,
    verbose: bool = True,
    train_dataset: CrashTensorDataset | None = None,
) -> tuple[CrashPredictionMLP, TrainingHistory]:
    """Train a neural network on the provided data.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_features: Number of input features.
        num_classes: Number of output classes.
        config: Training configuration. Uses defaults if None.
        verbose: Whether to print progress.
        train_dataset: Training dataset (needed for class weight computation).

    Returns:
        Tuple of (trained model, training history).
    """
    if config is None:
        config = TrainingConfig()

    device = torch.device(config.device)

    # Initialize model
    model = CrashPredictionMLP(
        num_features=num_features,
        num_classes=num_classes,
    ).to(device)

    # Compute class weights if requested
    class_weights = None
    if config.use_class_weights and train_dataset is not None:
        class_weights = compute_class_weights(train_dataset, num_classes).to(device)
        if verbose:
            print(f"Using class weights: {class_weights.cpu().numpy().round(2)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    # Training history
    history = TrainingHistory(
        train_losses=[],
        val_losses=[],
        train_accuracies=[],
        val_accuracies=[],
        best_epoch=0,
    )

    # Early stopping state
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    if verbose:
        print(f"\nTraining Neural Network on {device}")
        print(
            f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Train Acc':<12} {'Val Acc':<12}"
        )
        print("-" * 56)

    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Record history
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.train_accuracies.append(train_acc)
        history.val_accuracies.append(val_acc)

        if verbose:
            print(
                f"{epoch + 1:<8} {train_loss:<12.4f} {val_loss:<12.4f} {train_acc:<12.4f} {val_acc:<12.4f}"
            )

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


def save_model(model: CrashPredictionMLP, path: Path | str) -> None:
    """Save the model state dict to disk.

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
            "num_classes": model.num_classes,
            "dropout": model.dropout,
        },
        path,
    )


def load_model(path: Path | str, device: str = "cpu") -> CrashPredictionMLP:
    """Load a model from disk.

    Args:
        path: Path to the saved model (.pt file).
        device: Device to load the model to.

    Returns:
        Loaded model.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = CrashPredictionMLP(
        num_features=checkpoint["num_features"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model
