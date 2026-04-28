"""Multiclass neural network classifier with training and inference.

This module provides the main classifier class that wraps the neural network
model with training, evaluation, and checkpoint functionality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from training.multiclass.config import MulticlassNeuralConfig
from training.multiclass.losses import FocalLoss, compute_class_weights
from training.multiclass.neural_network import MulticlassMLP

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Training history for monitoring and analysis.

    Attributes:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_accuracies: Training accuracy per epoch.
        val_accuracies: Validation accuracy per epoch.
        learning_rates: Learning rate per epoch.
        best_epoch: Epoch with best validation performance.
        best_val_loss: Best validation loss achieved.
        best_val_accuracy: Best validation accuracy achieved.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0


class MulticlassNeuralClassifier:
    """Neural network classifier for multiclass crash severity prediction.

    This classifier wraps a MulticlassMLP model and provides:
    - Training with early stopping and learning rate scheduling
    - Class weighting via focal loss or weighted cross-entropy
    - Checkpoint saving and loading
    - Probability and label prediction

    Example:
        >>> config = MulticlassNeuralConfig(epochs=50, batch_size=128)
        >>> classifier = MulticlassNeuralClassifier(config)
        >>> classifier.fit(X_train, y_train, X_val, y_val)
        >>> predictions = classifier.predict(X_test)
        >>> probabilities = classifier.predict_proba(X_test)
    """

    def __init__(self, config: MulticlassNeuralConfig | None = None) -> None:
        """Initialize the classifier.

        Args:
            config: Configuration for the classifier. Uses defaults if None.
        """
        self.config = config or MulticlassNeuralConfig()
        self.device = torch.device(self.config.get_device())

        # Model (created during fit)
        self.model: MulticlassMLP | None = None
        self.num_features: int | None = None

        # Training state
        self.history = TrainingHistory()
        self.class_weights: torch.Tensor | None = None
        self.label_encoder: dict[int, str] | None = None

        # Set random seed for reproducibility
        self._set_seed(self.config.random_seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(
        self,
        X_train: NDArray[np.floating[Any]],
        y_train: NDArray[np.integer[Any]],
        X_val: NDArray[np.floating[Any]] | None = None,
        y_val: NDArray[np.integer[Any]] | None = None,
    ) -> "MulticlassNeuralClassifier":
        """Train the classifier.

        Args:
            X_train: Training features of shape (n_samples, n_features).
            y_train: Training labels of shape (n_samples,).
            X_val: Validation features (optional, for early stopping).
            y_val: Validation labels (optional).

        Returns:
            Self for method chaining.
        """
        self.num_features = X_train.shape[1]

        # Create model
        self.model = MulticlassMLP(
            num_features=self.num_features,
            num_classes=self.config.num_classes,
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout,
            use_batch_norm=self.config.use_batch_norm,
            activation=self.config.activation,
        ).to(self.device)

        if self.config.verbose:
            logger.info(f"Model architecture:\n{self.model.summary()}")
            logger.info(f"Training on device: {self.device}")

        # Compute class weights
        self._compute_class_weights(y_train)

        # Create data loaders
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = (
            self._create_dataloader(X_val, y_val, shuffle=False)
            if X_val is not None and y_val is not None
            else None
        )

        # Create loss function
        loss_fn = self._create_loss_fn()

        # Create optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Create learning rate scheduler
        scheduler = self._create_scheduler(optimizer)

        # Training loop
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train one epoch
            train_loss, train_acc = self._train_epoch(
                train_loader, loss_fn, optimizer
            )

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, loss_fn)
            else:
                val_loss, val_acc = train_loss, train_acc

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Update history
            self.history.train_losses.append(train_loss)
            self.history.val_losses.append(val_loss)
            self.history.train_accuracies.append(train_acc)
            self.history.val_accuracies.append(val_acc)
            self.history.learning_rates.append(current_lr)

            # Check for improvement
            improved = val_loss < self.history.best_val_loss - self.config.early_stopping_min_delta

            if improved:
                self.history.best_val_loss = val_loss
                self.history.best_val_accuracy = val_acc
                self.history.best_epoch = epoch
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            # Update learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Logging
            if self.config.verbose and (epoch % 10 == 0 or epoch == self.config.epochs - 1):
                logger.info(
                    f"Epoch {epoch + 1:3d}/{self.config.epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
                    f"lr={current_lr:.2e}"
                )

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                if self.config.verbose:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1}. "
                        f"Best epoch: {self.history.best_epoch + 1}"
                    )
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            self.model.to(self.device)

        if self.config.verbose:
            logger.info(
                f"Training complete. Best val_loss: {self.history.best_val_loss:.4f} "
                f"at epoch {self.history.best_epoch + 1}"
            )

        return self

    def _train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            logits = self.model(features)
            loss = loss_fn(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )

            optimizer.step()

            # Track metrics
            total_loss += loss.item() * features.size(0)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += features.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _validate(
        self, val_loader: DataLoader, loss_fn: nn.Module
    ) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(features)
                loss = loss_fn(logits, labels)

                total_loss += loss.item() * features.size(0)
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += features.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _create_dataloader(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.integer[Any]],
        shuffle: bool = False,
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        features = torch.tensor(X, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(features, labels)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def _compute_class_weights(
        self, y: NDArray[np.integer[Any]]
    ) -> None:
        """Compute class weights from training labels."""
        if self.config.class_weight_strategy is None:
            self.class_weights = None
            return

        # Count samples per class
        unique, counts = np.unique(y, return_counts=True)
        class_counts = {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}

        if self.config.verbose:
            logger.info(f"Class distribution: {class_counts}")

        if self.config.class_weight_strategy == "custom":
            self.class_weights = torch.tensor(
                self.config.custom_class_weights, dtype=torch.float32
            )
        else:
            strategy_map = {
                "inverse_frequency": "inverse_frequency",
                "effective_samples": "effective_samples",
            }
            strategy = strategy_map[self.config.class_weight_strategy]
            self.class_weights = compute_class_weights(
                class_counts,
                strategy=strategy,
                beta=self.config.effective_samples_beta,
            )

        if self.config.verbose:
            logger.info(f"Class weights: {self.class_weights.tolist()}")

    def _create_loss_fn(self) -> nn.Module:
        """Create the loss function."""
        weights = self.class_weights.to(self.device) if self.class_weights is not None else None

        if self.config.loss_type == "focal":
            return FocalLoss(
                alpha=weights,
                gamma=self.config.focal_gamma,
                label_smoothing=self.config.label_smoothing,
            )
        else:
            return nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=self.config.label_smoothing,
            )

    def _create_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.lr_scheduler_factor,
                patience=self.config.lr_scheduler_patience,
                verbose=self.config.verbose,
            )
        elif self.config.lr_scheduler == "cosine":
            return CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        else:
            return None

    def predict(self, X: NDArray[np.floating[Any]]) -> NDArray[np.integer[Any]]:
        """Predict class labels for samples.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        self._check_fitted()
        self.model.eval()

        features = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions = self.model.predict(features)

        return predictions.cpu().numpy()

    def predict_proba(
        self, X: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Predict class probabilities for samples.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """
        self._check_fitted()
        self.model.eval()

        features = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            probabilities = self.model.predict_proba(features)

        return probabilities.cpu().numpy()

    def _check_fitted(self) -> None:
        """Check if the model has been fitted."""
        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict()."
            )

    def save(self, path: str | Path) -> None:
        """Save the trained model and configuration.

        Args:
            path: Path to save the model checkpoint.
        """
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "num_features": self.num_features,
            "class_weights": self.class_weights,
            "history": self.history,
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, force_cpu: bool = True) -> "MulticlassNeuralClassifier":
        """Load a trained model from checkpoint.

        Args:
            path: Path to the model checkpoint.
            force_cpu: If True, force model to load on CPU (avoids Windows deadlocks).

        Returns:
            Loaded classifier ready for inference.
        """
        import os
        
        # Prevent multiprocessing issues on Windows
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        config = checkpoint["config"]
        
        # Force CPU to avoid CUDA initialization deadlocks
        if force_cpu:
            config.device = "cpu"
        
        # Create classifier without triggering CUDA checks
        classifier = object.__new__(cls)
        classifier.config = config
        classifier.device = torch.device("cpu")  # Always CPU for API inference
        classifier.model = None
        classifier.num_features = checkpoint["num_features"]
        classifier.history = checkpoint["history"]
        classifier.class_weights = checkpoint["class_weights"]
        classifier.label_encoder = None

        # Recreate model on CPU
        classifier.model = MulticlassMLP(
            num_features=classifier.num_features,
            num_classes=config.num_classes,
            hidden_sizes=config.hidden_sizes,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
            activation=config.activation,
        )
        classifier.model.load_state_dict(checkpoint["model_state_dict"])
        classifier.model.to(classifier.device)
        classifier.model.eval()

        logger.info(f"Model loaded from {path} (device: {classifier.device})")
        return classifier
