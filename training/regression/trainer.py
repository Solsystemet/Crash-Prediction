"""Training utilities for crash count regression.

Provides the training loop with:
- Multiple loss functions (MSE, Huber, Poisson)
- Early stopping
- Learning rate scheduling
- Training history tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from training.regression.config import RegressionConfig

logger = logging.getLogger(__name__)


def get_loss_function(
    loss_type: str,
    huber_delta: float = 1.0,
) -> nn.Module:
    """Get loss function by name.

    Args:
        loss_type: Type of loss ('mse', 'huber', 'poisson').
        huber_delta: Delta parameter for Huber loss.

    Returns:
        PyTorch loss module.
    """
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "huber":
        return nn.HuberLoss(delta=huber_delta)
    elif loss_type == "poisson":
        return nn.PoissonNLLLoss(log_input=False)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


@dataclass
class EarlyStoppingState:
    """State tracker for early stopping."""

    best_loss: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0
    best_state_dict: dict | None = None


class RegressionTrainer:
    """Trainer for crash count regression models.

    Handles:
    - Data loading and batching
    - Training loop with early stopping
    - Validation monitoring
    - Learning rate scheduling
    """

    def __init__(self, config: RegressionConfig) -> None:
        """Initialize trainer.

        Args:
            config: Regression configuration.
        """
        self.config = config
        self.device = torch.device(config.get_device())

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create DataLoader from numpy arrays.

        Args:
            X: Feature array.
            y: Target array.
            shuffle: Whether to shuffle data.

        Returns:
            PyTorch DataLoader.
        """
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    def _train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Run one training epoch.

        Args:
            model: Model to train.
            dataloader: Training data loader.
            optimizer: Optimizer.
            criterion: Loss function.

        Returns:
            Average training loss for the epoch.
        """
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def _validate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Compute validation loss.

        Args:
            model: Model to evaluate.
            dataloader: Validation data loader.
            criterion: Loss function.

        Returns:
            Average validation loss.
        """
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = model(X_batch)
                loss = criterion(preds, y_batch)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def fit(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the model.

        Args:
            model: Model to train (should already be on device).
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional but recommended).
            y_val: Validation targets.

        Returns:
            Training history with 'train_loss' and optionally 'val_loss'.
        """
        # Setup
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        criterion = get_loss_function(self.config.loss_type, self.config.huber_delta)
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=False,
        )

        # Early stopping state
        es_state = EarlyStoppingState()

        # Training history
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        # Training loop
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            history["train_loss"].append(train_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            # Validate
            if val_loader is not None:
                val_loss = self._validate(model, val_loader, criterion)
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < es_state.best_loss - self.config.min_delta:
                    es_state.best_loss = val_loss
                    es_state.best_epoch = epoch
                    es_state.epochs_without_improvement = 0
                    es_state.best_state_dict = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                else:
                    es_state.epochs_without_improvement += 1

                if es_state.epochs_without_improvement >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                # Log progress
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.epochs} - "
                        f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.epochs} - "
                        f"train_loss: {train_loss:.4f}"
                    )

        # Restore best model
        if es_state.best_state_dict is not None:
            model.load_state_dict(es_state.best_state_dict)
            logger.info(f"Restored best model from epoch {es_state.best_epoch + 1}")

        return history


class SequenceTrainer:
    """Trainer for sequence models (LSTM).

    Similar to RegressionTrainer but handles sequence data.
    """

    def __init__(
        self,
        config: RegressionConfig,
        sequence_length: int = 24,
    ) -> None:
        """Initialize sequence trainer.

        Args:
            config: Regression configuration.
            sequence_length: Length of input sequences.
        """
        self.config = config
        self.sequence_length = sequence_length
        self.device = torch.device(config.get_device())

    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create overlapping sequences from time series data.

        Args:
            X: Feature array of shape (n_timesteps, n_features).
            y: Target array of shape (n_timesteps,).

        Returns:
            Tuple of (X_seq, y_seq) where X_seq has shape
            (n_sequences, sequence_length, n_features).
        """
        n_timesteps = len(X)
        n_sequences = n_timesteps - self.sequence_length

        if n_sequences <= 0:
            raise ValueError(
                f"Not enough timesteps ({n_timesteps}) for sequence length {self.sequence_length}"
            )

        X_sequences = []
        y_sequences = []

        for i in range(n_sequences):
            X_sequences.append(X[i : i + self.sequence_length])
            y_sequences.append(y[i + self.sequence_length])

        return np.array(X_sequences), np.array(y_sequences)

    def fit(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train LSTM model on sequences.

        Args:
            model: LSTM model.
            X_train: Training sequences (n, seq_len, features).
            y_train: Training targets (n,).
            X_val: Validation sequences.
            y_val: Validation targets.

        Returns:
            Training history.
        """
        # Convert to sequences if needed
        if X_train.ndim == 2:
            X_train, y_train = self.create_sequences(X_train, y_train)
            if X_val is not None:
                X_val, y_val = self.create_sequences(X_val, y_val)

        # Create dataloaders
        train_tensor = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        )
        train_loader = DataLoader(
            train_tensor,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_tensor = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).float(),
            )
            val_loader = DataLoader(
                val_tensor,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        # Training setup
        criterion = get_loss_function(self.config.loss_type, self.config.huber_delta)
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training
            model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)

            # Validation
            if val_loader:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        preds = model(X_batch)
                        loss = criterion(preds, y_batch)
                        val_losses.append(loss.item())

                avg_val_loss = np.mean(val_losses)
                history["val_loss"].append(avg_val_loss)

                if avg_val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state:
            model.load_state_dict(best_state)

        return history
