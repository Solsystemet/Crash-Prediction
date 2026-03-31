"""Neural network hierarchical classifier implementation.

Uses 4 independent binary MLP classifiers, one per hierarchy level.
Uses class weighting instead of SMOTE for handling imbalance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from training.hierarchical.base import HierarchicalClassifierBase
from training.hierarchical.config import NeuralHierarchicalConfig
from training.hierarchical.structure import HierarchicalTargets, HierarchyLevel

logger = logging.getLogger(__name__)


class BinaryMLP(nn.Module):
    """Binary classification MLP for a single hierarchy level.

    Architecture: input → hidden1 → ReLU → Dropout → hidden2 → ReLU → Dropout → 1
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: tuple[int, int] = (128, 64),
        dropout: float = 0.3,
    ) -> None:
        """Initialize binary MLP.

        Args:
            num_features: Number of input features.
            hidden_sizes: Tuple of hidden layer sizes.
            dropout: Dropout probability.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.network(x).squeeze(-1)


@dataclass
class LevelTrainingHistory:
    """Training history for a single level."""

    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float


class HierarchicalNeuralClassifier(HierarchicalClassifierBase):
    """Neural network hierarchical classifier with class weighting.

    Uses 4 independent binary MLP classifiers:
    - L1: INJURY vs NO_INJURY (all samples)
    - L2: SEVERE vs MINOR (injury samples only)
    - L2.5: FATAL vs INCAPACITATING (severe samples only)
    - L3: REPORTED vs VISIBLE (minor injury samples only)

    Each level uses class weighting in the loss function instead of SMOTE.
    """

    def __init__(self, config: NeuralHierarchicalConfig | None = None):
        """Initialize neural network hierarchical classifier.

        Args:
            config: Neural network configuration. Uses defaults if None.
        """
        if config is None:
            config = NeuralHierarchicalConfig()
        super().__init__(config)
        self.config: NeuralHierarchicalConfig = config
        self.device = torch.device(config.get_device())

        # Level models (created during fit)
        self.l1_model: BinaryMLP | None = None
        self.l2_model: BinaryMLP | None = None
        self.l25_model: BinaryMLP | None = None
        self.l3_model: BinaryMLP | None = None

        # Training histories
        self.histories: dict[HierarchyLevel, LevelTrainingHistory] = {}

    def fit(
        self,
        X_train: np.ndarray,
        targets: HierarchicalTargets,
        X_val: np.ndarray | None = None,
        targets_val: HierarchicalTargets | None = None,
    ) -> "HierarchicalNeuralClassifier":
        """Train all hierarchy level models.

        Args:
            X_train: Training feature matrix.
            targets: HierarchicalTargets with binary labels for training.
            X_val: Optional validation feature matrix for early stopping.
            targets_val: Optional validation targets.

        Returns:
            Self for method chaining.
        """
        num_features = X_train.shape[1]

        # Level 1: INJURY vs NO_INJURY (all samples)
        self.l1_model, self.histories[HierarchyLevel.L1_INJURY] = self._fit_level(
            X_train, targets.y_injury,
            X_val, targets_val.y_injury if targets_val else None,
            level=HierarchyLevel.L1_INJURY,
            num_features=num_features,
        )

        # Level 2: SEVERE vs MINOR (injury samples only)
        injury_mask_train = targets.y_injury == 1
        injury_mask_val = targets_val.y_injury == 1 if targets_val else None

        self.l2_model, self.histories[HierarchyLevel.L2_SEVERITY] = self._fit_level(
            X_train[injury_mask_train], targets.y_severe[injury_mask_train],
            X_val[injury_mask_val] if X_val is not None and injury_mask_val is not None else None,
            targets_val.y_severe[injury_mask_val] if targets_val and injury_mask_val is not None else None,
            level=HierarchyLevel.L2_SEVERITY,
            num_features=num_features,
        )

        # Level 2.5: FATAL vs INCAPACITATING (severe samples only)
        severe_mask_train = (targets.y_injury == 1) & (targets.y_severe == 1)
        severe_mask_val = (targets_val.y_injury == 1) & (targets_val.y_severe == 1) if targets_val else None

        n_fatal = np.sum(targets.y_fatal[severe_mask_train] == 1)
        n_incap = np.sum(targets.y_fatal[severe_mask_train] == 0)

        if n_fatal > 0 and n_incap > 0:
            self.l25_model, self.histories[HierarchyLevel.L25_FATAL] = self._fit_level(
                X_train[severe_mask_train], targets.y_fatal[severe_mask_train],
                X_val[severe_mask_val] if X_val is not None and severe_mask_val is not None else None,
                targets_val.y_fatal[severe_mask_val] if targets_val and severe_mask_val is not None else None,
                level=HierarchyLevel.L25_FATAL,
                num_features=num_features,
            )
        else:
            logger.warning("Skipping L2.5 training - insufficient FATAL samples")

        # Level 3: REPORTED vs VISIBLE (minor injury samples only)
        minor_mask_train = (targets.y_injury == 1) & (targets.y_severe == 0)
        minor_mask_val = (targets_val.y_injury == 1) & (targets_val.y_severe == 0) if targets_val else None

        n_reported = np.sum(targets.y_reported[minor_mask_train] == 1)
        n_visible = np.sum(targets.y_reported[minor_mask_train] == 0)

        if n_reported > 0 and n_visible > 0:
            self.l3_model, self.histories[HierarchyLevel.L3_REPORTED] = self._fit_level(
                X_train[minor_mask_train], targets.y_reported[minor_mask_train],
                X_val[minor_mask_val] if X_val is not None and minor_mask_val is not None else None,
                targets_val.y_reported[minor_mask_val] if targets_val and minor_mask_val is not None else None,
                level=HierarchyLevel.L3_REPORTED,
                num_features=num_features,
            )
        else:
            logger.warning("Skipping L3 training - insufficient class samples")

        return self

    def _fit_level(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        level: HierarchyLevel,
        num_features: int,
    ) -> tuple[BinaryMLP, LevelTrainingHistory]:
        """Fit a single hierarchy level model.

        Args:
            X_train: Training features for this level.
            y_train: Training labels for this level.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            level: Which hierarchy level.
            num_features: Number of input features.

        Returns:
            Tuple of (trained model, training history).
        """
        logger.info("=" * 60)
        logger.info(f"{level.name}: Training neural network")
        logger.info("=" * 60)

        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        logger.info(f"Class distribution: negative={n_neg}, positive={n_pos}")

        # Create model
        model = BinaryMLP(
            num_features=num_features,
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout,
        ).to(self.device)

        # Compute class weight for positive class
        pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0]).to(self.device)
        if self.config.use_class_weights:
            logger.info(f"Using positive class weight: {pos_weight.item():.2f}")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        # Training loop
        history = LevelTrainingHistory(
            train_losses=[],
            val_losses=[],
            best_epoch=0,
            best_val_loss=float("inf"),
        )

        best_model_state = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(X_batch)

            train_loss /= len(train_dataset)
            history.train_losses.append(train_loss)

            # Validate
            val_loss = train_loss  # Default if no validation set
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        logits = model(X_batch)
                        loss = criterion(logits, y_batch)
                        val_loss += loss.item() * len(X_batch)
                val_loss /= len(val_loader.dataset)

            history.val_losses.append(val_loss)

            # Early stopping check
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == self.config.epochs - 1:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {history.best_epoch + 1}")

        return model, history

    def predict_proba(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get probability predictions for all levels.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (l1_proba, l2_proba, l25_proba, l3_proba).
        """
        if self.l1_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_tensor = torch.FloatTensor(X).to(self.device)

        def get_proba(model: BinaryMLP | None) -> np.ndarray:
            if model is None:
                return np.full(len(X), 0.5)
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                proba = torch.sigmoid(logits).cpu().numpy()
            return proba

        l1_proba = get_proba(self.l1_model)
        l2_proba = get_proba(self.l2_model)
        l25_proba = get_proba(self.l25_model)
        l3_proba = get_proba(self.l3_model)

        return l1_proba, l2_proba, l25_proba, l3_proba
