"""Random Forest training utilities.

This module provides functions for training and evaluating Random Forest models.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier

from data_preparation.tensor_dataset import CrashTensorDataset


@dataclass
class RandomForestConfig:
    """Configuration for Random Forest training.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees. None for unlimited.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        n_jobs: Number of parallel jobs. -1 uses all processors.
        random_state: Random seed for reproducibility.
        class_weight: How to weight classes. "balanced" adjusts weights
            inversely proportional to class frequencies.
    """

    n_estimators: int = 100
    max_depth: int | None = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    n_jobs: int = -1
    random_state: int = 42
    class_weight: str | None = "balanced"


def train_random_forest(
    train_dataset: CrashTensorDataset,
    config: RandomForestConfig | None = None,
    verbose: bool = True,
) -> RandomForestClassifier:
    """Train a Random Forest classifier on the provided data.

    Args:
        train_dataset: Training dataset (CrashTensorDataset).
        config: Training configuration. Uses defaults if None.
        verbose: Whether to print progress.

    Returns:
        Trained RandomForestClassifier.
    """
    if config is None:
        config = RandomForestConfig()

    # Convert tensors to numpy
    X_train = train_dataset.features.numpy()
    y_train = train_dataset.labels.numpy()

    if verbose:
        print(f"\nTraining Random Forest")
        print(f"  n_estimators: {config.n_estimators}")
        print(f"  max_depth: {config.max_depth}")
        print(f"  Training samples: {len(X_train):,}")

    # Create and train model
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        class_weight=config.class_weight,
        verbose=1 if verbose else 0,
    )

    if verbose and config.class_weight:
        print(f"  class_weight: {config.class_weight}")

    model.fit(X_train, y_train)

    if verbose:
        print(f"  Training complete!")

    return model


def predict_random_forest(
    model: RandomForestClassifier,
    dataset: CrashTensorDataset,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Get predictions from a Random Forest model.

    Args:
        model: Trained RandomForestClassifier.
        dataset: Dataset to predict on.

    Returns:
        Tuple of (predicted_labels, predicted_probabilities).
    """
    X = dataset.features.numpy()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return predictions, probabilities


def save_random_forest(model: RandomForestClassifier, path: Path | str) -> None:
    """Save the Random Forest model to disk.

    Args:
        model: Trained model to save.
        path: Path to save the model (.joblib file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_random_forest(path: Path | str) -> RandomForestClassifier:
    """Load a Random Forest model from disk.

    Args:
        path: Path to the saved model (.joblib file).

    Returns:
        Loaded model.
    """
    return joblib.load(path)
