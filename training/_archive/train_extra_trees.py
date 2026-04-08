"""Extra Trees training utilities.

Research shows Extra Trees (Extremely Randomized Trees) often outperforms
Random Forest and other ensemble methods for crash severity prediction
due to its ability to handle large feature spaces and capture complex interactions.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier

from data_preparation.tensor_dataset import CrashTensorDataset


@dataclass
class ExtraTreesConfig:
    """Configuration for Extra Trees training.

    Attributes:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees. None for unlimited.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf node.
        max_features: Number of features to consider for best split.
            "sqrt" is common for classification.
        n_jobs: Number of parallel jobs. -1 uses all processors.
        random_state: Random seed for reproducibility.
        class_weight: How to weight classes. "balanced" adjusts weights
            inversely proportional to class frequencies.
        bootstrap: Whether to use bootstrap samples (False for classic ET).
    """

    n_estimators: int = 200
    max_depth: int | None = 20
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str | int | float = "sqrt"
    n_jobs: int = -1
    random_state: int = 42
    class_weight: str | dict | None = "balanced"
    bootstrap: bool = False  # Classic Extra Trees uses all samples


def train_extra_trees(
    train_dataset: CrashTensorDataset,
    config: ExtraTreesConfig | None = None,
    verbose: bool = True,
) -> ExtraTreesClassifier:
    """Train an Extra Trees classifier on the provided data.

    Args:
        train_dataset: Training dataset (CrashTensorDataset).
        config: Training configuration. Uses defaults if None.
        verbose: Whether to print progress.

    Returns:
        Trained ExtraTreesClassifier.
    """
    if config is None:
        config = ExtraTreesConfig()

    # Convert tensors to numpy
    X_train = train_dataset.features.numpy()
    y_train = train_dataset.labels.numpy()

    if verbose:
        print(f"\nTraining Extra Trees")
        print(f"  n_estimators: {config.n_estimators}")
        print(f"  max_depth: {config.max_depth}")
        print(f"  max_features: {config.max_features}")
        print(f"  bootstrap: {config.bootstrap}")
        print(f"  Training samples: {len(X_train):,}")

    # Create and train model
    model = ExtraTreesClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        class_weight=config.class_weight,
        bootstrap=config.bootstrap,
        verbose=1 if verbose else 0,
    )

    if verbose and config.class_weight:
        print(f"  class_weight: {config.class_weight}")

    model.fit(X_train, y_train)

    if verbose:
        print(f"  Training complete!")

    return model


def predict_extra_trees(
    model: ExtraTreesClassifier,
    dataset: CrashTensorDataset,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Get predictions from an Extra Trees model.

    Args:
        model: Trained ExtraTreesClassifier.
        dataset: Dataset to predict on.

    Returns:
        Tuple of (predicted_labels, predicted_probabilities).
    """
    X = dataset.features.numpy()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return predictions, probabilities


def get_feature_importance(
    model: ExtraTreesClassifier,
    feature_names: list[str] | None = None,
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """Get feature importances from trained model.

    Args:
        model: Trained ExtraTreesClassifier.
        feature_names: Names of features. Uses indices if None.
        top_n: Number of top features to return.

    Returns:
        List of (feature_name, importance) tuples, sorted by importance.
    """
    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # Sort by importance
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance[:top_n]


def print_feature_importance(
    model: ExtraTreesClassifier,
    feature_names: list[str] | None = None,
    top_n: int = 15,
) -> None:
    """Print feature importances in a formatted way.

    Args:
        model: Trained model.
        feature_names: Names of features.
        top_n: Number of top features to show.
    """
    importances = get_feature_importance(model, feature_names, top_n)

    print(f"\nTop {top_n} Feature Importances:")
    print("-" * 50)
    for name, importance in importances:
        bar = "█" * int(importance * 50)
        print(f"  {name:30s} {importance:.4f} {bar}")


def save_extra_trees(model: ExtraTreesClassifier, path: Path | str) -> None:
    """Save the Extra Trees model to disk.

    Args:
        model: Trained model to save.
        path: Path to save the model (.joblib file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_extra_trees(path: Path | str) -> ExtraTreesClassifier:
    """Load an Extra Trees model from disk.

    Args:
        path: Path to the saved model (.joblib file).

    Returns:
        Loaded model.
    """
    return joblib.load(path)
