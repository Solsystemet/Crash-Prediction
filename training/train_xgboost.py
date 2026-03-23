"""XGBoost training utilities.

XGBoost is a powerful gradient boosting implementation that often achieves
state-of-the-art results on tabular data. It handles class imbalance well
through scale_pos_weight and provides excellent feature importance analysis.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from data_preparation.tensor_dataset import CrashTensorDataset


# Try to import xgboost, provide helpful message if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def check_xgboost_available() -> None:
    """Check if XGBoost is available, raise helpful error if not."""
    if not XGBOOST_AVAILABLE:
        raise ImportError(
            "XGBoost is required for this module. "
            "Install it with: pip install xgboost"
        )


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost training.

    Attributes:
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage (eta).
        min_child_weight: Minimum sum of instance weight in a child.
        subsample: Fraction of samples used per tree.
        colsample_bytree: Fraction of features used per tree.
        gamma: Minimum loss reduction for split.
        reg_alpha: L1 regularization term.
        reg_lambda: L2 regularization term.
        scale_pos_weight: Balance of positive and negative weights.
            Set to ratio of negative to positive for imbalanced data.
        use_label_encoder: Deprecated, always False.
        eval_metric: Evaluation metric for multi-class.
        random_state: Random seed.
        n_jobs: Number of parallel jobs.
        early_stopping_rounds: Stop if no improvement for this many rounds.
    """

    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: float | None = None  # Auto-calculated if None
    eval_metric: str = "mlogloss"
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int | None = 10


def calculate_scale_pos_weight(y: NDArray) -> float:
    """Calculate scale_pos_weight for binary classification.

    For multi-class, returns 1.0 (use sample_weight instead).

    Args:
        y: Labels array.

    Returns:
        Ratio of negative to positive class.
    """
    unique = np.unique(y)
    if len(unique) == 2:
        # Binary: ratio of negative to positive
        neg_count = np.sum(y == unique[0])
        pos_count = np.sum(y == unique[1])
        return neg_count / pos_count
    else:
        # Multi-class: use sample weights instead
        return 1.0


def calculate_sample_weights(y: NDArray) -> NDArray:
    """Calculate sample weights for class balancing.

    Each sample gets a weight inversely proportional to its class frequency.

    Args:
        y: Labels array.

    Returns:
        Array of sample weights.
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    n_classes = len(unique)

    # Weight per class
    class_weights = {label: total / (n_classes * count) for label, count in zip(unique, counts)}

    # Sample weights
    sample_weights = np.array([class_weights[label] for label in y])

    return sample_weights


def train_xgboost(
    train_dataset: CrashTensorDataset,
    val_dataset: CrashTensorDataset | None = None,
    config: XGBoostConfig | None = None,
    verbose: bool = True,
) -> "xgb.XGBClassifier":
    """Train an XGBoost classifier on the provided data.

    Args:
        train_dataset: Training dataset (CrashTensorDataset).
        val_dataset: Validation dataset for early stopping. Optional.
        config: Training configuration. Uses defaults if None.
        verbose: Whether to print progress.

    Returns:
        Trained XGBClassifier.
    """
    check_xgboost_available()

    if config is None:
        config = XGBoostConfig()

    # Convert tensors to numpy
    X_train = train_dataset.features.numpy()
    y_train = train_dataset.labels.numpy()

    # Calculate sample weights for class balancing
    sample_weights = calculate_sample_weights(y_train)

    # Determine number of classes
    n_classes = len(np.unique(y_train))

    if verbose:
        print(f"\nTraining XGBoost")
        print(f"  n_estimators: {config.n_estimators}")
        print(f"  max_depth: {config.max_depth}")
        print(f"  learning_rate: {config.learning_rate}")
        print(f"  n_classes: {n_classes}")
        print(f"  Training samples: {len(X_train):,}")

    # Create model
    model = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        gamma=config.gamma,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        objective="multi:softmax" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric=config.eval_metric,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        verbosity=1 if verbose else 0,
    )

    # Prepare evaluation set for early stopping
    eval_set = None
    if val_dataset is not None and config.early_stopping_rounds:
        X_val = val_dataset.features.numpy()
        y_val = val_dataset.labels.numpy()
        eval_set = [(X_val, y_val)]
        if verbose:
            print(f"  Early stopping: {config.early_stopping_rounds} rounds")

    # Train
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=verbose,
    )

    if verbose:
        print(f"  Training complete!")

    return model


def predict_xgboost(
    model: "xgb.XGBClassifier",
    dataset: CrashTensorDataset,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Get predictions from an XGBoost model.

    Args:
        model: Trained XGBClassifier.
        dataset: Dataset to predict on.

    Returns:
        Tuple of (predicted_labels, predicted_probabilities).
    """
    check_xgboost_available()

    X = dataset.features.numpy()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return predictions.astype(np.int64), probabilities


def get_feature_importance(
    model: "xgb.XGBClassifier",
    feature_names: list[str] | None = None,
    importance_type: str = "gain",
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """Get feature importances from trained model.

    Args:
        model: Trained XGBClassifier.
        feature_names: Names of features. Uses indices if None.
        importance_type: Type of importance ("gain", "weight", "cover").
        top_n: Number of top features to return.

    Returns:
        List of (feature_name, importance) tuples, sorted by importance.
    """
    check_xgboost_available()

    # Get importance scores
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type=importance_type)

    # Map to feature names
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(model.n_features_in_)]

    # XGBoost uses f0, f1, ... as keys
    result = []
    for i, name in enumerate(feature_names):
        key = f"f{i}"
        importance = importance_dict.get(key, 0.0)
        result.append((name, importance))

    # Normalize and sort
    total = sum(imp for _, imp in result) or 1.0
    result = [(name, imp / total) for name, imp in result]
    result.sort(key=lambda x: x[1], reverse=True)

    return result[:top_n]


def print_feature_importance(
    model: "xgb.XGBClassifier",
    feature_names: list[str] | None = None,
    top_n: int = 15,
) -> None:
    """Print feature importances in a formatted way.

    Args:
        model: Trained model.
        feature_names: Names of features.
        top_n: Number of top features to show.
    """
    importances = get_feature_importance(model, feature_names, top_n=top_n)

    print(f"\nTop {top_n} Feature Importances (XGBoost):")
    print("-" * 50)
    for name, importance in importances:
        bar = "█" * int(importance * 50)
        print(f"  {name:30s} {importance:.4f} {bar}")


def save_xgboost(model: "xgb.XGBClassifier", path: Path | str) -> None:
    """Save the XGBoost model to disk.

    Args:
        model: Trained model to save.
        path: Path to save the model (.json or .ubj file).
    """
    check_xgboost_available()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use XGBoost's native format for smaller files
    model.save_model(str(path))


def load_xgboost(path: Path | str) -> "xgb.XGBClassifier":
    """Load an XGBoost model from disk.

    Args:
        path: Path to the saved model.

    Returns:
        Loaded model.
    """
    check_xgboost_available()

    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model
