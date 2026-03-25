"""LightGBM training utilities.

This module provides functions for training and evaluating LightGBM models.
"""

from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from imblearn.over_sampling import SMOTE
from numpy.typing import NDArray

from data_preparation.tensor_dataset import CrashTensorDataset


@dataclass
class LightGBMConfig:
    """Configuration for LightGBM training.

    Attributes:
        n_estimators: Number of boosting iterations.
        max_depth: Maximum depth of trees. -1 for no limit.
        learning_rate: Learning rate for boosting.
        num_leaves: Maximum number of leaves in one tree.
        min_child_samples: Minimum number of samples in one leaf.
        subsample: Subsample ratio of the training instance.
        colsample_bytree: Subsample ratio of columns when constructing each tree.
        reg_alpha: L1 regularization term on weights.
        reg_lambda: L2 regularization term on weights.
        n_jobs: Number of parallel jobs. -1 uses all processors.
        random_state: Random seed for reproducibility.
        class_weight: How to weight classes. "balanced" adjusts weights
            inversely proportional to class frequencies.
        verbose: Verbosity level for LightGBM. -1 to suppress output.
        device: Device to train on. "cpu" only (GPU requires special build).
        use_smote: Whether to apply SMOTE for handling class imbalance.
        smote_k_neighbors: Number of nearest neighbors for SMOTE.
        smote_sampling_strategy: SMOTE sampling strategy. "auto" balances all classes,
            or dict/float for custom ratios.
    """

    n_estimators: int = 500
    max_depth: int = -1
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    n_jobs: int = -1
    random_state: int = 42
    class_weight: str | None = "balanced"
    verbose: int = -1
    device: str = "cpu"
    use_smote: bool = False
    smote_k_neighbors: int = 5
    smote_sampling_strategy: str = "auto"


def train_lightgbm(
    train_dataset: CrashTensorDataset,
    val_dataset: CrashTensorDataset | None = None,
    config: LightGBMConfig | None = None,
    verbose: bool = True,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier on the provided data.

    Args:
        train_dataset: Training dataset (CrashTensorDataset).
        val_dataset: Optional validation dataset for early stopping.
        config: Training configuration. Uses defaults if None.
        verbose: Whether to print progress.

    Returns:
        Trained LGBMClassifier.
    """
    if config is None:
        config = LightGBMConfig()

    # Convert tensors to numpy
    X_train = train_dataset.features.numpy()
    y_train = train_dataset.labels.numpy()

    if verbose:
        print(f"\nTraining LightGBM")
        print(f"  device: {config.device}")
        print(f"  n_estimators: {config.n_estimators}")
        print(f"  max_depth: {config.max_depth}")
        print(f"  learning_rate: {config.learning_rate}")
        print(f"  num_leaves: {config.num_leaves}")
        print(f"  Training samples (before SMOTE): {len(X_train):,}")

    # Apply SMOTE if enabled
    if config.use_smote:
        if verbose:
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"  Class distribution before SMOTE:")
            for cls, count in zip(unique, counts):
                print(f"    Class {cls}: {count:,} samples")

        smote = SMOTE(
            sampling_strategy=config.smote_sampling_strategy,
            k_neighbors=config.smote_k_neighbors,
            random_state=config.random_state,
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)

        if verbose:
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"  Class distribution after SMOTE:")
            for cls, count in zip(unique, counts):
                print(f"    Class {cls}: {count:,} samples")
            print(f"  Training samples (after SMOTE): {len(X_train):,}")

    # Prepare validation data for early stopping if provided
    eval_set = None
    callbacks = None
    if val_dataset is not None:
        X_val = val_dataset.features.numpy()
        y_val = val_dataset.labels.numpy()
        eval_set = [(X_val, y_val)]  # type: ignore[assignment]
        if verbose:
            print(f"  Validation samples: {len(X_val):,}")
            callbacks = [lgb.log_evaluation(period=50)]  # type: ignore[list-item]

    # Create and train model
    model = lgb.LGBMClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        class_weight=config.class_weight,
        verbose=config.verbose,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        callbacks=callbacks,
    )

    if verbose:
        print(f"  Best iteration: {model.best_iteration_}")
        print("  Training complete!")

    return model


def save_lightgbm(model: lgb.LGBMClassifier, path: Path | str) -> None:
    """Save the LightGBM model to disk using joblib.

    Args:
        model: Trained LightGBM model to save.
        path: Path to save the model (.joblib file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_lightgbm(path: Path | str) -> lgb.LGBMClassifier:
    """Load a LightGBM model from disk.

    Args:
        path: Path to the saved model (.joblib file).

    Returns:
        Loaded LightGBM model.
    """
    path = Path(path)
    model: lgb.LGBMClassifier = joblib.load(path)
    return model
