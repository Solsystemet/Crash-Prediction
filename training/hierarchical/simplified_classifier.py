"""Simplified 2-level tree classifier for 3-class severity prediction.

Uses only L1 (injury) and L2 (severity) classifiers, eliminating the
problematic L2.5 and L3 levels from the full hierarchical approach.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import precision_recall_curve
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from training.hierarchical.simplified_targets import (
    SimplifiedTargets,
    map_simplified_predictions,
    SIMPLIFIED_CLASS_NAMES,
)
from training.hierarchical.tree_classifier import _create_tree_model

logger = logging.getLogger(__name__)


class SimplifiedClassifierBase(ABC):
    """Abstract base class for simplified 2-level classifiers.

    Subclasses must implement:
    - fit(): Train L1 and L2 models
    - predict_proba(): Return probability predictions for both levels
    """

    def __init__(
        self,
        l1_threshold: float = 0.3,
        l2_threshold: float = 0.3,
        random_state: int = 42,
    ):
        """Initialize simplified classifier.

        Args:
            l1_threshold: Probability threshold for injury detection.
            l2_threshold: Probability threshold for severity detection.
            random_state: Random seed for reproducibility.
        """
        self.l1_threshold = l1_threshold
        self.l2_threshold = l2_threshold
        self.random_state = random_state
        self.feature_cols: list[str] | None = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        targets: SimplifiedTargets,
    ) -> "SimplifiedClassifierBase":
        """Train L1 and L2 models."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get probability predictions for L1 and L2."""
        pass

    def predict_binary(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict binary labels for both levels using thresholds.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (l1_pred, l2_pred) binary arrays.
        """
        l1_proba, l2_proba = self.predict_proba(X)
        l1_pred = (l1_proba >= self.l1_threshold).astype(int)
        l2_pred = (l2_proba >= self.l2_threshold).astype(int)
        return l1_pred, l2_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict simplified 3-class labels.

        Args:
            X: Feature matrix.

        Returns:
            Array of 3-class predictions (0=NO_INJURY, 1=MINOR, 2=SEVERE).
        """
        l1_pred, l2_pred = self.predict_binary(X)
        return map_simplified_predictions(l1_pred, l2_pred)

    def optimize_thresholds(
        self,
        X_val: np.ndarray,
        targets: SimplifiedTargets,
        l1_target_recall: float = 0.7,
        l2_target_recall: float = 0.5,
    ) -> None:
        """Optimize thresholds on validation data.

        Args:
            X_val: Validation features.
            targets: Validation targets.
            l1_target_recall: Target recall for L1 (injury detection).
            l2_target_recall: Target recall for L2 (severity detection).
        """
        l1_proba, l2_proba = self.predict_proba(X_val)

        # Optimize L1 threshold
        self.l1_threshold = self._find_threshold_for_recall(
            targets.y_injury, l1_proba, l1_target_recall, "L1"
        )

        # Optimize L2 threshold (on injury cases only)
        injury_mask = targets.y_injury == 1
        if np.sum(injury_mask) > 0:
            self.l2_threshold = self._find_threshold_for_recall(
                targets.y_severe[injury_mask],
                l2_proba[injury_mask],
                l2_target_recall,
                "L2",
            )

        logger.info(f"Optimized thresholds: L1={self.l1_threshold:.3f}, L2={self.l2_threshold:.3f}")

    def _find_threshold_for_recall(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        target_recall: float,
        level_name: str,
    ) -> float:
        """Find threshold that achieves target recall."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        # recall and precision have len(thresholds) + 1 elements
        # Last element of recall is always 1.0 (at threshold 0)
        # We want the highest threshold that still achieves target recall
        recall_at_thresholds = recall[:-1]  # Align with thresholds

        # Find thresholds that achieve target recall
        valid_mask = recall_at_thresholds >= target_recall
        if not np.any(valid_mask):
            logger.warning(f"{level_name}: Cannot achieve {target_recall:.0%} recall, using minimum threshold")
            return float(thresholds[0]) if len(thresholds) > 0 else 0.5

        # Get highest threshold that still meets recall target
        valid_thresholds = thresholds[valid_mask]
        if len(valid_thresholds) > 0:
            return float(valid_thresholds[-1])
        return 0.5


class SimplifiedTreeClassifier(SimplifiedClassifierBase):
    """Tree-based simplified 2-level classifier with SMOTE resampling.

    Uses separate tree ensemble models for:
    - L1: INJURY vs NO_INJURY (all samples)
    - L2: SEVERE vs MINOR (injury samples only)
    """

    def __init__(
        self,
        l1_model: str = "lgbm",
        l2_model: str = "lgbm",
        l1_sampling_strategy: float = 0.5,
        l2_sampling_strategy: float = 0.7,
        l2_weight_multiplier: float = 1.5,
        n_estimators: int = 200,
        max_depth: int = 15,
        n_jobs: int = -1,
        l1_threshold: float = 0.3,
        l2_threshold: float = 0.3,
        random_state: int = 42,
    ):
        """Initialize simplified tree classifier.

        Args:
            l1_model: Model type for L1 ('rf', 'xgb', 'et', 'lgbm').
            l2_model: Model type for L2.
            l1_sampling_strategy: SMOTE ratio for L1.
            l2_sampling_strategy: SMOTE ratio for L2.
            l2_weight_multiplier: Extra weight for severe class.
            n_estimators: Number of trees in ensemble.
            max_depth: Maximum tree depth.
            n_jobs: Parallel jobs (-1 for all cores).
            l1_threshold: Initial L1 threshold.
            l2_threshold: Initial L2 threshold.
            random_state: Random seed.
        """
        super().__init__(l1_threshold, l2_threshold, random_state)

        self.l1_model_type = l1_model
        self.l2_model_type = l2_model
        self.l1_sampling_strategy = l1_sampling_strategy
        self.l2_sampling_strategy = l2_sampling_strategy
        self.l2_weight_multiplier = l2_weight_multiplier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs

        # Models (created during fit)
        self.l1_model: RandomForestClassifier | XGBClassifier | ExtraTreesClassifier | LGBMClassifier | None = None
        self.l2_model: RandomForestClassifier | XGBClassifier | ExtraTreesClassifier | LGBMClassifier | None = None

    def fit(
        self,
        X_train: np.ndarray,
        targets: SimplifiedTargets,
    ) -> "SimplifiedTreeClassifier":
        """Train L1 and L2 models.

        Args:
            X_train: Feature matrix.
            targets: SimplifiedTargets with binary labels.

        Returns:
            Self for method chaining.
        """
        # Level 1: INJURY vs NO_INJURY (all samples)
        n_injury_classes = len(np.unique(targets.y_injury))
        
        if n_injury_classes < 2:
            # Only one class present - skip L1 training
            logger.info("=" * 60)
            logger.info(f"L1 (INJURY): Skipping - only {n_injury_classes} class present")
            logger.info("=" * 60)
            self.l1_model = None
            self._l1_default = int(np.mean(targets.y_injury) >= 0.5)
        else:
            logger.info("=" * 60)
            logger.info("L1 (INJURY): Training classifier")
            logger.info("=" * 60)
            self.l1_model = self._fit_level(
                X_train,
                targets.y_injury,
                model_type=self.l1_model_type,
                sampling_strategy=self.l1_sampling_strategy,
                weight_multiplier=1.0,
            )

        # Level 2: SEVERE vs MINOR (injury samples only)
        injury_mask = targets.y_injury == 1
        y_severe_subset = targets.y_severe[injury_mask]
        n_severe_classes = len(np.unique(y_severe_subset))
        
        if n_severe_classes < 2:
            # Only one class present (no SEVERE or no MINOR) - skip L2
            logger.info("=" * 60)
            logger.info(f"L2 (SEVERE): Skipping - only {n_severe_classes} class present")
            logger.info("=" * 60)
            self.l2_model = None
            # Store dominant class for fallback predictions
            self._l2_default = int(np.mean(y_severe_subset) >= 0.5) if len(y_severe_subset) > 0 else 0
        else:
            logger.info("=" * 60)
            logger.info("L2 (SEVERE): Training classifier")
            logger.info("=" * 60)
            self.l2_model = self._fit_level(
                X_train[injury_mask],
                y_severe_subset,
                model_type=self.l2_model_type,
                sampling_strategy=self.l2_sampling_strategy,
                weight_multiplier=self.l2_weight_multiplier,
            )

        return self

    def _fit_level(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        sampling_strategy: float,
        weight_multiplier: float,
    ) -> RandomForestClassifier | XGBClassifier | ExtraTreesClassifier | LGBMClassifier:
        """Fit a single level model with SMOTE."""
        # Class distribution
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        logger.info(f"Class distribution: negative={n_neg}, positive={n_pos}")
        logger.info(f"Positive class weight: {pos_weight:.2f}")

        # Apply SMOTE
        X_resampled, y_resampled = self._apply_smote(X, y, sampling_strategy)

        # Create and train model
        model = _create_tree_model(
            model_type=model_type,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            scale_pos_weight=pos_weight * weight_multiplier,
        )

        logger.info(f"Training {model_type.upper()} model...")
        model.fit(X_resampled, y_resampled)

        return model

    def _apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply BorderlineSMOTE with moderate oversampling."""
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Before SMOTE: {class_dist}")

        minority_class = min(class_dist, key=class_dist.get)
        majority_class = max(class_dist, key=class_dist.get)
        minority_count = class_dist[minority_class]
        majority_count = class_dist[majority_class]

        target_minority = int(majority_count * sampling_strategy)

        if target_minority <= minority_count:
            logger.info("Minority class already at target ratio, skipping SMOTE")
            return X, y

        try:
            smote = BorderlineSMOTE(
                sampling_strategy={minority_class: target_minority},
                k_neighbors=min(5, minority_count - 1),
                random_state=self.random_state,
            )
            X_res, y_res = smote.fit_resample(X, y)

            unique_res, counts_res = np.unique(y_res, return_counts=True)
            logger.info(f"After SMOTE: {dict(zip(unique_res, counts_res))}")

            return X_res, y_res
        except ValueError as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get probability predictions for L1 and L2.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (l1_proba, l2_proba) probability arrays.
        """
        # L1 predictions
        if self.l1_model is not None:
            l1_proba = self.l1_model.predict_proba(X)[:, 1]
        else:
            # No L1 model - use default prediction based on training data
            default_val = getattr(self, '_l1_default', 0)
            l1_proba = np.full(len(X), float(default_val))
        
        # L2 predictions
        if self.l2_model is not None:
            l2_proba = self.l2_model.predict_proba(X)[:, 1]
        else:
            # No L2 model - use default prediction based on training data
            default_val = getattr(self, '_l2_default', 0)
            l2_proba = np.full(len(X), float(default_val))

        return l1_proba, l2_proba


def save_simplified_model(clf: SimplifiedTreeClassifier, path: str) -> None:
    """Save a trained SimplifiedTreeClassifier to disk.

    Saves the L1 and L2 models, thresholds, and feature columns.

    Args:
        clf: Trained SimplifiedTreeClassifier.
        path: Path to save the model (will create a directory).
    """
    import joblib
    from pathlib import Path

    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    if clf.l1_model is not None:
        joblib.dump(clf.l1_model, save_dir / "l1_model.joblib")
    if clf.l2_model is not None:
        joblib.dump(clf.l2_model, save_dir / "l2_model.joblib")

    # Save metadata
    metadata = {
        "l1_threshold": clf.l1_threshold,
        "l2_threshold": clf.l2_threshold,
        "l1_model_type": clf.l1_model_type,
        "l2_model_type": clf.l2_model_type,
        "feature_cols": clf.feature_cols,
        "random_state": clf.random_state,
    }
    joblib.dump(metadata, save_dir / "metadata.joblib")

    logger.info(f"Saved simplified model to {save_dir}")


def load_simplified_model(path: str) -> SimplifiedTreeClassifier:
    """Load a trained SimplifiedTreeClassifier from disk.

    Args:
        path: Path to the saved model directory.

    Returns:
        Loaded SimplifiedTreeClassifier ready for inference.
    """
    import joblib
    from pathlib import Path

    load_dir = Path(path)

    # Load metadata
    metadata = joblib.load(load_dir / "metadata.joblib")

    # Create classifier with saved settings
    clf = SimplifiedTreeClassifier(
        l1_model=metadata["l1_model_type"],
        l2_model=metadata["l2_model_type"],
        l1_threshold=metadata["l1_threshold"],
        l2_threshold=metadata["l2_threshold"],
        random_state=metadata["random_state"],
    )
    clf.feature_cols = metadata["feature_cols"]

    # Load models
    clf.l1_model = joblib.load(load_dir / "l1_model.joblib")
    if (load_dir / "l2_model.joblib").exists():
        clf.l2_model = joblib.load(load_dir / "l2_model.joblib")

    logger.info(f"Loaded simplified model from {load_dir}")
    return clf
