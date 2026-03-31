"""Abstract base class for hierarchical classifiers.

Provides shared functionality:
- Threshold-based binary prediction
- Multiclass reconstruction from hierarchical predictions
- Threshold optimization on validation data
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelEncoder

from training.hierarchical.config import HierarchicalConfig
from training.hierarchical.structure import (
    HierarchicalTargets,
    map_hierarchical_to_multiclass,
)

logger = logging.getLogger(__name__)


class HierarchicalClassifierBase(ABC):
    """Abstract base class for hierarchical crash severity classifiers.

    Subclasses must implement:
    - fit(): Train models for all hierarchy levels
    - predict_proba(): Return probability predictions for all levels

    This base class provides:
    - predict_binary(): Apply thresholds to probabilities
    - predict_multiclass(): Map hierarchical predictions to 5 classes
    - optimize_thresholds(): Tune thresholds on validation data
    """

    def __init__(self, config: HierarchicalConfig):
        """Initialize classifier with configuration.

        Args:
            config: Hierarchical classification configuration.
        """
        self.config = config
        self.l1_threshold = config.l1_threshold
        self.l2_threshold = config.l2_threshold
        self.l25_threshold = config.l25_threshold
        self.l3_threshold = config.l3_threshold
        self.feature_cols: list[str] | None = None

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        targets: HierarchicalTargets,
    ) -> "HierarchicalClassifierBase":
        """Train all hierarchy level models.

        Args:
            X_train: Feature matrix.
            targets: HierarchicalTargets with binary labels for all levels.

        Returns:
            Self for method chaining.
        """
        pass

    @abstractmethod
    def predict_proba(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get probability predictions for all levels.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (l1_proba, l2_proba, l25_proba, l3_proba) where each
            is an array of positive class probabilities.
        """
        pass

    def predict_binary(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict binary labels for all levels using thresholds.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (l1_pred, l2_pred, l25_pred, l3_pred) binary arrays.
        """
        l1_proba, l2_proba, l25_proba, l3_proba = self.predict_proba(X)

        l1_pred = (l1_proba >= self.l1_threshold).astype(int)
        l2_pred = (l2_proba >= self.l2_threshold).astype(int)
        l25_pred = (l25_proba >= self.l25_threshold).astype(int)
        l3_pred = (l3_proba >= self.l3_threshold).astype(int)

        return l1_pred, l2_pred, l25_pred, l3_pred

    def predict_multiclass(
        self,
        X: np.ndarray,
        label_encoder: LabelEncoder | None = None,
    ) -> np.ndarray:
        """Predict original 5-class severity labels using hierarchy.

        Args:
            X: Feature matrix.
            label_encoder: Optional encoder for correct class indices.

        Returns:
            Array of multiclass predictions.
        """
        l1_pred, l2_pred, l25_pred, l3_pred = self.predict_binary(X)
        return map_hierarchical_to_multiclass(
            l1_pred, l2_pred, l25_pred, l3_pred, label_encoder
        )

    def optimize_thresholds(
        self,
        X_val: np.ndarray,
        targets: HierarchicalTargets,
        l1_target_recall: float = 0.7,
        l2_min_precision: float = 0.15,
        l25_target_recall: float = 0.4,
        l3_target_recall: float = 0.5,
        l3_min_threshold: float = 0.35,
    ) -> None:
        """Optimize classification thresholds on validation data.

        Args:
            X_val: Validation feature matrix.
            targets: HierarchicalTargets for validation set.
            l1_target_recall: Target recall for injury detection.
            l2_min_precision: Minimum precision for severity (F1 optimization).
            l25_target_recall: Target recall for FATAL detection.
            l3_target_recall: Target recall for reported detection.
            l3_min_threshold: Minimum threshold for L3 (balance REPORTED/NONINCAP).
        """
        logger.info("Optimizing classification thresholds...")

        l1_proba, l2_proba, l25_proba, l3_proba = self.predict_proba(X_val)

        # L1: Recall-based threshold
        self.l1_threshold = _find_optimal_threshold_recall(
            targets.y_injury, l1_proba, target_recall=l1_target_recall
        )
        logger.info(f"Optimized L1 threshold: {self.l1_threshold:.3f}")

        # L2: F1-based threshold with precision constraint (injury cases only)
        injury_mask = targets.y_injury == 1
        if np.sum(injury_mask) > 0:
            self.l2_threshold = _find_optimal_threshold_f1(
                targets.y_severe[injury_mask],
                l2_proba[injury_mask],
                min_precision=l2_min_precision,
            )
            logger.info(f"Optimized L2 threshold (F1-based): {self.l2_threshold:.3f}")

        # L2.5: Recall-based threshold (severe cases only)
        severe_mask = (targets.y_injury == 1) & (targets.y_severe == 1)
        if np.sum(severe_mask) > 0:
            self.l25_threshold = _find_optimal_threshold_recall(
                targets.y_fatal[severe_mask],
                l25_proba[severe_mask],
                target_recall=l25_target_recall,
            )
            logger.info(f"Optimized L2.5 threshold: {self.l25_threshold:.3f}")

        # L3: Recall-based with minimum threshold (minor injury cases)
        minor_mask = (targets.y_injury == 1) & (targets.y_severe == 0)
        if np.sum(minor_mask) > 0:
            optimal = _find_optimal_threshold_recall(
                targets.y_reported[minor_mask],
                l3_proba[minor_mask],
                target_recall=l3_target_recall,
            )
            self.l3_threshold = max(optimal, l3_min_threshold)
            logger.info(
                f"Optimized L3 threshold: {self.l3_threshold:.3f} "
                f"(optimal was {optimal:.3f})"
            )


def _find_optimal_threshold_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: float = 0.5,
) -> float:
    """Find threshold that achieves target recall for positive class.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for positive class.
        target_recall: Desired recall for positive class.

    Returns:
        Optimal threshold value.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # Find thresholds where recall >= target_recall
    valid_idx = np.where(recall >= target_recall)[0]
    if len(valid_idx) == 0:
        return float(thresholds[0]) if len(thresholds) > 0 else 0.5

    # Among valid thresholds, pick one with best F1
    best_f1 = 0.0
    best_threshold = 0.5

    for idx in valid_idx:
        if idx < len(thresholds):
            p = precision[idx]
            r = recall[idx]
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(thresholds[idx])

    return best_threshold


def _find_optimal_threshold_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.0,
) -> float:
    """Find threshold that maximizes F1 score with precision constraint.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for positive class.
        min_precision: Minimum precision constraint.

    Returns:
        Optimal threshold value.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    best_f1 = 0.0
    best_threshold = 0.5

    for i in range(len(thresholds)):
        p = precision[i]
        r = recall[i]

        if p < min_precision:
            continue

        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thresholds[i])

    return best_threshold
