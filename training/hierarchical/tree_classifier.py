"""Tree-based hierarchical classifier implementation.

Supports RandomForest, XGBoost, and ExtraTrees models with SMOTE resampling.
"""

from __future__ import annotations

import logging

import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from training.hierarchical.base import HierarchicalClassifierBase
from training.hierarchical.config import TreeHierarchicalConfig
from training.hierarchical.structure import HierarchicalTargets, HierarchyLevel

logger = logging.getLogger(__name__)


class HierarchicalTreeClassifier(HierarchicalClassifierBase):
    """Tree-based hierarchical classifier with SMOTE resampling.

    Uses separate tree ensemble models for each hierarchy level:
    - L1: INJURY vs NO_INJURY (all samples)
    - L2: SEVERE vs MINOR (injury samples only)
    - L2.5: FATAL vs INCAPACITATING (severe samples only)
    - L3: REPORTED vs VISIBLE (minor injury samples only)

    Each level applies BorderlineSMOTE for class balancing before training.
    """

    def __init__(self, config: TreeHierarchicalConfig | None = None):
        """Initialize tree-based hierarchical classifier.

        Args:
            config: Tree-specific configuration. Uses defaults if None.
        """
        if config is None:
            config = TreeHierarchicalConfig()
        super().__init__(config)
        self.config: TreeHierarchicalConfig = config

        # Level models (created during fit)
        self.l1_model: RandomForestClassifier | XGBClassifier | ExtraTreesClassifier | None = None
        self.l2_model: RandomForestClassifier | XGBClassifier | ExtraTreesClassifier | None = None
        self.l25_model: RandomForestClassifier | XGBClassifier | ExtraTreesClassifier | None = None
        self.l3_model: RandomForestClassifier | XGBClassifier | ExtraTreesClassifier | None = None

    def fit(
        self,
        X_train: np.ndarray,
        targets: HierarchicalTargets,
    ) -> "HierarchicalTreeClassifier":
        """Train all hierarchy level models.

        Args:
            X_train: Feature matrix.
            targets: HierarchicalTargets with binary labels for all levels.

        Returns:
            Self for method chaining.
        """
        # Level 1: INJURY vs NO_INJURY (all samples)
        self._fit_level(
            X_train,
            targets.y_injury,
            level=HierarchyLevel.L1_INJURY,
            model_type=self.config.l1_model,
            sampling_strategy=self.config.l1_sampling_strategy,
            weight_multiplier=1.0,
        )

        # Level 2: SEVERE vs MINOR (injury samples only)
        injury_mask = targets.y_injury == 1
        self._fit_level(
            X_train[injury_mask],
            targets.y_severe[injury_mask],
            level=HierarchyLevel.L2_SEVERITY,
            model_type=self.config.l2_model,
            sampling_strategy=self.config.l2_sampling_strategy,
            weight_multiplier=self.config.l2_weight_multiplier,
        )

        # Level 2.5: FATAL vs INCAPACITATING (severe samples only)
        severe_mask = (targets.y_injury == 1) & (targets.y_severe == 1)
        if np.sum(severe_mask) > 0:
            n_fatal = np.sum(targets.y_fatal[severe_mask] == 1)
            n_incap = np.sum(targets.y_fatal[severe_mask] == 0)
            if n_fatal > 0 and n_incap > 0:
                self._fit_level(
                    X_train[severe_mask],
                    targets.y_fatal[severe_mask],
                    level=HierarchyLevel.L25_FATAL,
                    model_type=self.config.l25_model,
                    sampling_strategy=self.config.l25_sampling_strategy,
                    weight_multiplier=self.config.l25_weight_multiplier,
                )
            else:
                logger.warning("Skipping L2.5 training - insufficient FATAL samples")

        # Level 3: REPORTED vs VISIBLE (minor injury samples only)
        minor_mask = (targets.y_injury == 1) & (targets.y_severe == 0)
        if np.sum(minor_mask) > 0:
            n_reported = np.sum(targets.y_reported[minor_mask] == 1)
            n_visible = np.sum(targets.y_reported[minor_mask] == 0)
            if n_reported > 0 and n_visible > 0:
                self._fit_level(
                    X_train[minor_mask],
                    targets.y_reported[minor_mask],
                    level=HierarchyLevel.L3_REPORTED,
                    model_type=self.config.l3_model,
                    sampling_strategy=self.config.l3_sampling_strategy,
                    weight_multiplier=1.0,
                )
            else:
                logger.warning("Skipping L3 training - insufficient class samples")

        return self

    def _fit_level(
        self,
        X: np.ndarray,
        y: np.ndarray,
        level: HierarchyLevel,
        model_type: str,
        sampling_strategy: float,
        weight_multiplier: float,
    ) -> None:
        """Fit a single hierarchy level model.

        Args:
            X: Feature matrix for this level.
            y: Binary labels for this level.
            level: Which hierarchy level.
            model_type: Model type ('rf', 'xgb', 'et').
            sampling_strategy: SMOTE sampling ratio.
            weight_multiplier: Extra weight for positive class.
        """
        logger.info("=" * 60)
        logger.info(f"{level.name}: Training classifier")
        logger.info("=" * 60)

        # Calculate class weights
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
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            n_jobs=self.config.n_jobs,
            scale_pos_weight=pos_weight * weight_multiplier,
        )

        logger.info(f"Training {model_type.upper()} model...")
        model.fit(X_resampled, y_resampled)

        # Store model
        if level == HierarchyLevel.L1_INJURY:
            self.l1_model = model
        elif level == HierarchyLevel.L2_SEVERITY:
            self.l2_model = model
        elif level == HierarchyLevel.L25_FATAL:
            self.l25_model = model
        elif level == HierarchyLevel.L3_REPORTED:
            self.l3_model = model

    def _apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sampling_strategy: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply BorderlineSMOTE with moderate oversampling.

        Args:
            X: Feature matrix.
            y: Binary labels.
            sampling_strategy: Ratio of minority to majority.

        Returns:
            Resampled (X, y).
        """
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
                random_state=self.config.random_state,
            )
            X_res, y_res = smote.fit_resample(X, y)

            unique_res, counts_res = np.unique(y_res, return_counts=True)
            logger.info(f"After SMOTE: {dict(zip(unique_res, counts_res))}")

            return X_res, y_res
        except ValueError as e:
            logger.warning(f"SMOTE failed: {e}. Using original data.")
            return X, y

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

        l1_proba = self.l1_model.predict_proba(X)[:, 1]
        l2_proba = self.l2_model.predict_proba(X)[:, 1] if self.l2_model else np.full(len(X), 0.5)
        l25_proba = self.l25_model.predict_proba(X)[:, 1] if self.l25_model else np.full(len(X), 0.5)
        l3_proba = self.l3_model.predict_proba(X)[:, 1] if self.l3_model else np.full(len(X), 0.5)

        return l1_proba, l2_proba, l25_proba, l3_proba


def _create_tree_model(
    model_type: str,
    n_estimators: int = 200,
    max_depth: int = 15,
    n_jobs: int = -1,
    scale_pos_weight: float = 1.0,
) -> RandomForestClassifier | XGBClassifier | ExtraTreesClassifier:
    """Create a tree ensemble classifier.

    Args:
        model_type: One of 'rf', 'xgb', 'et'.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        n_jobs: Parallel jobs (-1 for all cores).
        scale_pos_weight: Weight for positive class (XGBoost).

    Returns:
        Configured classifier.
    """
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=42,
        )
    elif model_type == "xgb":
        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            learning_rate=0.1,
            n_jobs=n_jobs,
            random_state=42,
            eval_metric="logloss",
        )
    elif model_type == "et":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
