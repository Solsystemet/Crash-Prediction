"""Hierarchical classification pipeline for crash severity prediction.

This module implements a two-stage classification approach:
- Level 1: INJURY vs NO_INJURY (14% vs 86% - more balanced than 5-class)
- Level 2: SEVERE vs MINOR (for injury cases only - better minority detection)

This approach addresses the extreme class imbalance that caused the original
pipeline to achieve only 4.3% FATAL recall despite 80% overall accuracy.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.feature_engineering import (
    add_binary_targets,
    add_ordinal_severity,
    engineer_all_features,
)
from data_preparation.triple_merge import triple_merge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical classification."""

    # Data parameters
    sample_size: int | None = None  # None = use all data
    test_size: float = 0.2
    random_state: int = 42

    # Level 1: Injury vs No Injury
    l1_sampling_strategy: float = 0.5  # Moderate oversampling (50% of majority)
    l1_model: Literal["rf", "xgb", "et"] = "xgb"
    l1_threshold: float = 0.3  # Lower threshold to catch more injuries

    # Level 2: Severe vs Minor
    l2_sampling_strategy: float = 0.7  # Higher oversampling for severity
    l2_model: Literal["rf", "xgb", "et"] = "xgb"
    l2_threshold: float = 0.2  # Lower threshold to catch more severe cases
    l2_weight_multiplier: float = 1.5  # Amplify class weight to reduce false positives

    # Level 2.5: Fatal vs Incapacitating (among severe cases)
    l25_sampling_strategy: float = 0.8  # High oversampling for rare FATAL class
    l25_model: Literal["rf", "xgb", "et"] = "xgb"
    l25_threshold: float = 0.15  # Low threshold to catch more FATAL cases
    l25_weight_multiplier: float = 3.0  # Extra weight for extremely rare FATAL

    # Level 3: Visible (NONINCAPACITATING) vs Reported (among minor injuries)
    l3_sampling_strategy: float = 0.5  # Moderate oversampling
    l3_model: Literal["rf", "xgb", "et"] = "xgb"
    l3_threshold: float = 0.5  # Default threshold

    # Model hyperparameters
    n_estimators: int = 200
    max_depth: int = 15
    n_jobs: int = -1


def create_model(
    model_type: str,
    n_estimators: int = 200,
    max_depth: int = 15,
    n_jobs: int = -1,
    scale_pos_weight: float = 1.0,
) -> RandomForestClassifier | XGBClassifier | ExtraTreesClassifier:
    """Create a classifier model.

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


def apply_moderate_smote(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = 0.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE with moderate oversampling.

    Instead of fully balancing classes (which created 619K synthetic from 766 real),
    this uses a ratio-based strategy that limits synthetic sample generation.

    Args:
        X: Feature matrix.
        y: Target labels.
        sampling_strategy: Ratio of minority to majority (0.5 = 50% of majority size).
        random_state: Random seed.

    Returns:
        Resampled (X, y).
    """
    # Calculate current class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    logger.info(f"Before SMOTE: {class_dist}")

    # Only oversample if minority is significantly smaller
    minority_class = min(class_dist, key=class_dist.get)
    majority_class = max(class_dist, key=class_dist.get)

    minority_count = class_dist[minority_class]
    majority_count = class_dist[majority_class]

    target_minority = int(majority_count * sampling_strategy)

    # Only apply SMOTE if target is larger than current minority count
    if target_minority <= minority_count:
        logger.info("Minority class already at target ratio, skipping SMOTE")
        return X, y

    try:
        smote = BorderlineSMOTE(
            sampling_strategy={minority_class: target_minority},
            k_neighbors=min(5, minority_count - 1),
            random_state=random_state,
        )
        X_res, y_res = smote.fit_resample(X, y)

        unique_res, counts_res = np.unique(y_res, return_counts=True)
        logger.info(f"After SMOTE: {dict(zip(unique_res, counts_res))}")

        return X_res, y_res
    except ValueError as e:
        logger.warning(f"SMOTE failed: {e}. Using original data.")
        return X, y


def find_optimal_threshold(
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

    # Find threshold where recall >= target_recall
    valid_idx = np.where(recall >= target_recall)[0]
    if len(valid_idx) == 0:
        # Can't achieve target recall, use lowest threshold
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


def find_optimal_threshold_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.0,
) -> float:
    """Find threshold that maximizes F1 score with optional precision constraint.

    This is an alternative to recall-based thresholding that balances
    precision and recall more evenly.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for positive class.
        min_precision: Minimum precision constraint (default 0 = no constraint).

    Returns:
        Optimal threshold value that maximizes F1.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    best_f1 = 0.0
    best_threshold = 0.5

    for i in range(len(thresholds)):
        p = precision[i]
        r = recall[i]

        # Skip if precision is below minimum constraint
        if p < min_precision:
            continue

        if p + r > 0:
            f1 = 2 * p * r / (p + r)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(thresholds[i])

    return best_threshold


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare feature matrix from DataFrame.

    Args:
        df: DataFrame with engineered features.

    Returns:
        Tuple of (feature DataFrame, feature column names).
    """
    # Define target and metadata columns to exclude
    # Also exclude outcome-correlated features (data leakage)
    exclude_cols = {
        # Identifiers and timestamps
        "CRASH_RECORD_ID",
        "CRASH_DATE",
        "DATE_POLICE_NOTIFIED",
        "CRASH_DATE_EST_I",
        # Direct injury indicators (targets)
        "MOST_SEVERE_INJURY",
        "INJURIES_TOTAL",
        "INJURIES_FATAL",
        "INJURIES_INCAPACITATING",
        "INJURIES_NON_INCAPACITATING",
        "INJURIES_REPORTED_NOT_EVIDENT",
        "INJURIES_NO_INDICATION",
        "INJURIES_UNKNOWN",
        # Binary targets
        "IS_INJURY",
        "IS_SEVERE",
        "IS_FATAL",  # L2.5 binary target
        "IS_REPORTED",  # L3 binary target
        "SEVERITY_LEVEL",
        "SEVERITY_ENCODED",  # Label-encoded MOST_SEVERE_INJURY
        # Geographic coordinates (use clusters instead)
        "LATITUDE",
        "LONGITUDE",
        "LOCATION",
        # Post-crash investigation fields (data leakage!)
        "STATEMENTS_TAKEN_I",
        "WORK_ZONE_I",
        "PHOTOS_TAKEN_I",
        "DOORING_I",
        # People-derived outcome features (data leakage!)
        # These are known only AFTER the crash/injury assessment
        "ejected_any",  # Ejection status correlates perfectly with injury
        "using_seatbelt_mean",  # Recorded during injury investigation
        "cell_phone_any",  # Often determined post-crash
        "bac_positive_any",  # BAC test results often post-crash
        # CRITICAL: CRASH_TYPE directly encodes injury!
        # Values: "NO INJURY / DRIVE AWAY", "INJURY AND / OR TOW DUE TO CRASH"
        "CRASH_TYPE",
        # REPORT_TYPE may correlate with injury severity
        "REPORT_TYPE",
        # Redundant temporal
        "HOUR",
        # High-cardinality location features (overfit risk)
        "STREET_NAME",
    }

    # Safe categorical features that are known BEFORE crash outcome
    # These describe pre-crash conditions, not post-crash investigation results
    safe_categorical_cols = [
        "WEATHER_CONDITION",      # Weather at time of crash
        "LIGHTING_CONDITION",     # Lighting at time of crash  
        "FIRST_CRASH_TYPE",       # Type of collision (PEDESTRIAN, REAR END, etc.)
        "TRAFFICWAY_TYPE",        # Road type
        "ROADWAY_SURFACE_COND",   # Road surface condition
        "TRAFFIC_CONTROL_DEVICE", # Traffic signals, signs
        "DEVICE_CONDITION",       # Condition of traffic device
        "ALIGNMENT",              # Road alignment (straight, curve)
        "ROAD_DEFECT",            # Road defects
        "PRIM_CONTRIBUTORY_CAUSE", # Primary cause (driver behavior)
        "DAMAGE",                 # Property damage level
    ]

    # Get numeric columns
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            feature_cols.append(col)

    # Create copy with numeric features
    df_features = df[feature_cols].copy()

    # Add label-encoded safe categorical features
    for col in safe_categorical_cols:
        if col in df.columns and col not in exclude_cols:
            le = LabelEncoder()
            values = df[col].fillna("UNKNOWN").astype(str)
            df_features[col] = le.fit_transform(values)
            feature_cols.append(col)

    feature_cols = list(df_features.columns)

    # Fill any remaining NaN
    df_features = df_features.fillna(0)

    return df_features, feature_cols


class HierarchicalClassifier:
    """Four-stage hierarchical classifier for crash severity.

    Level 1: Predicts INJURY vs NO_INJURY
    Level 2: For injury cases, predicts SEVERE vs NON_SEVERE
    Level 2.5: For severe cases, predicts FATAL vs INCAPACITATING
    Level 3: For minor injury cases, predicts VISIBLE (NONINCAPACITATING) vs REPORTED

    Final mapping:
    - NO_INJURY predicted at L1 → NO INDICATION OF INJURY
    - INJURY at L1, SEVERE at L2, FATAL at L2.5 → FATAL
    - INJURY at L1, SEVERE at L2, not FATAL at L2.5 → INCAPACITATING INJURY
    - INJURY at L1, MINOR at L2, VISIBLE at L3 → NONINCAPACITATING INJURY
    - INJURY at L1, MINOR at L2, REPORTED at L3 → REPORTED, NOT EVIDENT
    """

    def __init__(self, config: HierarchicalConfig):
        """Initialize hierarchical classifier.

        Args:
            config: Configuration parameters.
        """
        self.config = config
        self.l1_model = None
        self.l2_model = None
        self.l25_model = None  # L2.5: FATAL vs INCAPACITATING
        self.l3_model = None
        self.l1_threshold = config.l1_threshold
        self.l2_threshold = config.l2_threshold
        self.l25_threshold = config.l25_threshold
        self.l3_threshold = config.l3_threshold
        self.feature_cols = None

    def fit(
        self,
        X_train: np.ndarray,
        y_injury: np.ndarray,
        y_severe: np.ndarray,
        y_reported: np.ndarray | None = None,
        y_fatal: np.ndarray | None = None,
    ) -> "HierarchicalClassifier":
        """Fit all levels of the hierarchical classifier.

        Args:
            X_train: Feature matrix.
            y_injury: Binary labels (1=injury, 0=no injury).
            y_severe: Binary labels (1=severe, 0=not severe).
            y_reported: Binary labels (1=reported, 0=nonincapacitating).
            y_fatal: Binary labels (1=fatal, 0=incapacitating).

        Returns:
            Self.
        """
        # Level 1: Train on all data for injury detection
        logger.info("=" * 60)
        logger.info("LEVEL 1: Training INJURY vs NO_INJURY classifier")
        logger.info("=" * 60)

        # Calculate class weight for L1
        n_no_injury = np.sum(y_injury == 0)
        n_injury = np.sum(y_injury == 1)
        l1_weight = n_no_injury / n_injury if n_injury > 0 else 1.0
        logger.info(f"L1 Class distribution: NO_INJURY={n_no_injury}, INJURY={n_injury}")
        logger.info(f"L1 Positive class weight: {l1_weight:.2f}")

        # Apply moderate SMOTE
        X_l1, y_l1 = apply_moderate_smote(
            X_train,
            y_injury,
            sampling_strategy=self.config.l1_sampling_strategy,
            random_state=self.config.random_state,
        )

        self.l1_model = create_model(
            self.config.l1_model,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            n_jobs=self.config.n_jobs,
            scale_pos_weight=l1_weight,
        )

        logger.info(f"Training L1 model ({self.config.l1_model})...")
        self.l1_model.fit(X_l1, y_l1)

        # Level 2: Train only on injury cases for severity detection
        logger.info("=" * 60)
        logger.info("LEVEL 2: Training SEVERE vs NON-SEVERE classifier")
        logger.info("=" * 60)

        injury_mask = y_injury == 1
        X_injury = X_train[injury_mask]
        y_severe_injury = y_severe[injury_mask]

        n_minor = np.sum(y_severe_injury == 0)
        n_severe = np.sum(y_severe_injury == 1)
        l2_weight = n_minor / n_severe if n_severe > 0 else 1.0
        logger.info(f"L2 Class distribution: MINOR={n_minor}, SEVERE={n_severe}")
        logger.info(f"L2 Positive class weight: {l2_weight:.2f}")

        # Apply SMOTE for severity
        X_l2, y_l2 = apply_moderate_smote(
            X_injury,
            y_severe_injury,
            sampling_strategy=self.config.l2_sampling_strategy,
            random_state=self.config.random_state,
        )

        self.l2_model = create_model(
            self.config.l2_model,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            n_jobs=self.config.n_jobs,
            scale_pos_weight=l2_weight * self.config.l2_weight_multiplier,
        )

        logger.info(f"Training L2 model ({self.config.l2_model})...")
        self.l2_model.fit(X_l2, y_l2)

        # Level 2.5: Train only on severe cases for FATAL vs INCAPACITATING
        if y_fatal is not None:
            logger.info("=" * 60)
            logger.info("LEVEL 2.5: Training FATAL vs INCAPACITATING classifier")
            logger.info("=" * 60)

            # Severe injuries: injury=1 AND severe=1
            severe_mask = (y_injury == 1) & (y_severe == 1)
            X_severe = X_train[severe_mask]
            y_fatal_severe = y_fatal[severe_mask]

            n_incap = np.sum(y_fatal_severe == 0)
            n_fatal = np.sum(y_fatal_severe == 1)
            l25_weight = n_incap / n_fatal if n_fatal > 0 else 1.0
            logger.info(f"L2.5 Class distribution: INCAPACITATING={n_incap}, FATAL={n_fatal}")
            logger.info(f"L2.5 Positive class weight: {l25_weight:.2f}")

            if n_fatal > 0 and n_incap > 0:
                # Apply SMOTE for FATAL detection
                X_l25, y_l25 = apply_moderate_smote(
                    X_severe,
                    y_fatal_severe,
                    sampling_strategy=self.config.l25_sampling_strategy,
                    random_state=self.config.random_state,
                )

                self.l25_model = create_model(
                    self.config.l25_model,
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    n_jobs=self.config.n_jobs,
                    scale_pos_weight=l25_weight * self.config.l25_weight_multiplier,
                )

                logger.info(f"Training L2.5 model ({self.config.l25_model})...")
                self.l25_model.fit(X_l25, y_l25)
            else:
                logger.warning("Skipping L2.5 training - insufficient FATAL samples")

        # Level 3: Train only on minor injury cases for reported vs visible
        if y_reported is not None:
            logger.info("=" * 60)
            logger.info("LEVEL 3: Training REPORTED vs VISIBLE classifier")
            logger.info("=" * 60)

            # Minor injuries: injury=1 AND severe=0
            minor_mask = (y_injury == 1) & (y_severe == 0)
            X_minor = X_train[minor_mask]
            y_reported_minor = y_reported[minor_mask]

            # Now IS_REPORTED=1 means REPORTED, IS_REPORTED=0 means VISIBLE
            n_reported = np.sum(y_reported_minor == 1)
            n_visible = np.sum(y_reported_minor == 0)
            l3_weight = n_visible / n_reported if n_reported > 0 else 1.0
            logger.info(f"L3 Class distribution: VISIBLE={n_visible}, REPORTED={n_reported}")
            logger.info(f"L3 Positive class weight: {l3_weight:.2f}")

            if n_reported > 0 and n_visible > 0:
                # Apply SMOTE
                X_l3, y_l3 = apply_moderate_smote(
                    X_minor,
                    y_reported_minor,
                    sampling_strategy=self.config.l3_sampling_strategy,
                    random_state=self.config.random_state,
                )

                self.l3_model = create_model(
                    self.config.l3_model,
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    n_jobs=self.config.n_jobs,
                    scale_pos_weight=l3_weight,
                )

                logger.info(f"Training L3 model ({self.config.l3_model})...")
                self.l3_model.fit(X_l3, y_l3)
            else:
                logger.warning("Skipping L3 training - insufficient class samples")

        return self

    def optimize_thresholds(
        self,
        X_val: np.ndarray,
        y_injury: np.ndarray,
        y_severe: np.ndarray,
        y_reported: np.ndarray | None = None,
        y_fatal: np.ndarray | None = None,
        l1_target_recall: float = 0.7,
        l2_target_recall: float = 0.5,
        l25_target_recall: float = 0.4,
        l3_target_recall: float = 0.5,
    ) -> None:
        """Optimize classification thresholds on validation data.

        Args:
            X_val: Validation feature matrix.
            y_injury: Binary injury labels.
            y_severe: Binary severity labels.
            y_reported: Binary reported/nonincapacitating labels.
            y_fatal: Binary fatal/incapacitating labels.
            l1_target_recall: Target recall for injury detection.
            l2_target_recall: Target recall for severity detection.
            l25_target_recall: Target recall for FATAL detection.
            l3_target_recall: Target recall for reported detection.
        """
        logger.info("Optimizing classification thresholds...")

        # L1 threshold optimization
        l1_proba = self.l1_model.predict_proba(X_val)[:, 1]
        self.l1_threshold = find_optimal_threshold(
            y_injury, l1_proba, target_recall=l1_target_recall
        )
        logger.info(f"Optimized L1 threshold: {self.l1_threshold:.3f}")

        # L2 threshold optimization (on injury cases)
        # Use F1-based optimization with minimum precision constraint
        injury_mask = y_injury == 1
        if np.sum(injury_mask) > 0:
            X_val_injury = X_val[injury_mask]
            y_val_severe = y_severe[injury_mask]
            l2_proba = self.l2_model.predict_proba(X_val_injury)[:, 1]
            # Use F1 optimization with precision constraint to reduce false positives
            self.l2_threshold = find_optimal_threshold_f1(
                y_val_severe, l2_proba, min_precision=0.15
            )
            logger.info(f"Optimized L2 threshold (F1-based): {self.l2_threshold:.3f}")

        # L2.5 threshold optimization (on severe cases)
        if self.l25_model is not None and y_fatal is not None:
            severe_mask = (y_injury == 1) & (y_severe == 1)
            if np.sum(severe_mask) > 0:
                X_val_severe = X_val[severe_mask]
                y_val_fatal = y_fatal[severe_mask]
                l25_proba = self.l25_model.predict_proba(X_val_severe)[:, 1]
                self.l25_threshold = find_optimal_threshold(
                    y_val_fatal, l25_proba, target_recall=l25_target_recall
                )
                logger.info(f"Optimized L2.5 threshold: {self.l25_threshold:.3f}")

        # L3 threshold optimization (on minor injury cases)
        # Use minimum threshold of 0.35 to balance REPORTED vs NONINCAP
        if self.l3_model is not None and y_reported is not None:
            minor_mask = (y_injury == 1) & (y_severe == 0)
            if np.sum(minor_mask) > 0:
                X_val_minor = X_val[minor_mask]
                y_val_reported = y_reported[minor_mask]
                l3_proba = self.l3_model.predict_proba(X_val_minor)[:, 1]
                optimal_threshold = find_optimal_threshold(
                    y_val_reported, l3_proba, target_recall=l3_target_recall
                )
                # Apply minimum threshold to balance REPORTED vs NONINCAP
                self.l3_threshold = max(optimal_threshold, 0.35)
                logger.info(f"Optimized L3 threshold: {self.l3_threshold:.3f} (optimal was {optimal_threshold:.3f})")

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get probability predictions for all levels.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (L1 probabilities, L2 probabilities, L2.5 probabilities, L3 probabilities).
        """
        l1_proba = self.l1_model.predict_proba(X)[:, 1]

        # L2 prediction for all samples (will only use for predicted injuries)
        l2_proba = self.l2_model.predict_proba(X)[:, 1]

        # L2.5 prediction (if model exists)
        if self.l25_model is not None:
            l25_proba = self.l25_model.predict_proba(X)[:, 1]
        else:
            l25_proba = np.full(len(X), 0.5)  # Default to 50% if no L2.5 model

        # L3 prediction (if model exists)
        if self.l3_model is not None:
            l3_proba = self.l3_model.predict_proba(X)[:, 1]
        else:
            l3_proba = np.full(len(X), 0.5)  # Default to 50% if no L3 model

        return l1_proba, l2_proba, l25_proba, l3_proba

    def predict_binary(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict binary labels for all levels.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (L1 predictions, L2 predictions, L2.5 predictions, L3 predictions).
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
        """Predict original 5-class severity labels using 4-level hierarchy.

        LabelEncoder sorts alphabetically, so typical indices are:
        - 0: FATAL
        - 1: INCAPACITATING INJURY
        - 2: NO INDICATION OF INJURY
        - 3: NONINCAPACITATING INJURY
        - 4: REPORTED, NOT EVIDENT
        - 5: UNKNOWN

        Mapping from hierarchical to multiclass:
        - L1=0 (no injury) → NO INDICATION OF INJURY
        - L1=1, L2=1, L2.5=1 (injury, severe, fatal) → FATAL
        - L1=1, L2=1, L2.5=0 (injury, severe, not fatal) → INCAPACITATING INJURY
        - L1=1, L2=0, L3=1 (injury, minor, reported) → REPORTED, NOT EVIDENT
        - L1=1, L2=0, L3=0 (injury, minor, visible) → NONINCAPACITATING INJURY

        Args:
            X: Feature matrix.
            label_encoder: Optional encoder to get correct class indices.
                           If None, uses hardcoded alphabetical indices.

        Returns:
            Array of multiclass predictions.
        """
        l1_pred, l2_pred, l25_pred, l3_pred = self.predict_binary(X)

        # Get class indices
        if label_encoder is not None:
            classes = list(label_encoder.classes_)
            idx_no_indication = classes.index("NO INDICATION OF INJURY") if "NO INDICATION OF INJURY" in classes else 2
            idx_nonincapacitating = classes.index("NONINCAPACITATING INJURY") if "NONINCAPACITATING INJURY" in classes else 3
            idx_incapacitating = classes.index("INCAPACITATING INJURY") if "INCAPACITATING INJURY" in classes else 1
            idx_fatal = classes.index("FATAL") if "FATAL" in classes else 0
            idx_reported = classes.index("REPORTED, NOT EVIDENT") if "REPORTED, NOT EVIDENT" in classes else 4
        else:
            # Alphabetical encoding (sklearn default)
            idx_fatal = 0
            idx_incapacitating = 1
            idx_no_indication = 2
            idx_nonincapacitating = 3
            idx_reported = 4

        # Start with no injury
        multiclass = np.full(len(X), idx_no_indication, dtype=int)

        # Level 1: Injury cases
        injury_mask = l1_pred == 1

        # Level 2: Severe vs Minor (among injuries)
        severe_mask = injury_mask & (l2_pred == 1)
        minor_mask = injury_mask & (l2_pred == 0)

        # Level 2.5: FATAL vs INCAPACITATING (among severe cases)
        # Use L2.5 model predictions instead of percentile heuristic
        fatal_mask = severe_mask & (l25_pred == 1)
        incap_mask = severe_mask & (l25_pred == 0)

        multiclass[fatal_mask] = idx_fatal
        multiclass[incap_mask] = idx_incapacitating

        # Level 3: Reported vs Visible (among minor injuries)
        # IS_REPORTED=1 → L3=1 → REPORTED, NOT EVIDENT
        # IS_REPORTED=0 → L3=0 → NONINCAPACITATING INJURY
        reported_mask = minor_mask & (l3_pred == 1)  # REPORTED, NOT EVIDENT
        visible_mask = minor_mask & (l3_pred == 0)   # NONINCAPACITATING

        multiclass[reported_mask] = idx_reported
        multiclass[visible_mask] = idx_nonincapacitating

        return multiclass


def evaluate_hierarchical(
    clf: HierarchicalClassifier,
    X_test: np.ndarray,
    y_injury: np.ndarray,
    y_severe: np.ndarray,
    y_reported: np.ndarray,
    y_fatal: np.ndarray,
    y_original: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    """Evaluate hierarchical classifier performance.

    Args:
        clf: Trained hierarchical classifier.
        X_test: Test feature matrix.
        y_injury: Binary injury labels.
        y_severe: Binary severity labels.
        y_reported: Binary reported injury labels (REPORTED vs NONINCAP).
        y_fatal: Binary fatal injury labels (FATAL vs INCAPACITATING).
        y_original: Original 5-class labels.
        label_encoder: Encoder for original labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    results = {}

    # Level 1 evaluation
    logger.info("=" * 60)
    logger.info("LEVEL 1 EVALUATION: INJURY vs NO_INJURY")
    logger.info("=" * 60)

    l1_pred, l2_pred, l25_pred, l3_pred = clf.predict_binary(X_test)

    l1_acc = accuracy_score(y_injury, l1_pred)
    l1_prec = precision_score(y_injury, l1_pred)
    l1_rec = recall_score(y_injury, l1_pred)
    l1_f1 = f1_score(y_injury, l1_pred)

    logger.info(f"L1 Accuracy:  {l1_acc:.4f}")
    logger.info(f"L1 Precision: {l1_prec:.4f}")
    logger.info(f"L1 Recall:    {l1_rec:.4f} (INJURY detection)")
    logger.info(f"L1 F1:        {l1_f1:.4f}")

    results["l1_accuracy"] = l1_acc
    results["l1_precision"] = l1_prec
    results["l1_recall"] = l1_rec
    results["l1_f1"] = l1_f1

    # L1 Confusion Matrix
    l1_cm = confusion_matrix(y_injury, l1_pred)
    logger.info(f"\nL1 Confusion Matrix:\n{l1_cm}")

    # Level 2 evaluation (on injury cases only)
    logger.info("=" * 60)
    logger.info("LEVEL 2 EVALUATION: SEVERE vs NON-SEVERE (injury cases)")
    logger.info("=" * 60)

    injury_mask = y_injury == 1
    y_severe_test = y_severe[injury_mask]
    l2_pred_injury = l2_pred[injury_mask]

    if len(y_severe_test) > 0:
        l2_acc = accuracy_score(y_severe_test, l2_pred_injury)
        l2_prec = precision_score(y_severe_test, l2_pred_injury, zero_division=0)
        l2_rec = recall_score(y_severe_test, l2_pred_injury, zero_division=0)
        l2_f1 = f1_score(y_severe_test, l2_pred_injury, zero_division=0)

        logger.info(f"L2 Accuracy:  {l2_acc:.4f}")
        logger.info(f"L2 Precision: {l2_prec:.4f}")
        logger.info(f"L2 Recall:    {l2_rec:.4f} (SEVERE detection)")
        logger.info(f"L2 F1:        {l2_f1:.4f}")

        results["l2_accuracy"] = l2_acc
        results["l2_precision"] = l2_prec
        results["l2_recall"] = l2_rec
        results["l2_f1"] = l2_f1

        l2_cm = confusion_matrix(y_severe_test, l2_pred_injury)
        logger.info(f"\nL2 Confusion Matrix:\n{l2_cm}")

    # Level 2.5 evaluation (on severe injury cases only)
    logger.info("=" * 60)
    logger.info("LEVEL 2.5 EVALUATION: FATAL vs INCAPACITATING (severe cases)")
    logger.info("=" * 60)

    severe_mask = (y_injury == 1) & (y_severe == 1)
    y_fatal_test = y_fatal[severe_mask]
    l25_pred_severe = l25_pred[severe_mask]

    if len(y_fatal_test) > 0 and clf.l25_model is not None:
        l25_acc = accuracy_score(y_fatal_test, l25_pred_severe)
        l25_prec = precision_score(y_fatal_test, l25_pred_severe, zero_division=0)
        l25_rec = recall_score(y_fatal_test, l25_pred_severe, zero_division=0)
        l25_f1 = f1_score(y_fatal_test, l25_pred_severe, zero_division=0)

        logger.info(f"L2.5 Accuracy:  {l25_acc:.4f}")
        logger.info(f"L2.5 Precision: {l25_prec:.4f}")
        logger.info(f"L2.5 Recall:    {l25_rec:.4f} (FATAL detection)")
        logger.info(f"L2.5 F1:        {l25_f1:.4f}")

        results["l25_accuracy"] = l25_acc
        results["l25_precision"] = l25_prec
        results["l25_recall"] = l25_rec
        results["l25_f1"] = l25_f1

        l25_cm = confusion_matrix(y_fatal_test, l25_pred_severe)
        logger.info(f"\nL2.5 Confusion Matrix:\n{l25_cm}")
    else:
        logger.info("L2.5 model not trained or no severe injury cases in test set")

    # Level 3 evaluation (on minor injury cases only)
    logger.info("=" * 60)
    logger.info("LEVEL 3 EVALUATION: REPORTED vs NONINCAP (minor injury cases)")
    logger.info("=" * 60)

    # Minor injury = injury + not severe
    minor_mask = (y_injury == 1) & (y_severe == 0)
    y_reported_test = y_reported[minor_mask]
    l3_pred_minor = l3_pred[minor_mask]

    if len(y_reported_test) > 0 and clf.l3_model is not None:
        l3_acc = accuracy_score(y_reported_test, l3_pred_minor)
        l3_prec = precision_score(y_reported_test, l3_pred_minor, zero_division=0)
        l3_rec = recall_score(y_reported_test, l3_pred_minor, zero_division=0)
        l3_f1 = f1_score(y_reported_test, l3_pred_minor, zero_division=0)

        logger.info(f"L3 Accuracy:  {l3_acc:.4f}")
        logger.info(f"L3 Precision: {l3_prec:.4f}")
        logger.info(f"L3 Recall:    {l3_rec:.4f} (REPORTED detection)")
        logger.info(f"L3 F1:        {l3_f1:.4f}")

        results["l3_accuracy"] = l3_acc
        results["l3_precision"] = l3_prec
        results["l3_recall"] = l3_rec
        results["l3_f1"] = l3_f1

        l3_cm = confusion_matrix(y_reported_test, l3_pred_minor)
        logger.info(f"\nL3 Confusion Matrix:\n{l3_cm}")
    else:
        logger.info("L3 model not trained or no minor injury cases in test set")

    # Multiclass evaluation
    logger.info("=" * 60)
    logger.info("MULTICLASS EVALUATION (mapped to original 5 classes)")
    logger.info("=" * 60)

    y_pred_multi = clf.predict_multiclass(X_test, label_encoder=label_encoder)

    mc_acc = accuracy_score(y_original, y_pred_multi)
    mc_f1_macro = f1_score(y_original, y_pred_multi, average="macro", zero_division=0)
    mc_f1_weighted = f1_score(y_original, y_pred_multi, average="weighted", zero_division=0)

    logger.info(f"Multiclass Accuracy:    {mc_acc:.4f}")
    logger.info(f"Multiclass F1 (macro):  {mc_f1_macro:.4f}")
    logger.info(f"Multiclass F1 (weighted): {mc_f1_weighted:.4f}")

    results["mc_accuracy"] = mc_acc
    results["mc_f1_macro"] = mc_f1_macro
    results["mc_f1_weighted"] = mc_f1_weighted

    # Per-class metrics
    class_names = label_encoder.classes_
    logger.info("\nPer-class Classification Report:")
    report = classification_report(
        y_original,
        y_pred_multi,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )
    logger.info(f"\n{report}")

    # Per-class recall (THE KEY METRIC)
    logger.info("\n" + "=" * 60)
    logger.info("KEY METRICS: PER-CLASS RECALL")
    logger.info("=" * 60)
    for i, cls in enumerate(class_names):
        mask = y_original == i
        if np.sum(mask) > 0:
            cls_recall = recall_score(
                y_original[mask] == i,
                y_pred_multi[mask] == i,
                zero_division=0,
            )
            # More accurate: calculate directly
            cls_pred_correct = np.sum((y_original == i) & (y_pred_multi == i))
            cls_total = np.sum(y_original == i)
            cls_recall = cls_pred_correct / cls_total if cls_total > 0 else 0
            results[f"recall_{cls}"] = cls_recall
            logger.info(f"  {cls}: {cls_recall:.4f} ({cls_pred_correct}/{cls_total})")

    # Confusion matrix
    mc_cm = confusion_matrix(y_original, y_pred_multi)
    logger.info(f"\nMulticlass Confusion Matrix:\n{mc_cm}")

    return results


def run_hierarchical_pipeline(config: HierarchicalConfig | None = None) -> dict:
    """Run the full hierarchical classification pipeline.

    Args:
        config: Configuration parameters. Uses defaults if None.

    Returns:
        Dictionary of evaluation results.
    """
    if config is None:
        config = HierarchicalConfig()

    logger.info("=" * 60)
    logger.info("HIERARCHICAL CLASSIFICATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config}")

    # Load and merge data
    logger.info("\n[1/6] Loading and merging data...")
    df = triple_merge()
    logger.info(f"Merged dataset shape: {df.shape}")

    # Sample if specified
    if config.sample_size is not None and len(df) > config.sample_size:
        df = df.sample(n=config.sample_size, random_state=config.random_state)
        logger.info(f"Sampled to {len(df)} rows")

    # Add binary targets
    logger.info("\n[2/6] Engineering features and adding targets...")
    df = engineer_all_features(df, include_interactions=True, include_clusters=False)
    df = add_binary_targets(df)
    df = add_ordinal_severity(df)

    # Encode original target
    label_encoder = LabelEncoder()
    df["SEVERITY_ENCODED"] = label_encoder.fit_transform(
        df["MOST_SEVERE_INJURY"].fillna("UNKNOWN")
    )
    logger.info(f"Severity classes: {list(label_encoder.classes_)}")

    # Show target distributions
    logger.info("\nTarget distributions:")
    logger.info(f"  IS_INJURY: {df['IS_INJURY'].value_counts().to_dict()}")
    logger.info(f"  IS_SEVERE: {df['IS_SEVERE'].value_counts().to_dict()}")
    logger.info(f"  IS_FATAL: {df['IS_FATAL'].value_counts().to_dict()}")
    logger.info(f"  IS_REPORTED: {df['IS_REPORTED'].value_counts().to_dict()}")
    logger.info(f"  Original: {df['MOST_SEVERE_INJURY'].value_counts().to_dict()}")

    # Prepare features
    logger.info("\n[3/6] Preparing feature matrix...")
    X_df, feature_cols = prepare_features(df)
    logger.info(f"Feature matrix shape: {X_df.shape}")
    logger.info(f"Features: {feature_cols[:10]}... ({len(feature_cols)} total)")

    X = X_df.values
    y_injury = df["IS_INJURY"].values
    y_severe = df["IS_SEVERE"].values
    y_fatal = df["IS_FATAL"].values
    y_reported = df["IS_REPORTED"].values
    y_original = df["SEVERITY_ENCODED"].values

    # Split data
    logger.info("\n[4/6] Splitting data...")
    (
        X_train,
        X_test,
        y_inj_train,
        y_inj_test,
        y_sev_train,
        y_sev_test,
        y_fat_train,
        y_fat_test,
        y_rep_train,
        y_rep_test,
        y_orig_train,
        y_orig_test,
    ) = train_test_split(
        X,
        y_injury,
        y_severe,
        y_fatal,
        y_reported,
        y_original,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_original,
    )

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Further split training into train and validation for threshold optimization
    # This prevents test set leakage during threshold tuning
    (
        X_train_final,
        X_val,
        y_inj_train_final,
        y_inj_val,
        y_sev_train_final,
        y_sev_val,
        y_fat_train_final,
        y_fat_val,
        y_rep_train_final,
        y_rep_val,
    ) = train_test_split(
        X_train,
        y_inj_train,
        y_sev_train,
        y_fat_train,
        y_rep_train,
        test_size=0.2,  # 20% of training data for validation
        random_state=config.random_state,
        stratify=y_inj_train,
    )

    logger.info(f"Train (final): {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train hierarchical classifier on training set (not validation)
    logger.info("\n[5/6] Training hierarchical classifier...")
    clf = HierarchicalClassifier(config)
    clf.feature_cols = feature_cols
    clf.fit(X_train_final, y_inj_train_final, y_sev_train_final, y_rep_train_final, y_fat_train_final)

    # Optimize thresholds on VALIDATION set (not test set!)
    logger.info("\nOptimizing thresholds on validation set...")
    clf.optimize_thresholds(
        X_val,
        y_inj_val,
        y_sev_val,
        y_rep_val,
        y_fat_val,
        l1_target_recall=0.7,
        l25_target_recall=0.4,
        l3_target_recall=0.35,  # Lower target for balanced REPORTED/NONINCAP
    )

    # Evaluate
    logger.info("\n[6/6] Evaluating...")
    results = evaluate_hierarchical(
        clf,
        X_test,
        y_inj_test,
        y_sev_test,
        y_rep_test,
        y_fat_test,
        y_orig_test,
        label_encoder,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    target_metrics = ["recall_FATAL", "recall_INCAPACITATING INJURY"]
    for metric in target_metrics:
        if metric in results:
            old_val = 0.043 if "FATAL" in metric else 0.129
            new_val = results[metric]
            improvement = (new_val - old_val) / old_val * 100 if old_val > 0 else 0
            logger.info(f"{metric}: {new_val:.4f} (was {old_val:.4f}, {improvement:+.1f}%)")

    return results


if __name__ == "__main__":
    # Run with default config
    results = run_hierarchical_pipeline()

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
