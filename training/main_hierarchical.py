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

    # Get numeric and encoded categorical columns
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            feature_cols.append(col)
        elif df[col].dtype == "object":
            # Skip categorical columns for now - they cause data leakage
            # TODO: investigate which categorical features leak and add safe ones
            pass

    # Commented out categorical encoding due to data leakage
    # For now, use only numeric features
    # categorical_cols = df.select_dtypes(include=["object"]).columns
    # categorical_cols = [c for c in categorical_cols if c not in exclude_cols]

    # Create copy with only numeric features
    df_features = df[feature_cols].copy()

    feature_cols = list(df_features.columns)

    # Fill any remaining NaN
    df_features = df_features.fillna(0)

    return df_features, feature_cols


class HierarchicalClassifier:
    """Two-stage hierarchical classifier for crash severity.

    Level 1: Predicts INJURY vs NO_INJURY
    Level 2: For injury cases, predicts SEVERE vs NON_SEVERE

    Final mapping:
    - NO_INJURY predicted at L1 → NO INDICATION OF INJURY
    - INJURY at L1, NON_SEVERE at L2 → NONINCAPACITATING INJURY
    - INJURY at L1, SEVERE at L2 → INCAPACITATING INJURY (default) or FATAL (if very confident)
    """

    def __init__(self, config: HierarchicalConfig):
        """Initialize hierarchical classifier.

        Args:
            config: Configuration parameters.
        """
        self.config = config
        self.l1_model = None
        self.l2_model = None
        self.l1_threshold = config.l1_threshold
        self.l2_threshold = config.l2_threshold
        self.feature_cols = None

    def fit(
        self,
        X_train: np.ndarray,
        y_injury: np.ndarray,
        y_severe: np.ndarray,
    ) -> "HierarchicalClassifier":
        """Fit both levels of the hierarchical classifier.

        Args:
            X_train: Feature matrix.
            y_injury: Binary labels (1=injury, 0=no injury).
            y_severe: Binary labels (1=severe, 0=not severe).

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
            scale_pos_weight=l2_weight,
        )

        logger.info(f"Training L2 model ({self.config.l2_model})...")
        self.l2_model.fit(X_l2, y_l2)

        return self

    def optimize_thresholds(
        self,
        X_val: np.ndarray,
        y_injury: np.ndarray,
        y_severe: np.ndarray,
        l1_target_recall: float = 0.7,
        l2_target_recall: float = 0.5,
    ) -> None:
        """Optimize classification thresholds on validation data.

        Args:
            X_val: Validation feature matrix.
            y_injury: Binary injury labels.
            y_severe: Binary severity labels.
            l1_target_recall: Target recall for injury detection.
            l2_target_recall: Target recall for severity detection.
        """
        logger.info("Optimizing classification thresholds...")

        # L1 threshold optimization
        l1_proba = self.l1_model.predict_proba(X_val)[:, 1]
        self.l1_threshold = find_optimal_threshold(
            y_injury, l1_proba, target_recall=l1_target_recall
        )
        logger.info(f"Optimized L1 threshold: {self.l1_threshold:.3f}")

        # L2 threshold optimization (on injury cases)
        injury_mask = y_injury == 1
        if np.sum(injury_mask) > 0:
            X_val_injury = X_val[injury_mask]
            y_val_severe = y_severe[injury_mask]
            l2_proba = self.l2_model.predict_proba(X_val_injury)[:, 1]
            self.l2_threshold = find_optimal_threshold(
                y_val_severe, l2_proba, target_recall=l2_target_recall
            )
            logger.info(f"Optimized L2 threshold: {self.l2_threshold:.3f}")

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get probability predictions for both levels.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (L1 probabilities, L2 probabilities).
        """
        l1_proba = self.l1_model.predict_proba(X)[:, 1]

        # L2 prediction for all samples (will only use for predicted injuries)
        l2_proba = self.l2_model.predict_proba(X)[:, 1]

        return l1_proba, l2_proba

    def predict_binary(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict binary labels for both levels.

        Args:
            X: Feature matrix.

        Returns:
            Tuple of (L1 predictions, L2 predictions).
        """
        l1_proba, l2_proba = self.predict_proba(X)

        l1_pred = (l1_proba >= self.l1_threshold).astype(int)
        l2_pred = (l2_proba >= self.l2_threshold).astype(int)

        return l1_pred, l2_pred

    def predict_multiclass(
        self,
        X: np.ndarray,
        label_encoder: LabelEncoder | None = None,
    ) -> np.ndarray:
        """Predict original 5-class severity labels.

        LabelEncoder sorts alphabetically, so typical indices are:
        - 0: FATAL
        - 1: INCAPACITATING INJURY
        - 2: NO INDICATION OF INJURY
        - 3: NONINCAPACITATING INJURY
        - 4: REPORTED, NOT EVIDENT
        - 5: UNKNOWN

        Mapping from hierarchical to multiclass:
        - L1=0 (no injury) → NO INDICATION OF INJURY
        - L1=1, L2=0 (injury, not severe) → NONINCAPACITATING INJURY
        - L1=1, L2=1 (injury, severe) → INCAPACITATING INJURY
        - Very high severity confidence → FATAL

        Args:
            X: Feature matrix.
            label_encoder: Optional encoder to get correct class indices.
                           If None, uses hardcoded alphabetical indices.

        Returns:
            Array of multiclass predictions.
        """
        l1_pred, l2_pred = self.predict_binary(X)
        l1_proba, l2_proba = self.predict_proba(X)

        # Get class indices
        if label_encoder is not None:
            classes = list(label_encoder.classes_)
            idx_no_indication = classes.index("NO INDICATION OF INJURY") if "NO INDICATION OF INJURY" in classes else 2
            idx_nonincapacitating = classes.index("NONINCAPACITATING INJURY") if "NONINCAPACITATING INJURY" in classes else 3
            idx_incapacitating = classes.index("INCAPACITATING INJURY") if "INCAPACITATING INJURY" in classes else 1
            idx_fatal = classes.index("FATAL") if "FATAL" in classes else 0
        else:
            # Alphabetical encoding (sklearn default)
            idx_fatal = 0
            idx_incapacitating = 1
            idx_no_indication = 2
            idx_nonincapacitating = 3

        # Start with no injury
        multiclass = np.full(len(X), idx_no_indication, dtype=int)

        # Injury cases: default to minor injury
        injury_mask = l1_pred == 1
        multiclass[injury_mask] = idx_nonincapacitating

        # Severe cases
        severe_mask = injury_mask & (l2_pred == 1)
        multiclass[severe_mask] = idx_incapacitating

        # FATAL: Use top percentile of severe cases based on probability
        # FATAL is ~6% of severe cases (1094 FATAL vs 16804 INCAPACITATING)
        # So we classify the top ~6% of severe predictions as FATAL
        if severe_mask.sum() > 0:
            severe_probas = l2_proba[severe_mask]
            # Use 94th percentile as threshold (top 6% → FATAL)
            fatal_threshold = np.percentile(severe_probas, 94)
            very_severe_mask = severe_mask & (l2_proba >= fatal_threshold)
            multiclass[very_severe_mask] = idx_fatal

        return multiclass


def evaluate_hierarchical(
    clf: HierarchicalClassifier,
    X_test: np.ndarray,
    y_injury: np.ndarray,
    y_severe: np.ndarray,
    y_original: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    """Evaluate hierarchical classifier performance.

    Args:
        clf: Trained hierarchical classifier.
        X_test: Test feature matrix.
        y_injury: Binary injury labels.
        y_severe: Binary severity labels.
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

    l1_pred, l2_pred = clf.predict_binary(X_test)

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
    logger.info(f"  Original: {df['MOST_SEVERE_INJURY'].value_counts().to_dict()}")

    # Prepare features
    logger.info("\n[3/6] Preparing feature matrix...")
    X_df, feature_cols = prepare_features(df)
    logger.info(f"Feature matrix shape: {X_df.shape}")
    logger.info(f"Features: {feature_cols[:10]}... ({len(feature_cols)} total)")

    X = X_df.values
    y_injury = df["IS_INJURY"].values
    y_severe = df["IS_SEVERE"].values
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
        y_orig_train,
        y_orig_test,
    ) = train_test_split(
        X,
        y_injury,
        y_severe,
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
    ) = train_test_split(
        X_train,
        y_inj_train,
        y_sev_train,
        test_size=0.2,  # 20% of training data for validation
        random_state=config.random_state,
        stratify=y_inj_train,
    )

    logger.info(f"Train (final): {len(X_train_final)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train hierarchical classifier on training set (not validation)
    logger.info("\n[5/6] Training hierarchical classifier...")
    clf = HierarchicalClassifier(config)
    clf.feature_cols = feature_cols
    clf.fit(X_train_final, y_inj_train_final, y_sev_train_final)

    # Optimize thresholds on VALIDATION set (not test set!)
    logger.info("\nOptimizing thresholds on validation set...")
    clf.optimize_thresholds(
        X_val,
        y_inj_val,
        y_sev_val,
        l1_target_recall=0.7,
        l2_target_recall=0.5,
    )

    # Evaluate
    logger.info("\n[6/6] Evaluating...")
    results = evaluate_hierarchical(
        clf,
        X_test,
        y_inj_test,
        y_sev_test,
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
