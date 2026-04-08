"""Simplified 3-class severity classification pipeline.

This module implements a simplified 2-level classification approach:
- Level 1: INJURY vs NO_INJURY
- Level 2: SEVERE vs MINOR (for injury cases only)

Output classes:
- NO_INJURY: No indication of injury
- MINOR: Nonincapacitating + Reported injuries
- SEVERE: Fatal + Incapacitating injuries

This simplifies the 5-class problem into 3 actionable categories,
eliminating the problematic L2.5 (Fatal vs Incapacitating) and
L3 (Reported vs Nonincapacitating) levels.

Usage:
    python training/main_simplified.py
    python training/main_simplified.py --sample 10000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.feature_engineering import (
    add_binary_targets,
    add_ordinal_severity,
    engineer_all_features,
)
from data_preparation.triple_merge import triple_merge

from training.hierarchical.config import SimplifiedTreeConfig
from training.hierarchical.simplified_targets import (
    SimplifiedTargets,
    prepare_simplified_targets,
    SIMPLIFIED_CLASS_NAMES,
)
from training.hierarchical.simplified_classifier import (
    SimplifiedTreeClassifier,
    save_simplified_model,
)
from training.feature_selection import filter_by_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def stratified_sample_severe(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample dataset ensuring SEVERE cases (FATAL + INCAPACITATING) are represented.

    Args:
        df: Full DataFrame with MOST_SEVERE_INJURY column.
        sample_size: Target sample size.
        random_state: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with severe cases well-represented.
    """
    # Separate SEVERE (FATAL + INCAPACITATING) from rest
    severe_mask = df["MOST_SEVERE_INJURY"].isin(["FATAL", "INCAPACITATING INJURY"])
    severe_df = df[severe_mask]
    other_df = df[~severe_mask]

    n_severe = len(severe_df)
    logger.info(f"  SEVERE cases in full data: {n_severe}")

    # If we have more severe than sample_size, just sample from severe
    if n_severe >= sample_size:
        result = severe_df.sample(n=sample_size, random_state=random_state)
        logger.info(f"  Sampled {sample_size} from {n_severe} SEVERE cases")
        return result.reset_index(drop=True)

    # Otherwise, take all severe and sample from rest
    n_other = sample_size - n_severe

    if len(other_df) > n_other:
        other_sample = other_df.sample(n=n_other, random_state=random_state)
    else:
        other_sample = other_df

    result = pd.concat([severe_df, other_sample], ignore_index=True)
    logger.info(f"  Stratified sample: {n_severe} SEVERE + {len(other_sample)} other = {len(result)}")

    return result.sample(frac=1, random_state=random_state).reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare feature matrix from DataFrame.

    Args:
        df: DataFrame with engineered features.

    Returns:
        Tuple of (feature DataFrame, feature column names).
    """
    exclude_cols = {
        # Identifiers and timestamps
        "CRASH_RECORD_ID", "CRASH_DATE", "DATE_POLICE_NOTIFIED", "CRASH_DATE_EST_I",
        # Direct injury indicators (targets)
        "MOST_SEVERE_INJURY", "INJURIES_TOTAL", "INJURIES_FATAL",
        "INJURIES_INCAPACITATING", "INJURIES_NON_INCAPACITATING",
        "INJURIES_REPORTED_NOT_EVIDENT", "INJURIES_NO_INDICATION", "INJURIES_UNKNOWN",
        # Binary targets
        "IS_INJURY", "IS_SEVERE", "IS_FATAL", "IS_REPORTED",
        "SEVERITY_LEVEL", "SEVERITY_ENCODED",
        # Geographic coordinates
        "LATITUDE", "LONGITUDE", "LOCATION",
        # Post-crash investigation fields (data leakage!)
        "STATEMENTS_TAKEN_I", "WORK_ZONE_I", "PHOTOS_TAKEN_I", "DOORING_I",
        # People-derived outcome features (data leakage!)
        "ejected_any", "using_seatbelt_mean", "cell_phone_any",
        "bac_positive_any", "bac_clean_max",
        # CRASH_TYPE directly encodes injury!
        "CRASH_TYPE", "REPORT_TYPE",
        # Redundant temporal
        "HOUR",
        # High-cardinality location features
        "STREET_NAME",
    }

    safe_categorical_cols = [
        "WEATHER_CONDITION", "LIGHTING_CONDITION", "FIRST_CRASH_TYPE",
        "TRAFFICWAY_TYPE", "ROADWAY_SURFACE_COND", "TRAFFIC_CONTROL_DEVICE",
        "DEVICE_CONDITION", "ALIGNMENT", "ROAD_DEFECT",
        "PRIM_CONTRIBUTORY_CAUSE", "DAMAGE",
    ]

    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            feature_cols.append(col)

    df_features = df[feature_cols].copy()

    for col in safe_categorical_cols:
        if col in df.columns and col not in exclude_cols:
            le = LabelEncoder()
            values = df[col].fillna("UNKNOWN").astype(str)
            df_features[col] = le.fit_transform(values)
            feature_cols.append(col)

    feature_cols = list(df_features.columns)
    df_features = df_features.fillna(0)

    return df_features, feature_cols


def evaluate_simplified(
    clf: SimplifiedTreeClassifier,
    X: np.ndarray,
    targets: SimplifiedTargets,
) -> dict:
    """Evaluate simplified classifier on test data.

    Args:
        clf: Trained simplified classifier.
        X: Feature matrix.
        targets: Test targets.

    Returns:
        Dictionary of evaluation metrics.
    """
    results = {}

    # Get predictions
    l1_proba, l2_proba = clf.predict_proba(X)
    l1_pred, l2_pred = clf.predict_binary(X)
    y_pred = clf.predict(X)

    # Binary metrics for each level
    logger.info("=" * 60)
    logger.info("LEVEL 1 (INJURY) EVALUATION")
    logger.info("=" * 60)
    l1_metrics = _binary_metrics(targets.y_injury, l1_pred, "l1")
    results.update(l1_metrics)

    # L2 metrics on injury cases only
    injury_mask = targets.y_injury == 1
    if np.sum(injury_mask) > 0:
        logger.info("=" * 60)
        logger.info("LEVEL 2 (SEVERE) EVALUATION - injury cases only")
        logger.info("=" * 60)
        l2_metrics = _binary_metrics(
            targets.y_severe[injury_mask],
            l2_pred[injury_mask],
            "l2",
        )
        results.update(l2_metrics)

    # 3-class metrics
    logger.info("=" * 60)
    logger.info("3-CLASS EVALUATION")
    logger.info("=" * 60)

    accuracy = accuracy_score(targets.y_simplified, y_pred)
    f1_macro = f1_score(targets.y_simplified, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(targets.y_simplified, y_pred, average="weighted", zero_division=0)

    logger.info(f"Accuracy:      {accuracy:.4f}")
    logger.info(f"F1 (macro):    {f1_macro:.4f}")
    logger.info(f"F1 (weighted): {f1_weighted:.4f}")

    results["accuracy"] = accuracy
    results["f1_macro"] = f1_macro
    results["f1_weighted"] = f1_weighted

    # Per-class report
    logger.info("\nClassification Report:")
    report = classification_report(
        targets.y_simplified,
        y_pred,
        target_names=SIMPLIFIED_CLASS_NAMES,
        zero_division=0,
    )
    logger.info(f"\n{report}")

    # Per-class recall
    logger.info("=" * 60)
    logger.info("PER-CLASS RECALL")
    logger.info("=" * 60)
    for i, class_name in enumerate(SIMPLIFIED_CLASS_NAMES):
        class_mask = targets.y_simplified == i
        if np.sum(class_mask) > 0:
            class_recall = np.mean(y_pred[class_mask] == i)
            n_correct = np.sum(y_pred[class_mask] == i)
            n_total = np.sum(class_mask)
            logger.info(f"  {class_name}: {class_recall:.4f} ({n_correct}/{n_total})")
            results[f"recall_{class_name}"] = class_recall

    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(targets.y_simplified, y_pred)
    logger.info(f"\n{cm}")
    results["confusion_matrix"] = cm

    return results


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    """Compute binary classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1:        {f1:.4f}")

    return {
        f"{prefix}_accuracy": acc,
        f"{prefix}_precision": prec,
        f"{prefix}_recall": rec,
        f"{prefix}_f1": f1,
    }


def run_simplified_pipeline(
    sample_size: int | None = None,
    feature_filter: str = "drop-low",
) -> dict:
    """Run the simplified 3-class classification pipeline.

    Args:
        sample_size: Optional limit on dataset size for faster experimentation.
        feature_filter: Feature filtering mode ("none" or "drop-low").

    Returns:
        Dictionary of evaluation results.
    """
    logger.info("=" * 60)
    logger.info("SIMPLIFIED 3-CLASS CLASSIFICATION PIPELINE")
    logger.info("=" * 60)

    config = SimplifiedTreeConfig(sample_size=sample_size)
    logger.info(f"Configuration: {config}")

    # Load and merge data
    logger.info("\n[1/6] Loading and merging data...")
    df = triple_merge()
    logger.info(f"Merged dataset shape: {df.shape}")

    # Sample if specified - use stratified sampling for SEVERE cases
    if config.sample_size is not None and len(df) > config.sample_size:
        df = stratified_sample_severe(df, config.sample_size, config.random_state)

    # Add binary targets
    logger.info("\n[2/6] Engineering features and adding targets...")
    df = engineer_all_features(df, include_interactions=True, include_clusters=False)
    df = add_binary_targets(df)

    # Show simplified target distribution
    severe_count = df["MOST_SEVERE_INJURY"].isin(["FATAL", "INCAPACITATING INJURY"]).sum()
    minor_count = df["MOST_SEVERE_INJURY"].isin(["NONINCAPACITATING INJURY", "REPORTED, NOT EVIDENT"]).sum()
    no_injury_count = (df["MOST_SEVERE_INJURY"] == "NO INDICATION OF INJURY").sum()

    logger.info("\nSimplified target distribution:")
    logger.info(f"  SEVERE (Fatal+Incap): {severe_count}")
    logger.info(f"  MINOR (NonIncap+Reported): {minor_count}")
    logger.info(f"  NO_INJURY: {no_injury_count}")

    # Prepare features
    logger.info("\n[3/6] Preparing feature matrix...")
    X_df, feature_cols = prepare_features(df)
    logger.info(f"Feature matrix shape (before filter): {X_df.shape}")

    # Apply importance-based filtering
    X_df, feature_cols, filter_result = filter_by_importance(
        X_df, feature_cols, mode=feature_filter
    )
    logger.info(f"Feature matrix shape (after filter): {X_df.shape}")
    logger.info(f"Features: {feature_cols[:10]}... ({len(feature_cols)} total)")

    X = X_df.values

    # Prepare simplified targets
    targets = prepare_simplified_targets(df)

    # Split data
    logger.info("\n[4/6] Splitting data...")

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=targets.y_simplified,
    )

    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=config.val_size,
        random_state=config.random_state,
        stratify=targets.y_simplified[train_val_idx],
    )

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    def subset_targets(indices: np.ndarray) -> SimplifiedTargets:
        return SimplifiedTargets(
            y_injury=targets.y_injury[indices],
            y_severe=targets.y_severe[indices],
            y_simplified=targets.y_simplified[indices],
            label_encoder=targets.label_encoder,
        )

    targets_train = subset_targets(train_idx)
    targets_val = subset_targets(val_idx)
    targets_test = subset_targets(test_idx)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create and train classifier
    logger.info("\n[5/6] Training simplified classifier...")

    clf = SimplifiedTreeClassifier(
        l1_model=config.l1_model,
        l2_model=config.l2_model,
        l1_sampling_strategy=config.l1_sampling_strategy,
        l2_sampling_strategy=config.l2_sampling_strategy,
        l2_weight_multiplier=config.l2_weight_multiplier,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        n_jobs=config.n_jobs,
        l1_threshold=config.l1_threshold,
        l2_threshold=config.l2_threshold,
        random_state=config.random_state,
    )
    clf.feature_cols = feature_cols
    clf.fit(X_train, targets_train)

    # Optimize thresholds on validation set
    logger.info("\nOptimizing thresholds on validation set...")
    clf.optimize_thresholds(
        X_val,
        targets_val,
        l1_target_recall=0.7,
        l2_target_recall=0.5,
    )

    # Evaluate
    logger.info("\n[6/6] Evaluating on test set...")
    results = evaluate_simplified(clf, X_test, targets_test)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"3-Class Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Macro F1: {results['f1_macro']:.4f}")
    for class_name in SIMPLIFIED_CLASS_NAMES:
        key = f"recall_{class_name}"
        if key in results:
            logger.info(f"  {class_name} Recall: {results[key]:.4f}")

    # Save model
    model_dir = PROJECT_ROOT / "models" / "trained" / "simplified_3class"
    save_simplified_model(clf, str(model_dir))
    logger.info(f"Model saved to {model_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run simplified 3-class crash severity classification"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for faster experimentation (default: use all data)",
    )
    parser.add_argument(
        "--feature-filter",
        type=str,
        choices=["none", "drop-low", "drop-review"],
        default="drop-low",
        help="Feature filtering mode: 'none' (all features), 'drop-low' (drop DROP features), 'drop-review' (drop DROP + REVIEW features). Default: drop-low",
    )

    args = parser.parse_args()

    results = run_simplified_pipeline(
        sample_size=args.sample,
        feature_filter=args.feature_filter,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif key != "confusion_matrix":
            print(f"  {key}: {value}")
