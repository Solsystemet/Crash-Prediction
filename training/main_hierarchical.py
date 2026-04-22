"""Hierarchical classification pipeline for crash severity prediction.

This module implements a 4-level classification approach:
- Level 1: INJURY vs NO_INJURY (14% vs 86% - more balanced than 5-class)
- Level 2: SEVERE vs MINOR (for injury cases only)
- Level 2.5: FATAL vs INCAPACITATING (for severe cases)
- Level 3: REPORTED vs VISIBLE (for minor injuries)

This approach addresses the extreme class imbalance that caused the original
pipeline to achieve only 4.3% FATAL recall despite 80% overall accuracy.

Supports multiple model backends via the hierarchical package:
- Tree-based: XGBoost, RandomForest, ExtraTrees (with SMOTE)
- Neural network: MLP classifiers (with class weighting)

Usage:
    # Tree-based (default)
    python main_hierarchical.py

    # Neural network
    python main_hierarchical.py --model neural
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
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

# Import hierarchical classification package
from training.hierarchical import (
    HierarchicalClassifierBase,
    HierarchicalTargets,
    HierarchicalTreeClassifier,
    HierarchicalNeuralClassifier,
    TreeHierarchicalConfig,
    NeuralHierarchicalConfig,
    prepare_hierarchical_targets,
    evaluate_hierarchical,
)
from training.hierarchical.evaluation import plot_roc_curves, export_roc_data
from training.hierarchical.tree_classifier import save_hierarchical_model
from training.feature_selection import filter_by_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def stratified_sample_with_fatal(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample dataset ensuring FATAL cases are well-represented.

    Takes ALL FATAL cases plus stratified sample of the rest to ensure
    the L2.5 classifier has sufficient training data.

    Args:
        df: Full DataFrame with MOST_SEVERE_INJURY column.
        sample_size: Target sample size.
        random_state: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with all FATAL cases included.
    """
    # Separate FATAL from rest
    fatal_mask = df["MOST_SEVERE_INJURY"] == "FATAL"
    fatal_df = df[fatal_mask]
    other_df = df[~fatal_mask]

    n_fatal = len(fatal_df)
    n_other = sample_size - n_fatal

    logger.info(f"  FATAL cases in full data: {n_fatal}")

    if n_other > 0 and len(other_df) > n_other:
        other_sample = other_df.sample(n=n_other, random_state=random_state)
    else:
        other_sample = other_df

    result = pd.concat([fatal_df, other_sample], ignore_index=True)
    logger.info(f"  Stratified sample: {n_fatal} FATAL + {len(other_sample)} other = {len(result)}")

    # Shuffle to avoid any ordering effects
    return result.sample(frac=1, random_state=random_state).reset_index(drop=True)


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
        "IS_FATAL",
        "IS_REPORTED",
        "SEVERITY_LEVEL",
        "SEVERITY_ENCODED",
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
        "ejected_any",
        "using_seatbelt_mean",
        "cell_phone_any",
        "bac_positive_any",
        "bac_clean_max",
        # CRASH_TYPE directly encodes injury!
        "CRASH_TYPE",
        "REPORT_TYPE",
        # Redundant temporal
        "HOUR",
        # High-cardinality location features
        "STREET_NAME",
    }

    # Safe categorical features known BEFORE crash outcome
    safe_categorical_cols = [
        "WEATHER_CONDITION",
        "LIGHTING_CONDITION",
        "FIRST_CRASH_TYPE",
        "TRAFFICWAY_TYPE",
        "ROADWAY_SURFACE_COND",
        "TRAFFIC_CONTROL_DEVICE",
        "DEVICE_CONDITION",
        "ALIGNMENT",
        "ROAD_DEFECT",
        "PRIM_CONTRIBUTORY_CAUSE",
        "DAMAGE",
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


def run_hierarchical_pipeline(
    model_type: Literal["tree", "neural"] = "tree",
    sample_size: int | None = None,
    feature_filter: str = "drop-low",
) -> dict:
    """Run the full hierarchical classification pipeline.

    Args:
        model_type: Which classifier to use ('tree' or 'neural').
        sample_size: Optional limit on dataset size for faster experimentation.
        feature_filter: Feature filtering mode ("none" or "drop-low").

    Returns:
        Dictionary of evaluation results.
    """
    logger.info("=" * 60)
    logger.info("HIERARCHICAL CLASSIFICATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Model type: {model_type}")

    # Create config based on model type
    if model_type == "tree":
        config = TreeHierarchicalConfig(sample_size=sample_size)
    else:
        config = NeuralHierarchicalConfig(sample_size=sample_size)

    logger.info(f"Configuration: {config}")

    # Load and merge data
    logger.info("\n[1/6] Loading and merging data...")
    df = triple_merge()
    logger.info(f"Merged dataset shape: {df.shape}")

    # Sample if specified - use stratified sampling to ensure FATAL representation
    if config.sample_size is not None and len(df) > config.sample_size:
        df = stratified_sample_with_fatal(df, config.sample_size, config.random_state)

    # Add binary targets
    logger.info("\n[2/6] Engineering features and adding targets...")
    df = engineer_all_features(df, include_interactions=True, include_clusters=False)
    df = add_binary_targets(df)
    df = add_ordinal_severity(df)

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
    logger.info(f"Feature matrix shape (before filter): {X_df.shape}")

    # Apply importance-based filtering
    X_df, feature_cols, filter_result = filter_by_importance(
        X_df, feature_cols, mode=feature_filter
    )
    logger.info(f"Feature matrix shape (after filter): {X_df.shape}")
    logger.info(f"Features: {feature_cols[:10]}... ({len(feature_cols)} total)")

    X = X_df.values

    # Prepare hierarchical targets
    targets = prepare_hierarchical_targets(df)
    logger.info(f"Severity classes: {list(targets.label_encoder.classes_)}")

    # Split data
    logger.info("\n[4/6] Splitting data...")

    # First split: train+val vs test
    split_indices = train_test_split(
        np.arange(len(X)),
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=targets.y_original,
    )
    train_val_idx, test_idx = split_indices

    # Second split: train vs val (for threshold optimization)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=config.val_size,
        random_state=config.random_state,
        stratify=targets.y_injury[train_val_idx],
    )

    # Create train/val/test splits
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    def subset_targets(indices: np.ndarray) -> HierarchicalTargets:
        """Create HierarchicalTargets subset for given indices."""
        return HierarchicalTargets(
            y_injury=targets.y_injury[indices],
            y_severe=targets.y_severe[indices],
            y_fatal=targets.y_fatal[indices],
            y_reported=targets.y_reported[indices],
            y_original=targets.y_original[indices],
            label_encoder=targets.label_encoder,
        )

    targets_train = subset_targets(train_idx)
    targets_val = subset_targets(val_idx)
    targets_test = subset_targets(test_idx)

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create and train classifier
    logger.info("\n[5/6] Training hierarchical classifier...")

    if model_type == "tree":
        clf = HierarchicalTreeClassifier(config)
        clf.feature_cols = feature_cols
        clf.fit(X_train, targets_train)
    else:
        clf = HierarchicalNeuralClassifier(config)
        clf.feature_cols = feature_cols
        clf.fit(X_train, targets_train, X_val, targets_val)

    # Optimize thresholds on validation set
    logger.info("\nOptimizing thresholds on validation set...")
    clf.optimize_thresholds(
        X_val,
        targets_val,
        l1_target_recall=0.7,
        l25_target_recall=0.4,
        l3_target_recall=0.35,
    )

    # Evaluate
    logger.info("\n[6/6] Evaluating on test set...")
    results = evaluate_hierarchical(clf, X_test, targets_test)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Multiclass Accuracy: {results.get('mc_accuracy', 0):.4f}")
    logger.info(f"Macro F1: {results.get('mc_f1_macro', 0):.4f}")
    logger.info(f"Micro F1: {results.get('mc_f1_micro', 0):.4f}")

    target_metrics = ["recall_FATAL", "recall_INCAPACITATING INJURY"]
    for metric in target_metrics:
        if metric in results:
            old_val = 0.043 if "FATAL" in metric else 0.129
            new_val = results[metric]
            improvement = (new_val - old_val) / old_val * 100 if old_val > 0 else 0
            logger.info(f"{metric}: {new_val:.4f} (was {old_val:.4f}, {improvement:+.1f}%)")

    # Generate ROC curves
    logger.info("\nGenerating ROC curves...")
    plots_dir = PROJECT_ROOT / "models" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get probabilities for ROC (4-level hierarchical)
    l1_proba, l2_proba, l25_proba, l3_proba = clf.predict_proba(X_test)

    y_true_levels = {
        "L1 (INJURY vs NO_INJURY)": targets_test.y_injury,
    }
    y_proba_levels = {
        "L1 (INJURY vs NO_INJURY)": l1_proba,
    }

    # L2: SEVERE vs MINOR (injury cases only)
    injury_mask = targets_test.y_injury == 1
    if np.sum(injury_mask) > 0:
        y_true_levels["L2 (SEVERE vs MINOR)"] = targets_test.y_severe[injury_mask]
        y_proba_levels["L2 (SEVERE vs MINOR)"] = l2_proba[injury_mask]

    # L2.5: FATAL vs INCAPACITATING (severe cases only)
    severe_mask = (targets_test.y_injury == 1) & (targets_test.y_severe == 1)
    if np.sum(severe_mask) > 0:
        y_true_levels["L2.5 (FATAL vs INCAP)"] = targets_test.y_fatal[severe_mask]
        y_proba_levels["L2.5 (FATAL vs INCAP)"] = l25_proba[severe_mask]

    # L3: REPORTED vs VISIBLE (minor injury cases only)
    minor_mask = (targets_test.y_injury == 1) & (targets_test.y_severe == 0)
    if np.sum(minor_mask) > 0:
        y_true_levels["L3 (REPORTED vs VISIBLE)"] = targets_test.y_reported[minor_mask]
        y_proba_levels["L3 (REPORTED vs VISIBLE)"] = l3_proba[minor_mask]

    roc_path = plots_dir / "roc_curves_hierarchical.png"
    auc_scores = plot_roc_curves(
        y_true_levels=y_true_levels,
        y_proba_levels=y_proba_levels,
        save_path=roc_path,
        title="ROC Curves - Hierarchical 5-Class Classifier",
    )

    # Export ROC data as JSON for frontend visualization
    model_dir = PROJECT_ROOT / "models" / "trained" / "hierarchical_5class"
    roc_json_path = model_dir / "roc_data.json"
    export_roc_data(
        y_true_levels=y_true_levels,
        y_proba_levels=y_proba_levels,
        save_path=roc_json_path,
        model_name="hierarchical_5class",
    )

    logger.info(f"ROC curves saved to {roc_path}")
    for level_name, auc in auc_scores.items():
        logger.info(f"  {level_name}: AUC = {auc:.4f}")
        results[f"auc_{level_name}"] = auc

    # Save model (only for tree-based classifiers)
    if model_type == "tree":
        model_dir = PROJECT_ROOT / "models" / "trained" / "hierarchical_5class"
        save_hierarchical_model(clf, str(model_dir))
        logger.info(f"Model saved to {model_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run hierarchical crash severity classification"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["tree", "neural"],
        default="tree",
        help="Model type: 'tree' (XGBoost/RF/ET) or 'neural' (MLP)",
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

    results = run_hierarchical_pipeline(
        model_type=args.model,
        sample_size=args.sample,
        feature_filter=args.feature_filter,
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
