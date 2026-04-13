"""Compare XGBoost vs LightGBM for hierarchical crash severity prediction.

Runs both model configurations on the same train/val/test splits and compares
key metrics to determine which model should be used for final testing.

Usage:
    python training/compare_models.py
    python training/compare_models.py --sample 10000  # Quick test
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

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.feature_engineering import (
    add_binary_targets,
    add_ordinal_severity,
    engineer_all_features,
)
from data_preparation.triple_merge import triple_merge
from training.hierarchical import (
    HierarchicalTargets,
    HierarchicalTreeClassifier,
    TreeHierarchicalConfig,
    evaluate_hierarchical,
    prepare_hierarchical_targets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare feature matrix from DataFrame (copied from main_hierarchical)."""
    exclude_cols = {
        "CRASH_RECORD_ID", "CRASH_DATE", "DATE_POLICE_NOTIFIED", "CRASH_DATE_EST_I",
        "MOST_SEVERE_INJURY", "INJURIES_TOTAL", "INJURIES_FATAL",
        "INJURIES_INCAPACITATING", "INJURIES_NON_INCAPACITATING",
        "INJURIES_REPORTED_NOT_EVIDENT", "INJURIES_NO_INDICATION", "INJURIES_UNKNOWN",
        "IS_INJURY", "IS_SEVERE", "IS_FATAL", "IS_REPORTED",
        "SEVERITY_LEVEL", "SEVERITY_ENCODED",
        "LATITUDE", "LONGITUDE", "LOCATION",
        "STATEMENTS_TAKEN_I", "WORK_ZONE_I", "PHOTOS_TAKEN_I", "DOORING_I",
        "ejected_any", "using_seatbelt_mean", "cell_phone_any", "bac_positive_any", "bac_clean_max",
        "CRASH_TYPE", "REPORT_TYPE", "HOUR", "STREET_NAME",
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

    from sklearn.preprocessing import LabelEncoder
    for col in safe_categorical_cols:
        if col in df.columns and col not in exclude_cols:
            le = LabelEncoder()
            values = df[col].fillna("UNKNOWN").astype(str)
            df_features[col] = le.fit_transform(values)
            feature_cols.append(col)

    feature_cols = list(df_features.columns)
    df_features = df_features.fillna(0)

    return df_features, feature_cols


def run_model(
    model_name: Literal["xgb", "lgbm"],
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    targets_train: HierarchicalTargets,
    targets_val: HierarchicalTargets,
    targets_test: HierarchicalTargets,
    feature_cols: list[str],
    random_state: int = 42,
) -> dict:
    """Train and evaluate a single model configuration.

    Args:
        model_name: Model type to use for all levels.
        X_train: Training features.
        X_val: Validation features.
        X_test: Test features.
        targets_train: Training targets.
        targets_val: Validation targets.
        targets_test: Test targets.
        feature_cols: Feature column names.
        random_state: Random seed.

    Returns:
        Dictionary of evaluation metrics.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {model_name.upper()}")
    logger.info(f"{'='*60}")

    config = TreeHierarchicalConfig(
        l1_model=model_name,
        l2_model=model_name,
        l25_model=model_name,
        l3_model=model_name,
        random_state=random_state,
    )

    clf = HierarchicalTreeClassifier(config)
    clf.feature_cols = feature_cols
    clf.fit(X_train, targets_train)

    # Optimize thresholds on validation set
    clf.optimize_thresholds(
        X_val,
        targets_val,
        l1_target_recall=0.7,
        l25_target_recall=0.4,
        l3_target_recall=0.35,
    )

    # Evaluate on validation set (for model selection)
    logger.info(f"\nEvaluating {model_name.upper()} on validation set...")
    results = evaluate_hierarchical(clf, X_val, targets_val)
    results["model"] = model_name

    return results


def compare_results(xgb_results: dict, lgbm_results: dict) -> str:
    """Compare results and determine winner.

    Args:
        xgb_results: XGBoost evaluation metrics.
        lgbm_results: LightGBM evaluation metrics.

    Returns:
        Name of the winning model ('xgb' or 'lgbm').
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 70)

    # Key metrics to compare (use actual result keys from evaluate_hierarchical)
    key_metrics = [
        ("recall_FATAL", "FATAL Recall", 3.0),  # (metric_key, display_name, weight)
        ("recall_INCAPACITATING INJURY", "INCAP Recall", 2.0),
        ("mc_f1_macro", "Macro F1", 1.5),
        ("mc_accuracy", "Accuracy", 1.0),
    ]

    print(f"\n{'Metric':<25} {'XGBoost':>12} {'LightGBM':>12} {'Winner':>12}")
    print("-" * 65)

    xgb_score = 0.0
    lgbm_score = 0.0

    for metric_key, display_name, weight in key_metrics:
        xgb_val = xgb_results.get(metric_key, 0.0)
        lgbm_val = lgbm_results.get(metric_key, 0.0)

        if xgb_val > lgbm_val:
            winner = "XGBoost"
            xgb_score += weight
        elif lgbm_val > xgb_val:
            winner = "LightGBM"
            lgbm_score += weight
        else:
            winner = "Tie"

        print(f"{display_name:<25} {xgb_val:>12.4f} {lgbm_val:>12.4f} {winner:>12}")

    print("-" * 65)
    print(f"{'Weighted Score':<25} {xgb_score:>12.1f} {lgbm_score:>12.1f}")

    print("\n" + "=" * 70)
    if lgbm_score > xgb_score:
        overall_winner = "lgbm"
        print("WINNER: LightGBM")
        print("Recommendation: Update TreeHierarchicalConfig defaults to 'lgbm'")
    elif xgb_score > lgbm_score:
        overall_winner = "xgb"
        print("WINNER: XGBoost")
        print("Recommendation: Keep XGBoost as default, LightGBM available as option")
    else:
        overall_winner = "xgb"  # Tie goes to incumbent
        print("RESULT: Tie - keeping XGBoost as default")
    print("=" * 70)

    return overall_winner


def main(sample_size: int | None = None) -> str:
    """Run the model comparison.

    Args:
        sample_size: Optional limit on dataset size for faster testing.

    Returns:
        Name of the winning model.
    """
    logger.info("=" * 60)
    logger.info("XGBoost vs LightGBM COMPARISON")
    logger.info("=" * 60)

    random_state = 42

    # Load and prepare data
    logger.info("\n[1/5] Loading and merging data...")
    df = triple_merge()
    logger.info(f"Merged dataset shape: {df.shape}")

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)
        logger.info(f"Sampled to {len(df)} rows")

    # Feature engineering
    logger.info("\n[2/5] Engineering features...")
    df = engineer_all_features(df, include_interactions=True, include_clusters=False)
    df = add_binary_targets(df)
    df = add_ordinal_severity(df)

    # Prepare features
    logger.info("\n[3/5] Preparing feature matrix...")
    X_df, feature_cols = prepare_features(df)
    X = X_df.values

    targets = prepare_hierarchical_targets(df)

    # Split data (same splits for both models)
    logger.info("\n[4/5] Splitting data...")
    train_val_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.2,
        random_state=random_state,
        stratify=targets.y_original,
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        random_state=random_state,
        stratify=targets.y_injury[train_val_idx],
    )

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    def subset_targets(indices: np.ndarray) -> HierarchicalTargets:
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

    # Run both models
    logger.info("\n[5/5] Training and evaluating models...")

    xgb_results = run_model(
        "xgb", X_train, X_val, X_test,
        targets_train, targets_val, targets_test,
        feature_cols, random_state,
    )

    lgbm_results = run_model(
        "lgbm", X_train, X_val, X_test,
        targets_train, targets_val, targets_test,
        feature_cols, random_state,
    )

    # Compare and determine winner
    winner = compare_results(xgb_results, lgbm_results)

    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare XGBoost vs LightGBM for hierarchical classification"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for faster experimentation (default: use all data)",
    )

    args = parser.parse_args()
    winner = main(sample_size=args.sample)
    print(f"\nFinal winner: {winner}")
