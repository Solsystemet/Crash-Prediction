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
<<<<<<< HEAD
from data_preparation.triple_merge import triple_merge, DataSourceConfig
=======
from data_preparation.triple_merge import triple_merge
>>>>>>> origin/dev

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
from training.hierarchical.evaluation import plot_roc_curves, export_roc_data
from training.feature_selection import filter_by_importance
from training.baselines import (
    ClassificationBaseline,
    compare_to_baseline,
    log_comparison,
<<<<<<< HEAD
    print_dual_baseline_comparison,
    add_baseline_args,
    create_imbalance_baselines,
)
from training.metrics_schema import export_model_vs_baselines_csv
from utils.logging_config import setup_logging

logger = setup_logging(__name__)

# Base model name for output directory
MODEL_NAME = "simplified_3class"


def get_output_dir(config: DataSourceConfig) -> Path:
    """Get output directory based on data source configuration.
    
    Args:
        config: Data source configuration specifying which datasets are included.
        
    Returns:
        Path to model output directory.
    """
    return PROJECT_ROOT / "models" / "trained" / f"{MODEL_NAME}_{config.get_name_suffix()}"
=======
    print_baseline_comparison_box,
    add_baseline_args,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
>>>>>>> origin/dev


def stratified_sample_severe(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample dataset ensuring all classes are represented.

    Args:
        df: Full DataFrame with MOST_SEVERE_INJURY column.
        sample_size: Target sample size.
        random_state: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with all classes represented.
    """
    # Separate SEVERE (FATAL + INCAPACITATING) from rest
    severe_mask = df["MOST_SEVERE_INJURY"].isin(["FATAL", "INCAPACITATING INJURY"])
    severe_df = df[severe_mask]
    other_df = df[~severe_mask]

    n_severe = len(severe_df)
    n_other = len(other_df)
    logger.info(f"  SEVERE cases in full data: {n_severe}")
    logger.info(f"  Other cases in full data: {n_other}")

    # Always include a mix of classes
    # Target: up to 50% SEVERE, at least 50% other (to ensure class diversity)
    max_severe = min(n_severe, sample_size // 2)
    n_other_needed = sample_size - max_severe

    # Sample from each group
    if n_severe > max_severe:
        severe_sample = severe_df.sample(n=max_severe, random_state=random_state)
    else:
        severe_sample = severe_df

    if n_other > n_other_needed:
        other_sample = other_df.sample(n=n_other_needed, random_state=random_state)
    else:
        other_sample = other_df
        # If we don't have enough other, take more severe
        extra_severe_needed = n_other_needed - len(other_sample)
        remaining_severe = severe_df.drop(severe_sample.index)
        if len(remaining_severe) >= extra_severe_needed:
            extra_severe = remaining_severe.sample(n=extra_severe_needed, random_state=random_state)
            severe_sample = pd.concat([severe_sample, extra_severe], ignore_index=True)

    result = pd.concat([severe_sample, other_sample], ignore_index=True)
    logger.info(f"  Stratified sample: {len(severe_sample)} SEVERE + {len(other_sample)} other = {len(result)}")

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
    f1_micro = f1_score(targets.y_simplified, y_pred, average="micro", zero_division=0)
    f1_weighted = f1_score(targets.y_simplified, y_pred, average="weighted", zero_division=0)

    logger.info(f"Accuracy:      {accuracy:.4f}")
    logger.info(f"F1 (macro):    {f1_macro:.4f}")
    logger.info(f"F1 (micro):    {f1_micro:.4f}")
    logger.info(f"F1 (weighted): {f1_weighted:.4f}")

    results["accuracy"] = accuracy
    results["f1_macro"] = f1_macro
    results["f1_micro"] = f1_micro
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
    l2_weight_multiplier: float | None = None,
    l2_resampling_method: str | None = None,
    l2_target_recall: float | None = None,
    with_baseline: bool = True,
<<<<<<< HEAD
    data_config: DataSourceConfig | None = None,
=======
>>>>>>> origin/dev
) -> dict:
    """Run the simplified 3-class classification pipeline.

    Args:
        sample_size: Optional limit on dataset size for faster experimentation.
        feature_filter: Feature filtering mode ("none" or "drop-low").
        l2_weight_multiplier: Override for L2 class weight multiplier.
        l2_resampling_method: Override for L2 resampling method ('borderline' or 'adasyn').
        l2_target_recall: Override for L2 target recall during threshold optimization.
<<<<<<< HEAD
        data_config: Data source configuration (default: crash only).
=======
>>>>>>> origin/dev

    Returns:
        Dictionary of evaluation results.
    """
<<<<<<< HEAD
    if data_config is None:
        data_config = DataSourceConfig(use_vehicles=False, use_people=False, use_weather=False)
    
    output_dir = get_output_dir(data_config)
    
    logger.info("=" * 60)
    logger.info("SIMPLIFIED 3-CLASS CLASSIFICATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Data configuration: {data_config}")
    logger.info(f"Output directory: {output_dir}")
=======
    logger.info("=" * 60)
    logger.info("SIMPLIFIED 3-CLASS CLASSIFICATION PIPELINE")
    logger.info("=" * 60)
>>>>>>> origin/dev

    config = SimplifiedTreeConfig(sample_size=sample_size)

    # Apply CLI overrides to config
    if l2_weight_multiplier is not None:
        config.l2_weight_multiplier = l2_weight_multiplier
    if l2_resampling_method is not None:
        config.l2_resampling_method = l2_resampling_method
    if l2_target_recall is not None:
        config.l2_target_recall = l2_target_recall

    logger.info(f"Configuration: {config}")

    # Load and merge data
    logger.info("\n[1/6] Loading and merging data...")
<<<<<<< HEAD
    df = triple_merge(config=data_config, verbose=False)
=======
    df = triple_merge()
>>>>>>> origin/dev
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
        l2_resampling_method=config.l2_resampling_method,
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
        l2_target_recall=config.l2_target_recall,
    )

    # Evaluate
    logger.info("\n[6/6] Evaluating on test set...")
    results = evaluate_simplified(clf, X_test, targets_test)

<<<<<<< HEAD
    # Baseline evaluation (coin flip baselines for imbalanced data)
    if with_baseline:
        baseline = create_imbalance_baselines(random_state=config.random_state)
=======
    # Baseline evaluation
    if with_baseline:
        baseline = ClassificationBaseline(
            strategies=["most_frequent", "stratified"],
            random_state=config.random_state,
        )
>>>>>>> origin/dev
        baseline.fit(targets_train.y_simplified)
        baseline_results = baseline.evaluate(
            targets_test.y_simplified,
            class_names=SIMPLIFIED_CLASS_NAMES,
        )

<<<<<<< HEAD
        # Compare model to both baselines
=======
        # Compare model to baseline
>>>>>>> origin/dev
        model_metrics = {
            "accuracy": results["accuracy"],
            "f1_macro": results["f1_macro"],
            "f1_weighted": results["f1_weighted"],
            "recall_SEVERE": results.get("recall_SEVERE", 0),
        }
        comparison = compare_to_baseline(model_metrics, baseline_results)

<<<<<<< HEAD
        # Print comparison against both baselines
        print_dual_baseline_comparison(
            model_metrics=model_metrics,
            baseline_results=baseline_results,
            model_name="Simplified",
            primary_metric="recall_SEVERE",
=======
        # Print comparison box (use stratified baseline since we stratified-sample the training data)
        print_baseline_comparison_box(
            model_metrics=model_metrics,
            baseline_results=baseline_results,
            comparison=comparison,
            model_name="Simplified Classifier",
            baseline_strategy="stratified",
            primary_metric="recall_SEVERE",
            metric_labels={
                "accuracy": "Accuracy",
                "f1_macro": "F1 Macro",
                "f1_weighted": "F1 Weighted",
                "recall_SEVERE": "SEVERE Recall",
            },
>>>>>>> origin/dev
        )

        # Store baseline results
        results["baseline"] = {
            strategy: result.metrics
            for strategy, result in baseline_results.items()
        }
        results["baseline_comparison"] = comparison

<<<<<<< HEAD
        # Export baseline comparison CSV
        export_model_vs_baselines_csv(
            model_name="simplified_3class",
            model_metrics=model_metrics,
            baseline_results=baseline_results,
            output_dir=output_dir,
        )
        logger.info(f"Saved timestamped baseline comparison to {output_dir}")

=======
>>>>>>> origin/dev
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"3-Class Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Macro F1: {results['f1_macro']:.4f}")
    logger.info(f"Micro F1: {results['f1_micro']:.4f}")
    for class_name in SIMPLIFIED_CLASS_NAMES:
        key = f"recall_{class_name}"
        if key in results:
            logger.info(f"  {class_name} Recall: {results[key]:.4f}")

    # Generate ROC curves
    logger.info("\nGenerating ROC curves...")
    plots_dir = PROJECT_ROOT / "models" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get probabilities for ROC
    l1_proba, l2_proba = clf.predict_proba(X_test)
    injury_mask = targets_test.y_injury == 1

    y_true_levels = {
        "L1 (INJURY vs NO_INJURY)": targets_test.y_injury,
    }
    y_proba_levels = {
        "L1 (INJURY vs NO_INJURY)": l1_proba,
    }

    # Add L2 if we have injury cases
    if np.sum(injury_mask) > 0:
        y_true_levels["L2 (SEVERE vs MINOR)"] = targets_test.y_severe[injury_mask]
        y_proba_levels["L2 (SEVERE vs MINOR)"] = l2_proba[injury_mask]

    roc_path = plots_dir / "roc_curves_simplified.png"
    auc_scores = plot_roc_curves(
        y_true_levels=y_true_levels,
        y_proba_levels=y_proba_levels,
        save_path=roc_path,
        title="ROC Curves - Simplified 3-Class Classifier",
    )

    # Export ROC data as JSON for frontend visualization
<<<<<<< HEAD
    roc_json_path = output_dir / "roc_data.json"
=======
    model_dir = PROJECT_ROOT / "models" / "trained" / "simplified_3class"
    roc_json_path = model_dir / "roc_data.json"
>>>>>>> origin/dev
    export_roc_data(
        y_true_levels=y_true_levels,
        y_proba_levels=y_proba_levels,
        save_path=roc_json_path,
        model_name="simplified_3class",
    )

    logger.info(f"ROC curves saved to {roc_path}")
    for level_name, auc in auc_scores.items():
        logger.info(f"  {level_name}: AUC = {auc:.4f}")
        results[f"auc_{level_name}"] = auc

<<<<<<< HEAD
    # Generate additional evaluation plots
    logger.info("\nGenerating additional evaluation plots...")
    
    # Confusion matrix heatmap
    try:
        from training.plotting.calibration import plot_calibration_curve
        from training.plotting.confidence import plot_confidence_histogram
        
        # Confusion matrix heatmap
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(targets_test.y_simplified, y_pred)
            cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=SIMPLIFIED_CLASS_NAMES,
                yticklabels=SIMPLIFIED_CLASS_NAMES,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix - Simplified 3-Class")
            plt.tight_layout()
            cm_path = output_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Confusion matrix saved to {cm_path}")
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix plot: {e}")
        
        # Calibration curve for L1 (injury detection)
        cal_path = output_dir / "calibration_L1.png"
        plot_calibration_curve(
            targets_test.y_injury, l1_proba, cal_path,
            title="Calibration - L1 Injury Detection",
        )
        logger.info(f"Calibration curve (L1) saved to {cal_path}")
        
        # Calibration curve for L2 (severity) if we have injury cases
        if np.sum(injury_mask) > 0:
            cal_path_l2 = output_dir / "calibration_L2.png"
            plot_calibration_curve(
                targets_test.y_severe[injury_mask], l2_proba[injury_mask], cal_path_l2,
                title="Calibration - L2 Severity",
            )
            logger.info(f"Calibration curve (L2) saved to {cal_path_l2}")
        
        # Prediction confidence distribution
        # Calculate 3-class probabilities
        p_no_injury = 1 - l1_proba
        p_minor = l1_proba * (1 - l2_proba)
        p_severe = l1_proba * l2_proba
        all_proba = np.column_stack([p_no_injury, p_minor, p_severe])
        max_conf = np.max(all_proba, axis=1)
        
        conf_path = output_dir / "confidence_distribution.png"
        plot_confidence_histogram(
            max_conf, conf_path,
            title="Prediction Confidence Distribution",
        )
        logger.info(f"Confidence distribution saved to {conf_path}")
        
        # Per-class recall bar chart
        try:
            y_pred = clf.predict(X_test)
            recalls = {}
            for i, cls_name in enumerate(SIMPLIFIED_CLASS_NAMES):
                mask = targets_test.y_simplified == i
                if mask.sum() > 0:
                    correct = ((targets_test.y_simplified == i) & (y_pred == i)).sum()
                    recalls[cls_name] = correct / mask.sum()
                else:
                    recalls[cls_name] = 0.0
            
            fig, ax = plt.subplots(figsize=(8, 5))
            y_pos = np.arange(len(SIMPLIFIED_CLASS_NAMES))
            recall_values = [recalls[cls] for cls in SIMPLIFIED_CLASS_NAMES]
            colors = ["#22c55e" if r >= 0.5 else "#f97316" if r >= 0.3 else "#ef4444" for r in recall_values]
            
            bars = ax.barh(y_pos, recall_values, color=colors, edgecolor="black", alpha=0.8)
            for bar, val in zip(bars, recall_values):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center")
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(SIMPLIFIED_CLASS_NAMES)
            ax.set_xlabel("Recall")
            ax.set_title("Per-Class Recall - Simplified 3-Class")
            ax.set_xlim(0, 1.1)
            ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
            plt.tight_layout()
            
            recall_path = output_dir / "per_class_recall.png"
            plt.savefig(recall_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Per-class recall plot saved to {recall_path}")
        except Exception as e:
            logger.warning(f"Could not generate recall plot: {e}")
            
    except ImportError as e:
        logger.warning(f"Could not import plotting modules: {e}")

    # Save model
    save_simplified_model(clf, str(output_dir))
    logger.info(f"Model saved to {output_dir}")
=======
    # Save model
    model_dir = PROJECT_ROOT / "models" / "trained" / "simplified_3class"
    save_simplified_model(clf, str(model_dir))
    logger.info(f"Model saved to {model_dir}")
>>>>>>> origin/dev

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
    # L2 tuning options (same as main_simplified_zones.py)
    parser.add_argument(
        "--l2-weight",
        type=float,
        default=None,
        help="Class weight multiplier for severe class (default: 3.0). "
             "Higher values prioritize severe recall over precision.",
    )
    parser.add_argument(
        "--l2-resampling",
        type=str,
        choices=["borderline", "adasyn"],
        default=None,
        help="Resampling method for L2 classifier (default: adasyn). "
             "ADASYN focuses on harder-to-learn samples.",
    )
    parser.add_argument(
        "--l2-target-recall",
        type=float,
        default=None,
        help="Target recall for L2 threshold optimization (default: 0.6). "
             "Higher values catch more severe cases but increase false positives.",
    )
    add_baseline_args(parser)
<<<<<<< HEAD
    # Data source configuration flags
    parser.add_argument(
        "--include-vehicle",
        action="store_true",
        help="Include vehicle data (count, age, types, speed violations)",
    )
    parser.add_argument(
        "--include-people",
        action="store_true",
        help="Include people data (demographics, BAC, safety equipment)",
    )
    parser.add_argument(
        "--include-weather",
        action="store_true",
        help="Include weather data (temperature, humidity, rain, wind)",
    )

    args = parser.parse_args()
    
    # Build data source configuration from CLI flags
    data_config = DataSourceConfig(
        use_vehicles=args.include_vehicle,
        use_people=args.include_people,
        use_weather=args.include_weather,
    )
=======

    args = parser.parse_args()
>>>>>>> origin/dev

    results = run_simplified_pipeline(
        sample_size=args.sample,
        feature_filter=args.feature_filter,
        l2_weight_multiplier=args.l2_weight,
        l2_resampling_method=args.l2_resampling,
        l2_target_recall=args.l2_target_recall,
        with_baseline=not args.no_baseline,
<<<<<<< HEAD
        data_config=data_config,
=======
>>>>>>> origin/dev
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif key != "confusion_matrix":
            print(f"  {key}: {value}")
