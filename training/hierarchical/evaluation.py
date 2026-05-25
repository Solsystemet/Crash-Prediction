"""Evaluation utilities for hierarchical classifiers.

Provides model-agnostic evaluation that works with any HierarchicalClassifierBase.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder

from training.hierarchical.base import HierarchicalClassifierBase
from training.hierarchical.structure import HierarchicalTargets

logger = logging.getLogger(__name__)


def evaluate_hierarchical(
    clf: HierarchicalClassifierBase,
    X_test: np.ndarray,
    targets: HierarchicalTargets,
) -> dict:
    """Evaluate hierarchical classifier performance.

    Args:
        clf: Trained hierarchical classifier (any subclass of base).
        X_test: Test feature matrix.
        targets: HierarchicalTargets for test set.

    Returns:
        Dictionary of evaluation metrics.
    """
    results = {}

    # Get predictions
    l1_pred, l2_pred, l25_pred, l3_pred = clf.predict_binary(X_test)

    # Level 1 evaluation
    results.update(_evaluate_level(
        "L1 (INJURY vs NO_INJURY)",
        targets.y_injury,
        l1_pred,
        positive_name="INJURY",
    ))

    # Level 2 evaluation (injury cases only)
    injury_mask = targets.y_injury == 1
    if np.sum(injury_mask) > 0:
        results.update(_evaluate_level(
            "L2 (SEVERE vs MINOR)",
            targets.y_severe[injury_mask],
            l2_pred[injury_mask],
            positive_name="SEVERE",
            prefix="l2_",
        ))

    # Level 2.5 evaluation (severe cases only)
    severe_mask = (targets.y_injury == 1) & (targets.y_severe == 1)
    if np.sum(severe_mask) > 0:
        results.update(_evaluate_level(
            "L2.5 (FATAL vs INCAPACITATING)",
            targets.y_fatal[severe_mask],
            l25_pred[severe_mask],
            positive_name="FATAL",
            prefix="l25_",
        ))

    # Level 3 evaluation (minor injury cases only)
    minor_mask = (targets.y_injury == 1) & (targets.y_severe == 0)
    if np.sum(minor_mask) > 0:
        results.update(_evaluate_level(
            "L3 (REPORTED vs VISIBLE)",
            targets.y_reported[minor_mask],
            l3_pred[minor_mask],
            positive_name="REPORTED",
            prefix="l3_",
        ))

    # Multiclass evaluation
    logger.info("=" * 60)
    logger.info("MULTICLASS EVALUATION (mapped to original 5 classes)")
    logger.info("=" * 60)

    y_pred_multi = clf.predict_multiclass(X_test, label_encoder=targets.label_encoder)

    mc_acc = accuracy_score(targets.y_original, y_pred_multi)
    mc_f1_macro = f1_score(targets.y_original, y_pred_multi, average="macro", zero_division=0)
    mc_f1_micro = f1_score(targets.y_original, y_pred_multi, average="micro", zero_division=0)
    mc_f1_weighted = f1_score(targets.y_original, y_pred_multi, average="weighted", zero_division=0)

    logger.info(f"Multiclass Accuracy:      {mc_acc:.4f}")
    logger.info(f"Multiclass F1 (macro):    {mc_f1_macro:.4f}")
    logger.info(f"Multiclass F1 (micro):    {mc_f1_micro:.4f}")
    logger.info(f"Multiclass F1 (weighted): {mc_f1_weighted:.4f}")

    results["mc_accuracy"] = mc_acc
    results["mc_f1_macro"] = mc_f1_macro
    results["mc_f1_micro"] = mc_f1_micro
    results["mc_f1_weighted"] = mc_f1_weighted

    # Per-class metrics
    class_names = targets.label_encoder.classes_
    logger.info("\nPer-class Classification Report:")
    report = classification_report(
        targets.y_original,
        y_pred_multi,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )
    logger.info(f"\n{report}")

    # Per-class recall (KEY METRIC)
    logger.info("\n" + "=" * 60)
    logger.info("KEY METRICS: PER-CLASS RECALL")
    logger.info("=" * 60)

    for i, cls in enumerate(class_names):
        cls_pred_correct = np.sum((targets.y_original == i) & (y_pred_multi == i))
        cls_total = np.sum(targets.y_original == i)
        cls_recall = cls_pred_correct / cls_total if cls_total > 0 else 0
        results[f"recall_{cls}"] = cls_recall
        logger.info(f"  {cls}: {cls_recall:.4f} ({cls_pred_correct}/{cls_total})")

    # Confusion matrix
    mc_cm = confusion_matrix(targets.y_original, y_pred_multi)
    logger.info(f"\nMulticlass Confusion Matrix:\n{mc_cm}")

    return results


def _evaluate_level(
    level_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_name: str = "POSITIVE",
    prefix: str = "l1_",
) -> dict:
    """Evaluate a single hierarchy level.

    Args:
        level_name: Display name for logging.
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        positive_name: Name of positive class for logging.
        prefix: Prefix for result dictionary keys.

    Returns:
        Dictionary of level metrics.
    """
    logger.info("=" * 60)
    logger.info(f"{level_name} EVALUATION")
    logger.info("=" * 60)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f} ({positive_name} detection)")
    logger.info(f"F1:        {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    return {
        f"{prefix}accuracy": acc,
        f"{prefix}precision": prec,
        f"{prefix}recall": rec,
        f"{prefix}f1": f1,
    }


def plot_roc_curves(
    y_true_levels: dict[str, np.ndarray],
    y_proba_levels: dict[str, np.ndarray],
    save_path: str | Path,
    title: str = "ROC Curves - Hierarchical Classifier",
) -> dict[str, float]:
    """Plot ROC curves for each level of the hierarchical classifier.

    Args:
        y_true_levels: Dict mapping level name to true binary labels.
            e.g., {"L1 (INJURY)": y_injury, "L2 (SEVERE)": y_severe[mask], ...}
        y_proba_levels: Dict mapping level name to predicted probabilities.
            e.g., {"L1 (INJURY)": l1_proba, "L2 (SEVERE)": l2_proba[mask], ...}
        save_path: Path to save the ROC plot PNG.
        title: Plot title.

    Returns:
        Dictionary of AUC scores per level.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping ROC plot")
        return {}

    auc_scores = {}

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color palette for different levels
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (level_name, y_true) in enumerate(y_true_levels.items()):
        y_proba = y_proba_levels.get(level_name)

        if y_proba is None or len(y_true) == 0:
            logger.warning(f"Skipping ROC for {level_name}: no data")
            continue

        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            logger.warning(f"Skipping ROC for {level_name}: only one class present")
            continue

        # Compute ROC curve and AUC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            auc_scores[level_name] = auc

            # Plot
            color = colors[i % len(colors)]
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{level_name} (AUC = {auc:.3f})")

        except Exception as e:
            logger.warning(f"Error computing ROC for {level_name}: {e}")
            continue

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")

    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"ROC curve saved to {save_path}")

    # Log AUC scores
    for level_name, auc in auc_scores.items():
        logger.info(f"  {level_name} AUC: {auc:.4f}")
        if auc < 0.5:
            logger.warning(f"  WARNING: {level_name} AUC < 0.5, model may be inverted")

    return auc_scores


def _downsample_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    max_points: int = 100,
) -> tuple[list[float], list[float]]:
    """Downsample ROC curve points for efficient JSON export.

    Keeps first, last, and evenly spaced points in between.

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.
        max_points: Maximum number of points to keep.

    Returns:
        Tuple of (fpr_list, tpr_list) downsampled and converted to Python lists.
    """
    n_points = len(fpr)

    if n_points <= max_points:
        return fpr.tolist(), tpr.tolist()

    # Keep evenly spaced indices, always including first and last
    indices = np.linspace(0, n_points - 1, max_points, dtype=int)
    indices = np.unique(indices)  # Remove duplicates

    return fpr[indices].tolist(), tpr[indices].tolist()


def export_roc_data(
    y_true_levels: dict[str, np.ndarray],
    y_proba_levels: dict[str, np.ndarray],
    save_path: str | Path,
    model_name: str = "model",
    max_points: int = 100,
) -> dict:
    """Export ROC curve data as JSON for frontend visualization.

    Args:
        y_true_levels: Dict mapping level name to true binary labels.
        y_proba_levels: Dict mapping level name to predicted probabilities.
        save_path: Path to save the JSON file.
        model_name: Name of the model for metadata.
        max_points: Maximum points per curve (downsampled for performance).

    Returns:
        The exported data dictionary.
    """
    curves = []
    colors = ["#3b82f6", "#f97316", "#22c55e", "#ef4444"]  # Tailwind blue, orange, green, red

    for i, (level_name, y_true) in enumerate(y_true_levels.items()):
        y_proba = y_proba_levels.get(level_name)

        if y_proba is None or len(y_true) == 0:
            continue

        if len(np.unique(y_true)) < 2:
            continue

        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)

            # Downsample for JSON export
            fpr_list, tpr_list = _downsample_curve(fpr, tpr, max_points)

            curves.append({
                "name": level_name,
                "auc": round(auc, 4),
                "color": colors[i % len(colors)],
                "fpr": fpr_list,
                "tpr": tpr_list,
                "n_samples": int(len(y_true)),
                "n_positive": int(np.sum(y_true)),
            })

        except Exception as e:
            logger.warning(f"Error computing ROC for {level_name}: {e}")
            continue

    data = {
        "model_name": model_name,
        "generated_at": datetime.now().isoformat(),
        "curves": curves,
    }

    # Save JSON
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"ROC data exported to {save_path}")

    return data
<<<<<<< HEAD


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot confusion matrix as a heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Path to save the plot.
        title: Plot title.
        normalize: Whether to normalize by row (true class).
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available, skipping confusion matrix plot")
        return

    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Confusion matrix heatmap saved to {save_path}")


def plot_per_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    title: str = "Per-Class Recall",
) -> dict[str, float]:
    """Plot per-class recall as a horizontal bar chart.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        save_path: Path to save the plot.
        title: Plot title.

    Returns:
        Dictionary mapping class name to recall value.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping recall plot")
        return {}

    recalls = {}
    for i, cls_name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            correct = ((y_true == i) & (y_pred == i)).sum()
            recalls[cls_name] = correct / mask.sum()
        else:
            recalls[cls_name] = 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(class_names))
    recall_values = [recalls[cls] for cls in class_names]
    
    colors = ["#22c55e" if r >= 0.5 else "#f97316" if r >= 0.3 else "#ef4444" for r in recall_values]
    
    bars = ax.barh(y_pos, recall_values, color=colors, edgecolor="black", alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, recall_values):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=10,
        )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1.1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="50% threshold")
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Per-class recall plot saved to {save_path}")

    return recalls


def generate_all_evaluation_plots(
    clf: HierarchicalClassifierBase,
    X_test: np.ndarray,
    targets: HierarchicalTargets,
    output_dir: str | Path,
    model_name: str = "hierarchical",
) -> dict:
    """Generate all evaluation plots for a hierarchical classifier.

    This is the main entry point for comprehensive evaluation visualization.
    Generates:
    - ROC curves for each level
    - Confusion matrix heatmap
    - Per-class recall bar chart
    - Calibration curves (if probability estimates available)
    - Confidence distribution histogram

    Args:
        clf: Trained hierarchical classifier.
        X_test: Test feature matrix.
        targets: HierarchicalTargets for test set.
        output_dir: Directory to save all plots.
        model_name: Name of the model for titles/filenames.

    Returns:
        Dictionary of generated file paths and metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"plots": [], "metrics": {}}
    
    logger.info(f"Generating evaluation plots in {output_dir}")

    # Get predictions
    l1_pred, l2_pred, l25_pred, l3_pred = clf.predict_binary(X_test)
    l1_proba, l2_proba, l25_proba, l3_proba = clf.predict_proba(X_test)
    y_pred_multi = clf.predict_multiclass(X_test, label_encoder=targets.label_encoder)

    class_names = list(targets.label_encoder.classes_)

    # 1. ROC Curves
    injury_mask = targets.y_injury == 1
    severe_mask = (targets.y_injury == 1) & (targets.y_severe == 1)
    minor_mask = (targets.y_injury == 1) & (targets.y_severe == 0)

    y_true_levels = {"L1 (INJURY)": targets.y_injury}
    y_proba_levels = {"L1 (INJURY)": l1_proba}

    if np.sum(injury_mask) > 0:
        y_true_levels["L2 (SEVERE)"] = targets.y_severe[injury_mask]
        y_proba_levels["L2 (SEVERE)"] = l2_proba[injury_mask]

    if np.sum(severe_mask) > 0:
        y_true_levels["L2.5 (FATAL)"] = targets.y_fatal[severe_mask]
        y_proba_levels["L2.5 (FATAL)"] = l25_proba[severe_mask]

    if np.sum(minor_mask) > 0:
        y_true_levels["L3 (REPORTED)"] = targets.y_reported[minor_mask]
        y_proba_levels["L3 (REPORTED)"] = l3_proba[minor_mask]

    roc_path = output_dir / f"{model_name}_roc_curves.png"
    auc_scores = plot_roc_curves(
        y_true_levels, y_proba_levels, roc_path,
        title=f"ROC Curves - {model_name}",
    )
    results["plots"].append(str(roc_path))
    results["metrics"]["auc_scores"] = auc_scores

    # Export ROC data for frontend
    roc_json_path = output_dir / f"{model_name}_roc_data.json"
    export_roc_data(
        y_true_levels, y_proba_levels, roc_json_path,
        model_name=model_name,
    )
    results["plots"].append(str(roc_json_path))

    # 2. Confusion Matrix
    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix_heatmap(
        targets.y_original, y_pred_multi, class_names, cm_path,
        title=f"Confusion Matrix - {model_name}",
    )
    results["plots"].append(str(cm_path))

    # 3. Per-class Recall
    recall_path = output_dir / f"{model_name}_per_class_recall.png"
    recalls = plot_per_class_recall(
        targets.y_original, y_pred_multi, class_names, recall_path,
        title=f"Per-Class Recall - {model_name}",
    )
    results["metrics"]["recalls"] = recalls
    results["plots"].append(str(recall_path))

    # 4. Calibration Curves (using plotting module)
    try:
        from training.plotting.calibration import plot_calibration_curve
        
        # L1 calibration (most important - injury detection)
        cal_path = output_dir / f"{model_name}_calibration_L1.png"
        plot_calibration_curve(
            targets.y_injury, l1_proba, cal_path,
            title=f"Calibration - L1 Injury Detection ({model_name})",
        )
        results["plots"].append(str(cal_path))

        # L2 calibration (severity)
        if np.sum(injury_mask) > 0:
            cal_path_l2 = output_dir / f"{model_name}_calibration_L2.png"
            plot_calibration_curve(
                targets.y_severe[injury_mask], l2_proba[injury_mask], cal_path_l2,
                title=f"Calibration - L2 Severity ({model_name})",
            )
            results["plots"].append(str(cal_path_l2))
    except ImportError:
        logger.warning("Calibration plotting module not available")

    # 5. Confidence Distribution
    try:
        from training.plotting.confidence import plot_confidence_histogram
        
        # Get max probability for each prediction
        all_proba = np.column_stack([
            1 - l1_proba,  # NO_INJURY prob
            l1_proba * (1 - l2_proba),  # MINOR prob (simplified)
            l1_proba * l2_proba,  # SEVERE prob (simplified)
        ])
        max_conf = np.max(all_proba, axis=1)
        
        conf_path = output_dir / f"{model_name}_confidence_distribution.png"
        plot_confidence_histogram(
            max_conf, conf_path,
            title=f"Prediction Confidence Distribution - {model_name}",
        )
        results["plots"].append(str(conf_path))
    except ImportError:
        logger.warning("Confidence plotting module not available")

    # 6. Export metrics summary as JSON
    metrics_path = output_dir / f"{model_name}_metrics_summary.json"
    metrics_summary = {
        "model_name": model_name,
        "generated_at": datetime.now().isoformat(),
        "auc_scores": {k: round(v, 4) for k, v in auc_scores.items()},
        "recalls": {k: round(v, 4) for k, v in recalls.items()},
        "n_test_samples": len(targets.y_original),
        "class_distribution": {
            cls: int((targets.y_original == i).sum())
            for i, cls in enumerate(class_names)
        },
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    results["plots"].append(str(metrics_path))

    logger.info(f"Generated {len(results['plots'])} evaluation artifacts in {output_dir}")

    return results
=======
>>>>>>> origin/dev
