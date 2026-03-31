"""Evaluation utilities for hierarchical classifiers.

Provides model-agnostic evaluation that works with any HierarchicalClassifierBase.
"""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
    mc_f1_weighted = f1_score(targets.y_original, y_pred_multi, average="weighted", zero_division=0)

    logger.info(f"Multiclass Accuracy:      {mc_acc:.4f}")
    logger.info(f"Multiclass F1 (macro):    {mc_f1_macro:.4f}")
    logger.info(f"Multiclass F1 (weighted): {mc_f1_weighted:.4f}")

    results["mc_accuracy"] = mc_acc
    results["mc_f1_macro"] = mc_f1_macro
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
