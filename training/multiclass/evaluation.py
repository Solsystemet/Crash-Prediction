"""Evaluation metrics and visualization for multiclass classification.

This module provides comprehensive evaluation utilities for assessing
the performance of multiclass crash severity models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# Default class names for 5-class severity prediction
DEFAULT_CLASS_NAMES = [
    "NO_INJURY",
    "REPORTED_NOT_EVIDENT",
    "NON_INCAPACITATING",
    "INCAPACITATING",
    "FATAL",
]


@dataclass
class ClassMetrics:
    """Metrics for a single class.

    Attributes:
        name: Class name.
        precision: Precision score.
        recall: Recall score.
        f1: F1 score.
        support: Number of true samples for this class.
        roc_auc: ROC AUC score (one-vs-rest).
    """

    name: str
    precision: float
    recall: float
    f1: float
    support: int
    roc_auc: float | None = None


@dataclass
class MulticlassEvaluationResult:
    """Complete evaluation results for multiclass classification.

    Attributes:
        accuracy: Overall accuracy.
        macro_f1: Macro-averaged F1 score.
        micro_f1: Micro-averaged F1 score.
        weighted_f1: Weighted F1 score.
        macro_precision: Macro-averaged precision.
        macro_recall: Macro-averaged recall.
        per_class_metrics: Metrics for each class.
        confusion_matrix: Confusion matrix as 2D array.
        classification_report_str: Formatted classification report.
        class_names: Names of classes.
        roc_auc_macro: Macro-averaged ROC AUC (if probabilities provided).
        roc_auc_weighted: Weighted ROC AUC (if probabilities provided).
    """

    accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class_metrics: list[ClassMetrics]
    confusion_matrix: NDArray[np.integer[Any]]
    classification_report_str: str
    class_names: list[str]
    roc_auc_macro: float | None = None
    roc_auc_weighted: float | None = None


def evaluate_multiclass(
    y_true: NDArray[np.integer[Any]],
    y_pred: NDArray[np.integer[Any]],
    y_proba: NDArray[np.floating[Any]] | None = None,
    class_names: list[str] | None = None,
) -> MulticlassEvaluationResult:
    """Evaluate multiclass classification performance.

    Args:
        y_true: Ground truth labels of shape (n_samples,).
        y_pred: Predicted labels of shape (n_samples,).
        y_proba: Predicted probabilities of shape (n_samples, n_classes).
            Required for ROC AUC computation.
        class_names: Names for each class. Uses defaults if None.

    Returns:
        MulticlassEvaluationResult with comprehensive metrics.
    """
    num_classes = len(np.unique(y_true))
    class_names = class_names or DEFAULT_CLASS_NAMES[:num_classes]

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    per_class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Support (count per class)
    _, counts = np.unique(y_true, return_counts=True)

    # ROC AUC (one-vs-rest) if probabilities provided
    per_class_roc_auc: list[float | None] = [None] * num_classes
    roc_auc_macro = None
    roc_auc_weighted = None

    if y_proba is not None:
        try:
            roc_auc_macro = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
            roc_auc_weighted = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"
            )

            # Per-class ROC AUC
            for cls in range(num_classes):
                if np.sum(y_true == cls) > 0:
                    y_true_binary = (y_true == cls).astype(int)
                    per_class_roc_auc[cls] = roc_auc_score(
                        y_true_binary, y_proba[:, cls]
                    )
        except ValueError as e:
            logger.warning(f"Could not compute ROC AUC: {e}")

    # Build per-class metrics
    per_class_metrics = []
    for i, name in enumerate(class_names):
        metrics = ClassMetrics(
            name=name,
            precision=float(per_class_precision[i]),
            recall=float(per_class_recall[i]),
            f1=float(per_class_f1[i]),
            support=int(counts[i]) if i < len(counts) else 0,
            roc_auc=per_class_roc_auc[i],
        )
        per_class_metrics.append(metrics)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    return MulticlassEvaluationResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        per_class_metrics=per_class_metrics,
        confusion_matrix=cm,
        classification_report_str=report,
        class_names=class_names,
        roc_auc_macro=roc_auc_macro,
        roc_auc_weighted=roc_auc_weighted,
    )


def print_evaluation_summary(result: MulticlassEvaluationResult) -> None:
    """Print a formatted summary of evaluation results.

    Args:
        result: Evaluation result to print.
    """
    print("\n" + "=" * 60)
    print("MULTICLASS CLASSIFICATION EVALUATION")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:         {result.accuracy:.4f}")
    print(f"  Macro F1:         {result.macro_f1:.4f}")
    print(f"  Micro F1:         {result.micro_f1:.4f}")
    print(f"  Weighted F1:      {result.weighted_f1:.4f}")
    print(f"  Macro Precision:  {result.macro_precision:.4f}")
    print(f"  Macro Recall:     {result.macro_recall:.4f}")

    if result.roc_auc_macro is not None:
        print(f"  ROC AUC (macro):  {result.roc_auc_macro:.4f}")
        print(f"  ROC AUC (weighted): {result.roc_auc_weighted:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"  {'Class':<22} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8} {'AUC':>8}")
    print("  " + "-" * 62)

    for metrics in result.per_class_metrics:
        auc_str = f"{metrics.roc_auc:.4f}" if metrics.roc_auc is not None else "N/A"
        print(
            f"  {metrics.name:<22} {metrics.precision:>8.4f} {metrics.recall:>8.4f} "
            f"{metrics.f1:>8.4f} {metrics.support:>8d} {auc_str:>8}"
        )

    print(f"\nConfusion Matrix:")
    print(f"  Predicted →")
    print(f"  True ↓")

    # Print header
    header = "  " + " " * 12
    for name in result.class_names:
        header += f"{name[:8]:>10}"
    print(header)

    # Print matrix rows
    for i, row in enumerate(result.confusion_matrix):
        row_str = f"  {result.class_names[i][:10]:<10}"
        for val in row:
            row_str += f"{val:>10d}"
        print(row_str)

    print("\n" + "=" * 60)


def get_roc_curve_data(
    y_true: NDArray[np.integer[Any]],
    y_proba: NDArray[np.floating[Any]],
    class_names: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Compute ROC curve data for each class (one-vs-rest).

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities.
        class_names: Names for each class.

    Returns:
        Dict mapping class name to dict with 'fpr', 'tpr', 'thresholds'.
    """
    num_classes = y_proba.shape[1]
    class_names = class_names or DEFAULT_CLASS_NAMES[:num_classes]

    roc_data = {}

    for cls in range(num_classes):
        y_true_binary = (y_true == cls).astype(int)

        if np.sum(y_true_binary) == 0:
            continue

        fpr, tpr, thresholds = roc_curve(y_true_binary, y_proba[:, cls])

        roc_data[class_names[cls]] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    return roc_data


def get_pr_curve_data(
    y_true: NDArray[np.integer[Any]],
    y_proba: NDArray[np.floating[Any]],
    class_names: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Compute precision-recall curve data for each class (one-vs-rest).

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities.
        class_names: Names for each class.

    Returns:
        Dict mapping class name to dict with 'precision', 'recall', 'thresholds'.
    """
    num_classes = y_proba.shape[1]
    class_names = class_names or DEFAULT_CLASS_NAMES[:num_classes]

    pr_data = {}

    for cls in range(num_classes):
        y_true_binary = (y_true == cls).astype(int)

        if np.sum(y_true_binary) == 0:
            continue

        precision, recall, thresholds = precision_recall_curve(
            y_true_binary, y_proba[:, cls]
        )

        pr_data[class_names[cls]] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
        }

    return pr_data


def compare_with_baseline(
    y_true: NDArray[np.integer[Any]],
    model_predictions: NDArray[np.integer[Any]],
    model_name: str = "Multiclass NN",
) -> dict[str, dict[str, float]]:
    """Compare model performance against simple baselines.

    Args:
        y_true: Ground truth labels.
        model_predictions: Model's predicted labels.
        model_name: Name for the model in output.

    Returns:
        Dict with metrics for model and baselines.
    """
    from sklearn.dummy import DummyClassifier

    results = {}

    # Model metrics
    results[model_name] = {
        "accuracy": accuracy_score(y_true, model_predictions),
        "macro_f1": f1_score(y_true, model_predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, model_predictions, average="weighted", zero_division=0),
    }

    # Baseline: most frequent
    dummy_mf = DummyClassifier(strategy="most_frequent")
    dummy_mf.fit(y_true, y_true)  # Just needs target to learn most frequent
    pred_mf = dummy_mf.predict(y_true)

    results["Most Frequent"] = {
        "accuracy": accuracy_score(y_true, pred_mf),
        "macro_f1": f1_score(y_true, pred_mf, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred_mf, average="weighted", zero_division=0),
    }

    # Baseline: stratified
    dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
    dummy_strat.fit(y_true, y_true)
    pred_strat = dummy_strat.predict(y_true)

    results["Stratified"] = {
        "accuracy": accuracy_score(y_true, pred_strat),
        "macro_f1": f1_score(y_true, pred_strat, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred_strat, average="weighted", zero_division=0),
    }

    return results
