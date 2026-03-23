"""Evaluation utilities for model comparison.

This module provides functions for computing metrics and comparing models.
Enhanced with AUC-ROC, per-class recall, and confusion matrix visualization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_preparation.tensor_dataset import CrashTensorDataset
from training.models import CrashPredictionMLP


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics.

    Attributes:
        accuracy: Overall accuracy.
        precision: Macro-averaged precision.
        recall: Macro-averaged recall.
        f1: Macro-averaged F1 score.
        predictions: Predicted labels.
        true_labels: Ground truth labels.
        probabilities: Predicted probabilities (for AUC-ROC).
        auc_roc: AUC-ROC score (macro, one-vs-rest).
        per_class_recall: Recall for each class.
        confusion_mat: Confusion matrix.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    predictions: NDArray[np.int64]
    true_labels: NDArray[np.int64]
    probabilities: NDArray[np.float64] | None = None
    auc_roc: float | None = None
    per_class_recall: dict[int, float] = field(default_factory=dict)
    confusion_mat: NDArray[np.int64] | None = None


def evaluate_neural_network(
    model: CrashPredictionMLP,
    dataset: CrashTensorDataset,
    device: str = "cpu",
) -> EvaluationMetrics:
    """Evaluate a neural network on a dataset.

    Args:
        model: Trained neural network.
        dataset: Dataset to evaluate on.
        device: Device to run evaluation on.

    Returns:
        EvaluationMetrics containing results.
    """
    model.eval()
    model.to(device)

    features = dataset.features.to(device)
    true_labels = dataset.labels.numpy()

    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        _, predictions = outputs.max(1)
        predictions = predictions.cpu().numpy()

    return _compute_metrics(predictions, true_labels, probabilities)


def evaluate_random_forest(
    model: RandomForestClassifier,
    dataset: CrashTensorDataset,
) -> EvaluationMetrics:
    """Evaluate a Random Forest on a dataset.

    Args:
        model: Trained Random Forest.
        dataset: Dataset to evaluate on.

    Returns:
        EvaluationMetrics containing results.
    """
    X = dataset.features.numpy()
    true_labels = dataset.labels.numpy()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return _compute_metrics(predictions, true_labels, probabilities)


def evaluate_extra_trees(
    model: ExtraTreesClassifier,
    dataset: CrashTensorDataset,
) -> EvaluationMetrics:
    """Evaluate an Extra Trees model on a dataset.

    Args:
        model: Trained Extra Trees classifier.
        dataset: Dataset to evaluate on.

    Returns:
        EvaluationMetrics containing results.
    """
    X = dataset.features.numpy()
    true_labels = dataset.labels.numpy()
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    return _compute_metrics(predictions, true_labels, probabilities)


def evaluate_xgboost(
    model: Any,  # xgb.XGBClassifier
    dataset: CrashTensorDataset,
) -> EvaluationMetrics:
    """Evaluate an XGBoost model on a dataset.

    Args:
        model: Trained XGBoost classifier.
        dataset: Dataset to evaluate on.

    Returns:
        EvaluationMetrics containing results.
    """
    X = dataset.features.numpy()
    true_labels = dataset.labels.numpy()
    predictions = model.predict(X).astype(np.int64)
    probabilities = model.predict_proba(X)

    return _compute_metrics(predictions, true_labels, probabilities)


def _compute_metrics(
    predictions: NDArray[np.int64],
    true_labels: NDArray[np.int64],
    probabilities: NDArray[np.float64] | None = None,
) -> EvaluationMetrics:
    """Compute evaluation metrics from predictions.

    Args:
        predictions: Predicted labels.
        true_labels: Ground truth labels.
        probabilities: Predicted probabilities (optional, for AUC-ROC).

    Returns:
        EvaluationMetrics containing results.
    """
    # Basic metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="macro", zero_division=0)
    recall = recall_score(true_labels, predictions, average="macro", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="macro", zero_division=0)

    # Per-class recall
    unique_labels = np.unique(true_labels)
    per_class_recall = {}
    for label in unique_labels:
        mask = true_labels == label
        if mask.sum() > 0:
            per_class_recall[int(label)] = recall_score(
                true_labels[mask] == label,
                predictions[mask] == label,
                zero_division=0,
            )

    # Confusion matrix
    conf_mat = confusion_matrix(true_labels, predictions)

    # AUC-ROC (if probabilities available)
    auc_roc = None
    if probabilities is not None:
        try:
            n_classes = len(unique_labels)
            if n_classes == 2:
                auc_roc = roc_auc_score(true_labels, probabilities[:, 1])
            else:
                auc_roc = roc_auc_score(
                    true_labels, probabilities, multi_class="ovr", average="macro"
                )
        except Exception:
            # AUC can fail with single-class predictions
            pass

    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        predictions=predictions,
        true_labels=true_labels,
        probabilities=probabilities,
        auc_roc=auc_roc,
        per_class_recall=per_class_recall,
        confusion_mat=conf_mat,
    )


def print_metrics(metrics: EvaluationMetrics, title: str = "Evaluation Results") -> None:
    """Print evaluation metrics in a formatted table.

    Args:
        metrics: Metrics to print.
        title: Title for the output.
    """
    print(f"\n{title}")
    print("-" * 40)
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1 (macro): {metrics.f1:.4f}")
    if metrics.auc_roc is not None:
        print(f"  AUC-ROC:   {metrics.auc_roc:.4f}")
    print("", flush=True)


def print_per_class_recall(
    metrics: EvaluationMetrics,
    class_names: list[str] | None = None,
) -> None:
    """Print per-class recall scores.

    Args:
        metrics: Metrics containing per_class_recall.
        class_names: Names for each class index.
    """
    print("\nPer-Class Recall:")
    print("-" * 50)

    for label, rec in sorted(metrics.per_class_recall.items()):
        name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        bar = "█" * int(rec * 20)
        status = "⚠️" if rec < 0.3 else "✓" if rec > 0.5 else ""
        print(f"  {name:30s} {rec:.4f} {bar} {status}")


def print_confusion_matrix(
    metrics: EvaluationMetrics,
    class_names: list[str] | None = None,
) -> None:
    """Print confusion matrix.

    Args:
        metrics: Metrics containing confusion_mat.
        class_names: Names for each class index.
    """
    if metrics.confusion_mat is None:
        return

    print("\nConfusion Matrix:")
    print("-" * 60)

    n_classes = metrics.confusion_mat.shape[0]

    # Header
    if class_names:
        short_names = [n[:8] for n in class_names[:n_classes]]
    else:
        short_names = [f"C{i}" for i in range(n_classes)]

    header = "          " + " ".join(f"{n:>8}" for n in short_names)
    print(header)
    print("          " + "-" * (9 * n_classes))

    # Rows
    for i, row in enumerate(metrics.confusion_mat):
        name = short_names[i] if i < len(short_names) else f"C{i}"
        row_str = " ".join(f"{val:>8}" for val in row)
        print(f"  {name:>6} | {row_str}")


def save_confusion_matrix_plot(
    metrics: EvaluationMetrics,
    save_path: Path | str,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> None:
    """Save confusion matrix as an image.

    Args:
        metrics: Metrics containing confusion_mat.
        save_path: Path to save the plot.
        class_names: Names for each class.
        title: Title for the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping confusion matrix plot")
        return

    if metrics.confusion_mat is None:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Normalize for better visualization
    cm_normalized = metrics.confusion_mat.astype(float)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm_normalized, row_sums, where=row_sums != 0)

    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    n_classes = metrics.confusion_mat.shape[0]
    if class_names and len(class_names) >= n_classes:
        tick_labels = class_names[:n_classes]
    else:
        tick_labels = [f"Class {i}" for i in range(n_classes)]

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_normalized.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, f"{metrics.confusion_mat[i, j]}",
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")


def print_classification_report_full(
    metrics: EvaluationMetrics,
    class_names: list[str] | None = None,
) -> None:
    """Print detailed sklearn classification report.

    Args:
        metrics: Metrics containing predictions and true labels.
        class_names: Names for each class. Uses indices if None.
    """
    print("\nDetailed Classification Report:")
    print("-" * 60)
    
    # Get unique labels that appear in the data
    unique_labels = sorted(set(metrics.true_labels) | set(metrics.predictions))
    
    # Filter class names to only those present in the data
    if class_names is not None:
        filtered_names = [class_names[i] for i in unique_labels if i < len(class_names)]
    else:
        filtered_names = None
    
    print(
        classification_report(
            metrics.true_labels,
            metrics.predictions,
            labels=unique_labels,
            target_names=filtered_names,
            zero_division=0,
        ),
        flush=True,
    )


def compare_models(
    nn_metrics: EvaluationMetrics,
    rf_metrics: EvaluationMetrics,
) -> str:
    """Compare two models and return the winner.

    Args:
        nn_metrics: Neural network evaluation metrics.
        rf_metrics: Random Forest evaluation metrics.

    Returns:
        "neural_network" or "random_forest" based on F1 score.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 60)
    print(f"\n{'Metric':<15} {'Neural Network':<18} {'Random Forest':<18}")
    print("-" * 51)
    print(f"{'Accuracy':<15} {nn_metrics.accuracy:<18.4f} {rf_metrics.accuracy:<18.4f}")
    print(f"{'Precision':<15} {nn_metrics.precision:<18.4f} {rf_metrics.precision:<18.4f}")
    print(f"{'Recall':<15} {nn_metrics.recall:<18.4f} {rf_metrics.recall:<18.4f}")
    print(f"{'F1 (macro)':<15} {nn_metrics.f1:<18.4f} {rf_metrics.f1:<18.4f}")

    # Add AUC-ROC if available
    nn_auc = nn_metrics.auc_roc if nn_metrics.auc_roc else 0.0
    rf_auc = rf_metrics.auc_roc if rf_metrics.auc_roc else 0.0
    if nn_auc > 0 or rf_auc > 0:
        print(f"{'AUC-ROC':<15} {nn_auc:<18.4f} {rf_auc:<18.4f}")

    print("-" * 51)

    if nn_metrics.f1 >= rf_metrics.f1:
        winner = "neural_network"
        print(f"\n>>> Winner: Neural Network (F1: {nn_metrics.f1:.4f})", flush=True)
    else:
        winner = "random_forest"
        print(f"\n>>> Winner: Random Forest (F1: {rf_metrics.f1:.4f})", flush=True)

    return winner


def compare_multiple_models(
    model_metrics: dict[str, EvaluationMetrics],
    primary_metric: str = "f1",
) -> str:
    """Compare multiple models and return the winner.

    Args:
        model_metrics: Dictionary mapping model name to EvaluationMetrics.
        primary_metric: Metric to use for comparison ("f1", "auc_roc", "recall").

    Returns:
        Name of the winning model.
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 80)

    # Build header
    model_names = list(model_metrics.keys())
    col_width = max(15, max(len(n) for n in model_names) + 2)

    header = f"{'Metric':<15}"
    for name in model_names:
        header += f" {name:<{col_width}}"
    print(header)
    print("-" * (15 + (col_width + 1) * len(model_names)))

    # Print metrics rows
    metrics_to_print = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 (macro)", "f1"),
        ("AUC-ROC", "auc_roc"),
    ]

    for label, attr in metrics_to_print:
        row = f"{label:<15}"
        for name in model_names:
            m = model_metrics[name]
            val = getattr(m, attr, None)
            if val is not None:
                row += f" {val:<{col_width}.4f}"
            else:
                row += f" {'N/A':<{col_width}}"
        print(row)

    print("-" * (15 + (col_width + 1) * len(model_names)))

    # Determine winner
    def get_metric_value(m: EvaluationMetrics) -> float:
        val = getattr(m, primary_metric, None)
        return val if val is not None else 0.0

    winner = max(model_metrics.keys(), key=lambda n: get_metric_value(model_metrics[n]))
    winner_score = get_metric_value(model_metrics[winner])

    print(f"\n>>> Winner: {winner} ({primary_metric.upper()}: {winner_score:.4f})", flush=True)

    return winner
