"""Evaluation utilities for model comparison.

This module provides functions for computing metrics and comparing models.
"""

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn import metrics as skmetrics  # type: ignore[import-untyped]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]

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
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    predictions: NDArray[np.int64]
    true_labels: NDArray[np.int64]


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
        _, predictions = outputs.max(1)
        predictions = predictions.cpu().numpy()

    return _compute_metrics(predictions, true_labels)


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

    return _compute_metrics(predictions, true_labels)


def evaluate_lightgbm(
    model: lgb.LGBMClassifier,
    dataset: CrashTensorDataset,
) -> EvaluationMetrics:
    """Evaluate a LightGBM model on a dataset.

    Args:
        model: Trained LightGBM model.
        dataset: Dataset to evaluate on.

    Returns:
        EvaluationMetrics containing results.
    """
    X = dataset.features.numpy()
    X_pred: Any = X
    true_labels = dataset.labels.numpy()
    if hasattr(model, "feature_name_") and len(model.feature_name_) == X.shape[1]:
        X_pred = pd.DataFrame(X, columns=list(model.feature_name_))
    predictions = model.predict(X_pred)

    return _compute_metrics(predictions, true_labels)


def _compute_metrics(
    predictions: NDArray[np.int64],
    true_labels: NDArray[np.int64],
) -> EvaluationMetrics:
    """Compute evaluation metrics from predictions.

    Args:
        predictions: Predicted labels.
        true_labels: Ground truth labels.

    Returns:
        EvaluationMetrics containing results.
    """
    return EvaluationMetrics(
        accuracy=skmetrics.accuracy_score(true_labels, predictions),
        precision=skmetrics.precision_score(
            true_labels, predictions, average="macro", zero_division=0
        ),
        recall=skmetrics.recall_score(true_labels, predictions, average="macro", zero_division=0),
        f1=skmetrics.f1_score(true_labels, predictions, average="macro", zero_division=0),
        predictions=predictions,
        true_labels=true_labels,
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
    print(f"  F1 (macro): {metrics.f1:.4f}", flush=True)


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
        skmetrics.classification_report(
            metrics.true_labels,
            metrics.predictions,
            labels=unique_labels,
            target_names=filtered_names,
            zero_division=0,
        ),
        flush=True,
    )


def compare_models(
    lgbm_metrics: EvaluationMetrics,
    rf_metrics: EvaluationMetrics,
) -> str:
    """Compare two models and return the winner.

    Args:
        lgbm_metrics: LightGBM evaluation metrics.
        rf_metrics: Random Forest evaluation metrics.

    Returns:
        "lightgbm" or "random_forest" based on F1 score.
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 60)
    print(f"\n{'Metric':<15} {'LightGBM':<18} {'Random Forest':<18}")
    print("-" * 51)
    print(f"{'Accuracy':<15} {lgbm_metrics.accuracy:<18.4f} {rf_metrics.accuracy:<18.4f}")
    print(f"{'Precision':<15} {lgbm_metrics.precision:<18.4f} {rf_metrics.precision:<18.4f}")
    print(f"{'Recall':<15} {lgbm_metrics.recall:<18.4f} {rf_metrics.recall:<18.4f}")
    print(f"{'F1 (macro)':<15} {lgbm_metrics.f1:<18.4f} {rf_metrics.f1:<18.4f}")
    print("-" * 51)

    if lgbm_metrics.f1 >= rf_metrics.f1:
        winner = "lightgbm"
        print(f"\n>>> Winner: LightGBM (F1: {lgbm_metrics.f1:.4f})", flush=True)
    else:
        winner = "random_forest"
        print(f"\n>>> Winner: Random Forest (F1: {rf_metrics.f1:.4f})", flush=True)

    return winner
