"""Evaluation utilities for multiclass crash severity models.

This module provides comprehensive metrics for imbalanced classification:
- Per-class precision, recall, F1
- Macro/micro/weighted averages
- ROC-AUC and PR-AUC per class
- Confusion matrices
- Bootstrap confidence intervals
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# Standard severity class order
SEVERITY_CLASS_ORDER = [
    "FATAL",
    "INCAPACITATING INJURY",
    "NO INDICATION OF INJURY",
    "NONINCAPACITATING INJURY",
    "REPORTED, NOT EVIDENT",
]


@dataclass
class ClassMetrics:
    """Metrics for a single class."""

    name: str
    precision: float
    recall: float
    f1: float
    support: int  # Number of true samples
    roc_auc: float | None
    pr_auc: float | None


@dataclass
class EvaluationResult:
    """Complete evaluation results for a model."""

    # Overall metrics
    accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float

    # Per-class metrics
    class_metrics: list[ClassMetrics]

    # Aggregated metrics
    macro_precision: float
    macro_recall: float
    weighted_precision: float
    weighted_recall: float

    # Probability quality
    brier_score: float | None = None
    ece: float | None = None

    # Confusion matrix
    confusion_matrix: NDArray[np.int64] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "brier_score": self.brier_score,
            "ece": self.ece,
            "class_metrics": [asdict(cm) for cm in self.class_metrics],
        }
        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix.tolist()
        return result

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def evaluate_predictions(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    probs: NDArray[np.float64] | None = None,
    class_names: list[str] | None = None,
) -> EvaluationResult:
    """Comprehensive evaluation of classification predictions.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        probs: Predicted probabilities (optional, for AUC/Brier)
        class_names: Names for each class

    Returns:
        EvaluationResult with all metrics
    """
    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Averages
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class ROC-AUC and PR-AUC (if probabilities available)
    class_metrics = []
    for i, name in enumerate(class_names):
        roc_auc = None
        pr_auc = None

        if probs is not None and len(np.unique(y_true)) > 1:
            # Binary: class i vs rest
            y_binary = (y_true == i).astype(int)
            if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
                try:
                    roc_auc = float(roc_auc_score(y_binary, probs[:, i]))
                    pr_auc = float(average_precision_score(y_binary, probs[:, i]))
                except ValueError:
                    pass

        class_metrics.append(
            ClassMetrics(
                name=name,
                precision=float(precision[i]) if i < len(precision) else 0.0,
                recall=float(recall[i]) if i < len(recall) else 0.0,
                f1=float(f1[i]) if i < len(f1) else 0.0,
                support=int(support[i]) if i < len(support) else 0,
                roc_auc=roc_auc,
                pr_auc=pr_auc,
            )
        )

    # Probability quality metrics
    brier = None
    ece = None
    if probs is not None:
        from training.multiclass_focal.calibration import (
            brier_score_multiclass,
            expected_calibration_error,
        )
        brier = brier_score_multiclass(probs, y_true)
        ece = expected_calibration_error(probs, y_true)

    return EvaluationResult(
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        micro_f1=float(micro_f1),
        weighted_f1=float(weighted_f1),
        class_metrics=class_metrics,
        macro_precision=float(macro_prec),
        macro_recall=float(macro_rec),
        weighted_precision=float(weighted_prec),
        weighted_recall=float(weighted_rec),
        brier_score=brier,
        ece=ece,
        confusion_matrix=cm,
    )


def bootstrap_confidence_interval(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_fn: Function(y_true, y_pred) -> float
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)

    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)

    # Bootstrap
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        metric = metric_fn(y_true[indices], y_pred[indices])
        bootstrap_metrics.append(metric)

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Percentile confidence interval
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(bootstrap_metrics, 100 * alpha))
    upper = float(np.percentile(bootstrap_metrics, 100 * (1 - alpha)))

    return point_estimate, lower, upper


def compare_models(
    results: dict[str, EvaluationResult],
    primary_metric: str = "macro_f1",
) -> str:
    """Generate comparison table for multiple models.

    Args:
        results: Dict mapping model name to EvaluationResult
        primary_metric: Metric to sort by

    Returns:
        Formatted comparison string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL COMPARISON")
    lines.append("=" * 80)

    # Header
    header = f"{'Model':<20} {'Acc':>8} {'Macro-F1':>10} {'Weight-F1':>10}"
    if any(r.brier_score is not None for r in results.values()):
        header += f" {'Brier':>8} {'ECE':>8}"
    lines.append(header)
    lines.append("-" * 80)

    # Sort by primary metric
    sorted_models = sorted(
        results.items(),
        key=lambda x: getattr(x[1], primary_metric),
        reverse=True,
    )

    for name, result in sorted_models:
        row = f"{name:<20} {result.accuracy:>8.4f} {result.macro_f1:>10.4f} {result.weighted_f1:>10.4f}"
        if result.brier_score is not None:
            row += f" {result.brier_score:>8.4f} {result.ece:>8.4f}"
        lines.append(row)

    lines.append("=" * 80)

    # Per-class comparison for best model
    best_name, best_result = sorted_models[0]
    lines.append(f"\nBest model ({best_name}) per-class metrics:")
    lines.append("-" * 60)
    lines.append(f"{'Class':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    lines.append("-" * 60)

    for cm in best_result.class_metrics:
        lines.append(
            f"{cm.name:<30} {cm.precision:>8.4f} {cm.recall:>8.4f} {cm.f1:>8.4f} {cm.support:>8d}"
        )

    return "\n".join(lines)


def plot_confusion_matrix(
    cm: NDArray[np.int64],
    class_names: list[str],
    normalize: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix
        class_names: Class names
        normalize: Whether to normalize by row (true class)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(10, 8))
    
    if HAS_SEABORN:
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
    else:
        # Fallback to matplotlib imshow
        im = ax.imshow(cm_plot, cmap="Blues")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_plot[i, j]
                text = f"{val:{fmt[1:]}}" if isinstance(val, float) else str(val)
                ax.text(j, i, text, ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_per_class_metrics(
    results: dict[str, EvaluationResult],
    metric: str = "f1",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot per-class metrics for multiple models.

    Args:
        results: Dict mapping model name to EvaluationResult
        metric: Metric to plot ('precision', 'recall', 'f1')
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    model_names = list(results.keys())
    first_result = list(results.values())[0]
    class_names = [cm.name for cm in first_result.class_metrics]
    num_classes = len(class_names)
    num_models = len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(num_classes)
    width = 0.8 / num_models

    for i, (name, result) in enumerate(results.items()):
        values = [getattr(cm, metric) for cm in result.class_metrics]
        offset = (i - num_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, alpha=0.8)

    ax.set_xlabel("Class")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Per-Class {metric.capitalize()} Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def print_evaluation_summary(result: EvaluationResult, model_name: str = "Model") -> None:
    """Print formatted evaluation summary to console."""
    print("=" * 60)
    print(f"EVALUATION: {model_name}")
    print("=" * 60)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:        {result.accuracy:.4f}")
    print(f"  Macro F1:        {result.macro_f1:.4f}")
    print(f"  Weighted F1:     {result.weighted_f1:.4f}")
    print(f"  Macro Precision: {result.macro_precision:.4f}")
    print(f"  Macro Recall:    {result.macro_recall:.4f}")

    if result.brier_score is not None:
        print(f"\nProbability Quality:")
        print(f"  Brier Score:     {result.brier_score:.4f}")
        print(f"  ECE:             {result.ece:.4f}")

    print(f"\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 60)

    for cm in result.class_metrics:
        print(
            f"{cm.name:<30} {cm.precision:>8.4f} {cm.recall:>8.4f} "
            f"{cm.f1:>8.4f} {cm.support:>8d}"
        )

        if cm.roc_auc is not None:
            print(f"{'  ROC-AUC: ' + f'{cm.roc_auc:.4f}':<30} PR-AUC: {cm.pr_auc:.4f}")

    print("=" * 60)
