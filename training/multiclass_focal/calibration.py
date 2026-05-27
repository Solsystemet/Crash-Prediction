"""Probability calibration for multiclass classification.

This module provides utilities for:
- Calibrating model probabilities using isotonic regression or Platt scaling
- Evaluating calibration quality (reliability diagrams, ECE)
- Class-specific threshold optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt


@dataclass
class CalibrationResult:
    """Results from calibration fitting.

    Attributes:
        calibrated_probs: Calibrated probability matrix (n_samples, n_classes)
        ece: Expected Calibration Error
        per_class_ece: ECE for each class
        reliability_data: Data for plotting reliability diagrams
    """

    calibrated_probs: NDArray[np.float64]
    ece: float
    per_class_ece: NDArray[np.float64]
    reliability_data: dict[int, dict[str, NDArray[np.float64]]]


class CalibratedClassifier:
    """Wrapper that adds probability calibration to a trained model.

    Fits one calibrator per class using isotonic regression (recommended)
    or Platt scaling (logistic regression).

    Focal Loss and other modified loss functions often produce miscalibrated
    probabilities. This wrapper corrects them using a held-out validation set.

    Args:
        method: Calibration method ('isotonic' or 'platt')
        cv: Whether to use cross-validation for calibration fitting
    """

    def __init__(
        self,
        method: Literal["isotonic", "platt"] = "isotonic",
    ) -> None:
        self.method = method
        self.calibrators: list[IsotonicRegression | LogisticRegression] = []
        self.num_classes: int = 0
        self.is_fitted: bool = False

    def fit(
        self,
        probs: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> "CalibratedClassifier":
        """Fit calibrators on validation data.

        Args:
            probs: Uncalibrated probabilities, shape (n_samples, n_classes)
            labels: True class labels, shape (n_samples,)

        Returns:
            Self for chaining.
        """
        self.num_classes = probs.shape[1]
        self.calibrators = []

        for c in range(self.num_classes):
            # Binary labels: is this class c?
            y_binary = (labels == c).astype(np.float64)
            p_class = probs[:, c]

            if self.method == "isotonic":
                calibrator = IsotonicRegression(
                    y_min=0.0,
                    y_max=1.0,
                    out_of_bounds="clip",
                )
                calibrator.fit(p_class, y_binary)
            else:  # platt
                calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
                # Reshape for sklearn
                calibrator.fit(p_class.reshape(-1, 1), y_binary)

            self.calibrators.append(calibrator)

        self.is_fitted = True
        return self

    def calibrate(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply calibration to new probabilities.

        Args:
            probs: Uncalibrated probabilities, shape (n_samples, n_classes)

        Returns:
            Calibrated probabilities (normalized to sum to 1).
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        calibrated = np.zeros_like(probs)

        for c, calibrator in enumerate(self.calibrators):
            p_class = probs[:, c]
            if self.method == "isotonic":
                calibrated[:, c] = calibrator.predict(p_class)
            else:
                calibrated[:, c] = calibrator.predict_proba(
                    p_class.reshape(-1, 1)
                )[:, 1]

        # Normalize to ensure probabilities sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / (row_sums + 1e-8)

        return calibrated


def expected_calibration_error(
    probs: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well the predicted probabilities match empirical
    frequencies across confidence bins.

    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|

    A perfectly calibrated model has ECE = 0.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True class labels, shape (n_samples,)
        n_bins: Number of confidence bins

    Returns:
        Expected Calibration Error (lower is better)
    """
    # Get predicted class and confidence
    pred_labels = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    accuracies = (pred_labels == labels).astype(np.float64)

    # Bin edges
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        # Samples in this bin
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)


def per_class_ece(
    probs: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_bins: int = 15,
) -> NDArray[np.float64]:
    """Compute ECE for each class separately.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True class labels, shape (n_samples,)
        n_bins: Number of confidence bins

    Returns:
        Array of ECE values, one per class
    """
    num_classes = probs.shape[1]
    ece_values = np.zeros(num_classes)

    for c in range(num_classes):
        # Binary problem: class c vs rest
        y_binary = (labels == c).astype(np.float64)
        p_class = probs[:, c]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (p_class > bin_edges[i]) & (p_class <= bin_edges[i + 1])
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                avg_confidence = p_class[in_bin].mean()
                avg_accuracy = y_binary[in_bin].mean()
                ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

        ece_values[c] = ece

    return ece_values


def brier_score_multiclass(
    probs: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    """Compute multiclass Brier score.

    Brier score measures the mean squared error between predicted probabilities
    and the one-hot encoded true labels.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True class labels, shape (n_samples,)

    Returns:
        Brier score (lower is better)
    """
    num_classes = probs.shape[1]
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def reliability_diagram_data(
    probs: NDArray[np.float64],
    labels: NDArray[np.int64],
    n_bins: int = 10,
    class_idx: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute data for reliability diagram.

    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins
        class_idx: If specified, compute for binary (class vs rest).
            If None, use top-1 prediction confidence.

    Returns:
        Dict with 'bin_centers', 'accuracies', 'counts', 'confidences'
    """
    if class_idx is not None:
        # Binary: class vs rest
        y_binary = (labels == class_idx).astype(np.float64)
        confidences = probs[:, class_idx]
        accuracies = y_binary
    else:
        # Multiclass: use top-1
        pred_labels = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        accuracies = (pred_labels == labels).astype(np.float64)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        bin_counts[i] = in_bin.sum()

        if bin_counts[i] > 0:
            bin_accuracies[i] = accuracies[in_bin].mean()
            bin_confidences[i] = confidences[in_bin].mean()

    return {
        "bin_centers": bin_centers,
        "accuracies": bin_accuracies,
        "confidences": bin_confidences,
        "counts": bin_counts,
    }


def plot_reliability_diagram(
    probs: NDArray[np.float64],
    labels: NDArray[np.int64],
    class_names: list[str] | None = None,
    n_bins: int = 10,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot reliability diagrams for all classes.

    Args:
        probs: Predicted probabilities
        labels: True labels
        class_names: Names for each class
        n_bins: Number of bins
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    num_classes = probs.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Create subplot grid
    n_cols = min(3, num_classes)
    n_rows = (num_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for c in range(num_classes):
        ax = axes[c]
        data = reliability_diagram_data(probs, labels, n_bins=n_bins, class_idx=c)

        # Plot diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")

        # Plot actual calibration
        mask = data["counts"] > 0
        ax.bar(
            data["bin_centers"][mask],
            data["accuracies"][mask],
            width=0.8 / n_bins,
            alpha=0.7,
            color="steelblue",
            label="Actual",
        )

        # Compute ECE for this class
        y_binary = (labels == c).astype(np.float64)
        ece = expected_calibration_error(
            np.column_stack([1 - probs[:, c], probs[:, c]]),
            y_binary.astype(np.int64),
            n_bins=n_bins,
        )

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"{class_names[c]}\nECE={ece:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")

    # Hide empty subplots
    for i in range(num_classes, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


@dataclass
class ThresholdOptimizationResult:
    """Results from class-specific threshold optimization.

    Attributes:
        thresholds: Optimal threshold per class
        f1_scores: F1 at optimal threshold per class
        precision: Precision at optimal threshold per class
        recall: Recall at optimal threshold per class
    """

    thresholds: NDArray[np.float64]
    f1_scores: NDArray[np.float64]
    precision: NDArray[np.float64]
    recall: NDArray[np.float64]


def optimize_thresholds(
    probs: NDArray[np.float64],
    labels: NDArray[np.int64],
    metric: Literal["f1", "f2", "f0.5"] = "f1",
    n_thresholds: int = 100,
) -> ThresholdOptimizationResult:
    """Find optimal classification threshold for each class.

    Unlike using 0.5 or argmax for all classes, this finds the threshold
    that maximizes F1 (or F-beta) for each class separately.

    For minority classes, optimal thresholds are typically lower (e.g., 0.15)
    to allow more positive predictions.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        labels: True class labels, shape (n_samples,)
        metric: Metric to optimize ('f1', 'f2' for recall-heavy, 'f0.5' for precision-heavy)
        n_thresholds: Number of thresholds to try

    Returns:
        ThresholdOptimizationResult with optimal thresholds and metrics
    """
    num_classes = probs.shape[1]
    thresholds_to_try = np.linspace(0.01, 0.99, n_thresholds)

    # Parse beta for F-beta score
    if metric == "f1":
        beta = 1.0
    elif metric == "f2":
        beta = 2.0
    elif metric == "f0.5":
        beta = 0.5
    else:
        raise ValueError(f"Unknown metric: {metric}")

    optimal_thresholds = np.zeros(num_classes)
    optimal_f1 = np.zeros(num_classes)
    optimal_precision = np.zeros(num_classes)
    optimal_recall = np.zeros(num_classes)

    for c in range(num_classes):
        y_binary = (labels == c).astype(np.int64)
        p_class = probs[:, c]

        best_fbeta = -1
        best_threshold = 0.5
        best_prec = 0.0
        best_rec = 0.0

        for threshold in thresholds_to_try:
            pred = (p_class >= threshold).astype(np.int64)

            tp = ((pred == 1) & (y_binary == 1)).sum()
            fp = ((pred == 1) & (y_binary == 0)).sum()
            fn = ((pred == 0) & (y_binary == 1)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            # F-beta score
            fbeta = (
                (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
            )

            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_threshold = threshold
                best_prec = precision
                best_rec = recall

        optimal_thresholds[c] = best_threshold
        optimal_f1[c] = best_fbeta
        optimal_precision[c] = best_prec
        optimal_recall[c] = best_rec

    return ThresholdOptimizationResult(
        thresholds=optimal_thresholds,
        f1_scores=optimal_f1,
        precision=optimal_precision,
        recall=optimal_recall,
    )


def predict_with_thresholds(
    probs: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    default_class: int = 0,
) -> NDArray[np.int64]:
    """Make predictions using class-specific thresholds.

    For each sample, predicts the class with highest probability that
    exceeds its threshold. If no class exceeds threshold, returns default.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes)
        thresholds: Threshold per class, shape (num_classes,)
        default_class: Class to predict if no threshold exceeded

    Returns:
        Predicted class indices
    """
    n_samples, num_classes = probs.shape

    # Compute "margin over threshold" for each class
    margins = probs - thresholds

    # Predict class with highest positive margin
    predictions = np.full(n_samples, default_class, dtype=np.int64)
    max_margins = margins.max(axis=1)

    # Only update where at least one class exceeds threshold
    valid = max_margins > 0
    predictions[valid] = margins[valid].argmax(axis=1)

    return predictions
