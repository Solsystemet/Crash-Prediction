"""
Calibration curve plotting utilities.

Provides reliability diagrams and calibration metrics for probability calibration analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration metrics.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins for calibration

    Returns:
        Dict containing calibration metrics
    """
    from sklearn.metrics import brier_score_loss

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Brier score (lower is better)
    brier = brier_score_loss(y_true, y_prob)

    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_confidence = np.mean(y_prob[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            ece += (bin_size / total_samples) * abs(avg_accuracy - avg_confidence)

    # Maximum Calibration Error (MCE)
    mce = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_confidence = np.mean(y_prob[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            mce = max(mce, abs(avg_accuracy - avg_confidence))

    return {
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
    }


def _compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities

    Returns:
        Tuple of (mean_predicted_probs, fraction_positives, bin_counts)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mean_predicted = []
    fraction_positive = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            mean_predicted.append(np.mean(y_prob[in_bin]))
            fraction_positive.append(np.mean(y_true[in_bin]))
            bin_counts.append(bin_size)

    return np.array(mean_predicted), np.array(fraction_positive), np.array(bin_counts)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[str | Path] = None,
    n_bins: int = 10,
    title: str = "Calibration Curve (Reliability Diagram)",
    figsize: tuple = (10, 8),
    show_histogram: bool = True,
) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram).

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities for positive class
        output_path: Path to save the plot
        n_bins: Number of bins for calibration
        title: Plot title
        figsize: Figure size
        show_histogram: If True, show prediction distribution histogram

    Returns:
        matplotlib Figure object
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    mean_predicted, fraction_positive, bin_counts = _compute_calibration_curve(
        y_true, y_prob, n_bins
    )
    metrics = compute_calibration_metrics(y_true, y_prob, n_bins)

    if show_histogram:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax1 = plt.subplots(figsize=(figsize[0], figsize[1] * 0.7))
        ax2 = None

    # Main calibration plot
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", linewidth=2)
    ax1.plot(
        mean_predicted,
        fraction_positive,
        "s-",
        color="steelblue",
        label="Model",
        linewidth=2,
        markersize=8,
    )

    # Fill gap between perfect and actual
    for mp, fp in zip(mean_predicted, fraction_positive):
        color = "red" if fp < mp else "green"
        ax1.plot([mp, mp], [mp, fp], color=color, alpha=0.3, linewidth=2)

    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)

    # Add metrics annotation
    metrics_text = (
        f"Brier Score: {metrics['brier_score']:.4f}\n"
        f"ECE: {metrics['ece']:.4f}\n"
        f"MCE: {metrics['mce']:.4f}"
    )
    ax1.text(
        0.98,
        0.02,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Histogram of predictions
    if ax2 is not None:
        ax2.hist(y_prob, bins=n_bins, range=(0, 1), color="steelblue", edgecolor="black", alpha=0.7)
        ax2.set_xlabel("Predicted Probability", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_xlim([0, 1])
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

        # Save metrics as JSON
        import json

        json_path = output_path.with_suffix(".json")
        save_data = {
            **metrics,
            "mean_predicted": mean_predicted.tolist(),
            "fraction_positive": fraction_positive.tolist(),
            "bin_counts": bin_counts.tolist(),
        }
        with open(json_path, "w") as f:
            json.dump(save_data, f, indent=2)

    return fig


def plot_multiclass_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
    n_bins: int = 10,
    title: str = "Multiclass Calibration Curves",
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plot calibration curves for multiclass classification (one-vs-rest).

    Args:
        y_true: True class labels (integers 0 to n_classes-1)
        y_prob: Predicted probabilities, shape (n_samples, n_classes)
        class_names: Names for each class
        output_path: Path to save the plot
        n_bins: Number of bins
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_classes = y_prob.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Calculate number of subplot rows/cols
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    axes = np.atleast_2d(axes).flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        ax = axes[i]

        # One-vs-rest binary problem
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]

        mean_predicted, fraction_positive, _ = _compute_calibration_curve(
            y_true_binary, y_prob_class, n_bins
        )
        metrics = compute_calibration_metrics(y_true_binary, y_prob_class, n_bins)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.plot(
            mean_predicted,
            fraction_positive,
            "s-",
            color=colors[i],
            linewidth=2,
            markersize=6,
        )

        ax.set_xlabel("Mean Predicted Prob")
        ax.set_ylabel("Fraction Positive")
        ax.set_title(f"{class_names[i]} (ECE={metrics['ece']:.3f})")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
