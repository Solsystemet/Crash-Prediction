"""
Prediction confidence distribution plotting utilities.

Visualizes model confidence and uncertainty in predictions.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_confidence_histogram(
    probabilities: np.ndarray,
    output_path: Optional[str | Path] = None,
    n_bins: int = 20,
    title: str = "Prediction Confidence Distribution",
    figsize: tuple = (10, 6),
    threshold: float = 0.6,
) -> plt.Figure:
    """
    Plot histogram of prediction confidence (max probability).

    Args:
        probabilities: Predicted probabilities, shape (n_samples,) or (n_samples, n_classes)
        output_path: Path to save the plot
        n_bins: Number of histogram bins
        title: Plot title
        figsize: Figure size
        threshold: Confidence threshold to highlight low-confidence predictions

    Returns:
        matplotlib Figure object
    """
    probabilities = np.asarray(probabilities)

    # Get max probability for each sample (confidence)
    if probabilities.ndim == 2:
        confidence = np.max(probabilities, axis=1)
    else:
        confidence = probabilities

    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram
    n, bins, patches = ax.hist(
        confidence, bins=n_bins, range=(0, 1), color="steelblue", edgecolor="black", alpha=0.7
    )

    # Color low-confidence bins differently
    for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
        if left_edge + (bins[1] - bins[0]) / 2 < threshold:
            patch.set_facecolor("coral")

    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold})")

    # Statistics
    mean_conf = np.mean(confidence)
    median_conf = np.median(confidence)
    low_conf_pct = np.mean(confidence < threshold) * 100

    ax.axvline(x=mean_conf, color="green", linestyle="-", linewidth=2, alpha=0.7, label=f"Mean ({mean_conf:.3f})")
    ax.axvline(x=median_conf, color="purple", linestyle="-", linewidth=2, alpha=0.7, label=f"Median ({median_conf:.3f})")

    ax.set_xlabel("Confidence (Max Probability)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Add statistics annotation
    stats_text = (
        f"Mean: {mean_conf:.3f}\n"
        f"Median: {median_conf:.3f}\n"
        f"Std: {np.std(confidence):.3f}\n"
        f"Low confidence (<{threshold}): {low_conf_pct:.1f}%"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

        # Save statistics as JSON
        import json

        json_path = output_path.with_suffix(".json")
        stats = {
            "mean": float(mean_conf),
            "median": float(median_conf),
            "std": float(np.std(confidence)),
            "min": float(np.min(confidence)),
            "max": float(np.max(confidence)),
            "low_confidence_pct": float(low_conf_pct),
            "threshold": threshold,
            "n_samples": len(confidence),
        }
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

    return fig


def plot_confidence_by_class(
    probabilities: np.ndarray,
    predicted_classes: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
    title: str = "Confidence Distribution by Predicted Class",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot confidence distribution broken down by predicted class.

    Args:
        probabilities: Predicted probabilities, shape (n_samples, n_classes)
        predicted_classes: Predicted class labels
        class_names: Names for each class
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    probabilities = np.asarray(probabilities)
    predicted_classes = np.asarray(predicted_classes)

    if probabilities.ndim == 2:
        confidence = np.max(probabilities, axis=1)
    else:
        confidence = probabilities

    unique_classes = np.unique(predicted_classes)
    n_classes = len(unique_classes)

    if class_names is None:
        class_names = [f"Class {c}" for c in unique_classes]

    fig, ax = plt.subplots(figsize=figsize)

    # Box plot for each class
    data = [confidence[predicted_classes == c] for c in unique_classes]

    bp = ax.boxplot(
        data,
        labels=class_names,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markeredgecolor="black"),
    )

    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add sample counts
    for i, c in enumerate(unique_classes):
        count = np.sum(predicted_classes == c)
        ax.text(i + 1, 0.02, f"n={count}", ha="center", fontsize=9)

    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confidence_vs_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    output_path: Optional[str | Path] = None,
    n_bins: int = 10,
    title: str = "Confidence vs Accuracy",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot relationship between prediction confidence and actual accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence: Prediction confidence values
        output_path: Path to save the plot
        n_bins: Number of confidence bins
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence)

    correct = (y_true == y_pred).astype(int)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(np.mean(correct[in_bin]))
            bin_counts.append(bin_size)

    fig, ax1 = plt.subplots(figsize=figsize)

    # Accuracy line
    ax1.plot(
        bin_centers, bin_accuracies, "o-", color="steelblue", linewidth=2, markersize=8, label="Accuracy"
    )
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12, color="steelblue")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.grid(True, alpha=0.3)

    # Sample count bars (secondary axis)
    ax2 = ax1.twinx()
    ax2.bar(
        bin_centers, bin_counts, width=0.08, alpha=0.3, color="gray", label="Sample count"
    )
    ax2.set_ylabel("Sample Count", fontsize=12, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title(title, fontsize=14, fontweight="bold")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
