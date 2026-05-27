"""
Training curves plotting utilities.

Visualizes loss and metrics over epochs for model training progress analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = "Accuracy",
    output_path: Optional[str | Path] = None,
    title: str = "Training Curves",
    figsize: tuple = (12, 5),
    save_json: bool = True,
) -> plt.Figure:
    """
    Plot training curves showing loss and optionally metrics over epochs.

    Args:
        train_losses: Training loss values per epoch
        val_losses: Validation loss values per epoch (optional)
        train_metrics: Training metric values per epoch (optional)
        val_metrics: Validation metric values per epoch (optional)
        metric_name: Name of the metric being plotted
        output_path: Path to save the plot (PNG). If None, returns figure without saving.
        title: Plot title
        figsize: Figure size (width, height)
        save_json: If True, save curve data as JSON alongside PNG

    Returns:
        matplotlib Figure object
    """
    epochs = range(1, len(train_losses) + 1)

    # Determine number of subplots
    has_metrics = train_metrics is not None or val_metrics is not None
    n_plots = 2 if has_metrics else 1

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Plot loss
    ax_loss = axes[0]
    ax_loss.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    if val_losses is not None:
        ax_loss.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss Over Epochs")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Mark best validation loss
    if val_losses is not None:
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        ax_loss.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7)
        ax_loss.annotate(
            f"Best: {best_val_loss:.4f} (epoch {best_epoch})",
            xy=(best_epoch, best_val_loss),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            color="green",
        )

    # Plot metrics if provided
    if has_metrics:
        ax_metric = axes[1]
        if train_metrics is not None:
            ax_metric.plot(
                epochs, train_metrics, "b-", label=f"Train {metric_name}", linewidth=2
            )
        if val_metrics is not None:
            ax_metric.plot(
                epochs, val_metrics, "r-", label=f"Val {metric_name}", linewidth=2
            )
        ax_metric.set_xlabel("Epoch")
        ax_metric.set_ylabel(metric_name)
        ax_metric.set_title(f"{metric_name} Over Epochs")
        ax_metric.legend()
        ax_metric.grid(True, alpha=0.3)

        # Mark best validation metric
        if val_metrics is not None:
            best_epoch = np.argmax(val_metrics) + 1
            best_val_metric = max(val_metrics)
            ax_metric.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7)
            ax_metric.annotate(
                f"Best: {best_val_metric:.4f} (epoch {best_epoch})",
                xy=(best_epoch, best_val_metric),
                xytext=(10, -10),
                textcoords="offset points",
                fontsize=9,
                color="green",
            )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

        # Save JSON data
        if save_json:
            json_path = output_path.with_suffix(".json")
            curve_data = {
                "epochs": list(epochs),
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_metric": train_metrics,
                "val_metric": val_metrics,
                "metric_name": metric_name,
            }
            with open(json_path, "w") as f:
                json.dump(curve_data, f, indent=2)

    return fig


def plot_training_curves_from_history(
    history: Dict[str, List[float]] | pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "Training Curves",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot training curves from a history dictionary or DataFrame.

    Expects keys/columns: 'train_loss', 'val_loss', optionally 'train_metric', 'val_metric'.

    Args:
        history: Training history as dict or DataFrame
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if isinstance(history, pd.DataFrame):
        history = history.to_dict(orient="list")

    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss")
    train_metrics = history.get("train_metric")
    val_metrics = history.get("val_metric")
    metric_name = history.get("metric_name", ["Metric"])[0] if "metric_name" in history else "Metric"

    return plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        metric_name=metric_name,
        output_path=output_path,
        title=title,
        figsize=figsize,
    )


def plot_learning_rate_schedule(
    learning_rates: List[float],
    output_path: Optional[str | Path] = None,
    title: str = "Learning Rate Schedule",
) -> plt.Figure:
    """
    Plot learning rate schedule over epochs.

    Args:
        learning_rates: Learning rate values per epoch
        output_path: Path to save the plot
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    epochs = range(1, len(learning_rates) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, learning_rates, "g-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
