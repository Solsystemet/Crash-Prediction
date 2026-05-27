"""
Recall trends analysis and plotting utilities.

Tracks minority class recall and other metrics across historical training runs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_historical_metrics(
    comparison_dir: str | Path,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """
    Load historical training metrics from comparison CSV files.

    Args:
        comparison_dir: Directory containing comparison CSV files
        pattern: Glob pattern for CSV files

    Returns:
        DataFrame with all historical metrics
    """
    comparison_dir = Path(comparison_dir)

    if not comparison_dir.exists():
        return pd.DataFrame()

    all_dfs = []
    for csv_path in comparison_dir.glob(pattern):
        try:
            df = pd.read_csv(csv_path)

            # Try to extract timestamp from filename
            # Expected format: comparison_YYYYMMDD_HHMMSS.csv
            filename = csv_path.stem
            if "_" in filename:
                parts = filename.split("_")
                if len(parts) >= 3:
                    try:
                        date_str = f"{parts[-2]}_{parts[-1]}"
                        timestamp = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                        df["run_timestamp"] = timestamp
                    except ValueError:
                        df["run_timestamp"] = datetime.fromtimestamp(csv_path.stat().st_mtime)
                else:
                    df["run_timestamp"] = datetime.fromtimestamp(csv_path.stat().st_mtime)
            else:
                df["run_timestamp"] = datetime.fromtimestamp(csv_path.stat().st_mtime)

            df["source_file"] = csv_path.name
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_path}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined.sort_values("run_timestamp")


def plot_recall_trends(
    metrics_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    model_filter: Optional[str] = None,
    metric_columns: Optional[List[str]] = None,
    title: str = "Recall Trends Over Time",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Plot recall metrics trends over time.

    Args:
        metrics_df: DataFrame with historical metrics (from load_historical_metrics)
        output_path: Path to save the plot
        model_filter: Filter to specific model name (substring match)
        metric_columns: List of metric columns to plot. Defaults to recall columns.
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if metrics_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No historical metrics available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    df = metrics_df.copy()

    # Apply model filter
    if model_filter and "model" in df.columns:
        df = df[df["model"].str.contains(model_filter, case=False, na=False)]

    # Determine metric columns to plot
    if metric_columns is None:
        # Default to recall columns
        recall_cols = [c for c in df.columns if "recall" in c.lower()]
        if not recall_cols:
            recall_cols = ["f1_macro", "f1_weighted", "accuracy"]
        metric_columns = [c for c in recall_cols if c in df.columns]

    if not metric_columns or "run_timestamp" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Required columns not found", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(metric_columns)))

    for col, color in zip(metric_columns, colors):
        if col not in df.columns:
            continue

        # Group by timestamp and take mean if multiple models
        grouped = df.groupby("run_timestamp")[col].mean().reset_index()

        ax.plot(
            grouped["run_timestamp"],
            grouped[col],
            "o-",
            color=color,
            label=col,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Run Date", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison_trends(
    metrics_df: pd.DataFrame,
    metric: str = "f1_macro",
    output_path: Optional[str | Path] = None,
    title: str = "Model Performance Comparison Over Time",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Plot performance trends comparing different models.

    Args:
        metrics_df: DataFrame with historical metrics
        metric: Metric column to compare
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if metrics_df.empty or metric not in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Metric '{metric}' not found", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    df = metrics_df.copy()

    # Get unique models
    if "model" not in df.columns:
        df["model"] = "Unknown"

    models = df["model"].unique()

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for model, color in zip(models, colors):
        model_df = df[df["model"] == model].sort_values("run_timestamp")

        if len(model_df) > 0:
            ax.plot(
                model_df["run_timestamp"],
                model_df[metric],
                "o-",
                color=color,
                label=model,
                linewidth=2,
                markersize=6,
                alpha=0.8,
            )

    ax.set_xlabel("Run Date", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_class_recall_heatmap(
    metrics_df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    title: str = "Per-Class Recall Over Time",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Plot heatmap of per-class recall over time.

    Args:
        metrics_df: DataFrame with historical metrics
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    if metrics_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No historical metrics available", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    df = metrics_df.copy()

    # Find per-class recall columns
    recall_cols = [c for c in df.columns if c.startswith("recall_class_")]

    if not recall_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No per-class recall columns found", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    # Aggregate by timestamp
    if "run_timestamp" in df.columns:
        pivot_data = df.groupby("run_timestamp")[recall_cols].mean()
    else:
        pivot_data = df[recall_cols]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(pivot_data.T.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(len(recall_cols)))
    ax.set_yticklabels([c.replace("recall_class_", "Class ") for c in recall_cols])

    if "run_timestamp" in df.columns:
        n_ticks = min(10, len(pivot_data))
        tick_indices = np.linspace(0, len(pivot_data) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(
            [pivot_data.index[i].strftime("%Y-%m-%d") for i in tick_indices], rotation=45, ha="right"
        )

    ax.set_xlabel("Run Date", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Recall")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
