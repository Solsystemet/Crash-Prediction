"""
Data quality reporting and visualization utilities.

Provides tools for analyzing and visualizing data quality issues like
missing values, class distributions, and schema compliance.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.csv_filename_generator import generate_csv_filename


def analyze_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze missing values in a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with missingness statistics per column
    """
    total_rows = len(df)
    stats = []

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0

        stats.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": missing_count,
                "missing_pct": round(missing_pct, 2),
                "non_null_count": total_rows - missing_count,
                "n_unique": df[col].nunique(dropna=True),
            }
        )

    return pd.DataFrame(stats).sort_values("missing_pct", ascending=False)


def plot_missingness_heatmap(
    df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    max_cols: int = 50,
    title: str = "Missing Values Heatmap",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Plot a heatmap showing missing value patterns.

    Args:
        df: Input DataFrame
        output_path: Path to save the plot
        max_cols: Maximum number of columns to show
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Select columns with most missing values
    missing_counts = df.isna().sum()
    cols_with_missing = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if len(cols_with_missing) == 0:
        # No missing values - create simple message plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No missing values detected!", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    # Limit columns
    selected_cols = cols_with_missing.head(max_cols).index.tolist()

    # Create binary missingness matrix
    missing_matrix = df[selected_cols].isna().astype(int)

    # Sample rows if too many
    max_rows = 200
    if len(missing_matrix) > max_rows:
        sample_idx = np.linspace(0, len(missing_matrix) - 1, max_rows, dtype=int)
        missing_matrix = missing_matrix.iloc[sample_idx]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(missing_matrix.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xlabel("Column", fontsize=12)
    ax.set_ylabel("Row (sampled)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Set x-tick labels (column names)
    ax.set_xticks(range(len(selected_cols)))
    ax.set_xticklabels(selected_cols, rotation=45, ha="right", fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Missing (1) / Present (0)")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_missingness_bars(
    df: pd.DataFrame,
    output_path: Optional[str | Path] = None,
    top_n: int = 20,
    title: str = "Missing Values by Column",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot bar chart of missing value percentages.

    Args:
        df: Input DataFrame
        output_path: Path to save the plot
        top_n: Number of top columns to show
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    missing_pct = (df.isna().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)

    if len(missing_pct) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No missing values detected!", ha="center", va="center", fontsize=14)
        ax.axis("off")
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return fig

    # Take top N
    missing_pct = missing_pct.tail(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.YlOrRd(missing_pct.values / 100)
    bars = ax.barh(missing_pct.index, missing_pct.values, color=colors, edgecolor="black", linewidth=0.5)

    # Add percentage labels
    for bar, pct in zip(bars, missing_pct.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Missing (%)", fontsize=12)
    ax.set_ylabel("Column", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim([0, max(missing_pct.values) * 1.15])
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_class_distribution(
    labels: np.ndarray | pd.Series,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
    title: str = "Class Distribution",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot class distribution bar chart.

    Args:
        labels: Class labels
        class_names: Names for each class
        output_path: Path to save the plot
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    percentages = (counts / total) * 100

    if class_names is None:
        class_names = [str(u) for u in unique]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 1, len(unique)))
    bars = ax.bar(class_names, counts, color=colors, edgecolor="black", linewidth=0.5)

    # Add count and percentage labels
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + total * 0.01, f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add imbalance ratio annotation
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
    ax.text(
        0.98,
        0.98,
        f"Imbalance ratio: {imbalance_ratio:.1f}:1",
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

    return fig


def generate_data_quality_report(
    df: pd.DataFrame,
    output_dir: str | Path,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report.

    Creates:
    - missingness.csv - Missing value statistics per column
    - missingness_heatmap.png - Visual missing pattern
    - missingness_bars.png - Bar chart of missing percentages
    - schema_summary.csv - Column schema information
    - class_distribution.png - Target class distribution (if target_column provided)
    - report_summary.json - Overall statistics

    Args:
        df: Input DataFrame
        output_dir: Directory to save report files
        target_column: Name of target column for class distribution plot

    Returns:
        Dict containing summary statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Missingness analysis
    missingness_df = analyze_missingness(df)
    missingness_filename = generate_csv_filename("missingness", "data_quality")
    missingness_df.to_csv(output_dir / missingness_filename, index=False)

    # 2. Missingness visualizations
    plot_missingness_heatmap(df, output_dir / "missingness_heatmap.png")
    plot_missingness_bars(df, output_dir / "missingness_bars.png")

    # 3. Schema summary
    schema_data = []
    for col in df.columns:
        schema_data.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": df[col].nunique(dropna=True),
                "n_null": df[col].isna().sum(),
                "sample_values": str(df[col].dropna().head(3).tolist()),
            }
        )
    schema_df = pd.DataFrame(schema_data)
    schema_filename = generate_csv_filename("schema_summary", "data_quality")
    schema_df.to_csv(output_dir / schema_filename, index=False)

    # 4. Class distribution (if target column provided)
    if target_column and target_column in df.columns:
        plot_class_distribution(
            df[target_column].dropna(), output_path=output_dir / "class_distribution.png"
        )

    # 5. Summary statistics
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "total_cells": int(total_cells),
        "missing_cells": int(missing_cells),
        "missing_pct": round((missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0,
        "columns_with_missing": int((df.isna().sum() > 0).sum()),
        "complete_columns": int((df.isna().sum() == 0).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    with open(output_dir / "report_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
