"""Evaluation metrics and visualization for crash count regression.

Provides:
- Standard regression metrics (MAE, RMSE, MAPE, R²)
- Per-zone breakdown
- Actual vs predicted plots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metric name -> value.
    """
    # Ensure non-negative predictions
    y_pred = np.maximum(y_pred, 0)

    # Handle edge cases
    n = len(y_true)
    if n == 0:
        return {}

    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Mean Absolute Percentage Error (avoid division by zero)
    mask = y_true > 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    # R² (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Additional metrics
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Symmetric MAPE (handles zeros in both)
    smape = np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "mean_true": mean_true,
        "mean_pred": mean_pred,
        "n_samples": n,
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zone_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate regression model with optional per-zone breakdown.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        zone_ids: Optional zone IDs for per-zone metrics.

    Returns:
        Dictionary with overall and per-zone metrics.
    """
    results: dict[str, Any] = {}

    # Overall metrics
    overall = compute_metrics(y_true, y_pred)
    results["overall"] = overall

    logger.info("Overall Metrics:")
    logger.info(f"  MAE:   {overall['mae']:.4f}")
    logger.info(f"  RMSE:  {overall['rmse']:.4f}")
    logger.info(f"  MAPE:  {overall['mape']:.2f}%")
    logger.info(f"  R²:    {overall['r2']:.4f}")

    # Per-zone metrics
    if zone_ids is not None:
        results["per_zone"] = {}
        unique_zones = np.unique(zone_ids)

        logger.info(f"\nPer-Zone Metrics ({len(unique_zones)} zones):")
        for zone_id in unique_zones:
            mask = zone_ids == zone_id
            zone_metrics = compute_metrics(y_true[mask], y_pred[mask])
            results["per_zone"][zone_id] = zone_metrics

            logger.info(
                f"  Zone {zone_id}: "
                f"MAE={zone_metrics['mae']:.2f}, "
                f"R²={zone_metrics['r2']:.3f}, "
                f"n={zone_metrics['n_samples']}"
            )

    return results


def create_results_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray | None = None,
    zone_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create DataFrame with predictions and errors.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        timestamps: Optional timestamps.
        zone_ids: Optional zone IDs.

    Returns:
        DataFrame with actual, predicted, error columns.
    """
    data = {
        "actual": y_true,
        "predicted": y_pred,
        "error": y_pred - y_true,
        "abs_error": np.abs(y_pred - y_true),
    }

    if timestamps is not None:
        data["timestamp"] = timestamps

    if zone_ids is not None:
        data["zone_id"] = zone_ids

    return pd.DataFrame(data)


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray | None = None,
    title: str = "Actual vs Predicted Crash Counts",
    save_path: str | Path | None = None,
) -> None:
    """Plot actual vs predicted values.

    Creates:
    1. Time series plot (if timestamps provided)
    2. Scatter plot of actual vs predicted
    3. Residuals distribution

    Args:
        y_true: True values.
        y_pred: Predicted values.
        timestamps: Optional timestamps for time series plot.
        title: Plot title.
        save_path: Path to save figure (optional).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    n_plots = 3 if timestamps is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    if n_plots == 2:
        ax_scatter, ax_residuals = axes
        ax_time = None
    else:
        ax_time, ax_scatter, ax_residuals = axes

    # Time series plot
    if ax_time is not None and timestamps is not None:
        ax_time.plot(timestamps, y_true, label="Actual", alpha=0.7)
        ax_time.plot(timestamps, y_pred, label="Predicted", alpha=0.7)
        ax_time.set_xlabel("Time")
        ax_time.set_ylabel("Crash Count")
        ax_time.set_title("Time Series")
        ax_time.legend()
        ax_time.tick_params(axis="x", rotation=45)

    # Scatter plot
    ax_scatter.scatter(y_true, y_pred, alpha=0.5, s=10)
    max_val = max(y_true.max(), y_pred.max())
    ax_scatter.plot([0, max_val], [0, max_val], "r--", label="Perfect")
    ax_scatter.set_xlabel("Actual")
    ax_scatter.set_ylabel("Predicted")
    ax_scatter.set_title("Actual vs Predicted")
    ax_scatter.legend()

    # Residuals
    residuals = y_pred - y_true
    ax_residuals.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax_residuals.axvline(x=0, color="r", linestyle="--")
    ax_residuals.set_xlabel("Residual (Pred - Actual)")
    ax_residuals.set_ylabel("Frequency")
    ax_residuals.set_title(f"Residuals (mean={np.mean(residuals):.2f})")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    plt.close()


def plot_zone_comparison(
    results: dict[str, Any],
    metric: str = "mae",
    save_path: str | Path | None = None,
) -> None:
    """Plot per-zone metric comparison.

    Args:
        results: Results dictionary from evaluate_regression.
        metric: Metric to plot ('mae', 'rmse', 'r2').
        save_path: Path to save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    if "per_zone" not in results:
        logger.warning("No per-zone results to plot")
        return

    zone_metrics = results["per_zone"]
    zones = list(zone_metrics.keys())
    values = [zone_metrics[z].get(metric, 0) for z in zones]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(zones)), values, tick_label=[f"Zone {z}" for z in zones])

    # Highlight best/worst
    if metric in ["mae", "rmse", "mape"]:
        # Lower is better
        best_idx = np.argmin(values)
        worst_idx = np.argmax(values)
    else:  # r2
        # Higher is better
        best_idx = np.argmax(values)
        worst_idx = np.argmin(values)

    bars[best_idx].set_color("green")
    bars[worst_idx].set_color("red")

    ax.set_xlabel("Zone")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Per-Zone {metric.upper()} Comparison")

    # Add overall line
    overall_val = results["overall"].get(metric, 0)
    ax.axhline(y=overall_val, color="blue", linestyle="--", label=f"Overall: {overall_val:.2f}")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    plt.close()


def generate_report(
    results: dict[str, Any],
    save_path: str | Path | None = None,
) -> str:
    """Generate text report of evaluation results.

    Args:
        results: Results dictionary from evaluate_regression.
        save_path: Path to save report.

    Returns:
        Report as string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CRASH COUNT REGRESSION EVALUATION REPORT")
    lines.append("=" * 60)

    # Overall metrics
    lines.append("\nOVERALL METRICS")
    lines.append("-" * 40)
    overall = results.get("overall", {})
    for metric, value in overall.items():
        if isinstance(value, float):
            lines.append(f"  {metric:15}: {value:.4f}")
        else:
            lines.append(f"  {metric:15}: {value}")

    # Per-zone metrics
    if "per_zone" in results:
        lines.append("\nPER-ZONE METRICS")
        lines.append("-" * 40)
        lines.append(f"{'Zone':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'N':<10}")
        lines.append("-" * 40)

        for zone_id, metrics in sorted(results["per_zone"].items()):
            lines.append(
                f"{zone_id:<10} "
                f"{metrics.get('mae', 0):<10.2f} "
                f"{metrics.get('rmse', 0):<10.2f} "
                f"{metrics.get('r2', 0):<10.3f} "
                f"{metrics.get('n_samples', 0):<10}"
            )

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)

    if save_path:
        Path(save_path).write_text(report)
        logger.info(f"Report saved to {save_path}")

    return report
