"""Baseline models for classification and regression tasks.

Provides simple baseline models using sklearn's DummyClassifier and DummyRegressor
to establish performance floors that trained models must beat.

Baselines are controlled via CLI flag `--with-baseline` (default: enabled).
Disable with `--no-baseline` to skip baseline evaluation.

Usage:
    from training.baselines import ClassificationBaseline, RegressionBaseline

    # Classification baseline
    baseline = ClassificationBaseline(strategies=["most_frequent", "stratified"])
    baseline.fit(y_train)
    baseline_metrics = baseline.evaluate(y_test, class_names=["A", "B", "C"])

    # Regression baseline
    baseline = RegressionBaseline(strategies=["mean", "median"])
    baseline.fit(y_train)
    baseline_metrics = baseline.evaluate(y_test)

    # Compare model vs baseline
    comparison = compare_to_baseline(model_metrics, baseline_metrics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


ClassificationStrategy = Literal["most_frequent", "stratified", "prior", "uniform"]
RegressionStrategy = Literal["mean", "median", "constant"]


@dataclass
class BaselineResult:
    """Container for baseline evaluation results."""

    strategy: str
    metrics: dict[str, float]

    def __repr__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"BaselineResult(strategy={self.strategy}, {metrics_str})"


class ClassificationBaseline:
    """Baseline classifier using sklearn DummyClassifier.

    Trains multiple baseline strategies and returns metrics for each.
    Common strategies:
        - "most_frequent": Always predicts the majority class
        - "stratified": Random predictions weighted by class distribution
        - "prior": Same as stratified (alias)
        - "uniform": Random uniform predictions
    """

    def __init__(
        self,
        strategies: list[ClassificationStrategy] | None = None,
        random_state: int = 42,
    ):
        """Initialize classification baseline.

        Args:
            strategies: List of baseline strategies to evaluate.
                Default: ["most_frequent", "stratified"]
            random_state: Random seed for reproducibility.
        """
        self.strategies = strategies or ["most_frequent", "stratified"]
        self.random_state = random_state
        self._models: dict[str, DummyClassifier] = {}
        self._is_fitted = False

    def fit(self, y_train: np.ndarray) -> "ClassificationBaseline":
        """Fit baseline models on training labels.

        Args:
            y_train: Training labels.

        Returns:
            Self for chaining.
        """
        # Create dummy X (DummyClassifier ignores features but requires X)
        X_dummy = np.zeros((len(y_train), 1))

        for strategy in self.strategies:
            model = DummyClassifier(strategy=strategy, random_state=self.random_state)
            model.fit(X_dummy, y_train)
            self._models[strategy] = model

        self._is_fitted = True
        return self

    def predict(self, n_samples: int, strategy: str | None = None) -> np.ndarray:
        """Generate baseline predictions.

        Args:
            n_samples: Number of predictions to generate.
            strategy: Which strategy to use. Default: first strategy.

        Returns:
            Predicted labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")

        strategy = strategy or self.strategies[0]
        model = self._models[strategy]
        X_dummy = np.zeros((n_samples, 1))
        return model.predict(X_dummy)

    def evaluate(
        self,
        y_test: np.ndarray,
        class_names: list[str] | None = None,
    ) -> dict[str, BaselineResult]:
        """Evaluate all baseline strategies on test data.

        Args:
            y_test: True test labels.
            class_names: Optional class names for per-class metrics.

        Returns:
            Dictionary mapping strategy name to BaselineResult.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before evaluate()")

        results = {}
        X_dummy = np.zeros((len(y_test), 1))

        for strategy, model in self._models.items():
            y_pred = model.predict(X_dummy)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }

            # Per-class recall if class names provided
            if class_names:
                unique_classes = sorted(np.unique(y_test))
                for i, class_name in enumerate(class_names):
                    if i < len(unique_classes):
                        class_mask = y_test == unique_classes[i]
                        if np.sum(class_mask) > 0:
                            class_recall = np.mean(y_pred[class_mask] == unique_classes[i])
                            metrics[f"recall_{class_name}"] = class_recall

            results[strategy] = BaselineResult(strategy=strategy, metrics=metrics)

        return results

    def log_results(
        self,
        results: dict[str, BaselineResult],
        prefix: str = "BASELINE",
    ) -> None:
        """Log baseline evaluation results.

        Args:
            results: Results from evaluate().
            prefix: Logging prefix.
        """
        logger.info("=" * 60)
        logger.info(f"{prefix} EVALUATION")
        logger.info("=" * 60)

        for strategy, result in results.items():
            logger.info(f"\n{strategy.upper()} baseline:")
            for metric, value in result.metrics.items():
                logger.info(f"  {metric}: {value:.4f}")


class RegressionBaseline:
    """Baseline regressor using sklearn DummyRegressor.

    Trains multiple baseline strategies and returns metrics for each.
    Common strategies:
        - "mean": Always predicts training mean
        - "median": Always predicts training median
    """

    def __init__(
        self,
        strategies: list[RegressionStrategy] | None = None,
    ):
        """Initialize regression baseline.

        Args:
            strategies: List of baseline strategies to evaluate.
                Default: ["mean", "median"]
        """
        self.strategies = strategies or ["mean", "median"]
        self._models: dict[str, DummyRegressor] = {}
        self._is_fitted = False

    def fit(self, y_train: np.ndarray) -> "RegressionBaseline":
        """Fit baseline models on training targets.

        Args:
            y_train: Training target values.

        Returns:
            Self for chaining.
        """
        X_dummy = np.zeros((len(y_train), 1))

        for strategy in self.strategies:
            model = DummyRegressor(strategy=strategy)
            model.fit(X_dummy, y_train)
            self._models[strategy] = model

        self._is_fitted = True
        return self

    def predict(self, n_samples: int, strategy: str | None = None) -> np.ndarray:
        """Generate baseline predictions.

        Args:
            n_samples: Number of predictions to generate.
            strategy: Which strategy to use. Default: first strategy.

        Returns:
            Predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")

        strategy = strategy or self.strategies[0]
        model = self._models[strategy]
        X_dummy = np.zeros((n_samples, 1))
        return model.predict(X_dummy)

    def evaluate(self, y_test: np.ndarray) -> dict[str, BaselineResult]:
        """Evaluate all baseline strategies on test data.

        Args:
            y_test: True test values.

        Returns:
            Dictionary mapping strategy name to BaselineResult.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before evaluate()")

        results = {}
        X_dummy = np.zeros((len(y_test), 1))

        for strategy, model in self._models.items():
            y_pred = model.predict(X_dummy)

            # Ensure non-negative for count predictions
            y_pred = np.maximum(y_pred, 0)

            # Compute metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # MAPE (avoid division by zero)
            mask = y_test > 0
            if np.any(mask):
                mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            else:
                mape = np.nan

            metrics = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
                "mean_pred": float(np.mean(y_pred)),
            }

            results[strategy] = BaselineResult(strategy=strategy, metrics=metrics)

        return results

    def log_results(
        self,
        results: dict[str, BaselineResult],
        prefix: str = "BASELINE",
    ) -> None:
        """Log baseline evaluation results.

        Args:
            results: Results from evaluate().
            prefix: Logging prefix.
        """
        logger.info("=" * 60)
        logger.info(f"{prefix} EVALUATION")
        logger.info("=" * 60)

        for strategy, result in results.items():
            logger.info(f"\n{strategy.upper()} baseline:")
            for metric, value in result.metrics.items():
                if np.isnan(value):
                    logger.info(f"  {metric}: N/A")
                else:
                    logger.info(f"  {metric}: {value:.4f}")


def compare_to_baseline(
    model_metrics: dict[str, float],
    baseline_results: dict[str, BaselineResult],
    key_metrics: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compare model metrics to baseline results.

    Args:
        model_metrics: Dictionary of model metric values.
        baseline_results: Baseline results from evaluate().
        key_metrics: Metrics to compare. Default: all available.

    Returns:
        Dictionary with improvement percentages for each baseline strategy.
        Positive values indicate model is better than baseline.
    """
    if key_metrics is None:
        # Auto-detect shared metrics
        key_metrics = list(model_metrics.keys())

    comparison = {}

    for strategy, baseline in baseline_results.items():
        improvements = {}

        for metric in key_metrics:
            if metric not in model_metrics or metric not in baseline.metrics:
                continue

            model_val = model_metrics[metric]
            baseline_val = baseline.metrics[metric]

            if baseline_val == 0:
                continue

            # For error metrics (MAE, RMSE), lower is better
            # For score metrics (accuracy, F1, R²), higher is better
            error_metrics = {"mae", "rmse", "mape", "smape"}

            if metric.lower() in error_metrics:
                # Improvement = baseline - model (positive means model is better)
                improvement_pct = (baseline_val - model_val) / baseline_val * 100
            else:
                # Improvement = model - baseline (positive means model is better)
                improvement_pct = (model_val - baseline_val) / baseline_val * 100

            improvements[metric] = improvement_pct

        comparison[strategy] = improvements

    return comparison


def log_comparison(
    comparison: dict[str, dict[str, float]],
    model_name: str = "Model",
) -> None:
    """Log model vs baseline comparison.

    Args:
        comparison: Results from compare_to_baseline().
        model_name: Name of the model for logging.
    """
    logger.info("=" * 60)
    logger.info(f"{model_name.upper()} VS BASELINE COMPARISON")
    logger.info("=" * 60)

    for strategy, improvements in comparison.items():
        logger.info(f"\nVs {strategy.upper()} baseline:")
        for metric, improvement in improvements.items():
            sign = "+" if improvement > 0 else ""
            status = "better" if improvement > 0 else "worse"
            logger.info(f"  {metric}: {sign}{improvement:.1f}% ({status})")


def format_baseline_report(
    model_metrics: dict[str, float],
    baseline_results: dict[str, BaselineResult],
    comparison: dict[str, dict[str, float]],
) -> str:
    """Format baseline comparison as text report.

    Args:
        model_metrics: Model metric values.
        baseline_results: Baseline results.
        comparison: Comparison results.

    Returns:
        Formatted text report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("BASELINE COMPARISON REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Model metrics
    lines.append("MODEL METRICS:")
    for metric, value in model_metrics.items():
        lines.append(f"  {metric}: {value:.4f}")
    lines.append("")

    # Baseline metrics
    lines.append("BASELINE METRICS:")
    for strategy, result in baseline_results.items():
        lines.append(f"\n  {strategy.upper()}:")
        for metric, value in result.metrics.items():
            if np.isnan(value):
                lines.append(f"    {metric}: N/A")
            else:
                lines.append(f"    {metric}: {value:.4f}")
    lines.append("")

    # Improvements
    lines.append("IMPROVEMENT OVER BASELINE:")
    for strategy, improvements in comparison.items():
        lines.append(f"\n  vs {strategy.upper()}:")
        for metric, improvement in improvements.items():
            sign = "+" if improvement > 0 else ""
            status = "✓" if improvement > 0 else "✗"
            lines.append(f"    {metric}: {sign}{improvement:.1f}% {status}")

    return "\n".join(lines)


def print_baseline_comparison_box(
    model_metrics: dict[str, float],
    baseline_results: dict[str, BaselineResult],
    comparison: dict[str, dict[str, float]],
    model_name: str = "Model",
    baseline_strategy: str = "most_frequent",
    primary_metric: str | None = None,
    metric_labels: dict[str, str] | None = None,
    task_type: str = "classification",
) -> None:
    """Print a formatted box comparing model vs baseline metrics.

    Args:
        model_metrics: Dictionary of model metric values.
        baseline_results: Baseline results from evaluate().
        comparison: Comparison results from compare_to_baseline().
        model_name: Display name for the model.
        baseline_strategy: Which baseline strategy to compare against.
        primary_metric: Key metric to highlight with ★ (e.g., "recall_SEVERE").
        metric_labels: Optional display names for metrics.
        task_type: "classification" or "regression" (affects metric interpretation).
    """
    # Get the baseline to compare against
    if baseline_strategy not in baseline_results:
        baseline_strategy = list(baseline_results.keys())[0]
    
    baseline = baseline_results[baseline_strategy]
    improvements = comparison.get(baseline_strategy, {})
    
    # Default metric labels
    if metric_labels is None:
        metric_labels = {}
    
    # Determine which metrics to show
    metrics_to_show = [m for m in model_metrics.keys() if m in baseline.metrics]
    
    # Error metrics (lower is better)
    error_metrics = {"mae", "rmse", "mape", "smape"}
    
    # Build table data
    rows = []
    wins = 0
    total = 0
    
    for metric in metrics_to_show:
        model_val = model_metrics[metric]
        baseline_val = baseline.metrics.get(metric, 0)
        improvement = improvements.get(metric, 0)
        
        # Determine winner
        is_error_metric = metric.lower() in error_metrics
        if is_error_metric:
            model_wins = model_val < baseline_val
        else:
            model_wins = model_val > baseline_val
        
        # Handle special cases
        if baseline_val == 0 and model_val > 0:
            improvement_str = "+∞%"
            model_wins = not is_error_metric  # For non-error metrics, higher is better
        elif baseline_val == 0:
            improvement_str = "N/A"
        else:
            sign = "+" if improvement > 0 else ""
            improvement_str = f"{sign}{improvement:.1f}%"
        
        # Display label
        label = metric_labels.get(metric, metric)
        is_primary = metric == primary_metric
        if is_primary:
            label = f"{label} ★"
        
        winner = "✓ Model" if model_wins else "✗ Base"
        
        rows.append({
            "label": label,
            "model": model_val,
            "baseline": baseline_val,
            "improvement": improvement_str,
            "winner": winner,
            "model_wins": model_wins,
        })
        
        if baseline_val > 0 or model_val > 0:
            total += 1
            if model_wins:
                wins += 1
    
    # Calculate column widths
    label_width = max(len(r["label"]) for r in rows) + 2
    label_width = max(label_width, 16)
    
    # Box characters
    TL, TR, BL, BR = "╔", "╗", "╚", "╝"
    H, V = "═", "║"
    LT, RT, TT, BT, X = "╠", "╣", "╦", "╩", "╬"
    HL, VL = "─", "│"
    
    # Total width
    total_width = 68
    
    # Print header
    print()
    print(f"{TL}{H * (total_width - 2)}{TR}")
    
    title = f"MODEL VS BASELINE COMPARISON ({baseline_strategy.upper()})"
    padding = (total_width - 2 - len(title)) // 2
    print(f"{V}{' ' * padding}{title}{' ' * (total_width - 2 - padding - len(title))}{V}")
    
    print(f"{LT}{H * (total_width - 2)}{RT}")
    
    # Column headers
    header = f"{V}  {'Metric':<{label_width}} {VL} {'Model':^8} {VL} {'Baseline':^8} {VL} {'Change':^10} {VL} {'Winner':^8} {V}"
    print(header)
    
    print(f"{LT}{HL * (label_width + 2)}{X}{HL * 10}{X}{HL * 10}{X}{HL * 12}{X}{HL * 10}{RT}")
    
    # Data rows
    for row in rows:
        model_str = f"{row['model']:.4f}" if isinstance(row['model'], float) else str(row['model'])
        baseline_str = f"{row['baseline']:.4f}" if isinstance(row['baseline'], float) else str(row['baseline'])
        
        line = f"{V}  {row['label']:<{label_width}} {VL} {model_str:^8} {VL} {baseline_str:^8} {VL} {row['improvement']:^10} {VL} {row['winner']:^8} {V}"
        print(line)
    
    # Footer
    print(f"{LT}{H * (total_width - 2)}{RT}")
    
    # Primary metric note
    if primary_metric:
        note = "★ = Primary metric for this task"
        print(f"{V}  {note:<{total_width - 4}}{V}")
    
    # Summary
    if total > 0:
        result = "MODEL WINS" if wins > total / 2 else "BASELINE WINS"
        summary = f"Result: {result} on {wins}/{total} metrics"
        print(f"{V}  {summary:<{total_width - 4}}{V}")
    
    print(f"{BL}{H * (total_width - 2)}{BR}")
    print()


def add_baseline_args(parser: "argparse.ArgumentParser") -> None:
    """Add baseline-related CLI arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to.
    """
    import argparse

    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--with-baseline",
        action="store_true",
        default=True,
        help="Run baseline evaluation (default: enabled)",
    )
    baseline_group.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline evaluation",
    )
