"""Crash count regression package.

Neural network regression models for predicting crash counts
within time windows, with zone-based ensemble support.
"""

from training.regression.config import RegressionConfig
from training.regression.models import CrashCountMLP, ZoneAdjustmentMLP
from training.regression.ensemble import CrashCountEnsemble
from training.regression.trainer import RegressionTrainer
from training.regression.dataset import TimeSeriesDataset
from training.regression.evaluation import (
    evaluate_regression,
    compute_metrics,
    plot_predictions,
)
from training.regression.predict import CrashCountPredictor

__all__ = [
    "RegressionConfig",
    "CrashCountMLP",
    "ZoneAdjustmentMLP",
    "CrashCountEnsemble",
    "RegressionTrainer",
    "TimeSeriesDataset",
    "evaluate_regression",
    "compute_metrics",
    "plot_predictions",
    "CrashCountPredictor",
]
