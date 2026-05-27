"""
Plotting utilities for training visualization.

Provides standardized plotting functions for:
- Training curves (loss/metrics over epochs)
- Feature importance
- Calibration diagrams
- Confidence distributions
- Data quality reports
"""

from training.plotting.training_curves import (
    plot_training_curves,
    plot_training_curves_from_history,
)
from training.plotting.feature_importance import (
    plot_feature_importance,
    get_feature_importance,
)
from training.plotting.calibration import (
    plot_calibration_curve,
    compute_calibration_metrics,
)
from training.plotting.confidence import (
    plot_confidence_histogram,
    plot_confidence_by_class,
)
from training.plotting.data_quality import (
    generate_data_quality_report,
    plot_missingness_heatmap,
    plot_class_distribution,
)
from training.plotting.recall_trends import (
    plot_recall_trends,
    load_historical_metrics,
)

__all__ = [
    # Training curves
    "plot_training_curves",
    "plot_training_curves_from_history",
    # Feature importance
    "plot_feature_importance",
    "get_feature_importance",
    # Calibration
    "plot_calibration_curve",
    "compute_calibration_metrics",
    # Confidence
    "plot_confidence_histogram",
    "plot_confidence_by_class",
    # Data quality
    "generate_data_quality_report",
    "plot_missingness_heatmap",
    "plot_class_distribution",
    # Recall trends
    "plot_recall_trends",
    "load_historical_metrics",
]
