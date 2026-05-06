"""Multiclass focal loss training pipeline for crash severity prediction.

This package implements a 5-class severity model using:
- Focal Loss and LDAM Loss for handling class imbalance
- Neural networks and gradient boosting models
- Probability calibration for risk scoring
- Class-specific threshold optimization

Classes:
    NO INDICATION OF INJURY (~86%)
    REPORTED, NOT EVIDENT (~4-5%)
    NONINCAPACITATING INJURY (~8-10%)
    INCAPACITATING INJURY (~1-2%)
    FATAL (~0.1%)

Usage:
    python -m training.multiclass_focal.main --sample 50000
    python -m training.multiclass_focal.main --loss focal --model all
"""

from training.multiclass_focal.losses import FocalLoss, LDAMLoss, get_loss_function
from training.multiclass_focal.model import SeverityMLP, SeverityMLPConfig, MCDropoutWrapper
from training.multiclass_focal.sampler import (
    ClassBalancedSampler,
    SquareRootSampler,
    ProgressiveBalancedSampler,
    get_sampler,
)
from training.multiclass_focal.calibration import (
    CalibratedClassifier,
    plot_reliability_diagram,
    expected_calibration_error,
    optimize_thresholds,
    predict_with_thresholds,
)
from training.multiclass_focal.evaluation import (
    evaluate_predictions,
    EvaluationResult,
    print_evaluation_summary,
    compare_models,
    SEVERITY_CLASS_ORDER,
)

__all__ = [
    # Losses
    "FocalLoss",
    "LDAMLoss",
    "get_loss_function",
    # Model
    "SeverityMLP",
    "SeverityMLPConfig",
    "MCDropoutWrapper",
    # Sampling
    "ClassBalancedSampler",
    "SquareRootSampler",
    "ProgressiveBalancedSampler",
    "get_sampler",
    # Calibration
    "CalibratedClassifier",
    "plot_reliability_diagram",
    "expected_calibration_error",
    "optimize_thresholds",
    "predict_with_thresholds",
    # Evaluation
    "evaluate_predictions",
    "EvaluationResult",
    "print_evaluation_summary",
    "compare_models",
    "SEVERITY_CLASS_ORDER",
]
