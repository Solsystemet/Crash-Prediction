"""Accuracy evaluation service for comparing predictions against real data.

This module handles fetching real crash data from the Chicago API,
running predictions on it, and computing accuracy metrics.

Supports multiple model types:
- simplified: 3-class (NO_INJURY, MINOR, SEVERE)
- hierarchical: 5-class (NO_INJURY, REPORTED_NOT_EVIDENT, NONINCAPACITATING, INCAPACITATING, FATAL)
- zones: 3-class with geographic zone assignment
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from api.chicago_client import fetch_crash_data_for_accuracy, ChicagoAPIError
from api.data_transformer import (
    transform_all_crashes,
    extract_ground_truth,
    extract_ground_truth_5class,
    parse_crash_datetime,
    _safe_float,
)
from api.prediction import (
    predict,
    predict_hierarchical,
    predict_zones,
    predict_multiclass_nn,
    model_manager,
)
from api.models import PredictionRequest
from api.config import MODEL_REGISTRY, DEFAULT_MODEL

logger = logging.getLogger(__name__)

# Class labels for different model types
CLASS_LABELS_3CLASS = ["NO_INJURY", "MINOR", "SEVERE"]
CLASS_LABELS_5CLASS = [
    "NO_INJURY",
    "REPORTED_NOT_EVIDENT",
    "NONINCAPACITATING",
    "INCAPACITATING",
    "FATAL",
]

# Backwards compatibility
CLASS_LABELS = CLASS_LABELS_3CLASS


def get_class_labels_for_model(model_name: str) -> list[str]:
    """Get the appropriate class labels for a model.

    Args:
        model_name: Name of the model in the registry.

    Returns:
        List of class labels for this model type.
    """
    if model_name not in MODEL_REGISTRY:
        return CLASS_LABELS_3CLASS

    model_type = MODEL_REGISTRY[model_name].model_type
    if model_type in ("hierarchical", "multiclass_nn"):
        return CLASS_LABELS_5CLASS
    return CLASS_LABELS_3CLASS


def compute_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = CLASS_LABELS,
) -> list[list[int]]:
    """Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels in order

    Returns:
        Confusion matrix as 2D list [actual][predicted]
    """
    n_classes = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    matrix = [[0] * n_classes for _ in range(n_classes)]

    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            true_idx = label_to_idx[true]
            pred_idx = label_to_idx[pred]
            matrix[true_idx][pred_idx] += 1

    return matrix


def compute_class_metrics(
    confusion_matrix: list[list[int]],
    labels: list[str] = CLASS_LABELS,
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, and F1 for each class.

    Args:
        confusion_matrix: Confusion matrix [actual][predicted]
        labels: List of class labels

    Returns:
        Dictionary mapping class name to metrics dict
    """
    metrics = {}
    n_classes = len(labels)

    for i, label in enumerate(labels):
        # True positives: diagonal element
        tp = confusion_matrix[i][i]

        # False positives: sum of column i minus true positives
        fp = sum(confusion_matrix[j][i] for j in range(n_classes)) - tp

        # False negatives: sum of row i minus true positives
        fn = sum(confusion_matrix[i][j] for j in range(n_classes)) - tp

        # Support: total actual instances of this class
        support = sum(confusion_matrix[i])

        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }

    return metrics


def compute_overall_accuracy(
    y_true: list[str],
    y_pred: list[str],
) -> float:
    """Compute overall accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Accuracy as float between 0 and 1
    """
    if not y_true:
        return 0.0

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return round(correct / len(y_true), 4)


def compute_f1_scores(
    class_metrics: dict[str, dict[str, float]],
    labels: list[str] = CLASS_LABELS,
) -> dict[str, float]:
    """Compute macro and micro F1 scores.

    Args:
        class_metrics: Per-class metrics from compute_class_metrics
        labels: List of class labels

    Returns:
        Dictionary with f1_macro and f1_micro
    """
    # Macro F1: simple average of per-class F1 scores
    f1_scores = [
        class_metrics[label]["f1_score"] for label in labels if label in class_metrics
    ]
    f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Micro F1: compute from total TP, FP, FN across all classes
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in labels:
        if label not in class_metrics:
            continue
        metrics = class_metrics[label]
        support = metrics["support"]
        precision = metrics["precision"]
        recall = metrics["recall"]

        # Reverse-engineer TP, FP, FN from precision, recall, support
        # recall = TP / (TP + FN), and support = TP + FN
        # So TP = recall * support
        tp = recall * support
        fn = support - tp

        # precision = TP / (TP + FP)
        # So FP = TP / precision - TP (if precision > 0)
        if precision > 0:
            fp = tp / precision - tp
        else:
            fp = 0

        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Micro precision and recall
    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )

    # Micro F1
    f1_micro = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "f1_macro": round(f1_macro, 4),
        "f1_micro": round(f1_micro, 4),
    }


def evaluate_accuracy(
    days: int = 7,
    max_crashes: int = 10000,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate model accuracy on recent real crash data.

    Args:
        days: Number of days back to fetch data
        max_crashes: Maximum number of crashes to evaluate
        model_name: Name of the model to evaluate (from MODEL_REGISTRY).
                   If None, uses DEFAULT_MODEL.

    Returns:
        Dictionary containing accuracy metrics and individual predictions

    Raises:
        RuntimeError: If the model is not trained/available
    """
    # Resolve model name
    if model_name is None:
        model_name = DEFAULT_MODEL

    # Validate model exists
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    model_info = MODEL_REGISTRY[model_name]
    model_type = model_info.model_type

    logger.info(
        f"Starting accuracy evaluation (model={model_name}, type={model_type}, "
        f"days={days}, max_crashes={max_crashes})"
    )

    # Load the model
    try:
        model_manager.load_model(model_name)
    except ValueError as e:
        raise RuntimeError(
            f"Model '{model_name}' not available. Please train the model first."
        ) from e

    # Determine class labels for this model
    class_labels = get_class_labels_for_model(model_name)

    # Fetch data from Chicago API
    try:
        data = fetch_crash_data_for_accuracy(days=days, max_crashes=max_crashes)
    except ChicagoAPIError as e:
        logger.error(f"Failed to fetch data from Chicago API: {e}")
        raise RuntimeError(f"Failed to fetch crash data: {e}")

    crashes = data["crashes"]
    if not crashes:
        return {
            "metrics": {
                "overall_accuracy": 0.0,
                "sample_count": 0,
                "per_class_metrics": {},
                "confusion_matrix": [],
                "class_labels": class_labels,
                "time_range_days": days,
                "computed_at": datetime.now().isoformat(),
                "f1_macro": 0.0,
                "f1_micro": 0.0,
                "model_name": model_name,
            },
            "predictions": [],
        }

    # Transform crashes to prediction requests
    transformed = transform_all_crashes(
        crashes=crashes,
        people=data["people"],
        vehicles=data["vehicles"],
        weather=data["weather"],
    )

    if not transformed:
        return {
            "metrics": {
                "overall_accuracy": 0.0,
                "sample_count": 0,
                "per_class_metrics": {},
                "confusion_matrix": [],
                "class_labels": class_labels,
                "time_range_days": days,
                "computed_at": datetime.now().isoformat(),
                "f1_macro": 0.0,
                "f1_micro": 0.0,
                "model_name": model_name,
            },
            "predictions": [],
        }

    # Make predictions based on model type
    y_true = []
    y_pred = []
    predictions = []

    for crash, request, ground_truth_3class in transformed:
        try:
            # Get ground truth based on model type
            if model_type in ("hierarchical", "multiclass_nn"):
                ground_truth = extract_ground_truth_5class(crash)
                if ground_truth is None:
                    continue
            else:
                ground_truth = ground_truth_3class

            # Get prediction based on model type
            if model_type == "simplified":
                response = predict(request, model_name)
                predicted = response.prediction
                confidence = response.confidence
            elif model_type == "hierarchical":
                response = predict_hierarchical(request)
                predicted = response.prediction
                confidence = response.confidence
            elif model_type == "zones":
                # Zone model requires lat/lng
                if request.latitude is None or request.longitude is None:
                    continue
                response = predict_zones(request)
                predicted = response.prediction
                confidence = response.confidence
            elif model_type == "multiclass_nn":
                response = predict_multiclass_nn(request)
                predicted = response.prediction
                confidence = response.confidence
            else:
                # Skip unsupported model types (e.g., regression)
                continue

            y_true.append(ground_truth)
            y_pred.append(predicted)

            # Parse crash date
            crash_dt = parse_crash_datetime(crash)

            # Build prediction record
            predictions.append(
                {
                    "crash_record_id": crash.get("crash_record_id", ""),
                    "crash_date": crash_dt.isoformat() if crash_dt else None,
                    "predicted_severity": predicted,
                    "actual_severity": ground_truth,
                    "is_correct": predicted == ground_truth,
                    "confidence": confidence,
                    "latitude": _safe_float(crash.get("latitude")),
                    "longitude": _safe_float(crash.get("longitude")),
                    "weather_condition": crash.get("weather_condition"),
                    "lighting_condition": crash.get("lighting_condition"),
                    "first_crash_type": crash.get("first_crash_type"),
                    "posted_speed_limit": crash.get("posted_speed_limit"),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to predict for crash: {e}")
            continue

    # Compute metrics with appropriate class labels
    confusion_matrix = compute_confusion_matrix(y_true, y_pred, labels=class_labels)
    class_metrics = compute_class_metrics(confusion_matrix, labels=class_labels)
    overall_accuracy = compute_overall_accuracy(y_true, y_pred)
    f1_scores = compute_f1_scores(class_metrics, labels=class_labels)

    logger.info(
        f"Accuracy evaluation complete: {len(predictions)} predictions, "
        f"accuracy={overall_accuracy:.2%}, model={model_name}"
    )

    return {
        "metrics": {
            "overall_accuracy": overall_accuracy,
            "sample_count": len(predictions),
            "per_class_metrics": class_metrics,
            "confusion_matrix": confusion_matrix,
            "class_labels": class_labels,
            "time_range_days": days,
            "computed_at": datetime.now().isoformat(),
            "f1_macro": f1_scores["f1_macro"],
            "f1_micro": f1_scores["f1_micro"],
            "model_name": model_name,
        },
        "predictions": predictions,
    }


def evaluate_all_models(
    days: int = 7,
    max_crashes: int = 2000,
) -> dict[str, Any]:
    """Evaluate all classification models for comparison.

    Args:
        days: Number of days back to fetch data
        max_crashes: Maximum number of crashes to evaluate (lower for comparison)

    Returns:
        Dictionary with comparison results for each model
    """
    # Only compare classification models, not regression
    classification_models = [
        name
        for name, info in MODEL_REGISTRY.items()
        if info.model_type in ("simplified", "hierarchical", "zones", "multiclass_nn")
    ]

    results = {}

    for model_name in classification_models:
        try:
            logger.info(f"Evaluating model: {model_name}")
            result = evaluate_accuracy(
                days=days,
                max_crashes=max_crashes,
                model_name=model_name,
            )
            results[model_name] = {
                "model_name": model_name,
                "display_name": MODEL_REGISTRY[model_name].name,
                "model_type": MODEL_REGISTRY[model_name].model_type,
                "metrics": result["metrics"],
                "status": "success",
            }
        except Exception as e:
            logger.warning(f"Failed to evaluate {model_name}: {e}")
            results[model_name] = {
                "model_name": model_name,
                "display_name": MODEL_REGISTRY[model_name].name,
                "model_type": MODEL_REGISTRY[model_name].model_type,
                "metrics": None,
                "status": "error",
                "error": str(e),
            }

    return {
        "models": results,
        "time_range_days": days,
        "max_crashes": max_crashes,
        "computed_at": datetime.now().isoformat(),
    }


def get_map_data(
    days: int = 7,
    max_crashes: int = 200,
    filter_correct: bool | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Get prediction data formatted for map display.

    Args:
        days: Number of days back to fetch
        max_crashes: Maximum number of crashes
        filter_correct: If True, only correct; if False, only incorrect; if None, all
        model_name: Name of the model to use (defaults to DEFAULT_MODEL)

    Returns:
        List of prediction records with coordinates for map display
    """
    result = evaluate_accuracy(
        days=days, max_crashes=max_crashes, model_name=model_name
    )
    predictions = result["predictions"]

    # Filter based on correctness if specified
    if filter_correct is not None:
        predictions = [p for p in predictions if p["is_correct"] == filter_correct]

    # Filter to only predictions with valid coordinates
    map_data = []
    for p in predictions:
        lat = p.get("latitude")
        lon = p.get("longitude")

        # Check for valid Chicago coordinates
        if (
            lat is not None
            and lon is not None
            and 41.6 < lat < 42.1  # Chicago latitude range
            and -88.0 < lon < -87.4  # Chicago longitude range
        ):
            map_data.append(
                {
                    "crash_record_id": p["crash_record_id"],
                    "crash_date": p["crash_date"],
                    "predicted_severity": p["predicted_severity"],
                    "actual_severity": p["actual_severity"],
                    "is_correct": p["is_correct"],
                    "confidence": p["confidence"],
                    "latitude": lat,
                    "longitude": lon,
                    "weather_condition": p.get("weather_condition"),
                    "lighting_condition": p.get("lighting_condition"),
                }
            )

    return map_data


# ============================================================================
# ROC Curve Computation
# ============================================================================

# Colors for ROC curves (consistent with frontend)
ROC_COLORS_3CLASS = {
    "NO_INJURY": "#22c55e",  # Green
    "MINOR": "#f59e0b",  # Amber
    "SEVERE": "#ef4444",  # Red
}

ROC_COLORS_5CLASS = {
    "NO_INJURY": "#22c55e",  # Green
    "REPORTED_NOT_EVIDENT": "#84cc16",  # Lime
    "NONINCAPACITATING": "#f59e0b",  # Amber
    "INCAPACITATING": "#f97316",  # Orange
    "FATAL": "#ef4444",  # Red
}


def _get_probabilities_for_crash(
    request: PredictionRequest,
    model_type: str,
    model_name: str,
) -> tuple[str, list[float]] | None:
    """Get prediction and class probabilities for a single crash.

    Returns:
        Tuple of (predicted_class, probabilities) or None if failed.
        Probabilities are ordered by class labels.
    """
    try:
        if model_type == "simplified":
            response = predict(request, model_name)
            probs = [
                response.probabilities.no_injury,
                response.probabilities.minor,
                response.probabilities.severe,
            ]
            return response.prediction, probs

        elif model_type == "hierarchical":
            response = predict_hierarchical(request)
            # 5-class probabilities
            probs = [
                response.probabilities.no_injury,
                response.probabilities.reported_not_evident,
                response.probabilities.nonincapacitating,
                response.probabilities.incapacitating,
                response.probabilities.fatal,
            ]
            return response.prediction, probs

        elif model_type == "zones":
            if request.latitude is None or request.longitude is None:
                return None
            response = predict_zones(request)
            probs = [
                response.probabilities.no_injury,
                response.probabilities.minor,
                response.probabilities.severe,
            ]
            return response.prediction, probs

        elif model_type == "multiclass_nn":
            response = predict_multiclass_nn(request)
            probs = [
                response.probabilities.no_injury,
                response.probabilities.reported_not_evident,
                response.probabilities.nonincapacitating,
                response.probabilities.incapacitating,
                response.probabilities.fatal,
            ]
            return response.prediction, probs

        return None
    except Exception as e:
        logger.warning(f"Failed to get probabilities: {e}")
        return None


def compute_roc_data(
    days: int = 7,
    max_crashes: int = 2000,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Compute ROC curve data for a model using real crash data.

    Uses One-vs-Rest strategy for multi-class ROC curves.

    Args:
        days: Number of days back to fetch data
        max_crashes: Maximum number of crashes to evaluate
        model_name: Name of the model to evaluate

    Returns:
        Dictionary with ROC curve data for each class
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.preprocessing import label_binarize
    import numpy as np

    # Resolve model name
    if model_name is None:
        model_name = DEFAULT_MODEL

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    model_info = MODEL_REGISTRY[model_name]
    model_type = model_info.model_type

    logger.info(
        f"Computing ROC data (model={model_name}, type={model_type}, "
        f"days={days}, max_crashes={max_crashes})"
    )

    # Load model
    try:
        model_manager.load_model(model_name)
    except ValueError as e:
        raise RuntimeError(f"Model '{model_name}' not available.") from e

    # Get class labels
    class_labels = get_class_labels_for_model(model_name)
    n_classes = len(class_labels)
    colors = ROC_COLORS_5CLASS if n_classes == 5 else ROC_COLORS_3CLASS

    # Fetch data
    try:
        data = fetch_crash_data_for_accuracy(days=days, max_crashes=max_crashes)
    except ChicagoAPIError as e:
        raise RuntimeError(f"Failed to fetch crash data: {e}")

    crashes = data["crashes"]
    if not crashes:
        return {
            "model_name": model_name,
            "model_type": model_type,
            "curves": [],
            "computed_at": datetime.now().isoformat(),
        }

    # Transform crashes
    transformed = transform_all_crashes(
        crashes=crashes,
        people=data["people"],
        vehicles=data["vehicles"],
        weather=data["weather"],
    )

    if not transformed:
        return {
            "model_name": model_name,
            "model_type": model_type,
            "curves": [],
            "computed_at": datetime.now().isoformat(),
        }

    # Collect ground truth and probabilities
    y_true = []
    y_proba = []

    for crash, request, ground_truth_3class in transformed:
        # Get ground truth
        if model_type in ("hierarchical", "multiclass_nn"):
            ground_truth = extract_ground_truth_5class(crash)
            if ground_truth is None:
                continue
        else:
            ground_truth = ground_truth_3class

        # Get probabilities
        result = _get_probabilities_for_crash(request, model_type, model_name)
        if result is None:
            continue

        _, probs = result
        y_true.append(ground_truth)
        y_proba.append(probs)

    if len(y_true) < 10:
        logger.warning(f"Not enough samples for ROC: {len(y_true)}")
        return {
            "model_name": model_name,
            "model_type": model_type,
            "curves": [],
            "computed_at": datetime.now().isoformat(),
        }

    # Convert to numpy arrays
    y_proba = np.array(y_proba)

    # Binarize labels for One-vs-Rest
    y_true_bin = label_binarize(y_true, classes=class_labels)
    if n_classes == 2:
        # Special handling for binary case
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    # Compute ROC curve for each class
    curves = []
    for i, label in enumerate(class_labels):
        try:
            # Get binary labels and scores for this class
            y_binary = y_true_bin[:, i]
            y_scores = y_proba[:, i]

            n_positive = int(np.sum(y_binary))
            n_samples = len(y_binary)

            # Skip if no positive samples
            if n_positive == 0 or n_positive == n_samples:
                logger.warning(f"Skipping {label}: no variation in labels")
                continue

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_binary, y_scores)
            auc = roc_auc_score(y_binary, y_scores)

            # Downsample curve points for efficient transfer (max 100 points)
            if len(fpr) > 100:
                indices = np.linspace(0, len(fpr) - 1, 100, dtype=int)
                fpr = fpr[indices]
                tpr = tpr[indices]

            curves.append({
                "name": label,
                "auc": float(auc),
                "color": colors.get(label, "#666666"),
                "fpr": [float(x) for x in fpr],
                "tpr": [float(x) for x in tpr],
                "n_samples": n_samples,
                "n_positive": n_positive,
            })

        except Exception as e:
            logger.warning(f"Failed to compute ROC for {label}: {e}")
            continue

    logger.info(f"ROC computation complete: {len(curves)} curves")

    return {
        "model_name": model_name,
        "model_type": model_type,
        "curves": curves,
        "computed_at": datetime.now().isoformat(),
    }


def compute_roc_comparison(
    days: int = 7,
    max_crashes: int = 2000,
) -> dict[str, Any]:
    """Compute ROC curves for all classification models.

    Args:
        days: Number of days back to fetch data
        max_crashes: Maximum number of crashes per model

    Returns:
        Dictionary with ROC results for each model
    """
    classification_models = [
        name
        for name, info in MODEL_REGISTRY.items()
        if info.model_type in ("simplified", "hierarchical", "zones", "multiclass_nn")
    ]

    results = {}

    for model_name in classification_models:
        try:
            logger.info(f"Computing ROC for model: {model_name}")
            roc_data = compute_roc_data(
                days=days,
                max_crashes=max_crashes,
                model_name=model_name,
            )
            results[model_name] = {
                "model_name": model_name,
                "display_name": MODEL_REGISTRY[model_name].name,
                "model_type": MODEL_REGISTRY[model_name].model_type,
                "curves": roc_data["curves"],
                "status": "success",
            }
        except Exception as e:
            logger.warning(f"Failed to compute ROC for {model_name}: {e}")
            results[model_name] = {
                "model_name": model_name,
                "display_name": MODEL_REGISTRY[model_name].name,
                "model_type": MODEL_REGISTRY[model_name].model_type,
                "curves": [],
                "status": "error",
                "error": str(e),
            }

    return {
        "models": results,
        "time_range_days": days,
        "max_crashes": max_crashes,
        "computed_at": datetime.now().isoformat(),
    }

