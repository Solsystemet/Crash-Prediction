"""Accuracy evaluation service for comparing predictions against real data.

This module handles fetching real crash data from the Chicago API,
running predictions on it, and computing accuracy metrics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from api.chicago_client import fetch_crash_data_for_accuracy, ChicagoAPIError
from api.data_transformer import (
    transform_all_crashes,
    extract_ground_truth,
    parse_crash_datetime,
    _safe_float,
)
from api.prediction import predict, model_manager
from api.models import PredictionRequest

logger = logging.getLogger(__name__)

# Class labels in order
CLASS_LABELS = ["NO_INJURY", "MINOR", "SEVERE"]


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


def evaluate_accuracy(
    days: int = 7,
    max_crashes: int = 500,
) -> dict[str, Any]:
    """Evaluate model accuracy on recent real crash data.

    Args:
        days: Number of days back to fetch data
        max_crashes: Maximum number of crashes to evaluate

    Returns:
        Dictionary containing accuracy metrics and individual predictions

    Raises:
        RuntimeError: If the model is not trained/available
    """
    logger.info(
        f"Starting accuracy evaluation (days={days}, max_crashes={max_crashes})"
    )

    # Ensure model is loaded
    if not model_manager.is_model_loaded():
        from api.config import DEFAULT_MODEL

        try:
            model_manager.load_model(DEFAULT_MODEL)
        except ValueError as e:
            raise RuntimeError(
                "Model not available. Please train the model first by running: "
                "python training/main_simplified.py"
            ) from e

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
                "time_range_days": days,
                "computed_at": datetime.now().isoformat(),
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
                "time_range_days": days,
                "computed_at": datetime.now().isoformat(),
            },
            "predictions": [],
        }

    # Make predictions
    y_true = []
    y_pred = []
    predictions = []

    for crash, request, ground_truth in transformed:
        try:
            # Get prediction
            response = predict(request)
            predicted = response.prediction
            confidence = response.confidence

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

    # Compute metrics
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)
    class_metrics = compute_class_metrics(confusion_matrix)
    overall_accuracy = compute_overall_accuracy(y_true, y_pred)

    logger.info(
        f"Accuracy evaluation complete: {len(predictions)} predictions, "
        f"accuracy={overall_accuracy:.2%}"
    )

    return {
        "metrics": {
            "overall_accuracy": overall_accuracy,
            "sample_count": len(predictions),
            "per_class_metrics": class_metrics,
            "confusion_matrix": confusion_matrix,
            "class_labels": CLASS_LABELS,
            "time_range_days": days,
            "computed_at": datetime.now().isoformat(),
        },
        "predictions": predictions,
    }


def get_map_data(
    days: int = 7,
    max_crashes: int = 200,
    filter_correct: bool | None = None,
) -> list[dict[str, Any]]:
    """Get prediction data formatted for map display.

    Args:
        days: Number of days back to fetch
        max_crashes: Maximum number of crashes
        filter_correct: If True, only correct; if False, only incorrect; if None, all

    Returns:
        List of prediction records with coordinates for map display
    """
    result = evaluate_accuracy(days=days, max_crashes=max_crashes)
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
