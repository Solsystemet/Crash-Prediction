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
    predict_simple,
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
    if model_type == "hierarchical":
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


def _downsample_curve(
    fpr: list[float],
    tpr: list[float],
    max_points: int = 100,
) -> tuple[list[float], list[float]]:
    """Downsample ROC curve points for efficient JSON export.

    Args:
        fpr: False positive rate values.
        tpr: True positive rate values.
        max_points: Maximum number of points to keep.

    Returns:
        Tuple of (fpr_list, tpr_list) downsampled.
    """
    import numpy as np

    fpr_arr = np.array(fpr)
    tpr_arr = np.array(tpr)
    n_points = len(fpr_arr)

    if n_points <= max_points:
        return fpr, tpr

    # Keep evenly spaced indices, always including first and last
    indices = np.linspace(0, n_points - 1, max_points, dtype=int)
    indices = np.unique(indices)

    return fpr_arr[indices].tolist(), tpr_arr[indices].tolist()


def compute_roc_curves(
    predictions: list[dict[str, Any]],
    model_type: str,
    max_points: int = 100,
) -> list[dict[str, Any]]:
    """Compute ROC curves from prediction results.

    Uses the full probability distributions stored in predictions to compute
    proper ROC curves for each classification level.

    Args:
        predictions: List of prediction dicts with actual_severity, probabilities dict.
        model_type: Type of model (simplified, hierarchical, zones).
        max_points: Maximum points per curve for JSON export.

    Returns:
        List of ROC curve dictionaries with name, auc, fpr, tpr, color, etc.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    import numpy as np

    curves = []
    colors = ["#3b82f6", "#f97316", "#22c55e", "#ef4444"]  # blue, orange, green, red

    # For 3-class models (simplified, zones)
    if model_type in ("simplified", "zones"):
        # L1: Injury (MINOR or SEVERE) vs No Injury
        # P(injury) = P(minor) + P(severe) = 1 - P(no_injury)
        l1_y_true = []
        l1_proba = []

        for p in predictions:
            actual = p["actual_severity"]
            probs = p.get("probabilities", {})

            # Ground truth: 1 if injury, 0 if no injury
            l1_y_true.append(1 if actual in ("MINOR", "SEVERE") else 0)

            # P(injury) = 1 - P(no_injury)
            p_no_injury = probs.get("no_injury", 0.5)
            l1_proba.append(1.0 - p_no_injury)

        if len(set(l1_y_true)) >= 2 and l1_proba:
            try:
                fpr, tpr, _ = roc_curve(l1_y_true, l1_proba)
                auc = roc_auc_score(l1_y_true, l1_proba)
                fpr_ds, tpr_ds = _downsample_curve(
                    fpr.tolist(), tpr.tolist(), max_points
                )
                curves.append(
                    {
                        "name": "L1: Injury vs No Injury",
                        "auc": round(float(auc), 4),
                        "color": colors[0],
                        "fpr": fpr_ds,
                        "tpr": tpr_ds,
                        "n_samples": len(l1_y_true),
                        "n_positive": sum(l1_y_true),
                    }
                )
            except Exception as e:
                logger.warning(f"Error computing L1 ROC: {e}")

        # L2: Severe vs Minor (only for injury cases)
        # P(severe | injury) = P(severe) / (P(minor) + P(severe))
        injury_preds = [
            p for p in predictions if p["actual_severity"] in ("MINOR", "SEVERE")
        ]

        if len(injury_preds) > 10:
            l2_y_true = []
            l2_proba = []

            for p in injury_preds:
                actual = p["actual_severity"]
                probs = p.get("probabilities", {})

                # Ground truth: 1 if severe, 0 if minor
                l2_y_true.append(1 if actual == "SEVERE" else 0)

                # P(severe | injury) = P(severe) / (P(minor) + P(severe))
                p_severe = probs.get("severe", 0.33)
                p_minor = probs.get("minor", 0.33)
                p_injury = p_severe + p_minor
                if p_injury > 0.001:
                    l2_proba.append(p_severe / p_injury)
                else:
                    l2_proba.append(0.5)

            if len(set(l2_y_true)) >= 2 and l2_proba:
                try:
                    fpr, tpr, _ = roc_curve(l2_y_true, l2_proba)
                    auc = roc_auc_score(l2_y_true, l2_proba)
                    fpr_ds, tpr_ds = _downsample_curve(
                        fpr.tolist(), tpr.tolist(), max_points
                    )
                    curves.append(
                        {
                            "name": "L2: Severe vs Minor",
                            "auc": round(float(auc), 4),
                            "color": colors[1],
                            "fpr": fpr_ds,
                            "tpr": tpr_ds,
                            "n_samples": len(l2_y_true),
                            "n_positive": sum(l2_y_true),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error computing L2 ROC: {e}")

    # For 5-class hierarchical model
    elif model_type == "hierarchical":
        # L1: Any injury vs No Injury
        l1_y_true = []
        l1_proba = []

        for p in predictions:
            actual = p["actual_severity"]
            probs = p.get("probabilities", {})

            # Ground truth: 1 if injury, 0 if no injury
            l1_y_true.append(0 if actual == "NO_INJURY" else 1)

            # P(injury) = 1 - P(no_injury)
            p_no_injury = probs.get("no_injury", 0.5)
            l1_proba.append(1.0 - p_no_injury)

        if len(set(l1_y_true)) >= 2 and l1_proba:
            try:
                fpr, tpr, _ = roc_curve(l1_y_true, l1_proba)
                auc = roc_auc_score(l1_y_true, l1_proba)
                fpr_ds, tpr_ds = _downsample_curve(
                    fpr.tolist(), tpr.tolist(), max_points
                )
                curves.append(
                    {
                        "name": "L1: Injury vs No Injury",
                        "auc": round(float(auc), 4),
                        "color": colors[0],
                        "fpr": fpr_ds,
                        "tpr": tpr_ds,
                        "n_samples": len(l1_y_true),
                        "n_positive": sum(l1_y_true),
                    }
                )
            except Exception as e:
                logger.warning(f"Error computing L1 ROC: {e}")

        # L2: Severe (INCAPACITATING/FATAL) vs Minor (REPORTED_NOT_EVIDENT/NONINCAPACITATING)
        injury_preds = [p for p in predictions if p["actual_severity"] != "NO_INJURY"]

        if len(injury_preds) > 10:
            l2_y_true = []
            l2_proba = []
            severe_classes = {"INCAPACITATING", "FATAL"}

            for p in injury_preds:
                actual = p["actual_severity"]
                probs = p.get("probabilities", {})

                # Ground truth: 1 if severe, 0 if minor
                l2_y_true.append(1 if actual in severe_classes else 0)

                # P(severe | injury) = (P(incap) + P(fatal)) / P(injury)
                p_incap = probs.get("incapacitating", 0.1)
                p_fatal = probs.get("fatal", 0.05)
                p_severe = p_incap + p_fatal

                p_reported = probs.get("reported_not_evident", 0.2)
                p_nonincap = probs.get("nonincapacitating", 0.2)
                p_injury = p_severe + p_reported + p_nonincap

                if p_injury > 0.001:
                    l2_proba.append(p_severe / p_injury)
                else:
                    l2_proba.append(0.5)

            if len(set(l2_y_true)) >= 2 and l2_proba:
                try:
                    fpr, tpr, _ = roc_curve(l2_y_true, l2_proba)
                    auc = roc_auc_score(l2_y_true, l2_proba)
                    fpr_ds, tpr_ds = _downsample_curve(
                        fpr.tolist(), tpr.tolist(), max_points
                    )
                    curves.append(
                        {
                            "name": "L2: Severe vs Minor",
                            "auc": round(float(auc), 4),
                            "color": colors[1],
                            "fpr": fpr_ds,
                            "tpr": tpr_ds,
                            "n_samples": len(l2_y_true),
                            "n_positive": sum(l2_y_true),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error computing L2 ROC: {e}")

        # L2.5: Fatal vs Incapacitating (severe cases only)
        severe_preds = [
            p
            for p in predictions
            if p["actual_severity"] in {"INCAPACITATING", "FATAL"}
        ]

        if len(severe_preds) > 10:
            l25_y_true = []
            l25_proba = []

            for p in severe_preds:
                actual = p["actual_severity"]
                probs = p.get("probabilities", {})

                # Ground truth: 1 if fatal, 0 if incapacitating
                l25_y_true.append(1 if actual == "FATAL" else 0)

                # P(fatal | severe) = P(fatal) / (P(fatal) + P(incap))
                p_fatal = probs.get("fatal", 0.1)
                p_incap = probs.get("incapacitating", 0.2)
                p_severe = p_fatal + p_incap

                if p_severe > 0.001:
                    l25_proba.append(p_fatal / p_severe)
                else:
                    l25_proba.append(0.5)

            if len(set(l25_y_true)) >= 2 and l25_proba:
                try:
                    fpr, tpr, _ = roc_curve(l25_y_true, l25_proba)
                    auc = roc_auc_score(l25_y_true, l25_proba)
                    fpr_ds, tpr_ds = _downsample_curve(
                        fpr.tolist(), tpr.tolist(), max_points
                    )
                    curves.append(
                        {
                            "name": "L2.5: Fatal vs Incapacitating",
                            "auc": round(float(auc), 4),
                            "color": colors[2],
                            "fpr": fpr_ds,
                            "tpr": tpr_ds,
                            "n_samples": len(l25_y_true),
                            "n_positive": sum(l25_y_true),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error computing L2.5 ROC: {e}")

        # L3: Reported vs Visible (minor injury cases only)
        minor_preds = [
            p
            for p in predictions
            if p["actual_severity"] in {"REPORTED_NOT_EVIDENT", "NONINCAPACITATING"}
        ]

        if len(minor_preds) > 10:
            l3_y_true = []
            l3_proba = []

            for p in minor_preds:
                actual = p["actual_severity"]
                probs = p.get("probabilities", {})

                # Ground truth: 1 if reported, 0 if nonincapacitating
                l3_y_true.append(1 if actual == "REPORTED_NOT_EVIDENT" else 0)

                # P(reported | minor) = P(reported) / (P(reported) + P(nonincap))
                p_reported = probs.get("reported_not_evident", 0.25)
                p_nonincap = probs.get("nonincapacitating", 0.25)
                p_minor = p_reported + p_nonincap

                if p_minor > 0.001:
                    l3_proba.append(p_reported / p_minor)
                else:
                    l3_proba.append(0.5)

            if len(set(l3_y_true)) >= 2 and l3_proba:
                try:
                    fpr, tpr, _ = roc_curve(l3_y_true, l3_proba)
                    auc = roc_auc_score(l3_y_true, l3_proba)
                    fpr_ds, tpr_ds = _downsample_curve(
                        fpr.tolist(), tpr.tolist(), max_points
                    )
                    curves.append(
                        {
                            "name": "L3: Reported vs Visible",
                            "auc": round(float(auc), 4),
                            "color": colors[3],
                            "fpr": fpr_ds,
                            "tpr": tpr_ds,
                            "n_samples": len(l3_y_true),
                            "n_positive": sum(l3_y_true),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error computing L3 ROC: {e}")

    return curves


def evaluate_accuracy(
    days: int | None = 7,
    max_crashes: int = 10000,
    model_name: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate model accuracy on recent real crash data.

    Args:
        days: Number of days back to fetch data (used if start_date/end_date not provided)
        max_crashes: Maximum number of crashes to evaluate
        model_name: Name of the model to evaluate (from MODEL_REGISTRY).
                   If None, uses DEFAULT_MODEL.
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)

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

    # Calculate time_range_days for metrics response
    if start_date and end_date:
        time_range_days = (end_date - start_date).days + 1
        logger.info(
            f"Starting accuracy evaluation (model={model_name}, type={model_type}, "
            f"range={start_date.date()} to {end_date.date()}, max_crashes={max_crashes})"
        )
    else:
        time_range_days = days or 7
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
        data = fetch_crash_data_for_accuracy(
            days=days,
            max_crashes=max_crashes,
            start_date=start_date,
            end_date=end_date,
        )
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
                "time_range_days": time_range_days,
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
                "time_range_days": time_range_days,
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
            if model_type == "hierarchical":
                ground_truth = extract_ground_truth_5class(crash)
                if ground_truth is None:
                    continue
            else:
                ground_truth = ground_truth_3class

            # Get prediction based on model type
            # Store full probabilities for ROC curve computation
            probabilities = {}

            if model_type == "simplified":
                response = predict(request, model_name)
                predicted = response.prediction
                confidence = response.confidence
                probabilities = {
                    "no_injury": response.probabilities.no_injury,
                    "minor": response.probabilities.minor,
                    "severe": response.probabilities.severe,
                }
            elif model_type == "hierarchical":
                response = predict_hierarchical(request)
                predicted = response.prediction
                confidence = response.confidence
                probabilities = {
                    "no_injury": response.probabilities.no_injury,
                    "reported_not_evident": response.probabilities.reported_not_evident,
                    "nonincapacitating": response.probabilities.nonincapacitating,
                    "incapacitating": response.probabilities.incapacitating,
                    "fatal": response.probabilities.fatal,
                }
            elif model_type == "zones":
                # Zone model requires lat/lng
                if request.latitude is None or request.longitude is None:
                    continue
                response = predict_zones(request)
                predicted = response.prediction
                confidence = response.confidence
                probabilities = {
                    "no_injury": response.probabilities.no_injury,
                    "minor": response.probabilities.minor,
                    "severe": response.probabilities.severe,
                }
            elif model_type == "simple":
                # Simple vanilla RF model
                response = predict_simple(request, model_name)
                predicted = response.prediction
                confidence = response.confidence
                probabilities = {
                    "no_injury": response.probabilities.no_injury,
                    "minor": response.probabilities.minor,
                    "severe": response.probabilities.severe,
                }
            else:
                # Skip unsupported model types (e.g., regression)
                continue

            y_true.append(ground_truth)
            y_pred.append(predicted)

            # Parse crash date
            crash_dt = parse_crash_datetime(crash)

            # Build prediction record (including probabilities for ROC)
            predictions.append(
                {
                    "crash_record_id": crash.get("crash_record_id", ""),
                    "crash_date": crash_dt.isoformat() if crash_dt else None,
                    "predicted_severity": predicted,
                    "actual_severity": ground_truth,
                    "is_correct": predicted == ground_truth,
                    "confidence": confidence,
                    "probabilities": probabilities,
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
            "time_range_days": time_range_days,
            "computed_at": datetime.now().isoformat(),
            "f1_macro": f1_scores["f1_macro"],
            "f1_micro": f1_scores["f1_micro"],
            "model_name": model_name,
        },
        "predictions": predictions,
    }


def evaluate_all_models(
    days: int | None = 7,
    max_crashes: int = 2000,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate all classification models for comparison.

    Args:
        days: Number of days back to fetch data (used if start_date/end_date not provided)
        max_crashes: Maximum number of crashes to evaluate (lower for comparison)
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)

    Returns:
        Dictionary with comparison results for each model
    """
    # Calculate time_range_days for response
    if start_date and end_date:
        time_range_days = (end_date - start_date).days + 1
    else:
        time_range_days = days or 7

    # Only compare classification models, not regression
    classification_models = [
        name
        for name, info in MODEL_REGISTRY.items()
        if info.model_type in ("simplified", "hierarchical", "zones", "simple")
    ]

    results = {}

    for model_name in classification_models:
        try:
            logger.info(f"Evaluating model: {model_name}")
            result = evaluate_accuracy(
                days=days,
                max_crashes=max_crashes,
                model_name=model_name,
                start_date=start_date,
                end_date=end_date,
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
        "time_range_days": time_range_days,
        "max_crashes": max_crashes,
        "computed_at": datetime.now().isoformat(),
    }


def get_roc_data(
    days: int | None = 7,
    max_crashes: int = 2000,
    model_name: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, Any]:
    """Get ROC curve data for a model.

    Args:
        days: Number of days back to fetch data (used if start_date/end_date not provided)
        max_crashes: Maximum number of crashes to evaluate
        model_name: Name of the model (defaults to DEFAULT_MODEL)
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)

    Returns:
        Dictionary with ROC curves data for frontend visualization
    """
    # Resolve model name
    if model_name is None:
        model_name = DEFAULT_MODEL

    # Validate model exists
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    model_type = MODEL_REGISTRY[model_name].model_type

    # Get predictions (reuse evaluate_accuracy)
    result = evaluate_accuracy(
        days=days,
        max_crashes=max_crashes,
        model_name=model_name,
        start_date=start_date,
        end_date=end_date,
    )

    predictions = result["predictions"]

    if not predictions:
        return {
            "model_name": model_name,
            "model_type": model_type,
            "curves": [],
            "computed_at": datetime.now().isoformat(),
        }

    # Compute ROC curves
    curves = compute_roc_curves(predictions, model_type)

    return {
        "model_name": model_name,
        "model_type": model_type,
        "curves": curves,
        "computed_at": datetime.now().isoformat(),
    }


def get_all_models_roc_data(
    days: int | None = 7,
    max_crashes: int = 2000,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, Any]:
    """Get ROC curve data for all classification models.

    Args:
        days: Number of days back to fetch data (used if start_date/end_date not provided)
        max_crashes: Maximum number of crashes to evaluate
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)

    Returns:
        Dictionary with ROC data for each model
    """
    # Calculate time_range_days for response
    if start_date and end_date:
        time_range_days = (end_date - start_date).days + 1
    else:
        time_range_days = days or 7

    # Only compare classification models, not regression
    classification_models = [
        name
        for name, info in MODEL_REGISTRY.items()
        if info.model_type in ("simplified", "hierarchical", "zones", "simple")
    ]

    results = {}

    for model_name in classification_models:
        try:
            logger.info(f"Computing ROC for model: {model_name}")
            roc_data = get_roc_data(
                days=days,
                max_crashes=max_crashes,
                model_name=model_name,
                start_date=start_date,
                end_date=end_date,
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
        "time_range_days": time_range_days,
        "max_crashes": max_crashes,
        "computed_at": datetime.now().isoformat(),
    }


def get_map_data(
    days: int | None = 7,
    max_crashes: int = 200,
    filter_correct: bool | None = None,
    model_name: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[dict[str, Any]]:
    """Get prediction data formatted for map display.

    Args:
        days: Number of days back to fetch (used if start_date/end_date not provided)
        max_crashes: Maximum number of crashes
        filter_correct: If True, only correct; if False, only incorrect; if None, all
        model_name: Name of the model to use (defaults to DEFAULT_MODEL)
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)

    Returns:
        List of prediction records with coordinates for map display
    """
    result = evaluate_accuracy(
        days=days,
        max_crashes=max_crashes,
        model_name=model_name,
        start_date=start_date,
        end_date=end_date,
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
