"""Prediction logic for the API.

This module handles loading trained models and making predictions
based on input features from API requests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from api.config import MODEL_REGISTRY, DEFAULT_MODEL, ModelInfo
from api.models import PredictionRequest, PredictionResponse, PredictionProbabilities
from training.hierarchical.simplified_classifier import (
    SimplifiedTreeClassifier,
    load_simplified_model,
)
from training.hierarchical.simplified_targets import SIMPLIFIED_CLASS_NAMES

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and caching of trained models."""

    def __init__(self):
        self._models: dict[str, SimplifiedTreeClassifier] = {}
        self._label_encoders: dict[str, dict[str, LabelEncoder]] = {}
        self._current_model: str | None = None

    def load_model(self, model_name: str = DEFAULT_MODEL) -> SimplifiedTreeClassifier:
        """Load a model by name from the registry.

        Args:
            model_name: Name of the model in the registry.

        Returns:
            Loaded SimplifiedTreeClassifier.

        Raises:
            ValueError: If model is not in registry or not found on disk.
        """
        if model_name in self._models:
            logger.info(f"Using cached model: {model_name}")
            self._current_model = model_name
            return self._models[model_name]

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found in registry")

        model_info = MODEL_REGISTRY[model_name]
        model_path = model_info.path

        if not model_path.exists():
            raise ValueError(
                f"Model directory not found: {model_path}. "
                f"Please train the model first using: python training/main_simplified.py"
            )

        logger.info(f"Loading model from {model_path}")
        model = load_simplified_model(str(model_path))
        self._models[model_name] = model
        self._current_model = model_name

        # Create label encoders for categorical features
        self._create_label_encoders(model_name)

        logger.info(f"Model loaded successfully: {model_name}")
        return model

    def _create_label_encoders(self, model_name: str) -> None:
        """Create label encoders for categorical features.

        These must match the encoders used during training.
        """
        from api.config import (
            FIRST_CRASH_TYPE_OPTIONS,
            DAMAGE_OPTIONS,
            PRIM_CONTRIBUTORY_CAUSE_OPTIONS,
            WEATHER_CONDITION_OPTIONS,
            LIGHTING_CONDITION_OPTIONS,
            ROADWAY_SURFACE_COND_OPTIONS,
            TRAFFIC_CONTROL_DEVICE_OPTIONS,
            DEVICE_CONDITION_OPTIONS,
            TRAFFICWAY_TYPE_OPTIONS,
            ROAD_DEFECT_OPTIONS,
            ALIGNMENT_OPTIONS,
        )

        encoders = {}

        # Map feature names to their options
        categorical_features = {
            "FIRST_CRASH_TYPE": FIRST_CRASH_TYPE_OPTIONS + ["UNKNOWN"],
            "DAMAGE": DAMAGE_OPTIONS + ["UNKNOWN"],
            "PRIM_CONTRIBUTORY_CAUSE": PRIM_CONTRIBUTORY_CAUSE_OPTIONS + ["UNKNOWN"],
            "WEATHER_CONDITION": WEATHER_CONDITION_OPTIONS + ["UNKNOWN"],
            "LIGHTING_CONDITION": LIGHTING_CONDITION_OPTIONS + ["UNKNOWN"],
            "ROADWAY_SURFACE_COND": ROADWAY_SURFACE_COND_OPTIONS + ["UNKNOWN"],
            "TRAFFIC_CONTROL_DEVICE": TRAFFIC_CONTROL_DEVICE_OPTIONS + ["UNKNOWN"],
            "DEVICE_CONDITION": DEVICE_CONDITION_OPTIONS + ["UNKNOWN"],
            "TRAFFICWAY_TYPE": TRAFFICWAY_TYPE_OPTIONS + ["UNKNOWN"],
            "ROAD_DEFECT": ROAD_DEFECT_OPTIONS + ["UNKNOWN"],
            "ALIGNMENT": ALIGNMENT_OPTIONS + ["UNKNOWN"],
        }

        for feature_name, options in categorical_features.items():
            le = LabelEncoder()
            le.fit(options)
            encoders[feature_name] = le

        self._label_encoders[model_name] = encoders

    def get_current_model(self) -> SimplifiedTreeClassifier | None:
        """Get the currently loaded model."""
        if self._current_model is None:
            return None
        return self._models.get(self._current_model)

    def get_current_model_name(self) -> str | None:
        """Get the name of the currently loaded model."""
        return self._current_model

    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._current_model is not None

    def get_label_encoder(self, feature_name: str) -> LabelEncoder | None:
        """Get the label encoder for a categorical feature."""
        if self._current_model is None:
            return None
        encoders = self._label_encoders.get(self._current_model, {})
        return encoders.get(feature_name)


# Global model manager instance
model_manager = ModelManager()


def prepare_features_from_request(request: PredictionRequest) -> np.ndarray:
    """Convert API request to feature array for prediction.

    This function maps the request fields to the feature columns
    expected by the model, applying label encoding for categorical features.

    Args:
        request: PredictionRequest with input features.

    Returns:
        NumPy array of features ready for prediction.
    """
    model = model_manager.get_current_model()
    if model is None:
        raise RuntimeError("No model loaded")

    feature_cols = model.feature_cols
    if feature_cols is None:
        raise RuntimeError("Model has no feature columns defined")

    # Map request fields to model feature names
    # The model uses uppercase column names from the original dataset
    feature_mapping = {
        # Crash info
        "PERSON_COUNT": request.person_count,
        "VEHICLE_COUNT": request.vehicle_count,
        "NUM_UNITS": request.vehicle_count,  # Same as vehicle count
        "FIRST_CRASH_TYPE": request.first_crash_type,
        "DAMAGE": request.damage,
        "PRIM_CONTRIBUTORY_CAUSE": request.prim_contributory_cause,
        # People features
        "age_clean_mean": request.age_mean,
        "age_clean_min": request.age_min,
        "age_clean_max": request.age_max,
        "is_driver_sum": request.driver_count,
        # Vehicle features
        "AVG_VEHICLE_YEAR": request.avg_vehicle_year,
        "OLDEST_VEHICLE_YEAR": request.oldest_vehicle_year,
        "NEWEST_VEHICLE_YEAR": request.avg_vehicle_year,  # Default to avg
        "VEHICLE_AGE": 2024 - request.oldest_vehicle_year,
        # Road/Location features
        "POSTED_SPEED_LIMIT": request.posted_speed_limit,
        "TRAFFIC_CONTROL_DEVICE": request.traffic_control_device,
        "DEVICE_CONDITION": request.device_condition,
        "TRAFFICWAY_TYPE": request.trafficway_type,
        "LIGHTING_CONDITION": request.lighting_condition,
        "ROAD_DEFECT": request.road_defect,
        "ROADWAY_SURFACE_COND": request.roadway_surface_cond,
        "ALIGNMENT": request.alignment,
        # Weather
        "WEATHER_CONDITION": request.weather_condition,
        "Air Temperature": request.air_temperature,
        "Humidity": request.humidity,
        "Wind Speed": request.wind_speed,
        "Rain Intensity": request.rain_intensity,
        "Total Rain": request.rain_intensity,  # Approximate
        "Interval Rain": request.rain_intensity,
        "Barometric Pressure": 30.0,  # Default value
        "Precipitation Type": 0,  # Default
        # Time features
        "CRASH_HOUR": request.crash_hour,
        "CRASH_DAY_OF_WEEK": request.crash_day_of_week,
        "CRASH_MONTH": request.crash_month,
        # Derived features
        "IS_PEAK_HOUR": 1 if request.crash_hour in [7, 8, 9, 16, 17, 18] else 0,
        "IS_WEEKEND": 1 if request.crash_day_of_week in [1, 7] else 0,
        "IS_NIGHT": 1 if request.crash_hour < 6 or request.crash_hour >= 20 else 0,
        "MULTI_VEHICLE": 1 if request.vehicle_count > 2 else 0,
        "OLD_VEHICLE_FLAG": 1 if (2024 - request.oldest_vehicle_year) > 10 else 0,
        "HIGH_SPEED_AREA": 1 if request.posted_speed_limit >= 40 else 0,
        "WET_ROAD": 1
        if request.roadway_surface_cond in ["WET", "SNOW OR SLUSH", "ICE"]
        else 0,
        "NIGHT_POOR_LIGHTING": 1
        if (request.crash_hour < 6 or request.crash_hour >= 20)
        and request.lighting_condition == "DARKNESS"
        else 0,
        "ADVERSE_CONDITIONS_COUNT": sum(
            [
                1
                if request.weather_condition not in ["CLEAR", "CLOUDY/OVERCAST"]
                else 0,
                1 if request.roadway_surface_cond != "DRY" else 0,
                1 if request.lighting_condition in ["DARKNESS", "DUSK", "DAWN"] else 0,
            ]
        ),
        # Beat/Street (use defaults)
        "BEAT_OF_OCCURRENCE": 0,
        "STREET_NO": 0,
        "LANE_CNT": 2,  # Default
    }

    # Build feature array
    features = []
    for col in feature_cols:
        if col in feature_mapping:
            value = feature_mapping[col]

            # Apply label encoding for categorical features
            if col in [
                "FIRST_CRASH_TYPE",
                "DAMAGE",
                "PRIM_CONTRIBUTORY_CAUSE",
                "WEATHER_CONDITION",
                "LIGHTING_CONDITION",
                "ROADWAY_SURFACE_COND",
                "TRAFFIC_CONTROL_DEVICE",
                "DEVICE_CONDITION",
                "TRAFFICWAY_TYPE",
                "ROAD_DEFECT",
                "ALIGNMENT",
            ]:
                encoder = model_manager.get_label_encoder(col)
                if encoder is not None:
                    try:
                        value = encoder.transform([str(value)])[0]
                    except ValueError:
                        # Unknown value, use 0
                        value = encoder.transform(["UNKNOWN"])[0]

            features.append(
                float(value) if not isinstance(value, (int, float)) else value
            )
        else:
            # Feature not in mapping, use default value
            logger.warning(f"Feature {col} not in mapping, using 0")
            features.append(0)

    return np.array([features])


def predict(
    request: PredictionRequest, model_name: str = DEFAULT_MODEL
) -> PredictionResponse:
    """Make a prediction for the given request.

    Args:
        request: PredictionRequest with input features.
        model_name: Name of the model to use.

    Returns:
        PredictionResponse with prediction and probabilities.
    """
    # Ensure model is loaded
    model = model_manager.load_model(model_name)

    # Prepare features
    features = prepare_features_from_request(request)

    # Get predictions
    l1_proba, l2_proba = model.predict_proba(features)
    prediction_class = model.predict(features)[0]

    # Calculate 3-class probabilities
    # L1: probability of injury (1) vs no injury (0)
    # L2: probability of severe (1) vs minor (0) given injury

    p_no_injury = 1.0 - l1_proba[0]  # P(no injury)
    p_injury = l1_proba[0]  # P(injury)
    p_severe_given_injury = l2_proba[0]  # P(severe | injury)
    p_minor_given_injury = 1.0 - l2_proba[0]  # P(minor | injury)

    # Final probabilities
    prob_no_injury = p_no_injury
    prob_minor = p_injury * p_minor_given_injury
    prob_severe = p_injury * p_severe_given_injury

    # Normalize to sum to 1
    total = prob_no_injury + prob_minor + prob_severe
    prob_no_injury /= total
    prob_minor /= total
    prob_severe /= total

    # Determine prediction based on highest probability
    # (more intuitive than threshold-based prediction for users)
    probs = [prob_no_injury, prob_minor, prob_severe]
    class_labels = ["NO_INJURY", "MINOR", "SEVERE"]
    max_idx = probs.index(max(probs))
    prediction_label = class_labels[max_idx]

    # Confidence is the max probability
    confidence = max(probs)

    return PredictionResponse(
        prediction=prediction_label,  # type: ignore
        probabilities=PredictionProbabilities(
            no_injury=float(prob_no_injury),
            minor=float(prob_minor),
            severe=float(prob_severe),
        ),
        confidence=float(confidence),
        model_name=model_name,
    )
