"""Prediction logic for the API.

This module handles loading trained models and making predictions
based on input features from API requests.
"""

from __future__ import annotations

# IMPORTANT: Set these BEFORE importing torch to prevent Windows deadlocks
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable CUDA entirely

import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
from sklearn.preprocessing import LabelEncoder

from api.config import MODEL_REGISTRY, DEFAULT_MODEL, MODEL_TYPE_REGISTRY, ModelInfo
from api.models import (
    ModelType,
    PredictionRequest,
    PredictionResponse,
    PredictionProbabilities,
    HierarchicalPredictionResponse,
    HierarchicalProbabilities,
    ZonePredictionResponse,
    RegressionPredictionResponse,
)
from training.hierarchical.simplified_classifier import (
    SimplifiedTreeClassifier,
    load_simplified_model,
)
from training.hierarchical.tree_classifier import (
    HierarchicalTreeClassifier,
    load_hierarchical_model,
)
from training.hierarchical.simplified_targets import SIMPLIFIED_CLASS_NAMES

logger = logging.getLogger(__name__)

# 5-class labels for hierarchical model
HIERARCHICAL_CLASS_NAMES = [
    "NO_INJURY",
    "REPORTED_NOT_EVIDENT",
    "NONINCAPACITATING",
    "INCAPACITATING",
    "FATAL",
]

# Type alias for response types
PredictionResponseType = Union[
    PredictionResponse,
    HierarchicalPredictionResponse,
    ZonePredictionResponse,
    RegressionPredictionResponse,
]


class ModelManager:
    """Manages loading and caching of trained models for all model types."""

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._label_encoders: dict[str, dict[str, LabelEncoder]] = {}
        self._current_model: str | None = None
        self._zone_predictor: Any = None
        self._regression_predictor: Any = None
        self._nn_classifier: Any = None

    def load_model(self, model_name: str = DEFAULT_MODEL) -> Any:
        """Load a model by name from the registry.

        Args:
            model_name: Name of the model in the registry.

        Returns:
            Loaded model (type depends on model_type).

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
                f"Please train the model first."
            )

        logger.info(f"Loading model from {model_path}")

        # Load based on model type
        if model_info.model_type == "simplified":
            model = load_simplified_model(str(model_path))
        elif model_info.model_type == "hierarchical":
            # Hierarchical 5-class uses different loader
            model = load_hierarchical_model(str(model_path))
        elif model_info.model_type == "zones":
            from training.predict_zones import ZonePredictor
            model = ZonePredictor(model_path)
            self._zone_predictor = model
        elif model_info.model_type == "regression":
            from training.regression.predict import CrashCountPredictor
            model = CrashCountPredictor(model_path)
            self._regression_predictor = model
        elif model_info.model_type == "multiclass_nn":
            from training.multiclass.classifier import MulticlassNeuralClassifier
            nn_model_path = model_path / "multiclass_nn.pt"
            model = MulticlassNeuralClassifier.load(nn_model_path)
            self._nn_classifier = model
        else:
            raise ValueError(f"Unknown model type: {model_info.model_type}")

        self._models[model_name] = model
        self._current_model = model_name

        # Create label encoders for categorical features (for tree-based models)
        if model_info.model_type in ["simplified", "hierarchical", "zones"]:
            self._create_label_encoders(model_name)

        logger.info(f"Model loaded successfully: {model_name}")
        return model

    def load_model_by_type(self, model_type: ModelType) -> Any:
        """Load a model by its type.

        Args:
            model_type: Type of model to load.

        Returns:
            Loaded model.
        """
        model_name = MODEL_TYPE_REGISTRY.get(model_type.value)
        if model_name is None:
            raise ValueError(f"No model registered for type: {model_type}")
        return self.load_model(model_name)

    def get_model_name_for_type(self, model_type: ModelType) -> str:
        """Get the registry name for a model type."""
        model_name = MODEL_TYPE_REGISTRY.get(model_type.value)
        if model_name is None:
            raise ValueError(f"No model registered for type: {model_type}")
        return model_name

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


def prepare_features_for_nn(request: PredictionRequest, num_classes: int = 5) -> np.ndarray:
    """Prepare features specifically for the neural network model.

    The neural network expects a specific set of features including
    engineered features that are computed from the raw input.

    Args:
        request: PredictionRequest with input features.
        num_classes: Number of output classes (default 5).

    Returns:
        NumPy array of features ready for neural network prediction.
    """
    import json
    from api.config import MODELS_DIR

    # Load the expected feature list
    features_path = MODELS_DIR / "multiclass_nn" / "features.json"
    with open(features_path) as f:
        feature_cols = json.load(f)

    # Determine high-impact crash types
    high_impact_types = [
        "HEAD ON", "PEDESTRIAN", "PEDALCYCLIST", "FIXED OBJECT",
        "OVERTURNED", "TRAIN"
    ]

    # Build comprehensive feature mapping
    feature_mapping = {
        # Core crash info
        "POSTED_SPEED_LIMIT": request.posted_speed_limit,
        "LANE_CNT": 2,  # Default
        "STREET_NO": 0,  # Default
        "BEAT_OF_OCCURRENCE": 0,  # Default
        "NUM_UNITS": request.vehicle_count,
        "CRASH_HOUR": request.crash_hour,
        "CRASH_DAY_OF_WEEK": request.crash_day_of_week,
        "CRASH_MONTH": request.crash_month,
        "VEHICLE_COUNT": request.vehicle_count,
        "NEWEST_VEHICLE_YEAR": request.avg_vehicle_year,
        "PERSON_COUNT": request.person_count,

        # Age features
        "age_clean_min": request.age_min,
        "age_clean_max": request.age_max,
        "age_clean_mean": request.age_mean,
        "is_driver_sum": request.driver_count,

        # Weather features
        "Air Temperature": request.air_temperature,
        "Humidity": request.humidity,
        "Rain Intensity": request.rain_intensity,
        "Interval Rain": request.rain_intensity,
        "Total Rain": request.rain_intensity,
        "Precipitation Type": 0,  # Default
        "Wind Speed": request.wind_speed,
        "Barometric Pressure": 30.0,  # Default

        # Derived time features
        "IS_PEAK_HOUR": 1 if request.crash_hour in [7, 8, 9, 16, 17, 18] else 0,
        "IS_NIGHT": 1 if request.crash_hour < 6 or request.crash_hour >= 20 else 0,
        "IS_WEEKEND": 1 if request.crash_day_of_week in [1, 7] else 0,

        # Vehicle features
        "VEHICLE_AGE": 2024 - request.oldest_vehicle_year,
        "OLD_VEHICLE_FLAG": 1 if (2024 - request.oldest_vehicle_year) > 10 else 0,

        # Risk features
        "ADVERSE_CONDITIONS_COUNT": sum([
            1 if request.weather_condition not in ["CLEAR", "CLOUDY/OVERCAST"] else 0,
            1 if request.roadway_surface_cond != "DRY" else 0,
            1 if request.lighting_condition in ["DARKNESS", "DUSK", "DAWN"] else 0,
        ]),
        "NIGHT_POOR_LIGHTING": 1 if (
            (request.crash_hour < 6 or request.crash_hour >= 20) and
            request.lighting_condition == "DARKNESS"
        ) else 0,
        "WET_ROAD": 1 if request.roadway_surface_cond in ["WET", "SNOW OR SLUSH", "ICE"] else 0,
        "HIGH_SPEED_CRASH": 1 if request.posted_speed_limit >= 40 else 0,
        "VERY_HIGH_SPEED": 1 if request.posted_speed_limit >= 55 else 0,
        "MULTI_VEHICLE": 1 if request.vehicle_count > 2 else 0,
        "INTERSECTION_CRASH": 1 if request.traffic_control_device in [
            "TRAFFIC SIGNAL", "STOP SIGN/FLASHER", "YIELD"
        ] else 0,
        "HIGH_IMPACT_CRASH_TYPE": 1 if request.first_crash_type in high_impact_types else 0,
        "SEVERE_DAMAGE_INDICATOR": 1 if request.damage == "OVER $1,500" else 0,
        "HIT_AND_RUN": 0,  # Not available in request
        "WORK_ZONE_CRASH": 0,  # Not available in request
        "SEVERITY_RISK_SCORE": 0,  # Computed during training, use 0

        # Categorical features (will be encoded)
        "WEATHER_CONDITION": request.weather_condition,
        "LIGHTING_CONDITION": request.lighting_condition,
        "FIRST_CRASH_TYPE": request.first_crash_type,
        "TRAFFICWAY_TYPE": request.trafficway_type,
        "ROADWAY_SURFACE_COND": request.roadway_surface_cond,
        "TRAFFIC_CONTROL_DEVICE": request.traffic_control_device,
        "DEVICE_CONDITION": request.device_condition,
        "ALIGNMENT": request.alignment,
        "ROAD_DEFECT": request.road_defect,
        "PRIM_CONTRIBUTORY_CAUSE": request.prim_contributory_cause,
        "DAMAGE": request.damage,
    }

    # Get label encoders from the neural network model's training data
    # Load label encoders saved during training
    import joblib
    encoders_path = MODELS_DIR / "multiclass_nn" / "label_encoders.joblib"
    label_encoders = {}
    if encoders_path.exists():
        label_encoders = joblib.load(encoders_path)

    # Build feature array
    features = []
    for col in feature_cols:
        if col in feature_mapping:
            value = feature_mapping[col]

            # Apply label encoding using saved encoders from training
            if col in label_encoders:
                le = label_encoders[col]
                str_value = str(value)
                if str_value in le.classes_:
                    value = int(le.transform([str_value])[0])
                else:
                    value = 0  # Default for unknown

            features.append(float(value) if not isinstance(value, (int, float)) else value)
        else:
            # Feature not in mapping, use default value
            logger.warning(f"NN Feature {col} not in mapping, using 0")
            features.append(0.0)

    # Load and apply the scaler (same as training)
    scaler_path = MODELS_DIR / "multiclass_nn" / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        features_array = scaler.transform(np.array([features]))
        return features_array.astype(np.float32)

    return np.array([features], dtype=np.float32)


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
    """Make a prediction for the given request (simplified 3-class).

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


def predict_zones(request: PredictionRequest) -> ZonePredictionResponse:
    """Make a zone-based severity prediction.

    Args:
        request: PredictionRequest with input features including lat/lng.

    Returns:
        ZonePredictionResponse with prediction and zone information.

    Raises:
        ValueError: If latitude/longitude not provided.
    """
    if request.latitude is None or request.longitude is None:
        raise ValueError(
            "Latitude and longitude are required for zone-based prediction"
        )

    model_name = MODEL_TYPE_REGISTRY["zones"]
    zone_predictor = model_manager.load_model(model_name)

    # Assign zone based on coordinates
    coords = np.array([[request.latitude, request.longitude]])
    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(coords, zone_predictor.centroids)
    zone_id = int(np.argmin(distances))
    zone_center = tuple(zone_predictor.centroids[zone_id])

    # Get zone-specific model
    if zone_id in zone_predictor.zone_models:
        zone_model = zone_predictor.zone_models[zone_id]
    else:
        # Fallback to any available model
        available_zones = list(zone_predictor.zone_models.keys())
        if not available_zones:
            raise RuntimeError("No zone models available")
        zone_id = available_zones[0]
        zone_model = zone_predictor.zone_models[zone_id]
        zone_center = tuple(zone_predictor.centroids[zone_id])

    # Prepare features using the zone model
    model_manager._current_model = model_name
    features = prepare_features_from_request(request)

    # Get predictions
    l1_proba, l2_proba = zone_model.predict_proba(features)

    # Calculate 3-class probabilities
    p_no_injury = 1.0 - l1_proba[0]
    p_injury = l1_proba[0]
    p_severe_given_injury = l2_proba[0]
    p_minor_given_injury = 1.0 - l2_proba[0]

    prob_no_injury = p_no_injury
    prob_minor = p_injury * p_minor_given_injury
    prob_severe = p_injury * p_severe_given_injury

    total = prob_no_injury + prob_minor + prob_severe
    prob_no_injury /= total
    prob_minor /= total
    prob_severe /= total

    probs = [prob_no_injury, prob_minor, prob_severe]
    class_labels = ["NO_INJURY", "MINOR", "SEVERE"]
    max_idx = probs.index(max(probs))
    prediction_label = class_labels[max_idx]
    confidence = max(probs)

    return ZonePredictionResponse(
        prediction=prediction_label,  # type: ignore
        probabilities=PredictionProbabilities(
            no_injury=float(prob_no_injury),
            minor=float(prob_minor),
            severe=float(prob_severe),
        ),
        confidence=float(confidence),
        model_name=model_name,
        zone_id=zone_id,
        zone_center=zone_center,
    )


def predict_regression(request: PredictionRequest) -> RegressionPredictionResponse:
    """Make a crash count regression prediction.

    Args:
        request: PredictionRequest with input features.

    Returns:
        RegressionPredictionResponse with predicted count.
    """
    model_name = MODEL_TYPE_REGISTRY["regression"]
    
    try:
        predictor = model_manager.load_model(model_name)
    except Exception:
        # If regression model not available, return a placeholder
        logger.warning("Regression model not available, returning placeholder")
        return RegressionPredictionResponse(
            predicted_count=0.0,
            confidence_interval=(0.0, 0.0),
            zone_id=None,
            time_period="daily",
            model_name=model_name,
        )

    # For regression, we need time-based features
    # Create a simple feature vector based on the request
    import pandas as pd

    # Use available features to estimate crash count
    # This is simplified - real implementation would use proper time series
    hour = request.crash_hour
    day_of_week = request.crash_day_of_week
    month = request.crash_month

    # Base prediction on time patterns (simplified heuristic if model fails)
    try:
        # Attempt to use the actual predictor
        features = {
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "weather_condition": request.weather_condition,
        }
        
        # If predictor has predict method, use it
        if hasattr(predictor, "predict"):
            # Create minimal input for prediction
            pred = predictor.predict(
                hour=hour,
                day_of_week=day_of_week,
                month=month,
            )
            count = float(pred) if pred is not None else 5.0
        else:
            count = 5.0  # Default
            
        # Estimate confidence interval
        std_estimate = count * 0.3
        ci_lower = max(0, count - 1.96 * std_estimate)
        ci_upper = count + 1.96 * std_estimate

    except Exception as e:
        logger.warning(f"Regression prediction failed: {e}, using heuristic")
        # Heuristic based on time
        base = 3.0
        if hour in [7, 8, 9, 16, 17, 18]:  # Rush hours
            base *= 1.5
        if day_of_week in [1, 7]:  # Weekend
            base *= 0.8
        count = base
        ci_lower = max(0, count - 2)
        ci_upper = count + 2

    return RegressionPredictionResponse(
        predicted_count=float(count),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        zone_id=None,
        time_period="daily",
        model_name=model_name,
    )


def predict_hierarchical(request: PredictionRequest) -> HierarchicalPredictionResponse:
    """Make a 5-class hierarchical severity prediction.

    Args:
        request: PredictionRequest with input features.

    Returns:
        HierarchicalPredictionResponse with 5-class prediction and probabilities.
    """
    model_name = MODEL_TYPE_REGISTRY["hierarchical"]
    model = model_manager.load_model(model_name)

    # Prepare features
    features = prepare_features_from_request(request)

    # Get predictions from all 4 levels
    # L1: injury vs no_injury
    # L2: severe vs minor (given injury)
    # L2.5: fatal vs incapacitating (given severe)
    # L3: reported vs visible/nonincapacitating (given minor)
    l1_proba, l2_proba, l25_proba, l3_proba = model.predict_proba(features)

    # Calculate 5-class probabilities using hierarchical structure
    # P(no_injury) = P(L1=0)
    p_no_injury = 1.0 - l1_proba[0]

    # P(injury) = P(L1=1)
    p_injury = l1_proba[0]

    # P(minor | injury) = P(L2=0), P(severe | injury) = P(L2=1)
    p_minor_given_injury = 1.0 - l2_proba[0]
    p_severe_given_injury = l2_proba[0]

    # P(reported | minor) = P(L3=1), P(nonincap/visible | minor) = P(L3=0)
    p_reported_given_minor = l3_proba[0]
    p_nonincap_given_minor = 1.0 - l3_proba[0]

    # P(fatal | severe) = P(L25=1), P(incapacitating | severe) = P(L25=0)
    p_fatal_given_severe = l25_proba[0]
    p_incap_given_severe = 1.0 - l25_proba[0]

    # Final 5-class probabilities
    prob_no_injury = p_no_injury
    prob_reported = p_injury * p_minor_given_injury * p_reported_given_minor
    prob_nonincap = p_injury * p_minor_given_injury * p_nonincap_given_minor
    prob_incap = p_injury * p_severe_given_injury * p_incap_given_severe
    prob_fatal = p_injury * p_severe_given_injury * p_fatal_given_severe

    # Normalize to sum to 1
    total = prob_no_injury + prob_reported + prob_nonincap + prob_incap + prob_fatal
    prob_no_injury /= total
    prob_reported /= total
    prob_nonincap /= total
    prob_incap /= total
    prob_fatal /= total

    # Find prediction
    probs = [prob_no_injury, prob_reported, prob_nonincap, prob_incap, prob_fatal]
    max_idx = probs.index(max(probs))
    prediction_label = HIERARCHICAL_CLASS_NAMES[max_idx]
    confidence = max(probs)

    return HierarchicalPredictionResponse(
        prediction=prediction_label,  # type: ignore
        probabilities=HierarchicalProbabilities(
            no_injury=float(prob_no_injury),
            reported_not_evident=float(prob_reported),
            nonincapacitating=float(prob_nonincap),
            incapacitating=float(prob_incap),
            fatal=float(prob_fatal),
        ),
        confidence=float(confidence),
        model_name=model_name,
    )


def predict_multiclass_nn(request: PredictionRequest) -> HierarchicalPredictionResponse:
    """Make a 5-class severity prediction using the neural network model.

    Args:
        request: PredictionRequest with input features.

    Returns:
        HierarchicalPredictionResponse with 5-class prediction and probabilities.
    """
    model_name = MODEL_TYPE_REGISTRY["multiclass_nn"]
    model = model_manager.load_model(model_name)

    # Prepare features - neural network uses same features as tree models
    features = prepare_features_for_nn(request, model.config.num_classes)

    # Get predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    # probabilities is shape (1, 5) for 5 classes
    probs = probabilities[0]

    # Map to class names
    prediction_idx = int(predictions[0])
    prediction_label = HIERARCHICAL_CLASS_NAMES[prediction_idx]
    confidence = float(probs[prediction_idx])

    return HierarchicalPredictionResponse(
        prediction=prediction_label,  # type: ignore
        probabilities=HierarchicalProbabilities(
            no_injury=float(probs[0]),
            reported_not_evident=float(probs[1]),
            nonincapacitating=float(probs[2]),
            incapacitating=float(probs[3]),
            fatal=float(probs[4]),
        ),
        confidence=confidence,
        model_name=model_name,
    )


def predict_by_type(request: PredictionRequest) -> PredictionResponseType:
    """Make a prediction using the model type specified in the request.

    This is the main entry point for predictions that dispatches
    to the appropriate prediction function based on model_type.

    Args:
        request: PredictionRequest with input features and model_type.

    Returns:
        Response type depends on model_type:
        - simplified -> PredictionResponse
        - hierarchical -> HierarchicalPredictionResponse (5-class)
        - zones -> ZonePredictionResponse
        - regression -> RegressionPredictionResponse
    """
    model_type = request.model_type

    if model_type == ModelType.SIMPLIFIED:
        model_name = MODEL_TYPE_REGISTRY["simplified"]
        return predict(request, model_name)

    elif model_type == ModelType.HIERARCHICAL:
        return predict_hierarchical(request)

    elif model_type == ModelType.MULTICLASS_NN:
        return predict_multiclass_nn(request)

    elif model_type == ModelType.ZONES:
        return predict_zones(request)

    elif model_type == ModelType.REGRESSION:
        return predict_regression(request)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_all_zones() -> list:
    """Get all zones with their centroids for map visualization.

    Returns:
        List of ZoneInfo objects with zone_id and center coordinates.
    """
    from api.models import ZoneInfo

    model_name = MODEL_TYPE_REGISTRY["zones"]
    zone_predictor = model_manager.load_model(model_name)

    zones = []
    for zone_id in range(zone_predictor.n_clusters):
        centroid = zone_predictor.centroids[zone_id]
        zones.append(
            ZoneInfo(
                zone_id=zone_id,
                center=(float(centroid[0]), float(centroid[1])),
                crash_count=None,  # Could be populated from training data
            )
        )

    return zones


def predict_by_zone_id(request) -> ZonePredictionResponse:
    """Make a zone-based severity prediction using zone ID instead of coordinates.

    Args:
        request: ZonePredictionByIdRequest with input features and zone_id.

    Returns:
        ZonePredictionResponse with prediction and zone information.
    """
    from api.models import ZonePredictionByIdRequest

    model_name = MODEL_TYPE_REGISTRY["zones"]
    zone_predictor = model_manager.load_model(model_name)

    zone_id = request.zone_id

    # Validate zone ID
    if zone_id < 0 or zone_id >= zone_predictor.n_clusters:
        raise ValueError(
            f"Invalid zone_id {zone_id}. Must be 0-{zone_predictor.n_clusters - 1}"
        )

    # Get zone centroid and model
    zone_center = tuple(zone_predictor.centroids[zone_id])

    if zone_id in zone_predictor.zone_models:
        zone_model = zone_predictor.zone_models[zone_id]
    else:
        # Fallback to nearest available model
        available_zones = list(zone_predictor.zone_models.keys())
        if not available_zones:
            raise RuntimeError("No zone models available")
        # Find closest available zone
        fallback_zone = min(available_zones, key=lambda z: abs(z - zone_id))
        zone_model = zone_predictor.zone_models[fallback_zone]
        logger.warning(f"Zone {zone_id} model not found, using zone {fallback_zone}")

    # Convert request to a format compatible with prepare_features_from_request
    # Create a PredictionRequest-like object with the zone's centroid
    class RequestAdapter:
        def __init__(self, req, center):
            self.person_count = req.person_count
            self.vehicle_count = req.vehicle_count
            self.first_crash_type = req.first_crash_type
            self.damage = req.damage
            self.prim_contributory_cause = req.prim_contributory_cause
            self.age_mean = req.age_mean
            self.age_min = req.age_min
            self.age_max = req.age_max
            self.driver_count = req.driver_count
            self.avg_vehicle_year = req.avg_vehicle_year
            self.oldest_vehicle_year = req.oldest_vehicle_year
            self.posted_speed_limit = req.posted_speed_limit
            self.traffic_control_device = req.traffic_control_device
            self.device_condition = req.device_condition
            self.trafficway_type = req.trafficway_type
            self.lighting_condition = req.lighting_condition
            self.road_defect = req.road_defect
            self.roadway_surface_cond = req.roadway_surface_cond
            self.alignment = req.alignment
            self.weather_condition = req.weather_condition
            self.air_temperature = req.air_temperature
            self.humidity = req.humidity
            self.wind_speed = req.wind_speed
            self.rain_intensity = req.rain_intensity
            self.crash_hour = req.crash_hour
            self.crash_day_of_week = req.crash_day_of_week
            self.crash_month = req.crash_month
            self.latitude = center[0]
            self.longitude = center[1]

    adapted_request = RequestAdapter(request, zone_center)

    # Prepare features
    model_manager._current_model = model_name
    features = prepare_features_from_request(adapted_request)

    # Get predictions
    l1_proba, l2_proba = zone_model.predict_proba(features)

    # Calculate 3-class probabilities
    p_no_injury = 1.0 - l1_proba[0]
    p_injury = l1_proba[0]
    p_severe_given_injury = l2_proba[0]
    p_minor_given_injury = 1.0 - l2_proba[0]

    prob_no_injury = p_no_injury
    prob_minor = p_injury * p_minor_given_injury
    prob_severe = p_injury * p_severe_given_injury

    total = prob_no_injury + prob_minor + prob_severe
    prob_no_injury /= total
    prob_minor /= total
    prob_severe /= total

    probs = [prob_no_injury, prob_minor, prob_severe]
    class_labels = ["NO_INJURY", "MINOR", "SEVERE"]
    max_idx = probs.index(max(probs))
    prediction_label = class_labels[max_idx]
    confidence = max(probs)

    return ZonePredictionResponse(
        prediction=prediction_label,  # type: ignore
        probabilities=PredictionProbabilities(
            no_injury=float(prob_no_injury),
            minor=float(prob_minor),
            severe=float(prob_severe),
        ),
        confidence=float(confidence),
        model_name=model_name,
        zone_id=zone_id,
        zone_center=zone_center,
    )


def predict_all_zones(request) -> list[ZonePredictionResponse]:
    """Predict severity for all zones at once.

    Args:
        request: ZonePredictionByIdRequest with input features.

    Returns:
        List of ZonePredictionResponse for each zone.
    """
    model_name = MODEL_TYPE_REGISTRY["zones"]
    zone_predictor = model_manager.load_model(model_name)

    predictions = []
    for zone_id in range(zone_predictor.n_clusters):
        # Create a modified request with this zone_id
        class ZoneRequest:
            pass

        zone_request = ZoneRequest()
        for attr in [
            "person_count",
            "vehicle_count",
            "first_crash_type",
            "damage",
            "prim_contributory_cause",
            "age_mean",
            "age_min",
            "age_max",
            "driver_count",
            "avg_vehicle_year",
            "oldest_vehicle_year",
            "posted_speed_limit",
            "traffic_control_device",
            "device_condition",
            "trafficway_type",
            "lighting_condition",
            "road_defect",
            "roadway_surface_cond",
            "alignment",
            "weather_condition",
            "air_temperature",
            "humidity",
            "wind_speed",
            "rain_intensity",
            "crash_hour",
            "crash_day_of_week",
            "crash_month",
        ]:
            setattr(zone_request, attr, getattr(request, attr))
        zone_request.zone_id = zone_id

        try:
            prediction = predict_by_zone_id(zone_request)
            predictions.append(prediction)
        except Exception as e:
            logger.warning(f"Failed to predict for zone {zone_id}: {e}")
            # Skip zones that fail

    return predictions
