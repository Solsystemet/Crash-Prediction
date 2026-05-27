"""Focal Loss 5-class model predictor for API.

This module loads the trained multiclass focal model and provides
prediction functionality that exactly matches the training pipeline.

Supports multiple threshold modes:
- "training": Original thresholds from training (optimized on balanced validation)
- "balanced": Aggressive thresholds to improve minority class detection
- "argmax": Simple argmax prediction (no thresholds)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from api.models import PredictionRequest
from training.multiclass_focal.model import SeverityMLP, SeverityMLPConfig
from training.multiclass_focal.calibration import predict_with_thresholds

logger = logging.getLogger(__name__)

# Class names in the same order as training (alphabetical)
FOCAL_CLASS_NAMES = [
    "FATAL",
    "INCAPACITATING INJURY",
    "NO INDICATION OF INJURY",
    "NONINCAPACITATING INJURY",
    "REPORTED, NOT EVIDENT",
]

# Mapping to API-friendly labels
FOCAL_TO_API_LABELS = {
    "FATAL": "FATAL",
    "INCAPACITATING INJURY": "INCAPACITATING",
    "NO INDICATION OF INJURY": "NO_INJURY",
    "NONINCAPACITATING INJURY": "NONINCAPACITATING",
    "REPORTED, NOT EVIDENT": "REPORTED_NOT_EVIDENT",
}

# Threshold mode type
ThresholdMode = Literal["training", "balanced", "argmax"]

# Production class priors (approximate from Chicago data)
# Used to adjust predictions for class imbalance
PRODUCTION_PRIORS = {
    0: 0.001,   # FATAL
    1: 0.011,   # INCAPACITATING INJURY  
    2: 0.86,    # NO INDICATION OF INJURY
    3: 0.10,    # NONINCAPACITATING INJURY
    4: 0.07,    # REPORTED, NOT EVIDENT
}

# Balanced thresholds - designed to improve minority detection
# Lower thresholds for NO_INJURY, lower for minorities too
# Key: make the margins more competitive
BALANCED_THRESHOLDS = np.array([
    0.02,   # FATAL - very low, predict if any signal
    0.05,   # INCAPACITATING - low threshold
    0.70,   # NO_INJURY - high threshold, requires strong signal
    0.08,   # NONINCAPACITATING - low threshold  
    0.06,   # REPORTED_NOT_EVIDENT - low threshold
])


class FocalPredictor:
    """Predictor class for the focal loss 5-class model.
    
    Supports multiple threshold modes for different use cases:
    - "training": Thresholds optimized during training (balanced validation)
    - "balanced": Aggressive thresholds to improve minority class detection
    - "argmax": Simple argmax, ignores thresholds
    """

    def __init__(
        self, 
        model_dir: Path | str,
        threshold_mode: ThresholdMode = "balanced",
    ):
        """Initialize the predictor.

        Args:
            model_dir: Directory containing model artifacts
            threshold_mode: Which threshold strategy to use
        """
        self.model_dir = Path(model_dir)
        self.model: SeverityMLP | None = None
        self.calibrator: Any = None
        self.artifacts: dict = {}
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_loaded = False
        self.threshold_mode = threshold_mode

    def get_thresholds(self, mode: ThresholdMode | None = None) -> np.ndarray:
        """Get thresholds for specified mode.
        
        Args:
            mode: Threshold mode. If None, uses instance default.
            
        Returns:
            Array of thresholds per class.
        """
        mode = mode or self.threshold_mode
        
        if mode == "training":
            return np.array(self.artifacts["thresholds"])
        elif mode == "balanced":
            return BALANCED_THRESHOLDS
        else:  # argmax
            # Return very low thresholds so argmax effectively decides
            return np.zeros(5)

    def load(self) -> None:
        """Load all model artifacts."""
        if self._is_loaded:
            return

        # Find the latest artifacts file
        artifacts_files = list(self.model_dir.glob("artifacts_*.json"))
        if not artifacts_files:
            raise FileNotFoundError(f"No artifacts file found in {self.model_dir}")
        
        # Sort by name (timestamp) and get latest
        artifacts_path = sorted(artifacts_files)[-1]
        logger.info(f"Loading artifacts from {artifacts_path}")

        with open(artifacts_path) as f:
            self.artifacts = json.load(f)

        # Load model
        model_path = self.model_dir / self.artifacts["model_path"]
        logger.info(f"Loading model from {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]
        
        self.model = SeverityMLP(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load calibrator
        calibrator_path = self.model_dir / self.artifacts["calibrator_path"]
        logger.info(f"Loading calibrator from {calibrator_path}")

        with open(calibrator_path, "rb") as f:
            self.calibrator = pickle.load(f)

        # Create label encoders for categorical features
        self._create_label_encoders()

        self._is_loaded = True
        logger.info("Focal model loaded successfully")

    def _create_label_encoders(self) -> None:
        """Create label encoders matching training pipeline."""
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

        categorical_features = {
            "WEATHER_CONDITION": WEATHER_CONDITION_OPTIONS + ["UNKNOWN"],
            "LIGHTING_CONDITION": LIGHTING_CONDITION_OPTIONS + ["UNKNOWN"],
            "FIRST_CRASH_TYPE": FIRST_CRASH_TYPE_OPTIONS + ["UNKNOWN"],
            "TRAFFICWAY_TYPE": TRAFFICWAY_TYPE_OPTIONS + ["UNKNOWN"],
            "ROADWAY_SURFACE_COND": ROADWAY_SURFACE_COND_OPTIONS + ["UNKNOWN"],
            "TRAFFIC_CONTROL_DEVICE": TRAFFIC_CONTROL_DEVICE_OPTIONS + ["UNKNOWN"],
            "DEVICE_CONDITION": DEVICE_CONDITION_OPTIONS + ["UNKNOWN"],
            "ALIGNMENT": ALIGNMENT_OPTIONS + ["UNKNOWN"],
            "ROAD_DEFECT": ROAD_DEFECT_OPTIONS + ["UNKNOWN"],
            "PRIM_CONTRIBUTORY_CAUSE": PRIM_CONTRIBUTORY_CAUSE_OPTIONS + ["UNKNOWN"],
            "DAMAGE": DAMAGE_OPTIONS + ["UNKNOWN"],
        }

        for feature_name, options in categorical_features.items():
            le = LabelEncoder()
            le.fit(options)
            self.label_encoders[feature_name] = le

    def _encode_categorical(self, feature_name: str, value: str) -> int:
        """Encode a categorical feature value."""
        if feature_name not in self.label_encoders:
            return 0

        le = self.label_encoders[feature_name]
        try:
            return int(le.transform([value])[0])
        except ValueError:
            # Unknown value, try UNKNOWN
            try:
                return int(le.transform(["UNKNOWN"])[0])
            except ValueError:
                return 0

    def _compute_engineered_features(self, request: PredictionRequest) -> dict[str, float]:
        """Compute engineered features matching training pipeline."""
        features = {}

        # Time-based features
        features["IS_PEAK_HOUR"] = 1 if request.crash_hour in [7, 8, 9, 16, 17, 18] else 0
        features["IS_NIGHT"] = 1 if request.crash_hour >= 20 or request.crash_hour < 6 else 0
        features["IS_WEEKEND"] = 1 if request.crash_day_of_week in [1, 7] else 0  # Sun=1, Sat=7

        # Vehicle age features
        current_year = 2026
        features["VEHICLE_AGE"] = current_year - request.avg_vehicle_year
        features["OLD_VEHICLE_FLAG"] = 1 if features["VEHICLE_AGE"] > 15 else 0

        # Adverse conditions
        adverse_count = 0
        if request.weather_condition not in ["CLEAR", "CLOUDY/OVERCAST"]:
            adverse_count += 1
        if request.lighting_condition in ["DARKNESS", "DUSK", "DAWN"]:
            adverse_count += 1
        if request.roadway_surface_cond not in ["DRY"]:
            adverse_count += 1
        features["ADVERSE_CONDITIONS_COUNT"] = adverse_count

        # Night + poor lighting
        features["NIGHT_POOR_LIGHTING"] = 1 if (
            features["IS_NIGHT"] and 
            request.lighting_condition in ["DARKNESS", "UNKNOWN"]
        ) else 0

        # Wet road
        features["WET_ROAD"] = 1 if request.roadway_surface_cond in [
            "WET", "SNOW OR SLUSH", "ICE"
        ] else 0

        # Speed-related
        features["HIGH_SPEED_CRASH"] = 1 if request.posted_speed_limit >= 40 else 0
        features["VERY_HIGH_SPEED"] = 1 if request.posted_speed_limit >= 55 else 0

        # Multi-vehicle
        features["MULTI_VEHICLE"] = 1 if request.vehicle_count > 1 else 0

        # Intersection (based on crash type)
        intersection_types = ["ANGLE", "TURNING", "HEAD ON"]
        features["INTERSECTION_CRASH"] = 1 if request.first_crash_type in intersection_types else 0

        # High impact crash types
        high_impact_types = ["HEAD ON", "PEDESTRIAN", "PEDALCYCLIST", "OVERTURNED", "FIXED OBJECT"]
        features["HIGH_IMPACT_CRASH_TYPE"] = 1 if request.first_crash_type in high_impact_types else 0

        # Severe damage
        features["SEVERE_DAMAGE_INDICATOR"] = 1 if request.damage == "OVER $1,500" else 0

        # Hit and run / work zone (use 0 as default, not available in request)
        features["HIT_AND_RUN"] = 0
        features["WORK_ZONE_CRASH"] = 0

        # Severity risk score - sum of risk factors
        risk_score = (
            features["HIGH_SPEED_CRASH"] * 2 +
            features["VERY_HIGH_SPEED"] * 2 +
            features["HIGH_IMPACT_CRASH_TYPE"] * 3 +
            features["NIGHT_POOR_LIGHTING"] * 1 +
            features["ADVERSE_CONDITIONS_COUNT"] +
            features["SEVERE_DAMAGE_INDICATOR"] * 2
        )
        features["SEVERITY_RISK_SCORE"] = risk_score

        return features

    def _build_feature_vector(self, request: PredictionRequest) -> np.ndarray:
        """Build feature vector matching training pipeline exactly."""
        feature_columns = self.artifacts["feature_columns"]
        scaler_mean = np.array(self.artifacts["scaler_mean"])
        scaler_scale = np.array(self.artifacts["scaler_scale"])

        # Compute engineered features
        eng_features = self._compute_engineered_features(request)

        # Build raw feature vector
        raw_features = []
        
        for col in feature_columns:
            if col == "POSTED_SPEED_LIMIT":
                raw_features.append(request.posted_speed_limit)
            elif col == "LANE_CNT":
                raw_features.append(2)  # Default
            elif col == "STREET_NO":
                raw_features.append(3000)  # Default
            elif col == "BEAT_OF_OCCURRENCE":
                raw_features.append(1200)  # Default
            elif col == "NUM_UNITS":
                raw_features.append(request.vehicle_count)
            elif col == "CRASH_HOUR":
                raw_features.append(request.crash_hour)
            elif col == "CRASH_DAY_OF_WEEK":
                raw_features.append(request.crash_day_of_week)
            elif col == "CRASH_MONTH":
                raw_features.append(request.crash_month)
            elif col == "VEHICLE_COUNT":
                raw_features.append(request.vehicle_count)
            elif col == "OLDEST_VEHICLE_YEAR":
                raw_features.append(request.oldest_vehicle_year)
            elif col == "NEWEST_VEHICLE_YEAR":
                raw_features.append(request.avg_vehicle_year)  # Use avg as proxy
            elif col == "AVG_VEHICLE_YEAR":
                raw_features.append(request.avg_vehicle_year)
            elif col == "PERSON_COUNT":
                raw_features.append(request.person_count)
            elif col == "age_clean_min":
                raw_features.append(request.age_min)
            elif col == "age_clean_max":
                raw_features.append(request.age_max)
            elif col == "age_clean_mean":
                raw_features.append(request.age_mean)
            elif col == "is_driver_sum":
                raw_features.append(request.driver_count)
            elif col == "Air Temperature":
                # Convert from Fahrenheit to Celsius (training used Celsius)
                raw_features.append((request.air_temperature - 32) * 5 / 9)
            elif col == "Humidity":
                raw_features.append(request.humidity)
            elif col == "Rain Intensity":
                raw_features.append(request.rain_intensity)
            elif col == "Interval Rain":
                raw_features.append(0)  # Default
            elif col == "Total Rain":
                raw_features.append(request.rain_intensity * 10)  # Estimate
            elif col == "Precipitation Type":
                raw_features.append(0 if request.rain_intensity == 0 else 1)
            elif col == "Wind Speed":
                raw_features.append(request.wind_speed)
            elif col == "Barometric Pressure":
                raw_features.append(878.8)  # Default from training mean
            # Engineered features
            elif col in eng_features:
                raw_features.append(eng_features[col])
            # Categorical features
            elif col == "WEATHER_CONDITION":
                raw_features.append(self._encode_categorical("WEATHER_CONDITION", request.weather_condition))
            elif col == "LIGHTING_CONDITION":
                raw_features.append(self._encode_categorical("LIGHTING_CONDITION", request.lighting_condition))
            elif col == "FIRST_CRASH_TYPE":
                raw_features.append(self._encode_categorical("FIRST_CRASH_TYPE", request.first_crash_type))
            elif col == "TRAFFICWAY_TYPE":
                raw_features.append(self._encode_categorical("TRAFFICWAY_TYPE", request.trafficway_type))
            elif col == "ROADWAY_SURFACE_COND":
                raw_features.append(self._encode_categorical("ROADWAY_SURFACE_COND", request.roadway_surface_cond))
            elif col == "TRAFFIC_CONTROL_DEVICE":
                raw_features.append(self._encode_categorical("TRAFFIC_CONTROL_DEVICE", request.traffic_control_device))
            elif col == "DEVICE_CONDITION":
                raw_features.append(self._encode_categorical("DEVICE_CONDITION", request.device_condition))
            elif col == "ALIGNMENT":
                raw_features.append(self._encode_categorical("ALIGNMENT", request.alignment))
            elif col == "ROAD_DEFECT":
                raw_features.append(self._encode_categorical("ROAD_DEFECT", request.road_defect))
            elif col == "PRIM_CONTRIBUTORY_CAUSE":
                raw_features.append(self._encode_categorical("PRIM_CONTRIBUTORY_CAUSE", request.prim_contributory_cause))
            elif col == "DAMAGE":
                raw_features.append(self._encode_categorical("DAMAGE", request.damage))
            else:
                logger.warning(f"Unknown feature column: {col}, using 0")
                raw_features.append(0)

        # Scale features
        raw_array = np.array(raw_features, dtype=np.float32)
        scaled = (raw_array - scaler_mean) / (scaler_scale + 1e-8)

        return scaled

    def predict(
        self, 
        request: PredictionRequest,
        threshold_mode: ThresholdMode | None = None,
    ) -> dict[str, Any]:
        """Make a prediction for a single crash.

        Args:
            request: PredictionRequest with crash features
            threshold_mode: Override default threshold mode for this prediction

        Returns:
            Dictionary with prediction, probabilities, and class names
        """
        if not self._is_loaded:
            self.load()

        # Build feature vector
        features = self._build_feature_vector(request)
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Calibrate probabilities
        calibrated_probs = self.calibrator.calibrate(probs)

        # Get thresholds based on mode
        mode = threshold_mode or self.threshold_mode
        thresholds = self.get_thresholds(mode)
        
        # Apply class-specific thresholds
        if mode == "argmax":
            pred_idx = int(calibrated_probs[0].argmax())
        else:
            pred_idx = predict_with_thresholds(calibrated_probs, thresholds, default_class=2)[0]

        # Get class name and API label
        class_name = FOCAL_CLASS_NAMES[pred_idx]
        api_label = FOCAL_TO_API_LABELS[class_name]

        # Build probability dict
        prob_dict = {
            FOCAL_TO_API_LABELS[name]: float(calibrated_probs[0, i])
            for i, name in enumerate(FOCAL_CLASS_NAMES)
        }

        return {
            "prediction": api_label,
            "prediction_full": class_name,
            "probabilities": prob_dict,
            "class_names": [FOCAL_TO_API_LABELS[n] for n in FOCAL_CLASS_NAMES],
            "thresholds": {
                FOCAL_TO_API_LABELS[name]: float(thresholds[i])
                for i, name in enumerate(FOCAL_CLASS_NAMES)
            },
            "threshold_mode": mode,
        }

    def predict_batch(
        self, 
        requests: list[PredictionRequest],
        threshold_mode: ThresholdMode | None = None,
    ) -> list[dict[str, Any]]:
        """Make predictions for multiple crashes.

        Args:
            requests: List of PredictionRequests
            threshold_mode: Override default threshold mode

        Returns:
            List of prediction dictionaries
        """
        if not self._is_loaded:
            self.load()

        if not requests:
            return []

        # Build feature matrix
        features_list = [self._build_feature_vector(req) for req in requests]
        X = torch.tensor(np.stack(features_list), dtype=torch.float32).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Calibrate probabilities
        calibrated_probs = self.calibrator.calibrate(probs)

        # Get thresholds based on mode
        mode = threshold_mode or self.threshold_mode
        thresholds = self.get_thresholds(mode)
        
        # Apply thresholds
        if mode == "argmax":
            pred_indices = calibrated_probs.argmax(axis=1)
        else:
            pred_indices = predict_with_thresholds(calibrated_probs, thresholds, default_class=2)

        # Build results
        results = []
        for i in range(len(requests)):
            class_name = FOCAL_CLASS_NAMES[pred_indices[i]]
            api_label = FOCAL_TO_API_LABELS[class_name]

            prob_dict = {
                FOCAL_TO_API_LABELS[name]: float(calibrated_probs[i, j])
                for j, name in enumerate(FOCAL_CLASS_NAMES)
            }

            results.append({
                "prediction": api_label,
                "prediction_full": class_name,
                "probabilities": prob_dict,
                "threshold_mode": mode,
            })

        return results
