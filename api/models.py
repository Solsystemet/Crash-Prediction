"""Pydantic models for API request and response schemas."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal, Optional


class PredictionRequest(BaseModel):
    """Request schema for crash severity prediction.

    Contains all features required by the model to make a prediction.
    """

    # Crash information (most important features)
    person_count: int = Field(
        ge=1, le=50, description="Number of people involved in the crash"
    )
    vehicle_count: int = Field(ge=1, le=20, description="Number of vehicles involved")
    first_crash_type: str = Field(
        description="Type of crash (e.g., 'REAR END', 'TURNING', 'ANGLE')"
    )
    damage: str = Field(
        description="Damage level (e.g., '$500 OR LESS', 'OVER $1,500')"
    )
    prim_contributory_cause: str = Field(
        description="Primary contributory cause of the crash"
    )

    # People aggregation features
    age_mean: float = Field(ge=0, le=120, description="Average age of people involved")
    age_min: int = Field(ge=0, le=120, description="Minimum age of people involved")
    age_max: int = Field(ge=0, le=120, description="Maximum age of people involved")
    driver_count: int = Field(ge=0, le=20, description="Number of drivers involved")

    # Vehicle features
    avg_vehicle_year: int = Field(ge=1900, le=2030, description="Average vehicle year")
    oldest_vehicle_year: int = Field(
        ge=1900, le=2030, description="Oldest vehicle year"
    )

    # Road/Location features
    posted_speed_limit: int = Field(
        ge=0, le=100, description="Posted speed limit in mph"
    )
    traffic_control_device: str = Field(
        description="Traffic control device (e.g., 'TRAFFIC SIGNAL', 'STOP SIGN')"
    )
    device_condition: str = Field(description="Condition of the traffic control device")
    trafficway_type: str = Field(
        description="Type of trafficway (e.g., 'DIVIDED', 'NOT DIVIDED')"
    )
    lighting_condition: str = Field(
        description="Lighting condition (e.g., 'DAYLIGHT', 'DARKNESS')"
    )
    road_defect: str = Field(
        description="Road defects (e.g., 'NO DEFECTS', 'RUT, HOLES')"
    )
    roadway_surface_cond: str = Field(
        description="Road surface condition (e.g., 'DRY', 'WET')"
    )
    alignment: str = Field(description="Road alignment (e.g., 'STRAIGHT AND LEVEL')")

    # Weather features
    weather_condition: str = Field(
        description="Weather condition (e.g., 'CLEAR', 'RAIN')"
    )
    air_temperature: float = Field(
        ge=-50, le=150, description="Air temperature in Fahrenheit"
    )
    humidity: float = Field(ge=0, le=100, description="Humidity percentage")
    wind_speed: float = Field(ge=0, le=200, description="Wind speed in mph")
    rain_intensity: float = Field(ge=0, le=10, description="Rain intensity")

    # Time features
    crash_hour: int = Field(ge=0, le=23, description="Hour of the crash (0-23)")
    crash_day_of_week: int = Field(
        ge=1, le=7, description="Day of week (1=Sunday, 7=Saturday)"
    )
    crash_month: int = Field(ge=1, le=12, description="Month of the crash (1-12)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "person_count": 2,
                    "vehicle_count": 2,
                    "first_crash_type": "REAR END",
                    "damage": "OVER $1,500",
                    "prim_contributory_cause": "FOLLOWING TOO CLOSELY",
                    "age_mean": 35.0,
                    "age_min": 25,
                    "age_max": 45,
                    "driver_count": 2,
                    "avg_vehicle_year": 2018,
                    "oldest_vehicle_year": 2015,
                    "posted_speed_limit": 30,
                    "traffic_control_device": "TRAFFIC SIGNAL",
                    "device_condition": "FUNCTIONING PROPERLY",
                    "trafficway_type": "NOT DIVIDED",
                    "lighting_condition": "DAYLIGHT",
                    "road_defect": "NO DEFECTS",
                    "roadway_surface_cond": "DRY",
                    "alignment": "STRAIGHT AND LEVEL",
                    "weather_condition": "CLEAR",
                    "air_temperature": 70.0,
                    "humidity": 50.0,
                    "wind_speed": 10.0,
                    "rain_intensity": 0.0,
                    "crash_hour": 14,
                    "crash_day_of_week": 3,
                    "crash_month": 6,
                }
            ]
        }
    }


class PredictionProbabilities(BaseModel):
    """Probability scores for each severity class."""

    no_injury: float = Field(ge=0, le=1, description="Probability of no injury")
    minor: float = Field(ge=0, le=1, description="Probability of minor injury")
    severe: float = Field(ge=0, le=1, description="Probability of severe injury")


class PredictionResponse(BaseModel):
    """Response schema for crash severity prediction."""

    prediction: Literal["NO_INJURY", "MINOR", "SEVERE"] = Field(
        description="Predicted severity class"
    )
    probabilities: PredictionProbabilities = Field(
        description="Probability scores for each class"
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence score (max probability)"
    )
    model_name: str = Field(description="Name of the model used for prediction")


class FeatureOptionsResponse(BaseModel):
    """Response schema for available feature options."""

    first_crash_type: list[str]
    damage: list[str]
    prim_contributory_cause: list[str]
    weather_condition: list[str]
    lighting_condition: list[str]
    roadway_surface_cond: list[str]
    traffic_control_device: list[str]
    device_condition: list[str]
    trafficway_type: list[str]
    road_defect: list[str]
    alignment: list[str]


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

    name: str
    description: str
    model_type: str


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str
    model_loaded: bool
    model_name: str | None


# ============================================================================
# Accuracy Evaluation Models
# ============================================================================


class ClassMetrics(BaseModel):
    """Metrics for a single severity class."""

    precision: float = Field(ge=0, le=1, description="Precision score")
    recall: float = Field(ge=0, le=1, description="Recall score")
    f1_score: float = Field(ge=0, le=1, description="F1 score")
    support: int = Field(ge=0, description="Number of samples in this class")


class AccuracyMetrics(BaseModel):
    """Overall accuracy metrics."""

    overall_accuracy: float = Field(ge=0, le=1, description="Overall accuracy")
    sample_count: int = Field(ge=0, description="Total number of samples evaluated")
    per_class_metrics: dict[str, ClassMetrics] = Field(
        description="Metrics per severity class"
    )
    confusion_matrix: list[list[int]] = Field(
        description="Confusion matrix [actual][predicted]"
    )
    class_labels: list[str] = Field(
        description="Class labels in order (for confusion matrix)"
    )
    time_range_days: int = Field(description="Number of days of data evaluated")
    computed_at: str = Field(description="ISO timestamp when metrics were computed")


class PredictionWithActual(BaseModel):
    """A single prediction compared against actual outcome."""

    crash_record_id: str = Field(description="Unique crash identifier")
    crash_date: Optional[str] = Field(description="Crash date/time in ISO format")
    predicted_severity: Literal["NO_INJURY", "MINOR", "SEVERE"] = Field(
        description="Model's predicted severity"
    )
    actual_severity: Literal["NO_INJURY", "MINOR", "SEVERE"] = Field(
        description="Actual recorded severity"
    )
    is_correct: bool = Field(description="Whether prediction matches actual")
    confidence: float = Field(ge=0, le=1, description="Model confidence score")
    latitude: Optional[float] = Field(description="Crash location latitude")
    longitude: Optional[float] = Field(description="Crash location longitude")
    weather_condition: Optional[str] = Field(description="Weather at time of crash")
    lighting_condition: Optional[str] = Field(description="Lighting conditions")
    first_crash_type: Optional[str] = Field(description="Type of crash")
    posted_speed_limit: Optional[int] = Field(description="Posted speed limit")


class AccuracyResponse(BaseModel):
    """Response containing accuracy metrics and individual predictions."""

    metrics: AccuracyMetrics
    predictions: list[PredictionWithActual]


class MapPrediction(BaseModel):
    """Prediction data formatted for map display."""

    crash_record_id: str
    crash_date: Optional[str]
    predicted_severity: Literal["NO_INJURY", "MINOR", "SEVERE"]
    actual_severity: Literal["NO_INJURY", "MINOR", "SEVERE"]
    is_correct: bool
    confidence: float
    latitude: float
    longitude: float
    weather_condition: Optional[str]
    lighting_condition: Optional[str]


class MapDataResponse(BaseModel):
    """Response containing prediction data for map visualization."""

    predictions: list[MapPrediction]
    total_count: int
    correct_count: int
    incorrect_count: int
