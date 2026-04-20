"""Pydantic models for API request and response schemas."""

from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal, Optional


class ModelType(str, Enum):
    """Available model types for prediction."""

    SIMPLIFIED = "simplified"  # 3-class severity
    HIERARCHICAL = "hierarchical"  # 5-class severity
    ZONES = "zones"  # Zone-based severity
    REGRESSION = "regression"  # Crash count prediction


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

    # Model selection (optional - defaults to simplified 3-class)
    model_type: ModelType = Field(
        default=ModelType.SIMPLIFIED,
        description="Type of model to use for prediction",
    )

    # Zone-based model fields (optional - used when model_type=zones)
    latitude: float | None = Field(
        default=None,
        ge=-90,
        le=90,
        description="Latitude of crash location (required for zone-based model)",
    )
    longitude: float | None = Field(
        default=None,
        ge=-180,
        le=180,
        description="Longitude of crash location (required for zone-based model)",
    )

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


# Hierarchical 5-class probabilities
class HierarchicalProbabilities(BaseModel):
    """Probability scores for hierarchical 5-class prediction."""

    no_injury: float = Field(ge=0, le=1)
    reported_not_evident: float = Field(ge=0, le=1)
    nonincapacitating: float = Field(ge=0, le=1)
    incapacitating: float = Field(ge=0, le=1)
    fatal: float = Field(ge=0, le=1)


class HierarchicalPredictionResponse(BaseModel):
    """Response schema for hierarchical 5-class prediction."""

    prediction: Literal[
        "NO_INJURY",
        "REPORTED_NOT_EVIDENT",
        "NONINCAPACITATING",
        "INCAPACITATING",
        "FATAL",
    ] = Field(description="Predicted severity class (5-class)")
    probabilities: HierarchicalProbabilities = Field(
        description="Probability scores for each class"
    )
    confidence: float = Field(ge=0, le=1)
    model_name: str


# Zone-based prediction response (extends simplified with zone info)
class ZonePredictionResponse(BaseModel):
    """Response schema for zone-based severity prediction."""

    prediction: Literal["NO_INJURY", "MINOR", "SEVERE"] = Field(
        description="Predicted severity class"
    )
    probabilities: PredictionProbabilities = Field(
        description="Probability scores for each class"
    )
    confidence: float = Field(ge=0, le=1)
    model_name: str
    zone_id: int = Field(description="Geographic zone ID used for prediction")
    zone_center: tuple[float, float] = Field(
        description="Center coordinates (lat, lng) of the zone"
    )


# Regression prediction response
class RegressionPredictionResponse(BaseModel):
    """Response schema for crash count regression prediction."""

    predicted_count: float = Field(ge=0, description="Predicted number of crashes")
    confidence_interval: tuple[float, float] = Field(
        description="95% confidence interval (lower, upper)"
    )
    zone_id: int | None = Field(
        default=None, description="Zone ID if zone-specific prediction"
    )
    time_period: str = Field(
        description="Time period for prediction (e.g., 'daily', 'hourly')"
    )
    model_name: str


# Zone information for map visualization
class ZoneInfo(BaseModel):
    """Information about a geographic zone."""

    zone_id: int = Field(description="Zone ID (0-indexed)")
    center: tuple[float, float] = Field(
        description="Center coordinates (lat, lng) of the zone"
    )
    crash_count: int | None = Field(
        default=None, description="Number of crashes in this zone (training data)"
    )


class ZonesResponse(BaseModel):
    """Response schema for listing all zones."""

    zones: list[ZoneInfo] = Field(description="List of all zones")
    total_zones: int = Field(description="Total number of zones")


# Request for zone-based prediction by zone ID (no lat/lng required)
class ZonePredictionByIdRequest(BaseModel):
    """Request schema for zone-based prediction using zone ID."""

    zone_id: int = Field(ge=0, le=100, description="Zone ID to predict for")

    # Crash information
    person_count: int = Field(ge=1, le=50)
    vehicle_count: int = Field(ge=1, le=20)
    first_crash_type: str
    damage: str
    prim_contributory_cause: str

    # People features
    age_mean: float = Field(ge=0, le=120)
    age_min: int = Field(ge=0, le=120)
    age_max: int = Field(ge=0, le=120)
    driver_count: int = Field(ge=0, le=20)

    # Vehicle features
    avg_vehicle_year: int = Field(ge=1900, le=2030)
    oldest_vehicle_year: int = Field(ge=1900, le=2030)

    # Road features
    posted_speed_limit: int = Field(ge=0, le=100)
    traffic_control_device: str
    device_condition: str
    trafficway_type: str
    lighting_condition: str
    road_defect: str
    roadway_surface_cond: str
    alignment: str

    # Weather features
    weather_condition: str
    air_temperature: float = Field(ge=-50, le=150)
    humidity: float = Field(ge=0, le=100)
    wind_speed: float = Field(ge=0, le=200)
    rain_intensity: float = Field(ge=0, le=10)

    # Time features
    crash_hour: int = Field(ge=0, le=23)
    crash_day_of_week: int = Field(ge=1, le=7)
    crash_month: int = Field(ge=1, le=12)


# Response for predicting all zones at once
class AllZonesPredictionResponse(BaseModel):
    """Response schema for predicting all zones at once."""

    predictions: list[ZonePredictionResponse] = Field(
        description="Predictions for each zone"
    )
    total_zones: int = Field(description="Total number of zones predicted")


# Accuracy evaluation models
class ClassMetrics(BaseModel):
    """Metrics for a single class."""

    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1_score: float = Field(ge=0, le=1)
    support: int = Field(ge=0)


class AccuracyMetrics(BaseModel):
    """Overall accuracy metrics."""

    overall_accuracy: float = Field(ge=0, le=1)
    sample_count: int = Field(ge=0)
    per_class_metrics: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]
    class_labels: list[str]
    time_range_days: int
    computed_at: datetime
    f1_macro: float = Field(ge=0, le=1)
    f1_micro: float = Field(ge=0, le=1)


class PredictionWithActual(BaseModel):
    """A prediction compared against actual outcome."""

    crash_record_id: str
    crash_date: datetime | str | None
    predicted_severity: str
    actual_severity: str
    is_correct: bool
    confidence: float = Field(ge=0, le=1)
    latitude: float | None = None
    longitude: float | None = None
    weather_condition: str | None = None
    lighting_condition: str | None = None
    first_crash_type: str | None = None
    posted_speed_limit: str | None = None


class AccuracyResponse(BaseModel):
    """Response schema for accuracy evaluation."""

    metrics: AccuracyMetrics
    predictions: list[PredictionWithActual]


class MapPrediction(BaseModel):
    """A prediction with coordinates for map display."""

    crash_record_id: str
    crash_date: datetime
    predicted_severity: str
    actual_severity: str
    is_correct: bool
    confidence: float = Field(ge=0, le=1)
    latitude: float
    longitude: float
    weather_condition: str | None = None
    lighting_condition: str | None = None


class MapDataResponse(BaseModel):
    """Response schema for map visualization data."""

    predictions: list[MapPrediction]
    total_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    incorrect_count: int = Field(ge=0)
