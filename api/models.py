"""Pydantic models for API request and response schemas."""

from pydantic import BaseModel, Field
from typing import Literal


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
