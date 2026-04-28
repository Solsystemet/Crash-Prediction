"""Configuration for the API and model registry.

This module provides a centralized configuration for available models
and their paths, making it easy to add new models in the future.
"""

from pathlib import Path
from dataclasses import dataclass

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Models directory
MODELS_DIR = PROJECT_ROOT / "models" / "trained"


@dataclass
class ModelInfo:
    """Information about an available model."""

    name: str
    path: Path
    description: str
    model_type: str  # "simplified" or "hierarchical"


# Registry of available models
MODEL_REGISTRY: dict[str, ModelInfo] = {
    "simplified_3class": ModelInfo(
        name="Simplified 3-Class",
        path=MODELS_DIR / "simplified_3class",
        description="3-class severity prediction (NO_INJURY, MINOR, SEVERE)",
        model_type="simplified",
    ),
    "hierarchical_5class": ModelInfo(
        name="Hierarchical 5-Class",
        path=MODELS_DIR / "hierarchical_5class",
        description="5-class hierarchical severity prediction with detailed injury levels",
        model_type="hierarchical",
    ),
    "multiclass_nn": ModelInfo(
        name="Neural Network 5-Class",
        path=MODELS_DIR / "multiclass_nn",
        description="5-class severity prediction using deep learning neural network",
        model_type="multiclass_nn",
    ),
    "simplified_zones": ModelInfo(
        name="Zone-Based Severity",
        path=MODELS_DIR / "simplified_zones",
        description="Geographic zone-aware 3-class severity prediction",
        model_type="zones",
    ),
    "regression_daily": ModelInfo(
        name="Crash Count Regression",
        path=MODELS_DIR / "regression_daily",
        description="Predict expected number of crashes per day",
        model_type="regression",
    ),
}

# Default model to use
DEFAULT_MODEL = "simplified_3class"

# Model type to registry key mapping
MODEL_TYPE_REGISTRY = {
    "simplified": "simplified_3class",
    "hierarchical": "hierarchical_5class",
    "multiclass_nn": "multiclass_nn",
    "zones": "simplified_zones",
    "regression": "regression_daily",
}


# Feature value options for categorical features
# These are the allowed values that the frontend can use in dropdowns

FIRST_CRASH_TYPE_OPTIONS = [
    "REAR END",
    "TURNING",
    "ANGLE",
    "SIDESWIPE SAME DIRECTION",
    "PARKED MOTOR VEHICLE",
    "HEAD ON",
    "FIXED OBJECT",
    "SIDESWIPE OPPOSITE DIRECTION",
    "PEDESTRIAN",
    "PEDALCYCLIST",
    "OTHER OBJECT",
    "ANIMAL",
    "OVERTURNED",
    "TRAIN",
    "OTHER NONCOLLISION",
]

DAMAGE_OPTIONS = [
    "$500 OR LESS",
    "$501 - $1,500",
    "OVER $1,500",
]

PRIM_CONTRIBUTORY_CAUSE_OPTIONS = [
    "FAILING TO YIELD RIGHT-OF-WAY",
    "FOLLOWING TOO CLOSELY",
    "IMPROPER BACKING",
    "IMPROPER LANE USAGE",
    "IMPROPER OVERTAKING/PASSING",
    "IMPROPER TURNING/NO SIGNAL",
    "DISREGARDING TRAFFIC SIGNALS",
    "DISREGARDING STOP SIGN",
    "DRIVING SKILLS/KNOWLEDGE/EXPERIENCE",
    "DISTRACTION - FROM INSIDE VEHICLE",
    "DISTRACTION - FROM OUTSIDE VEHICLE",
    "DISTRACTION - OTHER ELECTRONIC DEVICE",
    "TEXTING",
    "CELL PHONE USE OTHER THAN TEXTING",
    "UNDER THE INFLUENCE OF ALCOHOL/DRUGS",
    "PHYSICAL CONDITION OF DRIVER",
    "OPERATING VEHICLE IN ERRATIC/RECKLESS MANNER",
    "EXCEEDING SAFE SPEED FOR CONDITIONS",
    "EXCEEDING AUTHORIZED SPEED LIMIT",
    "WEATHER",
    "ROAD ENGINEERING/SURFACE/MARKING DEFECTS",
    "ROAD CONSTRUCTION/MAINTENANCE",
    "VISION OBSCURED",
    "ANIMAL",
    "OBSTRUCTED CROSSWALKS",
    "BICYCLE ADVANCING LEGALLY ON RED LIGHT",
    "EVASIVE ACTION DUE TO ANIMAL/OBJECT/NONMOTORIST",
    "NOT APPLICABLE",
    "UNABLE TO DETERMINE",
]

WEATHER_CONDITION_OPTIONS = [
    "CLEAR",
    "CLOUDY/OVERCAST",
    "RAIN",
    "SNOW",
    "SLEET/HAIL",
    "FREEZING RAIN/DRIZZLE",
    "FOG/SMOKE/HAZE",
    "SEVERE CROSS WIND GATE",
    "BLOWING SAND/SOIL/DIRT",
    "BLOWING SNOW",
    "OTHER",
    "UNKNOWN",
]

LIGHTING_CONDITION_OPTIONS = [
    "DAYLIGHT",
    "DAWN",
    "DUSK",
    "DARKNESS",
    "DARKNESS, LIGHTED ROAD",
    "UNKNOWN",
]

ROADWAY_SURFACE_COND_OPTIONS = [
    "DRY",
    "WET",
    "SNOW OR SLUSH",
    "ICE",
    "SAND, MUD, DIRT",
    "OTHER",
    "UNKNOWN",
]

TRAFFIC_CONTROL_DEVICE_OPTIONS = [
    "NO CONTROLS",
    "TRAFFIC SIGNAL",
    "STOP SIGN/FLASHER",
    "YIELD",
    "LANE USE MARKING",
    "SCHOOL ZONE",
    "RAILROAD CROSSING GATE",
    "POLICE/FLAGMAN",
    "OTHER WARNING SIGN",
    "OTHER REGULATORY SIGN",
    "OTHER",
    "UNKNOWN",
]

DEVICE_CONDITION_OPTIONS = [
    "FUNCTIONING PROPERLY",
    "NOT FUNCTIONING",
    "FUNCTIONING IMPROPERLY",
    "WORN REFLECTIVE MATERIAL",
    "MISSING",
    "NO CONTROLS",
    "OTHER",
    "UNKNOWN",
]

TRAFFICWAY_TYPE_OPTIONS = [
    "NOT DIVIDED",
    "DIVIDED - W/MEDIAN (NOT RAISED)",
    "DIVIDED - W/MEDIAN BARRIER",
    "ONE-WAY",
    "CENTER TURN LANE",
    "PARKING LOT",
    "ALLEY",
    "DRIVEWAY",
    "RAMP",
    "UNKNOWN",
    "OTHER",
]

ROAD_DEFECT_OPTIONS = [
    "NO DEFECTS",
    "RUT, HOLES",
    "WORN SURFACE",
    "DEBRIS ON ROADWAY",
    "SHOULDER DEFECT",
    "OTHER",
    "UNKNOWN",
]

ALIGNMENT_OPTIONS = [
    "STRAIGHT AND LEVEL",
    "STRAIGHT ON GRADE",
    "STRAIGHT ON HILLCREST",
    "CURVE, LEVEL",
    "CURVE ON GRADE",
    "CURVE ON HILLCREST",
    "OTHER",
    "UNKNOWN",
]

# All feature options grouped for API response
FEATURE_OPTIONS = {
    "first_crash_type": FIRST_CRASH_TYPE_OPTIONS,
    "damage": DAMAGE_OPTIONS,
    "prim_contributory_cause": PRIM_CONTRIBUTORY_CAUSE_OPTIONS,
    "weather_condition": WEATHER_CONDITION_OPTIONS,
    "lighting_condition": LIGHTING_CONDITION_OPTIONS,
    "roadway_surface_cond": ROADWAY_SURFACE_COND_OPTIONS,
    "traffic_control_device": TRAFFIC_CONTROL_DEVICE_OPTIONS,
    "device_condition": DEVICE_CONDITION_OPTIONS,
    "trafficway_type": TRAFFICWAY_TYPE_OPTIONS,
    "road_defect": ROAD_DEFECT_OPTIONS,
    "alignment": ALIGNMENT_OPTIONS,
}
