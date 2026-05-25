"""Configuration for the API and model registry.

This module provides a centralized configuration for available models
and their paths, making it easy to add new models in the future.
<<<<<<< HEAD

Supports dynamic discovery of models trained with different dataset combinations:
- Crash data (always included)
- Vehicle data (optional)
- People data (optional)
- Weather data (optional)
"""

import logging
import re
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
=======
"""

from pathlib import Path
from dataclasses import dataclass
>>>>>>> origin/dev

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Models directory
MODELS_DIR = PROJECT_ROOT / "models" / "trained"


@dataclass
<<<<<<< HEAD
class DatasetConfig:
    """Configuration for which datasets are included in a model.
    
    Crash data is always included as the base.
    """
    use_vehicles: bool = False
    use_people: bool = False
    use_weather: bool = False
    
    def get_suffix(self) -> str:
        """Get the naming suffix for this configuration."""
        parts = ["crash"]
        if self.use_vehicles:
            parts.append("vehicle")
        if self.use_people:
            parts.append("people")
        if self.use_weather:
            parts.append("weather")
        return "_".join(parts)
    
    def get_display_name(self) -> str:
        """Get human-readable description of datasets."""
        if not self.use_vehicles and not self.use_people and not self.use_weather:
            return "Crash Only"
        parts = []
        if self.use_vehicles:
            parts.append("Vehicle")
        if self.use_people:
            parts.append("People")
        if self.use_weather:
            parts.append("Weather")
        return "Crash + " + " + ".join(parts)
    
    @classmethod
    def from_suffix(cls, suffix: str) -> "DatasetConfig":
        """Parse a dataset suffix into a config.
        
        Examples:
            'crash' -> DatasetConfig(False, False, False)
            'crash_vehicle' -> DatasetConfig(True, False, False)
            'crash_vehicle_people_weather' -> DatasetConfig(True, True, True)
        """
        parts = suffix.lower().split("_")
        return cls(
            use_vehicles="vehicle" in parts,
            use_people="people" in parts,
            use_weather="weather" in parts,
        )


# All possible dataset combinations (8 total)
ALL_DATASET_CONFIGS = [
    DatasetConfig(v, p, w)
    for v in [False, True]
    for p in [False, True]
    for w in [False, True]
]


@dataclass
=======
>>>>>>> origin/dev
class ModelInfo:
    """Information about an available model."""

    name: str
    path: Path
    description: str
<<<<<<< HEAD
    model_type: str  # "simplified", "hierarchical", "zones", etc.
    datasets: DatasetConfig = field(default_factory=DatasetConfig)
    is_available: bool = True  # Whether the model exists on disk


# Base model definitions (without dataset suffix)
# These are the "default" models that may or may not have dataset variations
_BASE_MODELS: dict[str, dict] = {
    "simplified_3class": {
        "name": "Simplified 3-Class",
        "description": "3-class severity prediction (NO_INJURY, MINOR, SEVERE)",
        "model_type": "simplified",
    },
    "hierarchical_5class": {
        "name": "Hierarchical 5-Class",
        "description": "5-class hierarchical severity prediction with detailed injury levels",
        "model_type": "hierarchical",
    },
    "simplified_zones": {
        "name": "Zone-Based Severity",
        "description": "Geographic zone-aware 3-class severity prediction",
        "model_type": "zones",
    },
    "regression_daily": {
        "name": "Crash Count Regression",
        "description": "Predict expected number of crashes per day",
        "model_type": "regression",
    },
    "simple_rf": {
        "name": "Simple Random Forest (Baseline)",
        "description": "Vanilla Random Forest with no feature engineering - baseline model",
        "model_type": "simple",
    },
}

# Tuned boosting models (special path structure)
_TUNED_MODELS: dict[str, dict] = {
    "tuned_lgbm": {
        "name": "LightGBM (Tuned)",
        "path": MODELS_DIR / "tuned_boosting" / "lgbm",
        "description": "Hyperparameter-tuned LightGBM with Optuna",
        "model_type": "tuned",
    },
    "tuned_xgb": {
        "name": "XGBoost (Tuned)",
        "path": MODELS_DIR / "tuned_boosting" / "xgb",
        "description": "Hyperparameter-tuned XGBoost with Optuna",
        "model_type": "tuned",
    },
    "tuned_catboost": {
        "name": "CatBoost (Tuned)",
        "path": MODELS_DIR / "tuned_boosting" / "catboost",
        "description": "Hyperparameter-tuned CatBoost with Optuna",
        "model_type": "tuned",
    },
    "tuned_rf": {
        "name": "Random Forest (Tuned)",
        "path": MODELS_DIR / "tuned_boosting" / "rf",
        "description": "Hyperparameter-tuned Random Forest with Optuna",
        "model_type": "tuned",
    },
    "tabnet": {
        "name": "TabNet",
        "path": MODELS_DIR / "tabnet",
        "description": "Attention-based deep learning for tabular data",
        "model_type": "deep",
    },
}


def discover_models() -> dict[str, ModelInfo]:
    """Discover all available models including dataset variations.
    
    Scans the models/trained directory for:
    1. Base models (e.g., simplified_3class)
    2. Dataset variations (e.g., simplified_3class_crash_vehicle_people)
    
    Returns:
        Dictionary mapping model keys to ModelInfo objects.
    """
    registry: dict[str, ModelInfo] = {}
    
    if not MODELS_DIR.exists():
        logger.warning(f"Models directory not found: {MODELS_DIR}")
        return registry
    
    # Pattern to match base model name and optional dataset suffix
    # e.g., "simplified_3class" or "simplified_3class_crash_vehicle_people"
    dataset_suffix_pattern = re.compile(
        r"^(.+?)(_crash(?:_vehicle)?(?:_people)?(?:_weather)?)?$"
    )
    
    # Scan directory for model folders
    for path in MODELS_DIR.iterdir():
        if not path.is_dir():
            continue
            
        dir_name = path.name
        
        # Skip special directories
        if dir_name in ("tuned_boosting", "tmp", "__pycache__"):
            continue
        
        # Try to match against known base models with dataset suffix
        match = dataset_suffix_pattern.match(dir_name)
        if not match:
            continue
            
        base_name = match.group(1)
        suffix = match.group(2) or ""
        
        # Check if this is a known base model type
        if base_name in _BASE_MODELS:
            base_info = _BASE_MODELS[base_name]
            
            # Parse dataset config from suffix
            if suffix:
                datasets = DatasetConfig.from_suffix(suffix.lstrip("_"))
            else:
                datasets = DatasetConfig()  # Default: crash only
            
            # Create model info
            model_key = dir_name
            display_name = base_info["name"]
            if suffix:
                display_name += f" ({datasets.get_display_name()})"
            
            registry[model_key] = ModelInfo(
                name=display_name,
                path=path,
                description=base_info["description"],
                model_type=base_info["model_type"],
                datasets=datasets,
                is_available=True,
            )
        else:
            # Unknown model type - add with generic info
            registry[dir_name] = ModelInfo(
                name=dir_name.replace("_", " ").title(),
                path=path,
                description=f"Model: {dir_name}",
                model_type="unknown",
                datasets=DatasetConfig(),
                is_available=True,
            )
    
    # Add tuned/special models (fixed paths, always available if exists)
    for key, info in _TUNED_MODELS.items():
        path = info.get("path", MODELS_DIR / key)
        registry[key] = ModelInfo(
            name=info["name"],
            path=path,
            description=info["description"],
            model_type=info["model_type"],
            datasets=DatasetConfig(),
            is_available=path.exists(),
        )
    
    return registry


def get_all_possible_models() -> dict[str, ModelInfo]:
    """Get all possible model combinations (including not-yet-trained).
    
    Returns a registry that includes:
    - All discovered (trained) models
    - All possible dataset combinations for base model types (marked as unavailable if not trained)
    
    Returns:
        Dictionary mapping model keys to ModelInfo objects.
    """
    registry = discover_models()
    
    # For each base model type, ensure all 8 dataset combinations exist
    for base_key, base_info in _BASE_MODELS.items():
        for config in ALL_DATASET_CONFIGS:
            suffix = config.get_suffix()
            model_key = f"{base_key}_{suffix}"
            
            # Skip if already discovered
            if model_key in registry:
                continue
            
            # Also check if base model without suffix exists (backwards compat)
            if suffix == "crash" and base_key in registry:
                continue
            
            # Add as unavailable
            path = MODELS_DIR / model_key
            display_name = f"{base_info['name']} ({config.get_display_name()})"
            
            registry[model_key] = ModelInfo(
                name=display_name,
                path=path,
                description=base_info["description"],
                model_type=base_info["model_type"],
                datasets=config,
                is_available=False,
            )
    
    return registry


# Discover models at module load time
MODEL_REGISTRY: dict[str, ModelInfo] = discover_models()


def refresh_model_registry() -> None:
    """Re-scan the models directory and update the registry.
    
    Call this after training new models to pick up newly available models.
    """
    global MODEL_REGISTRY
    MODEL_REGISTRY = discover_models()
    logger.info(f"Model registry refreshed: {len(MODEL_REGISTRY)} models found")

=======
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

>>>>>>> origin/dev
# Default model to use
DEFAULT_MODEL = "simplified_3class"

# Model type to registry key mapping
MODEL_TYPE_REGISTRY = {
    "simplified": "simplified_3class",
<<<<<<< HEAD
    "simple": "simple_rf",
=======
>>>>>>> origin/dev
    "hierarchical": "hierarchical_5class",
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
