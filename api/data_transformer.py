"""Data transformer for converting Chicago API data to model input format.

This module transforms raw JSON data from the Chicago SODA API into the
format expected by the crash severity prediction model.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from api.models import PredictionRequest

logger = logging.getLogger(__name__)

# Mapping from API injury values to our model's severity classes
INJURY_TO_SEVERITY = {
    # Severe injuries
    "FATAL": "SEVERE",
    "INCAPACITATING INJURY": "SEVERE",
    # Minor injuries
    "NONINCAPACITATING INJURY": "MINOR",
    "REPORTED, NOT EVIDENT": "MINOR",
    # No injury
    "NO INDICATION OF INJURY": "NO_INJURY",
}

# Default values for missing fields
DEFAULTS = {
    "posted_speed_limit": 30,
    "traffic_control_device": "NO CONTROLS",
    "device_condition": "FUNCTIONING PROPERLY",
    "trafficway_type": "NOT DIVIDED",
    "lighting_condition": "DAYLIGHT",
    "road_defect": "NO DEFECTS",
    "roadway_surface_cond": "DRY",
    "alignment": "STRAIGHT AND LEVEL",
    "weather_condition": "CLEAR",
    "first_crash_type": "REAR END",
    "damage": "OVER $1,500",
    "prim_contributory_cause": "UNABLE TO DETERMINE",
    "air_temperature": 70.0,
    "humidity": 50.0,
    "wind_speed": 5.0,
    "rain_intensity": 0.0,
}


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    """Safely convert a value to string."""
    if value is None:
        return default
    return str(value).strip()


def extract_ground_truth(crash: dict[str, Any]) -> str | None:
    """Extract the ground truth severity class from a crash record.

    Args:
        crash: Crash record from the API

    Returns:
        Severity class (NO_INJURY, MINOR, SEVERE) or None if unknown
    """
    # Try most_severe_injury first (from crashes dataset)
    injury = crash.get("most_severe_injury")
    if injury and injury in INJURY_TO_SEVERITY:
        return INJURY_TO_SEVERITY[injury]

    # Fallback to injuries_fatal/injuries_incapacitating counts
    if _safe_int(crash.get("injuries_fatal")) > 0:
        return "SEVERE"
    if _safe_int(crash.get("injuries_incapacitating")) > 0:
        return "SEVERE"
    if _safe_int(crash.get("injuries_non_incapacitating")) > 0:
        return "MINOR"
    if _safe_int(crash.get("injuries_reported_not_evident")) > 0:
        return "MINOR"

    # If no injuries indicated
    total_injuries = _safe_int(crash.get("injuries_total"))
    if total_injuries == 0:
        return "NO_INJURY"

    return None


def aggregate_people_data(
    people: list[dict[str, Any]],
    crash_id: str,
) -> dict[str, Any]:
    """Aggregate people data for a single crash.

    Args:
        people: List of all people records
        crash_id: The crash_record_id to filter by

    Returns:
        Dictionary with aggregated people features
    """
    crash_people = [p for p in people if p.get("crash_record_id") == crash_id]

    if not crash_people:
        return {
            "person_count": 1,
            "age_mean": 35.0,
            "age_min": 25,
            "age_max": 45,
            "driver_count": 1,
        }

    # Extract ages
    ages = []
    for p in crash_people:
        age = _safe_int(p.get("age"))
        if 0 < age < 120:  # Valid age range
            ages.append(age)

    # Count drivers
    driver_count = sum(
        1 for p in crash_people if _safe_str(p.get("person_type")).upper() == "DRIVER"
    )

    return {
        "person_count": len(crash_people),
        "age_mean": sum(ages) / len(ages) if ages else 35.0,
        "age_min": min(ages) if ages else 25,
        "age_max": max(ages) if ages else 45,
        "driver_count": max(driver_count, 1),
    }


def aggregate_vehicles_data(
    vehicles: list[dict[str, Any]],
    crash_id: str,
) -> dict[str, Any]:
    """Aggregate vehicle data for a single crash.

    Args:
        vehicles: List of all vehicle records
        crash_id: The crash_record_id to filter by

    Returns:
        Dictionary with aggregated vehicle features
    """
    crash_vehicles = [v for v in vehicles if v.get("crash_record_id") == crash_id]

    if not crash_vehicles:
        return {
            "vehicle_count": 2,
            "avg_vehicle_year": 2018,
            "oldest_vehicle_year": 2015,
        }

    # Extract vehicle years
    years = []
    for v in crash_vehicles:
        year = _safe_int(v.get("vehicle_year"))
        if 1900 < year <= 2030:  # Valid year range
            years.append(year)

    return {
        "vehicle_count": len(crash_vehicles),
        "avg_vehicle_year": int(sum(years) / len(years)) if years else 2018,
        "oldest_vehicle_year": min(years) if years else 2015,
    }


def get_nearest_weather(
    weather_data: list[dict[str, Any]],
    crash_datetime: datetime,
) -> dict[str, Any]:
    """Find the nearest weather reading to a crash time.

    Args:
        weather_data: List of weather records
        crash_datetime: The datetime of the crash

    Returns:
        Weather data dictionary with relevant fields
    """
    if not weather_data:
        return {
            "air_temperature": DEFAULTS["air_temperature"],
            "humidity": DEFAULTS["humidity"],
            "wind_speed": DEFAULTS["wind_speed"],
            "rain_intensity": DEFAULTS["rain_intensity"],
        }

    # Find closest weather reading
    closest = None
    closest_diff = None

    for w in weather_data:
        try:
            timestamp_str = w.get("measurement_timestamp", "")
            if not timestamp_str:
                continue
            w_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            diff = abs((w_time.replace(tzinfo=None) - crash_datetime).total_seconds())

            if closest_diff is None or diff < closest_diff:
                closest = w
                closest_diff = diff
        except (ValueError, TypeError):
            continue

    if closest is None:
        return {
            "air_temperature": DEFAULTS["air_temperature"],
            "humidity": DEFAULTS["humidity"],
            "wind_speed": DEFAULTS["wind_speed"],
            "rain_intensity": DEFAULTS["rain_intensity"],
        }

    # Convert Celsius to Fahrenheit for air temperature
    air_temp_c = _safe_float(closest.get("air_temperature"), 21.0)  # ~70F default
    air_temp_f = (air_temp_c * 9 / 5) + 32

    return {
        "air_temperature": air_temp_f,
        "humidity": _safe_float(closest.get("humidity"), DEFAULTS["humidity"]),
        "wind_speed": _safe_float(closest.get("wind_speed"), DEFAULTS["wind_speed"]),
        "rain_intensity": _safe_float(
            closest.get("rain_intensity"), DEFAULTS["rain_intensity"]
        ),
    }


def parse_crash_datetime(crash: dict[str, Any]) -> datetime | None:
    """Parse the crash datetime from a crash record."""
    date_str = crash.get("crash_date")
    if not date_str:
        return None

    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(
            tzinfo=None
        )
    except (ValueError, TypeError):
        return None


def transform_crash_to_request(
    crash: dict[str, Any],
    people: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    weather: list[dict[str, Any]],
) -> PredictionRequest | None:
    """Transform a single crash record into a PredictionRequest.

    Args:
        crash: Crash record from the API
        people: All people records
        vehicles: All vehicle records
        weather: Weather records

    Returns:
        PredictionRequest ready for model prediction, or None if transformation fails
    """
    try:
        crash_id = crash.get("crash_record_id")
        if not crash_id:
            return None

        # Parse crash datetime
        crash_dt = parse_crash_datetime(crash)

        # Aggregate related data
        people_agg = aggregate_people_data(people, crash_id)
        vehicles_agg = aggregate_vehicles_data(vehicles, crash_id)

        # Get weather data
        if crash_dt:
            weather_agg = get_nearest_weather(weather, crash_dt)
        else:
            weather_agg = {
                "air_temperature": DEFAULTS["air_temperature"],
                "humidity": DEFAULTS["humidity"],
                "wind_speed": DEFAULTS["wind_speed"],
                "rain_intensity": DEFAULTS["rain_intensity"],
            }

        # Extract time features
        if crash_dt:
            crash_hour = crash_dt.hour
            crash_day_of_week = crash_dt.isoweekday()  # Monday=1, Sunday=7
            # Convert to Sunday=1 format used by model
            crash_day_of_week = 1 if crash_day_of_week == 7 else crash_day_of_week + 1
            crash_month = crash_dt.month
        else:
            crash_hour = _safe_int(crash.get("crash_hour"), 12)
            crash_day_of_week = _safe_int(crash.get("crash_day_of_week"), 3)
            crash_month = _safe_int(crash.get("crash_month"), 6)

        # Build the request
        return PredictionRequest(
            # Crash info
            person_count=min(max(people_agg["person_count"], 1), 50),
            vehicle_count=min(max(vehicles_agg["vehicle_count"], 1), 20),
            first_crash_type=_safe_str(
                crash.get("first_crash_type"), DEFAULTS["first_crash_type"]
            ),
            damage=_safe_str(crash.get("damage"), DEFAULTS["damage"]),
            prim_contributory_cause=_safe_str(
                crash.get("prim_contributory_cause"),
                DEFAULTS["prim_contributory_cause"],
            ),
            # People features
            age_mean=max(min(people_agg["age_mean"], 120), 0),
            age_min=max(min(people_agg["age_min"], 120), 0),
            age_max=max(min(people_agg["age_max"], 120), 0),
            driver_count=min(max(people_agg["driver_count"], 0), 20),
            # Vehicle features
            avg_vehicle_year=max(min(vehicles_agg["avg_vehicle_year"], 2030), 1900),
            oldest_vehicle_year=max(
                min(vehicles_agg["oldest_vehicle_year"], 2030), 1900
            ),
            # Road/Location features
            posted_speed_limit=max(
                min(
                    _safe_int(
                        crash.get("posted_speed_limit"), DEFAULTS["posted_speed_limit"]
                    ),
                    100,
                ),
                0,
            ),
            traffic_control_device=_safe_str(
                crash.get("traffic_control_device"), DEFAULTS["traffic_control_device"]
            ),
            device_condition=_safe_str(
                crash.get("device_condition"), DEFAULTS["device_condition"]
            ),
            trafficway_type=_safe_str(
                crash.get("trafficway_type"), DEFAULTS["trafficway_type"]
            ),
            lighting_condition=_safe_str(
                crash.get("lighting_condition"), DEFAULTS["lighting_condition"]
            ),
            road_defect=_safe_str(crash.get("road_defect"), DEFAULTS["road_defect"]),
            roadway_surface_cond=_safe_str(
                crash.get("roadway_surface_cond"), DEFAULTS["roadway_surface_cond"]
            ),
            alignment=_safe_str(crash.get("alignment"), DEFAULTS["alignment"]),
            # Weather features
            weather_condition=_safe_str(
                crash.get("weather_condition"), DEFAULTS["weather_condition"]
            ),
            air_temperature=max(min(weather_agg["air_temperature"], 150), -50),
            humidity=max(min(weather_agg["humidity"], 100), 0),
            wind_speed=max(min(weather_agg["wind_speed"], 200), 0),
            rain_intensity=max(min(weather_agg["rain_intensity"], 10), 0),
            # Time features
            crash_hour=max(min(crash_hour, 23), 0),
            crash_day_of_week=max(min(crash_day_of_week, 7), 1),
            crash_month=max(min(crash_month, 12), 1),
        )
    except Exception as e:
        logger.warning(f"Failed to transform crash {crash.get('crash_record_id')}: {e}")
        return None


def transform_all_crashes(
    crashes: list[dict[str, Any]],
    people: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    weather: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], PredictionRequest, str]]:
    """Transform all crashes into prediction requests with ground truth.

    Args:
        crashes: List of crash records
        people: List of people records
        vehicles: List of vehicle records
        weather: List of weather records

    Returns:
        List of tuples (crash_record, prediction_request, ground_truth_severity)
    """
    results = []

    for crash in crashes:
        # Extract ground truth
        ground_truth = extract_ground_truth(crash)
        if ground_truth is None:
            continue  # Skip crashes without clear ground truth

        # Transform to prediction request
        request = transform_crash_to_request(crash, people, vehicles, weather)
        if request is None:
            continue

        results.append((crash, request, ground_truth))

    logger.info(f"Transformed {len(results)} crashes out of {len(crashes)}")
    return results
