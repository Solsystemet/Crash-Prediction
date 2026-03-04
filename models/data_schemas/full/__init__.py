"""Full Pandera schemas for all datasets."""

from .traffic_crashes import TrafficCrashesSchema
from .traffic_crashes_people import TrafficCrashesPeopleSchema
from .traffic_crashes_vehicles import TrafficCrashesVehiclesSchema
from .traffic_tracker import TrafficTrackerSchema
from .weather_stations import WeatherStationsSchema

__all__ = [
    "TrafficCrashesSchema",
    "TrafficCrashesPeopleSchema",
    "TrafficCrashesVehiclesSchema",
    "TrafficTrackerSchema",
    "WeatherStationsSchema",
]
