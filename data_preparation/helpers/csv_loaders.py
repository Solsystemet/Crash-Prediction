from pathlib import Path

import pandas as pd

from models.data_schemas.full.traffic_crashes import TrafficCrashesSchema
from models.data_schemas.full.traffic_crashes_people import TrafficCrashesPeopleSchema
from models.data_schemas.full.traffic_crashes_vehicles import (
    TrafficCrashesVehiclesSchema,
)
from models.data_schemas.full.traffic_tracker import TrafficTrackerSchema
from models.data_schemas.full.weather_stations import WeatherStationsSchema


DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

TRAFFIC_CRASHES_CSV = DATA_DIR / "traffic_crashes.csv"
TRAFFIC_TRACKER_CSV = (
    DATA_DIR
    / "chicago_traffic_tracker_historical_congestion_estimates_by_segment_2024_current.csv"
)
TRAFFIC_CRASHES_PEOPLE_CSV = DATA_DIR / "traffic-crashes-people.csv"
TRAFFIC_CRASHES_VEHICLES_CSV = DATA_DIR / "traffic_crashes_vehicles.csv"
WEATHER_STATIONS_CSV = DATA_DIR / "beach_weather_stations_automated_sensors.csv"


def _load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path, low_memory=False)


def _strip_thousands_separators(
    dataframe: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Remove comma thousands separators so values can be coerced to numeric."""
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].replace(r",", "", regex=True)
    return dataframe


def get_traffic_crashes():
    dataframe = _load_and_validate(TRAFFIC_CRASHES_CSV)
    dataframe = _strip_thousands_separators(dataframe, ["LANE_CNT"])
    return TrafficCrashesSchema.validate(dataframe)


def get_crash_people():
    dataframe = _load_and_validate(TRAFFIC_CRASHES_PEOPLE_CSV)
    return TrafficCrashesPeopleSchema.validate(dataframe)


def get_crash_vehicles():
    dataframe = _load_and_validate(TRAFFIC_CRASHES_VEHICLES_CSV)
    return TrafficCrashesVehiclesSchema.validate(dataframe)


def get_traffic_tracker():
    dataframe = _load_and_validate(TRAFFIC_TRACKER_CSV)
    return TrafficTrackerSchema.validate(dataframe)


def get_weather_stations():
    dataframe = _load_and_validate(WEATHER_STATIONS_CSV)
    dataframe = _strip_thousands_separators(
        dataframe, ["Total Rain", "Solar Radiation"]
    )
    return WeatherStationsSchema.validate(dataframe)
