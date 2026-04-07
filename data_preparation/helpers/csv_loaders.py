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
    """Remove thousands separators (commas and dots) so values can be coerced to numeric."""
    for col in columns:
        if col in dataframe.columns:
            # Remove commas and dots used as thousands separators
            dataframe[col] = dataframe[col].replace(r"[,.]", "", regex=True)
    return dataframe


def _fix_decimal_separator(
    dataframe: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Convert European decimal separator (comma) to period for numeric columns."""
    for col in columns:
        if col in dataframe.columns:
            # Replace comma with period for decimal values
            dataframe[col] = dataframe[col].astype(str).str.replace(",", ".", regex=False)
    return dataframe


def _fix_number(val) -> str | None:
    """Normalize a numeric string to standard decimal format.
    
    Handles cases like:
    - "1.056,1" -> "1056.1" (European: dot thousands, comma decimal)
    - "1.056.1" -> "1056.1" (malformed with multiple dots)
    - "4,00" -> "4.00" (just comma decimal)
    - Single dot with no comma: unchanged (already correct)
    """
    s = str(val).strip()
    if s == "nan" or s == "None" or s == "":
        return None
    
    dots = s.count(".")
    commas = s.count(",")
    
    if dots > 1:
        # Multiple dots: remove all but last (e.g., "1.056.1" -> "1056.1")
        parts = s.rsplit(".", 1)
        s = parts[0].replace(".", "") + "." + parts[1]
    elif dots == 1 and commas == 1:
        # European format: "1.056,1" -> "1056.1"
        s = s.replace(".", "").replace(",", ".")
    elif commas == 1 and dots == 0:
        # Just comma decimal: "4,00" -> "4.00"
        s = s.replace(",", ".")
    # If just one dot and no comma, assume it's already correct
    return s


def _fix_european_number_format(
    dataframe: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Fix European number format in specified columns using _fix_number."""
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(_fix_number)
    return dataframe


def get_traffic_crashes():
    dataframe = _load_and_validate(TRAFFIC_CRASHES_CSV)
    dataframe = _strip_thousands_separators(dataframe, ["LANE_CNT"])
    return TrafficCrashesSchema.validate(dataframe)


def get_crash_people():
    dataframe = _load_and_validate(TRAFFIC_CRASHES_PEOPLE_CSV)
    # Fix European decimal separator in BAC value column
    dataframe = _fix_decimal_separator(dataframe, ["BAC_RESULT VALUE"])
    return TrafficCrashesPeopleSchema.validate(dataframe)


def get_crash_vehicles():
    dataframe = _load_and_validate(TRAFFIC_CRASHES_VEHICLES_CSV)
    return TrafficCrashesVehiclesSchema.validate(dataframe)


def get_traffic_tracker():
    dataframe = _load_and_validate(TRAFFIC_TRACKER_CSV)
    return TrafficTrackerSchema.validate(dataframe)


def get_weather_stations():
    dataframe = _load_and_validate(WEATHER_STATIONS_CSV)
    # Fix European number format in numeric columns
    numeric_cols = [
        "Air Temperature", "Wet Bulb Temperature", "Humidity",
        "Rain Intensity", "Interval Rain", "Total Rain",
        "Wind Direction", "Wind Speed", "Maximum Wind Speed",
        "Barometric Pressure", "Solar Radiation", "Battery Life", "Heading",
    ]
    dataframe = _fix_european_number_format(dataframe, numeric_cols)
    return WeatherStationsSchema.validate(dataframe)
