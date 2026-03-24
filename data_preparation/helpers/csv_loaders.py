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


def _convert_european_decimals(
    dataframe: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Convert European decimal format (comma) to standard format (period).
    
    Handles values like "11,40" -> "11.40" for proper float parsing.
    """
    for col in columns:
        if col in dataframe.columns:
            # Replace comma decimal separator with period
            dataframe[col] = (
                dataframe[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
    return dataframe


def _convert_european_thousands_and_decimals(
    dataframe: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Convert European format with period thousands separator and comma decimal.
    
    Handles:
    - "1.056,1" -> "1056.1" (period=thousands, comma=decimal)
    - "1.056.1" -> "1056.1" (multiple periods, last is decimal)
    - "123,45" -> "123.45" (just comma decimal)
    - "1234.5" -> "1234.5" (already correct)
    """
    for col in columns:
        if col in dataframe.columns:
            def convert_value(val) -> str:
                # Handle non-string or missing values
                if pd.isna(val):
                    return str(val)
                s = str(val).strip()
                if s in ("nan", "None", ""):
                    return s
                
                has_comma = "," in s
                period_count = s.count(".")
                
                if has_comma:
                    # European format: period=thousands, comma=decimal
                    # Remove all periods, replace comma with period
                    return s.replace(".", "").replace(",", ".")
                elif period_count > 1:
                    # Multiple periods: first ones are thousands separators
                    # Keep only the last period as decimal
                    parts = s.rsplit(".", 1)
                    return parts[0].replace(".", "") + "." + parts[1]
                else:
                    # Already fine (single period or no decimals)
                    return s
            
            dataframe[col] = dataframe[col].apply(convert_value)
    return dataframe


def _strip_thousands_separators(
    dataframe: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Remove thousands separators (commas and dots) so values can be coerced to numeric."""
    for col in columns:
        if col in dataframe.columns:
            # Remove commas and dots used as thousands separators
            dataframe[col] = dataframe[col].replace(r"[,.]", "", regex=True)
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
    
    # Columns with simple European decimal format (comma as decimal separator)
    simple_decimal_columns = [
        "Air Temperature",
        "Wet Bulb Temperature",
        "Humidity",
        "Rain Intensity",
        "Interval Rain",
        "Precipitation Type",
        "Wind Direction",
        "Wind Speed",
        "Maximum Wind Speed",
        "Barometric Pressure",
        "Heading",
        "Battery Life",
    ]
    dataframe = _convert_european_decimals(dataframe, simple_decimal_columns)
    
    # Columns with European thousands separator (period) AND decimal (comma or period)
    # e.g., "1.056,1" or "1.056.1" meaning 1056.1
    thousands_columns = ["Total Rain", "Solar Radiation"]
    dataframe = _convert_european_thousands_and_decimals(dataframe, thousands_columns)
    
    return WeatherStationsSchema.validate(dataframe)
