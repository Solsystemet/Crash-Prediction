# pyright: reportAny=false
"""Pandera schema for the Beach Weather Stations CSV dataset.

Source: Chicago Data Portal - Beach Weather Stations - Automated Sensors
File: data/Beach_Weather_Stations_-_Automated_Sensors.csv
"""

from typing import Literal
import pandera.pandas as pa
from pandera.typing import Series


class FilteredWeatherStationsSchema(pa.DataFrameModel):
    """Schema for the Beach Weather Stations - Automated Sensors dataset.

    Each row represents a single measurement from an automated weather sensor
    at a Chicago beach location.
    """

    Measurement_Timestamp: Series[str] = pa.Field(
        alias="Measurement Timestamp",
    )
    Air_Temperature: Series[float] | None = pa.Field(
        alias="Air Temperature",
        nullable=True,
    )
    Humidity: Series[float] | None = pa.Field(
        alias="Humidity",
        nullable=True,
        ge=0,
        le=100,
    )
    Rain_Intensity: Series[float] | None = pa.Field(
        alias="Rain Intensity",
        nullable=True,
        ge=0,
    )
    Interval_Rain: Series[float] | None = pa.Field(
        alias="Interval Rain",
        nullable=True,
    )
    Total_Rain: Series[float] | None = pa.Field(
        alias="Total Rain",
        nullable=True,
        ge=0,
    )
    Precipitation_Type: Series[float] | None = pa.Field(
        alias="Precipitation Type",
        nullable=True,
        isin=[0, 5, 40, 60, 70],
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
