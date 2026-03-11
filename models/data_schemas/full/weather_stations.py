# pyright: reportAny=false
"""Pandera schema for the Beach Weather Stations CSV dataset.

Source: Chicago Data Portal - Beach Weather Stations - Automated Sensors
File: data/Beach_Weather_Stations_-_Automated_Sensors.csv
"""

from typing import Literal
import pandera.pandas as pa
from pandera.typing import Series


class WeatherStationsSchema(pa.DataFrameModel):
    """Schema for the Beach Weather Stations - Automated Sensors dataset.

    Each row represents a single measurement from an automated weather sensor
    at a Chicago beach location.
    """

    Station_Name: Series[str] = pa.Field(
        alias="Station Name",
    )
    Measurement_Timestamp: Series[str] = pa.Field(
        alias="Measurement Timestamp",
    )
    Air_Temperature: Series[float] = pa.Field(
        alias="Air Temperature",
        nullable=True,
    )
    Wet_Bulb_Temperature: Series[float] = pa.Field(
        alias="Wet Bulb Temperature",
        nullable=True,
    )
    Humidity: Series[float] = pa.Field(
        alias="Humidity",
        nullable=True,
        ge=0,
        le=100,
    )
    Rain_Intensity: Series[float] = pa.Field(
        alias="Rain Intensity",
        nullable=True,
        ge=0,
    )
    Interval_Rain: Series[float] = pa.Field(
        alias="Interval Rain",
        nullable=True,
    )
    Total_Rain: Series[float] = pa.Field(
        alias="Total Rain",
        nullable=True,
        ge=0,
    )
    Precipitation_Type: Series[float] = pa.Field(
        alias="Precipitation Type",
        nullable=True,
        isin=[0, 5, 40, 60, 70],
    )
    Wind_Direction: Series[float] = pa.Field(
        alias="Wind Direction",
        nullable=True,
        ge=0,
        le=360,
    )
    Wind_Speed: Series[float] = pa.Field(
        alias="Wind Speed",
        nullable=True,
        ge=0,
    )
    Maximum_Wind_Speed: Series[float] = pa.Field(
        alias="Maximum Wind Speed",
        nullable=True,
        ge=0,
    )
    Barometric_Pressure: Series[float] = pa.Field(
        alias="Barometric Pressure",
        nullable=True,
    )
    Solar_Radiation: Series[float] = pa.Field(
        alias="Solar Radiation",
        nullable=True,
    )
    Heading: Series[float] = pa.Field(
        alias="Heading",
        nullable=True,
        ge=0,
        le=360,
    )
    Battery_Life: Series[float] = pa.Field(
        alias="Battery Life",
        nullable=True,
    )
    Measurement_Timestamp_Label: Series[str] = pa.Field(
        alias="Measurement Timestamp Label",
    )
    Measurement_ID: Series[str] = pa.Field(
        alias="Measurement ID",
        unique=True,
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
