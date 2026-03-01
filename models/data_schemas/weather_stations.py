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
        description="Name of the weather station.",
    )
    Measurement_Timestamp: Series[str] = pa.Field(
        alias="Measurement Timestamp",
        description="The date and time when the measurements were taken.",
    )
    Air_Temperature: Series[float] | None = pa.Field(
        alias="Air Temperature",
        nullable=True,
        description="Air Temperature in Celsius degrees.",
    )
    Wet_Bulb_Temperature: Series[float] | None = pa.Field(
        alias="Wet Bulb Temperature",
        nullable=True,
        description="Wet bulb temperature in Celsius degrees.",
    )
    Humidity: Series[float] | None = pa.Field(
        alias="Humidity",
        nullable=True,
        ge=0,
        le=100,
        description="Percent relative humidity.",
    )
    Rain_Intensity: Series[float] | None = pa.Field(
        alias="Rain Intensity",
        nullable=True,
        ge=0,
        description="Rain intensity in mm per hour.",
    )
    Interval_Rain: Series[float] | None = pa.Field(
        alias="Interval Rain",
        nullable=True,
        description="Rain since the last hourly measurement, in mm.",
    )
    Total_Rain: Series[float] | None = pa.Field(
        alias="Total Rain",
        nullable=True,
        ge=0,
        description="Total rain since midnight in mm.",
    )
    Precipitation_Type: Series[float] | None = pa.Field(
        alias="Precipitation Type",
        nullable=True,
        isin=[0, 5, 40, 60, 70],
        description=(
            "Type of precipitation: 0 = No precipitation, "
            "60 = Liquid precipitation (e.g. rain - ice, hail and sleet are "
            "transmitted as rain), 70 = Solid precipitation (e.g. snow), "
            "40 = Unspecified precipitation."
        ),
    )
    Wind_Direction: Series[float] | None = pa.Field(
        alias="Wind Direction",
        nullable=True,
        ge=0,
        le=360,
        description="Wind direction in degrees.",
    )
    Wind_Speed: Series[float] | None = pa.Field(
        alias="Wind Speed",
        nullable=True,
        ge=0,
        description=(
            "Wind speed in meters per second at the time the record was reported."
        ),
    )
    Maximum_Wind_Speed: Series[float] | None = pa.Field(
        alias="Maximum Wind Speed",
        nullable=True,
        ge=0,
        description=(
            "Maximum wind speed in the two-minute period immediately "
            "preceding the time the record was reported."
        ),
    )
    Barometric_Pressure: Series[float] | None = pa.Field(
        alias="Barometric Pressure",
        nullable=True,
        description="Barometric pressure in hPa.",
    )
    Solar_Radiation: Series[float] | None = pa.Field(
        alias="Solar Radiation",
        nullable=True,
        description="Solar radiation in watts per square meter.",
    )
    Heading: Series[float] | None = pa.Field(
        alias="Heading",
        nullable=True,
        ge=0,
        le=360,
        description=(
            "The current heading of the wind-measurement unit. The ideal "
            "value to get the most accurate measurements is true north "
            "(0 degrees) and the unit is manually adjusted, as necessary, "
            "to keep it close to this heading."
        ),
    )
    Battery_Life: Series[float] | None = pa.Field(
        alias="Battery Life",
        nullable=True,
        description=(
            "Battery voltage, an indicator of remaining battery life used by "
            "the Chicago Park District to know when batteries should be "
            "replaced."
        ),
    )
    Measurement_Timestamp_Label: Series[str] = pa.Field(
        alias="Measurement Timestamp Label",
        description=(
            "The Last Updated value in text format, suitable for use in the "
            "Visualize function."
        ),
    )
    Measurement_ID: Series[str] = pa.Field(
        alias="Measurement ID",
        unique=True,
        description=(
            "A unique record ID made up of the Station Name and Measurement Timestamp."
        ),
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
