# pyright: reportAny=false
"""Pandera schema for the Traffic Crashes CSV dataset.

Source: Chicago Data Portal - Traffic Crashes - Crashes
File: data/Traffic_Crashes_-_Crashes.csv
"""

from typing import Literal
import pandera.pandas as pa
from pandera.typing import Series


class FilteredTrafficCrashesSchema(pa.DataFrameModel):
    """Schema for the Traffic Crashes dataset.

    Each row represents a single traffic crash reported by the Chicago Police
    Department. The CRASH_RECORD_ID can be used to link to the Vehicles and
    People datasets.
    """

    CRASH_RECORD_ID: Series[str] = pa.Field(
        unique=True,
    )
    CRASH_DATE_EST_I: Series[str] = pa.Field(
        nullable=True,
    )
    CRASH_DATE: Series[str] = pa.Field()
    POSTED_SPEED_LIMIT: Series[int] = pa.Field(
        ge=0,
    )
    TRAFFIC_CONTROL_DEVICE: Series[str] = pa.Field()
    DEVICE_CONDITION: Series[str] = pa.Field()
    WEATHER_CONDITION: Series[str] = pa.Field()
    LIGHTING_CONDITION: Series[str] = pa.Field()
    LANE_CNT: Series[float] = pa.Field(
        nullable=True,
        ge=0,
    )
    ROADWAY_SURFACE_COND: Series[str] = pa.Field()
    ROAD_DEFECT: Series[str] = pa.Field()
    MOST_SEVERE_INJURY: Series[str] = pa.Field(
        nullable=True,
    )
    INJURIES_FATAL: Series[float] = pa.Field(
        nullable=True,
        ge=0,
    )
    INJURIES_INCAPACITATING: Series[float] = pa.Field(
        nullable=True,
        ge=0,
    )
    INJURIES_NON_INCAPACITATING: Series[float] = pa.Field(
        nullable=True,
        ge=0,
    )
    INJURIES_REPORTED_NOT_EVIDENT: Series[float] = pa.Field(
        nullable=True,
        ge=0,
    )
    INJURIES_NO_INDICATION: Series[float] = pa.Field(
        nullable=True,
        ge=0,
    )
    INJURIES_UNKNOWN: Series[float] = pa.Field(
        nullable=True,
        ge=0,
    )
    CRASH_HOUR: Series[int] = pa.Field(
        ge=0,
        le=23,
    )
    CRASH_DAY_OF_WEEK: Series[int] = pa.Field(
        ge=1,
        le=7,
    )
    CRASH_MONTH: Series[int] = pa.Field(
        ge=1,
        le=12,
    )
    LATITUDE: Series[float] = pa.Field(
        nullable=True,
    )
    LONGITUDE: Series[float] = pa.Field(
        nullable=True,
    )
    LOCATION: Series[str] = pa.Field(
        nullable=True,
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
