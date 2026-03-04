# pyright: reportAny=false
"""Pandera schema for the Traffic Crashes - Vehicles CSV dataset.

Source: Chicago Data Portal - Traffic Crashes - Vehicles
File: data/traffic_crashes_vehicles.csv

This dataset contains information about vehicles involved in traffic crashes.
Each row represents one vehicle involved in a crash and can be linked to crashes
via CRASH_RECORD_ID and people via VEHICLE_ID.
"""

from typing import Literal
import pandera.pandas as pa
from pandera.typing import Series


class TrafficCrashesVehiclesSchema(pa.DataFrameModel):
    """Schema for the Traffic Crashes - Vehicles dataset.

    Each row represents a single vehicle involved in a traffic crash. The
    CRASH_RECORD_ID can be used to link to the main Crashes dataset, and
    VEHICLE_ID to link to the People dataset.
    """

    # Identifiers
    CRASH_RECORD_ID: Series[str] = pa.Field()
    VEHICLE_ID: Series[float] | None = pa.Field(
        nullable=True,
    )

    # Vehicle identification
    MAKE: Series[str] | None = pa.Field(
        nullable=True,
    )
    MODEL: Series[str] | None = pa.Field(
        nullable=True,
    )
    VEHICLE_YEAR: Series[float] | None = pa.Field(
        nullable=True,
        ge=1900,
        le=9999,
    )
    VEHICLE_TYPE: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Movement and maneuvers
    EXCEED_SPEED_LIMIT_I: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Trailer information
    TOTAL_VEHICLE_LENGTH: Series[str] | None = pa.Field(
        nullable=True,
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
