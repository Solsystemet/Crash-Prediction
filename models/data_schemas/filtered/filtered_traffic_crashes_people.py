# pyright: reportAny=false
"""Pandera schema for the Traffic Crashes - People CSV dataset.

Source: Chicago Data Portal - Traffic Crashes - People
File: data/traffic-crashes-people.csv

This dataset contains information about individuals involved in traffic crashes,
including drivers, passengers, and pedestrians. Each row represents one person
involved in a crash, and can be linked to crashes via CRASH_RECORD_ID and
vehicles via VEHICLE_ID.
"""

from typing import Literal
import pandera.pandas as pa
from pandera.typing import Series


class FilteredTrafficCrashesPeopleSchema(pa.DataFrameModel):
    """Schema for the Traffic Crashes - People dataset.

    Each row represents a single person (driver, passenger, or pedestrian)
    involved in a traffic crash. The CRASH_RECORD_ID can be used to link to
    the main Crashes dataset, and VEHICLE_ID to link to the Vehicles dataset.
    """

    # Identifiers
    CRASH_RECORD_ID: Series[str] = pa.Field()
    VEHICLE_ID: Series[float] = pa.Field(
        nullable=True,
    )
    PERSON_TYPE: Series[str] = pa.Field(
        isin=[
            "DRIVER",
            "PASSENGER",
            "PEDESTRIAN",
            "BICYCLE",
            "NON-MOTOR VEHICLE",
            "NON-CONTACT VEHICLE",
        ],
    )

    # Demographics
    SEX: Series[str] = pa.Field(
        nullable=True,
        isin=["M", "F", "X", "UNKNOWN"],
    )
    AGE: Series[float] = pa.Field(
        nullable=True,
    )

    # Safety and injury information
    SAFETY_EQUIPMENT: Series[str] = pa.Field(
        nullable=True,
    )
    INJURY_CLASSIFICATION: Series[str] = pa.Field(
        nullable=True,
        isin=[
            "SEVERE",
            "MINOR",
            "NO_INJURY",
        ],
    )

    # Alcohol-related information
    BAC_RESULT: Series[str] = pa.Field(
        nullable=True,
    )
    BAC_RESULT_VALUE: Series[float] = pa.Field(
        nullable=True,
        ge=0.0,
    )

    # Cell phone usage
    CELL_PHONE_USE: Series[str] = pa.Field(
        nullable=True,
        isin=["Y", "N"],
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
