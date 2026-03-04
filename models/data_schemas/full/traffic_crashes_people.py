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


class TrafficCrashesPeopleSchema(pa.DataFrameModel):
    """Schema for the Traffic Crashes - People dataset.

    Each row represents a single person (driver, passenger, or pedestrian)
    involved in a traffic crash. The CRASH_RECORD_ID can be used to link to
    the main Crashes dataset, and VEHICLE_ID to link to the Vehicles dataset.
    """

    # Identifiers
    PERSON_ID: Series[str] = pa.Field(
        unique=True,
    )
    CRASH_RECORD_ID: Series[str] = pa.Field()
    VEHICLE_ID: Series[float] | None = pa.Field(
        nullable=True,
    )
    PERSON_TYPE: Series[str] = pa.Field(
        isin=["DRIVER", "PASSENGER", "PEDESTRIAN", "BICYCLE", "NON-MOTOR VEHICLE", "NON-CONTACT VEHICLE"],
    )

    # Temporal information
    CRASH_DATE: Series[str] = pa.Field()

    # Location information
    SEAT_NO: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
    )
    CITY: Series[str] | None = pa.Field(
        nullable=True,
    )
    STATE: Series[str] | None = pa.Field(
        nullable=True,
    )
    ZIPCODE: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Demographics
    SEX: Series[str] | None = pa.Field(
        nullable=True,
        isin=["M", "F", "X", "UNKNOWN"],
    )
    AGE: Series[float] | None = pa.Field(
        nullable=True,
    )

    # Driver license information
    DRIVERS_LICENSE_STATE: Series[str] | None = pa.Field(
        nullable=True,
    )
    DRIVERS_LICENSE_CLASS: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Safety and injury information
    SAFETY_EQUIPMENT: Series[str] | None = pa.Field(
        nullable=True,
    )
    AIRBAG_DEPLOYED: Series[str] | None = pa.Field(
        nullable=True,
    )
    EJECTION: Series[str] | None = pa.Field(
        nullable=True,
    )
    INJURY_CLASSIFICATION: Series[str] | None = pa.Field(
        nullable=True,
        isin=[
            "NO INDICATION OF INJURY",
            "NONINCAPACITATING INJURY",
            "REPORTED, NOT EVIDENT",
            "INCAPACITATING INJURY",
            "FATAL",
            "UNKNOWN",
        ],
    )
    HOSPITAL: Series[str] | None = pa.Field(
        nullable=True,
    )
    EMS_AGENCY: Series[str] | None = pa.Field(
        nullable=True,
    )
    EMS_RUN_NO: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Driver behavior and conditions
    DRIVER_ACTION: Series[str] | None = pa.Field(
        nullable=True,
    )
    DRIVER_VISION: Series[str] | None = pa.Field(
        nullable=True,
    )
    PHYSICAL_CONDITION: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Pedestrian information
    PEDPEDAL_ACTION: Series[str] | None = pa.Field(
        nullable=True,
    )
    PEDPEDAL_VISIBILITY: Series[str] | None = pa.Field(
        nullable=True,
    )
    PEDPEDAL_LOCATION: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Alcohol-related information
    BAC_RESULT: Series[str] | None = pa.Field(
        nullable=True,
    )
    BAC_RESULT_VALUE: Series[float] | None = pa.Field(
        nullable=True,
        ge=0.0,
    )

    # Cell phone usage
    CELL_PHONE_USE: Series[str] | None = pa.Field(
        nullable=True,
        isin=["Y", "N"],
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
