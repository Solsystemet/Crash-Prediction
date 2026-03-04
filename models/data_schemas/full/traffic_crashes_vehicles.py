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
    CRASH_UNIT_ID: Series[int] = pa.Field(
        unique=True,
        ge=0,
    )
    CRASH_RECORD_ID: Series[str] = pa.Field()
    VEHICLE_ID: Series[float] | None = pa.Field(
        nullable=True,
    )

    # Temporal information
    CRASH_DATE: Series[str] = pa.Field()

    # Unit information
    UNIT_NO: Series[str] | None = pa.Field(
        nullable=True,
    )
    UNIT_TYPE: Series[str] | None = pa.Field(
        nullable=True,
    )
    NUM_PASSENGERS: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
    )

    # Vehicle identification
    CMRC_VEH_I: Series[str] | None = pa.Field(
        nullable=True,
    )
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
    LIC_PLATE_STATE: Series[str] | None = pa.Field(
        nullable=True,
    )
    VEHICLE_TYPE: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Vehicle condition and damage
    VEHICLE_DEFECT: Series[str] | None = pa.Field(
        nullable=True,
    )
    VEHICLE_USE: Series[str] | None = pa.Field(
        nullable=True,
    )
    OCCUPANT_CNT: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
    )

    # Movement and maneuvers
    TRAVEL_DIRECTION: Series[str] | None = pa.Field(
        nullable=True,
    )
    MANEUVER: Series[str] | None = pa.Field(
        nullable=True,
    )
    EXCEED_SPEED_LIMIT_I: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Post-crash information
    TOWED_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    TOWED_BY: Series[str] | None = pa.Field(
        nullable=True,
    )
    TOWED_TO: Series[str] | None = pa.Field(
        nullable=True,
    )
    FIRE_I: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Crash area indicators (one-hot encoded for each police district/area)
    AREA_00_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_01_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_02_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_03_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_04_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_05_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_06_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_07_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_08_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_09_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_10_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_11_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_12_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    AREA_99_I: Series[str] | None = pa.Field(
        nullable=True,
    )

    # First contact point
    FIRST_CONTACT_POINT: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Commercial vehicle information (CMV)
    CMV_ID: Series[str] | None = pa.Field(
        nullable=True,
    )
    USDOT_NO: Series[str] | None = pa.Field(
        nullable=True,
    )
    CCMC_NO: Series[str] | None = pa.Field(
        nullable=True,
    )
    ILCC_NO: Series[str] | None = pa.Field(
        nullable=True,
    )
    COMMERCIAL_SRC: Series[str] | None = pa.Field(
        nullable=True,
    )
    GVWR: Series[str] | None = pa.Field(
        nullable=True,
    )
    CARRIER_NAME: Series[str] | None = pa.Field(
        nullable=True,
    )
    CARRIER_STATE: Series[str] | None = pa.Field(
        nullable=True,
    )
    CARRIER_CITY: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Hazardous materials information
    HAZMAT_PLACARDS_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    HAZMAT_NAME: Series[str] | None = pa.Field(
        nullable=True,
    )
    UN_NO: Series[str] | None = pa.Field(
        nullable=True,
    )
    HAZMAT_PRESENT_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    HAZMAT_REPORT_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    HAZMAT_REPORT_NO: Series[str] | None = pa.Field(
        nullable=True,
    )
    HAZMAT_CLASS: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Violations related to commercial vehicles
    MCS_REPORT_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    MCS_REPORT_NO: Series[str] | None = pa.Field(
        nullable=True,
    )
    HAZMAT_VIO_CAUSE_CRASH_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    MCS_VIO_CAUSE_CRASH_I: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Permits and special vehicle characteristics
    IDOT_PERMIT_NO: Series[str] | None = pa.Field(
        nullable=True,
    )
    WIDE_LOAD_I: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Trailer information
    TRAILER1_WIDTH: Series[str] | None = pa.Field(
        nullable=True,
    )
    TRAILER2_WIDTH: Series[str] | None = pa.Field(
        nullable=True,
    )
    TRAILER1_LENGTH: Series[str] | None = pa.Field(
        nullable=True,
    )
    TRAILER2_LENGTH: Series[str] | None = pa.Field(
        nullable=True,
    )
    TOTAL_VEHICLE_LENGTH: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Axle and configuration information
    AXLE_CNT: Series[str] | None = pa.Field(
        nullable=True,
    )
    VEHICLE_CONFIG: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Cargo information
    CARGO_BODY_TYPE: Series[str] | None = pa.Field(
        nullable=True,
    )
    LOAD_TYPE: Series[str] | None = pa.Field(
        nullable=True,
    )

    # Out of service indicators
    HAZMAT_OUT_OF_SERVICE_I: Series[str] | None = pa.Field(
        nullable=True,
    )
    MCS_OUT_OF_SERVICE_I: Series[str] | None = pa.Field(
        nullable=True,
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
