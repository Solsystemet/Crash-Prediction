# pyright: reportAny=false
"""Pandera schema for the Chicago Traffic Tracker CSV dataset.

Source: Chicago Data Portal - Chicago Traffic Tracker - Historical Congestion
        Estimates by Segment
File: data/Chicago_Traffic_Tracker_-_Historical_Congestion_Estimates_by_Segment_-_2024-Current.csv
"""

from typing import Literal
import pandera.pandas as pa
from pandera.typing import Series


class TrafficTrackerSchema(pa.DataFrameModel):
    """Schema for the Chicago Traffic Tracker - Historical Congestion Estimates.

    Each row represents congestion estimates for a particular traffic segment
    at a given time. Estimates are derived from GPS data from CTA buses.
    """

    TIME: Series[str] = pa.Field(
        description="Timestamp of the congestion estimate.",
    )
    SEGMENT_ID: Series[int] = pa.Field(
        description="Unique arbitrary number to represent each segment.",
    )
    SPEED: Series[float] = pa.Field(
        ge=-1,
        description=(
            "Estimated traffic speed in miles per hour. A value of -1 means "
            "no estimate is available."
        ),
    )
    STREET: Series[str] = pa.Field(
        description="Street name of the traffic segment.",
    )
    DIRECTION: Series[str] = pa.Field(
        description="Traffic flow direction for the segment.",
    )
    FROM_STREET: Series[str] = pa.Field(
        description=("Start street for the segment in the direction of traffic flow."),
    )
    TO_STREET: Series[str] = pa.Field(
        description=("End street for the segment in the direction of traffic flow."),
    )
    LENGTH: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
        description="Length of the segment in miles.",
    )
    STREET_HEADING: Series[str] = pa.Field(
        description=(
            "The position of the segment in the address grid. North, South, "
            "East, or West of State and Madison."
        ),
    )
    COMMENTS: Series[str] | None = pa.Field(
        nullable=True,
        description="Additional comments about the segment.",
    )
    BUS_COUNT: Series[int] = pa.Field(
        ge=0,
        description=(
            "Number of buses providing a GPS feed used to estimate congestion."
        ),
    )
    MESSAGE_COUNT: Series[int] = pa.Field(
        ge=0,
        description=(
            "Number of GPS probes received (or used) for estimating the speed "
            "for that segment."
        ),
    )
    HOUR: Series[int] = pa.Field(
        ge=0,
        le=23,
        description="Hour of the day.",
    )
    DAY_OF_WEEK: Series[int] = pa.Field(
        ge=1,
        le=7,
        description="Day of the week. Sunday = 1.",
    )
    MONTH: Series[int] = pa.Field(
        ge=1,
        le=12,
        description="Month of the year.",
    )
    RECORD_ID: Series[str] = pa.Field(
        unique=True,
        description="A unique identifier for each record in the dataset.",
    )
    START_LATITUDE: Series[float] | None = pa.Field(
        nullable=True,
        description="Latitude of the start of the segment.",
    )
    START_LONGITUDE: Series[float] | None = pa.Field(
        nullable=True,
        description="Longitude of the start of the segment.",
    )
    END_LATITUDE: Series[float] | None = pa.Field(
        nullable=True,
        description="Latitude of the end of the segment.",
    )
    END_LONGITUDE: Series[float] | None = pa.Field(
        nullable=True,
        description="Longitude of the end of the segment.",
    )
    START_LOCATION: Series[str] | None = pa.Field(
        nullable=True,
        description="Location of the start of the segment (POINT geometry).",
    )
    END_LOCATION: Series[str] | None = pa.Field(
        nullable=True,
        description="Location of the end of the segment (POINT geometry).",
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
