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

    TIME: Series[str] = pa.Field()
    SEGMENT_ID: Series[int] = pa.Field()
    SPEED: Series[float] = pa.Field(
        ge=-1,
    )
    STREET: Series[str] = pa.Field()
    DIRECTION: Series[str] = pa.Field()
    FROM_STREET: Series[str] = pa.Field()
    TO_STREET: Series[str] = pa.Field()
    LENGTH: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
    )
    STREET_HEADING: Series[str] = pa.Field()
    COMMENTS: Series[str] | None = pa.Field(
        nullable=True,
    )
    BUS_COUNT: Series[int] = pa.Field(
        ge=0,
    )
    MESSAGE_COUNT: Series[int] = pa.Field(
        ge=0,
    )
    HOUR: Series[int] = pa.Field(
        ge=0,
        le=23,
    )
    DAY_OF_WEEK: Series[int] = pa.Field(
        ge=1,
        le=7,
    )
    MONTH: Series[int] = pa.Field(
        ge=1,
        le=12,
    )
    RECORD_ID: Series[str] = pa.Field(
        unique=True,
    )
    START_LATITUDE: Series[float] | None = pa.Field(
        nullable=True,
    )
    START_LONGITUDE: Series[float] | None = pa.Field(
        nullable=True,
    )
    END_LATITUDE: Series[float] | None = pa.Field(
        nullable=True,
    )
    END_LONGITUDE: Series[float] | None = pa.Field(
        nullable=True,
    )
    START_LOCATION: Series[str] | None = pa.Field(
        nullable=True,
    )
    END_LOCATION: Series[str] | None = pa.Field(
        nullable=True,
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
