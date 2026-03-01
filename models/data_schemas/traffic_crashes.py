# pyright: reportAny=false
"""Pandera schema for the Traffic Crashes CSV dataset.

Source: Chicago Data Portal - Traffic Crashes - Crashes
File: data/Traffic_Crashes_-_Crashes.csv
"""

from typing import Literal
import pandera.pandas as pa
from pandera.typing import Series


class TrafficCrashesSchema(pa.DataFrameModel):
    """Schema for the Traffic Crashes dataset.

    Each row represents a single traffic crash reported by the Chicago Police
    Department. The CRASH_RECORD_ID can be used to link to the Vehicles and
    People datasets.
    """

    CRASH_RECORD_ID: Series[str] = pa.Field(
        description=(
            "Unique identifier for each crash record. Can be used to link to "
            "the same crash in the Vehicles and People datasets."
        ),
        unique=True,
    )
    CRASH_DATE_EST_I: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "Crash date estimated by desk officer or reporting party (only "
            "used in cases where crash is reported at police station days "
            "after the crash)."
        ),
    )
    CRASH_DATE: Series[str] = pa.Field(
        description="Date and time of crash as entered by the reporting officer.",
    )
    POSTED_SPEED_LIMIT: Series[int] = pa.Field(
        ge=0,
        description="Posted speed limit, as determined by reporting officer.",
    )
    TRAFFIC_CONTROL_DEVICE: Series[str] = pa.Field(
        description=(
            "Traffic control device present at crash location, as determined "
            "by reporting officer."
        ),
    )
    DEVICE_CONDITION: Series[str] = pa.Field(
        description=(
            "Condition of traffic control device, as determined by reporting officer."
        ),
    )
    WEATHER_CONDITION: Series[str] = pa.Field(
        description=(
            "Weather condition at time of crash, as determined by reporting officer."
        ),
    )
    LIGHTING_CONDITION: Series[str] = pa.Field(
        description=(
            "Light condition at time of crash, as determined by reporting officer."
        ),
    )
    FIRST_CRASH_TYPE: Series[str] = pa.Field(
        description="Type of first collision in crash.",
    )
    TRAFFICWAY_TYPE: Series[str] = pa.Field(
        description=("Trafficway type, as determined by reporting officer."),
    )
    LANE_CNT: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
        description=(
            "Total number of through lanes in either direction, excluding "
            "turn lanes, as determined by reporting officer (0 = intersection)."
        ),
    )
    ALIGNMENT: Series[str] = pa.Field(
        description=(
            "Street alignment at crash location, as determined by reporting officer."
        ),
    )
    ROADWAY_SURFACE_COND: Series[str] = pa.Field(
        description=("Road surface condition, as determined by reporting officer."),
    )
    ROAD_DEFECT: Series[str] = pa.Field(
        description="Road defects, as determined by reporting officer.",
    )
    REPORT_TYPE: Series[str] | None = pa.Field(
        nullable=True,
        description=("Administrative report type (at scene, at desk, amended)."),
    )
    CRASH_TYPE: Series[str] = pa.Field(
        description=(
            "A general severity classification for the crash. Can be either "
            "'Injury and/or Tow Due to Crash' or 'No Injury / Drive Away'."
        ),
    )
    INTERSECTION_RELATED_I: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "A field observation by the police officer whether an intersection "
            "played a role in the crash. Does not represent whether or not the "
            "crash occurred within the intersection."
        ),
    )
    NOT_RIGHT_OF_WAY_I: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "Whether the crash begun or first contact was made outside of the "
            "public right-of-way."
        ),
    )
    HIT_AND_RUN_I: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "Crash did/did not involve a driver who caused the crash and fled "
            "the scene without exchanging information and/or rendering aid."
        ),
    )
    DAMAGE: Series[str] = pa.Field(
        description="A field observation of estimated damage.",
    )
    DATE_POLICE_NOTIFIED: Series[str] = pa.Field(
        description=("Calendar date on which police were notified of the crash."),
    )
    PRIM_CONTRIBUTORY_CAUSE: Series[str] = pa.Field(
        description=(
            "The factor which was most significant in causing the crash, as "
            "determined by officer judgment."
        ),
    )
    SEC_CONTRIBUTORY_CAUSE: Series[str] = pa.Field(
        description=(
            "The factor which was second most significant in causing the "
            "crash, as determined by officer judgment."
        ),
    )
    STREET_NO: Series[int] = pa.Field(
        ge=0,
        description=(
            "Street address number of crash location, as determined by "
            "reporting officer."
        ),
    )
    STREET_DIRECTION: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "Street address direction (N,E,S,W) of crash location, as "
            "determined by reporting officer."
        ),
    )
    STREET_NAME: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "Street address name of crash location, as determined by reporting officer."
        ),
    )
    BEAT_OF_OCCURRENCE: Series[float] | None = pa.Field(
        nullable=True,
        description=(
            "Chicago Police Department Beat ID. Boundaries available at "
            "https://data.cityofchicago.org/d/aerh-rz74"
        ),
    )
    PHOTOS_TAKEN_I: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "Whether the Chicago Police Department took photos at the "
            "location of the crash."
        ),
    )
    STATEMENTS_TAKEN_I: Series[str] | None = pa.Field(
        nullable=True,
        description=("Whether statements were taken from unit(s) involved in crash."),
    )
    DOORING_I: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "Whether crash involved a motor vehicle occupant opening a door "
            "into the travel path of a bicyclist, causing a crash."
        ),
    )
    WORK_ZONE_I: Series[str] | None = pa.Field(
        nullable=True,
        description=("Whether the crash occurred in an active work zone."),
    )
    WORK_ZONE_TYPE: Series[str] | None = pa.Field(
        nullable=True,
        description="The type of work zone, if any.",
    )
    WORKERS_PRESENT_I: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "Whether construction workers were present in an active work zone "
            "at crash location."
        ),
    )
    NUM_UNITS: Series[int] = pa.Field(
        ge=1,
        description=(
            "Number of units involved in the crash. A unit can be a motor "
            "vehicle, a pedestrian, a bicyclist, or another non-passenger "
            "roadway user. Each unit represents a mode of traffic with an "
            "independent trajectory."
        ),
    )
    MOST_SEVERE_INJURY: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "Most severe injury sustained by any person involved in the crash."
        ),
    )
    INJURIES_TOTAL: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
        description=(
            "Total persons sustaining fatal, incapacitating, "
            "non-incapacitating, and possible injuries as determined by the "
            "reporting officer."
        ),
    )
    INJURIES_FATAL: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
        description="Total persons sustaining fatal injuries in the crash.",
    )
    INJURIES_INCAPACITATING: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
        description=(
            "Total persons sustaining incapacitating/serious injuries in the "
            "crash as determined by the reporting officer. Any injury other "
            "than fatal injury, which prevents the injured person from "
            "walking, driving, or normally continuing the activities they were "
            "capable of performing before the injury occurred. Includes severe "
            "lacerations, broken limbs, skull or chest injuries, and "
            "abdominal injuries."
        ),
    )
    INJURIES_NON_INCAPACITATING: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
        description=(
            "Total persons sustaining non-incapacitating injuries in the "
            "crash as determined by the reporting officer. Any injury, other "
            "than fatal or incapacitating injury, which is evident to "
            "observers at the scene of the crash. Includes lump on head, "
            "abrasions, bruises, and minor lacerations."
        ),
    )
    INJURIES_REPORTED_NOT_EVIDENT: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
        description=(
            "Total persons sustaining possible injuries in the crash as "
            "determined by the reporting officer. Includes momentary "
            "unconsciousness, claims of injuries not evident, limping, "
            "complaint of pain, nausea, and hysteria."
        ),
    )
    INJURIES_NO_INDICATION: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
        description=(
            "Total persons sustaining no injuries in the crash as determined "
            "by the reporting officer."
        ),
    )
    INJURIES_UNKNOWN: Series[float] | None = pa.Field(
        nullable=True,
        ge=0,
        description="Total persons for whom injuries sustained, if any, are unknown.",
    )
    CRASH_HOUR: Series[int] = pa.Field(
        ge=0,
        le=23,
        description="The hour of the day component of CRASH_DATE.",
    )
    CRASH_DAY_OF_WEEK: Series[int] = pa.Field(
        ge=1,
        le=7,
        description=("The day of the week component of CRASH_DATE. Sunday=1."),
    )
    CRASH_MONTH: Series[int] = pa.Field(
        ge=1,
        le=12,
        description="The month component of CRASH_DATE.",
    )
    LATITUDE: Series[float] | None = pa.Field(
        nullable=True,
        description=(
            "The latitude of the crash location, as determined by reporting "
            "officer, as derived from the reported address of crash."
        ),
    )
    LONGITUDE: Series[float] | None = pa.Field(
        nullable=True,
        description=(
            "The longitude of the crash location, as determined by reporting "
            "officer, as derived from the reported address of crash."
        ),
    )
    LOCATION: Series[str] | None = pa.Field(
        nullable=True,
        description=(
            "The crash location, as determined by reporting officer, as "
            "derived from the reported address of crash, in a column type "
            "that allows for mapping and other geographic analysis in the "
            "data portal software."
        ),
    )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = False
        coerce: bool = True
