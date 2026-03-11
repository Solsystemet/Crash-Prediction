"""Unit tests for the TrafficCrashesSchema."""

from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from models.data_schemas.full.traffic_crashes import TrafficCrashesSchema

DATA_FILE = Path("data/traffic_crashes.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_row(**overrides: object) -> dict[str, object]:
    """Return a dict representing one valid row. Override any field via kwargs."""
    row: dict[str, object] = {
        "CRASH_RECORD_ID": "abc123",
        "CRASH_DATE_EST_I": None,
        "CRASH_DATE": "01/01/2025 12:00:00 AM",
        "POSTED_SPEED_LIMIT": 30,
        "TRAFFIC_CONTROL_DEVICE": "TRAFFIC SIGNAL",
        "DEVICE_CONDITION": "FUNCTIONING PROPERLY",
        "WEATHER_CONDITION": "CLEAR",
        "LIGHTING_CONDITION": "DAYLIGHT",
        "FIRST_CRASH_TYPE": "REAR END",
        "TRAFFICWAY_TYPE": "DIVIDED - W/MEDIAN (NOT RAISED)",
        "LANE_CNT": None,
        "ALIGNMENT": "STRAIGHT AND LEVEL",
        "ROADWAY_SURFACE_COND": "DRY",
        "ROAD_DEFECT": "NO DEFECTS",
        "REPORT_TYPE": "ON SCENE",
        "CRASH_TYPE": "NO INJURY / DRIVE AWAY",
        "INTERSECTION_RELATED_I": None,
        "NOT_RIGHT_OF_WAY_I": None,
        "HIT_AND_RUN_I": None,
        "DAMAGE": "OVER $1,500",
        "DATE_POLICE_NOTIFIED": "01/01/2025 12:05:00 AM",
        "PRIM_CONTRIBUTORY_CAUSE": "FOLLOWING TOO CLOSELY",
        "SEC_CONTRIBUTORY_CAUSE": "NOT APPLICABLE",
        "STREET_NO": 100,
        "STREET_DIRECTION": "N",
        "STREET_NAME": "STATE ST",
        "BEAT_OF_OCCURRENCE": None,
        "PHOTOS_TAKEN_I": None,
        "STATEMENTS_TAKEN_I": None,
        "DOORING_I": None,
        "WORK_ZONE_I": None,
        "WORK_ZONE_TYPE": None,
        "WORKERS_PRESENT_I": None,
        "NUM_UNITS": 2,
        "MOST_SEVERE_INJURY": None,
        "INJURIES_TOTAL": 0.0,
        "INJURIES_FATAL": 0.0,
        "INJURIES_INCAPACITATING": 0.0,
        "INJURIES_NON_INCAPACITATING": 0.0,
        "INJURIES_REPORTED_NOT_EVIDENT": 0.0,
        "INJURIES_NO_INDICATION": 0.0,
        "INJURIES_UNKNOWN": 0.0,
        "CRASH_HOUR": 0,
        "CRASH_DAY_OF_WEEK": 4,
        "CRASH_MONTH": 1,
        "LATITUDE": 41.8781,
        "LONGITUDE": -87.6298,
        "LOCATION": "POINT (-87.6298 41.8781)",
    }
    row.update(overrides)
    return row


def _df_from_row(**overrides: object) -> pd.DataFrame:
    return pd.DataFrame([_make_valid_row(**overrides)])


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestTrafficCrashesSchemaValid:
    """Tests that valid data passes schema validation."""

    def test_minimal_valid_row(self) -> None:
        df = _df_from_row()
        validated = TrafficCrashesSchema.validate(df)
        assert len(validated) == 1

    def test_nullable_fields_accept_none(self) -> None:
        """All Optional columns should accept NaN / None."""
        df = _df_from_row(
            CRASH_DATE_EST_I=None,
            LANE_CNT=None,
            REPORT_TYPE=None,
            INTERSECTION_RELATED_I=None,
            NOT_RIGHT_OF_WAY_I=None,
            HIT_AND_RUN_I=None,
            BEAT_OF_OCCURRENCE=None,
            PHOTOS_TAKEN_I=None,
            STATEMENTS_TAKEN_I=None,
            DOORING_I=None,
            WORK_ZONE_I=None,
            WORK_ZONE_TYPE=None,
            WORKERS_PRESENT_I=None,
            MOST_SEVERE_INJURY=None,
            INJURIES_TOTAL=None,
            INJURIES_FATAL=None,
            INJURIES_INCAPACITATING=None,
            INJURIES_NON_INCAPACITATING=None,
            INJURIES_REPORTED_NOT_EVIDENT=None,
            INJURIES_NO_INDICATION=None,
            INJURIES_UNKNOWN=None,
            LATITUDE=None,
            LONGITUDE=None,
            LOCATION=None,
        )
        validated = TrafficCrashesSchema.validate(df)
        assert len(validated) == 1

    def test_multiple_rows(self) -> None:
        rows = [
            _make_valid_row(CRASH_RECORD_ID="id_1", CRASH_HOUR=0),
            _make_valid_row(CRASH_RECORD_ID="id_2", CRASH_HOUR=23),
        ]
        df = pd.DataFrame(rows)
        validated = TrafficCrashesSchema.validate(df)
        assert len(validated) == 2

    def test_boundary_crash_hour(self) -> None:
        for hour in (0, 12, 23):
            df = _df_from_row(CRASH_HOUR=hour)
            validated = TrafficCrashesSchema.validate(df)
            assert len(validated) == 1

    def test_boundary_day_of_week(self) -> None:
        for dow in (1, 4, 7):
            df = _df_from_row(CRASH_DAY_OF_WEEK=dow)
            validated = TrafficCrashesSchema.validate(df)
            assert len(validated) == 1

    def test_boundary_crash_month(self) -> None:
        for month in (1, 6, 12):
            df = _df_from_row(CRASH_MONTH=month)
            validated = TrafficCrashesSchema.validate(df)
            assert len(validated) == 1

    @pytest.mark.skipif(not DATA_FILE.exists(), reason=f"{DATA_FILE} not found")
    def test_csv_sample(self) -> None:
        """Validate a sample from the real CSV file."""
        df = pd.read_csv(DATA_FILE, nrows=500)
        validated = TrafficCrashesSchema.validate(df)
        assert len(validated) == 500


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


class TestTrafficCrashesSchemaInvalid:
    """Tests that invalid data is rejected."""

    def test_duplicate_crash_record_id(self) -> None:
        rows = [
            _make_valid_row(CRASH_RECORD_ID="dup"),
            _make_valid_row(CRASH_RECORD_ID="dup"),
        ]
        df = pd.DataFrame(rows)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficCrashesSchema.validate(df)

    def test_negative_speed_limit(self) -> None:
        df = _df_from_row(POSTED_SPEED_LIMIT=-10)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficCrashesSchema.validate(df)

    def test_crash_hour_out_of_range(self) -> None:
        df = _df_from_row(CRASH_HOUR=25)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficCrashesSchema.validate(df)

    def test_day_of_week_out_of_range(self) -> None:
        df = _df_from_row(CRASH_DAY_OF_WEEK=0)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficCrashesSchema.validate(df)

    def test_crash_month_out_of_range(self) -> None:
        df = _df_from_row(CRASH_MONTH=13)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficCrashesSchema.validate(df)

    def test_num_units_zero(self) -> None:
        df = _df_from_row(NUM_UNITS=0)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficCrashesSchema.validate(df)

    def test_negative_injuries_total(self) -> None:
        df = _df_from_row(INJURIES_TOTAL=-1.0)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficCrashesSchema.validate(df)

    def test_negative_injuries_fatal(self) -> None:
        df = _df_from_row(INJURIES_FATAL=-1.0)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficCrashesSchema.validate(df)

    def test_negative_lane_cnt(self) -> None:
        df = _df_from_row(LANE_CNT=-1.0)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficCrashesSchema.validate(df)
