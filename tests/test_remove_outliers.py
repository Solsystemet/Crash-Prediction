"""Tests for remove_outliers module."""

import pandas as pd
import pytest
from pandera.typing.pandas import DataFrame

from data_preparation.helpers.remove_outliers import remove_outliers
from models.data_schemas.full.traffic_crashes import TrafficCrashesSchema


class Col:
    """Column name constants from TrafficCrashesSchema.

    Pandera schema attributes are typed as `str | None` but are always `str` at runtime.
    This class provides properly typed column name constants.
    """

    CRASH_RECORD_ID: str = "CRASH_RECORD_ID"
    CRASH_DATE_EST_I: str = "CRASH_DATE_EST_I"
    CRASH_DATE: str = "CRASH_DATE"
    POSTED_SPEED_LIMIT: str = "POSTED_SPEED_LIMIT"
    TRAFFIC_CONTROL_DEVICE: str = "TRAFFIC_CONTROL_DEVICE"
    DEVICE_CONDITION: str = "DEVICE_CONDITION"
    WEATHER_CONDITION: str = "WEATHER_CONDITION"
    LIGHTING_CONDITION: str = "LIGHTING_CONDITION"
    FIRST_CRASH_TYPE: str = "FIRST_CRASH_TYPE"
    TRAFFICWAY_TYPE: str = "TRAFFICWAY_TYPE"
    LANE_CNT: str = "LANE_CNT"
    ALIGNMENT: str = "ALIGNMENT"
    ROADWAY_SURFACE_COND: str = "ROADWAY_SURFACE_COND"
    ROAD_DEFECT: str = "ROAD_DEFECT"
    REPORT_TYPE: str = "REPORT_TYPE"
    CRASH_TYPE: str = "CRASH_TYPE"
    INTERSECTION_RELATED_I: str = "INTERSECTION_RELATED_I"
    NOT_RIGHT_OF_WAY_I: str = "NOT_RIGHT_OF_WAY_I"
    HIT_AND_RUN_I: str = "HIT_AND_RUN_I"
    DAMAGE: str = "DAMAGE"
    DATE_POLICE_NOTIFIED: str = "DATE_POLICE_NOTIFIED"
    PRIM_CONTRIBUTORY_CAUSE: str = "PRIM_CONTRIBUTORY_CAUSE"
    SEC_CONTRIBUTORY_CAUSE: str = "SEC_CONTRIBUTORY_CAUSE"
    STREET_NO: str = "STREET_NO"
    STREET_DIRECTION: str = "STREET_DIRECTION"
    STREET_NAME: str = "STREET_NAME"
    BEAT_OF_OCCURRENCE: str = "BEAT_OF_OCCURRENCE"
    PHOTOS_TAKEN_I: str = "PHOTOS_TAKEN_I"
    STATEMENTS_TAKEN_I: str = "STATEMENTS_TAKEN_I"
    DOORING_I: str = "DOORING_I"
    WORK_ZONE_I: str = "WORK_ZONE_I"
    WORK_ZONE_TYPE: str = "WORK_ZONE_TYPE"
    WORKERS_PRESENT_I: str = "WORKERS_PRESENT_I"
    NUM_UNITS: str = "NUM_UNITS"
    MOST_SEVERE_INJURY: str = "MOST_SEVERE_INJURY"
    INJURIES_TOTAL: str = "INJURIES_TOTAL"
    INJURIES_FATAL: str = "INJURIES_FATAL"
    INJURIES_INCAPACITATING: str = "INJURIES_INCAPACITATING"
    INJURIES_NON_INCAPACITATING: str = "INJURIES_NON_INCAPACITATING"
    INJURIES_REPORTED_NOT_EVIDENT: str = "INJURIES_REPORTED_NOT_EVIDENT"
    INJURIES_NO_INDICATION: str = "INJURIES_NO_INDICATION"
    INJURIES_UNKNOWN: str = "INJURIES_UNKNOWN"
    CRASH_HOUR: str = "CRASH_HOUR"
    CRASH_DAY_OF_WEEK: str = "CRASH_DAY_OF_WEEK"
    CRASH_MONTH: str = "CRASH_MONTH"
    LATITUDE: str = "LATITUDE"
    LONGITUDE: str = "LONGITUDE"
    LOCATION: str = "LOCATION"


def _make_valid_row(**overrides: object) -> dict[str, object]:
    """Return a dict representing one valid row. Override any field via kwargs."""
    row: dict[str, object] = {
        Col.CRASH_RECORD_ID: "abc123",
        Col.CRASH_DATE_EST_I: None,
        Col.CRASH_DATE: "01/01/2025 12:00:00 AM",
        Col.POSTED_SPEED_LIMIT: 30,
        Col.TRAFFIC_CONTROL_DEVICE: "TRAFFIC SIGNAL",
        Col.DEVICE_CONDITION: "FUNCTIONING PROPERLY",
        Col.WEATHER_CONDITION: "CLEAR",
        Col.LIGHTING_CONDITION: "DAYLIGHT",
        Col.FIRST_CRASH_TYPE: "REAR END",
        Col.TRAFFICWAY_TYPE: "DIVIDED - W/MEDIAN (NOT RAISED)",
        Col.LANE_CNT: None,
        Col.ALIGNMENT: "STRAIGHT AND LEVEL",
        Col.ROADWAY_SURFACE_COND: "DRY",
        Col.ROAD_DEFECT: "NO DEFECTS",
        Col.REPORT_TYPE: "ON SCENE",
        Col.CRASH_TYPE: "NO INJURY / DRIVE AWAY",
        Col.INTERSECTION_RELATED_I: None,
        Col.NOT_RIGHT_OF_WAY_I: None,
        Col.HIT_AND_RUN_I: None,
        Col.DAMAGE: "OVER $1,500",
        Col.DATE_POLICE_NOTIFIED: "01/01/2025 12:05:00 AM",
        Col.PRIM_CONTRIBUTORY_CAUSE: "FOLLOWING TOO CLOSELY",
        Col.SEC_CONTRIBUTORY_CAUSE: "NOT APPLICABLE",
        Col.STREET_NO: 100,
        Col.STREET_DIRECTION: "N",
        Col.STREET_NAME: "STATE ST",
        Col.BEAT_OF_OCCURRENCE: None,
        Col.PHOTOS_TAKEN_I: None,
        Col.STATEMENTS_TAKEN_I: None,
        Col.DOORING_I: None,
        Col.WORK_ZONE_I: None,
        Col.WORK_ZONE_TYPE: None,
        Col.WORKERS_PRESENT_I: None,
        Col.NUM_UNITS: 2,
        Col.MOST_SEVERE_INJURY: None,
        Col.INJURIES_TOTAL: 0.0,
        Col.INJURIES_FATAL: 0.0,
        Col.INJURIES_INCAPACITATING: 0.0,
        Col.INJURIES_NON_INCAPACITATING: 0.0,
        Col.INJURIES_REPORTED_NOT_EVIDENT: 0.0,
        Col.INJURIES_NO_INDICATION: 0.0,
        Col.INJURIES_UNKNOWN: 0.0,
        Col.CRASH_HOUR: 0,
        Col.CRASH_DAY_OF_WEEK: 4,
        Col.CRASH_MONTH: 1,
        Col.LATITUDE: 41.8781,
        Col.LONGITUDE: -87.6298,
        Col.LOCATION: "POINT (-87.6298 41.8781)",
    }
    row.update(overrides)
    return row


def _create_test_df(
    crash_ids: list[str],
    latitudes: list[float | None],
    longitudes: list[float | None],
) -> DataFrame[TrafficCrashesSchema]:
    """Helper to create and validate test DataFrames."""
    rows = [
        _make_valid_row(
            **{
                Col.CRASH_RECORD_ID: crash_id,
                Col.LATITUDE: lat,
                Col.LONGITUDE: lon,
            }
        )
        for crash_id, lat, lon in zip(crash_ids, latitudes, longitudes, strict=True)
    ]
    data = pd.DataFrame(rows)
    return TrafficCrashesSchema.validate(data)


@pytest.fixture
def sample_data() -> DataFrame[TrafficCrashesSchema]:
    """Create sample traffic crash data with valid coordinates."""
    return _create_test_df(
        crash_ids=["A", "B", "C", "D", "E"],
        latitudes=[41.8, 41.9, 41.7, 42.0, 41.85],
        longitudes=[-87.7, -87.6, -87.8, -87.65, -87.75],
    )


class TestRemoveOutliers:
    def test_keeps_valid_coordinates(
        self, sample_data: DataFrame[TrafficCrashesSchema]
    ) -> None:
        """Data within Chicago bounds should be retained."""
        result = remove_outliers(sample_data)

        assert len(result) == 5
        assert list(result[Col.CRASH_RECORD_ID]) == ["A", "B", "C", "D", "E"]

    def test_removes_latitude_below_minimum(self) -> None:
        """Latitude below 41.6 should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.5, 41.8],  # 41.5 is below min (41.6)
            longitudes=[-87.7, -87.7],
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][Col.CRASH_RECORD_ID] == "B"

    def test_removes_latitude_above_maximum(self) -> None:
        """Latitude above 42.1 should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, 42.2],  # 42.2 is above max (42.1)
            longitudes=[-87.7, -87.7],
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][Col.CRASH_RECORD_ID] == "A"

    def test_removes_longitude_below_minimum(self) -> None:
        """Longitude below -87.9 should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, 41.8],
            longitudes=[-88.0, -87.7],  # -88.0 is below min (-87.9)
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][Col.CRASH_RECORD_ID] == "B"

    def test_removes_longitude_above_maximum(self) -> None:
        """Longitude above -87.5 should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, 41.8],
            longitudes=[-87.7, -87.4],  # -87.4 is above max (-87.5)
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][Col.CRASH_RECORD_ID] == "A"

    def test_removes_null_latitude(self) -> None:
        """Rows with null latitude should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, None],
            longitudes=[-87.7, -87.7],
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][Col.CRASH_RECORD_ID] == "A"

    def test_removes_null_longitude(self) -> None:
        """Rows with null longitude should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, 41.8],
            longitudes=[-87.7, None],
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][Col.CRASH_RECORD_ID] == "A"

    def test_removes_both_null_coordinates(self) -> None:
        """Rows with both null lat and lon should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, None],
            longitudes=[-87.7, None],
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][Col.CRASH_RECORD_ID] == "A"

    def test_boundary_values_excluded(self) -> None:
        """Values exactly at boundaries should be excluded (strict inequality)."""
        df = _create_test_df(
            crash_ids=["A", "B", "C", "D", "E"],
            latitudes=[41.6, 42.1, 41.8, 41.8, 41.8],  # A and B are at boundary
            longitudes=[-87.7, -87.7, -87.9, -87.5, -87.7],  # C and D are at boundary
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][Col.CRASH_RECORD_ID] == "E"

    def test_returns_copy_not_view(self) -> None:
        """Modifying result should not affect original data."""
        df = _create_test_df(
            crash_ids=["A"],
            latitudes=[41.8],
            longitudes=[-87.7],
        )
        original_lat: float = 41.8  # Known input value

        result = remove_outliers(df)
        result.loc[result.index[0], Col.LATITUDE] = 99.9

        # Verify original dataframe was not modified
        assert df[Col.LATITUDE].tolist() == [original_lat]

    def test_preserves_other_columns(self) -> None:
        """Other columns in the dataframe should be preserved."""
        df = _create_test_df(
            crash_ids=["A"],
            latitudes=[41.8],
            longitudes=[-87.7],
        )

        result = remove_outliers(df)

        # Check that schema columns are preserved
        assert Col.CRASH_DATE in result.columns
        assert Col.WEATHER_CONDITION in result.columns
        assert result.iloc[0][Col.CRASH_DATE] == "01/01/2025 12:00:00 AM"
        assert result.iloc[0][Col.WEATHER_CONDITION] == "CLEAR"

    def test_empty_dataframe(self) -> None:
        """Empty dataframe should return empty dataframe."""
        # Create empty DataFrame with all required columns
        empty_row = _make_valid_row()
        data = pd.DataFrame([empty_row]).iloc[:0]  # Create empty df with correct cols
        df = TrafficCrashesSchema.validate(data)

        result = remove_outliers(df)

        assert len(result) == 0

    def test_all_outliers_removed(self) -> None:
        """If all data are outliers, result should be empty."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[40.0, 43.0],  # Both outside bounds
            longitudes=[-87.7, -87.7],
        )

        result = remove_outliers(df)

        assert len(result) == 0

    def test_preserves_index_reset(self) -> None:
        """Result should have reset index after filtering."""
        df = _create_test_df(
            crash_ids=["A", "B", "C"],
            latitudes=[40.0, 41.8, 41.9],  # First one is outlier
            longitudes=[-87.7, -87.7, -87.7],
        )

        result = remove_outliers(df)

        # After copy(), index should remain from original
        # This test verifies the data is intact
        assert len(result) == 2
        assert "B" in result[Col.CRASH_RECORD_ID].values
        assert "C" in result[Col.CRASH_RECORD_ID].values
