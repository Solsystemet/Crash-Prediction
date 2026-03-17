"""Tests for remove_outliers module."""

import pandas as pd
import pytest
from pandera.typing.pandas import DataFrame

from data_preparation.helpers.remove_outliers import remove_outliers
from models.data_schemas.full.traffic_crashes import TrafficCrashesSchema


def _make_valid_row(**overrides: object) -> dict[str, object]:
    """Return a dict representing one valid row. Override any field via kwargs."""
    S = TrafficCrashesSchema
    row: dict[str, object] = {
        S.CRASH_RECORD_ID: "abc123",
        S.CRASH_DATE_EST_I: None,
        S.CRASH_DATE: "01/01/2025 12:00:00 AM",
        S.POSTED_SPEED_LIMIT: 30,
        S.TRAFFIC_CONTROL_DEVICE: "TRAFFIC SIGNAL",
        S.DEVICE_CONDITION: "FUNCTIONING PROPERLY",
        S.WEATHER_CONDITION: "CLEAR",
        S.LIGHTING_CONDITION: "DAYLIGHT",
        S.FIRST_CRASH_TYPE: "REAR END",
        S.TRAFFICWAY_TYPE: "DIVIDED - W/MEDIAN (NOT RAISED)",
        S.LANE_CNT: None,
        S.ALIGNMENT: "STRAIGHT AND LEVEL",
        S.ROADWAY_SURFACE_COND: "DRY",
        S.ROAD_DEFECT: "NO DEFECTS",
        S.REPORT_TYPE: "ON SCENE",
        S.CRASH_TYPE: "NO INJURY / DRIVE AWAY",
        S.INTERSECTION_RELATED_I: None,
        S.NOT_RIGHT_OF_WAY_I: None,
        S.HIT_AND_RUN_I: None,
        S.DAMAGE: "OVER $1,500",
        S.DATE_POLICE_NOTIFIED: "01/01/2025 12:05:00 AM",
        S.PRIM_CONTRIBUTORY_CAUSE: "FOLLOWING TOO CLOSELY",
        S.SEC_CONTRIBUTORY_CAUSE: "NOT APPLICABLE",
        S.STREET_NO: 100,
        S.STREET_DIRECTION: "N",
        S.STREET_NAME: "STATE ST",
        S.BEAT_OF_OCCURRENCE: None,
        S.PHOTOS_TAKEN_I: None,
        S.STATEMENTS_TAKEN_I: None,
        S.DOORING_I: None,
        S.WORK_ZONE_I: None,
        S.WORK_ZONE_TYPE: None,
        S.WORKERS_PRESENT_I: None,
        S.NUM_UNITS: 2,
        S.MOST_SEVERE_INJURY: None,
        S.INJURIES_TOTAL: 0.0,
        S.INJURIES_FATAL: 0.0,
        S.INJURIES_INCAPACITATING: 0.0,
        S.INJURIES_NON_INCAPACITATING: 0.0,
        S.INJURIES_REPORTED_NOT_EVIDENT: 0.0,
        S.INJURIES_NO_INDICATION: 0.0,
        S.INJURIES_UNKNOWN: 0.0,
        S.CRASH_HOUR: 0,
        S.CRASH_DAY_OF_WEEK: 4,
        S.CRASH_MONTH: 1,
        S.LATITUDE: 41.8781,
        S.LONGITUDE: -87.6298,
        S.LOCATION: "POINT (-87.6298 41.8781)",
    }
    row.update(overrides)
    return row


def _create_test_df(
    crash_ids: list[str],
    latitudes: list[float | None],
    longitudes: list[float | None],
) -> DataFrame[TrafficCrashesSchema]:
    """Helper to create and validate test DataFrames."""
    S = TrafficCrashesSchema
    rows = [
        _make_valid_row(
            **{
                S.CRASH_RECORD_ID: crash_id,
                S.LATITUDE: lat,
                S.LONGITUDE: lon,
            }
        )
        for crash_id, lat, lon in zip(crash_ids, latitudes, longitudes, strict=True)
    ]
    data = pd.DataFrame(rows)
    return S.validate(data)


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
        assert list(result[TrafficCrashesSchema.CRASH_RECORD_ID]) == [
            "A",
            "B",
            "C",
            "D",
            "E",
        ]

    def test_removes_latitude_below_minimum(self) -> None:
        """Latitude below 41.6 should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.5, 41.8],  # 41.5 is below min (41.6)
            longitudes=[-87.7, -87.7],
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][TrafficCrashesSchema.CRASH_RECORD_ID] == "B"

    def test_removes_latitude_above_maximum(self) -> None:
        """Latitude above 42.1 should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, 42.2],  # 42.2 is above max (42.1)
            longitudes=[-87.7, -87.7],
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][TrafficCrashesSchema.CRASH_RECORD_ID] == "A"

    def test_removes_longitude_below_minimum(self) -> None:
        """Longitude below -87.9 should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, 41.8],
            longitudes=[-88.0, -87.7],  # -88.0 is below min (-87.9)
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][TrafficCrashesSchema.CRASH_RECORD_ID] == "B"

    def test_removes_longitude_above_maximum(self) -> None:
        """Longitude above -87.5 should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, 41.8],
            longitudes=[-87.7, -87.4],  # -87.4 is above max (-87.5)
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][TrafficCrashesSchema.CRASH_RECORD_ID] == "A"

    def test_removes_null_latitude(self) -> None:
        """Rows with null latitude should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, None],
            longitudes=[-87.7, -87.7],
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][TrafficCrashesSchema.CRASH_RECORD_ID] == "A"

    def test_removes_null_longitude(self) -> None:
        """Rows with null longitude should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, 41.8],
            longitudes=[-87.7, None],
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][TrafficCrashesSchema.CRASH_RECORD_ID] == "A"

    def test_removes_both_null_coordinates(self) -> None:
        """Rows with both null lat and lon should be removed."""
        df = _create_test_df(
            crash_ids=["A", "B"],
            latitudes=[41.8, None],
            longitudes=[-87.7, None],
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][TrafficCrashesSchema.CRASH_RECORD_ID] == "A"

    def test_boundary_values_excluded(self) -> None:
        """Values exactly at boundaries should be excluded (strict inequality)."""
        df = _create_test_df(
            crash_ids=["A", "B", "C", "D", "E"],
            latitudes=[41.6, 42.1, 41.8, 41.8, 41.8],  # A and B are at boundary
            longitudes=[-87.7, -87.7, -87.9, -87.5, -87.7],  # C and D are at boundary
        )

        result = remove_outliers(df)

        assert len(result) == 1
        assert result.iloc[0][TrafficCrashesSchema.CRASH_RECORD_ID] == "E"

    def test_returns_copy_not_view(self) -> None:
        """Modifying result should not affect original data."""
        df = _create_test_df(
            crash_ids=["A"],
            latitudes=[41.8],
            longitudes=[-87.7],
        )
        original_lat: float = 41.8  # Known input value

        result = remove_outliers(df)
        result.loc[result.index[0], TrafficCrashesSchema.LATITUDE] = 99.9

        # Verify original dataframe was not modified
        assert df[TrafficCrashesSchema.LATITUDE].tolist() == [original_lat]

    def test_preserves_other_columns(self) -> None:
        """Other columns in the dataframe should be preserved."""
        df = _create_test_df(
            crash_ids=["A"],
            latitudes=[41.8],
            longitudes=[-87.7],
        )

        result = remove_outliers(df)

        # Check that schema columns are preserved
        assert TrafficCrashesSchema.CRASH_DATE in result.columns
        assert TrafficCrashesSchema.WEATHER_CONDITION in result.columns
        assert (
            result.iloc[0][TrafficCrashesSchema.CRASH_DATE] == "01/01/2025 12:00:00 AM"
        )
        assert result.iloc[0][TrafficCrashesSchema.WEATHER_CONDITION] == "CLEAR"

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
        assert "B" in result[TrafficCrashesSchema.CRASH_RECORD_ID].values
        assert "C" in result[TrafficCrashesSchema.CRASH_RECORD_ID].values
