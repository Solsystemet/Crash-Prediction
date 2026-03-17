"""Tests for merge_crash_with_people module."""

from unittest.mock import patch

import pandas as pd
import pytest

from data_preparation.merge_crash_with_people import (
    merge_crash_with_people,
    _aggregate_people_features,
)


@pytest.fixture
def sample_crash_df() -> pd.DataFrame:
    """Create sample crash data DataFrame."""
    return pd.DataFrame(
        {
            "CRASH_RECORD_ID": ["CR001", "CR002", "CR003"],
            "CRASH_DATE": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "WEATHER_CONDITION": ["CLEAR", "RAIN", "SNOW"],
            "POSTED_SPEED_LIMIT": [30, 45, 25],
        }
    )


@pytest.fixture
def sample_people_df() -> pd.DataFrame:
    """Create sample people data DataFrame with multiple people per crash."""
    return pd.DataFrame(
        {
            "PERSON_ID": ["P001", "P002", "P003", "P004", "P005", "P006"],
            "CRASH_RECORD_ID": ["CR001", "CR001", "CR002", "CR002", "CR002", "CR003"],
            "PERSON_TYPE": [
                "DRIVER",
                "PASSENGER",
                "DRIVER",
                "PEDESTRIAN",
                "PASSENGER",
                "DRIVER",
            ],
            "AGE": [35, 28, 72, 45, 19, 22],
            "INJURY_CLASSIFICATION": [
                "NO INDICATION OF INJURY",
                "NONINCAPACITATING INJURY",
                "INCAPACITATING INJURY",
                "FATAL",
                "NO INDICATION OF INJURY",
                "NO INDICATION OF INJURY",
            ],
            "SAFETY_EQUIPMENT": [
                "HELMET",
                "NONE USED",
                "LAP BELT",
                None,
                "SHOULDER BELT",
                "LAP/SHOULDER BELT",
            ],
            "BAC_RESULT": [
                "NEGATIVE",
                "NOT OFFERED",
                "POSITIVE",
                "NOT OFFERED",
                "NOT OFFERED",
                "NEGATIVE",
            ],
            "CELL_PHONE_USE": ["N", "N", "N", "N", "Y", "N"],
        }
    )


class TestAggregatePeopleFeatures:
    def test_counts_occupants_per_crash(
        self, sample_people_df: pd.DataFrame
    ) -> None:
        """Should count total people involved in each crash."""
        result = _aggregate_people_features(sample_people_df)

        cr001 = result[result["CRASH_RECORD_ID"] == "CR001"].iloc[0]
        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]
        cr003 = result[result["CRASH_RECORD_ID"] == "CR003"].iloc[0]

        assert cr001["num_occupants"] == 2
        assert cr002["num_occupants"] == 3
        assert cr003["num_occupants"] == 1

    def test_counts_pedestrians(self, sample_people_df: pd.DataFrame) -> None:
        """Should count pedestrians per crash."""
        result = _aggregate_people_features(sample_people_df)

        cr001 = result[result["CRASH_RECORD_ID"] == "CR001"].iloc[0]
        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]

        assert cr001["num_pedestrians"] == 0
        assert cr002["num_pedestrians"] == 1

    def test_detects_fatality(self, sample_people_df: pd.DataFrame) -> None:
        """Should flag crashes with any fatal injury."""
        result = _aggregate_people_features(sample_people_df)

        cr001 = result[result["CRASH_RECORD_ID"] == "CR001"].iloc[0]
        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]
        cr003 = result[result["CRASH_RECORD_ID"] == "CR003"].iloc[0]

        assert cr001["has_fatality"] == 0
        assert cr002["has_fatality"] == 1  # Pedestrian was fatal
        assert cr003["has_fatality"] == 0

    def test_detects_incapacitating_injury(
        self, sample_people_df: pd.DataFrame
    ) -> None:
        """Should flag crashes with incapacitating injuries."""
        result = _aggregate_people_features(sample_people_df)

        cr001 = result[result["CRASH_RECORD_ID"] == "CR001"].iloc[0]
        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]

        assert cr001["has_incapacitating_injury"] == 0
        assert cr002["has_incapacitating_injury"] == 1

    def test_detects_unbelted_occupants(
        self, sample_people_df: pd.DataFrame
    ) -> None:
        """Should flag crashes with unbelted occupants."""
        result = _aggregate_people_features(sample_people_df)

        cr001 = result[result["CRASH_RECORD_ID"] == "CR001"].iloc[0]
        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]
        cr003 = result[result["CRASH_RECORD_ID"] == "CR003"].iloc[0]

        assert cr001["has_unbelted"] == 1  # Passenger had NONE USED
        assert cr002["has_unbelted"] == 0
        assert cr003["has_unbelted"] == 0

    def test_detects_alcohol(self, sample_people_df: pd.DataFrame) -> None:
        """Should flag crashes with alcohol involvement."""
        result = _aggregate_people_features(sample_people_df)

        cr001 = result[result["CRASH_RECORD_ID"] == "CR001"].iloc[0]
        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]

        assert cr001["has_alcohol"] == 0
        assert cr002["has_alcohol"] == 1  # Driver had POSITIVE

    def test_age_statistics(self, sample_people_df: pd.DataFrame) -> None:
        """Should compute age min/max/mean correctly."""
        result = _aggregate_people_features(sample_people_df)

        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]

        assert cr002["age_min"] == 19
        assert cr002["age_max"] == 72
        assert cr002["age_mean"] == pytest.approx((72 + 45 + 19) / 3)

    def test_detects_elderly(self, sample_people_df: pd.DataFrame) -> None:
        """Should flag crashes with elderly occupants (65+)."""
        result = _aggregate_people_features(sample_people_df)

        cr001 = result[result["CRASH_RECORD_ID"] == "CR001"].iloc[0]
        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]

        assert cr001["has_elderly"] == 0
        assert cr002["has_elderly"] == 1  # 72-year-old driver

    def test_detects_young_driver(self, sample_people_df: pd.DataFrame) -> None:
        """Should flag crashes with young drivers (25 or under)."""
        result = _aggregate_people_features(sample_people_df)

        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]
        cr003 = result[result["CRASH_RECORD_ID"] == "CR003"].iloc[0]

        # CR002 has 19-year-old passenger, not driver
        assert cr002["has_young_driver"] == 0
        # CR003 has 22-year-old driver
        assert cr003["has_young_driver"] == 1

    def test_detects_cellphone_use(self, sample_people_df: pd.DataFrame) -> None:
        """Should flag crashes with cellphone use."""
        result = _aggregate_people_features(sample_people_df)

        cr001 = result[result["CRASH_RECORD_ID"] == "CR001"].iloc[0]
        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]

        assert cr001["has_cellphone_use"] == 0
        assert cr002["has_cellphone_use"] == 1


class TestMergeCrashWithPeople:
    def test_merges_crash_with_people_features(
        self,
        sample_crash_df: pd.DataFrame,
        sample_people_df: pd.DataFrame,
    ) -> None:
        """Should merge crash data with aggregated people features."""
        with (
            patch(
                "data_preparation.merge_crash_with_people.get_traffic_crashes",
                return_value=sample_crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_people.get_crash_people",
                return_value=sample_people_df,
            ),
        ):
            result = merge_crash_with_people(output_path=None)

        # Should have all 3 crashes
        assert len(result) == 3

        # Should have original crash columns
        assert "CRASH_RECORD_ID" in result.columns
        assert "WEATHER_CONDITION" in result.columns
        assert "POSTED_SPEED_LIMIT" in result.columns

        # Should have people-derived columns
        assert "num_occupants" in result.columns
        assert "has_fatality" in result.columns
        assert "has_alcohol" in result.columns

    def test_preserves_crash_data(
        self,
        sample_crash_df: pd.DataFrame,
        sample_people_df: pd.DataFrame,
    ) -> None:
        """Should preserve original crash data after merge."""
        with (
            patch(
                "data_preparation.merge_crash_with_people.get_traffic_crashes",
                return_value=sample_crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_people.get_crash_people",
                return_value=sample_people_df,
            ),
        ):
            result = merge_crash_with_people(output_path=None)

        cr001 = result[result["CRASH_RECORD_ID"] == "CR001"].iloc[0]
        assert cr001["WEATHER_CONDITION"] == "CLEAR"
        assert cr001["POSTED_SPEED_LIMIT"] == 30

    def test_handles_crash_without_people_data(
        self,
        sample_crash_df: pd.DataFrame,
    ) -> None:
        """Should handle crashes that don't have matching people records."""
        # People data only for CR001, not CR002 or CR003
        limited_people_df = pd.DataFrame(
            {
                "PERSON_ID": ["P001"],
                "CRASH_RECORD_ID": ["CR001"],
                "PERSON_TYPE": ["DRIVER"],
                "AGE": [35],
                "INJURY_CLASSIFICATION": ["NO INDICATION OF INJURY"],
                "SAFETY_EQUIPMENT": ["LAP BELT"],
                "BAC_RESULT": ["NEGATIVE"],
                "CELL_PHONE_USE": ["N"],
            }
        )

        with (
            patch(
                "data_preparation.merge_crash_with_people.get_traffic_crashes",
                return_value=sample_crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_people.get_crash_people",
                return_value=limited_people_df,
            ),
        ):
            result = merge_crash_with_people(output_path=None)

        # All crashes should be present
        assert len(result) == 3

        # Unmatched crashes should have 0 for numeric people features
        cr002 = result[result["CRASH_RECORD_ID"] == "CR002"].iloc[0]
        assert cr002["num_occupants"] == 0
        assert cr002["has_fatality"] == 0
