"""Tests for merge_crash_with_weather module."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from data_preparation.merge_crash_with_weather import merge_crash_with_weather


@pytest.fixture
def sample_crash_df() -> pd.DataFrame:
    """Create sample crash data DataFrame."""
    return pd.DataFrame(
        {
            "CRASH_DATE": [
                "2024-01-01 10:00:00",  # Exact hour
                "2024-01-01 10:15:00",  # Rounds down to 10:00
                "2024-01-01 10:30:00",  # Rounds up to 11:00
                "2024-01-01 10:45:00",  # Rounds up to 11:00
                "2024-01-01 14:20:00",  # Rounds down to 14:00
            ],
            "CRASH_ID": ["A", "B", "C", "D", "E"],
            "SEVERITY": [1, 2, 3, 1, 2],
        }
    )


@pytest.fixture
def sample_weather_df() -> pd.DataFrame:
    """Create sample weather data DataFrame."""
    return pd.DataFrame(
        {
            "Measurement Timestamp": [
                "2024-01-01 10:00:00",
                "2024-01-01 11:00:00",
                "2024-01-01 12:00:00",
                "2024-01-01 14:00:00",
            ],
            "Air Temperature": [32, 35, 38, 42],
            "Humidity": [80, 75, 70, 65],
        }
    )


@pytest.fixture
def output_csv(tmp_path: Path) -> Path:
    """Create output path for merged CSV."""
    return tmp_path / "merged.csv"


class TestMergeCrashWithWeather:
    def test_merges_exact_hour(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
        output_csv: Path,
    ) -> None:
        """Crash at exact hour should match that hour's weather."""
        with (
            patch(
                "data_preparation.merge_crash_with_weather.get_traffic_crashes",
                return_value=sample_crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_weather.get_weather_stations",
                return_value=sample_weather_df,
            ),
        ):
            result = merge_crash_with_weather(output_path=output_csv)

        row_a = result[result["CRASH_ID"] == "A"].iloc[0]
        assert row_a["Air Temperature"] == 32
        assert row_a["Humidity"] == 80

    def test_rounds_down_before_half_hour(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
        output_csv: Path,
    ) -> None:
        """Crash at 10:15 should use 10:00 weather."""
        with (
            patch(
                "data_preparation.merge_crash_with_weather.get_traffic_crashes",
                return_value=sample_crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_weather.get_weather_stations",
                return_value=sample_weather_df,
            ),
        ):
            result = merge_crash_with_weather(output_path=output_csv)

        row_b = result[result["CRASH_ID"] == "B"].iloc[0]
        assert row_b["Air Temperature"] == 32  # 10:00 weather

    def test_rounds_up_at_half_hour(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
        output_csv: Path,
    ) -> None:
        """Crash at 10:30 should use 11:00 weather."""
        with (
            patch(
                "data_preparation.merge_crash_with_weather.get_traffic_crashes",
                return_value=sample_crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_weather.get_weather_stations",
                return_value=sample_weather_df,
            ),
        ):
            result = merge_crash_with_weather(output_path=output_csv)

        row_c = result[result["CRASH_ID"] == "C"].iloc[0]
        assert row_c["Air Temperature"] == 35  # 11:00 weather

    def test_rounds_up_after_half_hour(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
        output_csv: Path,
    ) -> None:
        """Crash at 10:45 should use 11:00 weather."""
        with (
            patch(
                "data_preparation.merge_crash_with_weather.get_traffic_crashes",
                return_value=sample_crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_weather.get_weather_stations",
                return_value=sample_weather_df,
            ),
        ):
            result = merge_crash_with_weather(output_path=output_csv)

        row_d = result[result["CRASH_ID"] == "D"].iloc[0]
        assert row_d["Air Temperature"] == 35  # 11:00 weather

    def test_preserves_crash_data(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
        output_csv: Path,
    ) -> None:
        """Should keep all original crash columns."""
        with (
            patch(
                "data_preparation.merge_crash_with_weather.get_traffic_crashes",
                return_value=sample_crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_weather.get_weather_stations",
                return_value=sample_weather_df,
            ),
        ):
            result = merge_crash_with_weather(output_path=output_csv)

        assert "CRASH_ID" in result.columns
        assert "SEVERITY" in result.columns
        assert "CRASH_DATE" in result.columns
        assert len(result) == 5

    def test_adds_weather_columns(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
        output_csv: Path,
    ) -> None:
        """Should add weather columns to result."""
        with (
            patch(
                "data_preparation.merge_crash_with_weather.get_traffic_crashes",
                return_value=sample_crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_weather.get_weather_stations",
                return_value=sample_weather_df,
            ),
        ):
            result = merge_crash_with_weather(output_path=output_csv)

        assert "Air Temperature" in result.columns
        assert "Humidity" in result.columns

    def test_saves_merged_csv(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
        output_csv: Path,
    ) -> None:
        """Should create output CSV file."""
        with (
            patch(
                "data_preparation.merge_crash_with_weather.get_traffic_crashes",
                return_value=sample_crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_weather.get_weather_stations",
                return_value=sample_weather_df,
            ),
        ):
            merge_crash_with_weather(output_path=output_csv)

        assert output_csv.exists()
        saved_df = pd.read_csv(output_csv)
        assert len(saved_df) == 5
        assert "Air Temperature" in saved_df.columns

    def test_unmatched_crashes_have_null_weather(self, output_csv: Path) -> None:
        """Crashes with no matching weather should have null weather values."""
        crash_df = pd.DataFrame(
            {"CRASH_DATE": ["2024-01-01 20:00:00"], "CRASH_ID": ["X"]}
        )
        weather_df = pd.DataFrame(
            {"Measurement Timestamp": ["2024-01-01 10:00:00"], "Air Temperature": [32]}
        )

        with (
            patch(
                "data_preparation.merge_crash_with_weather.get_traffic_crashes",
                return_value=crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_weather.get_weather_stations",
                return_value=weather_df,
            ),
        ):
            result = merge_crash_with_weather(output_path=output_csv)

        assert pd.isna(result.iloc[0]["Air Temperature"])

    def test_duplicate_weather_entries_uses_first(self, output_csv: Path) -> None:
        """When multiple weather entries exist for same hour, use first."""
        crash_df = pd.DataFrame(
            {"CRASH_DATE": ["2024-01-01 10:00:00"], "CRASH_ID": ["A"]}
        )
        weather_df = pd.DataFrame(
            {
                "Measurement Timestamp": [
                    "2024-01-01 10:00:00",
                    "2024-01-01 10:00:00",
                ],
                "Air Temperature": [32, 99],
            }
        )

        with (
            patch(
                "data_preparation.merge_crash_with_weather.get_traffic_crashes",
                return_value=crash_df,
            ),
            patch(
                "data_preparation.merge_crash_with_weather.get_weather_stations",
                return_value=weather_df,
            ),
        ):
            result = merge_crash_with_weather(output_path=output_csv)

        assert result.iloc[0]["Air Temperature"] == 32
