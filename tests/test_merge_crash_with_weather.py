"""Tests for merge_crash_with_weather module."""

import pandas as pd
import pytest
from pathlib import Path

from data_preparation.merge_crash_with_weather import merge_crash_with_weather


@pytest.fixture
def sample_crash_csv(tmp_path: Path) -> Path:
    """Create sample crash data CSV."""
    df = pd.DataFrame({
        "CRASH_DATE": [
            "2024-01-01 10:00:00",  # Exact hour
            "2024-01-01 10:15:00",  # Rounds down to 10:00
            "2024-01-01 10:30:00",  # Rounds up to 11:00
            "2024-01-01 10:45:00",  # Rounds up to 11:00
            "2024-01-01 14:20:00",  # Rounds down to 14:00
        ],
        "CRASH_ID": ["A", "B", "C", "D", "E"],
        "SEVERITY": [1, 2, 3, 1, 2]
    })
    csv_path = tmp_path / "crashes.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_weather_csv(tmp_path: Path) -> Path:
    """Create sample weather data CSV."""
    df = pd.DataFrame({
        "MEASUREMENT_TIMESTAMP": [
            "2024-01-01 10:00:00",
            "2024-01-01 11:00:00",
            "2024-01-01 12:00:00",
            "2024-01-01 14:00:00",
        ],
        "TEMPERATURE": [32, 35, 38, 42],
        "HUMIDITY": [80, 75, 70, 65]
    })
    csv_path = tmp_path / "weather.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def output_csv(tmp_path: Path) -> Path:
    """Create output path for merged CSV."""
    return tmp_path / "merged.csv"


class TestMergeCrashWithWeather:
    def test_merges_exact_hour(
        self, sample_crash_csv: Path, sample_weather_csv: Path, output_csv: Path
    ):
        """Crash at exact hour should match that hour's weather."""
        result = merge_crash_with_weather(
            crash_path=sample_crash_csv,
            weather_path=sample_weather_csv,
            output_path=output_csv
        )
        
        row_a = result[result["CRASH_ID"] == "A"].iloc[0]
        assert row_a["TEMPERATURE"] == 32
        assert row_a["HUMIDITY"] == 80

    def test_rounds_down_before_half_hour(
        self, sample_crash_csv: Path, sample_weather_csv: Path, output_csv: Path
    ):
        """Crash at 10:15 should use 10:00 weather."""
        result = merge_crash_with_weather(
            crash_path=sample_crash_csv,
            weather_path=sample_weather_csv,
            output_path=output_csv
        )
        
        row_b = result[result["CRASH_ID"] == "B"].iloc[0]
        assert row_b["TEMPERATURE"] == 32  # 10:00 weather

    def test_rounds_up_at_half_hour(
        self, sample_crash_csv: Path, sample_weather_csv: Path, output_csv: Path
    ):
        """Crash at 10:30 should use 11:00 weather."""
        result = merge_crash_with_weather(
            crash_path=sample_crash_csv,
            weather_path=sample_weather_csv,
            output_path=output_csv
        )
        
        row_c = result[result["CRASH_ID"] == "C"].iloc[0]
        assert row_c["TEMPERATURE"] == 35  # 11:00 weather

    def test_rounds_up_after_half_hour(
        self, sample_crash_csv: Path, sample_weather_csv: Path, output_csv: Path
    ):
        """Crash at 10:45 should use 11:00 weather."""
        result = merge_crash_with_weather(
            crash_path=sample_crash_csv,
            weather_path=sample_weather_csv,
            output_path=output_csv
        )
        
        row_d = result[result["CRASH_ID"] == "D"].iloc[0]
        assert row_d["TEMPERATURE"] == 35  # 11:00 weather

    def test_preserves_crash_data(
        self, sample_crash_csv: Path, sample_weather_csv: Path, output_csv: Path
    ):
        """Should keep all original crash columns."""
        result = merge_crash_with_weather(
            crash_path=sample_crash_csv,
            weather_path=sample_weather_csv,
            output_path=output_csv
        )
        
        assert "CRASH_ID" in result.columns
        assert "SEVERITY" in result.columns
        assert "CRASH_DATE" in result.columns
        assert len(result) == 5

    def test_adds_weather_columns(
        self, sample_crash_csv: Path, sample_weather_csv: Path, output_csv: Path
    ):
        """Should add weather columns to result."""
        result = merge_crash_with_weather(
            crash_path=sample_crash_csv,
            weather_path=sample_weather_csv,
            output_path=output_csv
        )
        
        assert "TEMPERATURE" in result.columns
        assert "HUMIDITY" in result.columns

    def test_saves_merged_csv(
        self, sample_crash_csv: Path, sample_weather_csv: Path, output_csv: Path
    ):
        """Should create output CSV file."""
        merge_crash_with_weather(
            crash_path=sample_crash_csv,
            weather_path=sample_weather_csv,
            output_path=output_csv
        )
        
        assert output_csv.exists()
        saved_df = pd.read_csv(output_csv)
        assert len(saved_df) == 5
        assert "TEMPERATURE" in saved_df.columns

    def test_unmatched_crashes_have_null_weather(self, tmp_path: Path):
        """Crashes with no matching weather should have null weather values."""
        crash_df = pd.DataFrame({
            "CRASH_DATE": ["2024-01-01 20:00:00"],
            "CRASH_ID": ["X"]
        })
        crash_path = tmp_path / "crash.csv"
        crash_df.to_csv(crash_path, index=False)
        
        weather_df = pd.DataFrame({
            "MEASUREMENT_TIMESTAMP": ["2024-01-01 10:00:00"],
            "TEMPERATURE": [32]
        })
        weather_path = tmp_path / "weather.csv"
        weather_df.to_csv(weather_path, index=False)
        
        output_path = tmp_path / "merged.csv"
        
        result = merge_crash_with_weather(
            crash_path=crash_path,
            weather_path=weather_path,
            output_path=output_path
        )
        
        assert pd.isna(result.iloc[0]["TEMPERATURE"])

    def test_custom_timestamp_columns(self, tmp_path: Path):
        """Should work with custom timestamp column names."""
        crash_df = pd.DataFrame({
            "TIME_OF_CRASH": ["2024-01-01 10:00:00"],
            "CRASH_ID": ["A"]
        })
        crash_path = tmp_path / "crash.csv"
        crash_df.to_csv(crash_path, index=False)
        
        weather_df = pd.DataFrame({
            "WEATHER_TIME": ["2024-01-01 10:00:00"],
            "TEMPERATURE": [32]
        })
        weather_path = tmp_path / "weather.csv"
        weather_df.to_csv(weather_path, index=False)
        
        output_path = tmp_path / "merged.csv"
        
        result = merge_crash_with_weather(
            crash_path=crash_path,
            weather_path=weather_path,
            output_path=output_path,
            crash_timestamp_col="TIME_OF_CRASH",
            weather_timestamp_col="WEATHER_TIME"
        )
        
        assert result.iloc[0]["TEMPERATURE"] == 32

    def test_duplicate_weather_entries_uses_first(self, tmp_path: Path):
        """When multiple weather entries exist for same hour, use first."""
        crash_df = pd.DataFrame({
            "CRASH_DATE": ["2024-01-01 10:00:00"],
            "CRASH_ID": ["A"]
        })
        crash_path = tmp_path / "crash.csv"
        crash_df.to_csv(crash_path, index=False)
        
        weather_df = pd.DataFrame({
            "MEASUREMENT_TIMESTAMP": [
                "2024-01-01 10:00:00",
                "2024-01-01 10:00:00"
            ],
            "TEMPERATURE": [32, 99]
        })
        weather_path = tmp_path / "weather.csv"
        weather_df.to_csv(weather_path, index=False)
        
        output_path = tmp_path / "merged.csv"
        
        result = merge_crash_with_weather(
            crash_path=crash_path,
            weather_path=weather_path,
            output_path=output_path
        )
        
        assert result.iloc[0]["TEMPERATURE"] == 32