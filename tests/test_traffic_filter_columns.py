"""Tests for column_filter module."""

import pandas as pd
import pytest
from pathlib import Path
import tempfile

from data_preparation.column_filter import filter_columns


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV file for testing."""
    df = pd.DataFrame({
        "CRASH_RECORD_ID": ["ABC123", "DEF456"],
        "CRASH_DATE": ["2024-01-01", "2024-01-02"],
        "POSTED_SPEED_LIMIT": [30, 45],
        "WEATHER_CONDITION": ["CLEAR", "RAIN"],
        "LOCATION": ["(41.8, -87.6)", "(41.9, -87.7)"]
    })
    csv_path = tmp_path / "test_input.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def output_csv(tmp_path: Path) -> Path:
    """Create output path for filtered CSV."""
    return tmp_path / "test_output.csv"


class TestFilterColumns:
    def test_drops_specified_columns(self, sample_csv: Path, output_csv: Path):
        """Should remove columns in the drop list."""
        result = filter_columns(
            input_path=sample_csv,
            output_path=output_csv,
            columns_to_drop=["CRASH_RECORD_ID", "LOCATION"]
        )
        
        assert "CRASH_RECORD_ID" not in result.columns
        assert "LOCATION" not in result.columns
        assert "CRASH_DATE" in result.columns
        assert "POSTED_SPEED_LIMIT" in result.columns

    def test_keeps_unspecified_columns(self, sample_csv: Path, output_csv: Path):
        """Should retain columns not in the drop list."""
        result = filter_columns(
            input_path=sample_csv,
            output_path=output_csv,
            columns_to_drop=["CRASH_RECORD_ID"]
        )
        
        assert len(result.columns) == 4
        assert "WEATHER_CONDITION" in result.columns

    def test_saves_filtered_csv(self, sample_csv: Path, output_csv: Path):
        """Should create output CSV with filtered columns."""
        filter_columns(
            input_path=sample_csv,
            output_path=output_csv,
            columns_to_drop=["CRASH_RECORD_ID", "LOCATION"]
        )
        
        assert output_csv.exists()
        saved_df = pd.read_csv(output_csv)
        assert "CRASH_RECORD_ID" not in saved_df.columns
        assert len(saved_df.columns) == 3

    def test_ignores_nonexistent_columns(self, sample_csv: Path, output_csv: Path, capsys):
        """Should warn but not fail when dropping columns that don't exist."""
        result = filter_columns(
            input_path=sample_csv,
            output_path=output_csv,
            columns_to_drop=["CRASH_RECORD_ID", "FAKE_COLUMN"]
        )
        
        captured = capsys.readouterr()
        assert "FAKE_COLUMN" in captured.out
        assert len(result.columns) == 4

    def test_empty_drop_list(self, sample_csv: Path, output_csv: Path):
        """Should return all columns when drop list is empty."""
        result = filter_columns(
            input_path=sample_csv,
            output_path=output_csv,
            columns_to_drop=[]
        )
        
        assert len(result.columns) == 5

    def test_preserves_data_values(self, sample_csv: Path, output_csv: Path):
        """Should not alter data values in remaining columns."""
        result = filter_columns(
            input_path=sample_csv,
            output_path=output_csv,
            columns_to_drop=["CRASH_RECORD_ID"]
        )
        
        assert result["POSTED_SPEED_LIMIT"].tolist() == [30, 45]
        assert result["WEATHER_CONDITION"].tolist() == ["CLEAR", "RAIN"]