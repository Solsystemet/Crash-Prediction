"""Unit tests for the TrafficTrackerSchema."""

from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from models.data_schemas.full.traffic_tracker import TrafficTrackerSchema

DATA_FILE = Path(
    "data/chicago_traffic_tracker_historical_congestion_estimates_by_segment_2024_current.csv"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_row(**overrides: object) -> dict[str, object]:
    """Return a dict representing one valid traffic tracker row."""
    row: dict[str, object] = {
        "TIME": "01/01/2025 12:00:00 PM",
        "SEGMENT_ID": 1152,
        "SPEED": 25.0,
        "STREET": "Harlem",
        "DIRECTION": "NB",
        "FROM_STREET": "Chicago",
        "TO_STREET": "North Ave",
        "LENGTH": 1.01,
        "STREET_HEADING": "N",
        "COMMENTS": None,
        "BUS_COUNT": 3,
        "MESSAGE_COUNT": 15,
        "HOUR": 12,
        "DAY_OF_WEEK": 4,
        "MONTH": 1,
        "RECORD_ID": "1152-202501011200",
        "START_LATITUDE": 41.8943,
        "START_LONGITUDE": -87.8051,
        "END_LATITUDE": 41.9089,
        "END_LONGITUDE": -87.8056,
        "START_LOCATION": "POINT (-87.8051 41.8943)",
        "END_LOCATION": "POINT (-87.8056 41.9089)",
    }
    row.update(overrides)
    return row


def _df_from_row(**overrides: object) -> pd.DataFrame:
    return pd.DataFrame([_make_valid_row(**overrides)])


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestTrafficTrackerSchemaValid:
    """Tests that valid data passes schema validation."""

    def test_minimal_valid_row(self) -> None:
        df = _df_from_row()
        validated = TrafficTrackerSchema.validate(df)
        assert len(validated) == 1

    def test_nullable_fields_accept_none(self) -> None:
        df = _df_from_row(
            LENGTH=None,
            COMMENTS=None,
            START_LATITUDE=None,
            START_LONGITUDE=None,
            END_LATITUDE=None,
            END_LONGITUDE=None,
            START_LOCATION=None,
            END_LOCATION=None,
        )
        validated = TrafficTrackerSchema.validate(df)
        assert len(validated) == 1

    def test_multiple_rows(self) -> None:
        rows = [
            _make_valid_row(RECORD_ID="rec_1"),
            _make_valid_row(RECORD_ID="rec_2"),
        ]
        df = pd.DataFrame(rows)
        validated = TrafficTrackerSchema.validate(df)
        assert len(validated) == 2

    def test_speed_minus_one(self) -> None:
        """Speed of -1 means no estimate available and should be valid."""
        df = _df_from_row(SPEED=-1.0)
        validated = TrafficTrackerSchema.validate(df)
        assert len(validated) == 1

    def test_speed_zero(self) -> None:
        df = _df_from_row(SPEED=0.0)
        validated = TrafficTrackerSchema.validate(df)
        assert len(validated) == 1

    def test_boundary_hour(self) -> None:
        for hour in (0, 12, 23):
            df = _df_from_row(HOUR=hour)
            validated = TrafficTrackerSchema.validate(df)
            assert len(validated) == 1

    def test_boundary_day_of_week(self) -> None:
        for dow in (1, 4, 7):
            df = _df_from_row(DAY_OF_WEEK=dow)
            validated = TrafficTrackerSchema.validate(df)
            assert len(validated) == 1

    def test_boundary_month(self) -> None:
        for month in (1, 6, 12):
            df = _df_from_row(MONTH=month)
            validated = TrafficTrackerSchema.validate(df)
            assert len(validated) == 1

    @pytest.mark.skipif(not DATA_FILE.exists(), reason=f"{DATA_FILE} not found")
    def test_csv_sample(self) -> None:
        """Validate a sample from the real CSV file."""
        df = pd.read_csv(DATA_FILE, nrows=500)
        validated = TrafficTrackerSchema.validate(df)
        assert len(validated) == 500


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


class TestTrafficTrackerSchemaInvalid:
    """Tests that invalid data is rejected."""

    def test_duplicate_record_id(self) -> None:
        rows = [
            _make_valid_row(RECORD_ID="dup"),
            _make_valid_row(RECORD_ID="dup"),
        ]
        df = pd.DataFrame(rows)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficTrackerSchema.validate(df)

    def test_speed_below_minus_one(self) -> None:
        df = _df_from_row(SPEED=-2.0)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficTrackerSchema.validate(df)

    def test_negative_length(self) -> None:
        df = _df_from_row(LENGTH=-0.5)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficTrackerSchema.validate(df)

    def test_negative_bus_count(self) -> None:
        df = _df_from_row(BUS_COUNT=-1)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficTrackerSchema.validate(df)

    def test_negative_message_count(self) -> None:
        df = _df_from_row(MESSAGE_COUNT=-1)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficTrackerSchema.validate(df)

    def test_hour_out_of_range(self) -> None:
        df = _df_from_row(HOUR=24)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficTrackerSchema.validate(df)

    def test_day_of_week_out_of_range(self) -> None:
        df = _df_from_row(DAY_OF_WEEK=8)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficTrackerSchema.validate(df)

    def test_month_out_of_range(self) -> None:
        df = _df_from_row(MONTH=0)
        with pytest.raises(pandera.errors.SchemaError):
            _ = TrafficTrackerSchema.validate(df)
