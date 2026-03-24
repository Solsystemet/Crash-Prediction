"""Tests for hourly crash aggregation and lag features."""

import numpy as np
import pandas as pd
import pytest

from data_preparation.aggregate_hourly import (
    add_cyclical_time_features,
    add_weather_lag_features,
    aggregate_crashes_by_hour,
    get_hourly_feature_columns,
    merge_hourly_crashes_with_weather,
    prepare_hourly_data,
    DEFAULT_WEATHER_LAG_COLUMNS,
)
from data_preparation.prepare_tensor_data import _split_data
from data_preparation.tensor_config import HOURLY_CRASH_COUNT_CONFIG, TensorConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_crash_df() -> pd.DataFrame:
    """Create sample crash DataFrame with timestamps."""
    return pd.DataFrame({
        "CRASH_DATE": [
            "2024-01-01 10:15:00",
            "2024-01-01 10:45:00",  # Same hour as above
            "2024-01-01 12:30:00",
            "2024-01-01 14:00:00",
            "2024-01-02 08:00:00",
        ],
        "CRASH_RECORD_ID": ["A", "B", "C", "D", "E"],
    })


@pytest.fixture
def sample_weather_df() -> pd.DataFrame:
    """Create sample weather DataFrame aligned with crash times."""
    # Create hourly weather data for 2 days
    hours = pd.date_range("2024-01-01 00:00", periods=48, freq="h")
    return pd.DataFrame({
        "Measurement Timestamp": hours,
        "Air_Temperature": np.sin(np.arange(48) * np.pi / 12) * 10 + 15,  # Varies by time
        "Humidity": np.random.default_rng(42).uniform(40, 80, 48),
        "Rain_Intensity": [0] * 24 + [0.5] * 24,  # Rain on day 2
        "Wind_Speed": np.random.default_rng(42).uniform(0, 20, 48),
    })


@pytest.fixture
def sample_hourly_df() -> pd.DataFrame:
    """Create sample hourly DataFrame with weather and crash counts."""
    hours = pd.date_range("2024-01-01 00:00", periods=10, freq="h")
    return pd.DataFrame({
        "hour_timestamp": hours,
        "crash_count": [0, 1, 2, 0, 1, 3, 2, 1, 0, 1],
        "Air_Temperature": [10, 12, 14, 15, 16, 17, 18, 17, 15, 13],
        "Humidity": [60, 58, 55, 52, 50, 48, 50, 55, 60, 65],
        "Rain_Intensity": [0, 0, 0.1, 0.5, 0.3, 0, 0, 0, 0.2, 0.4],
        "Wind_Speed": [5, 7, 8, 10, 12, 10, 8, 6, 5, 4],
    })


# =============================================================================
# Aggregation Tests
# =============================================================================


class TestAggregateCrashesByHour:
    """Tests for aggregate_crashes_by_hour function."""

    def test_counts_crashes_in_same_hour(self, sample_crash_df: pd.DataFrame) -> None:
        """Crashes in the same hour should be counted together."""
        result = aggregate_crashes_by_hour(sample_crash_df, include_zero_hours=False)

        # 10:15 and 10:45 are in the same hour (10:00)
        hour_10 = result[result["hour_timestamp"] == pd.Timestamp("2024-01-01 10:00")]
        assert hour_10["crash_count"].iloc[0] == 2

    def test_includes_zero_count_hours(self, sample_crash_df: pd.DataFrame) -> None:
        """When include_zero_hours=True, missing hours should have count=0."""
        result = aggregate_crashes_by_hour(sample_crash_df, include_zero_hours=True)

        # Hour 11 should exist with 0 crashes
        hour_11 = result[result["hour_timestamp"] == pd.Timestamp("2024-01-01 11:00")]
        assert len(hour_11) == 1
        assert hour_11["crash_count"].iloc[0] == 0

    def test_excludes_zero_count_hours_when_disabled(
        self, sample_crash_df: pd.DataFrame
    ) -> None:
        """When include_zero_hours=False, missing hours should not appear."""
        result = aggregate_crashes_by_hour(sample_crash_df, include_zero_hours=False)

        # Hour 11 should not exist
        hour_11 = result[result["hour_timestamp"] == pd.Timestamp("2024-01-01 11:00")]
        assert len(hour_11) == 0

    def test_output_columns(self, sample_crash_df: pd.DataFrame) -> None:
        """Output should have exactly hour_timestamp and crash_count columns."""
        result = aggregate_crashes_by_hour(sample_crash_df)
        assert list(result.columns) == ["hour_timestamp", "crash_count"]

    def test_crash_count_is_integer(self, sample_crash_df: pd.DataFrame) -> None:
        """Crash counts should be integers."""
        result = aggregate_crashes_by_hour(sample_crash_df, include_zero_hours=True)
        assert result["crash_count"].dtype in [np.int64, np.int32, int]


# =============================================================================
# Lag Feature Tests
# =============================================================================


class TestAddWeatherLagFeatures:
    """Tests for add_weather_lag_features function."""

    def test_creates_lag_columns(self, sample_hourly_df: pd.DataFrame) -> None:
        """Should create columns for each weather column × lag combination."""
        result = add_weather_lag_features(
            sample_hourly_df,
            weather_columns=["Air_Temperature", "Rain_Intensity"],
            lag_hours=[1, 2],
        )

        expected_cols = [
            "Air_Temperature_lag_1h",
            "Air_Temperature_lag_2h",
            "Rain_Intensity_lag_1h",
            "Rain_Intensity_lag_2h",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_lag_values_are_shifted_correctly(
        self, sample_hourly_df: pd.DataFrame
    ) -> None:
        """Lag values should be shifted by the correct number of hours."""
        result = add_weather_lag_features(
            sample_hourly_df,
            weather_columns=["Air_Temperature"],
            lag_hours=[1],
        )

        # Value at index 5 with lag_1h should equal value at index 4
        original_at_4 = sample_hourly_df["Air_Temperature"].iloc[4]
        lag_at_5 = result["Air_Temperature_lag_1h"].iloc[5]
        assert lag_at_5 == original_at_4

    def test_first_rows_have_nan_for_lags(
        self, sample_hourly_df: pd.DataFrame
    ) -> None:
        """First N rows should have NaN for lag features where N = lag hours."""
        result = add_weather_lag_features(
            sample_hourly_df,
            weather_columns=["Air_Temperature"],
            lag_hours=[1, 2, 3],
        )

        # First row should have NaN for all lag features
        assert pd.isna(result["Air_Temperature_lag_1h"].iloc[0])
        assert pd.isna(result["Air_Temperature_lag_2h"].iloc[0])
        assert pd.isna(result["Air_Temperature_lag_3h"].iloc[0])

        # Third row should have NaN only for lag_3h
        assert pd.notna(result["Air_Temperature_lag_1h"].iloc[2])
        assert pd.notna(result["Air_Temperature_lag_2h"].iloc[2])
        assert pd.isna(result["Air_Temperature_lag_3h"].iloc[2])

    def test_skips_missing_columns(self, sample_hourly_df: pd.DataFrame) -> None:
        """Should skip weather columns that don't exist in DataFrame."""
        result = add_weather_lag_features(
            sample_hourly_df,
            weather_columns=["Air_Temperature", "NonExistentColumn"],
            lag_hours=[1],
        )

        assert "Air_Temperature_lag_1h" in result.columns
        assert "NonExistentColumn_lag_1h" not in result.columns

    def test_uses_default_columns_and_lags(
        self, sample_hourly_df: pd.DataFrame
    ) -> None:
        """Should use default weather columns and lag hours when not specified."""
        result = add_weather_lag_features(sample_hourly_df)

        # Check default lag hours (1, 2, 3) for a column in defaults
        for lag in [1, 2, 3]:
            assert f"Air_Temperature_lag_{lag}h" in result.columns


# =============================================================================
# Cyclical Time Feature Tests
# =============================================================================


class TestAddCyclicalTimeFeatures:
    """Tests for add_cyclical_time_features function."""

    def test_adds_all_cyclical_columns(self, sample_hourly_df: pd.DataFrame) -> None:
        """Should add sin/cos columns for hour, day of week, and month."""
        result = add_cyclical_time_features(sample_hourly_df)

        expected_cols = [
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_hour_encoding_wraps_correctly(self) -> None:
        """Hour 0 and hour 23 should have similar encodings (cyclical)."""
        df = pd.DataFrame({
            "hour_timestamp": [
                pd.Timestamp("2024-01-01 00:00"),
                pd.Timestamp("2024-01-01 23:00"),
            ]
        })
        result = add_cyclical_time_features(df)

        # Sin values should be close (both near 0 for 0 and 23 hours)
        # 0h: sin(0) = 0, 23h: sin(2π*23/24) ≈ -0.26
        # Instead check that the angle distance is small
        hour_0_angle = np.arctan2(result["hour_sin"].iloc[0], result["hour_cos"].iloc[0])
        hour_23_angle = np.arctan2(result["hour_sin"].iloc[1], result["hour_cos"].iloc[1])

        # Angular distance should be about 1 hour worth (2π/24 ≈ 0.26 radians)
        angle_diff = abs(hour_0_angle - hour_23_angle)
        # Account for wraparound
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        assert angle_diff < 0.3  # Close enough (1 hour difference)

    def test_values_bounded_between_negative_one_and_one(
        self, sample_hourly_df: pd.DataFrame
    ) -> None:
        """All cyclical values should be between -1 and 1."""
        result = add_cyclical_time_features(sample_hourly_df)

        cyclical_cols = [
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
        ]
        for col in cyclical_cols:
            assert result[col].min() >= -1.0
            assert result[col].max() <= 1.0


# =============================================================================
# Time-Based Split Tests
# =============================================================================


class TestTimeSplit:
    """Tests for time-based data splitting."""

    def test_time_split_maintains_order(self) -> None:
        """Time-based split should maintain temporal order."""
        n_samples = 100
        features = np.arange(n_samples).reshape(-1, 1).astype(np.float32)
        labels = np.arange(n_samples).astype(np.float32)

        config = TensorConfig(
            target_column="test",
            task_type="regression",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            split_by_time=True,
        )

        train_X, train_y, val_X, val_y, test_X, test_y = _split_data(
            features, labels, config
        )

        # All train values should be less than all val values
        assert train_X.max() < val_X.min()
        # All val values should be less than all test values
        assert val_X.max() < test_X.min()

    def test_random_split_shuffles_data(self) -> None:
        """Random split should shuffle the data."""
        n_samples = 100
        features = np.arange(n_samples).reshape(-1, 1).astype(np.float32)
        labels = np.arange(n_samples).astype(np.float32)

        config = TensorConfig(
            target_column="test",
            task_type="regression",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            split_by_time=False,  # Random split
        )

        train_X, train_y, val_X, val_y, test_X, test_y = _split_data(
            features, labels, config
        )

        # With random split, train should contain values from throughout the range
        # Not just the first 70 values
        assert train_X.max() > 70  # Should have some high values in train

    def test_split_sizes_are_correct(self) -> None:
        """Split sizes should match the configured ratios."""
        n_samples = 100
        features = np.arange(n_samples).reshape(-1, 1).astype(np.float32)
        labels = np.arange(n_samples).astype(np.float32)

        config = TensorConfig(
            target_column="test",
            task_type="regression",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            split_by_time=True,
        )

        train_X, train_y, val_X, val_y, test_X, test_y = _split_data(
            features, labels, config
        )

        assert len(train_X) == 70
        assert len(val_X) == 15
        assert len(test_X) == 15


# =============================================================================
# Integration Tests
# =============================================================================


class TestMergeHourlyCrashesWithWeather:
    """Tests for merge_hourly_crashes_with_weather function."""

    def test_merges_correctly(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
    ) -> None:
        """Should merge crash counts with weather data."""
        result = merge_hourly_crashes_with_weather(
            sample_crash_df,
            sample_weather_df,
            include_zero_hours=True,
        )

        assert "crash_count" in result.columns
        assert "Air_Temperature" in result.columns
        assert "hour_timestamp" in result.columns


class TestPrepareHourlyData:
    """Tests for the full prepare_hourly_data pipeline."""

    def test_full_pipeline_produces_expected_columns(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
    ) -> None:
        """Full pipeline should produce all expected columns."""
        result = prepare_hourly_data(
            crash_df=sample_crash_df,
            weather_df=sample_weather_df,
            weather_lag_columns=["Air_Temperature", "Rain_Intensity"],
            lag_hours=[1, 2],
            include_zero_hours=True,
        )

        expected_cols = [
            "hour_timestamp",
            "crash_count",
            "Air_Temperature",
            "Rain_Intensity",
            "Air_Temperature_lag_1h",
            "Air_Temperature_lag_2h",
            "Rain_Intensity_lag_1h",
            "Rain_Intensity_lag_2h",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_pipeline_drops_nan_rows(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
    ) -> None:
        """Pipeline should drop rows with NaN from lag features."""
        result = prepare_hourly_data(
            crash_df=sample_crash_df,
            weather_df=sample_weather_df,
            lag_hours=[1, 2],
            include_zero_hours=True,
        )

        # Check no NaN values in lag columns that exist
        lag_cols = [c for c in result.columns if "_lag_" in c]
        for col in lag_cols:
            assert result[col].isna().sum() == 0, f"Column {col} has NaN values"

    def test_pipeline_output_is_sorted_by_time(
        self,
        sample_crash_df: pd.DataFrame,
        sample_weather_df: pd.DataFrame,
    ) -> None:
        """Output should be sorted by timestamp."""
        result = prepare_hourly_data(
            crash_df=sample_crash_df,
            weather_df=sample_weather_df,
        )

        timestamps = result["hour_timestamp"].values
        assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))


class TestGetHourlyFeatureColumns:
    """Tests for get_hourly_feature_columns helper."""

    def test_returns_correct_columns(self) -> None:
        """Should return expected feature column lists."""
        cat_cols, num_cols = get_hourly_feature_columns(
            weather_lag_columns=["Air_Temperature"],
            lag_hours=[1, 2],
            include_current_weather=True,
        )

        assert len(cat_cols) == 0  # No categorical for hourly
        assert "hour_sin" in num_cols
        assert "Air_Temperature" in num_cols
        assert "Air_Temperature_lag_1h" in num_cols
        assert "Air_Temperature_lag_2h" in num_cols


class TestHourlyCrashCountConfig:
    """Tests for the HOURLY_CRASH_COUNT_CONFIG preset."""

    def test_config_is_regression(self) -> None:
        """Config should be set up for regression."""
        assert HOURLY_CRASH_COUNT_CONFIG.task_type == "regression"

    def test_config_uses_time_split(self) -> None:
        """Config should use time-based splitting."""
        assert HOURLY_CRASH_COUNT_CONFIG.split_by_time is True

    def test_config_has_expected_features(self) -> None:
        """Config should have cyclical time and weather lag features."""
        features = HOURLY_CRASH_COUNT_CONFIG.feature_columns
        assert features is not None

        # Check for cyclical time
        assert "hour_sin" in features
        assert "hour_cos" in features

        # Check for weather lags
        assert "Air_Temperature_lag_1h" in features
        assert "Rain_Intensity_lag_3h" in features
