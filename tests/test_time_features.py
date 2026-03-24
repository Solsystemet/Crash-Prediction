"""Tests for time feature configuration and generation.

Tests for data_preparation/time_features.py module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_preparation.time_features import (
    DEFAULT_TIME_CONFIG,
    FULL_TIME_CONFIG,
    TimeFeatureConfig,
    add_binary_time_features,
    add_cyclical_time_features,
    add_time_features,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame with hourly timestamps spanning various times."""
    return pd.DataFrame({
        "hour_timestamp": pd.to_datetime([
            "2024-01-01 02:00",   # Night, Monday
            "2024-01-01 08:00",   # Morning rush, Monday
            "2024-01-01 14:00",   # Afternoon, Monday
            "2024-01-01 17:00",   # Evening rush, Monday
            "2024-01-01 23:00",   # Night, Monday
            "2024-01-06 12:00",   # Noon, Saturday (weekend)
            "2024-01-07 09:00",   # Morning, Sunday (weekend)
            "2024-06-15 00:00",   # Midnight, Saturday (summer)
        ])
    })


@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Minimal DataFrame for simple tests."""
    return pd.DataFrame({
        "hour_timestamp": pd.date_range("2024-01-01", periods=24, freq="h")
    })


# =============================================================================
# TimeFeatureConfig Tests
# =============================================================================


class TestTimeFeatureConfig:
    """Tests for TimeFeatureConfig dataclass."""

    def test_default_config_values(self) -> None:
        """Default config should enable cyclical features, disable binary."""
        config = TimeFeatureConfig()
        assert config.include_hour_cycle is True
        assert config.include_day_of_week_cycle is True
        assert config.include_month_cycle is True
        assert config.include_is_night is False
        assert config.include_is_rush_hour is False
        assert config.include_is_weekend is False

    def test_default_thresholds(self) -> None:
        """Default thresholds should be set correctly."""
        config = TimeFeatureConfig()
        assert config.night_start == 22
        assert config.night_end == 6
        assert config.rush_morning_start == 7
        assert config.rush_morning_end == 9
        assert config.rush_evening_start == 16
        assert config.rush_evening_end == 19

    def test_custom_thresholds(self) -> None:
        """Custom thresholds should be respected."""
        config = TimeFeatureConfig(
            night_start=23,
            night_end=5,
            rush_morning_start=6,
            rush_morning_end=10,
            rush_evening_start=15,
            rush_evening_end=20,
        )
        assert config.night_start == 23
        assert config.night_end == 5
        assert config.rush_morning_start == 6
        assert config.rush_morning_end == 10

    def test_invalid_hour_range_raises_error(self) -> None:
        """Hours outside 0-23 should raise ValueError."""
        with pytest.raises(ValueError, match="night_start must be between 0 and 23"):
            TimeFeatureConfig(night_start=24)

        with pytest.raises(ValueError, match="rush_evening_end must be between 0 and 23"):
            TimeFeatureConfig(rush_evening_end=25)

    def test_invalid_rush_hour_ordering_raises_error(self) -> None:
        """Rush hour start >= end should raise ValueError."""
        with pytest.raises(ValueError, match="rush_morning_start.*must be less than"):
            TimeFeatureConfig(rush_morning_start=10, rush_morning_end=8)

        with pytest.raises(ValueError, match="rush_evening_start.*must be less than"):
            TimeFeatureConfig(rush_evening_start=19, rush_evening_end=16)

    def test_get_feature_columns_all_enabled(self) -> None:
        """get_feature_columns should return all columns when all enabled."""
        config = TimeFeatureConfig(
            include_hour_cycle=True,
            include_day_of_week_cycle=True,
            include_month_cycle=True,
            include_is_night=True,
            include_is_rush_hour=True,
            include_is_weekend=True,
        )
        columns = config.get_feature_columns()
        expected = [
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
            "is_night", "is_rush_hour", "is_weekend",
        ]
        assert columns == expected

    def test_get_feature_columns_selective(self) -> None:
        """get_feature_columns should return only enabled columns."""
        config = TimeFeatureConfig(
            include_hour_cycle=True,
            include_day_of_week_cycle=False,
            include_month_cycle=False,
            include_is_night=True,
            include_is_rush_hour=False,
            include_is_weekend=True,
        )
        columns = config.get_feature_columns()
        assert columns == ["hour_sin", "hour_cos", "is_night", "is_weekend"]

    def test_get_feature_columns_none_enabled(self) -> None:
        """get_feature_columns should return empty list when all disabled."""
        config = TimeFeatureConfig(
            include_hour_cycle=False,
            include_day_of_week_cycle=False,
            include_month_cycle=False,
            include_is_night=False,
            include_is_rush_hour=False,
            include_is_weekend=False,
        )
        assert config.get_feature_columns() == []


class TestPresetConfigs:
    """Tests for preset configuration instances."""

    def test_default_time_config(self) -> None:
        """DEFAULT_TIME_CONFIG should match original behavior."""
        assert DEFAULT_TIME_CONFIG.include_hour_cycle is True
        assert DEFAULT_TIME_CONFIG.include_day_of_week_cycle is True
        assert DEFAULT_TIME_CONFIG.include_month_cycle is True
        assert DEFAULT_TIME_CONFIG.include_is_night is False
        assert DEFAULT_TIME_CONFIG.include_is_rush_hour is False
        assert DEFAULT_TIME_CONFIG.include_is_weekend is False

    def test_full_time_config(self) -> None:
        """FULL_TIME_CONFIG should have all features enabled."""
        assert FULL_TIME_CONFIG.include_hour_cycle is True
        assert FULL_TIME_CONFIG.include_day_of_week_cycle is True
        assert FULL_TIME_CONFIG.include_month_cycle is True
        assert FULL_TIME_CONFIG.include_is_night is True
        assert FULL_TIME_CONFIG.include_is_rush_hour is True
        assert FULL_TIME_CONFIG.include_is_weekend is True


# =============================================================================
# Cyclical Feature Tests
# =============================================================================


class TestCyclicalTimeFeatures:
    """Tests for add_cyclical_time_features function."""

    def test_adds_hour_cycle_features(self, minimal_df: pd.DataFrame) -> None:
        """Should add hour_sin and hour_cos columns."""
        config = TimeFeatureConfig(
            include_hour_cycle=True,
            include_day_of_week_cycle=False,
            include_month_cycle=False,
        )
        result = add_cyclical_time_features(minimal_df, config=config)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "day_of_week_sin" not in result.columns
        assert "month_sin" not in result.columns

    def test_adds_day_of_week_cycle_features(self, minimal_df: pd.DataFrame) -> None:
        """Should add day_of_week_sin and day_of_week_cos columns."""
        config = TimeFeatureConfig(
            include_hour_cycle=False,
            include_day_of_week_cycle=True,
            include_month_cycle=False,
        )
        result = add_cyclical_time_features(minimal_df, config=config)
        assert "day_of_week_sin" in result.columns
        assert "day_of_week_cos" in result.columns
        assert "hour_sin" not in result.columns

    def test_adds_month_cycle_features(self, minimal_df: pd.DataFrame) -> None:
        """Should add month_sin and month_cos columns."""
        config = TimeFeatureConfig(
            include_hour_cycle=False,
            include_day_of_week_cycle=False,
            include_month_cycle=True,
        )
        result = add_cyclical_time_features(minimal_df, config=config)
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns
        assert "hour_sin" not in result.columns

    def test_hour_cycle_values_at_boundaries(self) -> None:
        """Hour cycle should have correct values at key hours."""
        df = pd.DataFrame({
            "hour_timestamp": pd.to_datetime([
                "2024-01-01 00:00",  # Midnight
                "2024-01-01 06:00",  # 6 AM
                "2024-01-01 12:00",  # Noon
                "2024-01-01 18:00",  # 6 PM
            ])
        })
        config = TimeFeatureConfig(
            include_hour_cycle=True,
            include_day_of_week_cycle=False,
            include_month_cycle=False,
        )
        result = add_cyclical_time_features(df, config=config)

        # At midnight (hour=0): sin=0, cos=1
        assert np.isclose(result["hour_sin"].iloc[0], 0, atol=1e-10)
        assert np.isclose(result["hour_cos"].iloc[0], 1, atol=1e-10)

        # At 6 AM (hour=6): sin=1, cos=0
        assert np.isclose(result["hour_sin"].iloc[1], 1, atol=1e-10)
        assert np.isclose(result["hour_cos"].iloc[1], 0, atol=1e-10)

        # At noon (hour=12): sin=0, cos=-1
        assert np.isclose(result["hour_sin"].iloc[2], 0, atol=1e-10)
        assert np.isclose(result["hour_cos"].iloc[2], -1, atol=1e-10)

        # At 6 PM (hour=18): sin=-1, cos=0
        assert np.isclose(result["hour_sin"].iloc[3], -1, atol=1e-10)
        assert np.isclose(result["hour_cos"].iloc[3], 0, atol=1e-10)

    def test_default_config_used_when_none(self, minimal_df: pd.DataFrame) -> None:
        """Should use DEFAULT_TIME_CONFIG when config is None."""
        result = add_cyclical_time_features(minimal_df, config=None)
        # Default has all cyclical enabled
        assert "hour_sin" in result.columns
        assert "day_of_week_sin" in result.columns
        assert "month_sin" in result.columns

    def test_does_not_modify_original_df(self, minimal_df: pd.DataFrame) -> None:
        """Should not modify the original DataFrame."""
        original_columns = minimal_df.columns.tolist()
        add_cyclical_time_features(minimal_df, config=DEFAULT_TIME_CONFIG)
        assert minimal_df.columns.tolist() == original_columns


# =============================================================================
# Binary Feature Tests
# =============================================================================


class TestBinaryTimeFeatures:
    """Tests for add_binary_time_features function."""

    def test_is_night_default_thresholds(self, sample_df: pd.DataFrame) -> None:
        """is_night should be 1 for hours 22-23 and 0-5 (default)."""
        config = TimeFeatureConfig(include_is_night=True)
        result = add_binary_time_features(sample_df, config=config)

        # Index mapping for sample_df:
        # 0: 02:00 (night), 1: 08:00, 2: 14:00, 3: 17:00, 4: 23:00 (night)
        # 5: 12:00, 6: 09:00, 7: 00:00 (night)
        expected_night = [1, 0, 0, 0, 1, 0, 0, 1]
        assert result["is_night"].tolist() == expected_night

    def test_is_night_custom_thresholds(self) -> None:
        """is_night should respect custom thresholds."""
        df = pd.DataFrame({
            "hour_timestamp": pd.to_datetime([
                "2024-01-01 22:00",  # Not night with custom threshold
                "2024-01-01 23:00",  # Night
                "2024-01-01 04:00",  # Night
                "2024-01-01 05:00",  # Not night
            ])
        })
        config = TimeFeatureConfig(
            include_is_night=True,
            night_start=23,
            night_end=5,
        )
        result = add_binary_time_features(df, config=config)
        assert result["is_night"].tolist() == [0, 1, 1, 0]

    def test_is_night_same_day_range(self) -> None:
        """is_night should handle non-wrapping ranges (e.g., 0-6)."""
        df = pd.DataFrame({
            "hour_timestamp": pd.to_datetime([
                "2024-01-01 00:00",  # Night
                "2024-01-01 03:00",  # Night
                "2024-01-01 06:00",  # Not night (exclusive end)
                "2024-01-01 23:00",  # Not night
            ])
        })
        config = TimeFeatureConfig(
            include_is_night=True,
            night_start=0,
            night_end=6,
        )
        result = add_binary_time_features(df, config=config)
        assert result["is_night"].tolist() == [1, 1, 0, 0]

    def test_is_rush_hour_default_thresholds(self, sample_df: pd.DataFrame) -> None:
        """is_rush_hour should be 1 for 7-9 AM and 4-7 PM (default)."""
        config = TimeFeatureConfig(include_is_rush_hour=True)
        result = add_binary_time_features(sample_df, config=config)

        # Index mapping: 08:00 (rush), 17:00 (rush)
        expected_rush = [0, 1, 0, 1, 0, 0, 0, 0]
        assert result["is_rush_hour"].tolist() == expected_rush

    def test_is_rush_hour_custom_thresholds(self) -> None:
        """is_rush_hour should respect custom thresholds."""
        df = pd.DataFrame({
            "hour_timestamp": pd.to_datetime([
                "2024-01-01 06:00",  # Morning rush (custom)
                "2024-01-01 07:00",  # Morning rush (custom)
                "2024-01-01 10:00",  # Not rush
                "2024-01-01 15:00",  # Evening rush (custom)
                "2024-01-01 20:00",  # Not rush (exclusive end)
            ])
        })
        config = TimeFeatureConfig(
            include_is_rush_hour=True,
            rush_morning_start=6,
            rush_morning_end=10,
            rush_evening_start=15,
            rush_evening_end=20,
        )
        result = add_binary_time_features(df, config=config)
        assert result["is_rush_hour"].tolist() == [1, 1, 0, 1, 0]

    def test_is_weekend(self, sample_df: pd.DataFrame) -> None:
        """is_weekend should be 1 for Saturday and Sunday."""
        config = TimeFeatureConfig(include_is_weekend=True)
        result = add_binary_time_features(sample_df, config=config)

        # Index 5, 6, 7 are Saturday/Sunday
        expected_weekend = [0, 0, 0, 0, 0, 1, 1, 1]
        assert result["is_weekend"].tolist() == expected_weekend

    def test_no_binary_features_when_disabled(self, sample_df: pd.DataFrame) -> None:
        """No binary columns should be added when all disabled."""
        config = TimeFeatureConfig(
            include_is_night=False,
            include_is_rush_hour=False,
            include_is_weekend=False,
        )
        result = add_binary_time_features(sample_df, config=config)
        assert "is_night" not in result.columns
        assert "is_rush_hour" not in result.columns
        assert "is_weekend" not in result.columns

    def test_does_not_modify_original_df(self, sample_df: pd.DataFrame) -> None:
        """Should not modify the original DataFrame."""
        original_columns = sample_df.columns.tolist()
        config = TimeFeatureConfig(
            include_is_night=True,
            include_is_rush_hour=True,
            include_is_weekend=True,
        )
        add_binary_time_features(sample_df, config=config)
        assert sample_df.columns.tolist() == original_columns


# =============================================================================
# Combined Feature Tests
# =============================================================================


class TestAddTimeFeatures:
    """Tests for the combined add_time_features function."""

    def test_adds_all_enabled_features(self, sample_df: pd.DataFrame) -> None:
        """Should add both cyclical and binary features based on config."""
        config = TimeFeatureConfig(
            include_hour_cycle=True,
            include_day_of_week_cycle=True,
            include_month_cycle=True,
            include_is_night=True,
            include_is_rush_hour=True,
            include_is_weekend=True,
        )
        result = add_time_features(sample_df, config=config)

        expected_columns = config.get_feature_columns()
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_selective_features(self, sample_df: pd.DataFrame) -> None:
        """Should only add selected features."""
        config = TimeFeatureConfig(
            include_hour_cycle=True,
            include_day_of_week_cycle=False,
            include_month_cycle=False,
            include_is_night=False,
            include_is_rush_hour=False,
            include_is_weekend=True,
        )
        result = add_time_features(sample_df, config=config)

        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "is_weekend" in result.columns
        assert "day_of_week_sin" not in result.columns
        assert "month_sin" not in result.columns
        assert "is_night" not in result.columns

    def test_preserves_original_columns(self, sample_df: pd.DataFrame) -> None:
        """Should preserve all original columns."""
        result = add_time_features(sample_df, config=FULL_TIME_CONFIG)
        assert "hour_timestamp" in result.columns

    def test_custom_timestamp_column(self) -> None:
        """Should work with custom timestamp column name."""
        df = pd.DataFrame({
            "my_time": pd.date_range("2024-01-01", periods=5, freq="h")
        })
        config = TimeFeatureConfig(include_hour_cycle=True)
        result = add_time_features(df, timestamp_col="my_time", config=config)
        assert "hour_sin" in result.columns
        assert "my_time" in result.columns

    def test_generated_columns_match_get_feature_columns(
        self, sample_df: pd.DataFrame
    ) -> None:
        """Generated columns should match what get_feature_columns returns."""
        config = TimeFeatureConfig(
            include_hour_cycle=True,
            include_day_of_week_cycle=False,
            include_month_cycle=True,
            include_is_night=True,
            include_is_rush_hour=False,
            include_is_weekend=True,
        )
        result = add_time_features(sample_df, config=config)

        expected_columns = config.get_feature_columns()
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Also verify no unexpected time feature columns
        time_feature_set = {
            "hour_sin", "hour_cos",
            "day_of_week_sin", "day_of_week_cos",
            "month_sin", "month_cos",
            "is_night", "is_rush_hour", "is_weekend",
        }
        actual_time_cols = [c for c in result.columns if c in time_feature_set]
        assert sorted(actual_time_cols) == sorted(expected_columns)
