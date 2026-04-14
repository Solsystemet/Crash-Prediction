"""Tests for crash count regression pipeline.

Tests:
- Time series aggregation
- Feature engineering
- Model creation and forward pass
- Dataset creation
- Temporal splits
"""

import numpy as np
import pandas as pd
import pytest
import torch

from data_preparation.aggregate_time_series import (
    TimeSeriesConfig,
    aggregate_crashes_by_time,
    _fill_time_gaps,
)
from data_preparation.time_series_features import (
    FeatureConfig,
    create_lag_features,
    create_rolling_features,
    add_cyclical_time_encoding,
    engineer_time_series_features,
)
from training.regression.config import RegressionConfig
from training.regression.models import CrashCountMLP, ZoneAdjustmentMLP
from training.regression.dataset import (
    TimeSeriesDataset,
    create_temporal_splits,
)
from training.regression.evaluation import compute_metrics


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_crash_df():
    """Create sample crash DataFrame for testing."""
    np.random.seed(42)
    n = 1000

    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "CRASH_DATE": dates,
        "LATITUDE": np.random.uniform(41.6, 42.0, n),
        "LONGITUDE": np.random.uniform(-87.9, -87.5, n),
        "LOCATION_CLUSTER": np.random.randint(0, 5, n),
    })


@pytest.fixture
def sample_ts_df():
    """Create sample time series DataFrame."""
    dates = pd.date_range("2023-01-01", periods=500, freq="h")
    return pd.DataFrame({
        "timestamp": dates,
        "zone_id": [0] * 500,
        "crash_count": np.random.poisson(5, 500),
    })


# ============================================================================
# Aggregation Tests
# ============================================================================

class TestAggregation:
    """Tests for time series aggregation."""

    def test_aggregate_hourly(self, sample_crash_df):
        """Test hourly aggregation."""
        config = TimeSeriesConfig(granularity="hourly", fill_gaps=False)
        result = aggregate_crashes_by_time(sample_crash_df, config)

        assert "timestamp" in result.columns
        assert "crash_count" in result.columns
        assert result["crash_count"].sum() == len(sample_crash_df)

    def test_aggregate_daily(self, sample_crash_df):
        """Test daily aggregation."""
        config = TimeSeriesConfig(granularity="daily", fill_gaps=False)
        result = aggregate_crashes_by_time(sample_crash_df, config)

        assert len(result) <= len(sample_crash_df)
        assert result["crash_count"].sum() == len(sample_crash_df)

    def test_aggregate_with_zones(self, sample_crash_df):
        """Test aggregation with zone grouping."""
        config = TimeSeriesConfig(
            granularity="hourly",
            zone_col="LOCATION_CLUSTER",
            fill_gaps=False,
        )
        result = aggregate_crashes_by_time(sample_crash_df, config)

        assert "zone_id" in result.columns
        assert result["crash_count"].sum() == len(sample_crash_df)

    def test_fill_gaps(self):
        """Test gap filling in time series."""
        # Create data with gaps
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2023-01-01 00:00", "2023-01-01 03:00"]),
            "crash_count": [5, 3],
        })

        config = TimeSeriesConfig(granularity="hourly")
        result = _fill_time_gaps(df, config)

        # Should fill hours 1 and 2 with zeros
        assert len(result) == 4
        assert result[result["crash_count"] == 0].shape[0] == 2


# ============================================================================
# Feature Engineering Tests
# ============================================================================

class TestFeatureEngineering:
    """Tests for time series feature engineering."""

    def test_lag_features(self, sample_ts_df):
        """Test lag feature creation."""
        config = FeatureConfig(lag_periods=[1, 2, 3])
        result = create_lag_features(sample_ts_df, config)

        assert "lag_1" in result.columns
        assert "lag_2" in result.columns
        assert "lag_3" in result.columns

        # First row should have NaN in lag features
        assert pd.isna(result["lag_1"].iloc[0])

        # Lag values should match shifted target
        assert result["lag_1"].iloc[3] == result["crash_count"].iloc[2]

    def test_rolling_features(self, sample_ts_df):
        """Test rolling statistics."""
        config = FeatureConfig(rolling_windows=[3, 6])
        result = create_rolling_features(sample_ts_df, config)

        assert "roll_3_mean" in result.columns
        assert "roll_3_std" in result.columns
        assert "roll_6_max" in result.columns

    def test_cyclical_encoding(self, sample_ts_df):
        """Test cyclical time encoding."""
        config = FeatureConfig(granularity="hourly", include_cyclical=True)
        result = add_cyclical_time_encoding(sample_ts_df, config=config)

        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "dow_sin" in result.columns

        # Sin/cos should be in [-1, 1]
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_engineer_all_features(self, sample_ts_df):
        """Test full feature engineering pipeline."""
        config = FeatureConfig.for_granularity("hourly")
        result, feature_cols = engineer_time_series_features(
            sample_ts_df,
            config=config,
            group_col=None,
        )

        assert len(feature_cols) > 10  # Should have many features
        assert "crash_count" in result.columns
        assert "timestamp" in result.columns
        assert len(result) < len(sample_ts_df)  # Some rows dropped due to NaN


# ============================================================================
# Model Tests
# ============================================================================

class TestModels:
    """Tests for neural network models."""

    def test_crash_count_mlp_forward(self):
        """Test CrashCountMLP forward pass."""
        model = CrashCountMLP(num_features=20, hidden_sizes=(64, 32))

        x = torch.randn(10, 20)
        output = model(x)

        assert output.shape == (10,)

    def test_crash_count_mlp_predict(self):
        """Test CrashCountMLP predict method (non-negative)."""
        model = CrashCountMLP(num_features=20)

        x = torch.randn(10, 20)
        output = model.predict(x)

        assert (output >= 0).all()

    def test_zone_adjustment_mlp(self):
        """Test ZoneAdjustmentMLP forward pass."""
        model = ZoneAdjustmentMLP(num_features=20, hidden_sizes=(32, 16))

        x = torch.randn(10, 20)
        output = model(x)

        assert output.shape == (10,)
        # Adjustments can be negative
        

# ============================================================================
# Dataset Tests
# ============================================================================

class TestDataset:
    """Tests for dataset utilities."""

    def test_time_series_dataset(self):
        """Test TimeSeriesDataset creation."""
        features = np.random.randn(100, 20)
        targets = np.random.rand(100)
        zone_ids = np.random.randint(0, 5, 100)

        dataset = TimeSeriesDataset(features, targets, zone_ids)

        assert len(dataset) == 100
        assert dataset.num_features == 20
        assert dataset.num_zones == 5

        # Test indexing
        f, t, z = dataset[0]
        assert f.shape == (20,)
        assert t.shape == ()
        assert z.shape == ()

    def test_temporal_splits(self, sample_ts_df):
        """Test temporal data splitting."""
        train, val, test = create_temporal_splits(
            sample_ts_df,
            train_ratio=0.7,
            val_ratio=0.15,
        )

        # Check sizes
        assert len(train) == int(500 * 0.7)
        assert len(val) == int(500 * 0.85) - int(500 * 0.7)
        assert len(test) == 500 - int(500 * 0.85)

        # Check temporal ordering (train before val before test)
        assert train["timestamp"].max() <= val["timestamp"].min()
        assert val["timestamp"].max() <= test["timestamp"].min()


# ============================================================================
# Evaluation Tests
# ============================================================================

class TestEvaluation:
    """Tests for evaluation metrics."""

    def test_compute_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mae"] == 0
        assert metrics["rmse"] == 0
        assert metrics["r2"] == 1.0

    def test_compute_metrics_with_error(self):
        """Test metrics with prediction errors."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mae"] == 2.0
        assert metrics["rmse"] > 0
        assert 0 < metrics["r2"] < 1

    def test_compute_metrics_empty(self):
        """Test metrics with empty arrays."""
        metrics = compute_metrics(np.array([]), np.array([]))
        assert metrics == {}


# ============================================================================
# Config Tests
# ============================================================================

class TestConfig:
    """Tests for configuration classes."""

    def test_regression_config_defaults(self):
        """Test default configuration values."""
        config = RegressionConfig()

        assert config.time_granularity == "hourly"
        assert config.n_zones == 10
        assert config.epochs == 100

    def test_regression_config_validation(self):
        """Test configuration validation."""
        # Invalid train/val ratios
        with pytest.raises(ValueError):
            RegressionConfig(train_ratio=0.8, val_ratio=0.3)

        # Invalid n_zones
        with pytest.raises(ValueError):
            RegressionConfig(n_zones=0)

    def test_feature_config_for_granularity(self):
        """Test feature config factory."""
        hourly = FeatureConfig.for_granularity("hourly")
        daily = FeatureConfig.for_granularity("daily")
        weekly = FeatureConfig.for_granularity("weekly")

        # Hourly should have more lags
        assert len(hourly.lag_periods) > len(daily.lag_periods)
        assert len(daily.lag_periods) > len(weekly.lag_periods)
