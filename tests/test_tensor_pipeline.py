"""Tests for tensor data preparation pipeline."""

import numpy as np
import pandas as pd
import pytest
import torch

from data_preparation.encoders import (
    CategoricalEncoder,
    EncoderRegistry,
    NumericalScaler,
)
from data_preparation.prepare_tensor_data import prepare_tensor_data
from data_preparation.tensor_config import (
    MINIMAL_TEST_CONFIG,
    SEVERITY_PREDICTION_CONFIG,
    TensorConfig,
)
from data_preparation.tensor_dataset import CrashTensorDataset


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "WEATHER_CONDITION": ["CLEAR", "RAIN", "SNOW", "CLEAR", "RAIN"] * 20,
            "LIGHTING_CONDITION": ["DAYLIGHT", "DARKNESS", "DAYLIGHT", "DUSK", "DARKNESS"]
            * 20,
            "POSTED_SPEED_LIMIT": [30, 45, 25, 35, 40] * 20,
            "CRASH_HOUR": [8, 14, 22, 6, 18] * 20,
            "CRASH_MONTH": [1, 6, 12, 3, 9] * 20,
            "MOST_SEVERE_INJURY": [
                "NO INDICATION OF INJURY",
                "NONINCAPACITATING INJURY",
                "INCAPACITATING INJURY",
                "FATAL",
                "NO INDICATION OF INJURY",
            ]
            * 20,
        }
    )


@pytest.fixture
def sample_config() -> TensorConfig:
    """Create a sample config for testing."""
    return TensorConfig(
        target_column="MOST_SEVERE_INJURY",
        feature_columns=["WEATHER_CONDITION", "POSTED_SPEED_LIMIT", "CRASH_HOUR"],
        categorical_columns=["WEATHER_CONDITION"],
        numerical_columns=["POSTED_SPEED_LIMIT", "CRASH_HOUR"],
        task_type="classification",
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_seed=42,
    )


# =============================================================================
# TensorConfig Tests
# =============================================================================


class TestTensorConfig:
    """Tests for TensorConfig dataclass."""

    def test_valid_config_creation(self) -> None:
        """Config with valid ratios should be created successfully."""
        config = TensorConfig(
            target_column="TARGET",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        assert config.target_column == "TARGET"

    def test_invalid_ratio_sum(self) -> None:
        """Config with ratios not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            TensorConfig(
                target_column="TARGET",
                train_ratio=0.5,
                val_ratio=0.2,
                test_ratio=0.2,
            )

    def test_negative_ratio(self) -> None:
        """Config with negative ratio should raise ValueError."""
        with pytest.raises(ValueError):
            TensorConfig(
                target_column="TARGET",
                train_ratio=-0.1,
                val_ratio=0.5,
                test_ratio=0.6,
            )

    def test_preset_configs_valid(self) -> None:
        """Preset configs should have valid ratios."""
        # These should not raise
        assert SEVERITY_PREDICTION_CONFIG.target_column == "MOST_SEVERE_INJURY"
        assert MINIMAL_TEST_CONFIG.target_column == "MOST_SEVERE_INJURY"


# =============================================================================
# CategoricalEncoder Tests
# =============================================================================


class TestCategoricalEncoder:
    """Tests for CategoricalEncoder."""

    def test_fit_transform(self) -> None:
        """Encoder should correctly transform categories to integers."""
        encoder = CategoricalEncoder(column_name="test")
        values = pd.Series(["A", "B", "C", "A", "B"])

        encoded = encoder.fit_transform(values)

        assert encoded.dtype == np.int64
        assert len(np.unique(encoded)) == 3  # 3 unique values

    def test_inverse_transform(self) -> None:
        """Inverse transform should recover original values."""
        encoder = CategoricalEncoder(column_name="test")
        values = pd.Series(["A", "B", "C", "A", "B"])

        encoded = encoder.fit_transform(values)
        decoded = encoder.inverse_transform(encoded)

        np.testing.assert_array_equal(decoded, values.values)

    def test_unknown_category_handling(self) -> None:
        """Unknown categories should be mapped to unknown_value."""
        encoder = CategoricalEncoder(column_name="test", unknown_value="UNKNOWN")
        encoder.fit(pd.Series(["A", "B", "C"]))

        # Transform with unknown category
        encoded = encoder.transform(pd.Series(["A", "D", "B"]))

        # "D" should be mapped to same index as "UNKNOWN"
        assert encoder.classes_ is not None
        unknown_idx = np.where(encoder.classes_ == "UNKNOWN")[0][0]
        assert encoded[1] == unknown_idx

    def test_missing_value_handling(self) -> None:
        """Missing values should be filled with unknown_value."""
        encoder = CategoricalEncoder(column_name="test", unknown_value="UNKNOWN")
        values = pd.Series(["A", None, "B"])

        encoded = encoder.fit_transform(values)

        assert encoder.classes_ is not None
        unknown_idx = np.where(encoder.classes_ == "UNKNOWN")[0][0]
        assert encoded[1] == unknown_idx

    def test_serialization(self) -> None:
        """Encoder should be serializable and deserializable."""
        encoder = CategoricalEncoder(column_name="test")
        encoder.fit(pd.Series(["A", "B", "C"]))

        data = encoder.to_dict()
        restored = CategoricalEncoder.from_dict(data)

        assert restored.column_name == encoder.column_name
        np.testing.assert_array_equal(restored.classes_, encoder.classes_)


# =============================================================================
# NumericalScaler Tests
# =============================================================================


class TestNumericalScaler:
    """Tests for NumericalScaler."""

    def test_fit_transform(self) -> None:
        """Scaler should standardize values."""
        scaler = NumericalScaler(column_names=["col1", "col2"])
        df = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [10.0, 20.0, 30.0]})

        scaled = scaler.fit_transform(df)

        # Scaled values should have mean ~0 and std ~1
        assert scaled.shape == (3, 2)
        np.testing.assert_almost_equal(scaled.mean(axis=0), [0, 0], decimal=5)

    def test_inverse_transform(self) -> None:
        """Inverse transform should recover original values."""
        scaler = NumericalScaler(column_names=["col1"])
        df = pd.DataFrame({"col1": [1.0, 2.0, 3.0, 4.0, 5.0]})

        scaled = scaler.fit_transform(df)
        unscaled = scaler.inverse_transform(scaled)

        np.testing.assert_almost_equal(unscaled.flatten(), df["col1"].values, decimal=5)

    def test_serialization(self) -> None:
        """Scaler should be serializable and deserializable."""
        scaler = NumericalScaler(column_names=["col1", "col2"])
        df = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [10.0, 20.0, 30.0]})
        scaler.fit(df)

        data = scaler.to_dict()
        restored = NumericalScaler.from_dict(data)

        assert restored.column_names == scaler.column_names
        np.testing.assert_array_equal(restored.mean_, scaler.mean_)


# =============================================================================
# CrashTensorDataset Tests
# =============================================================================


class TestCrashTensorDataset:
    """Tests for CrashTensorDataset."""

    def test_dataset_creation(self) -> None:
        """Dataset should be created with correct shapes."""
        features = torch.randn(100, 10)
        labels = torch.randint(0, 5, (100,))

        dataset = CrashTensorDataset(features, labels)

        assert len(dataset) == 100
        assert dataset.num_features == 10
        assert dataset.num_samples == 100

    def test_getitem(self) -> None:
        """Dataset should return correct items."""
        features = torch.randn(100, 10)
        labels = torch.randint(0, 5, (100,))

        dataset = CrashTensorDataset(features, labels)
        feat, label = dataset[0]

        assert feat.shape == (10,)
        assert label.shape == ()

    def test_mismatched_shapes_raises(self) -> None:
        """Mismatched feature/label counts should raise ValueError."""
        features = torch.randn(100, 10)
        labels = torch.randint(0, 5, (50,))  # Wrong size

        with pytest.raises(ValueError, match="same number of samples"):
            CrashTensorDataset(features, labels)

    def test_label_distribution(self) -> None:
        """Label distribution should be computed correctly."""
        features = torch.randn(10, 5)
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])

        dataset = CrashTensorDataset(features, labels, task_type="classification")
        dist = dataset.get_label_distribution()

        assert dist == {0: 3, 1: 2, 2: 4, 3: 1}


# =============================================================================
# Integration Tests
# =============================================================================


class TestPrepareTenosrData:
    """Integration tests for prepare_tensor_data."""

    def test_prepare_with_sample_data(
        self, sample_dataframe: pd.DataFrame, sample_config: TensorConfig
    ) -> None:
        """Pipeline should work with sample data."""
        result = prepare_tensor_data(sample_config, df=sample_dataframe)

        # Check datasets created
        assert result.train_dataset is not None
        assert result.val_dataset is not None
        assert result.test_dataset is not None

        # Check split ratios (approximately)
        total = (
            len(result.train_dataset)
            + len(result.val_dataset)
            + len(result.test_dataset)
        )
        assert total == len(sample_dataframe)

        train_ratio = len(result.train_dataset) / total
        assert 0.5 < train_ratio < 0.7  # Allow some variance

    def test_tensor_dtypes(
        self, sample_dataframe: pd.DataFrame, sample_config: TensorConfig
    ) -> None:
        """Tensors should have correct dtypes."""
        result = prepare_tensor_data(sample_config, df=sample_dataframe)

        # For classification
        assert result.train_dataset.features.dtype == torch.float32
        assert result.train_dataset.labels.dtype == torch.int64

    def test_encoder_registry_populated(
        self, sample_dataframe: pd.DataFrame, sample_config: TensorConfig
    ) -> None:
        """Encoder registry should be populated after preparation."""
        result = prepare_tensor_data(sample_config, df=sample_dataframe)

        registry = result.encoder_registry
        assert len(registry.categorical_encoders) == 1  # WEATHER_CONDITION
        assert registry.numerical_scaler is not None
        assert registry.target_encoder is not None
        assert registry.get_num_classes() > 0

    def test_config_change_works(self, sample_dataframe: pd.DataFrame) -> None:
        """Changing config should produce different results without errors."""
        config1 = TensorConfig(
            target_column="MOST_SEVERE_INJURY",
            feature_columns=["WEATHER_CONDITION", "POSTED_SPEED_LIMIT"],
            categorical_columns=["WEATHER_CONDITION"],
            numerical_columns=["POSTED_SPEED_LIMIT"],
        )
        config2 = TensorConfig(
            target_column="MOST_SEVERE_INJURY",
            feature_columns=["CRASH_HOUR", "CRASH_MONTH"],
            categorical_columns=[],
            numerical_columns=["CRASH_HOUR", "CRASH_MONTH"],
        )

        result1 = prepare_tensor_data(config1, df=sample_dataframe)
        result2 = prepare_tensor_data(config2, df=sample_dataframe)

        # Different features = different feature counts
        assert result1.train_dataset.num_features == 2  # 1 cat + 1 num
        assert result2.train_dataset.num_features == 2  # 0 cat + 2 num

    def test_regression_task(self, sample_dataframe: pd.DataFrame) -> None:
        """Regression task should produce float labels."""
        # Add a numerical target
        sample_dataframe["NUM_TARGET"] = np.random.random(len(sample_dataframe))

        config = TensorConfig(
            target_column="NUM_TARGET",
            feature_columns=["POSTED_SPEED_LIMIT"],
            categorical_columns=[],
            numerical_columns=["POSTED_SPEED_LIMIT"],
            task_type="regression",
        )

        result = prepare_tensor_data(config, df=sample_dataframe)

        assert result.train_dataset.labels.dtype == torch.float32
        assert result.encoder_registry.target_encoder is None


class TestEncoderRegistrySerialization:
    """Tests for EncoderRegistry serialization."""

    def test_round_trip(
        self, sample_dataframe: pd.DataFrame, sample_config: TensorConfig
    ) -> None:
        """Registry should survive serialization round-trip."""
        result = prepare_tensor_data(sample_config, df=sample_dataframe)

        # Serialize and deserialize
        data = result.encoder_registry.to_dict()
        restored = EncoderRegistry.from_dict(data)

        # Check properties match
        assert restored.feature_columns == result.encoder_registry.feature_columns
        assert restored.target_column == result.encoder_registry.target_column
        assert restored.task_type == result.encoder_registry.task_type
        assert restored.get_num_classes() == result.encoder_registry.get_num_classes()
