"""Main tensor data preparation pipeline.

This module provides the core function for converting DataFrame data
to PyTorch tensors ready for training, with train/val/test splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from data_preparation.encoders import (
    CategoricalEncoder,
    EncoderRegistry,
    NumericalScaler,
)
from data_preparation.helpers.csv_loaders import get_traffic_crashes
from data_preparation.tensor_config import TensorConfig
from data_preparation.tensor_dataset import CrashTensorDataset


def map_injury_severity_to_3_classes(severity: str) -> str:
    """Map original 5 injury severity categories to 3 simplified categories.
    
    Mapping:
        - SEVERE: FATAL, INCAPACITATING INJURY
        - MINOR: NONINCAPACITATING INJURY, REPORTED, NOT EVIDENT
        - NO_INJURY: NO INDICATION OF INJURY
    
    Args:
        severity: Original injury severity label (or already mapped label).
        
    Returns:
        Mapped severity category (SEVERE, MINOR, or NO_INJURY).
        
    Note:
        The output labels are prefixed with numbers to ensure proper ordering
        in classification reports (SEVERE first, then MINOR, then NO_INJURY).
    """
    # Handle already-mapped values (idempotent)
    if severity in ("SEVERE", "MINOR", "NO_INJURY"):
        return severity
    
    # Map original 5 categories to 3 new categories
    # Using numeric prefixes ensures proper severity-based ordering
    if severity in ("FATAL", "INCAPACITATING INJURY"):
        return "SEVERE"
    elif severity in ("NONINCAPACITATING INJURY", "REPORTED, NOT EVIDENT"):
        return "MINOR"
    elif severity == "NO INDICATION OF INJURY":
        return "NO_INJURY"
    else:
        raise ValueError(f"Unknown injury severity: {severity}")


@dataclass
class TensorPipelineResult:
    """Result of tensor data preparation.

    Attributes:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        encoder_registry: Registry of all encoders/scalers used.
        config: Configuration used for preparation.
    """

    train_dataset: CrashTensorDataset
    val_dataset: CrashTensorDataset
    test_dataset: CrashTensorDataset
    encoder_registry: EncoderRegistry
    config: TensorConfig

    def save(self, path: Path | str) -> None:
        """Save the pipeline result to disk.

        Args:
            path: Path to save the checkpoint file (.pt).
        """
        path = Path(path)
        checkpoint = {
            "train_features": self.train_dataset.features,
            "train_labels": self.train_dataset.labels,
            "val_features": self.val_dataset.features,
            "val_labels": self.val_dataset.labels,
            "test_features": self.test_dataset.features,
            "test_labels": self.test_dataset.labels,
            "encoder_registry": self.encoder_registry.to_dict(),
            "config": {
                "target_column": self.config.target_column,
                "feature_columns": self.config.feature_columns,
                "categorical_columns": self.config.categorical_columns,
                "numerical_columns": self.config.numerical_columns,
                "task_type": self.config.task_type,
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
                "random_seed": self.config.random_seed,
                "fill_categorical_na": self.config.fill_categorical_na,
                "fill_numerical_na": self.config.fill_numerical_na,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path | str) -> TensorPipelineResult:
        """Load a pipeline result from disk.

        Args:
            path: Path to the checkpoint file (.pt).

        Returns:
            Loaded TensorPipelineResult.
        """
        path = Path(path)
        checkpoint: dict[str, Any] = torch.load(path, weights_only=False)

        config = TensorConfig(**checkpoint["config"])
        encoder_registry = EncoderRegistry.from_dict(checkpoint["encoder_registry"])

        train_dataset = CrashTensorDataset(
            features=checkpoint["train_features"],
            labels=checkpoint["train_labels"],
            task_type=config.task_type,
        )
        val_dataset = CrashTensorDataset(
            features=checkpoint["val_features"],
            labels=checkpoint["val_labels"],
            task_type=config.task_type,
        )
        test_dataset = CrashTensorDataset(
            features=checkpoint["test_features"],
            labels=checkpoint["test_labels"],
            task_type=config.task_type,
        )

        return cls(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            encoder_registry=encoder_registry,
            config=config,
        )


def prepare_tensor_data(
    config: TensorConfig,
    df: pd.DataFrame | None = None,
) -> TensorPipelineResult:
    """Prepare tensor data from traffic crashes dataset.

    This function:
    1. Loads the traffic crashes CSV (or uses provided DataFrame)
    2. Filters to selected feature and target columns
    3. Handles missing values
    4. Encodes categorical columns (label encoding)
    5. Scales numerical columns (standard scaling)
    6. Splits data into train/val/test sets
    7. Converts to PyTorch tensors

    Args:
        config: Configuration specifying features, target, and split ratios.
        df: Optional pre-loaded DataFrame. If None, loads from CSV.

    Returns:
        TensorPipelineResult containing datasets and encoders.
    """
    # Step 1: Load data
    if df is None:
        df = get_traffic_crashes()

    # Step 2: Determine columns to use
    target_col = config.target_column
    feature_cols = config.feature_columns
    if feature_cols is None:
        # Use all columns except target
        feature_cols = [c for c in df.columns if c != target_col]

    # Validate columns exist
    all_cols = feature_cols + [target_col]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Step 3: Filter to relevant columns and drop rows with missing target
    df_subset = df[all_cols].copy()
    df_subset = df_subset.dropna(subset=[target_col])
    
    # Step 3a: Filter out UNKNOWN injury severity entries
    if target_col == "MOST_SEVERE_INJURY":
        initial_count = len(df_subset)
        df_subset = df_subset[df_subset[target_col] != "UNKNOWN"].copy()
        filtered_count = initial_count - len(df_subset)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} UNKNOWN injury severity entries")
    
    # Step 3b: Map 5 injury categories to 3 simplified categories
    if target_col == "MOST_SEVERE_INJURY":
        df_subset[target_col] = df_subset[target_col].apply(map_injury_severity_to_3_classes)
        print(f"Mapped injury severities to 3 categories: {df_subset[target_col].value_counts().to_dict()}")

    # Step 4: Handle missing values in features
    df_subset = _fill_missing_values(df_subset, config)

    # Step 5: Create encoder registry and encode data
    encoder_registry = _create_encoder_registry(df_subset, config, feature_cols)
    features_array, labels_array = _encode_data(df_subset, config, encoder_registry)

    # Step 6: Split into train/val/test
    train_features, train_labels, val_features, val_labels, test_features, test_labels = (
        _split_data(features_array, labels_array, config)
    )

    # Step 7: Convert to tensors
    train_dataset = _create_dataset(train_features, train_labels, config)
    val_dataset = _create_dataset(val_features, val_labels, config)
    test_dataset = _create_dataset(test_features, test_labels, config)

    return TensorPipelineResult(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        encoder_registry=encoder_registry,
        config=config,
    )


def _fill_missing_values(df: pd.DataFrame, config: TensorConfig) -> pd.DataFrame:
    """Fill missing values in the DataFrame.

    Args:
        df: DataFrame with potential missing values.
        config: Configuration specifying fill strategies.

    Returns:
        DataFrame with missing values filled.
    """
    # Fill categorical columns
    for col in config.categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(config.fill_categorical_na)

    # Fill numerical columns
    for col in config.numerical_columns:
        if col in df.columns:
            if config.fill_numerical_na == "median":
                fill_value = df[col].median()
            else:
                fill_value = df[col].mean()
            df[col] = df[col].fillna(fill_value)

    return df


def _create_encoder_registry(
    df: pd.DataFrame,
    config: TensorConfig,
    feature_cols: list[str],
) -> EncoderRegistry:
    """Create and fit all encoders/scalers.

    Args:
        df: DataFrame to fit encoders on.
        config: Configuration specifying column types.
        feature_cols: List of feature column names.

    Returns:
        Fitted EncoderRegistry.
    """
    registry = EncoderRegistry(
        feature_columns=feature_cols,
        target_column=config.target_column,
        task_type=config.task_type,
    )

    # Fit categorical encoders for features
    for col in config.categorical_columns:
        if col in feature_cols:
            encoder = CategoricalEncoder(
                column_name=col,
                unknown_value=config.fill_categorical_na,
            )
            encoder.fit(df[col])
            registry.categorical_encoders[col] = encoder

    # Fit numerical scaler
    numerical_feature_cols = [c for c in config.numerical_columns if c in feature_cols]
    if numerical_feature_cols:
        scaler = NumericalScaler(column_names=numerical_feature_cols)
        scaler.fit(df)
        registry.numerical_scaler = scaler

    # Fit target encoder (for classification)
    if config.task_type == "classification":
        target_encoder = CategoricalEncoder(
            column_name=config.target_column,
            unknown_value=config.fill_categorical_na,
            add_unknown_class=False,  # Don't add UNKNOWN for target after filtering
        )
        
        # For MOST_SEVERE_INJURY, manually control class ordering
        if config.target_column == "MOST_SEVERE_INJURY":
            # Get unique values from data
            unique_vals = set(df[config.target_column].unique())
            # Define severity-based order: SEVERE -> MINOR -> NO_INJURY
            desired_order = ["SEVERE", "MINOR", "NO_INJURY"]
            ordered_classes = [c for c in desired_order if c in unique_vals]
            # Directly set classes without fit (which would alphabetically sort)
            target_encoder.classes_ = np.array(ordered_classes, dtype=np.str_)
            target_encoder._encoder.classes_ = target_encoder.classes_
        else:
            # Normal fit for other target columns
            target_encoder.fit(df[config.target_column])
        
        registry.target_encoder = target_encoder

    return registry


def _encode_data(
    df: pd.DataFrame,
    config: TensorConfig,
    registry: EncoderRegistry,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode features and labels using the registry.

    Args:
        df: DataFrame to encode.
        config: Configuration.
        registry: Fitted encoder registry.

    Returns:
        Tuple of (features_array, labels_array).
    """
    feature_arrays: list[np.ndarray] = []

    # Encode categorical features (as floats for concatenation)
    for col in config.categorical_columns:
        if col in registry.categorical_encoders:
            encoded = registry.categorical_encoders[col].transform(df[col])
            feature_arrays.append(encoded.reshape(-1, 1).astype(np.float32))

    # Scale numerical features
    if registry.numerical_scaler is not None:
        scaled = registry.numerical_scaler.transform(df)
        feature_arrays.append(scaled.astype(np.float32))

    # Concatenate all features
    if feature_arrays:
        features = np.hstack(feature_arrays)
    else:
        raise ValueError("No features to encode - check config columns")

    # Encode labels
    if config.task_type == "classification":
        if registry.target_encoder is None:
            raise RuntimeError("Target encoder not found for classification task")
        labels = registry.target_encoder.transform(df[config.target_column])
    else:
        # Regression - use raw values (could scale if needed)
        labels = df[config.target_column].values.astype(np.float32)

    return features, labels


def _split_data(
    features: np.ndarray,
    labels: np.ndarray,
    config: TensorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test sets.

    Args:
        features: Feature array of shape (n_samples, n_features).
        labels: Label array of shape (n_samples,).
        config: Configuration with split ratios and random seed.

    Returns:
        Tuple of (train_X, train_y, val_X, val_y, test_X, test_y).
    """
    n_samples = features.shape[0]

    # Create shuffled indices
    rng = np.random.default_rng(config.random_seed)
    indices = rng.permutation(n_samples)

    # Calculate split points
    train_end = int(n_samples * config.train_ratio)
    val_end = train_end + int(n_samples * config.val_ratio)

    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        features[train_idx],
        labels[train_idx],
        features[val_idx],
        labels[val_idx],
        features[test_idx],
        labels[test_idx],
    )


def _create_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    config: TensorConfig,
) -> CrashTensorDataset:
    """Create a CrashTensorDataset from numpy arrays.

    Args:
        features: Feature array.
        labels: Label array.
        config: Configuration.

    Returns:
        CrashTensorDataset instance.
    """
    features_tensor = torch.from_numpy(features).float()

    if config.task_type == "classification":
        labels_tensor = torch.from_numpy(labels).long()
    else:
        labels_tensor = torch.from_numpy(labels).float()

    return CrashTensorDataset(
        features=features_tensor,
        labels=labels_tensor,
        task_type=config.task_type,
    )
