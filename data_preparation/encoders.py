"""Encoding and scaling utilities for tensor data preparation.

This module provides wrappers around sklearn encoders/scalers with
serialization support for inference pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class CategoricalEncoder:
    """Wrapper around sklearn LabelEncoder with mapping storage.

    Attributes:
        column_name: Name of the column this encoder is for.
        classes_: Array of unique classes after fitting.
        unknown_value: Value used for unknown categories during transform.
        add_unknown_class: Whether to always add unknown_value to classes.
            If True, unknown_value is always added (for handling unseen values).
            If False, unknown_value is only added if present in data or if NaN exists.
    """

    column_name: str
    unknown_value: str = "UNKNOWN"
    add_unknown_class: bool = True
    _encoder: LabelEncoder = field(default_factory=LabelEncoder, repr=False)
    classes_: NDArray[np.str_] | None = field(default=None, init=False)

    def fit(self, values: pd.Series) -> CategoricalEncoder:
        """Fit the encoder to the given values.

        Args:
            values: Pandas Series of categorical values.

        Returns:
            Self for method chaining.
        """
        # Fill NA with unknown_value
        filled = values.fillna(self.unknown_value).astype(str)
        unique_values = list(filled.unique())
        
        # Add unknown_value to classes based on settings
        if self.add_unknown_class:
            # Always add unknown for handling unseen categories
            if self.unknown_value not in unique_values:
                unique_values.append(self.unknown_value)
        else:
            # Only add if there are NaN values in original data
            has_nan = values.isna().any()
            if has_nan and self.unknown_value not in unique_values:
                unique_values.append(self.unknown_value)

        self._encoder.fit(unique_values)
        self.classes_ = self._encoder.classes_
        return self

    def transform(self, values: pd.Series) -> NDArray[np.int64]:
        """Transform categorical values to integer labels.

        Unknown categories are mapped to the unknown_value label.

        Args:
            values: Pandas Series of categorical values.

        Returns:
            Numpy array of integer labels.
        """
        if self.classes_ is None:
            raise RuntimeError("Encoder must be fit before transform")

        filled = values.fillna(self.unknown_value).astype(str)

        # Handle unknown categories by mapping to unknown_value
        known_mask = filled.isin(self.classes_)
        filled_safe = filled.copy()
        filled_safe[~known_mask] = self.unknown_value

        return self._encoder.transform(filled_safe)

    def fit_transform(self, values: pd.Series) -> NDArray[np.int64]:
        """Fit and transform in one step.

        Args:
            values: Pandas Series of categorical values.

        Returns:
            Numpy array of integer labels.
        """
        self.fit(values)
        return self.transform(values)

    def inverse_transform(self, labels: NDArray[np.int64]) -> NDArray[np.str_]:
        """Convert integer labels back to original categories.

        Args:
            labels: Numpy array of integer labels.

        Returns:
            Numpy array of original category strings.
        """
        if self.classes_ is None:
            raise RuntimeError("Encoder must be fit before inverse_transform")
        return self._encoder.inverse_transform(labels)

    def get_num_classes(self) -> int:
        """Return the number of unique classes."""
        if self.classes_ is None:
            raise RuntimeError("Encoder must be fit first")
        return len(self.classes_)

    def to_dict(self) -> dict[str, Any]:
        """Serialize encoder state to dictionary."""
        return {
            "column_name": self.column_name,
            "unknown_value": self.unknown_value,
            "classes_": self.classes_.tolist() if self.classes_ is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoricalEncoder:
        """Deserialize encoder from dictionary."""
        encoder = cls(
            column_name=data["column_name"],
            unknown_value=data["unknown_value"],
        )
        if data["classes_"] is not None:
            encoder.classes_ = np.array(data["classes_"])
            encoder._encoder.fit(encoder.classes_)
        return encoder


@dataclass
class NumericalScaler:
    """Wrapper around sklearn StandardScaler with column tracking.

    Attributes:
        column_names: Names of columns this scaler handles.
        mean_: Mean values after fitting (one per column).
        scale_: Standard deviation values after fitting (one per column).
    """

    column_names: list[str]
    _scaler: StandardScaler = field(default_factory=StandardScaler, repr=False)
    mean_: NDArray[np.float64] | None = field(default=None, init=False)
    scale_: NDArray[np.float64] | None = field(default=None, init=False)

    def fit(self, df: pd.DataFrame) -> NumericalScaler:
        """Fit the scaler to the given DataFrame columns.

        Args:
            df: DataFrame containing the columns to scale.

        Returns:
            Self for method chaining.
        """
        values = df[self.column_names].values.astype(np.float64)
        self._scaler.fit(values)
        self.mean_ = self._scaler.mean_
        self.scale_ = self._scaler.scale_
        return self

    def transform(self, df: pd.DataFrame) -> NDArray[np.float64]:
        """Transform DataFrame columns to scaled values.

        Args:
            df: DataFrame containing the columns to scale.

        Returns:
            Numpy array of scaled values.
        """
        if self.mean_ is None:
            raise RuntimeError("Scaler must be fit before transform")

        values = df[self.column_names].values.astype(np.float64)
        return self._scaler.transform(values)

    def fit_transform(self, df: pd.DataFrame) -> NDArray[np.float64]:
        """Fit and transform in one step.

        Args:
            df: DataFrame containing the columns to scale.

        Returns:
            Numpy array of scaled values.
        """
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert scaled values back to original scale.

        Args:
            values: Numpy array of scaled values.

        Returns:
            Numpy array of original-scale values.
        """
        if self.mean_ is None:
            raise RuntimeError("Scaler must be fit before inverse_transform")
        return self._scaler.inverse_transform(values)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scaler state to dictionary."""
        return {
            "column_names": self.column_names,
            "mean_": self.mean_.tolist() if self.mean_ is not None else None,
            "scale_": self.scale_.tolist() if self.scale_ is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NumericalScaler:
        """Deserialize scaler from dictionary."""
        scaler = cls(column_names=data["column_names"])
        if data["mean_"] is not None:
            scaler.mean_ = np.array(data["mean_"])
            scaler.scale_ = np.array(data["scale_"])
            scaler._scaler.mean_ = scaler.mean_
            scaler._scaler.scale_ = scaler.scale_
            scaler._scaler.var_ = scaler.scale_**2
            scaler._scaler.n_features_in_ = len(scaler.column_names)
        return scaler


@dataclass
class EncoderRegistry:
    """Container for all encoders/scalers used in a pipeline.

    Stores categorical encoders, numerical scalers, and target encoder
    for serialization and inference.

    Attributes:
        categorical_encoders: Dict mapping column name to CategoricalEncoder.
        numerical_scaler: NumericalScaler for all numerical columns.
        target_encoder: CategoricalEncoder for target column (if classification).
        feature_columns: Ordered list of feature column names.
        target_column: Name of target column.
        task_type: "classification" or "regression".
    """

    categorical_encoders: dict[str, CategoricalEncoder] = field(default_factory=dict)
    numerical_scaler: NumericalScaler | None = None
    target_encoder: CategoricalEncoder | None = None
    feature_columns: list[str] = field(default_factory=list)
    target_column: str = ""
    task_type: str = "classification"

    def get_num_features(self) -> int:
        """Return total number of input features after encoding."""
        num_cat = len(self.categorical_encoders)
        num_num = len(self.numerical_scaler.column_names) if self.numerical_scaler else 0
        return num_cat + num_num

    def get_num_classes(self) -> int:
        """Return number of target classes (for classification)."""
        if self.target_encoder is None:
            raise RuntimeError("No target encoder - task may be regression")
        return self.target_encoder.get_num_classes()

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire registry to dictionary."""
        return {
            "categorical_encoders": {
                name: enc.to_dict() for name, enc in self.categorical_encoders.items()
            },
            "numerical_scaler": (
                self.numerical_scaler.to_dict() if self.numerical_scaler else None
            ),
            "target_encoder": (
                self.target_encoder.to_dict() if self.target_encoder else None
            ),
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "task_type": self.task_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncoderRegistry:
        """Deserialize registry from dictionary."""
        registry = cls(
            feature_columns=data["feature_columns"],
            target_column=data["target_column"],
            task_type=data["task_type"],
        )

        for name, enc_data in data["categorical_encoders"].items():
            registry.categorical_encoders[name] = CategoricalEncoder.from_dict(enc_data)

        if data["numerical_scaler"]:
            registry.numerical_scaler = NumericalScaler.from_dict(data["numerical_scaler"])

        if data["target_encoder"]:
            registry.target_encoder = CategoricalEncoder.from_dict(data["target_encoder"])

        return registry
