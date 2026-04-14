"""Inference utilities for crash count prediction.

Provides:
- CrashCountPredictor: Main class for making predictions
- Support for adding new zones at inference time
- Feature preparation from recent history
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from data_preparation.aggregate_time_series import TimeSeriesConfig
from data_preparation.time_series_features import FeatureConfig, create_prediction_features

logger = logging.getLogger(__name__)


class CrashCountPredictor:
    """Predictor for crash counts using trained ensemble.

    Handles:
    - Loading trained models
    - Preparing features from recent history
    - Making predictions for specific zones and times
    - Adding new zones on-the-fly
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
    ) -> None:
        """Initialize predictor.

        Args:
            model_dir: Directory containing trained ensemble. If None,
                      must call load() separately.
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.ensemble = None
        self.feature_config: FeatureConfig | None = None
        self.time_config: TimeSeriesConfig | None = None
        self.feature_cols: list[str] = []
        self.zone_centroids: np.ndarray | None = None

        if model_dir:
            self.load(model_dir)

    def load(self, model_dir: str | Path) -> "CrashCountPredictor":
        """Load trained ensemble and metadata.

        Args:
            model_dir: Directory containing trained models.

        Returns:
            Self for chaining.
        """
        from training.regression.ensemble import CrashCountEnsemble

        model_dir = Path(model_dir)
        self.model_dir = model_dir

        # Load ensemble
        self.ensemble = CrashCountEnsemble.load(model_dir)
        self.feature_cols = self.ensemble.feature_cols

        # Load additional metadata
        metadata_path = model_dir / "predictor_metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.feature_config = metadata.get("feature_config")
            self.time_config = metadata.get("time_config")
            self.zone_centroids = metadata.get("zone_centroids")

        logger.info(f"Loaded predictor from {model_dir}")
        return self

    def save(self, save_dir: str | Path) -> None:
        """Save predictor state.

        Saves ensemble plus additional metadata needed for inference.

        Args:
            save_dir: Directory to save to.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save ensemble
        if self.ensemble:
            self.ensemble.save(save_dir)

        # Save predictor-specific metadata
        metadata = {
            "feature_config": self.feature_config,
            "time_config": self.time_config,
            "zone_centroids": self.zone_centroids,
        }
        joblib.dump(metadata, save_dir / "predictor_metadata.joblib")

        logger.info(f"Saved predictor to {save_dir}")

    def assign_zone(
        self,
        lat: float,
        lon: float,
    ) -> int:
        """Assign a location to the nearest zone.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Zone ID of nearest centroid.
        """
        if self.zone_centroids is None:
            raise RuntimeError("Zone centroids not loaded")

        point = np.array([[lat, lon]])
        distances = np.sqrt(np.sum((self.zone_centroids - point) ** 2, axis=1))
        return int(np.argmin(distances))

    def prepare_features(
        self,
        recent_history: pd.DataFrame,
        prediction_time: pd.Timestamp,
        zone_id: int | None = None,
    ) -> np.ndarray:
        """Prepare feature vector for prediction.

        Args:
            recent_history: Recent crash count data (must have enough
                          history for lags and rolling windows).
            prediction_time: Time to predict for.
            zone_id: Zone to predict for (optional).

        Returns:
            Feature vector of shape (1, n_features).
        """
        if self.feature_config is None:
            raise RuntimeError("Feature config not loaded")

        # Create feature dictionary
        features_dict = create_prediction_features(
            recent_history=recent_history,
            config=self.feature_config,
            prediction_time=prediction_time,
            zone_id=zone_id,
        )

        # Convert to array in correct column order
        feature_vector = np.array([
            features_dict.get(col, 0) for col in self.feature_cols
        ]).reshape(1, -1)

        return feature_vector

    def predict(
        self,
        features: np.ndarray,
        zone_id: int | None = None,
    ) -> float:
        """Make a single prediction.

        Args:
            features: Feature vector of shape (1, n_features) or (n_features,).
            zone_id: Zone to use for zone-specific adjustment.

        Returns:
            Predicted crash count (non-negative).
        """
        if self.ensemble is None:
            raise RuntimeError("Model not loaded")

        features = np.atleast_2d(features)
        zone_ids = np.array([zone_id]) if zone_id is not None else None

        preds = self.ensemble.predict(features, zone_ids)
        return float(preds[0])

    def predict_next_period(
        self,
        recent_history: pd.DataFrame,
        prediction_time: pd.Timestamp,
        zone_id: int | None = None,
    ) -> dict[str, Any]:
        """Predict crash count for the next time period.

        Convenience method that handles feature preparation.

        Args:
            recent_history: Recent crash count data.
            prediction_time: Time to predict for.
            zone_id: Zone to predict for.

        Returns:
            Dictionary with prediction and metadata.
        """
        # Prepare features
        features = self.prepare_features(
            recent_history, prediction_time, zone_id
        )

        # Make prediction
        prediction = self.predict(features, zone_id)

        return {
            "prediction_time": prediction_time,
            "zone_id": zone_id,
            "predicted_count": prediction,
            "model_type": "ensemble",
        }

    def predict_range(
        self,
        recent_history: pd.DataFrame,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        zone_id: int | None = None,
    ) -> pd.DataFrame:
        """Predict crash counts for a time range.

        Note: This uses the same features for each prediction (point forecast).
        For true multi-step forecasting, consider using an autoregressive approach.

        Args:
            recent_history: Recent crash count data.
            start_time: Start of prediction range.
            end_time: End of prediction range.
            zone_id: Zone to predict for.

        Returns:
            DataFrame with timestamp and predicted_count columns.
        """
        if self.time_config is None:
            raise RuntimeError("Time config not loaded")

        # Generate prediction times
        freq = self.time_config.freq
        prediction_times = pd.date_range(start=start_time, end=end_time, freq=freq)

        predictions = []
        for pred_time in prediction_times:
            result = self.predict_next_period(recent_history, pred_time, zone_id)
            predictions.append({
                "timestamp": pred_time,
                "predicted_count": result["predicted_count"],
                "zone_id": zone_id,
            })

        return pd.DataFrame(predictions)

    def add_zone(
        self,
        zone_id: int,
        training_data: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "crash_count",
    ) -> None:
        """Add a new zone to the ensemble.

        Trains a new zone adjustment model using existing global model.

        Args:
            zone_id: New zone identifier.
            training_data: Historical data for this zone.
            feature_cols: Feature column names.
            target_col: Target column name.
        """
        if self.ensemble is None:
            raise RuntimeError("Model not loaded")

        X = training_data[feature_cols].values
        y = training_data[target_col].values

        # Split for validation (simple 80/20)
        n_train = int(len(X) * 0.8)
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        self.ensemble.add_zone(zone_id, X_train, y_train, X_val, y_val)
        logger.info(f"Added zone {zone_id} with {len(X)} training samples")

    def get_zone_info(self) -> dict[int, dict]:
        """Get information about loaded zones.

        Returns:
            Dictionary mapping zone_id to zone metadata.
        """
        if self.ensemble is None:
            return {}

        info = {}
        for zone_id in self.ensemble.zone_models.keys():
            zone_config = self.ensemble.zone_configs.get(zone_id, {})
            info[zone_id] = {
                "has_model": True,
                "centroid": self.zone_centroids[zone_id] if self.zone_centroids is not None else None,
                **zone_config,
            }

        return info


def batch_predict(
    predictor: CrashCountPredictor,
    test_data: pd.DataFrame,
    feature_cols: list[str],
    zone_col: str | None = "zone_id",
) -> np.ndarray:
    """Make batch predictions on test data.

    Args:
        predictor: Loaded predictor.
        test_data: Test DataFrame.
        feature_cols: Feature column names.
        zone_col: Zone ID column name.

    Returns:
        Predictions array.
    """
    X = test_data[feature_cols].values
    zone_ids = test_data[zone_col].values if zone_col and zone_col in test_data.columns else None

    return predictor.ensemble.predict(X, zone_ids)
