"""Ensemble model combining global and zone-specific predictions.

The ensemble architecture:
1. Global model: Trained on all zones, captures city-wide patterns
2. Zone models: Per-zone adjustment models that learn residuals
3. Final prediction: global_pred + zone_adjustment
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn

from training.regression.config import RegressionConfig, ZoneConfig, EnsembleConfig
from training.regression.models import CrashCountMLP, ZoneAdjustmentMLP
from utils.csv_filename_generator import generate_csv_filename
    @classmethod
    def load(cls, load_dir: str | Path) -> "CrashCountEnsemble":
        """Load ensemble from disk.

        Args:
            load_dir: Directory containing saved models.

        Returns:
            Loaded ensemble.
        """
        load_dir = Path(load_dir)

        # Load metadata
        metadata = joblib.load(load_dir / "metadata.joblib")

        # Create ensemble
        ensemble = cls(
            config=metadata["config"],
            ensemble_config=metadata["ensemble_config"],
        )
        ensemble.feature_cols = metadata["feature_cols"]
        ensemble.zone_configs = metadata["zone_configs"]
        ensemble.global_history = metadata["global_history"]
        ensemble.zone_histories = metadata["zone_histories"]

        # Infer number of features from first zone model or raise error
        # Try to get num_features from a model file
        global_path = load_dir / "global_model.pt"
        if global_path.exists():
            state_dict = torch.load(global_path, map_location=ensemble.device, weights_only=True)
            # Get input size from first layer
            first_layer_key = "network.0.weight"
            if first_layer_key in state_dict:
                num_features = state_dict[first_layer_key].shape[1]

                # Load global model
                ensemble.global_model = ensemble._create_global_model(num_features)
                ensemble.global_model.load_state_dict(state_dict)

        # Load zone models
        for zone_id in metadata["zone_ids"]:
            zone_path = load_dir / f"zone_{zone_id}_model.pt"
            if zone_path.exists():
                state_dict = torch.load(zone_path, map_location=ensemble.device, weights_only=True)
                first_layer_key = "network.0.weight"
                if first_layer_key in state_dict:
                    num_features = state_dict[first_layer_key].shape[1]
                    zone_model = ensemble._create_zone_model(num_features)
                    zone_model.load_state_dict(state_dict)
                    ensemble.zone_models[zone_id] = zone_model

        ensemble.is_fitted = True
        logger.info(f"Ensemble loaded from {load_dir}")
        logger.info(f"  Global model: {'loaded' if ensemble.global_model else 'missing'}")
        logger.info(f"  Zone models: {len(ensemble.zone_models)}")

        return ensemble
