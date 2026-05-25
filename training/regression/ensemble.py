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
<<<<<<< HEAD
from utils.csv_filename_generator import generate_csv_filename
=======
>>>>>>> origin/dev

logger = logging.getLogger(__name__)


class CrashCountEnsemble:
    """Ensemble of global model + zone-specific adjustment models.

    Training strategy:
    1. Train global model on all data across all zones
    2. Compute residuals (actual - global_prediction) per zone
    3. Train zone adjustment models on their respective residuals

    Prediction:
    final_pred = global_model(X) + zone_model(X)

    This approach allows the global model to learn city-wide patterns
    while zone models capture local effects (construction, events, etc.).
    """

    def __init__(
        self,
        config: RegressionConfig | None = None,
        ensemble_config: EnsembleConfig | None = None,
    ) -> None:
        """Initialize ensemble.

        Args:
            config: Base regression configuration.
            ensemble_config: Ensemble-specific configuration.
        """
        self.config = config or RegressionConfig()
        self.ensemble_config = ensemble_config or EnsembleConfig(base_config=self.config)
        self.device = torch.device(self.config.get_device())

        # Models
        self.global_model: CrashCountMLP | None = None
        self.zone_models: dict[int, ZoneAdjustmentMLP] = {}

        # Metadata
        self.feature_cols: list[str] = []
        self.zone_configs: dict[int, ZoneConfig] = {}
        self.is_fitted: bool = False

        # Training history
        self.global_history: dict[str, list[float]] = {}
        self.zone_histories: dict[int, dict[str, list[float]]] = {}

    def _create_global_model(self, num_features: int) -> CrashCountMLP:
        """Create the global MLP model.

        Args:
            num_features: Number of input features.

        Returns:
            Initialized global model.
        """
        model = CrashCountMLP(
            num_features=num_features,
            hidden_sizes=self.config.hidden_sizes,
            dropout=self.config.dropout,
        )
        return model.to(self.device)

    def _create_zone_model(self, num_features: int) -> ZoneAdjustmentMLP:
        """Create a zone adjustment model.

        Args:
            num_features: Number of input features.

        Returns:
            Initialized zone adjustment model.
        """
        model = ZoneAdjustmentMLP(
            num_features=num_features,
            hidden_sizes=self.config.zone_hidden_sizes,
            dropout=self.config.dropout * 0.5,  # Less dropout for smaller model
        )
        return model.to(self.device)

    def fit_global(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the global model on all data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,).
            X_val: Validation features.
            y_val: Validation targets.

        Returns:
            Training history dictionary.
        """
        from training.regression.trainer import RegressionTrainer

        num_features = X.shape[1]
        self.global_model = self._create_global_model(num_features)

        trainer = RegressionTrainer(self.config)
        history = trainer.fit(
            model=self.global_model,
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
        )

        self.global_history = history
        logger.info(f"Global model trained. Best val loss: {min(history.get('val_loss', [float('inf')])):.4f}")

        return history

    def fit_zone(
        self,
        zone_id: int,
        X: np.ndarray,
        residuals: np.ndarray,
        X_val: np.ndarray | None = None,
        residuals_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train a zone adjustment model on residuals.

        Args:
            zone_id: Zone identifier.
            X: Training features for this zone.
            residuals: Residuals (actual - global_pred) for training.
            X_val: Validation features.
            residuals_val: Validation residuals.

        Returns:
            Training history for this zone.
        """
        from training.regression.trainer import RegressionTrainer

        num_features = X.shape[1]
        zone_model = self._create_zone_model(num_features)

        # Use different config for zone models (fewer epochs, higher patience)
        zone_config = RegressionConfig(
            **{
                **self.config.__dict__,
                "epochs": min(50, self.config.epochs),
                "patience": 10,
                "learning_rate": self.config.learning_rate * 0.5,
            }
        )

        trainer = RegressionTrainer(zone_config)
        history = trainer.fit(
            model=zone_model,
            X_train=X,
            y_train=residuals,
            X_val=X_val,
            y_val=residuals_val,
        )

        self.zone_models[zone_id] = zone_model
        self.zone_histories[zone_id] = history
        logger.info(f"Zone {zone_id} model trained on {len(X)} samples")

        return history

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        zone_ids: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        zone_ids_val: np.ndarray | None = None,
    ) -> "CrashCountEnsemble":
        """Train the full ensemble (global + zone models).

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training targets of shape (n_samples,).
            zone_ids: Zone ID for each sample.
            X_val: Validation features.
            y_val: Validation targets.
            zone_ids_val: Validation zone IDs.

        Returns:
            Self for method chaining.
        """
        logger.info("Training ensemble model...")
        logger.info(f"  Training samples: {len(X)}")
        logger.info(f"  Zones: {len(np.unique(zone_ids))}")

        # Step 1: Train global model on all data
        logger.info("[1/2] Training global model...")
        self.fit_global(X, y, X_val, y_val)

        # Step 2: Compute residuals and train zone models
        if self.ensemble_config.train_zone_models:
            logger.info("[2/2] Training zone adjustment models...")

            # Get global predictions for training data
            global_preds = self.predict_global(X)
            residuals = y - global_preds

            global_preds_val = self.predict_global(X_val) if X_val is not None else None
            residuals_val = y_val - global_preds_val if y_val is not None else None

            # Train per-zone models
            unique_zones = np.unique(zone_ids)
            for zone_id in unique_zones:
                zone_mask = zone_ids == zone_id
                n_zone_samples = np.sum(zone_mask)

                if n_zone_samples < 10:
                    logger.warning(f"Zone {zone_id} has only {n_zone_samples} samples, skipping")
                    continue

                X_zone = X[zone_mask]
                residuals_zone = residuals[zone_mask]

                # Validation data for this zone
                X_val_zone = None
                residuals_val_zone = None
                if X_val is not None and zone_ids_val is not None:
                    val_mask = zone_ids_val == zone_id
                    if np.sum(val_mask) > 0:
                        X_val_zone = X_val[val_mask]
                        residuals_val_zone = residuals_val[val_mask]

                self.fit_zone(
                    zone_id=zone_id,
                    X=X_zone,
                    residuals=residuals_zone,
                    X_val=X_val_zone,
                    residuals_val=residuals_val_zone,
                )
        else:
            logger.info("[2/2] Skipping zone models (disabled in config)")

        self.is_fitted = True
        return self

    def predict_global(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from global model only.

        Args:
            X: Features of shape (n_samples, n_features).

        Returns:
            Global predictions of shape (n_samples,).
        """
        if self.global_model is None:
            raise RuntimeError("Global model not fitted")

        self.global_model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            preds = self.global_model(X_tensor)
            return torch.clamp(preds, min=0).cpu().numpy()

    def predict_zone_adjustment(
        self,
        X: np.ndarray,
        zone_id: int,
    ) -> np.ndarray:
        """Get residual adjustment from zone model.

        Args:
            X: Features for the zone.
            zone_id: Zone identifier.

        Returns:
            Adjustment values (can be positive or negative).
        """
        if zone_id not in self.zone_models:
            # No zone model, return zero adjustment
            return np.zeros(len(X))

        zone_model = self.zone_models[zone_id]
        zone_model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            adjustment = zone_model(X_tensor)
            return adjustment.cpu().numpy()

    def predict(
        self,
        X: np.ndarray,
        zone_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Get ensemble predictions.

        Args:
            X: Features of shape (n_samples, n_features).
            zone_ids: Zone ID for each sample. If None, uses global only.

        Returns:
            Predicted crash counts (non-negative).
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Start with global predictions
        preds = self.predict_global(X)

        # Add zone adjustments if available
        if zone_ids is not None and len(self.zone_models) > 0:
            unique_zones = np.unique(zone_ids)
            for zone_id in unique_zones:
                if zone_id in self.zone_models:
                    zone_mask = zone_ids == zone_id
                    X_zone = X[zone_mask]
                    adjustment = self.predict_zone_adjustment(X_zone, zone_id)

                    # Weight ensemble
                    adj_weight = self.ensemble_config.zone_weight
                    preds[zone_mask] += adj_weight * adjustment

        # Ensure non-negative
        return np.maximum(preds, 0)

    def add_zone(
        self,
        zone_id: int,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Add and train a new zone model.

        Used when a new zone is identified after initial training.

        Args:
            zone_id: New zone identifier.
            X: Training features for the new zone.
            y: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        if zone_id in self.zone_models:
            logger.warning(f"Zone {zone_id} already exists, will be overwritten")

        # Compute residuals from global model
        global_preds = self.predict_global(X)
        residuals = y - global_preds

        residuals_val = None
        if X_val is not None and y_val is not None:
            global_preds_val = self.predict_global(X_val)
            residuals_val = y_val - global_preds_val

        # Train zone model
        self.fit_zone(zone_id, X, residuals, X_val, residuals_val)
        logger.info(f"Added zone {zone_id} with {len(X)} samples")

    def save(self, save_dir: str | Path) -> None:
        """Save ensemble to disk.

        Saves:
        - global_model.pt: Global model state dict
        - zone_{id}_model.pt: Zone model state dicts
        - metadata.joblib: Config, feature cols, zone configs

        Args:
            save_dir: Directory to save models to.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save global model
        if self.global_model is not None:
            torch.save(
                self.global_model.state_dict(),
                save_dir / "global_model.pt",
            )

        # Save zone models
        for zone_id, model in self.zone_models.items():
            torch.save(
                model.state_dict(),
                save_dir / f"zone_{zone_id}_model.pt",
            )

        # Save metadata
        metadata = {
            "config": self.config,
            "ensemble_config": self.ensemble_config,
            "feature_cols": self.feature_cols,
            "zone_configs": self.zone_configs,
            "zone_ids": list(self.zone_models.keys()),
            "global_history": self.global_history,
            "zone_histories": self.zone_histories,
        }
        joblib.dump(metadata, save_dir / "metadata.joblib")

        logger.info(f"Ensemble saved to {save_dir}")

<<<<<<< HEAD
    def export_training_history(self, output_dir: str | Path) -> None:
        """Export training history to CSV files.
        
        Creates:
        - training_history_global.csv: Global model training history
        - training_history_zones.csv: Combined zone model histories
        - training_curves.png: Visualization of training curves (if plotting available)
        
        Args:
            output_dir: Directory to save CSV files.
        """
        import pandas as pd
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export global model history
        if self.global_history:
            df_global = pd.DataFrame(self.global_history)
            df_global["epoch"] = range(1, len(df_global) + 1)
            df_global["model"] = "global"
            df_global = df_global[["epoch", "model", "train_loss", "val_loss", "lr"]]
            global_filename = generate_csv_filename("training_history_global", "regression_ensemble")
            df_global.to_csv(output_dir / global_filename, index=False)
            logger.info(f"Exported global training history: {len(df_global)} epochs")
        
        # Export zone model histories
        if self.zone_histories:
            zone_dfs = []
            for zone_id, history in self.zone_histories.items():
                df_zone = pd.DataFrame(history)
                df_zone["epoch"] = range(1, len(df_zone) + 1)
                df_zone["zone_id"] = zone_id
                df_zone["model"] = f"zone_{zone_id}"
                zone_dfs.append(df_zone)
            
            if zone_dfs:
                df_zones = pd.concat(zone_dfs, ignore_index=True)
                cols = ["epoch", "zone_id", "model", "train_loss"]
                if "val_loss" in df_zones.columns:
                    cols.append("val_loss")
                if "lr" in df_zones.columns:
                    cols.append("lr")
                df_zones = df_zones[cols]
                zones_filename = generate_csv_filename("training_history_zones", "regression_ensemble")
                df_zones.to_csv(output_dir / zones_filename, index=False)
                logger.info(f"Exported zone training histories: {len(self.zone_histories)} zones")
        
        # Try to generate training curves plot
        try:
            from training.plotting.training_curves import plot_training_curves
            
            if self.global_history and "val_loss" in self.global_history:
                plot_training_curves(
                    train_losses=self.global_history["train_loss"],
                    val_losses=self.global_history["val_loss"],
                    output_path=output_dir / "training_curves_global.png",
                    title="Global Model Training Curves",
                )
                logger.info("Generated global training curves plot")
        except ImportError:
            pass  # Plotting module not available

=======
>>>>>>> origin/dev
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
