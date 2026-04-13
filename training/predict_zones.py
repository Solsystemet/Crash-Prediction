"""Zone-based inference for crash severity prediction.

Loads trained zone models and makes predictions on new crash data.
Each crash is assigned to a geographic zone based on its coordinates,
then the corresponding zone model predicts severity.

Usage:
    # Predict on a CSV file
    python predict_zones.py --input crashes.csv --output predictions.csv

    # Predict on sample from training data (for testing)
    python predict_zones.py --test-sample 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.feature_engineering import engineer_all_features
from training.hierarchical.simplified_classifier import (
    SimplifiedTreeClassifier,
    load_simplified_model,
)
from training.hierarchical.simplified_targets import SIMPLIFIED_CLASS_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default model directory
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "trained" / "simplified_zones"


class ZonePredictor:
    """Load and use zone-based models for prediction.

    Attributes:
        model_dir: Path to saved models.
        centroids: K-means cluster centroids for zone assignment.
        n_clusters: Number of zones.
        zone_models: Dictionary mapping zone_id to classifier.
        feature_cols: Feature column names expected by models.
        skipped_zones: Zone IDs that were skipped during training.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        """Initialize predictor by loading saved models.

        Args:
            model_dir: Path to directory containing saved zone models.
        """
        self.model_dir = Path(model_dir)
        self.centroids: np.ndarray | None = None
        self.n_clusters: int = 0
        self.zone_models: dict[int, SimplifiedTreeClassifier] = {}
        self.feature_cols: list[str] = []
        self.skipped_zones: list[int] = []
        self.lat_col: str = "LATITUDE"
        self.lon_col: str = "LONGITUDE"

        self._load_models()

    def _load_models(self) -> None:
        """Load centroids and zone models from disk."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Load centroids
        centroids_path = self.model_dir / "centroids.joblib"
        if not centroids_path.exists():
            raise FileNotFoundError(f"Centroids file not found: {centroids_path}")

        centroids_data = joblib.load(centroids_path)
        self.centroids = centroids_data["centroids"]
        self.n_clusters = centroids_data["n_clusters"]
        self.lat_col = centroids_data.get("lat_col", "LATITUDE")
        self.lon_col = centroids_data.get("lon_col", "LONGITUDE")

        logger.info(f"Loaded centroids for {self.n_clusters} zones")

        # Load metadata
        metadata_path = self.model_dir / "metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.feature_cols = metadata.get("feature_cols", [])
            self.skipped_zones = metadata.get("skipped_zones", [])
            logger.info(f"Feature columns: {len(self.feature_cols)}")
            if self.skipped_zones:
                logger.info(f"Skipped zones: {self.skipped_zones}")

        # Load zone models
        for zone_dir in sorted(self.model_dir.glob("zone_*")):
            try:
                zone_id = int(zone_dir.name.split("_")[1])
                clf = load_simplified_model(str(zone_dir))
                self.zone_models[zone_id] = clf
                logger.info(f"Loaded model for zone {zone_id}")
            except Exception as e:
                logger.warning(f"Failed to load {zone_dir}: {e}")

        logger.info(f"Loaded {len(self.zone_models)} zone models")

    def assign_zones(self, df: pd.DataFrame) -> np.ndarray:
        """Assign crashes to zones based on coordinates.

        Args:
            df: DataFrame with LATITUDE and LONGITUDE columns.

        Returns:
            Array of zone IDs (-1 for invalid coordinates).
        """
        if self.centroids is None:
            raise RuntimeError("Centroids not loaded")

        coords = df[[self.lat_col, self.lon_col]].values

        # Handle NaN values
        valid_mask = ~np.isnan(coords).any(axis=1)
        zones = np.full(len(coords), -1, dtype=np.int64)

        if valid_mask.sum() > 0:
            valid_coords = coords[valid_mask]
            tensor_coords = torch.from_numpy(valid_coords).float()
            tensor_centroids = torch.from_numpy(self.centroids).float()

            distances = torch.cdist(tensor_coords, tensor_centroids)
            _, cluster_labels = torch.min(distances, dim=1)
            zones[valid_mask] = cluster_labels.numpy()

        return zones

    def predict(
        self,
        df: pd.DataFrame,
        return_proba: bool = False,
    ) -> pd.DataFrame:
        """Make predictions on crash data.

        Args:
            df: DataFrame with crash features (raw or engineered).
            return_proba: Whether to include probability columns.

        Returns:
            DataFrame with zone and prediction columns added.
        """
        result = df.copy()

        # Engineer features if needed
        if "CRASH_HOUR" not in df.columns:
            logger.info("Engineering features...")
            result = engineer_all_features(result, include_interactions=True, include_clusters=False)

        # Assign zones
        logger.info("Assigning zones...")
        zones = self.assign_zones(result)
        result["ZONE"] = zones

        # Prepare features
        logger.info("Preparing features...")
        X_df = self._prepare_features(result)

        # Initialize prediction columns
        result["PREDICTION"] = "UNKNOWN"
        result["PREDICTION_ID"] = -1
        if return_proba:
            result["PROBA_INJURY"] = 0.0
            result["PROBA_SEVERE"] = 0.0

        # Predict per zone
        for zone_id, clf in self.zone_models.items():
            zone_mask = zones == zone_id
            if zone_mask.sum() == 0:
                continue

            X_zone = X_df.loc[zone_mask].values

            # Get predictions
            l1_proba, l2_proba = clf.predict_proba(X_zone)

            # Apply thresholds
            l1_pred = (l1_proba > clf.l1_threshold).astype(int)
            l2_pred = (l2_proba > clf.l2_threshold).astype(int)

            # Map to 3-class predictions
            # 0=NO_INJURY, 1=MINOR, 2=SEVERE
            pred_ids = np.where(
                l1_pred == 0,
                0,  # NO_INJURY
                np.where(l2_pred == 1, 2, 1),  # SEVERE or MINOR
            )

            pred_names = [SIMPLIFIED_CLASS_NAMES[i] for i in pred_ids]

            result.loc[zone_mask, "PREDICTION"] = pred_names
            result.loc[zone_mask, "PREDICTION_ID"] = pred_ids

            if return_proba:
                result.loc[zone_mask, "PROBA_INJURY"] = l1_proba
                result.loc[zone_mask, "PROBA_SEVERE"] = l2_proba

        # Handle skipped/unknown zones
        unknown_mask = result["PREDICTION"] == "UNKNOWN"
        n_unknown = unknown_mask.sum()
        if n_unknown > 0:
            logger.warning(f"{n_unknown} samples in zones without models")

        return result

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix matching training format.

        Uses saved feature_cols from training to ensure exact column match.

        Args:
            df: DataFrame with engineered features.

        Returns:
            DataFrame with feature columns ready for prediction.
        """
        safe_categorical_cols = [
            "WEATHER_CONDITION", "LIGHTING_CONDITION", "FIRST_CRASH_TYPE",
            "TRAFFICWAY_TYPE", "ROADWAY_SURFACE_COND", "TRAFFIC_CONTROL_DEVICE",
            "DEVICE_CONDITION", "ALIGNMENT", "ROAD_DEFECT",
            "PRIM_CONTRIBUTORY_CAUSE", "DAMAGE",
        ]

        # Build feature matrix using saved feature columns
        df_features = pd.DataFrame(index=df.index)

        for col in self.feature_cols:
            if col in df.columns:
                if col in safe_categorical_cols:
                    # Encode categorical
                    le = LabelEncoder()
                    values = df[col].fillna("UNKNOWN").astype(str)
                    df_features[col] = le.fit_transform(values)
                else:
                    # Numeric - copy directly
                    df_features[col] = df[col].fillna(0)
            else:
                # Missing column - fill with 0
                logger.warning(f"Missing feature column: {col}")
                df_features[col] = 0

        return df_features

    def summary(self) -> dict[str, Any]:
        """Get summary of loaded models.

        Returns:
            Dictionary with model statistics.
        """
        return {
            "model_dir": str(self.model_dir),
            "n_clusters": self.n_clusters,
            "n_models_loaded": len(self.zone_models),
            "zone_ids": sorted(self.zone_models.keys()),
            "skipped_zones": self.skipped_zones,
            "n_features": len(self.feature_cols),
        }


def predict_file(
    input_path: str,
    output_path: str,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    return_proba: bool = False,
) -> pd.DataFrame:
    """Load CSV, predict, and save results.

    Args:
        input_path: Path to input CSV file.
        output_path: Path to save predictions CSV.
        model_dir: Path to saved models.
        return_proba: Include probability columns.

    Returns:
        DataFrame with predictions.
    """
    logger.info(f"Loading input: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows")

    predictor = ZonePredictor(model_dir)
    result = predictor.predict(df, return_proba=return_proba)

    logger.info(f"Saving predictions: {output_path}")
    result.to_csv(output_path, index=False)

    # Summary
    pred_counts = result["PREDICTION"].value_counts()
    logger.info(f"Prediction distribution:\n{pred_counts}")

    return result


def test_on_sample(
    sample_size: int = 100,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> pd.DataFrame:
    """Test predictor on sample from training data.

    Args:
        sample_size: Number of samples to test.
        model_dir: Path to saved models.

    Returns:
        DataFrame with predictions and actual labels.
    """
    from data_preparation.triple_merge import triple_merge

    logger.info("Loading test data...")
    df = triple_merge()
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    logger.info(f"Sampled {len(df)} rows")

    # Save actual labels
    actual = df["MOST_SEVERE_INJURY"].copy()

    predictor = ZonePredictor(model_dir)
    logger.info(f"Model summary: {predictor.summary()}")

    result = predictor.predict(df, return_proba=True)
    result["ACTUAL"] = actual

    # Compare
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 60)

    pred_counts = result["PREDICTION"].value_counts()
    logger.info(f"\nPrediction distribution:\n{pred_counts}")

    actual_counts = result["ACTUAL"].value_counts()
    logger.info(f"\nActual distribution:\n{actual_counts}")

    # Zone distribution
    zone_counts = result["ZONE"].value_counts().sort_index()
    logger.info(f"\nZone distribution:\n{zone_counts}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zone-based crash severity prediction"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=f"Model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--test-sample",
        type=int,
        help="Run on N random samples from training data",
    )
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Include probability columns in output",
    )

    args = parser.parse_args()

    if args.test_sample:
        result = test_on_sample(args.test_sample, args.model_dir)
        print(f"\nSample predictions:")
        print(result[["ZONE", "PREDICTION", "ACTUAL", "PROBA_INJURY", "PROBA_SEVERE"]].head(20))
    elif args.input and args.output:
        predict_file(args.input, args.output, args.model_dir, args.proba)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python predict_zones.py --test-sample 100")
        print("  python predict_zones.py --input data.csv --output predictions.csv")
