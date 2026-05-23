"""TabNet training script for crash severity prediction.

Uses pytorch-tabnet for attention-based tabular learning with built-in
feature selection and interpretability.

Usage:
    python -m training.tabnet.train
    python -m training.tabnet.train --sample 50000 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.triple_merge import triple_merge, DataSourceConfig
from training.baselines import create_imbalance_baselines, print_dual_baseline_comparison
from training.feature_selection import filter_by_importance
from training.metrics_schema import export_model_vs_baselines_csv
from utils.logging_config import setup_logging
from utils.csv_filename_generator import generate_csv_filename

logger = setup_logging(__name__)

# Base model name for output directory
MODEL_NAME = "tabnet"


def get_output_dir(config: DataSourceConfig) -> Path:
    """Get output directory based on data source configuration.
    
    Args:
        config: Data source configuration specifying which datasets are included.
        
    Returns:
        Path to model output directory.
    """
    return PROJECT_ROOT / "models" / "trained" / f"{MODEL_NAME}_{config.get_name_suffix()}"


# Class names
CLASS_NAMES = ["NO_INJURY", "MINOR", "SEVERE"]

# Base features (always included)
CATEGORICAL_FEATURES = [
    "FIRST_CRASH_TYPE",
    "DAMAGE",
    "PRIM_CONTRIBUTORY_CAUSE",
    "TRAFFIC_CONTROL_DEVICE",
    "DEVICE_CONDITION",
    "WEATHER_CONDITION",
    "LIGHTING_CONDITION",
    "TRAFFICWAY_TYPE",
    "ROADWAY_SURFACE_COND",
    "ROAD_DEFECT",
    "ALIGNMENT",
]

NUMERICAL_FEATURES = [
    "POSTED_SPEED_LIMIT",
    "NUM_UNITS",
    "CRASH_HOUR",
    "CRASH_DAY_OF_WEEK",
    "CRASH_MONTH",
]

# Optional vehicle features (from aggregate_vehicle_data)
VEHICLE_CATEGORICAL = [
    "VEHICLE_TYPES",
]

VEHICLE_NUMERICAL = [
    "VEHICLE_COUNT",
    "OLDEST_VEHICLE_YEAR",
    "NEWEST_VEHICLE_YEAR",
    "AVG_VEHICLE_YEAR",
    "ANY_SPEED_VIOLATION",
]

# Optional people features (from aggregate_people_data)
PEOPLE_NUMERICAL = [
    "PERSON_COUNT",
    "DRIVER_COUNT",
    "MIN_AGE",
    "MAX_AGE",
    "AVG_AGE",
    "MAX_BAC",
    "ANY_BAC_POSITIVE",
    "SEATBELT_USAGE_RATE",
    "ANY_CELL_PHONE_USE",
    "ANY_EJECTION",
]

# Optional weather features (from merge_with_weather)
WEATHER_NUMERICAL = [
    "Air Temperature",
    "Humidity",
    "Rain Intensity",
    "Interval Rain",
    "Total Rain",
    "Wind Speed",
    "Barometric Pressure",
]

WEATHER_CATEGORICAL = [
    "Precipitation Type",
]

INJURY_MAPPING = {
    "NO INDICATION OF INJURY": "NO_INJURY",
    "REPORTED, NOT EVIDENT": "MINOR",
    "NONINCAPACITATING INJURY": "MINOR",
    "INCAPACITATING INJURY": "SEVERE",
    "FATAL": "SEVERE",
}


@dataclass
class TabNetConfig:
    """Configuration for TabNet model.
    
    Attributes:
        n_d: Width of decision prediction layer (default: 8)
        n_a: Width of attention embedding (default: 8)
        n_steps: Number of decision steps (default: 3)
        gamma: Coefficient for feature reusage (default: 1.3)
        n_independent: Number of independent GLU layers (default: 2)
        n_shared: Number of shared GLU layers (default: 2)
        lambda_sparse: Sparsity regularization coefficient (default: 1e-3)
        momentum: Momentum for batch normalization (default: 0.02)
        mask_type: 'sparsemax' or 'entmax' (default: 'sparsemax')
    """
    n_d: int = 8
    n_a: int = 8
    n_steps: int = 3
    gamma: float = 1.3
    n_independent: int = 2
    n_shared: int = 2
    lambda_sparse: float = 1e-3
    momentum: float = 0.02
    mask_type: str = "sparsemax"
    
    # Training params
    max_epochs: int = 200
    patience: int = 20
    batch_size: int = 1024
    virtual_batch_size: int = 128


def get_class_weights(y: np.ndarray) -> dict[int, float]:
    """Calculate balanced class weights."""
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = {int(c): n_samples / (n_classes * count) for c, count in zip(classes, counts)}
    return weights


def prepare_data(
    sample_size: int | None = None,
    config: DataSourceConfig | None = None,
    feature_filter: str = "drop-low",
) -> tuple[np.ndarray, np.ndarray, list[str], dict, list[int]]:
    """Load and prepare data for TabNet training.
    
    Args:
        sample_size: Optional sample size limit.
        config: Data source configuration (default: crash only).
        feature_filter: Feature filtering mode ('none', 'drop-low', 'drop-review').
    
    Returns:
        X, y, feature_names, encoders, cat_idxs (indices of categorical features)
    """
    if config is None:
        config = DataSourceConfig(use_vehicles=False, use_people=False, use_weather=False)
    
    logger.info(f"Data configuration: {config}")
    logger.info("Loading crash data...")
    df = triple_merge(config=config, verbose=False)
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} rows")
    
    # Extract time features
    crash_datetime = pd.to_datetime(df["CRASH_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df["CRASH_HOUR"] = crash_datetime.dt.hour.fillna(12).astype(int)
    df["CRASH_DAY_OF_WEEK"] = crash_datetime.dt.dayofweek.fillna(0).astype(int) + 1
    df["CRASH_MONTH"] = crash_datetime.dt.month.fillna(6).astype(int)
    
    # Map severity
    df["SEVERITY_3CLASS"] = df["MOST_SEVERE_INJURY"].map(INJURY_MAPPING)
    df = df.dropna(subset=["SEVERITY_3CLASS"])
    
    # Build dynamic feature lists based on config
    cat_features = CATEGORICAL_FEATURES.copy()
    num_features = NUMERICAL_FEATURES.copy()
    
    if config.use_vehicles:
        cat_features.extend(VEHICLE_CATEGORICAL)
        num_features.extend(VEHICLE_NUMERICAL)
    
    if config.use_people:
        num_features.extend(PEOPLE_NUMERICAL)
    
    if config.use_weather:
        cat_features.extend(WEATHER_CATEGORICAL)
        num_features.extend(WEATHER_NUMERICAL)
    
    # Filter to columns that exist in df
    cat_features = [c for c in cat_features if c in df.columns]
    num_features = [c for c in num_features if c in df.columns]
    
    logger.info(f"Using {len(cat_features)} categorical + {len(num_features)} numerical features")
    
    # Select features
    all_features = cat_features + num_features
    
    # Apply importance-based feature filtering if enabled
    if feature_filter != "none":
        # Create a temporary DataFrame with all features for filtering
        temp_df = df[all_features].copy()
        temp_df, kept_features, filter_result = filter_by_importance(
            temp_df, all_features, mode=feature_filter
        )
        logger.info(f"Feature filter '{feature_filter}': {filter_result.n_original} -> {filter_result.n_kept} features")
        if filter_result.dropped_features:
            logger.info(f"Dropped features: {filter_result.dropped_features[:5]}{'...' if len(filter_result.dropped_features) > 5 else ''}")
        
        # Rebuild cat_features and num_features based on what's kept
        cat_features = [f for f in cat_features if f in kept_features]
        num_features = [f for f in num_features if f in kept_features]
        all_features = kept_features
    
    columns_needed = all_features + ["SEVERITY_3CLASS"]
    df = df[[c for c in columns_needed if c in df.columns]].copy()
    
    # Fill missing values
    for col in num_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if df[col].notna().any() else 0)
    
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")
    
    # Encode categorical features
    encoders = {}
    cat_dims = []
    for col in cat_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            cat_dims.append(len(le.classes_))
    
    # Encode target
    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(CLASS_NAMES)
    df["SEVERITY_3CLASS"] = target_encoder.transform(df["SEVERITY_3CLASS"])
    encoders["target"] = target_encoder
    
    # Prepare X and y
    feature_cols = [c for c in cat_features + num_features if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y = df["SEVERITY_3CLASS"].values
    
    # Categorical feature indices (TabNet needs these)
    cat_idxs = list(range(len(cat_features)))
    
    # Store categorical dimensions for embeddings
    encoders["cat_dims"] = cat_dims
    encoders["cat_idxs"] = cat_idxs
    
    logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Categorical features: {len(cat_idxs)}, Numerical features: {len(num_features)}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_cols, encoders, cat_idxs


def train_tabnet(
    config: TabNetConfig | None = None,
    sample_size: int | None = None,
    save_model: bool = True,
    data_config: DataSourceConfig | None = None,
    feature_filter: str = "drop-low",
) -> tuple[Any, dict]:
    """Train TabNet model.
    
    Args:
        config: TabNet configuration (uses defaults if None)
        sample_size: Optional sample size for faster iteration
        save_model: Whether to save the trained model
        data_config: Data source configuration (default: crash only)
        feature_filter: Feature filtering mode ('none', 'drop-low', 'drop-review')
        
    Returns:
        Trained TabNet classifier and metrics dict
    """
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError:
        logger.error("pytorch-tabnet not installed. Run: pip install pytorch-tabnet")
        raise
    
    if config is None:
        config = TabNetConfig()
    
    if data_config is None:
        data_config = DataSourceConfig(use_vehicles=False, use_people=False, use_weather=False)
    
    output_dir = get_output_dir(data_config)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Feature filter: {feature_filter}")
    
    # Prepare data
    X, y, feature_names, encoders, cat_idxs = prepare_data(sample_size, data_config, feature_filter)
    
    # Train/val/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Class weights for imbalanced data
    class_weights = get_class_weights(y_train)
    weights = np.array([class_weights[c] for c in range(len(CLASS_NAMES))])
    
    # Categorical feature embeddings (embed dimension = min(50, (cardinality+1)//2))
    cat_dims = encoders.get("cat_dims", [])
    cat_emb_dim = [min(50, (dim + 1) // 2) for dim in cat_dims]
    
    logger.info(f"Training TabNet with config: {config}")
    
    # Initialize TabNet
    model = TabNetClassifier(
        n_d=config.n_d,
        n_a=config.n_a,
        n_steps=config.n_steps,
        gamma=config.gamma,
        n_independent=config.n_independent,
        n_shared=config.n_shared,
        lambda_sparse=config.lambda_sparse,
        momentum=config.momentum,
        mask_type=config.mask_type,
        cat_idxs=cat_idxs if cat_idxs else [],
        cat_dims=cat_dims if cat_dims else [],
        cat_emb_dim=cat_emb_dim if cat_emb_dim else [],
        verbose=1,
        seed=42,
    )
    
    # Train
    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["balanced_accuracy"],
        max_epochs=config.max_epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        virtual_batch_size=config.virtual_batch_size,
        weights=1,  # Use automatic class balancing
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    logger.info(f"\n{'='*60}")
    logger.info("TABNET TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Macro: {f1_macro:.4f}")
    logger.info(f"F1 Weighted: {f1_weighted:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    # Feature importances from attention mechanism
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importances,
    }).sort_values("importance", ascending=False)
    logger.info(f"\nTop 10 Feature Importances:")
    logger.info(importance_df.head(10).to_string(index=False))
    
    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "feature_importances": dict(zip(feature_names, [float(x) for x in feature_importances])),
    }
    
    # Baseline comparison
    baseline = create_imbalance_baselines()
    baseline.fit(y_train)
    baseline_results = baseline.evaluate(y_test, class_names=CLASS_NAMES)
    
    model_metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }
    
    print_dual_baseline_comparison(
        model_metrics=model_metrics,
        baseline_results=baseline_results,
        model_name="TabNet",
        primary_metric="f1_macro",
    )
    
    # Save model
    if save_model:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export baseline comparison CSV
        export_model_vs_baselines_csv(
            model_name="tabnet",
            model_metrics=model_metrics,
            baseline_results=baseline_results,
            output_dir=output_dir,
        )
        logger.info(f"Saved timestamped baseline comparison to {output_dir}")
        
        # TabNet has its own save method
        model.save_model(str(output_dir / "tabnet_model"))
        
        # Also save as joblib for compatibility with compare script
        # Create a wrapper that loads the TabNet model
        joblib.dump(model, output_dir / "model.joblib")
        
        # Save encoders
        joblib.dump(encoders, output_dir / "encoders.joblib")
        
        # Save feature names
        joblib.dump(feature_names, output_dir / "feature_names.joblib")
        
        # Save config and metrics
        results = {
            "config": {
                "n_d": config.n_d,
                "n_a": config.n_a,
                "n_steps": config.n_steps,
                "gamma": config.gamma,
                "lambda_sparse": config.lambda_sparse,
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Export training history to CSV with timestamp
        if hasattr(model, "history") and model.history:
            try:
                # TabNet history may be a dict of lists or custom object
                history_data = dict(model.history) if hasattr(model.history, "items") else model.history
                history_df = pd.DataFrame(history_data)
                history_filename = generate_csv_filename("training_history", "tabnet")
                history_path = output_dir / history_filename
                history_df.to_csv(history_path, index=False)
                logger.info(f"Saved training history to {history_path}")
            except Exception as e:
                logger.warning(f"Could not save training history: {e}")
        
        logger.info(f"\nSaved TabNet model to {output_dir}")
    
    return model, metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train TabNet for crash severity prediction")
    parser.add_argument("--sample", type=int, default=None, help="Sample size")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--n-d", type=int, default=8, help="Width of decision layer")
    parser.add_argument("--n-a", type=int, default=8, help="Width of attention layer")
    parser.add_argument("--n-steps", type=int, default=3, help="Number of decision steps")
    # Data source configuration flags
    parser.add_argument(
        "--include-vehicle",
        action="store_true",
        help="Include vehicle data (count, age, types, speed violations)",
    )
    parser.add_argument(
        "--include-people",
        action="store_true",
        help="Include people data (demographics, BAC, safety equipment)",
    )
    parser.add_argument(
        "--include-weather",
        action="store_true",
        help="Include weather data (temperature, humidity, rain, wind)",
    )
    parser.add_argument(
        "--feature-filter",
        type=str,
        choices=["none", "drop-low", "drop-review"],
        default="drop-low",
        help="Feature filtering mode: 'none' (all features), 'drop-low' (drop DROP features), 'drop-review' (drop DROP + REVIEW). Default: drop-low",
    )
    
    args = parser.parse_args()
    
    config = TabNetConfig(
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        max_epochs=args.epochs,
        patience=args.patience,
    )
    
    # Build data source configuration from CLI flags
    data_config = DataSourceConfig(
        use_vehicles=args.include_vehicle,
        use_people=args.include_people,
        use_weather=args.include_weather,
    )
    
    train_tabnet(
        config=config,
        sample_size=args.sample,
        data_config=data_config,
        feature_filter=args.feature_filter,
    )


if __name__ == "__main__":
    main()
