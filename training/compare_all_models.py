"""Compare all trained models on the same test set.

Generates a comparison table and confusion matrices for all models.

Usage:
    python training/compare_all_models.py
    python training/compare_all_models.py --sample 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.helpers.csv_loaders import get_traffic_crashes
from training.ensemble.stacking import StackingEnsemble  # Required for unpickling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
OUTPUT_DIR = PROJECT_ROOT / "models" / "plots"

# Class names
CLASS_NAMES = ["NO_INJURY", "MINOR", "SEVERE"]

# Features (same as baseline)
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

INJURY_MAPPING = {
    "NO INDICATION OF INJURY": "NO_INJURY",
    "REPORTED, NOT EVIDENT": "MINOR",
    "NONINCAPACITATING INJURY": "MINOR",
    "INCAPACITATING INJURY": "SEVERE",
    "FATAL": "SEVERE",
}


def prepare_test_data(sample_size: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and prepare test data using the same preprocessing as training."""
    logger.info("Loading crash data...")
    df = get_traffic_crashes()
    
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
    
    # Select features
    all_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    columns_needed = all_features + ["SEVERITY_3CLASS"]
    df = df[[c for c in columns_needed if c in df.columns]].copy()
    
    # Fill missing values
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")
    
    # We'll use the encoders from the simple_rf model for consistency
    encoders_path = MODELS_DIR / "simple_rf" / "encoders.joblib"
    if encoders_path.exists():
        encoders = joblib.load(encoders_path)
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and col in encoders:
                le = encoders[col]
                # Handle unseen categories
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    else:
        # Fallback: encode from scratch
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    
    # Encode target
    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(CLASS_NAMES)
    df["SEVERITY_3CLASS"] = target_encoder.transform(df["SEVERITY_3CLASS"])
    
    # Prepare X and y
    feature_cols = [c for c in CATEGORICAL_FEATURES + NUMERICAL_FEATURES if c in df.columns]
    X = df[feature_cols].values
    y = df["SEVERITY_3CLASS"].values
    
    # Use same test split as training (20%)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_test, y_test, feature_cols


def load_model(model_dir: Path) -> Any | None:
    """Load a trained model from a directory."""
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "flatten"):
        y_pred = y_pred.flatten()
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
    }
    
    # Per-class recall (important for minority classes)
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_recall = recall_score(y_test[class_mask], y_pred[class_mask], average="micro")
            metrics[f"recall_{class_name}"] = (y_pred[class_mask] == i).sum() / class_mask.sum()
    
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    
    return metrics


def find_all_models() -> list[tuple[str, Path]]:
    """Find all trained models in the models directory."""
    models = []
    
    # Simple RF
    simple_rf_dir = MODELS_DIR / "simple_rf"
    if (simple_rf_dir / "model.joblib").exists():
        models.append(("Random Forest (baseline)", simple_rf_dir))
    
    # Tuned boosting models
    tuned_dir = MODELS_DIR / "tuned_boosting"
    if tuned_dir.exists():
        for model_type in ["lgbm", "xgb", "catboost"]:
            model_dir = tuned_dir / model_type
            if (model_dir / "model.joblib").exists():
                name_map = {"lgbm": "LightGBM (tuned)", "xgb": "XGBoost (tuned)", "catboost": "CatBoost (tuned)"}
                models.append((name_map.get(model_type, model_type), model_dir))
    
    # TabNet
    tabnet_dir = MODELS_DIR / "tabnet"
    if (tabnet_dir / "model.joblib").exists():
        models.append(("TabNet", tabnet_dir))
    
    # Stacking ensemble
    ensemble_dir = MODELS_DIR / "stacking_ensemble"
    if (ensemble_dir / "model.joblib").exists():
        models.append(("Stacking Ensemble", ensemble_dir))
    
    return models


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    if not results:
        logger.warning("No results to display")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Select columns for display
    display_cols = ["model", "accuracy", "f1_macro", "f1_weighted", "recall_SEVERE"]
    display_cols = [c for c in display_cols if c in df.columns]
    
    df_display = df[display_cols].copy()
    
    # Format percentages
    for col in df_display.columns:
        if col != "model" and df_display[col].dtype in [float, np.float64]:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
    
    # Sort by F1 macro
    df_display = df_display.sort_values("f1_macro", ascending=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)
    logger.info(f"\n{df_display.to_string(index=False)}")
    
    # Print best model
    best_model = df.loc[df["f1_macro"].idxmax(), "model"]
    best_f1 = df["f1_macro"].max()
    logger.info(f"\nBest model: {best_model} (F1 Macro: {best_f1:.4f})")


def save_comparison(results: list[dict]) -> None:
    """Save comparison results to JSON and CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save full results as JSON
    with open(OUTPUT_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary as CSV
    df = pd.DataFrame(results)
    summary_cols = ["model", "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    for cls in CLASS_NAMES:
        col = f"recall_{cls}"
        if col in df.columns:
            summary_cols.append(col)
    
    df[summary_cols].to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    
    logger.info(f"\nSaved comparison to {OUTPUT_DIR}")


def main(sample_size: int | None = None) -> None:
    """Run comparison of all trained models."""
    logger.info("Starting model comparison...")
    
    # Prepare test data
    X_test, y_test, feature_names = prepare_test_data(sample_size)
    
    # Find all models
    models = find_all_models()
    
    if not models:
        logger.warning("No trained models found! Run training scripts first.")
        return
    
    logger.info(f"Found {len(models)} trained models")
    
    # Evaluate each model
    results = []
    for model_name, model_dir in models:
        logger.info(f"\nEvaluating {model_name}...")
        
        model = load_model(model_dir)
        if model is None:
            logger.warning(f"Could not load model from {model_dir}")
            continue
        
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        if "recall_SEVERE" in metrics:
            logger.info(f"  SEVERE Recall: {metrics['recall_SEVERE']:.4f}")
    
    # Print and save comparison
    print_comparison_table(results)
    save_comparison(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all trained models")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for faster iteration",
    )
    
    args = parser.parse_args()
    main(sample_size=args.sample)
