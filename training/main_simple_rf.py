"""Simple vanilla Random Forest classifier for 3-class severity prediction.

This is a baseline model with:
- No feature engineering
- No resampling (SMOTE/ADASYN)
- No hyperparameter tuning
- Default sklearn RandomForestClassifier parameters

Uses only raw columns from traffic_crashes.csv.

Output classes:
- NO_INJURY: No indication of injury
- MINOR: Nonincapacitating + Reported injuries
- SEVERE: Fatal + Incapacitating injuries

Usage:
    python training/main_simple_rf.py
    python training/main_simple_rf.py --sample 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.helpers.csv_loaders import get_traffic_crashes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "models" / "trained" / "simple_rf"

# Class names (same as simplified model for compatibility)
CLASS_NAMES = ["NO_INJURY", "MINOR", "SEVERE"]

# Features to use from raw crash data
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
    "NUM_UNITS",  # vehicle count
    "CRASH_HOUR",
    "CRASH_DAY_OF_WEEK",
    "CRASH_MONTH",
]

# Mapping from raw injury labels to simplified 3-class
INJURY_MAPPING = {
    "NO INDICATION OF INJURY": "NO_INJURY",
    "REPORTED, NOT EVIDENT": "MINOR",
    "NONINCAPACITATING INJURY": "MINOR",
    "INCAPACITATING INJURY": "SEVERE",
    "FATAL": "SEVERE",
}


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day of week, and month from CRASH_DATE.

    Args:
        df: DataFrame with CRASH_DATE column.

    Returns:
        DataFrame with added time columns.
    """
    df = df.copy()
    
    # Parse crash date
    crash_datetime = pd.to_datetime(df["CRASH_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    
    df["CRASH_HOUR"] = crash_datetime.dt.hour.fillna(12).astype(int)
    df["CRASH_DAY_OF_WEEK"] = crash_datetime.dt.dayofweek.fillna(0).astype(int) + 1  # 1-7
    df["CRASH_MONTH"] = crash_datetime.dt.month.fillna(6).astype(int)
    
    return df


def map_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Map MOST_SEVERE_INJURY to 3-class target.

    Args:
        df: DataFrame with MOST_SEVERE_INJURY column.

    Returns:
        DataFrame with SEVERITY_3CLASS column.
    """
    df = df.copy()
    df["SEVERITY_3CLASS"] = df["MOST_SEVERE_INJURY"].map(INJURY_MAPPING)
    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Prepare features and encode categoricals.

    Args:
        df: Raw DataFrame from traffic_crashes.csv.

    Returns:
        Tuple of (prepared DataFrame, dict of label encoders).
    """
    df = df.copy()
    
    # Extract time features
    df = extract_time_features(df)
    
    # Map severity target
    df = map_severity(df)
    
    # Drop rows with missing target
    df = df.dropna(subset=["SEVERITY_3CLASS"])
    
    # Select only needed columns
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
    
    # Encode categorical features
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    # Encode target
    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(CLASS_NAMES)
    df["SEVERITY_3CLASS"] = target_encoder.transform(df["SEVERITY_3CLASS"])
    encoders["target"] = target_encoder
    
    return df, encoders


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a vanilla Random Forest classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Random seed.

    Returns:
        Trained RandomForestClassifier.
    """
    logger.info("Training Random Forest with balanced class weights...")
    
    model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,  # Use all cores
        class_weight='balanced',  # Penalize misclassifying rare classes
    )
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate the model and print metrics.

    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    logger.info(f"\n{'='*60}")
    logger.info("SIMPLE RANDOM FOREST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Macro: {f1_macro:.4f}")
    logger.info(f"F1 Weighted: {f1_weighted:.4f}")
    
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    logger.info(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n{cm}")
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
    }


def save_model(
    model: RandomForestClassifier,
    encoders: dict[str, LabelEncoder],
    feature_names: list[str],
    metrics: dict,
) -> None:
    """Save model and encoders to disk.

    Args:
        model: Trained classifier.
        encoders: Dictionary of label encoders.
        feature_names: List of feature column names.
        metrics: Training metrics.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = OUTPUT_DIR / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save encoders
    encoders_path = OUTPUT_DIR / "encoders.joblib"
    joblib.dump(encoders, encoders_path)
    logger.info(f"Saved encoders to {encoders_path}")
    
    # Save feature names
    features_path = OUTPUT_DIR / "feature_names.joblib"
    joblib.dump(feature_names, features_path)
    logger.info(f"Saved feature names to {features_path}")
    
    # Save metrics
    metrics_path = OUTPUT_DIR / "metrics.joblib"
    joblib.dump(metrics, metrics_path)
    logger.info(f"Saved metrics to {metrics_path}")


def main(sample_size: int | None = None, random_state: int = 42) -> None:
    """Main training pipeline.

    Args:
        sample_size: Optional sample size for faster iteration.
        random_state: Random seed for reproducibility.
    """
    logger.info("Loading traffic crashes data...")
    df = get_traffic_crashes()
    logger.info(f"Loaded {len(df)} crashes")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size} crashes...")
        df = df.sample(n=sample_size, random_state=random_state)
    
    # Prepare data
    logger.info("Preparing data...")
    df, encoders = prepare_data(df)
    logger.info(f"Prepared {len(df)} samples with valid target")
    
    # Log class distribution
    target_counts = pd.Series(df["SEVERITY_3CLASS"]).value_counts().sort_index()
    for idx, count in target_counts.items():
        logger.info(f"  {CLASS_NAMES[idx]}: {count} ({100*count/len(df):.1f}%)")
    
    # Split features and target
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].values
    y = df["SEVERITY_3CLASS"].values
    
    # Train/test split
    logger.info("Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )
    logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    model = train_model(X_train, y_train, random_state)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save
    save_model(model, encoders, feature_cols, metrics)
    
    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train simple Random Forest model")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for faster iteration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    main(sample_size=args.sample, random_state=args.seed)
