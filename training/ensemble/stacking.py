"""Stacking ensemble for crash severity prediction.

Combines predictions from multiple base models using a meta-learner.
Uses out-of-fold predictions to avoid data leakage.

Base models:
- Random Forest (baseline)
- LightGBM (tuned)
- XGBoost (tuned)
- CatBoost (tuned)

Meta-learner:
- Logistic Regression (default) or MLP

Usage:
    python -m training.ensemble.stacking
    python -m training.ensemble.stacking --sample 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.helpers.csv_loaders import get_traffic_crashes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
OUTPUT_DIR = MODELS_DIR / "stacking_ensemble"

# Class names
CLASS_NAMES = ["NO_INJURY", "MINOR", "SEVERE"]
N_CLASSES = 3

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


@dataclass
class StackingConfig:
    """Configuration for stacking ensemble."""
    meta_learner: Literal["lr", "mlp"] = "lr"
    n_folds: int = 5
    use_probabilities: bool = True  # Use probabilities vs hard predictions
    include_original_features: bool = False  # Include original features in meta-learner


class StackingEnsemble:
    """Stacking ensemble classifier.
    
    Combines base model predictions using a meta-learner trained on
    out-of-fold predictions.
    """
    
    def __init__(
        self,
        base_models: dict[str, Any],
        meta_learner: Any,
        use_probabilities: bool = True,
        scaler: StandardScaler | None = None,
    ):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.use_probabilities = use_probabilities
        self.scaler = scaler
        self.classes_ = np.array(range(N_CLASSES))
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        meta_features = []
        
        for name, model in self.base_models.items():
            if self.use_probabilities and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                meta_features.append(proba)
            else:
                pred = model.predict(X)
                if hasattr(pred, "flatten"):
                    pred = pred.flatten()
                # One-hot encode predictions
                one_hot = np.zeros((len(pred), N_CLASSES))
                one_hot[np.arange(len(pred)), pred.astype(int)] = 1
                meta_features.append(one_hot)
        
        return np.hstack(meta_features)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        meta_X = self._get_meta_features(X)
        if self.scaler is not None:
            meta_X = self.scaler.transform(meta_X)
        return self.meta_learner.predict(meta_X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        meta_X = self._get_meta_features(X)
        if self.scaler is not None:
            meta_X = self.scaler.transform(meta_X)
        return self.meta_learner.predict_proba(meta_X)


def prepare_data(sample_size: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load and prepare data."""
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
    
    # Prepare X and y
    feature_cols = [c for c in CATEGORICAL_FEATURES + NUMERICAL_FEATURES if c in df.columns]
    X = df[feature_cols].values
    y = df["SEVERITY_3CLASS"].values
    
    logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, feature_cols, encoders


def load_or_train_base_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """Load pre-trained base models or train new ones."""
    base_models = {}
    
    # Try to load tuned models first
    tuned_dir = MODELS_DIR / "tuned_boosting"
    
    # LightGBM
    lgbm_path = tuned_dir / "lgbm" / "model.joblib"
    if lgbm_path.exists():
        logger.info("Loading tuned LightGBM...")
        base_models["lgbm"] = joblib.load(lgbm_path)
    else:
        logger.info("Training LightGBM with default params...")
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )
            model.fit(X_train, y_train)
            base_models["lgbm"] = model
        except ImportError:
            logger.warning("LightGBM not available")
    
    # XGBoost
    xgb_path = tuned_dir / "xgb" / "model.joblib"
    if xgb_path.exists():
        logger.info("Loading tuned XGBoost...")
        base_models["xgb"] = joblib.load(xgb_path)
    else:
        logger.info("Training XGBoost with default params...")
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                tree_method="hist",
            )
            model.fit(X_train, y_train)
            base_models["xgb"] = model
        except ImportError:
            logger.warning("XGBoost not available")
    
    # CatBoost
    catboost_path = tuned_dir / "catboost" / "model.joblib"
    if catboost_path.exists():
        logger.info("Loading tuned CatBoost...")
        base_models["catboost"] = joblib.load(catboost_path)
    else:
        logger.info("Training CatBoost with default params...")
        try:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(
                iterations=200,
                learning_rate=0.1,
                depth=6,
                auto_class_weights="Balanced",
                random_state=42,
                verbose=False,
            )
            model.fit(X_train, y_train)
            base_models["catboost"] = model
        except ImportError:
            logger.warning("CatBoost not available")
    
    # Random Forest (always train fresh for consistency)
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    base_models["rf"] = rf
    
    logger.info(f"Base models available: {list(base_models.keys())}")
    
    return base_models


def generate_oof_predictions(
    base_models_configs: dict[str, dict],
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    use_probabilities: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate out-of-fold predictions for stacking.
    
    Args:
        base_models_configs: Dict of model name -> config dict
        X: Features
        y: Labels
        n_folds: Number of CV folds
        use_probabilities: Use probabilities vs hard predictions
        
    Returns:
        meta_features, trained_base_models
    """
    n_samples = len(X)
    n_models = len(base_models_configs)
    
    if use_probabilities:
        meta_features = np.zeros((n_samples, n_models * N_CLASSES))
    else:
        meta_features = np.zeros((n_samples, n_models * N_CLASSES))
    
    # Store models from last fold for final ensemble
    final_models = {}
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        logger.info(f"Fold {fold_idx + 1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        for model_idx, (name, config) in enumerate(base_models_configs.items()):
            # Train model for this fold
            model = _create_base_model(name, config)
            model.fit(X_train, y_train)
            
            # Generate predictions for validation set
            start_col = model_idx * N_CLASSES
            end_col = start_col + N_CLASSES
            
            if use_probabilities and hasattr(model, "predict_proba"):
                meta_features[val_idx, start_col:end_col] = model.predict_proba(X_val)
            else:
                pred = model.predict(X_val)
                if hasattr(pred, "flatten"):
                    pred = pred.flatten()
                # One-hot encode
                for i, p in enumerate(pred):
                    meta_features[val_idx[i], start_col + int(p)] = 1
            
            # Keep model from last fold
            if fold_idx == n_folds - 1:
                final_models[name] = model
    
    return meta_features, final_models


def _create_base_model(name: str, config: dict) -> Any:
    """Create a base model from config."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=config.get("n_estimators", 200),
            max_depth=config.get("max_depth", None),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=config.get("n_estimators", 200),
            learning_rate=config.get("learning_rate", 0.1),
            max_depth=config.get("max_depth", 6),
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    elif name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=config.get("n_estimators", 200),
            learning_rate=config.get("learning_rate", 0.1),
            max_depth=config.get("max_depth", 6),
            random_state=42,
            tree_method="hist",
        )
    elif name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=config.get("iterations", 200),
            learning_rate=config.get("learning_rate", 0.1),
            depth=config.get("depth", 6),
            auto_class_weights="Balanced",
            random_state=42,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def train_stacking_ensemble(
    config: StackingConfig | None = None,
    sample_size: int | None = None,
    save_model: bool = True,
) -> tuple[StackingEnsemble, dict]:
    """Train stacking ensemble.
    
    Args:
        config: Stacking configuration
        sample_size: Optional sample size
        save_model: Whether to save the trained ensemble
        
    Returns:
        Trained StackingEnsemble and metrics dict
    """
    if config is None:
        config = StackingConfig()
    
    # Prepare data
    X, y, feature_names, encoders = prepare_data(sample_size)
    
    # Train/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train+Val: {len(X_trainval)}, Test: {len(X_test)}")
    
    # Define base model configs
    base_configs = {
        "rf": {"n_estimators": 200},
        "lgbm": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6},
        "xgb": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 6},
    }
    
    # Add CatBoost if available
    try:
        from catboost import CatBoostClassifier
        base_configs["catboost"] = {"iterations": 200, "learning_rate": 0.1, "depth": 6}
    except ImportError:
        logger.warning("CatBoost not available, skipping")
    
    # Generate OOF predictions
    logger.info("Generating out-of-fold predictions...")
    meta_features, base_models = generate_oof_predictions(
        base_configs,
        X_trainval,
        y_trainval,
        n_folds=config.n_folds,
        use_probabilities=config.use_probabilities,
    )
    
    # Scale meta-features
    scaler = StandardScaler()
    meta_features_scaled = scaler.fit_transform(meta_features)
    
    # Train meta-learner
    logger.info(f"Training meta-learner ({config.meta_learner})...")
    if config.meta_learner == "lr":
        meta_learner = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    else:  # mlp
        meta_learner = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
        )
    
    meta_learner.fit(meta_features_scaled, y_trainval)
    
    # Create ensemble
    ensemble = StackingEnsemble(
        base_models=base_models,
        meta_learner=meta_learner,
        use_probabilities=config.use_probabilities,
        scaler=scaler,
    )
    
    # Evaluate on test set
    y_pred = ensemble.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    logger.info(f"\n{'='*60}")
    logger.info("STACKING ENSEMBLE TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Macro: {f1_macro:.4f}")
    logger.info(f"F1 Weighted: {f1_weighted:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "base_models": list(base_models.keys()),
    }
    
    # Save ensemble
    if save_model:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble
        joblib.dump(ensemble, OUTPUT_DIR / "model.joblib")
        
        # Save encoders
        joblib.dump(encoders, OUTPUT_DIR / "encoders.joblib")
        
        # Save feature names
        joblib.dump(feature_names, OUTPUT_DIR / "feature_names.joblib")
        
        # Save config and metrics
        results = {
            "config": {
                "meta_learner": config.meta_learner,
                "n_folds": config.n_folds,
                "use_probabilities": config.use_probabilities,
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nSaved stacking ensemble to {OUTPUT_DIR}")
    
    return ensemble, metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train stacking ensemble")
    parser.add_argument("--sample", type=int, default=None, help="Sample size")
    parser.add_argument(
        "--meta-learner",
        type=str,
        choices=["lr", "mlp"],
        default="lr",
        help="Meta-learner type",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    
    args = parser.parse_args()
    
    config = StackingConfig(
        meta_learner=args.meta_learner,
        n_folds=args.n_folds,
    )
    
    train_stacking_ensemble(config=config, sample_size=args.sample)


if __name__ == "__main__":
    main()
