"""Hyperparameter tuning for tree-based models using Optuna.

Tunes LightGBM, XGBoost, CatBoost, and Random Forest with Bayesian optimization.
Uses 3-fold stratified CV with macro F1 as the objective. Includes:
- Optuna pruning to stop unpromising trials early
- GPU acceleration for XGBoost/CatBoost (auto-detected)
- Narrowed search ranges for faster convergence

Usage:
    python training/tune_boosting.py
    python training/tune_boosting.py --model lgbm --n-trials 100
    python training/tune_boosting.py --model rf --n-trials 50
    python training/tune_boosting.py --model all --sample 50000
    python training/tune_boosting.py --model xgb --no-gpu  # Force CPU
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Suppress warnings during optimization
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.helpers.csv_loaders import get_traffic_crashes
from training.baselines import (
    create_imbalance_baselines,
    print_dual_baseline_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "models" / "trained" / "tuned_boosting"

# Class names
CLASS_NAMES = ["NO_INJURY", "MINOR", "SEVERE"]

# Features (same as simple RF baseline for fair comparison)
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

# Tuning configuration
CV_FOLDS = 3  # Reduced from 5 for faster tuning
PRUNING_WARMUP_STEPS = 5  # Trials before pruning kicks in


def detect_gpu() -> bool:
    """Detect if GPU is available for XGBoost/CatBoost."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    # Fallback: check for CUDA libraries
    try:
        import xgboost as xgb
        # Try to create a small GPU model
        params = {"tree_method": "gpu_hist", "n_estimators": 1}
        model = xgb.XGBClassifier(**params)
        return True
    except Exception:
        return False


INJURY_MAPPING = {
    "NO INDICATION OF INJURY": "NO_INJURY",
    "REPORTED, NOT EVIDENT": "MINOR",
    "NONINCAPACITATING INJURY": "MINOR",
    "INCAPACITATING INJURY": "SEVERE",
    "FATAL": "SEVERE",
}


def prepare_data(sample_size: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Load and prepare data for training.
    
    Args:
        sample_size: Optional sample size limit.
        
    Returns:
        X, y, feature_names, encoders
    """
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
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_cols, encoders


def get_class_weights(y: np.ndarray) -> dict[int, float]:
    """Calculate balanced class weights."""
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = {int(c): n_samples / (n_classes * count) for c, count in zip(classes, counts)}
    return weights


# ============================================================================
# LightGBM Tuning
# ============================================================================

def objective_lgbm(trial, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> float:
    """Optuna objective function for LightGBM with pruning support."""
    import lightgbm as lgb
    
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "random_state": 42,
        # Tunable parameters (narrowed ranges for faster convergence)
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 150, 500),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
        "class_weight": "balanced",
    }
    
    f1_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        
        y_pred = model.predict(X_val)
        fold_f1 = f1_score(y_val, y_pred, average="macro")
        f1_scores.append(fold_f1)
        
        # Report intermediate value for pruning
        trial.report(np.mean(f1_scores), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(f1_scores)


def tune_lgbm(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> tuple[dict, float]:
    """Tune LightGBM hyperparameters with pruning."""
    import optuna
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=PRUNING_WARMUP_STEPS)
    
    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_tuning",
        pruner=pruner,
    )
    study.optimize(
        lambda trial: objective_lgbm(trial, X, y, cv),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,  # Sequential trials (model training is already parallel)
    )
    
    logger.info(f"LightGBM best F1 (CV): {study.best_value:.4f}")
    logger.info(f"LightGBM best params: {study.best_params}")
    logger.info(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    return study.best_params, study.best_value


# ============================================================================
# XGBoost Tuning
# ============================================================================

def objective_xgb(trial, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold, use_gpu: bool = False) -> float:
    """Optuna objective function for XGBoost with pruning and optional GPU."""
    import xgboost as xgb
    
    # Calculate sample weights for class imbalance
    class_weights = get_class_weights(y)
    
    params = {
        "objective": "multi:softmax",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": 42,
        "tree_method": "gpu_hist" if use_gpu else "hist",
        # Tunable parameters (narrowed ranges)
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 150, 500),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
        "gamma": trial.suggest_float("gamma", 1e-6, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
    }
    
    f1_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create sample weights
        sample_weights = np.array([class_weights[yi] for yi in y_train])
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        y_pred = model.predict(X_val)
        fold_f1 = f1_score(y_val, y_pred, average="macro")
        f1_scores.append(fold_f1)
        
        # Report intermediate value for pruning
        trial.report(np.mean(f1_scores), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(f1_scores)


def tune_xgb(X: np.ndarray, y: np.ndarray, n_trials: int = 50, use_gpu: bool = False) -> tuple[dict, float]:
    """Tune XGBoost hyperparameters with pruning and optional GPU."""
    import optuna
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=PRUNING_WARMUP_STEPS)
    
    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_tuning",
        pruner=pruner,
    )
    study.optimize(
        lambda trial: objective_xgb(trial, X, y, cv, use_gpu),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )
    
    logger.info(f"XGBoost best F1 (CV): {study.best_value:.4f}")
    logger.info(f"XGBoost best params: {study.best_params}")
    logger.info(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    return study.best_params, study.best_value


# ============================================================================
# CatBoost Tuning
# ============================================================================

def objective_catboost(trial, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold, use_gpu: bool = False) -> float:
    """Optuna objective function for CatBoost with pruning and optional GPU."""
    from catboost import CatBoostClassifier
    
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "verbose": False,
        "random_state": 42,
        "thread_count": -1,
        "auto_class_weights": "Balanced",
        # Tunable parameters (narrowed ranges)
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "iterations": trial.suggest_int("iterations", 200, 500),
        "depth": trial.suggest_int("depth", 5, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-6, 1.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 0.8),
        "random_strength": trial.suggest_float("random_strength", 1e-6, 1.0, log=True),
        "border_count": trial.suggest_int("border_count", 64, 200),
    }
    
    # GPU acceleration
    if use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    
    f1_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=30,
        )
        
        y_pred = model.predict(X_val).flatten()
        fold_f1 = f1_score(y_val, y_pred, average="macro")
        f1_scores.append(fold_f1)
        
        # Report intermediate value for pruning
        trial.report(np.mean(f1_scores), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(f1_scores)


def tune_catboost(X: np.ndarray, y: np.ndarray, n_trials: int = 50, use_gpu: bool = False) -> tuple[dict, float]:
    """Tune CatBoost hyperparameters with pruning and optional GPU."""
    import optuna
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=PRUNING_WARMUP_STEPS)
    
    study = optuna.create_study(
        direction="maximize",
        study_name="catboost_tuning",
        pruner=pruner,
    )
    study.optimize(
        lambda trial: objective_catboost(trial, X, y, cv, use_gpu),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )
    
    logger.info(f"CatBoost best F1 (CV): {study.best_value:.4f}")
    logger.info(f"CatBoost best params: {study.best_params}")
    logger.info(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    return study.best_params, study.best_value


# ============================================================================
# Random Forest Tuning
# ============================================================================

def objective_rf(trial, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> float:
    """Optuna objective function for Random Forest."""
    # Handle max_depth: can be None or an integer
    max_depth_choice = trial.suggest_categorical("max_depth_type", ["none", "int"])
    if max_depth_choice == "none":
        max_depth = None
    else:
        max_depth = trial.suggest_int("max_depth", 5, 50)
    
    # Handle max_features: can be "sqrt", "log2", or a float
    max_features_type = trial.suggest_categorical("max_features_type", ["sqrt", "log2", "float"])
    if max_features_type == "float":
        max_features = trial.suggest_float("max_features_float", 0.3, 0.9)
    else:
        max_features = max_features_type
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "max_depth": max_depth,
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": max_features,
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "n_jobs": -1,
        "random_state": 42,
    }
    
    # bootstrap must be True for balanced_subsample
    if params["class_weight"] == "balanced_subsample" and not params["bootstrap"]:
        params["bootstrap"] = True
    
    f1_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        fold_f1 = f1_score(y_val, y_pred, average="macro")
        f1_scores.append(fold_f1)
        
        # Report intermediate value for pruning
        trial.report(np.mean(f1_scores), fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return np.mean(f1_scores)


def tune_rf(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> tuple[dict, float]:
    """Tune Random Forest hyperparameters with pruning."""
    import optuna
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=PRUNING_WARMUP_STEPS)
    
    study = optuna.create_study(
        direction="maximize",
        study_name="rf_tuning",
        pruner=pruner,
    )
    study.optimize(
        lambda trial: objective_rf(trial, X, y, cv),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )
    
    logger.info(f"Random Forest best F1 (CV): {study.best_value:.4f}")
    logger.info(f"Random Forest best params: {study.best_params}")
    logger.info(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    return study.best_params, study.best_value


# ============================================================================
# Training Final Models
# ============================================================================

def train_final_model(
    model_type: Literal["lgbm", "xgb", "catboost", "rf"],
    best_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Any:
    """Train final model with best hyperparameters."""
    
    if model_type == "lgbm":
        import lightgbm as lgb
        
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": 42,
            "class_weight": "balanced",
            **best_params,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        
    elif model_type == "xgb":
        import xgboost as xgb
        
        class_weights = get_class_weights(y_train)
        sample_weights = np.array([class_weights[yi] for yi in y_train])
        
        params = {
            "objective": "multi:softmax",
            "num_class": 3,
            "verbosity": 0,
            "n_jobs": -1,
            "random_state": 42,
            "tree_method": "hist",
            **best_params,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        
        params = {
            "loss_function": "MultiClass",
            "verbose": False,
            "random_state": 42,
            "thread_count": -1,
            "auto_class_weights": "Balanced",
            **best_params,
        }
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
        )
        
    elif model_type == "rf":
        # Reconstruct RF params from trial params
        max_depth_type = best_params.get("max_depth_type", "int")
        if max_depth_type == "none":
            max_depth = None
        else:
            max_depth = best_params.get("max_depth", 20)
        
        max_features_type = best_params.get("max_features_type", "sqrt")
        if max_features_type == "float":
            max_features = best_params.get("max_features_float", 0.5)
        else:
            max_features = max_features_type
        
        bootstrap = best_params.get("bootstrap", True)
        class_weight = best_params.get("class_weight", "balanced")
        
        # Ensure bootstrap is True for balanced_subsample
        if class_weight == "balanced_subsample" and not bootstrap:
            bootstrap = True
        
        params = {
            "n_estimators": best_params.get("n_estimators", 500),
            "max_depth": max_depth,
            "min_samples_split": best_params.get("min_samples_split", 2),
            "min_samples_leaf": best_params.get("min_samples_leaf", 1),
            "max_features": max_features,
            "bootstrap": bootstrap,
            "class_weight": class_weight,
            "criterion": best_params.get("criterion", "gini"),
            "n_jobs": -1,
            "random_state": 42,
        }
        model = RandomForestClassifier(**params)
        # Combine train and val for final model (RF doesn't use early stopping)
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        model.fit(X_combined, y_combined)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Evaluate model on test set."""
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "flatten"):
        y_pred = y_pred.flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} TEST RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Macro: {f1_macro:.4f}")
    logger.info(f"F1 Weighted: {f1_weighted:.4f}")
    logger.info(f"\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
    }


def save_results(
    model_type: str,
    model: Any,
    best_params: dict,
    cv_score: float,
    test_metrics: dict,
    feature_names: list[str],
    encoders: dict,
) -> None:
    """Save tuned model and results."""
    model_dir = OUTPUT_DIR / model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, model_dir / "model.joblib")
    
    # Save encoders
    joblib.dump(encoders, model_dir / "encoders.joblib")
    
    # Save feature names
    joblib.dump(feature_names, model_dir / "feature_names.joblib")
    
    # Save results
    results = {
        "model_type": model_type,
        "best_params": best_params,
        "cv_f1_macro": float(cv_score),
        "test_metrics": test_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    with open(model_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved {model_type} model to {model_dir}")


def main(
    model_type: Literal["lgbm", "xgb", "catboost", "rf", "all"] = "all",
    n_trials: int = 50,
    sample_size: int | None = None,
    use_gpu: bool | None = None,
) -> None:
    """Main tuning pipeline.
    
    Args:
        model_type: Which model(s) to tune.
        n_trials: Number of Optuna trials per model.
        sample_size: Optional sample size for faster iteration.
        use_gpu: Use GPU for XGBoost/CatBoost. None=auto-detect.
    """
    # Auto-detect GPU if not specified
    if use_gpu is None:
        use_gpu = detect_gpu()
        logger.info(f"GPU auto-detected: {use_gpu}")
    else:
        logger.info(f"GPU mode: {'enabled' if use_gpu else 'disabled'}")
    
    logger.info(f"Starting hyperparameter tuning: model={model_type}, n_trials={n_trials}")
    logger.info(f"Using {CV_FOLDS}-fold CV with pruning (warmup={PRUNING_WARMUP_STEPS})")
    
    # Prepare data
    X, y, feature_names, encoders = prepare_data(sample_size)
    
    # Train/val/test split (60/20/20)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    models_to_tune = []
    if model_type == "all":
        models_to_tune = ["lgbm", "xgb", "catboost", "rf"]
    else:
        models_to_tune = [model_type]
    
    results_summary = {}
    
    for mt in models_to_tune:
        logger.info(f"\n{'='*60}")
        logger.info(f"TUNING {mt.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            if mt == "lgbm":
                best_params, cv_score = tune_lgbm(X_trainval, y_trainval, n_trials)
            elif mt == "xgb":
                best_params, cv_score = tune_xgb(X_trainval, y_trainval, n_trials, use_gpu)
            elif mt == "catboost":
                best_params, cv_score = tune_catboost(X_trainval, y_trainval, n_trials, use_gpu)
            elif mt == "rf":
                best_params, cv_score = tune_rf(X_trainval, y_trainval, n_trials)
            else:
                continue
            
            # Train final model
            logger.info(f"\nTraining final {mt.upper()} model with best params...")
            model = train_final_model(mt, best_params, X_train, y_train, X_val, y_val)
            
            # Evaluate on test set
            test_metrics = evaluate_model(model, X_test, y_test, mt.upper())
            
            # Save
            save_results(mt, model, best_params, cv_score, test_metrics, feature_names, encoders)
            
            results_summary[mt] = {
                "cv_f1": cv_score,
                "test_f1": test_metrics["f1_macro"],
            }
            
        except ImportError as e:
            logger.warning(f"Skipping {mt}: {e}")
        except Exception as e:
            logger.error(f"Error tuning {mt}: {e}")
            raise
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TUNING SUMMARY")
    logger.info(f"{'='*60}")
    for mt, scores in results_summary.items():
        logger.info(f"{mt.upper()}: CV F1={scores['cv_f1']:.4f}, Test F1={scores['test_f1']:.4f}")
    
    # Baseline comparison (coin flip baselines for imbalanced data)
    if results_summary:
        baseline = create_imbalance_baselines()
        baseline.fit(y_trainval)
        baseline_results = baseline.evaluate(y_test, class_names=CLASS_NAMES)
        
        # Compare best model against both baselines
        best_model = max(results_summary.items(), key=lambda x: x[1]["test_f1"])
        best_model_name = best_model[0].upper()
        best_model_metrics = {
            "f1_macro": best_model[1]["test_f1"],
        }
        
        print_dual_baseline_comparison(
            model_metrics=best_model_metrics,
            baseline_results=baseline_results,
            model_name=best_model_name,
            primary_metric="f1_macro",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune gradient boosting hyperparameters")
    parser.add_argument(
        "--model",
        type=str,
        choices=["lgbm", "xgb", "catboost", "rf", "all"],
        default="all",
        help="Model type to tune",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials per model",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for faster iteration (e.g., 50000)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    # Determine GPU mode: None = auto-detect, False = disabled
    use_gpu = None if not args.no_gpu else False
    
    main(model_type=args.model, n_trials=args.n_trials, sample_size=args.sample, use_gpu=use_gpu)
