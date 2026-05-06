"""5-class multiclass severity classification with Focal Loss.

This module implements the 5-class crash severity model using:
- Focal Loss / LDAM Loss for handling extreme class imbalance
- Neural networks and gradient boosting models
- Probability calibration for risk scoring
- Class-specific threshold optimization

Output classes (preserved from original data):
- FATAL (~0.1%)
- INCAPACITATING INJURY (~1-2%)
- NONINCAPACITATING INJURY (~8-10%)
- REPORTED, NOT EVIDENT (~4-5%)
- NO INDICATION OF INJURY (~86%)

Usage:
    python training/multiclass_focal/main.py
    python training/multiclass_focal/main.py --sample 50000
    python training/multiclass_focal/main.py --loss focal --model nn
    python training/multiclass_focal/main.py --loss ldam --model lgbm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.feature_engineering import (
    add_binary_targets,
    add_ordinal_severity,
    engineer_all_features,
)
from data_preparation.triple_merge import triple_merge

from training.multiclass_focal.losses import FocalLoss, LDAMLoss, get_loss_function
from training.multiclass_focal.model import SeverityMLP, SeverityMLPConfig
from training.multiclass_focal.sampler import get_sampler
from training.multiclass_focal.calibration import (
    CalibratedClassifier,
    optimize_thresholds,
    predict_with_thresholds,
    plot_reliability_diagram,
    expected_calibration_error,
)
from training.multiclass_focal.evaluation import (
    evaluate_predictions,
    print_evaluation_summary,
    compare_models,
    plot_confusion_matrix,
    SEVERITY_CLASS_ORDER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "models" / "trained" / "multiclass_focal"


def stratified_sample_with_minority_boost(
    df: pd.DataFrame,
    sample_size: int,
    minority_boost: float = 5.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample dataset with controlled minority oversampling.

    Instead of including ALL minorities (which destroys class balance),
    this function:
    1. Computes target counts that BOOST minorities by a factor (e.g., 5x)
    2. But ensures majority class still dominates (prevents overprediction)
    3. Guarantees minimum samples per class for training

    Args:
        df: Full DataFrame with MOST_SEVERE_INJURY column.
        sample_size: Target sample size.
        minority_boost: Factor to oversample minorities (e.g., 5.0 = 5x natural rate)
        random_state: Random seed.

    Returns:
        Sampled DataFrame with boosted but not overwhelming minorities.
    """
    counts = df["MOST_SEVERE_INJURY"].value_counts()
    total = len(df)
    logger.info(f"Original class distribution:\n{counts}")

    # Compute natural proportions
    natural_props = counts / total

    # Apply minority boost: multiply rare class proportions
    # But cap so no class exceeds 20% of sample (to preserve majority structure)
    boosted_props = natural_props.copy()
    for cls in boosted_props.index:
        if natural_props[cls] < 0.1:  # Boost classes under 10%
            boosted_props[cls] = min(0.20, natural_props[cls] * minority_boost)

    # Renormalize to sum to 1
    boosted_props = boosted_props / boosted_props.sum()

    # Compute target counts per class
    target_counts = (boosted_props * sample_size).astype(int)

    # Ensure minimum of 50 samples per class (or all available if fewer)
    for cls in target_counts.index:
        target_counts[cls] = max(target_counts[cls], min(50, counts[cls]))

    # Adjust to hit exact sample_size
    diff = sample_size - target_counts.sum()
    majority_cls = counts.idxmax()
    if diff > 0:
        target_counts[majority_cls] += diff
    elif diff < 0:
        target_counts[majority_cls] = max(100, target_counts[majority_cls] + diff)

    logger.info(f"Target counts per class:\n{target_counts}")

    # Sample from each class
    samples = []
    for cls in counts.index:
        cls_df = df[df["MOST_SEVERE_INJURY"] == cls]
        n_to_sample = min(target_counts[cls], len(cls_df))
        if n_to_sample > 0:
            cls_sample = cls_df.sample(n=n_to_sample, random_state=random_state)
            samples.append(cls_sample)

    result = pd.concat(samples, ignore_index=True)

    # Shuffle
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)

    final_counts = result["MOST_SEVERE_INJURY"].value_counts()
    logger.info(f"Final sample distribution:\n{final_counts}")
    logger.info(f"Final sample size: {len(result)}")

    return result


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare feature matrix from DataFrame.

    Excludes target columns and post-crash investigation fields to avoid leakage.

    Returns:
        Tuple of (feature DataFrame, feature column names).
    """
    exclude_cols = {
        # Identifiers and timestamps
        "CRASH_RECORD_ID", "CRASH_DATE", "DATE_POLICE_NOTIFIED", "CRASH_DATE_EST_I",
        # Direct injury indicators (targets)
        "MOST_SEVERE_INJURY", "INJURIES_TOTAL", "INJURIES_FATAL",
        "INJURIES_INCAPACITATING", "INJURIES_NON_INCAPACITATING",
        "INJURIES_REPORTED_NOT_EVIDENT", "INJURIES_NO_INDICATION", "INJURIES_UNKNOWN",
        # Binary targets
        "IS_INJURY", "IS_SEVERE", "IS_FATAL", "IS_REPORTED",
        "SEVERITY_LEVEL", "SEVERITY_ENCODED",
        # Geographic coordinates
        "LATITUDE", "LONGITUDE", "LOCATION",
        # Post-crash investigation fields (data leakage!)
        "STATEMENTS_TAKEN_I", "WORK_ZONE_I", "PHOTOS_TAKEN_I", "DOORING_I",
        # People-derived outcome features (data leakage!)
        "ejected_any", "using_seatbelt_mean", "cell_phone_any",
        "bac_positive_any", "bac_clean_max",
        # CRASH_TYPE directly encodes injury!
        "CRASH_TYPE", "REPORT_TYPE",
        # Redundant temporal
        "HOUR",
        # High-cardinality location features
        "STREET_NAME",
    }

    # Categorical columns to encode
    categorical_cols = [
        "WEATHER_CONDITION", "LIGHTING_CONDITION", "FIRST_CRASH_TYPE",
        "TRAFFICWAY_TYPE", "ROADWAY_SURFACE_COND", "TRAFFIC_CONTROL_DEVICE",
        "DEVICE_CONDITION", "ALIGNMENT", "ROAD_DEFECT",
        "PRIM_CONTRIBUTORY_CAUSE", "DAMAGE",
    ]

    # Get numeric columns
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            feature_cols.append(col)

    df_features = df[feature_cols].copy()

    # Encode categoricals
    for col in categorical_cols:
        if col in df.columns and col not in exclude_cols:
            le = LabelEncoder()
            values = df[col].fillna("UNKNOWN").astype(str)
            df_features[col] = le.fit_transform(values)
            if col not in feature_cols:
                feature_cols.append(col)

    feature_cols = list(df_features.columns)
    df_features = df_features.fillna(0)

    return df_features, feature_cols


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_counts: np.ndarray,
    loss_type: Literal["focal", "ldam", "ce"] = "focal",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> tuple[SeverityMLP, dict]:
    """Train neural network with focal/LDAM loss.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        class_counts: Number of samples per class
        loss_type: Loss function ('focal', 'ldam', 'ce')
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        seed: Random seed

    Returns:
        Tuple of (trained model, training history)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    config = SeverityMLPConfig(
        input_dim=X_train.shape[1],
        num_classes=5,
        hidden_dims=[256, 256, 128],
        dropout_rate=0.3,
        batch_norm=True,
    )
    model = SeverityMLP(config).to(device)

    # Create loss function
    loss_fn = get_loss_function(
        loss_type=loss_type,
        class_counts=class_counts,
        gamma=2.0 if loss_type == "focal" else 0.0,
    )
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create data loaders with class-balanced sampling
    sampler = get_sampler(y_train, strategy="sqrt", batch_size=batch_size, seed=seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
    )

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_val_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()
            val_preds = val_logits.argmax(dim=1).cpu().numpy()

        # Compute validation F1
        from sklearn.metrics import f1_score
        val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)

        # Save best model
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state = model.state_dict().copy()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_macro_f1={val_macro_f1:.4f}"
            )

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(f"Best validation macro F1: {best_val_f1:.4f}")

    return model, history


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_counts: np.ndarray,
    seed: int = 42,
) -> tuple:
    """Train LightGBM with class weighting.

    Returns:
        Tuple of (trained model, None for history compatibility)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not available, skipping")
        return None, None

    # Compute class weights (sqrt-balanced)
    freq = class_counts / class_counts.sum()
    weights = np.sqrt(1.0 / (freq + 1e-8))
    weights = weights / weights.sum() * len(class_counts)
    sample_weights = weights[y_train]

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": seed,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    return model, None


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_counts: np.ndarray,
    seed: int = 42,
) -> tuple:
    """Train CatBoost with auto class balancing.

    Returns:
        Tuple of (trained model, None for history compatibility)
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        logger.warning("CatBoost not available, skipping")
        return None, None

    # Compute class weights
    freq = class_counts / class_counts.sum()
    weights = np.sqrt(1.0 / (freq + 1e-8))
    class_weights = {i: w for i, w in enumerate(weights)}

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        loss_function="MultiClass",
        class_weights=class_weights,
        early_stopping_rounds=50,
        verbose=100,
        random_seed=seed,
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    return model, None


def predict_proba(model, X: np.ndarray, model_type: str, device: str = "cpu") -> np.ndarray:
    """Get probability predictions from any model type.

    Args:
        model: Trained model
        X: Features
        model_type: One of 'nn', 'lgbm', 'catboost'
        device: Device for neural network

    Returns:
        Probability matrix (n_samples, n_classes)
    """
    if model_type == "nn":
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(device)
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    elif model_type == "lgbm":
        probs = model.predict(X)
    elif model_type == "catboost":
        probs = model.predict_proba(X)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return probs


def main(
    sample_size: int | None = None,
    loss_type: Literal["focal", "ldam", "ce"] = "focal",
    model_type: Literal["nn", "lgbm", "catboost", "all"] = "all",
    epochs: int = 50,
    seed: int = 42,
) -> dict:
    """Main training pipeline.

    Args:
        sample_size: Number of samples to use (None for all)
        loss_type: Loss function for neural network
        model_type: Model type to train ('nn', 'lgbm', 'catboost', 'all')
        epochs: Training epochs for neural network
        seed: Random seed

    Returns:
        Dictionary of results for each model
    """
    logger.info("=" * 60)
    logger.info("5-CLASS MULTICLASS SEVERITY CLASSIFICATION")
    logger.info(f"Loss: {loss_type}, Model: {model_type}")
    logger.info("=" * 60)

    # Load and prepare data
    logger.info("Loading data...")
    df = triple_merge()
    logger.info(f"Loaded {len(df)} samples")

    # Sample if requested
    if sample_size and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} records...")
        df = stratified_sample_with_minority_boost(df, sample_size, minority_boost=5.0, random_state=seed)

    # Feature engineering
    logger.info("Engineering features...")
    df = engineer_all_features(df)
    df = add_binary_targets(df)

    # Prepare features
    logger.info("Preparing features...")
    X_df, feature_cols = prepare_features(df)
    logger.info(f"Using {len(feature_cols)} features")

    # Encode target
    le = LabelEncoder()
    le.classes_ = np.array(SEVERITY_CLASS_ORDER)
    y = le.transform(df["MOST_SEVERE_INJURY"].fillna("NO INDICATION OF INJURY"))

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_counts = np.zeros(5, dtype=np.int64)
    for u, c in zip(unique, counts):
        class_counts[u] = c
    logger.info("Class distribution:")
    for i, name in enumerate(SEVERITY_CLASS_ORDER):
        logger.info(f"  {name}: {class_counts[i]} ({100*class_counts[i]/len(y):.2f}%)")

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    # Train/val/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, stratify=y_trainval, random_state=seed
    )

    logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # Recompute class counts on training set
    train_counts = np.bincount(y_train, minlength=5)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Train models
    results = {}
    models = {}

    if model_type in ["nn", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info(f"TRAINING NEURAL NETWORK ({loss_type.upper()} LOSS)")
        logger.info("=" * 60)

        model_nn, history = train_neural_network(
            X_train, y_train, X_val, y_val, train_counts,
            loss_type=loss_type, epochs=epochs, device=device, seed=seed,
        )
        models["nn"] = model_nn

        # Evaluate
        probs = predict_proba(model_nn, X_test, "nn", device)
        preds = probs.argmax(axis=1)
        result = evaluate_predictions(y_test, preds, probs, SEVERITY_CLASS_ORDER)
        results[f"nn_{loss_type}"] = result
        print_evaluation_summary(result, f"Neural Network ({loss_type})")

    if model_type in ["lgbm", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING LIGHTGBM")
        logger.info("=" * 60)

        model_lgbm, _ = train_lightgbm(
            X_train, y_train, X_val, y_val, train_counts, seed=seed
        )
        if model_lgbm is not None:
            models["lgbm"] = model_lgbm
            probs = predict_proba(model_lgbm, X_test, "lgbm")
            preds = probs.argmax(axis=1)
            result = evaluate_predictions(y_test, preds, probs, SEVERITY_CLASS_ORDER)
            results["lgbm"] = result
            print_evaluation_summary(result, "LightGBM")

    if model_type in ["catboost", "all"]:
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING CATBOOST")
        logger.info("=" * 60)

        model_cb, _ = train_catboost(
            X_train, y_train, X_val, y_val, train_counts, seed=seed
        )
        if model_cb is not None:
            models["catboost"] = model_cb
            probs = predict_proba(model_cb, X_test, "catboost")
            preds = probs.argmax(axis=1)
            result = evaluate_predictions(y_test, preds, probs, SEVERITY_CLASS_ORDER)
            results["catboost"] = result
            print_evaluation_summary(result, "CatBoost")

    # Model comparison
    if len(results) > 1:
        logger.info("\n" + compare_models(results))

    # Calibration and threshold optimization for best model
    if results:
        best_name = max(results.keys(), key=lambda k: results[k].macro_f1)
        best_model = models.get(best_name.split("_")[0] if "_" in best_name else best_name)
        model_type_key = best_name.split("_")[0] if "_" in best_name else best_name

        logger.info("\n" + "=" * 60)
        logger.info(f"CALIBRATION & THRESHOLDS FOR BEST MODEL: {best_name}")
        logger.info("=" * 60)

        # Get validation probabilities
        val_probs = predict_proba(best_model, X_val, model_type_key, device)

        # Calibrate
        calibrator = CalibratedClassifier(method="isotonic")
        calibrator.fit(val_probs, y_val)

        # Test with calibration
        test_probs = predict_proba(best_model, X_test, model_type_key, device)
        calibrated_probs = calibrator.calibrate(test_probs)

        # ECE before/after
        ece_before = expected_calibration_error(test_probs, y_test)
        ece_after = expected_calibration_error(calibrated_probs, y_test)
        logger.info(f"ECE before calibration: {ece_before:.4f}")
        logger.info(f"ECE after calibration:  {ece_after:.4f}")

        # Optimize thresholds
        val_calibrated = calibrator.calibrate(val_probs)
        threshold_result = optimize_thresholds(val_calibrated, y_val, metric="f1")

        logger.info("\nOptimal thresholds per class:")
        for i, name in enumerate(SEVERITY_CLASS_ORDER):
            logger.info(
                f"  {name}: {threshold_result.thresholds[i]:.3f} "
                f"(F1={threshold_result.f1_scores[i]:.3f})"
            )

        # Final predictions with thresholds
        final_preds = predict_with_thresholds(
            calibrated_probs, threshold_result.thresholds, default_class=2
        )
        final_result = evaluate_predictions(
            y_test, final_preds, calibrated_probs, SEVERITY_CLASS_ORDER
        )
        print_evaluation_summary(final_result, f"{best_name} (calibrated + thresholds)")

        results[f"{best_name}_calibrated"] = final_result

        # ========================================
        # SAVE DEPLOYABLE MODEL ARTIFACTS
        # ========================================
        logger.info("\n" + "=" * 60)
        logger.info("SAVING MODEL ARTIFACTS FOR DEPLOYMENT")
        logger.info("=" * 60)

        import pickle

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "model_type": model_type_key,
            "class_names": SEVERITY_CLASS_ORDER,
            "feature_columns": feature_cols,
            "thresholds": threshold_result.thresholds.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }

        # Save neural network
        if model_type_key == "nn":
            model_path = OUTPUT_DIR / f"model_nn_{timestamp}.pt"
            torch.save({
                "model_state_dict": best_model.state_dict(),
                "config": best_model.config,
            }, model_path)
            artifacts["model_path"] = str(model_path.name)
            logger.info(f"  Neural network saved to {model_path}")
        elif model_type_key == "lgbm":
            model_path = OUTPUT_DIR / f"model_lgbm_{timestamp}.txt"
            best_model.save_model(str(model_path))
            artifacts["model_path"] = str(model_path.name)
            logger.info(f"  LightGBM model saved to {model_path}")
        elif model_type_key == "catboost":
            model_path = OUTPUT_DIR / f"model_catboost_{timestamp}.cbm"
            best_model.save_model(str(model_path))
            artifacts["model_path"] = str(model_path.name)
            logger.info(f"  CatBoost model saved to {model_path}")

        # Save calibrator
        calibrator_path = OUTPUT_DIR / f"calibrator_{timestamp}.pkl"
        with open(calibrator_path, "wb") as f:
            pickle.dump(calibrator, f)
        artifacts["calibrator_path"] = str(calibrator_path.name)
        logger.info(f"  Calibrator saved to {calibrator_path}")

        # Save artifacts metadata
        artifacts_path = OUTPUT_DIR / f"artifacts_{timestamp}.json"
        with open(artifacts_path, "w") as f:
            json.dump(artifacts, f, indent=2)
        logger.info(f"  Artifacts metadata saved to {artifacts_path}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = OUTPUT_DIR / f"results_{timestamp}.json"
    results_dict = {name: r.to_dict() for name, r in results.items()}
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 5-class severity models")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Number of samples to use (default: all)"
    )
    parser.add_argument(
        "--loss", type=str, default="focal", choices=["focal", "ldam", "ce"],
        help="Loss function for neural network"
    )
    parser.add_argument(
        "--model", type=str, default="all", choices=["nn", "lgbm", "catboost", "all"],
        help="Model type to train"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs for neural network"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    main(
        sample_size=args.sample,
        loss_type=args.loss,
        model_type=args.model,
        epochs=args.epochs,
        seed=args.seed,
    )
