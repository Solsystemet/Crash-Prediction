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

from data_preparation.triple_merge import triple_merge, DataSourceConfig
from training.baselines import (
    create_imbalance_baselines,
    print_dual_baseline_comparison,
)
from training.feature_selection import filter_by_importance
from training.metrics_schema import export_model_vs_baselines_csv
from utils.logging_config import setup_logging
from utils.csv_filename_generator import generate_csv_filename

logger = setup_logging(__name__)

# Base model name for output directory
MODEL_NAME = "simple_rf"


def get_output_dir(config: DataSourceConfig) -> Path:
    """Get output directory based on data source configuration.
    
    Args:
        config: Data source configuration specifying which datasets are included.
        
    Returns:
        Path to model output directory (e.g., models/trained/simple_rf_crash_vehicle/).
    """
    return PROJECT_ROOT / "models" / "trained" / f"{MODEL_NAME}_{config.get_name_suffix()}"


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
    output_dir: Path,
) -> None:
    """Save model and encoders to disk.

    Args:
        model: Trained classifier.
        encoders: Dictionary of label encoders.
        feature_names: List of feature column names.
        metrics: Training metrics.
        output_dir: Directory to save model artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save encoders
    encoders_path = output_dir / "encoders.joblib"
    joblib.dump(encoders, encoders_path)
    logger.info(f"Saved encoders to {encoders_path}")
    
    # Save feature names
    features_path = output_dir / "feature_names.joblib"
    joblib.dump(feature_names, features_path)
    logger.info(f"Saved feature names to {features_path}")
    
    # Save metrics
    metrics_path = output_dir / "metrics.joblib"
    joblib.dump(metrics, metrics_path)
    logger.info(f"Saved metrics to {metrics_path}")


def main(
    sample_size: int | None = None,
    random_state: int = 42,
    config: DataSourceConfig | None = None,
    feature_filter: str = "drop-low",
) -> dict:
    """Main training pipeline.

    Args:
        sample_size: Optional sample size for faster iteration.
        random_state: Random seed for reproducibility.
        config: Data source configuration (default: crash only).
        feature_filter: Feature filtering mode ('none', 'drop-low', 'drop-review').

    Returns:
        Dictionary of evaluation metrics.
    """
    if config is None:
        config = DataSourceConfig(use_vehicles=False, use_people=False, use_weather=False)
    
    output_dir = get_output_dir(config)
    
    logger.info(f"Data configuration: {config}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("Loading traffic crashes data...")
    df = triple_merge(config=config, verbose=False)
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
    
    # Apply importance-based feature filtering if enabled
    if feature_filter != "none":
        X_df = df[feature_cols].copy()
        X_df, feature_cols, filter_result = filter_by_importance(
            X_df, feature_cols, mode=feature_filter
        )
        logger.info(f"Feature filter '{feature_filter}': {filter_result.n_original} -> {filter_result.n_kept} features")
        if filter_result.dropped_features:
            logger.info(f"Dropped features: {filter_result.dropped_features[:5]}{'...' if len(filter_result.dropped_features) > 5 else ''}")
        X = X_df.values
    else:
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
    
    # Baseline comparison (coin flip baselines for imbalanced data)
    baseline = create_imbalance_baselines(random_state=random_state)
    baseline.fit(y_train)
    baseline_results = baseline.evaluate(y_test, class_names=CLASS_NAMES)
    
    model_metrics = {
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
    }
    print_dual_baseline_comparison(
        model_metrics=model_metrics,
        baseline_results=baseline_results,
        model_name="Simple RF",
        primary_metric="f1_macro",
    )
    
    # Export baseline comparison CSV
    export_model_vs_baselines_csv(
        model_name="simple_rf",
        model_metrics=model_metrics,
        baseline_results=baseline_results,
        output_dir=output_dir,
    )
    logger.info(f"Saved timestamped baseline comparison to {output_dir}")
    
    # Always generate feature importance CSV (no plotting dependencies required)
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": feature_importance,
    }).sort_values("importance", ascending=False)
    
    fi_csv_filename = generate_csv_filename("feature_importance", "simple_rf")
    fi_csv_path = output_dir / fi_csv_filename
    importance_df.to_csv(fi_csv_path, index=False)
    logger.info(f"Feature importance CSV saved to {fi_csv_path}")
    
    # Generate evaluation plots (optional, requires matplotlib/seaborn)
    logger.info("\nGenerating evaluation plots...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, roc_auc_score
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # 1. Confusion matrix heatmap
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm_normalized, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix - Simple RF")
        plt.tight_layout()
        cm_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Confusion matrix saved to {cm_path}")
        
        # 2. Per-class recall bar chart
        recalls = {}
        for i, cls_name in enumerate(CLASS_NAMES):
            mask = y_test == i
            if mask.sum() > 0:
                correct = ((y_test == i) & (y_pred == i)).sum()
                recalls[cls_name] = correct / mask.sum()
            else:
                recalls[cls_name] = 0.0
        
        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = np.arange(len(CLASS_NAMES))
        recall_values = [recalls[cls] for cls in CLASS_NAMES]
        colors = ["#22c55e" if r >= 0.5 else "#f97316" if r >= 0.3 else "#ef4444" for r in recall_values]
        
        bars = ax.barh(y_pos, recall_values, color=colors, edgecolor="black", alpha=0.8)
        for bar, val in zip(bars, recall_values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center")
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Recall")
        ax.set_title("Per-Class Recall - Simple RF")
        ax.set_xlim(0, 1.1)
        ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        recall_path = output_dir / "per_class_recall.png"
        plt.savefig(recall_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Per-class recall plot saved to {recall_path}")
        
        # 3. Feature importance
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": feature_importance,
        }).sort_values("importance", ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_n = min(20, len(importance_df))
        top_features = importance_df.head(top_n)
        
        ax.barh(range(top_n), top_features["importance"].values[::-1], color="#3b82f6", alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features["feature"].values[::-1])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances - Simple RF")
        plt.tight_layout()
        
        fi_path = output_dir / "feature_importance.png"
        plt.savefig(fi_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Feature importance plot saved to {fi_path}")
        
        # 4. Confidence distribution
        max_proba = np.max(y_proba, axis=1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(max_proba, bins=20, color="#3b82f6", edgecolor="black", alpha=0.7)
        ax.axvline(x=0.5, color="red", linestyle="--", label="50% confidence")
        ax.axvline(x=np.mean(max_proba), color="green", linestyle="--", label=f"Mean: {np.mean(max_proba):.2f}")
        ax.set_xlabel("Prediction Confidence (max probability)")
        ax.set_ylabel("Frequency")
        ax.set_title("Prediction Confidence Distribution - Simple RF")
        ax.legend()
        plt.tight_layout()
        
        conf_path = output_dir / "confidence_distribution.png"
        plt.savefig(conf_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Confidence distribution saved to {conf_path}")
        
        # 5. ROC curves (one-vs-rest)
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#3b82f6", "#f97316", "#22c55e"]
        
        for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
            y_true_binary = (y_test == i).astype(int)
            y_score = y_proba[:, i]
            
            if len(np.unique(y_true_binary)) < 2:
                continue
                
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            auc = roc_auc_score(y_true_binary, y_score)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC = {auc:.3f})")
        
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves (One-vs-Rest) - Simple RF")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        roc_path = output_dir / "roc_curves.png"
        plt.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ROC curves saved to {roc_path}")
        
    except ImportError as e:
        logger.warning(f"Could not generate plots (missing dependencies): {e}")
    except Exception as e:
        logger.warning(f"Error generating plots: {e}")
    
    # Save
    save_model(model, encoders, feature_cols, metrics, output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"{'='*60}")
    
    return metrics


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
    
    # Build data source configuration from CLI flags
    config = DataSourceConfig(
        use_vehicles=args.include_vehicle,
        use_people=args.include_people,
        use_weather=args.include_weather,
    )
    
    main(sample_size=args.sample, random_state=args.seed, config=config, feature_filter=args.feature_filter)
