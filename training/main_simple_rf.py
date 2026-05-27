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


class DataPreprocessor:
    """Preprocessor that separates fit and transform to prevent data leakage.
    
    Use fit() on training data only, then transform() on both train and test.
    This ensures test data does not influence preprocessing statistics.
    """
    
    def __init__(
        self,
        categorical_features: list[str] | None = None,
        numerical_features: list[str] | None = None,
    ):
        """Initialize preprocessor with feature lists.
        
        Args:
            categorical_features: List of categorical column names.
            numerical_features: List of numerical column names.
        """
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self.numerical_features = numerical_features or NUMERICAL_FEATURES
        self.medians: dict[str, float] = {}
        self.encoders: dict[str, LabelEncoder] = {}
        self.target_encoder: LabelEncoder | None = None
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit preprocessor on training data only.
        
        Computes medians for numerical features and fits label encoders
        for categorical features based ONLY on training data.
        
        Args:
            df: Training DataFrame (already has time features and severity mapped).
            
        Returns:
            Self for method chaining.
        """
        # Compute medians on training data only
        for col in self.numerical_features:
            if col in df.columns:
                self.medians[col] = df[col].median()
        
        # Fit encoders on training categories only
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                # Include "UNKNOWN" in fit to handle missing and unseen values
                all_values = df[col].fillna("UNKNOWN").astype(str).tolist()
                if "UNKNOWN" not in all_values:
                    all_values.append("UNKNOWN")
                le.fit(all_values)
                self.encoders[col] = le
        
        # Fit target encoder with fixed class order
        self.target_encoder = LabelEncoder()
        self.target_encoder.classes_ = np.array(CLASS_NAMES)
        
        self._is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame using fitted statistics.
        
        Args:
            df: DataFrame to transform (train or test).
            
        Returns:
            Transformed DataFrame with encoded features.
            
        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("DataPreprocessor must be fit before transform")
        
        df = df.copy()
        
        # Fill missing numerical values with training medians
        for col in self.numerical_features:
            if col in df.columns and col in self.medians:
                df[col] = df[col].fillna(self.medians[col])
        
        # Fill and encode categorical features (vectorized for speed)
        for col in self.categorical_features:
            if col in df.columns and col in self.encoders:
                df[col] = df[col].fillna("UNKNOWN").astype(str)
                le = self.encoders[col]
                
                # Create mapping dict for known classes
                class_to_idx = {cls: idx for idx, cls in enumerate(le.classes_)}
                
                # Get fallback value for unseen categories
                fallback = class_to_idx.get("UNKNOWN", -1)
                
                # Vectorized mapping with fallback
                df[col] = df[col].map(class_to_idx).fillna(fallback).astype(int)
        
        # Encode target
        if "SEVERITY_3CLASS" in df.columns and self.target_encoder is not None:
            df["SEVERITY_3CLASS"] = self.target_encoder.transform(df["SEVERITY_3CLASS"])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step (for training data).
        
        Args:
            df: Training DataFrame.
            
        Returns:
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)
    
    def get_encoders(self) -> dict[str, LabelEncoder]:
        """Get all encoders for saving with model.
        
        Returns:
            Dictionary of feature name to LabelEncoder.
        """
        encoders = dict(self.encoders)
        if self.target_encoder is not None:
            encoders["target"] = self.target_encoder
        return encoders


def prepare_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw data before train/test split.
    
    Extracts time features and maps severity, but does NOT compute
    statistics or fit encoders (those happen after split).
    
    Args:
        df: Raw DataFrame from traffic_crashes.csv.
        
    Returns:
        DataFrame with time features and severity column, rows with
        missing severity dropped.
    """
    df = df.copy()
    
    # Extract time features (deterministic, no statistics)
    df = extract_time_features(df)
    
    # Map severity target (deterministic mapping)
    df = map_severity(df)
    
    # Drop rows with missing target
    df = df.dropna(subset=["SEVERITY_3CLASS"])
    
    # Select only needed columns (keep CRASH_DATE for potential temporal split)
    all_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    columns_needed = all_features + ["SEVERITY_3CLASS", "CRASH_DATE"]
    df = df[[c for c in columns_needed if c in df.columns]].copy()
    
    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Prepare features and encode categoricals.
    
    DEPRECATED: This function computes statistics on the full dataset before
    splitting, which causes data leakage. Use prepare_raw_data() + 
    DataPreprocessor.fit_transform() instead.

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
    preprocessor: DataPreprocessor | None = None,
    metadata: dict | None = None,
) -> None:
    """Save model and encoders to disk.

    Args:
        model: Trained classifier.
        encoders: Dictionary of label encoders.
        feature_names: List of feature column names.
        metrics: Training metrics.
        output_dir: Directory to save model artifacts.
        preprocessor: Optional DataPreprocessor for leakage-free inference.
        metadata: Optional metadata dict (split type, date ranges, etc.).
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
    
    # Save preprocessor (for leakage-free inference)
    if preprocessor is not None:
        preprocessor_path = output_dir / "preprocessor.joblib"
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"Saved preprocessor to {preprocessor_path}")
    
    # Save metadata
    if metadata is not None:
        metadata_path = output_dir / "metadata.joblib"
        joblib.dump(metadata, metadata_path)
        logger.info(f"Saved metadata to {metadata_path}")


def main(
    sample_size: int | None = None,
    random_state: int = 42,
    config: DataSourceConfig | None = None,
    feature_filter: str = "drop-low",
    temporal_split: bool = False,
) -> dict:
    """Main training pipeline.

    Args:
        sample_size: Optional sample size for faster iteration.
        random_state: Random seed for reproducibility.
        config: Data source configuration (default: crash only).
        feature_filter: Feature filtering mode ('none', 'drop-low', 'drop-review').
        temporal_split: Use chronological split instead of random (prevents temporal leakage).

    Returns:
        Dictionary of evaluation metrics.
    """
    from data_preparation.resampling import temporal_train_test_split
    
    if config is None:
        config = DataSourceConfig(use_vehicles=False, use_people=False, use_weather=False)
    
    output_dir = get_output_dir(config)
    
    logger.info(f"Data configuration: {config}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Split type: {'temporal' if temporal_split else 'random stratified'}")
    logger.info("Loading traffic crashes data...")
    df = triple_merge(config=config, verbose=False)
    logger.info(f"Loaded {len(df)} crashes")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        logger.info(f"Sampling {sample_size} crashes...")
        df = df.sample(n=sample_size, random_state=random_state)
    
    # Prepare raw data (time features + severity mapping, NO encoding yet)
    logger.info("Preparing raw data (before split)...")
    df = prepare_raw_data(df)
    logger.info(f"Prepared {len(df)} samples with valid target")
    
    # ==========================================================================
    # SPLIT FIRST, THEN PREPROCESS (prevents data leakage)
    # ==========================================================================
    logger.info("Splitting data (80/20)...")
    
    if temporal_split:
        # Chronological split: train on past, test on future
        train_df, test_df = temporal_train_test_split(
            df, date_col="CRASH_DATE", test_size=0.2
        )
    else:
        # Random stratified split (original behavior)
        from sklearn.model_selection import train_test_split as sklearn_split
        train_df, test_df = sklearn_split(
            df, test_size=0.2, random_state=random_state, stratify=df["SEVERITY_3CLASS"]
        )
        logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    
    # ==========================================================================
    # FIT PREPROCESSOR ON TRAINING DATA ONLY
    # ==========================================================================
    logger.info("Fitting preprocessor on training data only...")
    preprocessor = DataPreprocessor()
    train_df = preprocessor.fit_transform(train_df)
    test_df = preprocessor.transform(test_df)
    
    encoders = preprocessor.get_encoders()
    
    # Log class distribution (on encoded training data)
    target_counts = pd.Series(train_df["SEVERITY_3CLASS"]).value_counts().sort_index()
    for idx, count in target_counts.items():
        logger.info(f"  {CLASS_NAMES[idx]}: {count} ({100*count/len(train_df):.1f}%)")
    
    # Split features and target
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    # Apply importance-based feature filtering if enabled
    if feature_filter != "none":
        X_train_df = train_df[feature_cols].copy()
        X_train_df, feature_cols, filter_result = filter_by_importance(
            X_train_df, feature_cols, mode=feature_filter
        )
        logger.info(f"Feature filter '{feature_filter}': {filter_result.n_original} -> {filter_result.n_kept} features")
        if filter_result.dropped_features:
            logger.info(f"Dropped features: {filter_result.dropped_features[:5]}{'...' if len(filter_result.dropped_features) > 5 else ''}")
        X_train = X_train_df.values
        X_test = test_df[feature_cols].values
    else:
        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
    
    y_train = train_df["SEVERITY_3CLASS"].values
    y_test = test_df["SEVERITY_3CLASS"].values
    
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
    
    # Save model with preprocessor and metadata
    metadata = {
        "split_type": "temporal" if temporal_split else "random_stratified",
        "random_state": random_state,
        "feature_filter": feature_filter,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    save_model(
        model, encoders, feature_cols, metrics, output_dir,
        preprocessor=preprocessor, metadata=metadata
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Split type: {metadata['split_type']}")
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
    parser.add_argument(
        "--temporal-split",
        action="store_true",
        help="Use chronological train/test split instead of random. Train on older data, test on newer. Prevents temporal leakage for more realistic evaluation.",
    )
    
    args = parser.parse_args()
    
    # Build data source configuration from CLI flags
    config = DataSourceConfig(
        use_vehicles=args.include_vehicle,
        use_people=args.include_people,
        use_weather=args.include_weather,
    )
    
    main(
        sample_size=args.sample,
        random_state=args.seed,
        config=config,
        feature_filter=args.feature_filter,
        temporal_split=args.temporal_split,
    )
