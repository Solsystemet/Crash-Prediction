"""Multiclass neural network training pipeline for crash severity prediction.

This module implements direct 5-class classification using a neural network,
as an alternative to the hierarchical binary cascade approach.

Target classes (ordered by severity):
- 0: NO INDICATION OF INJURY
- 1: REPORTED, NOT EVIDENT
- 2: NONINCAPACITATING INJURY
- 3: INCAPACITATING INJURY
- 4: FATAL

Usage:
    # Default configuration
    python main_multiclass.py

    # Custom configuration
    python main_multiclass.py --epochs 150 --batch-size 128 --hidden 512,256,128

    # Quick test with small sample
    python main_multiclass.py --sample-size 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.feature_engineering import engineer_all_features
from data_preparation.triple_merge import triple_merge

from training.multiclass import (
    MulticlassNeuralConfig,
    MulticlassNeuralClassifier,
)
from training.multiclass.evaluation import (
    evaluate_multiclass,
    print_evaluation_summary,
    compare_with_baseline,
    get_roc_curve_data,
    plot_multiclass_roc_curves,
)
from training.feature_selection import correlation_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Severity class labels in order (0-4)
SEVERITY_CLASSES = [
    "NO INDICATION OF INJURY",
    "REPORTED, NOT EVIDENT",
    "NONINCAPACITATING INJURY",
    "INCAPACITATING INJURY",
    "FATAL",
]

# Short names for display
SEVERITY_SHORT_NAMES = [
    "NO_INJURY",
    "REPORTED",
    "NON_INCAP",
    "INCAP",
    "FATAL",
]


def stratified_sample_with_fatal(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample dataset ensuring FATAL cases are well-represented.

    Takes ALL FATAL cases plus stratified sample of the rest to ensure
    the rare class is adequately represented.

    Args:
        df: Full DataFrame with MOST_SEVERE_INJURY column.
        sample_size: Target sample size.
        random_state: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with all FATAL cases included.
    """
    fatal_mask = df["MOST_SEVERE_INJURY"] == "FATAL"
    fatal_df = df[fatal_mask]
    other_df = df[~fatal_mask]

    n_fatal = len(fatal_df)
    n_other = sample_size - n_fatal

    logger.info(f"  FATAL cases in full data: {n_fatal}")

    if n_other > 0 and len(other_df) > n_other:
        other_sample = other_df.sample(n=n_other, random_state=random_state)
    else:
        other_sample = other_df

    result = pd.concat([fatal_df, other_sample], ignore_index=True)
    logger.info(f"  Stratified sample: {n_fatal} FATAL + {len(other_sample)} other = {len(result)}")

    return result.sample(frac=1, random_state=random_state).reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare feature matrix from DataFrame.

    Args:
        df: DataFrame with engineered features.

    Returns:
        Tuple of (feature DataFrame, feature column names).
    """
    # Columns to exclude (targets, IDs, post-crash data leakage)
    exclude_cols = {
        # Identifiers and timestamps
        "CRASH_RECORD_ID",
        "CRASH_DATE",
        "DATE_POLICE_NOTIFIED",
        "CRASH_DATE_EST_I",
        # Direct injury indicators (targets)
        "MOST_SEVERE_INJURY",
        "INJURIES_TOTAL",
        "INJURIES_FATAL",
        "INJURIES_INCAPACITATING",
        "INJURIES_NON_INCAPACITATING",
        "INJURIES_REPORTED_NOT_EVIDENT",
        "INJURIES_NO_INDICATION",
        "INJURIES_UNKNOWN",
        # Binary targets (from hierarchical)
        "IS_INJURY",
        "IS_SEVERE",
        "IS_FATAL",
        "IS_REPORTED",
        "SEVERITY_LEVEL",
        "SEVERITY_ENCODED",
        # Geographic coordinates (use clusters instead)
        "LATITUDE",
        "LONGITUDE",
        "LOCATION",
        # Post-crash investigation fields (data leakage!)
        "STATEMENTS_TAKEN_I",
        "WORK_ZONE_I",
        "PHOTOS_TAKEN_I",
        "DOORING_I",
        # People-derived outcome features (data leakage!)
        "ejected_any",
        "using_seatbelt_mean",
        "cell_phone_any",
        "bac_positive_any",
        "bac_clean_max",
        # CRASH_TYPE directly encodes injury!
        "CRASH_TYPE",
        "REPORT_TYPE",
        # Redundant temporal
        "HOUR",
        # High-cardinality location features
        "STREET_NAME",
    }

    # Safe categorical features known BEFORE crash outcome
    safe_categorical_cols = [
        "WEATHER_CONDITION",
        "LIGHTING_CONDITION",
        "FIRST_CRASH_TYPE",
        "TRAFFICWAY_TYPE",
        "ROADWAY_SURFACE_COND",
        "TRAFFIC_CONTROL_DEVICE",
        "DEVICE_CONDITION",
        "ALIGNMENT",
        "ROAD_DEFECT",
        "PRIM_CONTRIBUTORY_CAUSE",
        "DAMAGE",
    ]

    # Get numeric columns
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            feature_cols.append(col)

    # Create copy with numeric features
    df_features = df[feature_cols].copy()

    # Add label-encoded safe categorical features
    label_encoders = {}  # Store encoders for inference
    for col in safe_categorical_cols:
        if col in df.columns and col not in exclude_cols:
            le = LabelEncoder()
            values = df[col].fillna("UNKNOWN").astype(str)
            df_features[col] = le.fit_transform(values)
            label_encoders[col] = le  # Save for later
            feature_cols.append(col)

    feature_cols = list(df_features.columns)

    # Fill any remaining NaN
    df_features = df_features.fillna(0)

    return df_features, feature_cols, label_encoders


def encode_severity_target(df: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder]:
    """Encode severity target to 0-4 integer labels.

    Args:
        df: DataFrame with MOST_SEVERE_INJURY column.

    Returns:
        Tuple of (encoded labels, label encoder).
    """
    le = LabelEncoder()
    le.classes_ = np.array(SEVERITY_CLASSES)
    y = le.transform(df["MOST_SEVERE_INJURY"])
    return y, le


def run_multiclass_pipeline(
    config: MulticlassNeuralConfig | None = None,
    sample_size: int | None = None,
    correlation_threshold: float = 0.95,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    save_model: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full multiclass classification pipeline.

    Args:
        config: Model configuration. Uses defaults if None.
        sample_size: Optional limit on dataset size for faster experimentation.
        correlation_threshold: Threshold for correlation-based feature filtering.
        test_size: Proportion of data for test set.
        val_size: Proportion of training data for validation.
        random_state: Random seed for reproducibility.
        save_model: Whether to save the trained model.
        output_dir: Directory for output files.

    Returns:
        Dictionary containing evaluation results and model.
    """
    logger.info("=" * 60)
    logger.info("MULTICLASS NEURAL NETWORK PIPELINE")
    logger.info("=" * 60)

    config = config or MulticlassNeuralConfig()
    output_dir = output_dir or PROJECT_ROOT / "models" / "trained" / "multiclass_nn"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Configuration: {config}")

    # Load and merge data
    logger.info("\n[1/7] Loading and merging data...")
    df = triple_merge()
    logger.info(f"Merged dataset shape: {df.shape}")

    # Sample if specified
    if sample_size is not None and len(df) > sample_size:
        df = stratified_sample_with_fatal(df, sample_size, random_state)

    # Drop rows with missing target (can occur from joins in triple_merge)
    initial_len = len(df)
    df = df.dropna(subset=["MOST_SEVERE_INJURY"])
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with missing MOST_SEVERE_INJURY")

    # Engineer features
    logger.info("\n[2/7] Engineering features...")
    df = engineer_all_features(df, include_interactions=True, include_clusters=False)

    # Show class distribution
    class_dist = df["MOST_SEVERE_INJURY"].value_counts()
    logger.info("\nClass distribution:")
    for cls in SEVERITY_CLASSES:
        count = class_dist.get(cls, 0)
        pct = 100 * count / len(df)
        logger.info(f"  {cls}: {count:,} ({pct:.2f}%)")

    # Prepare features
    logger.info("\n[3/7] Preparing feature matrix...")
    X_df, feature_cols, label_encoders = prepare_features(df)
    logger.info(f"Feature matrix shape (before filtering): {X_df.shape}")

    # Apply correlation filtering
    X_filtered, kept_indices, removed_features = correlation_filter(
        X_df.values,
        feature_names=feature_cols,
        threshold=correlation_threshold,
        verbose=config.verbose,
    )
    filtered_feature_cols = [feature_cols[i] for i in kept_indices]
    logger.info(f"Feature matrix shape (after filtering): {X_filtered.shape}")
    logger.info(f"Removed {len(removed_features)} correlated features")

    # Scale features
    logger.info("\n[4/7] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Encode target
    y, label_encoder = encode_severity_target(df)
    logger.info(f"Target shape: {y.shape}")
    logger.info(f"Classes: {list(label_encoder.classes_)}")

    # Split data
    logger.info("\n[5/7] Splitting data...")

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_trainval,
    )

    logger.info(f"  Train: {len(X_train):,} samples")
    logger.info(f"  Val:   {len(X_val):,} samples")
    logger.info(f"  Test:  {len(X_test):,} samples")

    # Train model
    logger.info("\n[6/7] Training model...")
    classifier = MulticlassNeuralClassifier(config)
    classifier.fit(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    logger.info("\n[7/7] Evaluating on test set...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)

    # Compute metrics
    results = evaluate_multiclass(
        y_test, y_pred, y_proba,
        class_names=SEVERITY_SHORT_NAMES,
    )

    # Print summary
    print_evaluation_summary(results)

    # Compare to baselines
    baseline_comparison = compare_with_baseline(
        y_test, y_pred, model_name="Multiclass NN"
    )

    logger.info("\nBaseline Comparison:")
    for model_name, metrics in baseline_comparison.items():
        logger.info(
            f"  {model_name:20s}: "
            f"acc={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )

    # Save model and results
    if save_model:
        model_path = output_dir / "multiclass_nn.pt"
        classifier.save(model_path)

        # Save the scaler for inference
        import joblib
        scaler_path = output_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # Save label encoders for categorical features
        encoders_path = output_dir / "label_encoders.joblib"
        joblib.dump(label_encoders, encoders_path)
        logger.info(f"Label encoders saved to {encoders_path}")

        # Save ROC curve data
        roc_data = get_roc_curve_data(y_test, y_proba, SEVERITY_SHORT_NAMES)
        roc_path = output_dir / "roc_data.json"
        with open(roc_path, "w") as f:
            json.dump(roc_data, f, indent=2)
        logger.info(f"ROC data saved to {roc_path}")

        # Plot ROC curves
        roc_plot_path = output_dir / "roc_curves_multiclass.png"
        plot_multiclass_roc_curves(
            y_test, y_proba, SEVERITY_SHORT_NAMES, roc_plot_path,
            title="ROC Curves - Multiclass Neural Network"
        )

        # Save metrics summary
        metrics_summary = {
            "accuracy": results.accuracy,
            "macro_f1": results.macro_f1,
            "weighted_f1": results.weighted_f1,
            "roc_auc_macro": results.roc_auc_macro,
            "per_class": {
                m.name: {"precision": m.precision, "recall": m.recall, "f1": m.f1}
                for m in results.per_class_metrics
            },
            "confusion_matrix": results.confusion_matrix.tolist(),
            "class_names": SEVERITY_SHORT_NAMES,
            "baseline_comparison": baseline_comparison,
        }
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

        # Save feature list
        features_path = output_dir / "features.json"
        with open(features_path, "w") as f:
            json.dump(filtered_feature_cols, f, indent=2)
        logger.info(f"Feature list saved to {features_path}")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "classifier": classifier,
        "results": results,
        "baseline_comparison": baseline_comparison,
        "feature_cols": filtered_feature_cols,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "label_encoder": label_encoder,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multiclass neural network for crash severity prediction"
    )

    # Data options
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit dataset size for faster experimentation",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Threshold for correlation-based feature filtering (default: 0.95)",
    )

    # Architecture options
    parser.add_argument(
        "--hidden",
        type=str,
        default="256,128,64",
        help="Comma-separated hidden layer sizes (default: 256,128,64)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability (default: 0.3)",
    )
    parser.add_argument(
        "--no-batch-norm",
        action="store_true",
        help="Disable batch normalization",
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )

    # Loss options
    parser.add_argument(
        "--loss",
        type=str,
        choices=["focal", "cross_entropy"],
        default="focal",
        help="Loss function (default: focal)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (default: 2.0)",
    )

    # Other options
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Build config from args
    hidden_sizes = tuple(int(x) for x in args.hidden.split(","))

    config = MulticlassNeuralConfig(
        hidden_sizes=hidden_sizes,
        dropout=args.dropout,
        use_batch_norm=not args.no_batch_norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        verbose=not args.quiet,
    )

    run_multiclass_pipeline(
        config=config,
        sample_size=args.sample_size,
        correlation_threshold=args.correlation_threshold,
        save_model=not args.no_save,
    )


if __name__ == "__main__":
    main()
