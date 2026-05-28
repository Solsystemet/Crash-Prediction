"""Crash count regression pipeline.

This module implements a neural network ensemble for predicting crash counts
within time windows (hourly, daily, weekly). Uses a two-level approach:
- Global model: Trained on all data, captures city-wide patterns
- Zone models: Per-zone adjustments that capture local effects

The model uses historical crash patterns (lags, rolling statistics, trends)
to predict future counts. Time-based features (cyclical hour/day encodings)
are also included.

Usage:
    # Default (hourly granularity, 10 zones)
    python main_regression.py

    # Daily predictions with custom zones
    python main_regression.py --granularity daily --n-zones 15

    # Quick test run with sample
    python main_regression.py --sample 10000 --epochs 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.aggregate_time_series import (
    TimeSeriesConfig,
    aggregate_crashes_by_time,
    aggregate_with_zones,
)
from data_preparation.time_series_features import (
    FeatureConfig,
    engineer_time_series_features,
)
from data_preparation.triple_merge import triple_merge, DataSourceConfig
from splice.k_means import LocationClusterer

from training.regression.config import RegressionConfig, EnsembleConfig
from training.regression.dataset import create_per_zone_splits, prepare_datasets
from training.regression.ensemble import CrashCountEnsemble
from training.regression.evaluation import (
    evaluate_regression,
    plot_predictions,
    plot_zone_comparison,
    generate_report,
)
from training.regression.predict import CrashCountPredictor
from training.baselines import (
    RegressionBaseline,
    compare_to_baseline,
    log_comparison,
    print_baseline_comparison_box,
    add_baseline_args,
)
from training.metrics_schema import export_model_vs_baselines_csv
from utils.logging_config import setup_logging

logger = setup_logging(__name__)


def get_output_dir(granularity: str, config: DataSourceConfig) -> Path:
    """Get output directory based on granularity and data source configuration.
    
    Args:
        granularity: Time granularity ('hourly', 'daily', 'weekly').
        config: Data source configuration specifying which datasets are included.
        
    Returns:
        Path to model output directory.
    """
    return PROJECT_ROOT / "models" / "trained" / f"regression_{granularity}_{config.get_name_suffix()}"


def run_regression_pipeline(
    granularity: Literal["hourly", "daily", "weekly"] = "hourly",
    n_zones: int = 10,
    sample_size: int | None = None,
    epochs: int = 100,
    batch_size: int = 64,
    save_model: bool = True,
    with_baseline: bool = True,
    data_config: DataSourceConfig | None = None,
) -> dict:
    """Run the full crash count regression pipeline.

    Args:
        granularity: Time bucket size ('hourly', 'daily', 'weekly').
        n_zones: Number of geographic zones for clustering.
        sample_size: Optional limit on dataset size for faster experimentation.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        save_model: Whether to save the trained model.
        data_config: Data source configuration (default: crash only).

    Returns:
        Dictionary of evaluation results.
    """
    if data_config is None:
        data_config = DataSourceConfig(use_vehicles=False, use_people=False, use_weather=False)
    
    output_dir = get_output_dir(granularity, data_config)
    
    logger.info("=" * 60)
    logger.info("CRASH COUNT REGRESSION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Time granularity: {granularity}")
    logger.info(f"Number of zones: {n_zones}")
    logger.info(f"Data configuration: {data_config}")
    logger.info(f"Output directory: {output_dir}")
    if sample_size:
        logger.info(f"Sample size: {sample_size}")

    # Create configurations
    regression_config = RegressionConfig(
        time_granularity=granularity,
        n_zones=n_zones,
        epochs=epochs,
        batch_size=batch_size,
    )
    time_config = TimeSeriesConfig(
        granularity=granularity,
        fill_gaps=True,
    )
    feature_config = FeatureConfig.for_granularity(granularity)

    # Step 1: Load and merge data
    logger.info("\n[1/7] Loading and merging data...")
    df = triple_merge(config=data_config, verbose=False)
    logger.info(f"Merged dataset shape: {df.shape}")

    # Sample if specified
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=regression_config.random_state)
        logger.info(f"Sampled to {len(df)} rows")

    # Step 2: Geographic clustering (zones)
    logger.info(f"\n[2/7] Creating {n_zones} geographic zones...")

    # Filter rows with valid coordinates
    valid_coords = df[["LATITUDE", "LONGITUDE"]].notna().all(axis=1)
    df_valid = df[valid_coords].copy()
    logger.info(f"Rows with valid coordinates: {len(df_valid)}")

    clusterer = LocationClusterer(n_clusters=n_zones, random_state=regression_config.random_state)
    clusterer.fit(df_valid)
    zone_labels = clusterer.transform(df_valid)
    df_valid["LOCATION_CLUSTER"] = zone_labels

    # Store centroids for inference
    zone_centroids = clusterer.centroids.numpy() if clusterer.centroids is not None else None

    logger.info(f"Zone distribution: {pd.Series(zone_labels).value_counts().to_dict()}")

    # Step 3: Aggregate to time series
    logger.info(f"\n[3/7] Aggregating crashes to {granularity} counts...")
    time_config.zone_col = "LOCATION_CLUSTER"
    ts_df = aggregate_crashes_by_time(df_valid, time_config)
    logger.info(f"Time series shape: {ts_df.shape}")
    logger.info(f"Time range: {ts_df['timestamp'].min()} to {ts_df['timestamp'].max()}")
    logger.info(f"Crash count stats: {ts_df['crash_count'].describe().to_dict()}")

    # Step 4: Engineer time-series features
    logger.info("\n[4/7] Engineering time-series features...")
    ts_df, feature_cols = engineer_time_series_features(
        ts_df,
        config=feature_config,
        group_col="zone_id",
        timestamp_col="timestamp",
        drop_na=True,
    )
    logger.info(f"After feature engineering: {ts_df.shape}")
    logger.info(f"Features ({len(feature_cols)}): {feature_cols[:10]}...")

    # Step 5: Temporal train/val/test split
    logger.info("\n[5/7] Splitting data temporally...")
    train_df, val_df, test_df = create_per_zone_splits(
        ts_df,
        train_ratio=regression_config.train_ratio,
        val_ratio=regression_config.val_ratio,
        timestamp_col="timestamp",
        zone_col="zone_id",
    )
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    logger.info(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    # Prepare numpy arrays
    X_train = train_df[feature_cols].values
    y_train = train_df["crash_count"].values
    zone_ids_train = train_df["zone_id"].values

    X_val = val_df[feature_cols].values
    y_val = val_df["crash_count"].values
    zone_ids_val = val_df["zone_id"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["crash_count"].values
    zone_ids_test = test_df["zone_id"].values

    # Step 6: Train ensemble
    logger.info("\n[6/7] Training ensemble model...")
    ensemble_config = EnsembleConfig(
        base_config=regression_config,
        global_weight=0.7,
        train_zone_models=True,
    )
    ensemble = CrashCountEnsemble(regression_config, ensemble_config)
    ensemble.feature_cols = feature_cols

    ensemble.fit(
        X_train, y_train, zone_ids_train,
        X_val, y_val, zone_ids_val,
    )

    # Step 7: Evaluate
    logger.info("\n[7/7] Evaluating on test set...")

    # Predictions
    y_pred = ensemble.predict(X_test, zone_ids_test)

    # Also get global-only predictions for comparison
    y_pred_global = ensemble.predict_global(X_test)

    # Evaluate
    results = evaluate_regression(y_test, y_pred, zone_ids_test)

    # Compare with global-only
    from training.regression.evaluation import compute_metrics
    global_metrics = compute_metrics(y_test, y_pred_global)
    logger.info(f"\nGlobal-only baseline:")
    logger.info(f"  MAE: {global_metrics['mae']:.4f}")
    logger.info(f"  R²:  {global_metrics['r2']:.4f}")

    improvement = (
        (global_metrics["mae"] - results["overall"]["mae"]) / global_metrics["mae"] * 100
    )
    logger.info(f"\nEnsemble improvement: {improvement:.1f}% lower MAE")

    # Dummy baselines (mean/median)
    baseline_results = {}
    if with_baseline:
        baseline = RegressionBaseline(strategies=["mean", "median"])
        baseline.fit(y_train)
        baseline_results = baseline.evaluate(y_test)

        # Compare model to baseline
        model_metrics = {
            "mae": results["overall"]["mae"],
            "rmse": results["overall"]["rmse"],
            "r2": results["overall"]["r2"],
        }
        comparison = compare_to_baseline(model_metrics, baseline_results)

        # Print comparison box
        print_baseline_comparison_box(
            model_metrics=model_metrics,
            baseline_results=baseline_results,
            comparison=comparison,
            model_name="Ensemble",
            baseline_strategy="mean",
            primary_metric="mae",
            metric_labels={
                "mae": "MAE",
                "rmse": "RMSE",
                "r2": "R²",
            },
            task_type="regression",
        )

        # Export baseline comparison CSV
        export_model_vs_baselines_csv(
            model_name="regression_ensemble",
            model_metrics=model_metrics,
            baseline_results=baseline_results,
            output_dir=output_dir,
        )
        logger.info(f"Saved timestamped baseline comparison to {output_dir}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Granularity: {granularity}")
    logger.info(f"Zones: {n_zones}")
    logger.info(f"Test MAE: {results['overall']['mae']:.4f}")
    logger.info(f"Test RMSE: {results['overall']['rmse']:.4f}")
    logger.info(f"Test R²: {results['overall']['r2']:.4f}")

    # Save model
    if save_model:
        ensemble.save(output_dir)

        # Export training history to CSV
        ensemble.export_training_history(output_dir)

        # Save additional metadata for predictor
        predictor_metadata = {
            "feature_config": feature_config,
            "time_config": time_config,
            "zone_centroids": zone_centroids,
        }
        joblib.dump(predictor_metadata, output_dir / "predictor_metadata.joblib")

        logger.info(f"Model saved to {output_dir}")

        # Save plots
        plots_dir = PROJECT_ROOT / "models" / "plots"
        plots_dir.mkdir(exist_ok=True)

        plot_predictions(
            y_test, y_pred,
            timestamps=test_df["timestamp"].values,
            title=f"Crash Count Prediction ({granularity})",
            save_path=plots_dir / f"regression_{granularity}_predictions.png",
        )

        plot_zone_comparison(
            results,
            metric="mae",
            save_path=plots_dir / f"regression_{granularity}_zones.png",
        )

        report = generate_report(results, save_path=output_dir / "evaluation_report.txt")

    return {
        "overall": results["overall"],
        "per_zone": results.get("per_zone", {}),
        "global_baseline": global_metrics,
        "improvement_pct": improvement,
        "baseline": {
            strategy: result.metrics
            for strategy, result in baseline_results.items()
        } if baseline_results else {},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train crash count regression model"
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["hourly", "daily", "weekly"],
        default="hourly",
        help="Time bucket size for predictions (default: hourly)",
    )
    parser.add_argument(
        "--n-zones",
        type=int,
        default=10,
        help="Number of geographic zones (default: 10)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for faster experimentation (default: use all data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model",
    )
    add_baseline_args(parser)
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

    args = parser.parse_args()
    
    # Build data source configuration from CLI flags
    data_config = DataSourceConfig(
        use_vehicles=args.include_vehicle,
        use_people=args.include_people,
        use_weather=args.include_weather,
    )

    results = run_regression_pipeline(
        granularity=args.granularity,
        n_zones=args.n_zones,
        sample_size=args.sample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_model=not args.no_save,
        with_baseline=not args.no_baseline,
        data_config=data_config,
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for key, value in results["overall"].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print(f"\n  Ensemble vs Global improvement: {results['improvement_pct']:.1f}%")
