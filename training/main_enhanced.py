"""Enhanced training entry point implementing research-based approach.

This script implements the 6-point research methodology:
1. Multi-Source Data Integration (Triple Merge)
2. Aggressive Feature Engineering
3. Class Imbalance Handling (SMOTE)
4. Spatial Clustering Features
5. Multi-Model Comparison (NN, RF, Extra Trees, XGBoost)
6. Enhanced Evaluation Metrics (AUC-ROC, per-class recall)

Run with: python -m training.main_enhanced
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from data_preparation.triple_merge import triple_merge
from data_preparation.feature_engineering import engineer_all_features
from data_preparation.resampling import apply_resampling, print_class_distribution
from data_preparation.prepare_tensor_data import prepare_tensor_data
from data_preparation.tensor_config import SEVERITY_PREDICTION_CONFIG, TensorConfig
from splice.k_means import LocationClusterer

from training.evaluation import (
    EvaluationMetrics,
    compare_multiple_models,
    evaluate_neural_network,
    evaluate_random_forest,
    evaluate_extra_trees,
    evaluate_xgboost,
    print_classification_report_full,
    print_metrics,
    print_per_class_recall,
    save_confusion_matrix_plot,
)
from training.train_neural_net import (
    TrainingConfig,
    save_model as save_nn_model,
    train_neural_network,
)
from training.train_random_forest import (
    RandomForestConfig,
    save_random_forest,
    train_random_forest,
)
from training.train_extra_trees import (
    ExtraTreesConfig,
    save_extra_trees,
    train_extra_trees,
    print_feature_importance as print_et_importance,
)
from training.train_xgboost import (
    XGBoostConfig,
    save_xgboost,
    train_xgboost,
    XGBOOST_AVAILABLE,
)


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "models" / "plots"


# Enhanced config for triple-merged data
_ENHANCED_CATEGORICAL = [
    "TRAFFIC_CONTROL_DEVICE",
    "DEVICE_CONDITION",
    "WEATHER_CONDITION",
    "LIGHTING_CONDITION",
    "ROADWAY_SURFACE_COND",
    "ROAD_DEFECT",
    "TIME_OF_DAY",
    "SEASON",
    "VEHICLE_AGE_CATEGORY",
    "LOCATION_CLUSTER",
]

_ENHANCED_NUMERICAL = [
    "POSTED_SPEED_LIMIT",
    "LANE_CNT",
    "CRASH_HOUR",
    "CRASH_DAY_OF_WEEK",
    "CRASH_MONTH",
    "LATITUDE",
    "LONGITUDE",
    # Temporal
    "IS_PEAK_HOUR",
    "IS_WEEKEND",
    "IS_NIGHT",
    # Vehicle
    "VEHICLE_COUNT",
    "VEHICLE_AGE",
    "OLD_VEHICLE_FLAG",
    # People
    "PERSON_COUNT",
    "AVG_AGE",
    "ANY_BAC_POSITIVE",
    "SEATBELT_USAGE_RATE",
    # Interactions
    "ADVERSE_CONDITIONS_COUNT",
    "NIGHT_POOR_LIGHTING",
    "WET_ROAD",
    "HIGH_SPEED_AREA",
    "MULTI_VEHICLE",
    # Weather
    "Air Temperature",
    "Humidity",
]


ENHANCED_CONFIG = TensorConfig(
    target_column="MOST_SEVERE_INJURY",
    feature_columns=_ENHANCED_CATEGORICAL + _ENHANCED_NUMERICAL,
    categorical_columns=_ENHANCED_CATEGORICAL,
    numerical_columns=_ENHANCED_NUMERICAL,
    task_type="classification",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
)


def main(
    use_triple_merge: bool = True,
    use_feature_engineering: bool = True,
    use_smote: bool = True,
    use_clustering: bool = True,
    n_clusters: int = 10,
    smote_strategy: str = "smote",
) -> None:
    """Train all models with research-based methodology.

    Args:
        use_triple_merge: Whether to use triple merge (crash+vehicle+people+weather).
        use_feature_engineering: Whether to generate engineered features.
        use_smote: Whether to apply SMOTE resampling.
        use_clustering: Whether to add location cluster features.
        n_clusters: Number of location clusters.
        smote_strategy: SMOTE variant to use.
    """
    print("=" * 70)
    print("Crash Severity Prediction - Enhanced Training Pipeline")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Triple Merge:        {use_triple_merge}")
    print(f"  Feature Engineering: {use_feature_engineering}")
    print(f"  SMOTE Resampling:    {use_smote} ({smote_strategy})")
    print(f"  Location Clustering: {use_clustering} (k={n_clusters})")
    print()
    sys.stdout.flush()

    # =========================================================================
    # Step 1: Load and Merge Data
    # =========================================================================
    print("[1/7] Loading and merging data...")

    if use_triple_merge:
        df = triple_merge(use_weather=True, verbose=True)
    else:
        from data_preparation.helpers.csv_loaders import get_traffic_crashes
        df = get_traffic_crashes()
        print(f"  Loaded {len(df):,} crashes (no merge)")

    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================
    print("\n[2/7] Engineering features...")

    cluster_labels = None
    if use_clustering:
        print("  Fitting location clusterer...")
        clusterer = LocationClusterer(n_clusters=n_clusters, random_state=42)
        clusterer.fit(df)
        cluster_labels = clusterer.transform(df)
        print(f"  Created {n_clusters} location clusters")

    if use_feature_engineering:
        df = engineer_all_features(
            df,
            include_interactions=True,
            include_clusters=use_clustering,
            cluster_labels=cluster_labels,
        )
        print(f"  Engineered features. Total columns: {len(df.columns)}")

    # Select config based on features available
    if use_triple_merge and use_feature_engineering:
        config = ENHANCED_CONFIG
        # Filter to columns that actually exist
        available_cat = [c for c in config.categorical_columns if c in df.columns]
        available_num = [c for c in config.numerical_columns if c in df.columns]
        config = TensorConfig(
            target_column=config.target_column,
            feature_columns=available_cat + available_num,
            categorical_columns=available_cat,
            numerical_columns=available_num,
            task_type="classification",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )
        print(f"  Using {len(available_cat)} categorical + {len(available_num)} numerical features")
    else:
        config = SEVERITY_PREDICTION_CONFIG

    # =========================================================================
    # Step 3: Prepare Tensor Datasets
    # =========================================================================
    print("\n[3/7] Preparing tensor datasets...")

    result = prepare_tensor_data(config, df=df)

    print(f"  Train: {len(result.train_dataset):,} samples")
    print(f"  Val:   {len(result.val_dataset):,} samples")
    print(f"  Test:  {len(result.test_dataset):,} samples")
    print(f"  Features: {result.train_dataset.num_features}")
    print(f"  Classes: {result.encoder_registry.get_num_classes()}")

    # Get class names
    class_names = None
    if result.encoder_registry.target_encoder is not None:
        class_names = list(result.encoder_registry.target_encoder.classes_)

    # Show class distribution
    print_class_distribution(
        result.train_dataset.labels.numpy(),
        title="Training Set Class Distribution",
        class_names=class_names,
    )

    # =========================================================================
    # Step 4: Apply SMOTE (Training Data Only)
    # =========================================================================
    X_train = result.train_dataset.features.numpy()
    y_train = result.train_dataset.labels.numpy()

    if use_smote:
        print("\n[4/7] Applying SMOTE resampling...")
        X_train_resampled, y_train_resampled = apply_resampling(
            X_train, y_train,
            strategy=smote_strategy,
            verbose=True,
        )
    else:
        print("\n[4/7] Skipping SMOTE (using original class distribution)...")
        X_train_resampled, y_train_resampled = X_train, y_train

    # Create resampled dataset for sklearn models
    from data_preparation.tensor_dataset import CrashTensorDataset
    import torch

    train_dataset_resampled = CrashTensorDataset(
        features=torch.from_numpy(X_train_resampled).float(),
        labels=torch.from_numpy(y_train_resampled).long(),
    )

    sys.stdout.flush()

    # =========================================================================
    # Step 5: Train All Models
    # =========================================================================
    print("\n[5/7] Training models...")

    models: dict[str, Any] = {}
    val_metrics: dict[str, EvaluationMetrics] = {}

    # --- Neural Network ---
    print("\n" + "-" * 50)
    print("Training Neural Network...")
    print("-" * 50)

    train_loader = DataLoader(train_dataset_resampled, batch_size=64, shuffle=True)
    val_loader = DataLoader(result.val_dataset, batch_size=64, shuffle=False)

    nn_config = TrainingConfig(
        epochs=30,
        learning_rate=1e-3,
        early_stopping_patience=5,
        use_class_weights=not use_smote,  # Don't double-correct
    )

    nn_model, _ = train_neural_network(
        train_loader=train_loader,
        val_loader=val_loader,
        num_features=result.train_dataset.num_features,
        num_classes=result.encoder_registry.get_num_classes(),
        config=nn_config,
        verbose=True,
        train_dataset=train_dataset_resampled,
    )
    models["Neural Network"] = nn_model
    val_metrics["Neural Network"] = evaluate_neural_network(
        nn_model, result.val_dataset, device=nn_config.device
    )

    # --- Random Forest ---
    print("\n" + "-" * 50)
    print("Training Random Forest...")
    print("-" * 50)

    rf_config = RandomForestConfig(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced" if not use_smote else None,
    )

    rf_model = train_random_forest(
        train_dataset=train_dataset_resampled,
        config=rf_config,
        verbose=True,
    )
    models["Random Forest"] = rf_model
    val_metrics["Random Forest"] = evaluate_random_forest(rf_model, result.val_dataset)

    # --- Extra Trees ---
    print("\n" + "-" * 50)
    print("Training Extra Trees (Research Top Performer)...")
    print("-" * 50)

    et_config = ExtraTreesConfig(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced" if not use_smote else None,
    )

    et_model = train_extra_trees(
        train_dataset=train_dataset_resampled,
        config=et_config,
        verbose=True,
    )
    models["Extra Trees"] = et_model
    val_metrics["Extra Trees"] = evaluate_extra_trees(et_model, result.val_dataset)

    # --- XGBoost ---
    if XGBOOST_AVAILABLE:
        print("\n" + "-" * 50)
        print("Training XGBoost...")
        print("-" * 50)

        xgb_config = XGBoostConfig(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            early_stopping_rounds=10,
        )

        xgb_model = train_xgboost(
            train_dataset=train_dataset_resampled,
            val_dataset=result.val_dataset,
            config=xgb_config,
            verbose=True,
        )
        models["XGBoost"] = xgb_model
        val_metrics["XGBoost"] = evaluate_xgboost(xgb_model, result.val_dataset)
    else:
        print("\n[XGBoost not available - skipping]")

    # =========================================================================
    # Step 6: Compare Models
    # =========================================================================
    print("\n[6/7] Comparing models on validation set...")

    winner = compare_multiple_models(val_metrics, primary_metric="f1")

    # Print per-class recall for winner
    print_per_class_recall(val_metrics[winner], class_names=class_names)

    # =========================================================================
    # Step 7: Final Evaluation on Test Set
    # =========================================================================
    print("\n[7/7] Final evaluation on test set...")

    # Evaluate winner on test set
    if winner == "Neural Network":
        test_metrics = evaluate_neural_network(
            models[winner], result.test_dataset, device=nn_config.device
        )
    elif winner == "XGBoost":
        test_metrics = evaluate_xgboost(models[winner], result.test_dataset)
    elif winner == "Extra Trees":
        test_metrics = evaluate_extra_trees(models[winner], result.test_dataset)
    else:
        test_metrics = evaluate_random_forest(models[winner], result.test_dataset)

    print_metrics(test_metrics, title=f"{winner} - Test Set Results")
    print_classification_report_full(test_metrics, class_names=class_names)
    print_per_class_recall(test_metrics, class_names=class_names)

    # Check minority class recall
    min_recall = min(test_metrics.per_class_recall.values()) if test_metrics.per_class_recall else 0
    if min_recall < 0.3:
        print(f"\n⚠️  WARNING: Minimum class recall is {min_recall:.2%} - model may be missing rare classes")

    # =========================================================================
    # Save Models and Plots
    # =========================================================================
    print("\nSaving models and plots...")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save winner
    if winner == "Neural Network":
        save_nn_model(models[winner], MODELS_DIR / "best_model_nn.pt")
    elif winner == "Random Forest":
        save_random_forest(models[winner], MODELS_DIR / "best_model_rf.joblib")
    elif winner == "Extra Trees":
        save_extra_trees(models[winner], MODELS_DIR / "best_model_et.joblib")
    elif winner == "XGBoost":
        save_xgboost(models[winner], MODELS_DIR / "best_model_xgb.json")

    print(f"  Saved winning model ({winner})")

    # Save confusion matrix
    save_confusion_matrix_plot(
        test_metrics,
        PLOTS_DIR / "confusion_matrix.png",
        class_names=class_names,
        title=f"{winner} - Confusion Matrix",
    )

    # Print feature importance for tree models
    if winner == "Extra Trees":
        feature_names = config.categorical_columns + config.numerical_columns
        print_et_importance(models[winner], feature_names=feature_names, top_n=15)

    print("\n" + "=" * 70)
    print("Enhanced Training Complete!")
    print("=" * 70)
    print(f"\nBest Model: {winner}")
    print(f"  F1 (macro): {test_metrics.f1:.4f}")
    if test_metrics.auc_roc:
        print(f"  AUC-ROC:    {test_metrics.auc_roc:.4f}")
    print(f"\nFiles saved to:")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Plots:  {PLOTS_DIR}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
