"""Training entry point: train and compare LightGBM vs Random Forest.

Run with: python -m training.main
"""

import sys
from pathlib import Path

from data_preparation.data_prepper import prepare_data
from data_preparation.tensor_config import SEVERITY_PREDICTION_CONFIG
from training.evaluation import (
    compare_models,
    evaluate_lightgbm,
    evaluate_random_forest,
    print_classification_report_full,
    print_metrics,
)
from training.light_gbm import (
    LightGBMConfig,
    save_lightgbm,
    train_lightgbm,
)
from training.train_random_forest import (
    RandomForestConfig,
    train_random_forest,
)


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"


def main() -> None:
    """Train both models, compare on validation, evaluate winner on test."""
    print("=" * 60)
    print("Crash Severity Prediction - Model Training")
    print("=" * 60)

    # =========================================================================
    # Step 1: Prepare Data
    # =========================================================================
    print("\n[1/5] Loading data...")
    result = prepare_data(config=SEVERITY_PREDICTION_CONFIG, use_cache=True)

    print(f"  Train: {len(result.train_dataset):,} samples")
    print(f"  Val:   {len(result.val_dataset):,} samples")
    print(f"  Test:  {len(result.test_dataset):,} samples")
    print(f"  Features: {result.train_dataset.num_features}")
    print(f"  Classes: {result.encoder_registry.get_num_classes()}")
    sys.stdout.flush()

    # =========================================================================
    # Step 2: Train LightGBM
    # =========================================================================
    print("\n[2/5] Training LightGBM...")
    lgbm_config = LightGBMConfig(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=48,
        min_child_samples=200,
        class_weight="balanced",
        boosting_type="goss",
        top_rate=0.2,
        other_rate=0.1,
        use_smote=False,
        eval_metric="multi_logloss",
        early_stopping_rounds=50,
    )

    lgbm_model = train_lightgbm(
        train_dataset=result.train_dataset,
        val_dataset=result.val_dataset,
        config=lgbm_config,
        verbose=True,
    )

    # =========================================================================
    # Step 3: Train Random Forest
    # =========================================================================
    print("\n[3/5] Training Random Forest...")
    rf_config = RandomForestConfig(
        n_estimators=100,
        max_depth=15,
    )

    rf_model = train_random_forest(
        train_dataset=result.train_dataset,
        config=rf_config,
        verbose=True,
    )

    # =========================================================================
    # Step 4: Evaluate Both on Validation Set
    # =========================================================================
    print("\n[4/5] Evaluating on validation set...")

    lgbm_val_metrics = evaluate_lightgbm(
        model=lgbm_model,
        dataset=result.val_dataset,
    )

    rf_val_metrics = evaluate_random_forest(
        model=rf_model,
        dataset=result.val_dataset,
    )

    # Compare and pick winner
    winner = compare_models(lgbm_val_metrics, rf_val_metrics)

    # =========================================================================
    # Step 5: Final Evaluation on Test Set
    #   - Always print LightGBM test report (requested)
    #   - Save the validation winner
    # =========================================================================
    print("\n[5/5] Final evaluation on test set...")

    # Get class names for detailed report
    class_names = None
    if result.encoder_registry.target_encoder is not None:
        encoder_classes = result.encoder_registry.target_encoder.classes_
        if encoder_classes is not None:
            class_names = list(encoder_classes)

    # Always: LightGBM test metrics + report
    lgbm_test_metrics = evaluate_lightgbm(
        model=lgbm_model,
        dataset=result.test_dataset,
    )
    print_metrics(lgbm_test_metrics, title="LightGBM - Test Set Results")
    print_classification_report_full(lgbm_test_metrics, class_names=class_names)

    if winner == "lightgbm":
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_lightgbm(lgbm_model, MODELS_DIR / "lightgbm.joblib")
        print(f"\nModel saved to: {MODELS_DIR / 'lightgbm.joblib'}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
