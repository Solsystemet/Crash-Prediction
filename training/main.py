"""Training entry point: train and evaluate LightGBM.

Run with: python -m training.main
"""

import sys
from pathlib import Path

from data_preparation.data_prepper import prepare_data
from data_preparation.tensor_config import SEVERITY_PREDICTION_CONFIG
from training.evaluation import (
    evaluate_lightgbm,
    print_classification_report_full,
    print_metrics,
)
from training.light_gbm import (
    LightGBMConfig,
    save_lightgbm,
    train_lightgbm,
)


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"


def _build_manual_class_weights(
    class_names: list[str] | None,
    requested_weights: dict[str, float] | None,
) -> dict[int, float] | None:
    """Map user-friendly class-name weights to encoded class-id weights."""
    if requested_weights is None:
        return None
    if class_names is None:
        raise RuntimeError("Cannot apply manual class weights without target class names")

    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    unknown_classes = sorted(set(requested_weights) - set(name_to_idx))
    if unknown_classes:
        raise ValueError(
            f"Unknown class names in manual weights: {unknown_classes}. "
            f"Available classes: {class_names}"
        )

    return {name_to_idx[name]: weight for name, weight in requested_weights.items()}


def main() -> None:
    """Train LightGBM and evaluate on validation and test sets."""
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

    class_names = None
    if result.encoder_registry.target_encoder is not None:
        encoder_classes = result.encoder_registry.target_encoder.classes_
        if encoder_classes is not None:
            class_names = list(encoder_classes)

    # Manual class weights by class name (set to None to use "balanced")
    # Example:
    # manual_class_weights_by_name = {"SEVERE": 8.0, "MINOR": 3.0, "NO_INJURY": 1.0}
    manual_class_weights_by_name: dict[str, float] | None = None
    manual_class_weights = _build_manual_class_weights(
        class_names=class_names,
        requested_weights=manual_class_weights_by_name,
    )

    # =========================================================================
    # Step 2: Train LightGBM
    # =========================================================================
    print("\n[2/5] Training LightGBM...")
    lgbm_config = LightGBMConfig(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=48,
        min_child_samples=200,
        class_weight=manual_class_weights if manual_class_weights is not None else "balanced",
        boosting_type="goss",
        top_rate=0.2,
        other_rate=0.1,
        use_smote=True,
        eval_metric=["multi_logloss"],
        early_stopping_rounds=50,
    )

    lgbm_model = train_lightgbm(
        train_dataset=result.train_dataset,
        val_dataset=result.val_dataset,
        config=lgbm_config,
        verbose=True,
    )

    # =========================================================================
    # Step 3: Evaluate on Validation Set
    # =========================================================================
    print("\n[3/4] Evaluating on validation set...")

    lgbm_val_metrics = evaluate_lightgbm(
        model=lgbm_model,
        dataset=result.val_dataset,
    )
    print_metrics(lgbm_val_metrics, title="LightGBM - Validation Set Results")

    # =========================================================================
    # Step 4: Final Evaluation on Test Set
    # =========================================================================
    print("\n[4/4] Final evaluation on test set...")

    # Always: LightGBM test metrics + report
    lgbm_test_metrics = evaluate_lightgbm(
        model=lgbm_model,
        dataset=result.test_dataset,
    )
    print_metrics(lgbm_test_metrics, title="LightGBM - Test Set Results")
    print_classification_report_full(lgbm_test_metrics, class_names=class_names)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_lightgbm(lgbm_model, MODELS_DIR / "lightgbm.joblib")
    print(f"\nModel saved to: {MODELS_DIR / 'lightgbm.joblib'}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
