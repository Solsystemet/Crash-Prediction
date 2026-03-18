"""Training entry point: train Gradient Boosted Trees model.

Run with: python -m training.main
"""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

from data_preparation.data_prepper import prepare_data
from data_preparation.tensor_config import SEVERITY_PREDICTION_CONFIG
from training.evaluation import (
    EvaluationMetrics,
    evaluate_gradient_boosted_trees,
    print_classification_report_full,
    print_metrics,
)
from training.gradient_boosted_trees import (
    GradientBoostingConfig,
    save_gradient_boosted_trees,
    train_gradient_boosted_trees,
)


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"


def main() -> None:
    """Train Gradient Boosted Trees model and evaluate on test set."""
    print("=" * 60)
    print("Crash Severity Prediction - Gradient Boosted Trees")
    print("=" * 60)

    # =========================================================================
    # Step 1: Prepare Data
    # =========================================================================
    print("\n[1/4] Loading data...")
    result = prepare_data(config=SEVERITY_PREDICTION_CONFIG, use_cache=True)

    print(f"  Train: {len(result.train_dataset):,} samples")
    print(f"  Val:   {len(result.val_dataset):,} samples")
    print(f"  Test:  {len(result.test_dataset):,} samples")
    print(f"  Features: {result.train_dataset.num_features}")
    print(f"  Classes: {result.encoder_registry.get_num_classes()}")
    sys.stdout.flush()

    # Create DataLoaders
    train_loader = DataLoader(
        result.train_dataset,
        batch_size=len(result.train_dataset),  # Load all data at once for XGBoost
        shuffle=True,
    )
    val_loader = DataLoader(
        result.val_dataset,
        batch_size=len(result.val_dataset),
        shuffle=False,
    )

    # =========================================================================
    # Step 2: Train Gradient Boosted Trees
    # =========================================================================
    print("\n[2/4] Training Gradient Boosted Trees...")
    gbt_config = GradientBoostingConfig(
        objective="multi:softmax",
        max_depth=6,
        learning_rate=0.1,
        n=100,
    )

    gbt_model, train_accuracy = train_gradient_boosted_trees(
        train_loader=train_loader,
        val_loader=val_loader,
        num_features=result.train_dataset.num_features,
        num_classes=result.encoder_registry.get_num_classes(),
        config=gbt_config,
        verbose=True,
        train_dataset=result.train_dataset,
    )

    # =========================================================================
    # Step 3: Evaluate on Validation Set
    # =========================================================================
    print("\n[3/4] Evaluating on validation set...")

    val_metrics = evaluate_gradient_boosted_trees(
        model=gbt_model,
        dataset=result.val_dataset,
    )

    print_metrics(val_metrics, title="Gradient Boosted Trees - Validation Set")

    # =========================================================================
    # Step 4: Final Evaluation on Test Set
    # =========================================================================
    print("\n[4/4] Final evaluation on test set...")

    # Get class names for detailed report
    class_names = None
    if result.encoder_registry.target_encoder is not None:
        class_names = list(result.encoder_registry.target_encoder.classes_)

    test_metrics = evaluate_gradient_boosted_trees(
        model=gbt_model,
        dataset=result.test_dataset,
    )
    print_metrics(test_metrics, title="Gradient Boosted Trees - Test Set Results")
    print_classification_report_full(test_metrics, class_names=class_names)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_gradient_boosted_trees(gbt_model, MODELS_DIR / "gradient_boosted_trees.json")
    print(f"\nModel saved to: {MODELS_DIR / 'gradient_boosted_trees.json'}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
