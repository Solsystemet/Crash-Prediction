"""Training entry point: train and compare Neural Network vs Random Forest.

Run with: python -m training.main
"""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

from data_preparation.data_prepper import prepare_data
from data_preparation.tensor_config import SEVERITY_PREDICTION_CONFIG
from training.evaluation import (
    EvaluationMetrics,
    compare_models,
    evaluate_neural_network,
    evaluate_random_forest,
    print_classification_report_full,
    print_metrics,
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

    # Create DataLoaders for neural network
    train_loader = DataLoader(
        result.train_dataset,
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        result.val_dataset,
        batch_size=32,
        shuffle=False,
    )

    # =========================================================================
    # Step 2: Train Neural Network
    # =========================================================================
    print("\n[2/5] Training Neural Network...")
    nn_config = TrainingConfig(
        epochs=30,
        learning_rate=1e-3,
        early_stopping_patience=5,
    )

    nn_model, nn_history = train_neural_network(
        train_loader=train_loader,
        val_loader=val_loader,
        num_features=result.train_dataset.num_features,
        num_classes=result.encoder_registry.get_num_classes(),
        config=nn_config,
        verbose=True,
        train_dataset=result.train_dataset,
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

    nn_val_metrics = evaluate_neural_network(
        model=nn_model,
        dataset=result.val_dataset,
        device=nn_config.device,
    )

    rf_val_metrics = evaluate_random_forest(
        model=rf_model,
        dataset=result.val_dataset,
    )

    # Compare and pick winner
    winner = compare_models(nn_val_metrics, rf_val_metrics)

    # =========================================================================
    # Step 5: Final Evaluation on Test Set (Winner Only)
    # =========================================================================
    print("\n[5/5] Final evaluation on test set...")

    # Get class names for detailed report
    class_names = None
    if result.encoder_registry.target_encoder is not None:
        class_names = list(result.encoder_registry.target_encoder.classes_)

    if winner == "neural_network":
        test_metrics = evaluate_neural_network(
            model=nn_model,
            dataset=result.test_dataset,
            device=nn_config.device,
        )
        print_metrics(test_metrics, title="Neural Network - Test Set Results")
        print_classification_report_full(test_metrics, class_names=class_names)

        # Save winning model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_nn_model(nn_model, MODELS_DIR / "neural_network.pt")
        print(f"\nModel saved to: {MODELS_DIR / 'neural_network.pt'}")

    else:
        test_metrics = evaluate_random_forest(
            model=rf_model,
            dataset=result.test_dataset,
        )
        print_metrics(test_metrics, title="Random Forest - Test Set Results")
        print_classification_report_full(test_metrics, class_names=class_names)

        # Save winning model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_random_forest(rf_model, MODELS_DIR / "random_forest.joblib")
        print(f"\nModel saved to: {MODELS_DIR / 'random_forest.joblib'}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
