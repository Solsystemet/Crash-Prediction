"""Diagnostic script to investigate model predictions."""

import numpy as np
from torch.utils.data import DataLoader

from data_preparation.data_prepper import prepare_data
from data_preparation.tensor_config import SEVERITY_PREDICTION_CONFIG
from training.evaluation import evaluate_neural_network, evaluate_random_forest
from training.train_neural_net import TrainingConfig, train_neural_network
from training.train_random_forest import RandomForestConfig, train_random_forest


def main() -> None:
    print("Loading data...")
    result = prepare_data(config=SEVERITY_PREDICTION_CONFIG, use_cache=True)

    # Create DataLoaders
    train_loader = DataLoader(result.train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(result.val_dataset, batch_size=32, shuffle=False)

    # Train NN (quick - few epochs)
    print("\nTraining Neural Network (5 epochs)...")
    nn_config = TrainingConfig(epochs=5, early_stopping_patience=10, use_class_weights=True)
    nn_model, _ = train_neural_network(
        train_loader=train_loader,
        val_loader=val_loader,
        num_features=result.train_dataset.num_features,
        num_classes=result.encoder_registry.get_num_classes(),
        config=nn_config,
        verbose=True,
        train_dataset=result.train_dataset,
    )

    # Train RF
    print("\nTraining Random Forest...")
    rf_config = RandomForestConfig(n_estimators=50, max_depth=10, class_weight="balanced")
    rf_model = train_random_forest(
        train_dataset=result.train_dataset,
        config=rf_config,
        verbose=True,
    )

    # Evaluate both
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Checking predictions")
    print("=" * 60)

    nn_metrics = evaluate_neural_network(nn_model, result.val_dataset, device=nn_config.device)
    rf_metrics = evaluate_random_forest(rf_model, result.val_dataset)

    print(f"\nNN predictions unique values: {np.unique(nn_metrics.predictions, return_counts=True)}")
    print(f"RF predictions unique values: {np.unique(rf_metrics.predictions, return_counts=True)}")
    print(f"True labels unique values:    {np.unique(nn_metrics.true_labels, return_counts=True)}")

    print(f"\nNN first 20 predictions: {nn_metrics.predictions[:20]}")
    print(f"RF first 20 predictions: {rf_metrics.predictions[:20]}")
    print(f"True first 20 labels:    {nn_metrics.true_labels[:20]}")

    # Check if predictions are identical
    are_identical = np.array_equal(nn_metrics.predictions, rf_metrics.predictions)
    print(f"\nAre NN and RF predictions identical? {are_identical}")

    if are_identical:
        print("\n>>> PROBLEM: Both models produce identical predictions!")
        print("    This suggests both are predicting the majority class.")

    # Show class distribution in training data
    print("\n" + "=" * 60)
    print("Training data class distribution:")
    print("=" * 60)
    train_dist = result.train_dataset.get_label_distribution()
    total = sum(train_dist.values())
    for label, count in sorted(train_dist.items()):
        pct = count / total * 100
        if result.encoder_registry.target_encoder is not None:
            name = result.encoder_registry.target_encoder.classes_[label]
            print(f"  {label} ({name}): {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
