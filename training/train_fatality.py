"""Training entry point for fatality prediction with people data.

Run with: python -m training.train_fatality
"""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

from data_preparation.data_prepper import prepare_data
from data_preparation.merge_crash_with_people import get_crash_with_people_features
from data_preparation.tensor_config import FATALITY_PREDICTION_CONFIG
from training.evaluation import (
    compare_models,
    evaluate_neural_network,
    evaluate_random_forest,
    print_classification_report_full,
    print_metrics,
)
from training.train_neural_net import (
    TrainingConfig,
    create_weighted_sampler,
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
    """Train fatality prediction models using crash + people data."""
    print("=" * 60)
    print("Fatality Prediction - Model Training")
    print("=" * 60)

    # =========================================================================
    # Step 1: Prepare Merged Data
    # =========================================================================
    print("\n[1/5] Loading and merging crash + people data...")
    
    # Get merged crash + people data
    merged_df = get_crash_with_people_features()
    
    # Prepare tensors using fatality config
    result = prepare_data(
        config=FATALITY_PREDICTION_CONFIG,
        df=merged_df,
        use_cache=False,  # Don't cache merged data (it's dynamically generated)
        cache_name="fatality_pipeline",
    )

    print(f"\n  Train: {len(result.train_dataset):,} samples")
    print(f"  Val:   {len(result.val_dataset):,} samples")
    print(f"  Test:  {len(result.test_dataset):,} samples")
    print(f"  Features: {result.train_dataset.num_features}")
    print(f"  Classes: {result.encoder_registry.get_num_classes()}")
    
    # Show class distribution (important for imbalanced data)
    train_labels = result.train_dataset.labels.numpy()
    fatal_count = (train_labels == 1).sum()
    non_fatal_count = (train_labels == 0).sum()
    print(f"\n  Class distribution (train):")
    print(f"    Non-fatal (0): {non_fatal_count:,} ({non_fatal_count/len(train_labels)*100:.2f}%)")
    print(f"    Fatal (1):     {fatal_count:,} ({fatal_count/len(train_labels)*100:.2f}%)")
    sys.stdout.flush()

    # Create DataLoaders with weighted sampling for class imbalance
    train_sampler = create_weighted_sampler(result.train_dataset)

    train_loader = DataLoader(
        result.train_dataset,
        batch_size=256,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        result.val_dataset,
        batch_size=256,
        shuffle=False,
    )

    # =========================================================================
    # Step 2: Train Neural Network
    # =========================================================================
    print("\n[2/5] Training Neural Network...")
    
    # Config optimized for imbalanced binary classification
    nn_config = TrainingConfig(
        epochs=100,
        learning_rate=1e-3,
        early_stopping_patience=20,
        use_class_weights=True,  # Critical for rare fatal class
        loss_type="focal",       # Better for class imbalance
        use_lr_scheduler=True,
        gradient_clip_norm=1.0,
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
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",  # Handle imbalance in RF too
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

    class_names = ["Non-Fatal", "Fatal"]

    if winner == "neural_network":
        test_metrics = evaluate_neural_network(
            model=nn_model,
            dataset=result.test_dataset,
            device=nn_config.device,
        )
        print_metrics(test_metrics, title="Neural Network - Test Set Results")
        print_classification_report_full(test_metrics, class_names=class_names)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_nn_model(nn_model, MODELS_DIR / "fatality_neural_network.pt")
        print(f"\nModel saved to: {MODELS_DIR / 'fatality_neural_network.pt'}")

    else:
        test_metrics = evaluate_random_forest(
            model=rf_model,
            dataset=result.test_dataset,
        )
        print_metrics(test_metrics, title="Random Forest - Test Set Results")
        print_classification_report_full(test_metrics, class_names=class_names)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_random_forest(rf_model, MODELS_DIR / "fatality_random_forest.joblib")
        print(f"\nModel saved to: {MODELS_DIR / 'fatality_random_forest.joblib'}")

    print("\n" + "=" * 60)
    print("Fatality prediction training complete!")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
