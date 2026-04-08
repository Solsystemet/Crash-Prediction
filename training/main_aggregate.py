"""Main entry point for training the aggregate hourly injury prediction model.

Run with:
    python -m training.main_aggregate

Options:
    --use-clusters: Use geographic clustering (default: city-wide)
    --n-clusters: Number of clusters if using clustering (default: 10)
    --epochs: Maximum training epochs (default: 100)
    --batch-size: Batch size for training (default: 64)
    --lr: Learning rate (default: 0.001)
    --loss: Loss function - "mse", "poisson", or "negbin" (default: mse)
    --log-transform: Apply log(1+x) transform to targets (only with mse loss)

Examples:
    # Train with MSE loss (default)
    python -m training.main_aggregate

    # Train with Poisson loss (recommended for count data)
    python -m training.main_aggregate --loss poisson

    # Train with Negative Binomial loss (handles overdispersion)
    python -m training.main_aggregate --loss negbin
"""

import argparse
from pathlib import Path

from data_preparation.aggregate_data import (
    AggregateConfig,
    CITYWIDE_HOURLY_CONFIG,
    prepare_aggregate_data,
    create_cluster_config,
)
from training.train_aggregate import (
    AggregateTrainingConfig,
    train_aggregate_model,
    evaluate_aggregate_model,
    save_aggregate_model,
)
from training.device_utils import print_device_info


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"


def main() -> None:
    """Main function to train and evaluate the aggregate model."""
    parser = argparse.ArgumentParser(
        description="Train aggregate hourly injury prediction model"
    )
    parser.add_argument(
        "--use-clusters",
        action="store_true",
        help="Use geographic clustering instead of city-wide aggregation",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of geographic clusters (default: 10)",
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
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--log-transform",
        action="store_true",
        help="Apply log(1+x) transform to targets (only with mse loss)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "weighted_mse", "poisson", "weighted_poisson", "negbin"],
        default="mse",
        help="Loss function: mse, weighted_mse, poisson, weighted_poisson, or negbin",
    )
    args = parser.parse_args()

    # Validate: log-transform and count losses are mutually exclusive
    if args.log_transform and args.loss in ("poisson", "weighted_poisson", "negbin"):
        parser.error(
            f"--log-transform cannot be used with --loss {args.loss}. "
            "Poisson and Negative Binomial losses expect raw counts."
        )

    # Create data configuration
    if args.use_clusters:
        print(f"Using geographic clustering with {args.n_clusters} clusters")
        data_config = create_cluster_config(
            n_clusters=args.n_clusters,
            log_transform_targets=args.log_transform,
        )
    else:
        print("Using city-wide aggregation")
        data_config = AggregateConfig(log_transform_targets=args.log_transform)

    if args.log_transform:
        print("Target transformation: ENABLED (log(1+x))")
    print(f"Loss function: {args.loss.upper()}")

    # Prepare data
    print("\n" + "=" * 60)
    print("STEP 1: Preparing Aggregated Data")
    print("=" * 60)
    data_result = prepare_aggregate_data(config=data_config, verbose=True)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(data_result.train_dataset)} samples")
    print(f"  Val:   {len(data_result.val_dataset)} samples")
    print(f"  Test:  {len(data_result.test_dataset)} samples")
    print(f"  Features: {data_result.num_features}")
    print(f"  Targets: {data_result.num_targets}")

    # Detect and print device info
    print("\n" + "=" * 60)
    print("STEP 2: Detecting Compute Device")
    print("=" * 60)
    device = print_device_info()

    # Create training configuration
    training_config = AggregateTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        device=device,
        loss_function=args.loss,
    )

    # Update data_result with the loss function for saving
    data_result.loss_function = args.loss

    # Train model
    print("\n" + "=" * 60)
    print("STEP 3: Training Neural Network")
    print("=" * 60)
    model, history = train_aggregate_model(
        train_dataset=data_result.train_dataset,
        val_dataset=data_result.val_dataset,
        num_features=data_result.num_features,
        num_targets=data_result.num_targets,
        target_names=data_result.target_columns,
        config=training_config,
        verbose=True,
        log_transform_targets=data_result.log_transform_targets,
    )

    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating on Validation Set")
    print("=" * 60)
    val_metrics = evaluate_aggregate_model(
        model=model,
        dataset=data_result.val_dataset,
        target_names=data_result.target_columns,
        device=training_config.device,
        log_transform_targets=data_result.log_transform_targets,
        loss_function=args.loss,
    )
    print(val_metrics)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating on Test Set")
    print("=" * 60)
    test_metrics = evaluate_aggregate_model(
        model=model,
        dataset=data_result.test_dataset,
        target_names=data_result.target_columns,
        device=training_config.device,
        log_transform_targets=data_result.log_transform_targets,
        loss_function=args.loss,
    )
    print(test_metrics)

    # Save model
    print("\n" + "=" * 60)
    print("STEP 6: Saving Model")
    print("=" * 60)

    # Determine model name based on options
    name_parts = ["aggregate_injury_mlp"]
    if args.use_clusters:
        name_parts.append("clusters")
    if args.loss != "mse":
        name_parts.append(args.loss)
    if args.log_transform:
        name_parts.append("logtransform")
    model_name = "_".join(name_parts) + ".pt"

    model_path = MODELS_DIR / model_name
    save_aggregate_model(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save data preprocessing info
    data_path = MODELS_DIR / model_name.replace(".pt", "_data.pt")
    data_result.save(data_path)
    print(f"Data preprocessing saved to: {data_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best epoch: {history.best_epoch + 1}")
    print(f"Test MSE: {test_metrics.mse:.4f}")
    print(f"Test MAE: {test_metrics.mae:.4f}")
    print(f"Test R²:  {test_metrics.r2:.4f}")


if __name__ == "__main__":
    main()
