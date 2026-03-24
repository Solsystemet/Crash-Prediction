#!/usr/bin/env python
"""Train MLP to predict hourly crash counts with configurable time features.

Usage:
    uv run training/train_with_time_features.py

This script demonstrates using TimeFeatureConfig to control which
temporal patterns the model learns from:
- Cyclical features: hour_sin/cos, day_of_week_sin/cos, month_sin/cos
- Binary features: is_night, is_rush_hour, is_weekend

Experiment with different configurations to see which features help most.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_preparation.aggregate_hourly import (
    aggregate_crash_conditions,
    get_hourly_feature_columns,
    prepare_hourly_data,
)
from data_preparation.prepare_tensor_data import prepare_tensor_data
from data_preparation.tensor_config import TensorConfig
from data_preparation.time_features import (
    DEFAULT_TIME_CONFIG,
    FULL_TIME_CONFIG,
    TimeFeatureConfig,
)


class CrashCountMLP(nn.Module):
    """MLP for predicting hourly crash counts (regression)."""

    def __init__(
        self,
        num_features: int,
        hidden_sizes: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev_size = num_features

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        n_samples += features.size(0)

    return total_loss / n_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model, return (loss, MAE, RMSE)."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_se = 0.0
    n_samples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            total_mae += (outputs - labels).abs().sum().item()
            total_se += ((outputs - labels) ** 2).sum().item()
            n_samples += features.size(0)

    mse = total_loss / n_samples
    mae = total_mae / n_samples
    rmse = (total_se / n_samples) ** 0.5

    return mse, mae, rmse


def create_time_config(args: argparse.Namespace) -> TimeFeatureConfig:
    """Create TimeFeatureConfig from command line arguments."""
    if args.preset == "full":
        return FULL_TIME_CONFIG
    elif args.preset == "default":
        return DEFAULT_TIME_CONFIG
    elif args.preset == "binary_only":
        return TimeFeatureConfig(
            include_hour_cycle=False,
            include_day_of_week_cycle=False,
            include_month_cycle=False,
            include_is_night=True,
            include_is_rush_hour=True,
            include_is_weekend=True,
        )
    elif args.preset == "cyclical_only":
        return TimeFeatureConfig(
            include_hour_cycle=True,
            include_day_of_week_cycle=True,
            include_month_cycle=True,
            include_is_night=False,
            include_is_rush_hour=False,
            include_is_weekend=False,
        )
    else:
        # Custom from args
        return TimeFeatureConfig(
            include_hour_cycle=args.hour_cycle,
            include_day_of_week_cycle=args.day_cycle,
            include_month_cycle=args.month_cycle,
            include_is_night=args.is_night,
            include_is_rush_hour=args.is_rush_hour,
            include_is_weekend=args.is_weekend,
            night_start=args.night_start,
            night_end=args.night_end,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MLP with configurable time features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all time features (cyclical + binary)
  uv run training/train_with_time_features.py --preset full

  # Use only binary features (is_night, is_rush_hour, is_weekend)
  uv run training/train_with_time_features.py --preset binary_only

  # Custom: hour cycle + is_night with custom night hours
  uv run training/train_with_time_features.py --hour-cycle --is-night --night-start 23 --night-end 5
        """,
    )

    # Preset selection
    parser.add_argument(
        "--preset",
        choices=["full", "default", "binary_only", "cyclical_only", "custom"],
        default="full",
        help="Use a preset configuration (default: full)",
    )

    # Individual feature toggles (used when preset=custom)
    parser.add_argument("--hour-cycle", dest="hour_cycle", action="store_true", default=True)
    parser.add_argument("--no-hour-cycle", dest="hour_cycle", action="store_false")
    parser.add_argument("--day-cycle", dest="day_cycle", action="store_true", default=True)
    parser.add_argument("--no-day-cycle", dest="day_cycle", action="store_false")
    parser.add_argument("--month-cycle", dest="month_cycle", action="store_true", default=True)
    parser.add_argument("--no-month-cycle", dest="month_cycle", action="store_false")
    parser.add_argument("--is-night", dest="is_night", action="store_true", default=False)
    parser.add_argument("--is-rush-hour", dest="is_rush_hour", action="store_true", default=False)
    parser.add_argument("--is-weekend", dest="is_weekend", action="store_true", default=False)

    # Custom thresholds
    parser.add_argument("--night-start", type=int, default=22, help="Night start hour (default: 22)")
    parser.add_argument("--night-end", type=int, default=6, help="Night end hour (default: 6)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")

    # Additional feature flags
    parser.add_argument("--holidays", action="store_true", help="Add is_holiday binary feature")
    parser.add_argument("--conditions", action="store_true", help="Add crash condition percentages (pct_darkness, pct_wet_road, pct_bad_weather)")

    args = parser.parse_args()

    print("=" * 70)
    print("Hourly Crash Count Prediction - MLP with Configurable Time Features")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create time config
    time_config = create_time_config(args)
    time_features = time_config.get_feature_columns()

    print(f"\n[Config] Preset: {args.preset}")
    print(f"[Config] Time features ({len(time_features)}): {time_features}")
    if args.holidays:
        print("[Config] Holiday feature: enabled")
    if args.conditions:
        print("[Config] Condition features: enabled (pct_darkness, pct_wet_road, pct_bad_weather)")

    # Step 1: Prepare data with time features
    print("\n[1/4] Preparing hourly data...")
    hourly_df = prepare_hourly_data(
        time_config=time_config,
        include_holidays=args.holidays,
        include_conditions=args.conditions,
    )
    
    # Pre-compute condition lookup for prediction if conditions enabled
    condition_lookup_dict = None
    if args.conditions:
        from data_preparation.helpers.csv_loaders import get_traffic_crashes
        crash_df = get_traffic_crashes()
        condition_lookup = aggregate_crash_conditions(crash_df)
        # Convert to dict for checkpoint: {hour: {pct_darkness, pct_wet_road, pct_bad_weather}}
        condition_lookup_dict = condition_lookup.set_index("hour").to_dict(orient="index")
    
    print(f"  Total samples: {len(hourly_df):,}")
    print(f"  Date range: {hourly_df['hour_timestamp'].min()} to {hourly_df['hour_timestamp'].max()}")
    print(f"  Crash count - min: {hourly_df['crash_count'].min()}, max: {hourly_df['crash_count'].max()}, mean: {hourly_df['crash_count'].mean():.2f}")

    # Get feature columns
    categorical_cols, numerical_cols = get_hourly_feature_columns(
        time_config=time_config,
        include_holidays=args.holidays,
        include_conditions=args.conditions,
    )
    all_features = categorical_cols + numerical_cols

    print(f"\n  Feature breakdown:")
    print(f"    Time features: {len(time_features)}")
    extra_features = 0
    if args.holidays:
        extra_features += 1
    if args.conditions:
        extra_features += 3
    print(f"    Weather features: {len(numerical_cols) - len(time_features) - extra_features}")
    if extra_features > 0:
        print(f"    Additional features: {extra_features}")
    print(f"    Total features: {len(all_features)}")

    # Step 2: Create tensor config and convert
    print("\n[2/4] Converting to tensors with time-based split...")
    tensor_config = TensorConfig(
        target_column="crash_count",
        feature_columns=all_features,
        categorical_columns=categorical_cols,
        numerical_columns=numerical_cols,
        task_type="regression",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        split_by_time=True,
    )

    result = prepare_tensor_data(tensor_config, df=hourly_df)

    train_ds = result.train_dataset
    val_ds = result.val_dataset
    test_ds = result.test_dataset

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"  Input features: {train_ds.features.shape[1]}")

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Step 3: Initialize model
    print("\n[3/4] Training MLP...")
    num_features = train_ds.features.shape[1]
    model = CrashCountMLP(num_features=num_features).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"\n{'Epoch':<8} {'Train MSE':<12} {'Val MSE':<12} {'Val MAE':<12} {'Val RMSE':<12}")
    print("-" * 56)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_mse, val_mae, val_rmse = evaluate(model, val_loader, criterion, device)

        marker = " *" if val_mse < best_val_loss else ""
        print(f"{epoch + 1:<8} {train_loss:<12.4f} {val_mse:<12.4f} {val_mae:<12.4f} {val_rmse:<12.4f}{marker}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Step 4: Evaluate
    print("\n[4/4] Evaluating on test set...")
    test_mse, test_mae, test_rmse = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"  MSE:  {test_mse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")

    # Baseline
    train_mean = train_ds.labels.mean().item()
    baseline_mae = (test_ds.labels - train_mean).abs().mean().item()
    baseline_rmse = ((test_ds.labels - train_mean) ** 2).mean().item() ** 0.5

    print(f"\n  Baseline (predict mean={train_mean:.2f}):")
    print(f"    MAE:  {baseline_mae:.4f}")
    print(f"    RMSE: {baseline_rmse:.4f}")

    if baseline_mae > 0:
        print(f"\n  Improvement over baseline:")
        print(f"    MAE:  {(1 - test_mae / baseline_mae) * 100:+.1f}%")
        print(f"    RMSE: {(1 - test_rmse / baseline_rmse) * 100:+.1f}%")

    # Step 5: Save model
    from pathlib import Path
    
    models_dir = Path("trained_models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "crash_count_mlp.pt"
    
    # Save everything needed to reload and use the model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_features": num_features,
        "hidden_sizes": (128, 64, 32),
        "dropout": 0.3,
        "time_config": {
            "include_hour_cycle": time_config.include_hour_cycle,
            "include_day_of_week_cycle": time_config.include_day_of_week_cycle,
            "include_month_cycle": time_config.include_month_cycle,
            "include_is_night": time_config.include_is_night,
            "include_is_rush_hour": time_config.include_is_rush_hour,
            "include_is_weekend": time_config.include_is_weekend,
            "night_start": time_config.night_start,
            "night_end": time_config.night_end,
            "rush_morning_start": time_config.rush_morning_start,
            "rush_morning_end": time_config.rush_morning_end,
            "rush_evening_start": time_config.rush_evening_start,
            "rush_evening_end": time_config.rush_evening_end,
        },
        "include_holidays": args.holidays,
        "include_conditions": args.conditions,
        "condition_lookup": condition_lookup_dict,
        "feature_columns": all_features,
        "encoder_registry": result.encoder_registry.to_dict(),
        "test_metrics": {
            "mse": test_mse,
            "mae": test_mae,
            "rmse": test_rmse,
        },
        "baseline_metrics": {
            "mae": baseline_mae,
            "rmse": baseline_rmse,
        },
        "train_mean": train_mean,
    }
    
    torch.save(checkpoint, model_path)
    print(f"\n[5/5] Model saved to: {model_path}")

    print("\n" + "=" * 70)
    print(f"Time features used: {time_features}")
    print("=" * 70)


if __name__ == "__main__":
    main()
