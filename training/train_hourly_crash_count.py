#!/usr/bin/env python
"""Train MLP to predict hourly crash counts.

Usage:
    uv run training/train_hourly_crash_count.py

This script:
1. Loads and aggregates crash + weather data to hourly level
2. Adds lag features and cyclical time encodings
3. Splits data temporally (train < val < test in time)
4. Trains an MLP regressor to predict crash counts per hour
5. Evaluates on held-out test set
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_preparation.aggregate_hourly import prepare_hourly_data
from data_preparation.prepare_tensor_data import prepare_tensor_data
from data_preparation.tensor_config import HOURLY_CRASH_COUNT_CONFIG


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

        # Output layer: single value for regression
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


def main() -> None:
    print("=" * 60)
    print("Hourly Crash Count Prediction - MLP Regression")
    print("=" * 60)

    # Config
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    EARLY_STOPPING_PATIENCE = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Step 1: Prepare data
    print("\n[1/4] Preparing hourly data...")
    hourly_df = prepare_hourly_data()
    print(f"  Total hourly samples: {len(hourly_df):,}")
    print(f"  Date range: {hourly_df['hour_timestamp'].min()} to {hourly_df['hour_timestamp'].max()}")
    print(f"  Crash count range: {hourly_df['crash_count'].min()} - {hourly_df['crash_count'].max()}")
    print(f"  Mean crashes/hour: {hourly_df['crash_count'].mean():.2f}")

    # Step 2: Convert to tensors
    print("\n[2/4] Converting to tensors with time-based split...")
    result = prepare_tensor_data(HOURLY_CRASH_COUNT_CONFIG, df=hourly_df)

    train_ds = result.train_dataset
    val_ds = result.val_dataset
    test_ds = result.test_dataset

    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples: {len(val_ds):,}")
    print(f"  Test samples: {len(test_ds):,}")
    print(f"  Features: {train_ds.features.shape[1]}")

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Step 3: Initialize model
    print("\n[3/4] Training MLP...")
    num_features = train_ds.features.shape[1]
    model = CrashCountMLP(num_features=num_features).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"\n{'Epoch':<8} {'Train MSE':<12} {'Val MSE':<12} {'Val MAE':<12} {'Val RMSE':<12}")
    print("-" * 56)

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_mse, val_mae, val_rmse = evaluate(model, val_loader, criterion, device)

        print(f"{epoch + 1:<8} {train_loss:<12.4f} {val_mse:<12.4f} {val_mae:<12.4f} {val_rmse:<12.4f}")

        # Early stopping check
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Step 4: Evaluate on test set
    print("\n[4/4] Evaluating on test set...")
    test_mse, test_mae, test_rmse = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  MSE:  {test_mse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")

    # Baseline comparison (predicting mean)
    train_mean = train_ds.labels.mean().item()
    baseline_mae = (test_ds.labels - train_mean).abs().mean().item()
    baseline_rmse = ((test_ds.labels - train_mean) ** 2).mean().item() ** 0.5

    print(f"\n  Baseline (predict mean={train_mean:.2f}):")
    print(f"    MAE:  {baseline_mae:.4f}")
    print(f"    RMSE: {baseline_rmse:.4f}")
    print(f"\n  Improvement over baseline:")
    print(f"    MAE:  {(1 - test_mae / baseline_mae) * 100:.1f}%")
    print(f"    RMSE: {(1 - test_rmse / baseline_rmse) * 100:.1f}%")


if __name__ == "__main__":
    main()
