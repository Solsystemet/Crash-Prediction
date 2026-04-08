"""Evaluate the trained aggregate injury prediction model.

This script loads an existing trained model and evaluates it against
the validation or test set, showing per-target accuracy metrics.

Supports:
- Overall evaluation
- Per-zone (cluster) evaluation when model was trained with clusters

Run with:
    python -m training.evaluate_aggregate_model
    python -m training.evaluate_aggregate_model --include-test
    python -m training.evaluate_aggregate_model --per-zone
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_preparation.aggregate_data import AggregateDataResult
from training.train_aggregate import load_aggregate_model
from training.device_utils import print_device_info
from training.losses import is_count_loss


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"
DEFAULT_MODEL_PATH = MODELS_DIR / "aggregate_injury_mlp.pt"
DEFAULT_DATA_PATH = MODELS_DIR / "aggregate_injury_mlp_data.pt"
LOGTRANSFORM_MODEL_PATH = MODELS_DIR / "aggregate_injury_mlp_logtransform.pt"
LOGTRANSFORM_DATA_PATH = MODELS_DIR / "aggregate_injury_mlp_logtransform_data.pt"
POISSON_MODEL_PATH = MODELS_DIR / "aggregate_injury_mlp_poisson.pt"
POISSON_DATA_PATH = MODELS_DIR / "aggregate_injury_mlp_poisson_data.pt"
NEGBIN_MODEL_PATH = MODELS_DIR / "aggregate_injury_mlp_negbin.pt"
NEGBIN_DATA_PATH = MODELS_DIR / "aggregate_injury_mlp_negbin_data.pt"


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_names: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-target and overall metrics.

    Args:
        predictions: Predicted values, shape (num_samples, num_targets).
        targets: Actual values, shape (num_samples, num_targets).
        target_names: Names of target columns.

    Returns:
        Dictionary with per-target and overall metrics.
    """
    results = {}

    # Per-target metrics
    for i, name in enumerate(target_names):
        pred_col = predictions[:, i]
        tgt_col = targets[:, i]

        mse = float(np.mean((pred_col - tgt_col) ** 2))
        mae = float(np.mean(np.abs(pred_col - tgt_col)))
        rmse = float(np.sqrt(mse))

        # R-squared
        ss_res = np.sum((tgt_col - pred_col) ** 2)
        ss_tot = np.sum((tgt_col - np.mean(tgt_col)) ** 2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        # Mean of actual values (for context)
        mean_actual = float(np.mean(tgt_col))

        results[name] = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mean_actual": mean_actual,
        }

    # Overall metrics
    overall_mse = float(np.mean((predictions - targets) ** 2))
    overall_mae = float(np.mean(np.abs(predictions - targets)))
    overall_rmse = float(np.sqrt(overall_mse))

    ss_res_total = np.sum((targets - predictions) ** 2)
    ss_tot_total = np.sum((targets - np.mean(targets)) ** 2)
    overall_r2 = float(1 - (ss_res_total / ss_tot_total)) if ss_tot_total > 0 else 0.0

    results["_overall"] = {
        "mae": overall_mae,
        "mse": overall_mse,
        "rmse": overall_rmse,
        "r2": overall_r2,
        "mean_actual": float(np.mean(targets)),
    }

    return results


def get_predictions_and_targets(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str,
    log_transform_targets: bool = False,
    loss_function: str = "mse",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get predictions and targets from a dataset.

    Args:
        model: Trained model.
        dataset: Dataset to evaluate on.
        device: Device to run on.
        log_transform_targets: Whether targets were log-transformed.
        loss_function: Loss function used during training.

    Returns:
        Tuple of (predictions, targets, features) as numpy arrays.
    """
    model.eval()
    device_obj = torch.device(device)
    model = model.to(device_obj)

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_predictions = []
    all_targets = []
    all_features = []

    with torch.no_grad():
        for features, targets in loader:
            all_features.append(features.numpy())
            features = features.to(device_obj)
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    features = np.vstack(all_features)

    # Transform predictions/targets back to original scale
    if is_count_loss(loss_function):
        # Model outputs log-rates, convert to rates
        predictions = np.exp(predictions)
    elif log_transform_targets:
        # Inverse of log(1 + x) is exp(x) - 1
        predictions = np.expm1(predictions)
        targets = np.expm1(targets)

    return predictions, targets, features


def evaluate_per_target(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    target_names: list[str],
    device: str,
    log_transform_targets: bool = False,
    loss_function: str = "mse",
) -> dict[str, dict[str, float]]:
    """Evaluate model and return per-target metrics.

    Args:
        model: Trained model.
        dataset: Dataset to evaluate on.
        target_names: Names of target columns.
        device: Device to run on.
        log_transform_targets: Whether targets were log-transformed (needs inverse).
        loss_function: Loss function used during training.

    Returns:
        Dictionary with per-target metrics.
    """
    predictions, targets, _ = get_predictions_and_targets(
        model, dataset, device, log_transform_targets, loss_function
    )
    return compute_metrics(predictions, targets, target_names)


def evaluate_per_zone(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    target_names: list[str],
    feature_columns: list[str],
    device: str,
    log_transform_targets: bool = False,
    loss_function: str = "mse",
    feature_means: torch.Tensor | None = None,
    feature_stds: torch.Tensor | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    """Evaluate model per zone/cluster.

    Args:
        model: Trained model.
        dataset: Dataset to evaluate on.
        target_names: Names of target columns.
        feature_columns: Names of feature columns.
        device: Device to run on.
        log_transform_targets: Whether targets were log-transformed.
        loss_function: Loss function used during training.
        feature_means: Feature means for denormalization.
        feature_stds: Feature stds for denormalization.

    Returns:
        Dictionary mapping zone_id to per-target metrics.
    """
    # Check if cluster feature exists
    if "cluster" not in feature_columns:
        raise ValueError(
            "Model was not trained with clusters. "
            "Use --use-clusters when training to enable per-zone evaluation."
        )

    cluster_idx = feature_columns.index("cluster")

    # Get all predictions and features
    predictions, targets, features = get_predictions_and_targets(
        model, dataset, device, log_transform_targets, loss_function
    )

    # Denormalize cluster feature to get actual cluster IDs
    if feature_means is not None and feature_stds is not None:
        cluster_values = (
            features[:, cluster_idx] * feature_stds[cluster_idx].item()
            + feature_means[cluster_idx].item()
        )
    else:
        cluster_values = features[:, cluster_idx]

    # Round to get integer cluster IDs
    cluster_ids = np.round(cluster_values).astype(int)
    unique_clusters = np.unique(cluster_ids)

    # Evaluate per zone
    zone_results = {}
    for zone_id in sorted(unique_clusters):
        mask = cluster_ids == zone_id
        zone_predictions = predictions[mask]
        zone_targets = targets[mask]

        if len(zone_predictions) > 0:
            zone_results[int(zone_id)] = compute_metrics(
                zone_predictions, zone_targets, target_names
            )
            zone_results[int(zone_id)]["_num_samples"] = int(mask.sum())

    return zone_results


def print_metrics_table(
    results: dict[str, dict[str, float]],
    target_names: list[str],
    set_name: str,
    num_samples: int,
) -> None:
    """Print metrics in a formatted table.

    Args:
        results: Dictionary with per-target metrics.
        target_names: Names of target columns.
        set_name: Name of the dataset (e.g., "Validation", "Test").
        num_samples: Number of samples in the dataset.
    """
    print(f"\n{set_name} Set Results (N={num_samples} samples):")
    print()
    print("Per-Target Metrics:")
    print("-" * 85)
    print(f"{'Target':<35} {'MAE':>8} {'MSE':>10} {'RMSE':>8} {'R²':>8} {'Mean':>10}")
    print("-" * 85)

    for name in target_names:
        metrics = results[name]
        short_name = name.replace("INJURIES_", "").replace("_", " ").title()
        print(
            f"{short_name:<35} "
            f"{metrics['mae']:>8.2f} "
            f"{metrics['mse']:>10.2f} "
            f"{metrics['rmse']:>8.2f} "
            f"{metrics['r2']:>8.3f} "
            f"{metrics['mean_actual']:>10.2f}"
        )

    print("-" * 85)
    overall = results["_overall"]
    print(
        f"{'Overall':<35} "
        f"{overall['mae']:>8.2f} "
        f"{overall['mse']:>10.2f} "
        f"{overall['rmse']:>8.2f} "
        f"{overall['r2']:>8.3f} "
        f"{overall['mean_actual']:>10.2f}"
    )
    print()


def print_zone_metrics_table(
    zone_results: dict[int, dict[str, Any]],
    set_name: str,
) -> None:
    """Print per-zone metrics in a formatted table.

    Args:
        zone_results: Dictionary mapping zone_id to metrics.
        set_name: Name of the dataset (e.g., "Validation", "Test").
    """
    print(f"\n{set_name} Set - Per-Zone Summary:")
    print()
    print("-" * 75)
    print(
        f"{'Zone':<8} {'Samples':>10} {'MAE':>10} {'RMSE':>10} {'R²':>10} {'Mean':>10}"
    )
    print("-" * 75)

    total_samples = 0
    for zone_id in sorted(zone_results.keys()):
        metrics = zone_results[zone_id]
        overall = metrics["_overall"]
        num_samples = metrics.get("_num_samples", 0)
        total_samples += num_samples
        print(
            f"{zone_id:<8} "
            f"{num_samples:>10} "
            f"{overall['mae']:>10.2f} "
            f"{overall['rmse']:>10.2f} "
            f"{overall['r2']:>10.3f} "
            f"{overall['mean_actual']:>10.2f}"
        )

    print("-" * 75)
    print(f"{'Total':<8} {total_samples:>10}")
    print()


def print_zone_detailed_metrics(
    zone_results: dict[int, dict[str, Any]],
    target_names: list[str],
    zone_id: int,
) -> None:
    """Print detailed metrics for a specific zone.

    Args:
        zone_results: Dictionary mapping zone_id to metrics.
        target_names: Names of target columns.
        zone_id: Zone ID to print details for.
    """
    if zone_id not in zone_results:
        print(f"Zone {zone_id} not found in results.")
        return

    metrics = zone_results[zone_id]
    num_samples = metrics.get("_num_samples", 0)

    print(f"\nZone {zone_id} Detailed Results (N={num_samples} samples):")
    print("-" * 85)
    print(f"{'Target':<35} {'MAE':>8} {'MSE':>10} {'RMSE':>8} {'R²':>8} {'Mean':>10}")
    print("-" * 85)

    for name in target_names:
        if name in metrics:
            m = metrics[name]
            short_name = name.replace("INJURIES_", "").replace("_", " ").title()
            print(
                f"{short_name:<35} "
                f"{m['mae']:>8.2f} "
                f"{m['mse']:>10.2f} "
                f"{m['rmse']:>8.2f} "
                f"{m['r2']:>8.3f} "
                f"{m['mean_actual']:>10.2f}"
            )

    print("-" * 85)
    overall = metrics["_overall"]
    print(
        f"{'Overall':<35} "
        f"{overall['mae']:>8.2f} "
        f"{overall['mse']:>10.2f} "
        f"{overall['rmse']:>8.2f} "
        f"{overall['r2']:>8.3f} "
        f"{overall['mean_actual']:>10.2f}"
    )
    print()


def main() -> None:
    """Main function to evaluate the model."""
    parser = argparse.ArgumentParser(
        description="Evaluate aggregate injury prediction model"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to data preprocessing info (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also evaluate on test set (default: validation only)",
    )
    parser.add_argument(
        "--log-transform",
        action="store_true",
        help="Use the log-transformed model (aggregate_injury_mlp_logtransform.pt)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "weighted_mse", "poisson", "weighted_poisson", "negbin"],
        default=None,
        help="Override loss function (auto-detected from saved data if not specified)",
    )
    parser.add_argument(
        "--per-zone",
        action="store_true",
        help="Evaluate per zone/cluster (requires model trained with --use-clusters)",
    )
    parser.add_argument(
        "--zone",
        type=int,
        default=None,
        help="Show detailed metrics for a specific zone (use with --per-zone)",
    )
    args = parser.parse_args()

    # Update paths based on model type
    if args.log_transform:
        if args.model == DEFAULT_MODEL_PATH:
            args.model = LOGTRANSFORM_MODEL_PATH
        if args.data == DEFAULT_DATA_PATH:
            args.data = LOGTRANSFORM_DATA_PATH

    # Check if files exist
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first with: python -m training.main_aggregate")
        return

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        print("Please train the model first with: python -m training.main_aggregate")
        return

    # Print header
    print("=" * 60)
    print("Aggregate Injury Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data:  {args.data}")
    print()

    # Detect device
    device = print_device_info()

    # Load model and data
    print("\nLoading model and data...")
    model = load_aggregate_model(args.model, device=device)
    data_result = AggregateDataResult.load(args.data)

    # Determine loss function (from args or saved data)
    loss_function = args.loss if args.loss else data_result.loss_function

    print(f"Features: {data_result.feature_columns}")
    print(f"Targets:  {data_result.target_columns}")
    print(f"Loss function: {loss_function.upper()}")
    if data_result.log_transform_targets:
        print("Log transform: enabled")
    if data_result.config.use_clusters:
        print(f"Clusters: {data_result.config.n_clusters}")

    if args.per_zone:
        # Per-zone evaluation
        try:
            val_zone_results = evaluate_per_zone(
                model=model,
                dataset=data_result.val_dataset,
                target_names=data_result.target_columns,
                feature_columns=data_result.feature_columns,
                device=device,
                log_transform_targets=data_result.log_transform_targets,
                loss_function=loss_function,
                feature_means=data_result.feature_means,
                feature_stds=data_result.feature_stds,
            )
            print_zone_metrics_table(val_zone_results, "Validation")

            if args.zone is not None:
                print_zone_detailed_metrics(
                    val_zone_results, data_result.target_columns, args.zone
                )

            if args.include_test:
                test_zone_results = evaluate_per_zone(
                    model=model,
                    dataset=data_result.test_dataset,
                    target_names=data_result.target_columns,
                    feature_columns=data_result.feature_columns,
                    device=device,
                    log_transform_targets=data_result.log_transform_targets,
                    loss_function=loss_function,
                    feature_means=data_result.feature_means,
                    feature_stds=data_result.feature_stds,
                )
                print_zone_metrics_table(test_zone_results, "Test")

                if args.zone is not None:
                    print_zone_detailed_metrics(
                        test_zone_results, data_result.target_columns, args.zone
                    )

        except ValueError as e:
            print(f"\nError: {e}")
            return
    else:
        # Standard evaluation
        val_results = evaluate_per_target(
            model=model,
            dataset=data_result.val_dataset,
            target_names=data_result.target_columns,
            device=device,
            log_transform_targets=data_result.log_transform_targets,
            loss_function=loss_function,
        )
        print_metrics_table(
            results=val_results,
            target_names=data_result.target_columns,
            set_name="Validation",
            num_samples=len(data_result.val_dataset),
        )

        if args.include_test:
            test_results = evaluate_per_target(
                model=model,
                dataset=data_result.test_dataset,
                target_names=data_result.target_columns,
                device=device,
                log_transform_targets=data_result.log_transform_targets,
                loss_function=loss_function,
            )
            print_metrics_table(
                results=test_results,
                target_names=data_result.target_columns,
                set_name="Test",
                num_samples=len(data_result.test_dataset),
            )


if __name__ == "__main__":
    main()
