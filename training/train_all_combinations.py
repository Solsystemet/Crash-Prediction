"""Train all classification models across all dataset combinations.

This script trains 4 classification models with all 8 dataset combinations
(2^3 = 8 from vehicle/people/weather flags) and collects metrics for comparison.

Models:
- simple_rf: Simple Random Forest (3-class)
- simplified: Simplified 3-class hierarchy
- simplified_zones: Zone-based simplified (25 zones)
- hierarchical: Hierarchical 5-class (tree)

Dataset combinations (8):
1. crash_only: No additional data
2. vehicle: + vehicle data
3. people: + people data
4. weather: + weather data
5. vehicle_people: + vehicle + people
6. vehicle_weather: + vehicle + weather
7. people_weather: + people + weather
8. full: + vehicle + people + weather

Usage:
    python training/train_all_combinations.py
    python training/train_all_combinations.py --sample 10000  # Quick test
    python training/train_all_combinations.py --models simple_rf simplified  # Subset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.triple_merge import DataSourceConfig

# Import training functions
from training.main_simple_rf import main as train_simple_rf
from training.main_simplified import run_simplified_pipeline
from training.main_simplified_zones import run_zoned_pipeline
from training.main_hierarchical import run_hierarchical_pipeline
from training.tune_boosting import main as tune_boosting_main
from training.tabnet.train import train_tabnet
from utils.logging_config import setup_logging

logger = setup_logging(__name__)


# ============================================================================
# Dataset Combinations
# ============================================================================

@dataclass
class DatasetCombo:
    """A dataset combination configuration."""
    name: str
    use_vehicles: bool
    use_people: bool
    use_weather: bool
    
    def to_config(self) -> DataSourceConfig:
        """Convert to DataSourceConfig."""
        return DataSourceConfig(
            use_vehicles=self.use_vehicles,
            use_people=self.use_people,
            use_weather=self.use_weather,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for results."""
        return {
            "dataset": self.name,
            "use_vehicles": self.use_vehicles,
            "use_people": self.use_people,
            "use_weather": self.use_weather,
        }


# All 8 combinations
DATASET_COMBINATIONS = [
    DatasetCombo("crash_only", False, False, False),
    DatasetCombo("vehicle", True, False, False),
    DatasetCombo("people", False, True, False),
    DatasetCombo("weather", False, False, True),
    DatasetCombo("vehicle_people", True, True, False),
    DatasetCombo("vehicle_weather", True, False, True),
    DatasetCombo("people_weather", False, True, True),
    DatasetCombo("full", True, True, True),
]


# ============================================================================
# Model Definitions
# ============================================================================

@dataclass
class ModelSpec:
    """Specification for a model to train."""
    name: str
    train_fn: Callable
    extract_metrics: Callable[[Any], dict]  # Function to extract metrics from result


def extract_dict_metrics(result: dict) -> dict:
    """Extract metrics from a dict-returning function (simplified, hierarchical)."""
    # Common keys we want - expanded to include AUC, confusion matrix, and more
    keys_to_extract = [
        # Core metrics
        "accuracy", "f1_macro", "f1_micro", "f1_weighted",
        "precision_macro", "precision_weighted", 
        "recall_macro", "recall_weighted",
        # Hierarchical level metrics
        "l1_accuracy", "l1_f1", "l1_recall", "l1_auc",
        "l2_accuracy", "l2_f1", "l2_recall", "l2_auc",
        # AUC metrics
        "auc_injury", "auc_severe", "auc_macro", "auc_weighted",
        # Per-class recall (simplified 3-class)
        "recall_NO_INJURY", "recall_MINOR", "recall_SEVERE",
        # Per-class recall (hierarchical 5-class)
        "recall_FATAL", "recall_INCAPACITATING",
        "recall_NONINCAPACITATING", "recall_REPORTED",
        # mc_ prefixed metrics from hierarchical evaluation
        "mc_accuracy", "mc_f1_macro", "mc_f1_micro", "mc_f1_weighted",
    ]
    metrics = {}
    for key in keys_to_extract:
        if key in result:
            metrics[key] = result[key]
    
    # Handle confusion matrix - convert to JSON string for CSV compatibility
    if "confusion_matrix" in result:
        import json
        cm = result["confusion_matrix"]
        if hasattr(cm, "tolist"):
            cm = cm.tolist()
        metrics["confusion_matrix_json"] = json.dumps(cm)
    
    return metrics


def extract_zone_metrics(results: list) -> dict:
    """Extract aggregated metrics from zone results list."""
    # Filter out skipped zones
    valid_results = [r for r in results if not r.skipped]
    
    if not valid_results:
        return {"error": "All zones skipped"}
    
    # Compute weighted averages based on n_test
    total_test = sum(r.n_test for r in valid_results)
    
    metrics = {
        "n_zones_trained": len(valid_results),
        "n_zones_skipped": len(results) - len(valid_results),
    }
    
    # Weighted average of key metrics - expanded set
    metric_keys = [
        "accuracy", "f1_macro", "f1_micro", "f1_weighted",
        "precision_macro", "recall_macro",
        "l1_accuracy", "l1_f1", "l1_auc",
        "l2_accuracy", "l2_f1", "l2_auc",
        "recall_no_injury", "recall_minor", "recall_severe",
        "auc_injury", "auc_severe",
    ]
    
    for key in metric_keys:
        weighted_sum = sum(
            getattr(r, key, 0) * r.n_test for r in valid_results
        )
        metrics[key] = weighted_sum / total_test if total_test > 0 else 0.0
    
    return metrics


def extract_tuned_metrics(results: dict) -> dict:
    """Extract metrics from tune_boosting results."""
    # Results is dict like {"lgbm": {"cv_f1": 0.5, "test_f1": 0.48}}
    if not results:
        return {"error": "No results"}
    # Get first (only) model's metrics
    model_name, scores = next(iter(results.items()))
    return {
        "accuracy": scores.get("test_accuracy", 0),
        "f1_macro": scores.get("test_f1", 0),
        "cv_f1": scores.get("cv_f1", 0),
    }


def extract_tabnet_metrics(result: tuple) -> dict:
    """Extract metrics from tabnet train result (model, metrics)."""
    _, metrics = result
    return extract_dict_metrics(metrics)


def create_model_specs() -> list[ModelSpec]:
    """Create model specifications."""
    return [
        ModelSpec(
            name="simple_rf",
            train_fn=train_simple_rf,
            extract_metrics=extract_dict_metrics,
        ),
        ModelSpec(
            name="simplified",
            train_fn=run_simplified_pipeline,
            extract_metrics=extract_dict_metrics,
        ),
        ModelSpec(
            name="simplified_zones",
            train_fn=run_zoned_pipeline,
            extract_metrics=extract_zone_metrics,
        ),
        ModelSpec(
            name="hierarchical",
            train_fn=run_hierarchical_pipeline,
            extract_metrics=extract_dict_metrics,
        ),
        ModelSpec(
            name="tuned_lgbm",
            train_fn=tune_boosting_main,
            extract_metrics=extract_tuned_metrics,
        ),
        ModelSpec(
            name="tuned_xgb",
            train_fn=tune_boosting_main,
            extract_metrics=extract_tuned_metrics,
        ),
        ModelSpec(
            name="tuned_catboost",
            train_fn=tune_boosting_main,
            extract_metrics=extract_tuned_metrics,
        ),
        ModelSpec(
            name="tuned_rf",
            train_fn=tune_boosting_main,
            extract_metrics=extract_tuned_metrics,
        ),
        ModelSpec(
            name="tabnet",
            train_fn=train_tabnet,
            extract_metrics=extract_tabnet_metrics,
        ),
    ]


# ============================================================================
# Training Runner
# ============================================================================

def run_single_combination(
    model: ModelSpec,
    dataset: DatasetCombo,
    sample_size: int,
) -> dict[str, Any]:
    """Run a single model + dataset combination.
    
    Args:
        model: Model specification.
        dataset: Dataset combination.
        sample_size: Sample size for training.
        
    Returns:
        Dictionary with results and metadata.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING: {model.name} + {dataset.name}")
    logger.info(f"{'='*70}")
    
    config = dataset.to_config()
    start_time = time.time()
    
    try:
        # Run training with appropriate arguments
        if model.name == "simple_rf":
            result = model.train_fn(
                sample_size=sample_size,
                config=config,
            )
        elif model.name == "simplified":
            result = model.train_fn(
                sample_size=sample_size,
                data_config=config,
                with_baseline=False,  # Skip baseline for speed
            )
        elif model.name == "simplified_zones":
            result = model.train_fn(
                sample_size=sample_size,
                data_config=config,
                with_baseline=False,
            )
        elif model.name == "hierarchical":
            result = model.train_fn(
                model_type="tree",
                sample_size=sample_size,
                data_config=config,
                with_baseline=False,
            )
        elif model.name.startswith("tuned_"):
            # Extract model type from name (e.g., "tuned_lgbm" -> "lgbm")
            boosting_type = model.name.replace("tuned_", "")
            result = model.train_fn(
                model_type=boosting_type,
                n_trials=20,  # Reduced trials for speed
                sample_size=sample_size,
                config=config,
            )
        elif model.name == "tabnet":
            result = model.train_fn(
                sample_size=sample_size,
                data_config=config,
                save_model=True,  # Save trained models
            )
        else:
            raise ValueError(f"Unknown model: {model.name}")
        
        elapsed = time.time() - start_time
        
        # Extract metrics
        metrics = model.extract_metrics(result)
        
        return {
            "model": model.name,
            **dataset.to_dict(),
            "sample_size": sample_size,
            "duration_seconds": round(elapsed, 2),
            "status": "success",
            **metrics,
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"ERROR training {model.name} + {dataset.name}: {e}")
        return {
            "model": model.name,
            **dataset.to_dict(),
            "sample_size": sample_size,
            "duration_seconds": round(elapsed, 2),
            "status": "error",
            "error": str(e),
        }


def run_all_combinations(
    sample_size: int = 50000,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run all model × dataset combinations.
    
    Args:
        sample_size: Sample size for each training run.
        models: Optional subset of model names to run.
        datasets: Optional subset of dataset names to run.
        
    Returns:
        List of result dictionaries.
    """
    all_models = create_model_specs()
    all_datasets = DATASET_COMBINATIONS
    
    # Filter models if specified
    if models:
        all_models = [m for m in all_models if m.name in models]
        if not all_models:
            raise ValueError(f"No matching models found for: {models}")
    
    # Filter datasets if specified
    if datasets:
        all_datasets = [d for d in all_datasets if d.name in datasets]
        if not all_datasets:
            raise ValueError(f"No matching datasets found for: {datasets}")
    
    total_runs = len(all_models) * len(all_datasets)
    logger.info(f"\n{'#'*70}")
    logger.info(f"TRAINING ALL COMBINATIONS")
    logger.info(f"  Models: {[m.name for m in all_models]}")
    logger.info(f"  Datasets: {[d.name for d in all_datasets]}")
    logger.info(f"  Total runs: {total_runs}")
    logger.info(f"  Sample size: {sample_size:,}")
    logger.info(f"{'#'*70}\n")
    
    results = []
    run_num = 0
    
    for model in all_models:
        for dataset in all_datasets:
            run_num += 1
            logger.info(f"\n[Run {run_num}/{total_runs}]")
            
            result = run_single_combination(model, dataset, sample_size)
            results.append(result)
            
            # Log progress
            if result["status"] == "success":
                acc = result.get("accuracy", 0)
                f1 = result.get("f1_macro", 0)
                logger.info(f"  ✓ Completed: accuracy={acc:.4f}, f1_macro={f1:.4f}")
            else:
                logger.info(f"  ✗ Failed: {result.get('error', 'unknown')}")
    
    return results


# ============================================================================
# Output Functions
# ============================================================================

def save_results(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save results to JSON and CSV files.
    
    Args:
        results: List of result dictionaries.
        output_dir: Directory to save results.
        
    Returns:
        Tuple of (json_path, csv_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = output_dir / f"comparison_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "n_runs": len(results),
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved JSON results to: {json_path}")
    
    # Save CSV
    csv_path = output_dir / f"comparison_results_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV results to: {csv_path}")
    
    # Also save a "latest" copy for easy access
    latest_json = output_dir / "comparison_results_latest.json"
    latest_csv = output_dir / "comparison_results_latest.csv"
    
    with open(latest_json, "w") as f:
        json.dump({"timestamp": timestamp, "n_runs": len(results), "results": results}, f, indent=2)
    df.to_csv(latest_csv, index=False)
    
    return json_path, csv_path


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print a summary table of results."""
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Filter successful runs
    successful = df[df["status"] == "success"]
    
    if len(successful) == 0:
        logger.warning("No successful runs!")
        return
    
    # Pivot table: models as rows, datasets as columns, f1_macro as values
    if "f1_macro" in successful.columns:
        pivot = successful.pivot_table(
            index="model",
            columns="dataset",
            values="f1_macro",
            aggfunc="first",
        )
        logger.info("F1 Macro Score by Model × Dataset:")
        logger.info(pivot.round(4).to_string())
    
    # Best combination per model
    logger.info("-" * 40)
    logger.info("Best dataset per model (by F1 macro):")
    logger.info("-" * 40)
    
    for model in successful["model"].unique():
        model_runs = successful[successful["model"] == model]
        if "f1_macro" in model_runs.columns and model_runs["f1_macro"].notna().any():
            best_idx = model_runs["f1_macro"].idxmax()
            best = model_runs.loc[best_idx]
            logger.info(f"  {model}: {best['dataset']} (f1={best['f1_macro']:.4f})")
        else:
            logger.info(f"  {model}: (no f1_macro available)")
    
    # Overall best
    if "f1_macro" in successful.columns and successful["f1_macro"].notna().any():
        best_overall = successful.loc[successful["f1_macro"].idxmax()]
        logger.info("-" * 40)
        logger.info(f"BEST OVERALL: {best_overall['model']} + {best_overall['dataset']}")
        logger.info(f"  F1 Macro: {best_overall['f1_macro']:.4f}")
        logger.info(f"  Accuracy: {best_overall.get('accuracy', 'N/A')}")
    
    # Timing summary
    total_time = df["duration_seconds"].sum()
    logger.info("-" * 40)
    logger.info(f"Total training time: {total_time/60:.1f} minutes")
    logger.info(f"Average per run: {total_time/len(df):.1f} seconds")
    
    # Error summary
    failed = df[df["status"] != "success"]
    if len(failed) > 0:
        logger.info("-" * 40)
        logger.info(f"FAILED RUNS: {len(failed)}")
        for _, row in failed.iterrows():
            logger.info(f"  - {row['model']} + {row['dataset']}: {row.get('error', 'unknown')}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train all models across all dataset combinations"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50000,
        help="Sample size for each training run (default: 50000)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=[
            "simple_rf", "simplified", "simplified_zones", "hierarchical",
            "tuned_lgbm", "tuned_xgb", "tuned_catboost", "tuned_rf", "tabnet",
        ],
        default=None,
        help="Subset of models to train (default: all)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=[d.name for d in DATASET_COMBINATIONS],
        default=None,
        help="Subset of datasets to use (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: models/trained/comparison)",
    )
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "models" / "trained" / "comparison"
    )
    
    # Run all combinations
    results = run_all_combinations(
        sample_size=args.sample,
        models=args.models,
        datasets=args.datasets,
    )
    
    # Save results
    json_path, csv_path = save_results(results, output_dir)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
