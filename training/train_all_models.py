"""Train all models without feature filtering for comparison.

Runs each training script sequentially with --feature-filter none
to ensure all models are trained with the same full feature set.

Usage:
    python -m training.train_all_models
    python -m training.train_all_models --sample 50000  # For faster testing
    python -m training.train_all_models --skip-boosting  # Skip slow tuning
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logging_config import setup_logging

logger = setup_logging(__name__)


def run_training_script(
    script: str,
    args: list[str],
    description: str,
) -> bool:
    """Run a training script and return success status.
    
    Args:
        script: Module path (e.g., 'training.main_simple_rf').
        args: Additional command line arguments.
        description: Description for logging.
        
    Returns:
        True if successful, False otherwise.
    """
    cmd = [sys.executable, "-m", script] + args
    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
        )
        elapsed = time.time() - start_time
        logger.info(f"✓ {description} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ {description} failed after {elapsed:.1f}s (exit code {e.returncode})")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed: {e}")
        return False


def main(
    sample_size: int | None = None,
    skip_boosting: bool = False,
    skip_tabnet: bool = False,
    n_trials: int = 30,
) -> None:
    """Train all models without feature filtering.
    
    Args:
        sample_size: Optional sample size for faster training.
        skip_boosting: Skip the slow boosting hyperparameter tuning.
        skip_tabnet: Skip TabNet training.
        n_trials: Number of Optuna trials for boosting tuning.
    """
    logger.info("="*60)
    logger.info("TRAINING ALL MODELS (no feature filtering)")
    logger.info("="*60)
    
    base_args = ["--feature-filter", "none"]
    if sample_size:
        base_args.extend(["--sample", str(sample_size)])
    
    results = {}
    total_start = time.time()
    
    # 1. Random Forest baseline
    results["Random Forest"] = run_training_script(
        "training.main_simple_rf",
        base_args,
        "Random Forest (baseline)",
    )
    
    # 2. Boosting models (LightGBM, XGBoost, CatBoost)
    if not skip_boosting:
        for model_type in ["lgbm", "xgb", "catboost"]:
            model_name = {
                "lgbm": "LightGBM",
                "xgb": "XGBoost", 
                "catboost": "CatBoost",
            }[model_type]
            
            boosting_args = base_args + ["--model", model_type, "--n-trials", str(n_trials)]
            results[model_name] = run_training_script(
                "training.tune_boosting",
                boosting_args,
                f"{model_name} (tuned)",
            )
    else:
        logger.info("\nSkipping boosting models (--skip-boosting)")
    
    # 3. TabNet
    if not skip_tabnet:
        results["TabNet"] = run_training_script(
            "training.tabnet.train",
            base_args,
            "TabNet",
        )
    else:
        logger.info("\nSkipping TabNet (--skip-tabnet)")
    
    # Summary
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    
    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {name}: {status}")
    
    successful = sum(results.values())
    total = len(results)
    logger.info(f"\nCompleted {successful}/{total} models in {total_elapsed/60:.1f} minutes")
    
    if successful == total:
        logger.info("\nAll models trained successfully!")
        logger.info("Run: python -m training.compare_all_models")
    else:
        logger.warning("\nSome models failed. Check logs above for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all models without feature filtering"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for faster training",
    )
    parser.add_argument(
        "--skip-boosting",
        action="store_true",
        help="Skip slow boosting hyperparameter tuning",
    )
    parser.add_argument(
        "--skip-tabnet",
        action="store_true",
        help="Skip TabNet training",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for boosting tuning (default: 30)",
    )
    
    args = parser.parse_args()
    main(
        sample_size=args.sample,
        skip_boosting=args.skip_boosting,
        skip_tabnet=args.skip_tabnet,
        n_trials=args.n_trials,
    )
