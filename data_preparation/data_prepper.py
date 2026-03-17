"""Data preparation orchestration module.

This module provides high-level functions for preparing data,
with optional caching of processed tensor data.
"""

from pathlib import Path

import pandas as pd

from data_preparation.prepare_tensor_data import TensorPipelineResult, prepare_tensor_data
from data_preparation.tensor_config import SEVERITY_PREDICTION_CONFIG, TensorConfig


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def prepare_data(
    config: TensorConfig | None = None,
    use_cache: bool = True,
    cache_name: str = "tensor_pipeline",
    df: pd.DataFrame | None = None,
) -> TensorPipelineResult:
    """Prepare tensor data with optional caching.

    Args:
        config: Configuration for tensor preparation. If None, uses
            SEVERITY_PREDICTION_CONFIG.
        use_cache: Whether to use cached data if available.
        cache_name: Name for the cache file (without extension).
        df: Optional pre-loaded DataFrame. If None, loads from CSV.
            Use this to pass merged crash+people data.

    Returns:
        TensorPipelineResult containing train/val/test datasets and encoders.
    """
    if config is None:
        config = SEVERITY_PREDICTION_CONFIG

    cache_path = CACHE_DIR / f"{cache_name}.pt"

    # Try to load from cache (only if no custom DataFrame provided)
    if use_cache and cache_path.exists() and df is None:
        print(f"Loading cached tensor data from {cache_path}")
        return TensorPipelineResult.load(cache_path)

    # Prepare fresh data
    print("Preparing tensor data from CSV...")
    result = prepare_tensor_data(config, df=df)

    # Save to cache (only if no custom DataFrame provided)
    if use_cache and df is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.save(cache_path)
        print(f"Cached tensor data to {cache_path}")

    return result
