"""Data preparation orchestration module.

This module provides high-level functions for preparing data,
with optional caching of processed tensor data.
"""

import logging
from pathlib import Path

from data_preparation.prepare_tensor_data import TensorPipelineResult, prepare_tensor_data
from data_preparation.tensor_config import SEVERITY_PREDICTION_CONFIG, TensorConfig

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def prepare_data(
    config: TensorConfig | None = None,
    use_cache: bool = True,
    cache_name: str = "tensor_pipeline",
) -> TensorPipelineResult:
    """Prepare tensor data with optional caching.

    Args:
        config: Configuration for tensor preparation. If None, uses
            SEVERITY_PREDICTION_CONFIG.
        use_cache: Whether to use cached data if available.
        cache_name: Name for the cache file (without extension).

    Returns:
        TensorPipelineResult containing train/val/test datasets and encoders.
    """
    if config is None:
        config = SEVERITY_PREDICTION_CONFIG

    cache_path = CACHE_DIR / f"{cache_name}.pt"

    # Try to load from cache
    if use_cache and cache_path.exists():
        logger.info(f"Loading cached tensor data from {cache_path}")
        return TensorPipelineResult.load(cache_path)

    # Prepare fresh data
    logger.info("Preparing tensor data from CSV...")
    result = prepare_tensor_data(config)

    # Save to cache
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.save(cache_path)
        logger.info(f"Cached tensor data to {cache_path}")

    return result
