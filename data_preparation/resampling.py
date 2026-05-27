"""Resampling utilities for handling class imbalance.

This module provides SMOTE and related oversampling techniques to address
the natural class imbalance in traffic crash data (fewer severe/fatal crashes).

Research shows that ignoring class imbalance leads to biased models that
fail to identify rare but critical severe accident cases.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


# Try to import imbalanced-learn, provide helpful message if not available
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    from imblearn.combine import SMOTEENN, SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


ResamplingStrategy = Literal["none", "smote", "borderline", "adasyn", "smoteenn", "smotetomek"]


def check_imblearn_available() -> None:
    """Check if imbalanced-learn is available, raise helpful error if not."""
    if not IMBLEARN_AVAILABLE:
        raise ImportError(
            "imbalanced-learn is required for resampling. "
            "Install it with: pip install imbalanced-learn"
        )


def temporal_train_test_split(
    df: "pd.DataFrame",
    date_col: str = "CRASH_DATE",
    test_size: float = 0.2,
    date_format: str = "%m/%d/%Y %I:%M:%S %p",
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Split DataFrame chronologically for realistic future-prediction evaluation.
    
    This avoids temporal leakage by ensuring the test set contains only
    crashes that occurred AFTER all training crashes. Random splits can
    leak future information into training, leading to over-optimistic results.
    
    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column for sorting.
        test_size: Fraction of data to use for testing (default 0.2 = 20%).
        date_format: Format string for parsing dates.
        
    Returns:
        Tuple of (train_df, test_df) where test_df contains the most recent data.
        
    Example:
        >>> train_df, test_df = temporal_train_test_split(df, test_size=0.2)
        >>> # train_df has oldest 80%, test_df has newest 20%
    """
    import pandas as pd
    
    df = df.copy()
    
    # Parse dates if not already datetime
    if date_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df["_parsed_date"] = pd.to_datetime(
                df[date_col], format=date_format, errors="coerce"
            )
        else:
            df["_parsed_date"] = df[date_col]
    else:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    # Sort by date
    df_sorted = df.sort_values("_parsed_date", na_position="first")
    
    # Calculate split point
    n_total = len(df_sorted)
    n_train = int(n_total * (1 - test_size))
    
    train_df = df_sorted.iloc[:n_train].drop(columns=["_parsed_date"])
    test_df = df_sorted.iloc[n_train:].drop(columns=["_parsed_date"])
    
    # Log date ranges
    if "_parsed_date" in df_sorted.columns or date_col in df_sorted.columns:
        train_dates = df_sorted.iloc[:n_train]["_parsed_date"]
        test_dates = df_sorted.iloc[n_train:]["_parsed_date"]
        
        train_start = train_dates.min()
        train_end = train_dates.max()
        test_start = test_dates.min()
        test_end = test_dates.max()
        
        logger.info(f"Temporal split: train={n_train}, test={n_total - n_train}")
        logger.info(f"  Train period: {train_start} to {train_end}")
        logger.info(f"  Test period:  {test_start} to {test_end}")
    
    return train_df, test_df


    y: NDArray,
    title: str = "Class Distribution",
    class_names: list[str] | None = None,
) -> None:
    """Log class distribution in a formatted way.

    Args:
        y: Labels array.
        title: Title for the output.
        class_names: Optional names for each class index.
    """
    dist = get_class_distribution(y)
    total = len(y)

    logger.info(f"{title}")
    logger.info("-" * 50)

    for label, count in sorted(dist.items()):
        pct = count / total * 100
        name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        bar = "█" * int(pct / 2)
        logger.info(f"  {name:30s} {count:6,} ({pct:5.1f}%) {bar}")

    logger.info(f"  {'Total':30s} {total:6,}")


# Alias for backward compatibility
print_class_distribution = log_class_distribution


def apply_resampling(
    X: NDArray,
    y: NDArray,
    strategy: ResamplingStrategy = "smote",
    random_state: int = 42,
    verbose: bool = True,
    k_neighbors: int = 5,
) -> tuple[NDArray, NDArray]:
    """Apply resampling to address class imbalance.

    WARNING: Only apply to TRAINING data, never to validation/test sets!

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Labels array of shape (n_samples,).
        strategy: Resampling strategy to use.
            - "none": No resampling, return original data.
            - "smote": Standard SMOTE oversampling.
            - "borderline": Borderline-SMOTE (focuses on boundary samples).
            - "adasyn": Adaptive Synthetic Sampling.
            - "smoteenn": SMOTE + Edited Nearest Neighbors (combined over/under).
            - "smotetomek": SMOTE + Tomek links cleaning.
        random_state: Random seed for reproducibility.
        verbose: Whether to print before/after distribution.
        k_neighbors: Number of neighbors for SMOTE variants.

    Returns:
        Tuple of (resampled_X, resampled_y).
    """
    if strategy == "none":
        return X, y

    check_imblearn_available()

    if verbose:
        print_class_distribution(y, title="Before Resampling")

    # Adjust k_neighbors if any class has fewer samples
    min_class_size = min(get_class_distribution(y).values())
    effective_k = min(k_neighbors, min_class_size - 1)
    if effective_k < 1:
        effective_k = 1

    # Select resampling method
    match strategy:
        case "smote":
            sampler = SMOTE(
                random_state=random_state,
                k_neighbors=effective_k,
            )
        case "borderline":
            sampler = BorderlineSMOTE(
                random_state=random_state,
                k_neighbors=effective_k,
            )
        case "adasyn":
            sampler = ADASYN(
                random_state=random_state,
                n_neighbors=effective_k,
            )
        case "smoteenn":
            sampler = SMOTEENN(
                random_state=random_state,
                smote=SMOTE(random_state=random_state, k_neighbors=effective_k),
            )
        case "smotetomek":
            sampler = SMOTETomek(
                random_state=random_state,
                smote=SMOTE(random_state=random_state, k_neighbors=effective_k),
            )
        case _:
            raise ValueError(f"Unknown resampling strategy: {strategy}")

    # Apply resampling
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    if verbose:
        log_class_distribution(y_resampled, title="After Resampling")
        logger.info(f"  Samples: {len(y):,} → {len(y_resampled):,} "
              f"(+{len(y_resampled) - len(y):,})")

    return X_resampled, y_resampled


def calculate_class_weights(y: NDArray) -> dict[int, float]:
    """Calculate class weights inversely proportional to class frequencies.

    This is an alternative to resampling - use class weights in loss function.

    Args:
        y: Labels array.

    Returns:
        Dictionary mapping class label to weight.
    """
    dist = get_class_distribution(y)
    total = len(y)
    n_classes = len(dist)

    weights = {}
    for label, count in dist.items():
        # Weight = total / (n_classes * count)
        # This gives higher weight to minority classes
        weights[label] = total / (n_classes * count)

    return weights


def get_sampling_strategy_balanced(y: NDArray) -> dict[int, int]:
    """Get sampling strategy to achieve balanced classes.

    Returns target counts that would make all classes equal to the majority.

    Args:
        y: Labels array.

    Returns:
        Dictionary mapping class label to target count.
    """
    dist = get_class_distribution(y)
    max_count = max(dist.values())

    return {label: max_count for label in dist.keys()}


def get_sampling_strategy_moderate(y: NDArray, ratio: float = 0.5) -> dict[int, int]:
    """Get sampling strategy for moderate balancing.

    Instead of fully balancing, only oversample minority classes to a fraction
    of the majority class. This can help avoid overfitting from too much
    synthetic data.

    Args:
        y: Labels array.
        ratio: Target ratio of minority to majority (0.5 = half as many).

    Returns:
        Dictionary mapping class label to target count.
    """
    dist = get_class_distribution(y)
    max_count = max(dist.values())
    target_min = int(max_count * ratio)

    result = {}
    for label, count in dist.items():
        # Only oversample if below target
        result[label] = max(count, target_min)

    return result
