"""Resampling utilities for handling class imbalance.

This module provides SMOTE and related oversampling techniques to address
the natural class imbalance in traffic crash data (fewer severe/fatal crashes).

Research shows that ignoring class imbalance leads to biased models that
fail to identify rare but critical severe accident cases.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Literal


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


def get_class_distribution(y: NDArray) -> dict[int, int]:
    """Get the distribution of classes in labels array.

    Args:
        y: Labels array.

    Returns:
        Dictionary mapping class label to count.
    """
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def print_class_distribution(
    y: NDArray,
    title: str = "Class Distribution",
    class_names: list[str] | None = None,
) -> None:
    """Print class distribution in a formatted way.

    Args:
        y: Labels array.
        title: Title for the output.
        class_names: Optional names for each class index.
    """
    dist = get_class_distribution(y)
    total = len(y)

    print(f"\n{title}")
    print("-" * 50)

    for label, count in sorted(dist.items()):
        pct = count / total * 100
        name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        bar = "█" * int(pct / 2)
        print(f"  {name:30s} {count:6,} ({pct:5.1f}%) {bar}")

    print(f"  {'Total':30s} {total:6,}")


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
        print_class_distribution(y_resampled, title="After Resampling")
        print(f"\n  Samples: {len(y):,} → {len(y_resampled):,} "
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
