"""Hierarchical structure definitions for crash severity classification.

This module defines the 4-level hierarchy and provides utilities for:
- Target preparation (binary labels for each level)
- Multiclass reconstruction from hierarchical predictions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class HierarchyLevel(Enum):
    """Enumeration of hierarchy levels."""

    L1_INJURY = auto()       # INJURY vs NO_INJURY
    L2_SEVERITY = auto()     # SEVERE vs MINOR (among injuries)
    L25_FATAL = auto()       # FATAL vs INCAPACITATING (among severe)
    L3_REPORTED = auto()     # REPORTED vs VISIBLE (among minor)


# Standard severity class ordering (alphabetical, sklearn default)
SEVERITY_CLASS_ORDER = [
    "FATAL",
    "INCAPACITATING INJURY",
    "NO INDICATION OF INJURY",
    "NONINCAPACITATING INJURY",
    "REPORTED, NOT EVIDENT",
]


@dataclass
class HierarchicalTargets:
    """Container for hierarchical binary targets.

    Attributes:
        y_injury: Binary labels (1=injury, 0=no injury).
        y_severe: Binary labels (1=severe, 0=not severe). Valid only when y_injury=1.
        y_fatal: Binary labels (1=fatal, 0=incapacitating). Valid only when y_severe=1.
        y_reported: Binary labels (1=reported, 0=visible). Valid only when y_severe=0.
        y_original: Original multiclass labels (encoded integers).
        label_encoder: Encoder for original severity labels.
    """

    y_injury: np.ndarray
    y_severe: np.ndarray
    y_fatal: np.ndarray
    y_reported: np.ndarray
    y_original: np.ndarray
    label_encoder: LabelEncoder


def prepare_hierarchical_targets(df: pd.DataFrame) -> HierarchicalTargets:
    """Prepare hierarchical binary targets from DataFrame.

    Expects DataFrame to have these columns (from add_binary_targets):
    - IS_INJURY: 1 if any injury, 0 otherwise
    - IS_SEVERE: 1 if FATAL or INCAPACITATING, 0 otherwise
    - IS_FATAL: 1 if FATAL, 0 if INCAPACITATING (valid for severe injuries)
    - IS_REPORTED: 1 if REPORTED, NOT EVIDENT, 0 if NONINCAPACITATING
    - MOST_SEVERE_INJURY: Original severity label string

    Args:
        df: DataFrame with binary target columns.

    Returns:
        HierarchicalTargets containing all binary label arrays.

    Raises:
        KeyError: If required columns are missing.
    """
    required_cols = ["IS_INJURY", "IS_SEVERE", "IS_FATAL", "IS_REPORTED", "MOST_SEVERE_INJURY"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Encode original multiclass target
    label_encoder = LabelEncoder()
    y_original = label_encoder.fit_transform(
        df["MOST_SEVERE_INJURY"].fillna("UNKNOWN")
    )

    return HierarchicalTargets(
        y_injury=df["IS_INJURY"].values.astype(np.int32),
        y_severe=df["IS_SEVERE"].values.astype(np.int32),
        y_fatal=df["IS_FATAL"].values.astype(np.int32),
        y_reported=df["IS_REPORTED"].values.astype(np.int32),
        y_original=y_original,
        label_encoder=label_encoder,
    )


def get_multiclass_indices(
    label_encoder: LabelEncoder | None = None,
) -> dict[str, int]:
    """Get indices for each severity class.

    Args:
        label_encoder: Optional encoder to get correct indices.
                       If None, uses alphabetical ordering.

    Returns:
        Dictionary mapping class names to indices.
    """
    if label_encoder is not None:
        classes = list(label_encoder.classes_)
    else:
        classes = SEVERITY_CLASS_ORDER

    # Build index mapping with defaults for missing classes
    indices = {
        "FATAL": 0,
        "INCAPACITATING": 1,
        "NO_INDICATION": 2,
        "NONINCAPACITATING": 3,
        "REPORTED": 4,
    }

    # Map from standardized keys to actual class names in encoder
    class_mappings = {
        "FATAL": "FATAL",
        "INCAPACITATING": "INCAPACITATING INJURY",
        "NO_INDICATION": "NO INDICATION OF INJURY",
        "NONINCAPACITATING": "NONINCAPACITATING INJURY",
        "REPORTED": "REPORTED, NOT EVIDENT",
    }

    for key, class_name in class_mappings.items():
        if class_name in classes:
            indices[key] = classes.index(class_name)

    return indices


def map_hierarchical_to_multiclass(
    l1_pred: np.ndarray,
    l2_pred: np.ndarray,
    l25_pred: np.ndarray,
    l3_pred: np.ndarray,
    label_encoder: LabelEncoder | None = None,
) -> np.ndarray:
    """Map hierarchical binary predictions to original 5-class labels.

    Mapping logic:
    - L1=0 (no injury) → NO INDICATION OF INJURY
    - L1=1, L2=1, L2.5=1 (injury, severe, fatal) → FATAL
    - L1=1, L2=1, L2.5=0 (injury, severe, not fatal) → INCAPACITATING INJURY
    - L1=1, L2=0, L3=1 (injury, minor, reported) → REPORTED, NOT EVIDENT
    - L1=1, L2=0, L3=0 (injury, minor, visible) → NONINCAPACITATING INJURY

    Args:
        l1_pred: Level 1 binary predictions (injury).
        l2_pred: Level 2 binary predictions (severity).
        l25_pred: Level 2.5 binary predictions (fatal).
        l3_pred: Level 3 binary predictions (reported).
        label_encoder: Optional encoder for class indices.

    Returns:
        Array of multiclass predictions.
    """
    idx = get_multiclass_indices(label_encoder)

    # Start with no injury
    multiclass = np.full(len(l1_pred), idx["NO_INDICATION"], dtype=int)

    # Level 1: Injury cases
    injury_mask = l1_pred == 1

    # Level 2: Severe vs Minor (among injuries)
    severe_mask = injury_mask & (l2_pred == 1)
    minor_mask = injury_mask & (l2_pred == 0)

    # Level 2.5: FATAL vs INCAPACITATING (among severe cases)
    fatal_mask = severe_mask & (l25_pred == 1)
    incap_mask = severe_mask & (l25_pred == 0)

    multiclass[fatal_mask] = idx["FATAL"]
    multiclass[incap_mask] = idx["INCAPACITATING"]

    # Level 3: Reported vs Visible (among minor injuries)
    reported_mask = minor_mask & (l3_pred == 1)
    visible_mask = minor_mask & (l3_pred == 0)

    multiclass[reported_mask] = idx["REPORTED"]
    multiclass[visible_mask] = idx["NONINCAPACITATING"]

    return multiclass


def split_targets_for_level(
    X: np.ndarray,
    targets: HierarchicalTargets,
    level: HierarchyLevel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get features and labels for a specific hierarchy level.

    Args:
        X: Full feature matrix.
        targets: HierarchicalTargets container.
        level: Which hierarchy level to prepare data for.

    Returns:
        Tuple of (X_subset, y_binary, mask) where mask indicates
        which samples from the original X are included.
    """
    if level == HierarchyLevel.L1_INJURY:
        # All samples for L1
        return X, targets.y_injury, np.ones(len(X), dtype=bool)

    elif level == HierarchyLevel.L2_SEVERITY:
        # Only injury cases for L2
        mask = targets.y_injury == 1
        return X[mask], targets.y_severe[mask], mask

    elif level == HierarchyLevel.L25_FATAL:
        # Only severe injury cases for L2.5
        mask = (targets.y_injury == 1) & (targets.y_severe == 1)
        return X[mask], targets.y_fatal[mask], mask

    elif level == HierarchyLevel.L3_REPORTED:
        # Only minor injury cases for L3
        mask = (targets.y_injury == 1) & (targets.y_severe == 0)
        return X[mask], targets.y_reported[mask], mask

    else:
        raise ValueError(f"Unknown level: {level}")
