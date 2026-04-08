"""Simplified 3-class target definitions for crash severity.

Merges the 5 original severity classes into 3 actionable categories:
- SEVERE: FATAL + INCAPACITATING INJURY
- MINOR: NONINCAPACITATING INJURY + REPORTED, NOT EVIDENT
- NO_INJURY: NO INDICATION OF INJURY
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class SimplifiedSeverity(Enum):
    """Simplified 3-class severity enumeration."""

    NO_INJURY = 0
    MINOR = 1
    SEVERE = 2


# Mapping from original 5-class to simplified 3-class
SEVERITY_MAPPING = {
    "NO INDICATION OF INJURY": SimplifiedSeverity.NO_INJURY,
    "NONINCAPACITATING INJURY": SimplifiedSeverity.MINOR,
    "REPORTED, NOT EVIDENT": SimplifiedSeverity.MINOR,
    "INCAPACITATING INJURY": SimplifiedSeverity.SEVERE,
    "FATAL": SimplifiedSeverity.SEVERE,
    "UNKNOWN": SimplifiedSeverity.NO_INJURY,  # Treat unknown as no injury
}

# Class names for the simplified system
SIMPLIFIED_CLASS_NAMES = ["NO_INJURY", "MINOR", "SEVERE"]


@dataclass
class SimplifiedTargets:
    """Container for simplified hierarchical binary targets.

    Attributes:
        y_injury: Binary labels (1=any injury, 0=no injury).
        y_severe: Binary labels (1=severe, 0=minor). Valid only when y_injury=1.
        y_simplified: Simplified 3-class labels (0=NO_INJURY, 1=MINOR, 2=SEVERE).
        label_encoder: Encoder for simplified severity labels.
    """

    y_injury: np.ndarray
    y_severe: np.ndarray
    y_simplified: np.ndarray
    label_encoder: LabelEncoder


def prepare_simplified_targets(df: pd.DataFrame) -> SimplifiedTargets:
    """Prepare simplified 3-class targets from DataFrame.

    Expects DataFrame to have:
    - IS_INJURY: 1 if any injury, 0 otherwise
    - IS_SEVERE: 1 if FATAL or INCAPACITATING, 0 otherwise
    - MOST_SEVERE_INJURY: Original severity label string

    Args:
        df: DataFrame with binary target columns.

    Returns:
        SimplifiedTargets containing binary and 3-class labels.

    Raises:
        KeyError: If required columns are missing.
    """
    required_cols = ["IS_INJURY", "IS_SEVERE", "MOST_SEVERE_INJURY"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Map original labels to simplified classes
    simplified_labels = df["MOST_SEVERE_INJURY"].fillna("UNKNOWN").map(
        lambda x: SEVERITY_MAPPING.get(x, SimplifiedSeverity.NO_INJURY).value
    )

    # Create label encoder for simplified classes
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(SIMPLIFIED_CLASS_NAMES)

    return SimplifiedTargets(
        y_injury=df["IS_INJURY"].values.astype(np.int32),
        y_severe=df["IS_SEVERE"].values.astype(np.int32),
        y_simplified=simplified_labels.values.astype(np.int32),
        label_encoder=label_encoder,
    )


def map_simplified_predictions(
    l1_pred: np.ndarray,
    l2_pred: np.ndarray,
) -> np.ndarray:
    """Map 2-level binary predictions to simplified 3-class labels.

    Mapping logic:
    - L1=0 (no injury) → NO_INJURY (0)
    - L1=1, L2=0 (injury, minor) → MINOR (1)
    - L1=1, L2=1 (injury, severe) → SEVERE (2)

    Args:
        l1_pred: Level 1 binary predictions (injury).
        l2_pred: Level 2 binary predictions (severity).

    Returns:
        Array of simplified 3-class predictions.
    """
    # Start with no injury (class 0)
    result = np.zeros(len(l1_pred), dtype=np.int32)

    # Injury cases
    injury_mask = l1_pred == 1

    # Minor injuries (injury but not severe)
    minor_mask = injury_mask & (l2_pred == 0)
    result[minor_mask] = SimplifiedSeverity.MINOR.value

    # Severe injuries
    severe_mask = injury_mask & (l2_pred == 1)
    result[severe_mask] = SimplifiedSeverity.SEVERE.value

    return result
