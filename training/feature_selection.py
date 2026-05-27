"""Feature selection utilities.

This module implements hybrid feature selection combining:
- Correlation-based Feature Selection (CFS): Remove highly correlated features
- Recursive Feature Elimination (RFE): Iteratively remove least important features
- Importance-based filtering: Remove features based on pre-computed importance CSV

Research shows that removing noise and redundant features can significantly
boost accuracy and computational efficiency.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass

from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

# Default path to feature importance CSV (relative to project root)
DEFAULT_IMPORTANCE_CSV = Path(__file__).parent.parent / "models" / "plots" / "feature_importance.csv"

# Thresholds for auto-generating recommendations from importance values
IMPORTANCE_THRESHOLDS = {
    "drop": 0.05,    # Drop features with < 5% importance
    "review": 0.10,  # Review features with 5-10% importance
    # Features >= 10% importance are marked as KEEP
}


    Args:
        csv_path: Path to the feature_importance.csv file.
            If None, uses the default path.

    Returns:
        DataFrame with columns: feature_name, importance, recommendation.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
    """
    if csv_path is None:
        csv_path = DEFAULT_IMPORTANCE_CSV

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature importance CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Normalize column names
    col_mapping = {
        "feature": "feature_name",
        "importance_pct": "importance",
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Check for feature name column
    if "feature_name" not in df.columns:
        raise ValueError(f"CSV must contain 'feature_name' or 'feature' column. Found: {list(df.columns)}")
    
    # Auto-generate recommendations if not present
    if "recommendation" not in df.columns:
        if "importance" not in df.columns:
            raise ValueError(f"CSV must contain 'recommendation' or 'importance' column. Found: {list(df.columns)}")
        
        # Normalize importance to percentage if needed (values > 1 are already percentages)
        importance_vals = df["importance"].values
        if importance_vals.max() <= 1.0:
            importance_pct = importance_vals * 100
        else:
            importance_pct = importance_vals
        
        # Generate recommendations based on thresholds
        recommendations = []
        for imp in importance_pct:
            if imp < IMPORTANCE_THRESHOLDS["drop"] * 100:
                recommendations.append("DROP")
            elif imp < IMPORTANCE_THRESHOLDS["review"] * 100:
                recommendations.append("REVIEW")
            else:
                recommendations.append("KEEP")
        
        df["recommendation"] = recommendations
        logger.info(f"Auto-generated recommendations: {df['recommendation'].value_counts().to_dict()}")

    return df


def filter_by_importance(
    X_df: pd.DataFrame,
    feature_cols: list[str],
    mode: Literal["none", "drop-low", "drop-review"] = "drop-low",
    importance_csv: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str], ImportanceFilterResult]:
    """Filter features based on pre-computed importance recommendations.

    Args:
        X_df: Feature DataFrame.
        feature_cols: List of feature column names.
        mode: Filtering mode:
            - "none": No filtering, return original data.
            - "drop-low": Drop features marked as "DROP" in the CSV.
            - "drop-review": Drop features marked as "DROP" or "REVIEW" in the CSV.
        importance_csv: Path to importance CSV. Uses default if None.
        verbose: Whether to log filtering details.

    Returns:
        Tuple of (filtered DataFrame, filtered feature list, ImportanceFilterResult).
    """
    if mode == "none":
        result = ImportanceFilterResult(
            kept_features=feature_cols.copy(),
            dropped_features=[],
            n_original=len(feature_cols),
            n_kept=len(feature_cols),
            missing_features=[],
        )
        return X_df, feature_cols, result

    # Load importance data (gracefully handle missing CSV)
    try:
        importance_df = load_feature_importance(importance_csv)
    except FileNotFoundError:
        # Try to find any feature importance CSV in models directory
        models_dir = Path(__file__).parent.parent / "models"
        csv_files = list(models_dir.glob("**/feature_importance*.csv"))
        
        importance_df = None
        if csv_files:
            # Use the most recent one
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            if verbose:
                logger.info(f"Using feature importance CSV: {latest_csv}")
            try:
                importance_df = load_feature_importance(latest_csv)
            except (FileNotFoundError, ValueError) as e:
                if verbose:
                    logger.warning(f"Could not load {latest_csv}: {e}")
        
        if importance_df is None:
            if verbose:
                logger.warning(
                    f"No feature importance CSV found. Skipping feature filtering. "
                    f"Run feature analysis first or use --feature-filter none."
                )
            result = ImportanceFilterResult(
                kept_features=feature_cols.copy(),
                dropped_features=[],
                n_original=len(feature_cols),
                n_kept=len(feature_cols),
                missing_features=[],
            )
            return X_df, feature_cols, result
    
    importance_map = dict(zip(importance_df["feature_name"], importance_df["recommendation"]))

    # Categorize features
    kept_features = []
    dropped_features = []
    missing_features = []

    # Determine which recommendations to drop based on mode
    drop_recommendations = {"DROP"}
    if mode == "drop-review":
        drop_recommendations.add("REVIEW")

    for col in feature_cols:
        recommendation = importance_map.get(col)
        if recommendation is None:
            # Feature not in CSV - keep it by default with warning
            missing_features.append(col)
            kept_features.append(col)
        elif recommendation in drop_recommendations:
            dropped_features.append(col)
        else:
            # KEEP (or REVIEW when mode is drop-low) - retain
            kept_features.append(col)

    # Filter DataFrame
    X_filtered = X_df[kept_features].copy()

    result = ImportanceFilterResult(
        kept_features=kept_features,
        dropped_features=dropped_features,
        n_original=len(feature_cols),
        n_kept=len(kept_features),
        missing_features=missing_features,
    )

    if verbose:
        logger.info(f"Feature Importance Filter (mode={mode}):")
        logger.info(f"  Original: {result.n_original} features")
        logger.info(f"  Kept: {result.n_kept} features")
        logger.info(f"  Dropped: {len(dropped_features)} features")
        if dropped_features:
            logger.info(f"  Dropped features: {dropped_features[:10]}")
            if len(dropped_features) > 10:
                logger.info(f"    ... and {len(dropped_features) - 10} more")
        if missing_features:
            logger.warning(f"  Features not in CSV (kept by default): {missing_features}")

    return X_filtered, kept_features, result
