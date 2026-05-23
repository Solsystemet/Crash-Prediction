"""
Feature importance plotting utilities.

Provides standardized feature importance visualization for tree-based models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils.csv_filename_generator import generate_csv_filename


def get_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Extract feature importance from a model.

    Supports: RandomForest, XGBoost, LightGBM, CatBoost, and sklearn tree models.

    Args:
        model: Trained model with feature_importances_ or similar attribute
        feature_names: List of feature names. If None, uses indices.

    Returns:
        Tuple of (feature_names, importance_values)
    """
    # Try different importance extraction methods
    importances = None

    # Standard sklearn interface
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # XGBoost
    elif hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            importance_dict = booster.get_score(importance_type="gain")
            if feature_names is None:
                feature_names = list(importance_dict.keys())
            importances = np.array([importance_dict.get(f, 0) for f in feature_names])
        except Exception:
            pass

    # CatBoost
    elif hasattr(model, "get_feature_importance"):
        importances = model.get_feature_importance()

    # LightGBM
    elif hasattr(model, "booster_"):
        try:
            importances = model.booster_.feature_importance(importance_type="gain")
        except Exception:
            pass

    if importances is None:
        raise ValueError(
            f"Could not extract feature importance from model of type {type(model)}"
        )

    importances = np.array(importances)

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    return feature_names, importances


def plot_feature_importance(
    model: Any = None,
    feature_names: Optional[List[str]] = None,
    importances: Optional[np.ndarray] = None,
    output_path: Optional[str | Path] = None,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: tuple = (10, 8),
    color: str = "steelblue",
    horizontal: bool = True,
) -> plt.Figure:
    """
    Plot feature importance as a bar chart.

    Can either extract importance from a model or use provided values.

    Args:
        model: Trained model (if importances not provided)
        feature_names: List of feature names
        importances: Pre-computed importance values (if model not provided)
        output_path: Path to save the plot
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        color: Bar color
        horizontal: If True, use horizontal bars (better for long names)

    Returns:
        matplotlib Figure object
    """
    # Get importances from model if not provided
    if importances is None:
        if model is None:
            raise ValueError("Either model or importances must be provided")
        feature_names, importances = get_feature_importance(model, feature_names)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # Sort by importance
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]

    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]

    # Normalize to percentages
    total_importance = np.sum(importances)
    if total_importance > 0:
        top_importances_pct = (top_importances / total_importance) * 100
    else:
        top_importances_pct = top_importances

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        # Reverse for horizontal bar chart (top feature at top)
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importances_pct[::-1], color=color, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel("Importance (%)")
        ax.set_ylabel("Feature")

        # Add value labels
        for i, (val, feat) in enumerate(zip(top_importances_pct[::-1], top_features[::-1])):
            ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=9)
    else:
        x_pos = np.arange(len(top_features))
        ax.bar(x_pos, top_importances_pct, color=color, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_features, rotation=45, ha="right")
        ax.set_ylabel("Importance (%)")
        ax.set_xlabel("Feature")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x" if horizontal else "y")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

        # Save importance data as CSV with timestamp
        import pandas as pd

        csv_filename = generate_csv_filename(output_path.stem, model_name=None)
        csv_path = output_path.parent / csv_filename
        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
                "importance_pct": (importances / total_importance) * 100
                if total_importance > 0
                else importances,
            }
        )
        df = df.sort_values("importance", ascending=False)
        df.to_csv(csv_path, index=False)

    return fig


def plot_feature_importance_comparison(
    models_importances: Dict[str, Tuple[List[str], np.ndarray]],
    output_path: Optional[str | Path] = None,
    top_n: int = 15,
    title: str = "Feature Importance Comparison",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """
    Compare feature importance across multiple models.

    Args:
        models_importances: Dict mapping model name to (feature_names, importances) tuple
        output_path: Path to save the plot
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    # Aggregate importance across all models to find top features
    all_features = set()
    feature_scores: Dict[str, float] = {}

    for model_name, (features, importances) in models_importances.items():
        total = np.sum(importances)
        for feat, imp in zip(features, importances):
            all_features.add(feat)
            normalized = (imp / total) * 100 if total > 0 else 0
            feature_scores[feat] = feature_scores.get(feat, 0) + normalized

    # Get top N features by average importance
    top_features = sorted(feature_scores.keys(), key=lambda x: feature_scores[x], reverse=True)[
        :top_n
    ]

    # Prepare data for grouped bar chart
    n_models = len(models_importances)
    x = np.arange(len(top_features))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (model_name, (features, importances)) in enumerate(models_importances.items()):
        total = np.sum(importances)
        feat_to_imp = {f: (imp / total) * 100 if total > 0 else 0 for f, imp in zip(features, importances)}

        values = [feat_to_imp.get(f, 0) for f in top_features]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, color=colors[i])

    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
