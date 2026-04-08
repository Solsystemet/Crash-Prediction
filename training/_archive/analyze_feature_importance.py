"""Feature importance analysis with CSV and plot outputs.

This module trains a Random Forest on the FULL merged dataset (crashes +
vehicles + people + weather) and generates:
- CSV with ranked feature importances and recommendations
- Horizontal bar chart visualization

Uses the same data loading as main_simplified.py (triple_merge + 
engineer_all_features) to analyze all ~72 features.

Run with: python -m training.analyze_feature_importance
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_preparation.feature_engineering import (
    add_binary_targets,
    engineer_all_features,
)
from data_preparation.triple_merge import triple_merge


# Output directory
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "plots"

# Recommendation thresholds (percentage)
THRESHOLD_DROP = 0.5  # Below this -> DROP
THRESHOLD_KEEP = 2.0  # Above this -> KEEP, between -> REVIEW


def get_recommendation(importance_pct: float) -> str:
    """Return recommendation based on importance percentage."""
    if importance_pct < THRESHOLD_DROP:
        return "DROP"
    elif importance_pct >= THRESHOLD_KEEP:
        return "KEEP"
    else:
        return "REVIEW"


def get_recommendation_color(recommendation: str) -> str:
    """Return color for bar chart based on recommendation."""
    colors = {
        "KEEP": "#2ecc71",    # Green
        "REVIEW": "#f39c12",  # Yellow/Orange
        "DROP": "#e74c3c",    # Red
    }
    return colors.get(recommendation, "#95a5a6")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare feature matrix from DataFrame.
    
    Same logic as main_simplified.py to ensure consistency.

    Args:
        df: DataFrame with engineered features.

    Returns:
        Tuple of (feature DataFrame, feature column names).
    """
    exclude_cols = {
        # Identifiers and timestamps
        "CRASH_RECORD_ID", "CRASH_DATE", "DATE_POLICE_NOTIFIED", "CRASH_DATE_EST_I",
        # Direct injury indicators (targets)
        "MOST_SEVERE_INJURY", "INJURIES_TOTAL", "INJURIES_FATAL",
        "INJURIES_INCAPACITATING", "INJURIES_NON_INCAPACITATING",
        "INJURIES_REPORTED_NOT_EVIDENT", "INJURIES_NO_INDICATION", "INJURIES_UNKNOWN",
        # Binary targets
        "IS_INJURY", "IS_SEVERE", "IS_FATAL", "IS_REPORTED",
        "SEVERITY_LEVEL", "SEVERITY_ENCODED",
        # Geographic coordinates
        "LATITUDE", "LONGITUDE", "LOCATION",
        # Post-crash investigation fields (data leakage!)
        "STATEMENTS_TAKEN_I", "WORK_ZONE_I", "PHOTOS_TAKEN_I", "DOORING_I",
        # People-derived outcome features (data leakage!)
        "ejected_any", "using_seatbelt_mean", "cell_phone_any",
        "bac_positive_any", "bac_clean_max",
        # CRASH_TYPE directly encodes injury!
        "CRASH_TYPE", "REPORT_TYPE",
        # Redundant temporal
        "HOUR",
        # High-cardinality location features
        "STREET_NAME",
    }

    safe_categorical_cols = [
        "WEATHER_CONDITION", "LIGHTING_CONDITION", "FIRST_CRASH_TYPE",
        "TRAFFICWAY_TYPE", "ROADWAY_SURFACE_COND", "TRAFFIC_CONTROL_DEVICE",
        "DEVICE_CONDITION", "ALIGNMENT", "ROAD_DEFECT",
        "PRIM_CONTRIBUTORY_CAUSE", "DAMAGE",
    ]

    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            feature_cols.append(col)

    df_features = df[feature_cols].copy()

    for col in safe_categorical_cols:
        if col in df.columns and col not in exclude_cols:
            le = LabelEncoder()
            values = df[col].fillna("UNKNOWN").astype(str)
            df_features[col] = le.fit_transform(values)
            if col not in feature_cols:
                feature_cols.append(col)

    feature_cols = list(df_features.columns)
    df_features = df_features.fillna(0)

    return df_features, feature_cols


def analyze_feature_importance(
    sample_size: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train Random Forest and analyze feature importances.
    
    Uses triple_merge to load all datasets (crashes + vehicles + people + weather)
    and engineer_all_features to create the full feature set.
    
    Args:
        sample_size: Optional limit on dataset size for faster experimentation.
        verbose: Whether to print progress messages.
    
    Returns:
        DataFrame with feature importances and recommendations.
    """
    # =========================================================================
    # Step 1: Load and Merge Data (same as main_simplified.py)
    # =========================================================================
    if verbose:
        print("=" * 60)
        print("Feature Importance Analysis (Full 72-Feature Dataset)")
        print("=" * 60)
        print("\n[1/4] Loading and merging data...")
    
    df = triple_merge()
    
    if verbose:
        print(f"  Merged dataset shape: {df.shape}")
    
    # Sample if specified
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        if verbose:
            print(f"  Sampled to {len(df):,} rows")
    
    # Engineer features and add targets
    df = engineer_all_features(df, include_interactions=True, include_clusters=False)
    df = add_binary_targets(df)
    
    # Prepare features (same logic as main_simplified.py)
    X_df, feature_names = prepare_features(df)
    
    # Prepare target (IS_INJURY binary for simplicity)
    y = df["IS_INJURY"].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if verbose:
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Features: {len(feature_names)}")
    
    # =========================================================================
    # Step 2: Train Random Forest
    # =========================================================================
    if verbose:
        print("\n[2/4] Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    
    if verbose:
        print("  Training complete.")
    
    # =========================================================================
    # Step 3: Extract and Analyze Importances
    # =========================================================================
    if verbose:
        print("\n[3/4] Analyzing feature importances...")
    
    importances = model.feature_importances_
    
    # Create DataFrame
    results_df = pd.DataFrame({
        "feature_name": feature_names,
        "importance": importances,
    })
    
    # Sort by importance descending
    results_df = results_df.sort_values("importance", ascending=False).reset_index(drop=True)
    
    # Calculate percentage and cumulative percentage
    total_importance = results_df["importance"].sum()
    results_df["importance_pct"] = (results_df["importance"] / total_importance) * 100
    results_df["cumulative_pct"] = results_df["importance_pct"].cumsum()
    
    # Add recommendations
    results_df["recommendation"] = results_df["importance_pct"].apply(get_recommendation)
    
    # Add rank
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    
    return results_df


def save_csv(df: pd.DataFrame, verbose: bool = True) -> Path:
    """Save feature importances to CSV.
    
    Args:
        df: DataFrame with feature importances.
        verbose: Whether to print progress messages.
    
    Returns:
        Path to saved CSV file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "feature_importance.csv"
    
    df.to_csv(csv_path, index=False, float_format="%.6f")
    
    if verbose:
        print(f"  Saved CSV: {csv_path}")
    
    return csv_path


def save_plot(df: pd.DataFrame, max_features: int = 30, verbose: bool = True) -> Path:
    """Save horizontal bar chart of feature importances.
    
    Args:
        df: DataFrame with feature importances.
        max_features: Maximum number of features to show.
        verbose: Whether to print progress messages.
    
    Returns:
        Path to saved PNG file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "feature_importance.png"
    
    # Take top N features
    plot_df = df.head(max_features).copy()
    
    # Reverse for horizontal bar chart (top feature at top)
    plot_df = plot_df.iloc[::-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.4)))
    
    # Get colors based on recommendations
    colors = [get_recommendation_color(rec) for rec in plot_df["recommendation"]]
    
    # Create horizontal bars
    y_pos = range(len(plot_df))
    bars = ax.barh(y_pos, plot_df["importance_pct"], color=colors, edgecolor="white")
    
    # Add feature names as y-tick labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature_name"], fontsize=10)
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, plot_df["importance_pct"]):
        width = bar.get_width()
        ax.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )
    
    # Styling
    ax.set_xlabel("Importance (%)", fontsize=12)
    ax.set_title("Feature Importance Analysis", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label=f"KEEP (≥{THRESHOLD_KEEP}%)"),
        Patch(facecolor="#f39c12", label=f"REVIEW ({THRESHOLD_DROP}-{THRESHOLD_KEEP}%)"),
        Patch(facecolor="#e74c3c", label=f"DROP (<{THRESHOLD_DROP}%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    if verbose:
        print(f"  Saved plot: {plot_path}")
    
    return plot_path


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics to console."""
    total = len(df)
    keep_count = (df["recommendation"] == "KEEP").sum()
    review_count = (df["recommendation"] == "REVIEW").sum()
    drop_count = (df["recommendation"] == "DROP").sum()
    
    # Find how many features account for 90% of importance
    features_for_90pct = (df["cumulative_pct"] <= 90).sum() + 1
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal features: {total}")
    print(f"  KEEP:   {keep_count:3d} ({keep_count/total*100:.1f}%)")
    print(f"  REVIEW: {review_count:3d} ({review_count/total*100:.1f}%)")
    print(f"  DROP:   {drop_count:3d} ({drop_count/total*100:.1f}%)")
    print(f"\nFeatures for 90% importance: {features_for_90pct}")
    
    print("\n" + "-" * 60)
    print("TOP 10 FEATURES")
    print("-" * 60)
    for _, row in df.head(10).iterrows():
        print(f"  {row['rank']:2d}. {row['feature_name']:<30} {row['importance_pct']:5.1f}%  [{row['recommendation']}]")
    
    if drop_count > 0:
        print("\n" + "-" * 60)
        print("RECOMMENDED FOR REMOVAL")
        print("-" * 60)
        drop_features = df[df["recommendation"] == "DROP"]["feature_name"].tolist()
        for feat in drop_features:
            print(f"  - {feat}")


def main() -> None:
    """Run feature importance analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze feature importance")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for faster experimentation (default: use all data)",
    )
    args = parser.parse_args()
    
    # Analyze
    df = analyze_feature_importance(sample_size=args.sample, verbose=True)
    
    # Save outputs
    print("\n[4/4] Saving outputs...")
    save_csv(df, verbose=True)
    save_plot(df, verbose=True)
    
    # Print summary
    print_summary(df)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
