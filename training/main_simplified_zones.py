"""Zone-based simplified 3-class severity classification pipeline.

This module extends the simplified 2-level classification approach by training
independent models for geographic zones created via K-means clustering.

Strategy:
- Use K-means clustering on LATITUDE/LONGITUDE to create zones (default: 25)
- Train independent SimplifiedTreeClassifier per zone
- Produce per-zone metrics and aggregate summary

Output classes (same as main_simplified.py):
- NO_INJURY: No indication of injury
- MINOR: Nonincapacitating + Reported injuries
- SEVERE: Fatal + Incapacitating injuries

Usage:
    python training/main_simplified_zones.py
    python training/main_simplified_zones.py --clusters 25 --sample 50000
    python training/main_simplified_zones.py --min-zone-size 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.feature_engineering import (
    add_binary_targets,
    engineer_all_features,
)
from data_preparation.triple_merge import triple_merge
from splice.k_means import LocationClusterer

from training.hierarchical.config import SimplifiedTreeConfig
from training.hierarchical.simplified_targets import (
    SimplifiedTargets,
    prepare_simplified_targets,
    SIMPLIFIED_CLASS_NAMES,
)
from training.hierarchical.simplified_classifier import (
    SimplifiedTreeClassifier,
    save_simplified_model,
)
from training.hierarchical.evaluation import plot_roc_curves
from training.feature_selection import filter_by_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ZoneResults:
    """Container for per-zone evaluation metrics."""

    zone_id: int
    n_samples: int
    n_train: int
    n_val: int
    n_test: int
    l1_accuracy: float = 0.0
    l1_recall: float = 0.0
    l1_f1: float = 0.0
    l2_accuracy: float = 0.0
    l2_recall: float = 0.0
    l2_f1: float = 0.0
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    f1_weighted: float = 0.0
    recall_no_injury: float = 0.0
    recall_minor: float = 0.0
    recall_severe: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "zone_id": self.zone_id,
            "n_samples": self.n_samples,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "l1_accuracy": self.l1_accuracy,
            "l1_recall": self.l1_recall,
            "l1_f1": self.l1_f1,
            "l2_accuracy": self.l2_accuracy,
            "l2_recall": self.l2_recall,
            "l2_f1": self.l2_f1,
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_micro": self.f1_micro,
            "f1_weighted": self.f1_weighted,
            "recall_NO_INJURY": self.recall_no_injury,
            "recall_MINOR": self.recall_minor,
            "recall_SEVERE": self.recall_severe,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class ZoneROCData:
    """Container for ROC curve data from a single zone."""

    zone_id: int
    l1_y_true: np.ndarray | None = None
    l1_y_proba: np.ndarray | None = None
    l2_y_true: np.ndarray | None = None  # Only injury cases
    l2_y_proba: np.ndarray | None = None  # Only injury cases


def stratified_sample_severe(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample dataset ensuring class balance for simplified 3-class targets.

    Takes a stratified sample that maintains class proportions while ensuring
    minority classes (SEVERE) have sufficient representation for training.

    Args:
        df: Full DataFrame with MOST_SEVERE_INJURY column.
        sample_size: Target sample size.
        random_state: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with balanced class representation.
    """
    # Use stratified sampling to maintain proportions
    if len(df) <= sample_size:
        return df.reset_index(drop=True)

    # Create simplified class labels for stratification
    def simplify(x):
        if x in ["FATAL", "INCAPACITATING INJURY"]:
            return "SEVERE"
        elif x in ["NONINCAPACITATING INJURY", "REPORTED, NOT EVIDENT"]:
            return "MINOR"
        else:
            return "NO_INJURY"

    strat_labels = df["MOST_SEVERE_INJURY"].fillna("UNKNOWN").map(simplify)

    # Check if we can do stratified sampling
    min_class_count = strat_labels.value_counts().min()
    if min_class_count < 2:
        # Can't stratify, just random sample
        return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # Stratified sample
    from sklearn.model_selection import train_test_split
    _, sample_df = train_test_split(
        df,
        test_size=sample_size / len(df),
        stratify=strat_labels,
        random_state=random_state,
    )
    return sample_df.reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare feature matrix from DataFrame.

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
        # Geographic coordinates (already used for clustering)
        "LATITUDE", "LONGITUDE", "LOCATION",
        # Zone column (not a feature)
        "ZONE",
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
            feature_cols.append(col)

    feature_cols = list(df_features.columns)
    df_features = df_features.fillna(0)

    return df_features, feature_cols


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute binary classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_zone(
    df_zone: pd.DataFrame,
    zone_id: int,
    config: SimplifiedTreeConfig,
    feature_filter: str = "drop-low",
) -> tuple[ZoneResults, SimplifiedTreeClassifier | None, ZoneROCData | None]:
    """Train a simplified classifier for a single zone.

    Args:
        df_zone: DataFrame filtered to this zone.
        zone_id: Zone identifier.
        config: Training configuration.
        feature_filter: Feature filtering mode.

    Returns:
        Tuple of (ZoneResults, trained classifier or None, ROC data or None if skipped).
    """
    n_samples = len(df_zone)

    # Initialize results
    results = ZoneResults(
        zone_id=zone_id,
        n_samples=n_samples,
        n_train=0,
        n_val=0,
        n_test=0,
    )

    # Check minimum samples for training (need at least 10 for any meaningful split)
    min_for_split = 10
    if n_samples < min_for_split:
        results.skipped = True
        results.skip_reason = f"Insufficient samples ({n_samples} < {min_for_split})"
        return results, None, None

    # Engineer features for this zone
    df_zone = engineer_all_features(df_zone, include_interactions=True, include_clusters=False)
    df_zone = add_binary_targets(df_zone)

    # Prepare features
    X_df, feature_cols = prepare_features(df_zone)

    # Apply importance-based filtering
    X_df, feature_cols, _ = filter_by_importance(X_df, feature_cols, mode=feature_filter)
    X = X_df.values

    # Prepare targets
    targets = prepare_simplified_targets(df_zone)

    # Check class distribution - we need at least 2 classes to train
    unique, counts = np.unique(targets.y_simplified, return_counts=True)
    n_classes = len(unique)

    if n_classes < 2:
        results.skipped = True
        results.skip_reason = f"Only 1 class present (need at least 2 for classification)"
        return results, None, None

    # Split data - try stratified first, fall back to random if it fails
    try:
        train_val_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=targets.y_simplified,
        )

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=config.val_size,
            random_state=config.random_state,
            stratify=targets.y_simplified[train_val_idx],
        )
    except ValueError:
        # Stratified split failed - use random split instead
        train_val_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=config.test_size,
            random_state=config.random_state,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=config.val_size,
            random_state=config.random_state,
        )

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    results.n_train = len(train_idx)
    results.n_val = len(val_idx)
    results.n_test = len(test_idx)

    def subset_targets(indices: np.ndarray) -> SimplifiedTargets:
        return SimplifiedTargets(
            y_injury=targets.y_injury[indices],
            y_severe=targets.y_severe[indices],
            y_simplified=targets.y_simplified[indices],
            label_encoder=targets.label_encoder,
        )

    targets_train = subset_targets(train_idx)
    targets_val = subset_targets(val_idx)
    targets_test = subset_targets(test_idx)

    # Create and train classifier
    clf = SimplifiedTreeClassifier(
        l1_model=config.l1_model,
        l2_model=config.l2_model,
        l1_sampling_strategy=config.l1_sampling_strategy,
        l2_sampling_strategy=config.l2_sampling_strategy,
        l2_weight_multiplier=config.l2_weight_multiplier,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        n_jobs=config.n_jobs,
        l1_threshold=config.l1_threshold,
        l2_threshold=config.l2_threshold,
        random_state=config.random_state,
    )
    clf.feature_cols = feature_cols

    try:
        clf.fit(X_train, targets_train)
    except Exception as e:
        results.skipped = True
        results.skip_reason = f"Training failed: {e}"
        return results, None, None

    # Optimize thresholds on validation set
    clf.optimize_thresholds(
        X_val,
        targets_val,
        l1_target_recall=0.7,
        l2_target_recall=0.5,
    )

    # Evaluate on test set
    l1_pred, l2_pred = clf.predict_binary(X_test)
    y_pred = clf.predict(X_test)

    # L1 metrics
    l1_metrics = _binary_metrics(targets_test.y_injury, l1_pred)
    results.l1_accuracy = l1_metrics["accuracy"]
    results.l1_recall = l1_metrics["recall"]
    results.l1_f1 = l1_metrics["f1"]

    # L2 metrics (injury cases only)
    injury_mask = targets_test.y_injury == 1
    if np.sum(injury_mask) > 0:
        l2_metrics = _binary_metrics(
            targets_test.y_severe[injury_mask],
            l2_pred[injury_mask],
        )
        results.l2_accuracy = l2_metrics["accuracy"]
        results.l2_recall = l2_metrics["recall"]
        results.l2_f1 = l2_metrics["f1"]

    # 3-class metrics
    results.accuracy = accuracy_score(targets_test.y_simplified, y_pred)
    results.f1_macro = f1_score(targets_test.y_simplified, y_pred, average="macro", zero_division=0)
    results.f1_micro = f1_score(targets_test.y_simplified, y_pred, average="micro", zero_division=0)
    results.f1_weighted = f1_score(targets_test.y_simplified, y_pred, average="weighted", zero_division=0)

    # Per-class recall
    for i, class_name in enumerate(SIMPLIFIED_CLASS_NAMES):
        class_mask = targets_test.y_simplified == i
        if np.sum(class_mask) > 0:
            recall = np.mean(y_pred[class_mask] == i)
            if class_name == "NO_INJURY":
                results.recall_no_injury = recall
            elif class_name == "MINOR":
                results.recall_minor = recall
            elif class_name == "SEVERE":
                results.recall_severe = recall

    # Collect ROC data (probabilities)
    l1_proba, l2_proba = clf.predict_proba(X_test)
    roc_data = ZoneROCData(
        zone_id=zone_id,
        l1_y_true=targets_test.y_injury,
        l1_y_proba=l1_proba,
        l2_y_true=targets_test.y_severe[injury_mask] if np.sum(injury_mask) > 0 else None,
        l2_y_proba=l2_proba[injury_mask] if np.sum(injury_mask) > 0 else None,
    )

    return results, clf, roc_data


def save_zone_models(
    output_dir: Path,
    clusterer: LocationClusterer,
    zone_classifiers: dict[int, SimplifiedTreeClassifier],
    zone_results: list[ZoneResults],
    feature_cols: list[str],
) -> None:
    """Save all zone models and metadata.

    Args:
        output_dir: Base output directory.
        clusterer: Fitted LocationClusterer.
        zone_classifiers: Dictionary mapping zone_id to trained classifier.
        zone_results: List of ZoneResults for all zones.
        feature_cols: Feature column names (shared across zones).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save clusterer centroids
    centroids_path = output_dir / "centroids.joblib"
    joblib.dump({
        "centroids": clusterer.centroids.numpy() if clusterer.centroids is not None else None,
        "n_clusters": clusterer.n_clusters,
        "lat_col": clusterer.lat_col,
        "lon_col": clusterer.lon_col,
    }, centroids_path)
    logger.info(f"Saved clusterer to {centroids_path}")

    # Save global metadata
    metadata = {
        "n_zones": len(zone_classifiers),
        "zone_ids": list(zone_classifiers.keys()),
        "feature_cols": feature_cols,
        "total_samples": sum(r.n_samples for r in zone_results),
        "skipped_zones": [r.zone_id for r in zone_results if r.skipped],
    }
    joblib.dump(metadata, output_dir / "metadata.joblib")

    # Save each zone's models
    for zone_id, clf in zone_classifiers.items():
        zone_dir = output_dir / f"zone_{zone_id:02d}"
        save_simplified_model(clf, str(zone_dir))
        logger.info(f"Saved zone {zone_id} model to {zone_dir}")


def print_zone_summary(
    zone_results: list[ZoneResults],
    output_dir: Path,
) -> None:
    """Print and save aggregate summary of zone results.

    Args:
        zone_results: List of ZoneResults for all zones.
        output_dir: Directory to save summary CSV.
    """
    # Filter to non-skipped zones
    trained_zones = [r for r in zone_results if not r.skipped]
    skipped_zones = [r for r in zone_results if r.skipped]

    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)

    print(f"\nZones trained: {len(trained_zones)} / {len(zone_results)}")
    if skipped_zones:
        print(f"Zones skipped: {len(skipped_zones)}")
        for r in skipped_zones:
            print(f"  Zone {r.zone_id}: {r.skip_reason}")

    total_samples = sum(r.n_samples for r in trained_zones)
    print(f"\nTotal samples in trained zones: {total_samples:,}")

    if not trained_zones:
        print("\nNo zones were successfully trained!")
        return

    # Summary table
    print("\n" + "-" * 85)
    print(f"{'Zone':>6} | {'Samples':>8} | {'Accuracy':>8} | {'Macro F1':>8} | {'Micro F1':>8} | {'SEVERE Recall':>13}")
    print("-" * 85)

    for r in sorted(trained_zones, key=lambda x: x.zone_id):
        print(
            f"{r.zone_id:>6} | {r.n_samples:>8,} | {r.accuracy:>8.4f} | "
            f"{r.f1_macro:>8.4f} | {r.f1_micro:>8.4f} | {r.recall_severe:>13.4f}"
        )

    # Best and worst zones
    best_zone = max(trained_zones, key=lambda x: x.f1_macro)
    worst_zone = min(trained_zones, key=lambda x: x.f1_macro)

    print("-" * 85)
    print(
        f"{'BEST':>6} | {best_zone.n_samples:>8,} | {best_zone.accuracy:>8.4f} | "
        f"{best_zone.f1_macro:>8.4f} | {best_zone.f1_micro:>8.4f} | {best_zone.recall_severe:>13.4f}  (Zone {best_zone.zone_id})"
    )
    print(
        f"{'WORST':>6} | {worst_zone.n_samples:>8,} | {worst_zone.accuracy:>8.4f} | "
        f"{worst_zone.f1_macro:>8.4f} | {worst_zone.f1_micro:>8.4f} | {worst_zone.recall_severe:>13.4f}  (Zone {worst_zone.zone_id})"
    )

    # Weighted averages
    total_weight = sum(r.n_samples for r in trained_zones)
    avg_accuracy = sum(r.accuracy * r.n_samples for r in trained_zones) / total_weight
    avg_f1_macro = sum(r.f1_macro * r.n_samples for r in trained_zones) / total_weight
    avg_f1_micro = sum(r.f1_micro * r.n_samples for r in trained_zones) / total_weight
    avg_severe_recall = sum(r.recall_severe * r.n_samples for r in trained_zones) / total_weight

    print("-" * 85)
    print(
        f"{'AVG':>6} | {total_samples // len(trained_zones):>8,} | {avg_accuracy:>8.4f} | "
        f"{avg_f1_macro:>8.4f} | {avg_f1_micro:>8.4f} | {avg_severe_recall:>13.4f}  (weighted)"
    )
    print("-" * 85)

    # Save to CSV
    summary_df = pd.DataFrame([r.to_dict() for r in zone_results])
    csv_path = output_dir / "zone_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")


def run_zoned_pipeline(
    n_clusters: int = 25,
    sample_size: int | None = None,
    min_zone_size: int = 100,
    feature_filter: str = "drop-low",
) -> list[ZoneResults]:
    """Run the zone-based simplified classification pipeline.

    Args:
        n_clusters: Number of geographic zones to create via K-means.
        sample_size: Optional limit on total dataset size.
        min_zone_size: Minimum samples required per zone.
        feature_filter: Feature filtering mode.

    Returns:
        List of ZoneResults for all zones.
    """
    print("=" * 70)
    print(f"ZONE-BASED SIMPLIFIED CLASSIFICATION ({n_clusters} zones)")
    print("=" * 70)

    config = SimplifiedTreeConfig(sample_size=sample_size)
    logger.info(f"Configuration: {config}")

    # Load and merge data
    logger.info("\n[1/5] Loading and merging data...")
    df = triple_merge()
    logger.info(f"Merged dataset shape: {df.shape}")

    # Sample if specified
    if sample_size is not None and len(df) > sample_size:
        logger.info(f"Sampling {sample_size:,} rows...")
        df = stratified_sample_severe(df, sample_size, config.random_state)

    # Fit clusterer and assign zones
    logger.info(f"\n[2/5] Clustering into {n_clusters} geographic zones...")

    clusterer = LocationClusterer(
        n_clusters=n_clusters,
        n_iterations=100,
        random_state=config.random_state,
    )

    # Check for valid coordinates
    valid_coords = df[["LATITUDE", "LONGITUDE"]].notna().all(axis=1)
    n_invalid = (~valid_coords).sum()
    if n_invalid > 0:
        logger.warning(f"Dropping {n_invalid:,} rows with missing coordinates")
        df = df[valid_coords].reset_index(drop=True)

    clusterer.fit(df)
    df = clusterer.add_cluster_column(df, column_name="ZONE")

    # Show zone distribution
    zone_counts = df["ZONE"].value_counts().sort_index()
    logger.info(f"Zone distribution (min={zone_counts.min()}, max={zone_counts.max()}, median={zone_counts.median():.0f}):")
    for zone_id, count in zone_counts.items():
        status = "OK" if count >= min_zone_size else "SMALL"
        logger.info(f"  Zone {zone_id:>2}: {count:>6,} samples [{status}]")

    # Train models per zone
    logger.info(f"\n[3/5] Training models per zone (min_size={min_zone_size})...")

    zone_results: list[ZoneResults] = []
    zone_classifiers: dict[int, SimplifiedTreeClassifier] = {}
    zone_roc_data: list[ZoneROCData] = []
    shared_feature_cols: list[str] = []

    for zone_id in sorted(df["ZONE"].unique()):
        df_zone = df[df["ZONE"] == zone_id].copy()
        n_samples = len(df_zone)

        print(f"\n[Zone {zone_id}] ", end="")

        if n_samples < min_zone_size:
            results = ZoneResults(
                zone_id=int(zone_id),
                n_samples=n_samples,
                n_train=0,
                n_val=0,
                n_test=0,
                skipped=True,
                skip_reason=f"Below min_zone_size ({n_samples} < {min_zone_size})",
            )
            print(f"SKIPPED - {results.skip_reason}")
            zone_results.append(results)
            continue

        print(f"Training on {n_samples:,} samples... ", end="", flush=True)

        results, clf, roc_data = train_zone(df_zone, int(zone_id), config, feature_filter)
        zone_results.append(results)

        if results.skipped:
            print(f"SKIPPED - {results.skip_reason}")
        else:
            zone_classifiers[int(zone_id)] = clf
            if roc_data is not None:
                zone_roc_data.append(roc_data)
            if clf is not None and hasattr(clf, "feature_cols"):
                shared_feature_cols = clf.feature_cols
            print(
                f"OK | Accuracy: {results.accuracy:.4f} | "
                f"SEVERE Recall: {results.recall_severe:.4f}"
            )

    # Save models
    logger.info("\n[4/5] Saving models...")
    output_dir = PROJECT_ROOT / "models" / "trained" / "simplified_zones"

    if zone_classifiers:
        save_zone_models(
            output_dir,
            clusterer,
            zone_classifiers,
            zone_results,
            shared_feature_cols,
        )
        print(f"\nModels saved to: {output_dir}")
    else:
        print("\nNo models to save (all zones skipped)")

    # Print summary
    logger.info("\n[5/5] Generating summary...")
    plots_dir = PROJECT_ROOT / "models" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print_zone_summary(zone_results, plots_dir)

    # Generate aggregate ROC curves
    if zone_roc_data:
        logger.info("\nGenerating ROC curves...")
        _plot_aggregate_roc(zone_roc_data, plots_dir)

    return zone_results


def _plot_aggregate_roc(
    zone_roc_data: list[ZoneROCData],
    output_dir: Path,
) -> None:
    """Aggregate ROC data from all zones and plot combined ROC curves.

    Args:
        zone_roc_data: List of ZoneROCData from successfully trained zones.
        output_dir: Directory to save the ROC plot.
    """
    # Aggregate L1 and L2 data across zones
    l1_y_true_all = []
    l1_y_proba_all = []
    l2_y_true_all = []
    l2_y_proba_all = []

    for roc_data in zone_roc_data:
        if roc_data.l1_y_true is not None and roc_data.l1_y_proba is not None:
            l1_y_true_all.append(roc_data.l1_y_true)
            l1_y_proba_all.append(roc_data.l1_y_proba)

        if roc_data.l2_y_true is not None and roc_data.l2_y_proba is not None:
            l2_y_true_all.append(roc_data.l2_y_true)
            l2_y_proba_all.append(roc_data.l2_y_proba)

    # Build level dictionaries for plotting
    y_true_levels = {}
    y_proba_levels = {}

    if l1_y_true_all:
        y_true_levels["L1 (INJURY vs NO_INJURY)"] = np.concatenate(l1_y_true_all)
        y_proba_levels["L1 (INJURY vs NO_INJURY)"] = np.concatenate(l1_y_proba_all)

    if l2_y_true_all:
        y_true_levels["L2 (SEVERE vs MINOR)"] = np.concatenate(l2_y_true_all)
        y_proba_levels["L2 (SEVERE vs MINOR)"] = np.concatenate(l2_y_proba_all)

    if not y_true_levels:
        logger.warning("No valid ROC data to plot")
        return

    # Plot
    save_path = output_dir / "roc_curves.png"
    auc_scores = plot_roc_curves(
        y_true_levels=y_true_levels,
        y_proba_levels=y_proba_levels,
        save_path=save_path,
        title="ROC Curves - Zone-Based Simplified Classifier (Aggregated)",
    )

    # Log summary
    print(f"\nROC curves saved to: {save_path}")
    for level_name, auc in auc_scores.items():
        print(f"  {level_name}: AUC = {auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run zone-based simplified 3-class crash severity classification"
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=25,
        help="Number of geographic zones to create via K-means (default: 25)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for faster experimentation (default: use all data)",
    )
    parser.add_argument(
        "--min-zone-size",
        type=int,
        default=0,
        help="Minimum samples required per zone (default: 0, no minimum)",
    )
    parser.add_argument(
        "--feature-filter",
        type=str,
        choices=["none", "drop-low", "drop-review"],
        default="drop-low",
        help="Feature filtering mode (default: drop-low)",
    )

    args = parser.parse_args()

    results = run_zoned_pipeline(
        n_clusters=args.clusters,
        sample_size=args.sample,
        min_zone_size=args.min_zone_size,
        feature_filter=args.feature_filter,
    )

    # Final results
    trained = [r for r in results if not r.skipped]
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Zones trained: {len(trained)}")
    if trained:
        total = sum(r.n_samples for r in trained)
        avg_acc = sum(r.accuracy * r.n_samples for r in trained) / total
        avg_f1_macro = sum(r.f1_macro * r.n_samples for r in trained) / total
        avg_f1_micro = sum(r.f1_micro * r.n_samples for r in trained) / total
        avg_severe = sum(r.recall_severe * r.n_samples for r in trained) / total
        print(f"Weighted Accuracy: {avg_acc:.4f}")
        print(f"Weighted Macro F1: {avg_f1_macro:.4f}")
        print(f"Weighted Micro F1: {avg_f1_micro:.4f}")
        print(f"Weighted SEVERE Recall: {avg_severe:.4f}")
