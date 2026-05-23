"""Compare all trained models on the same test set.

Generates a comparison table and confusion matrices for all models.

Usage:
    python training/compare_all_models.py
    python training/compare_all_models.py --sample 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.triple_merge import triple_merge, CRASH_ONLY
from training.baselines import (
    create_imbalance_baselines,
    print_dual_baseline_comparison,
)
from training.feature_selection import filter_by_importance
from utils.logging_config import setup_logging
from utils.csv_filename_generator import generate_csv_filename

logger = setup_logging(__name__)

# Paths
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
OUTPUT_DIR = PROJECT_ROOT / "models" / "plots"

# Class names
CLASS_NAMES = ["NO_INJURY", "MINOR", "SEVERE"]

# Features (same as baseline)
CATEGORICAL_FEATURES = [
    "FIRST_CRASH_TYPE",
    "DAMAGE",
    "PRIM_CONTRIBUTORY_CAUSE",
    "TRAFFIC_CONTROL_DEVICE",
    "DEVICE_CONDITION",
    "WEATHER_CONDITION",
    "LIGHTING_CONDITION",
    "TRAFFICWAY_TYPE",
    "ROADWAY_SURFACE_COND",
    "ROAD_DEFECT",
    "ALIGNMENT",
]

NUMERICAL_FEATURES = [
    "POSTED_SPEED_LIMIT",
    "NUM_UNITS",
    "CRASH_HOUR",
    "CRASH_DAY_OF_WEEK",
    "CRASH_MONTH",
]

INJURY_MAPPING = {
    "NO INDICATION OF INJURY": "NO_INJURY",
    "REPORTED, NOT EVIDENT": "MINOR",
    "NONINCAPACITATING INJURY": "MINOR",
    "INCAPACITATING INJURY": "SEVERE",
    "FATAL": "SEVERE",
}


def prepare_test_data(
    sample_size: int | None = None,
    feature_filter: str = "drop-low",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load and prepare test data using the same preprocessing as training.
    
    Args:
        sample_size: Optional sample size limit.
        feature_filter: Feature filtering mode ('none', 'drop-low', 'drop-review').
                       Must match the filter used during training for valid comparison.
    
    Returns:
        X_test, y_test, y_train (for baseline fitting), feature_names
    """
    logger.info("Loading crash data...")
    df = triple_merge(config=CRASH_ONLY, verbose=False)
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} rows")
    
    # Extract time features
    crash_datetime = pd.to_datetime(df["CRASH_DATE"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df["CRASH_HOUR"] = crash_datetime.dt.hour.fillna(12).astype(int)
    df["CRASH_DAY_OF_WEEK"] = crash_datetime.dt.dayofweek.fillna(0).astype(int) + 1
    df["CRASH_MONTH"] = crash_datetime.dt.month.fillna(6).astype(int)
    
    # Map severity
    df["SEVERITY_3CLASS"] = df["MOST_SEVERE_INJURY"].map(INJURY_MAPPING)
    df = df.dropna(subset=["SEVERITY_3CLASS"])
    
    # Select features
    all_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    columns_needed = all_features + ["SEVERITY_3CLASS"]
    df = df[[c for c in columns_needed if c in df.columns]].copy()
    
    # Fill missing values
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")
    
    # We'll use the encoders from the simple_rf model for consistency
    encoders_path = MODELS_DIR / "simple_rf" / "encoders.joblib"
    if encoders_path.exists():
        encoders = joblib.load(encoders_path)
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and col in encoders:
                le = encoders[col]
                # Handle unseen categories
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    else:
        # Fallback: encode from scratch
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    
    # Encode target
    target_encoder = LabelEncoder()
    target_encoder.classes_ = np.array(CLASS_NAMES)
    df["SEVERITY_3CLASS"] = target_encoder.transform(df["SEVERITY_3CLASS"])
    
    # Prepare X and y
    feature_cols = [c for c in CATEGORICAL_FEATURES + NUMERICAL_FEATURES if c in df.columns]
    
    # Apply importance-based feature filtering if enabled
    if feature_filter != "none":
        logger.warning(
            f"Feature filter '{feature_filter}' is enabled. "
            "Ensure this matches the filter used during training for valid comparison!"
        )
        X_df = df[feature_cols].copy()
        X_df, feature_cols, filter_result = filter_by_importance(
            X_df, feature_cols, mode=feature_filter
        )
        logger.info(f"Feature filter '{feature_filter}': {filter_result.n_original} -> {filter_result.n_kept} features")
        X = X_df.values
    else:
        X = df[feature_cols].values
    
    y = df["SEVERITY_3CLASS"].values
    
    # Use same test split as training (20%)
    _, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_test, y_test, y_train, feature_cols


def load_model(model_dir: Path) -> Any | None:
    """Load a trained model from a directory."""
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Evaluate a model and return metrics.
    
    Returns dict with metrics and optionally y_proba for ROC curves.
    """
    y_pred = model.predict(X_test)
    if hasattr(y_pred, "flatten"):
        y_pred = y_pred.flatten()
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_micro": f1_score(y_test, y_pred, average="micro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "is_baseline": False,
    }
    
    # Per-class recall (important for minority classes)
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_recall = recall_score(y_test[class_mask], y_pred[class_mask], average="micro")
            metrics[f"recall_{class_name}"] = (y_pred[class_mask] == i).sum() / class_mask.sum()
    
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    
    # Capture prediction probabilities for ROC curves if available
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            metrics["y_proba"] = y_proba
            
            # Compute ROC AUC (One-vs-Rest macro average)
            try:
                metrics["roc_auc_ovr"] = roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                )
            except Exception:
                pass  # Skip if ROC AUC computation fails
        except Exception as e:
            logger.debug(f"Could not get predict_proba for {model_name}: {e}")
    
    return metrics


def find_all_models() -> list[tuple[str, Path]]:
    """Find all trained models in the models directory."""
    models = []
    
    # Simple RF (baseline)
    simple_rf_dir = MODELS_DIR / "simple_rf"
    if (simple_rf_dir / "model.joblib").exists():
        models.append(("Random Forest (baseline)", simple_rf_dir))
    
    # Tuned RF
    tuned_rf_dir = MODELS_DIR / "tuned_boosting" / "rf"
    if (tuned_rf_dir / "model.joblib").exists():
        models.append(("Random Forest (tuned)", tuned_rf_dir))
    
    # Tuned boosting models
    tuned_dir = MODELS_DIR / "tuned_boosting"
    if tuned_dir.exists():
        for model_type in ["lgbm", "xgb", "catboost"]:
            model_dir = tuned_dir / model_type
            if (model_dir / "model.joblib").exists():
                name_map = {"lgbm": "LightGBM (tuned)", "xgb": "XGBoost (tuned)", "catboost": "CatBoost (tuned)"}
                models.append((name_map.get(model_type, model_type), model_dir))
    
    # TabNet
    tabnet_dir = MODELS_DIR / "tabnet"
    if (tabnet_dir / "model.joblib").exists():
        models.append(("TabNet", tabnet_dir))
    
    return models


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    if not results:
        logger.warning("No results to display")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Select columns for display
    display_cols = ["model", "accuracy", "f1_macro", "f1_weighted", "recall_SEVERE"]
    display_cols = [c for c in display_cols if c in df.columns]
    
    df_display = df[display_cols].copy()
    
    # Format percentages
    for col in df_display.columns:
        if col != "model" and df_display[col].dtype in [float, np.float64]:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
    
    # Sort by F1 macro
    df_display = df_display.sort_values("f1_macro", ascending=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)
    logger.info(f"\n{df_display.to_string(index=False)}")
    
    # Print best model
    best_model = df.loc[df["f1_macro"].idxmax(), "model"]
    best_f1 = df["f1_macro"].max()
    logger.info(f"\nBest model: {best_model} (F1 Macro: {best_f1:.4f})")


def save_comparison(results: list[dict], baseline_results: list[dict] | None = None) -> None:
    """Save comparison results to JSON and CSV with timestamps, including baselines."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Combine model and baseline results, stripping non-serializable fields
    all_results = []
    for r in results:
        clean_result = {k: v for k, v in r.items() if k != "y_proba"}
        all_results.append(clean_result)
    if baseline_results:
        for r in baseline_results:
            clean_result = {k: v for k, v in r.items() if k != "y_proba"}
            all_results.append(clean_result)
    
    # Generate timestamped filenames
    json_filename = generate_csv_filename("model_comparison", "all_models", extension=".json")
    csv_filename = generate_csv_filename("model_comparison", "all_models")
    
    # Save full results as JSON
    with open(OUTPUT_DIR / json_filename, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary as CSV
    df = pd.DataFrame(all_results)
    summary_cols = ["model", "is_baseline", "accuracy", "f1_macro", "f1_micro", "f1_weighted", "precision_macro", "recall_macro", "roc_auc_ovr"]
    for cls in CLASS_NAMES:
        col = f"recall_{cls}"
        if col in df.columns:
            summary_cols.append(col)
    
    # Only include columns that exist
    summary_cols = [c for c in summary_cols if c in df.columns]
    
    df[summary_cols].to_csv(OUTPUT_DIR / csv_filename, index=False)
    
    logger.info(f"\nSaved comparison to {OUTPUT_DIR / csv_filename}")


def _downsample_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    max_points: int = 100,
) -> tuple[list[float], list[float]]:
    """Downsample ROC curve points for efficient JSON export."""
    n_points = len(fpr)
    if n_points <= max_points:
        return fpr.tolist(), tpr.tolist()
    indices = np.linspace(0, n_points - 1, max_points, dtype=int)
    indices = np.unique(indices)
    return fpr[indices].tolist(), tpr[indices].tolist()


def generate_roc_comparison(
    results: list[dict],
    y_test: np.ndarray,
    output_dir: Path | None = None,
) -> None:
    """Generate unified ROC comparison plot and JSON data for all models.
    
    Creates:
    - roc_curves_all_models.png: Matplotlib plot with all model ROC curves
    - roc_data_all_models.json: JSON export for frontend visualization
    
    Args:
        results: List of evaluation results from evaluate_model(), must include y_proba.
        y_test: True labels for the test set.
        output_dir: Output directory (defaults to OUTPUT_DIR).
    """
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to models that have probability outputs
    models_with_proba = [r for r in results if "y_proba" in r and r["y_proba"] is not None]
    
    if not models_with_proba:
        logger.warning("No models with predict_proba available for ROC curves")
        return
    
    logger.info(f"Generating ROC curves for {len(models_with_proba)} models...")
    
    # Color palette for models
    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]
    
    # --- 1. Generate One-vs-Rest ROC curves (macro-averaged) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    roc_json_data = {
        "model_name": "all_models_comparison",
        "generated_at": datetime.now().isoformat(),
        "curves": [],
    }
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        ax = axes[class_idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.50)")
        
        for model_idx, result in enumerate(models_with_proba):
            model_name = result["model"]
            y_proba = result["y_proba"]
            
            # Binary labels for this class (one-vs-rest)
            y_true_binary = (y_test == class_idx).astype(int)
            y_score = y_proba[:, class_idx]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            auc_score = roc_auc_score(y_true_binary, y_score)
            
            color = colors[model_idx % len(colors)]
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC={auc_score:.3f})")
            
            # Store for JSON export
            fpr_down, tpr_down = _downsample_curve(fpr, tpr)
            roc_json_data["curves"].append({
                "model": model_name,
                "class": class_name,
                "auc": round(auc_score, 4),
                "color": color,
                "fpr": fpr_down,
                "tpr": tpr_down,
                "n_samples": len(y_test),
                "n_positive": int(y_true_binary.sum()),
            })
        
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve: {class_name}")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("ROC Curves - All Models Comparison (One-vs-Rest)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    roc_plot_path = output_dir / "roc_curves_all_models.png"
    plt.savefig(roc_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC comparison plot saved to {roc_plot_path}")
    
    # --- 2. Generate single macro-averaged ROC plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.50)")
    
    for model_idx, result in enumerate(models_with_proba):
        model_name = result["model"]
        y_proba = result["y_proba"]
        
        # Compute macro-averaged ROC (average of per-class curves)
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        
        for class_idx in range(len(CLASS_NAMES)):
            y_true_binary = (y_test == class_idx).astype(int)
            y_score = y_proba[:, class_idx]
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        mean_tpr /= len(CLASS_NAMES)
        macro_auc = result.get("roc_auc_ovr", auc(all_fpr, mean_tpr))
        
        color = colors[model_idx % len(colors)]
        ax.plot(all_fpr, mean_tpr, color=color, lw=2.5, label=f"{model_name} (AUC={macro_auc:.3f})")
        
        # Add macro-averaged curve to JSON
        fpr_down, tpr_down = _downsample_curve(all_fpr, mean_tpr)
        roc_json_data["curves"].append({
            "model": model_name,
            "class": "MACRO_AVERAGE",
            "auc": round(float(macro_auc), 4),
            "color": color,
            "fpr": fpr_down,
            "tpr": tpr_down,
            "n_samples": len(y_test),
        })
    
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Macro-Averaged (All Models)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    macro_plot_path = output_dir / "roc_curves_macro_averaged.png"
    plt.savefig(macro_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Macro-averaged ROC plot saved to {macro_plot_path}")
    
    # --- 3. Save JSON data ---
    json_path = output_dir / "roc_data_all_models.json"
    with open(json_path, "w") as f:
        json.dump(roc_json_data, f, indent=2)
    logger.info(f"ROC data saved to {json_path}")


def main(sample_size: int | None = None, feature_filter: str = "drop-low") -> None:
    """Run comparison of all trained models.
    
    Args:
        sample_size: Optional sample size limit.
        feature_filter: Feature filtering mode. Must match training filter!
    """
    logger.info("Starting model comparison...")
    if feature_filter != "none":
        logger.info(f"Feature filter: {feature_filter}")
    
    # Prepare test data
    X_test, y_test, y_train, feature_names = prepare_test_data(sample_size, feature_filter)
    
    # Find all models
    models = find_all_models()
    
    if not models:
        logger.warning("No trained models found! Run training scripts first.")
        return
    
    logger.info(f"Found {len(models)} trained models")
    
    # Evaluate each model
    results = []
    for model_name, model_dir in models:
        logger.info(f"\nEvaluating {model_name}...")
        
        model = load_model(model_dir)
        if model is None:
            logger.warning(f"Could not load model from {model_dir}")
            continue
        
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        if "recall_SEVERE" in metrics:
            logger.info(f"  SEVERE Recall: {metrics['recall_SEVERE']:.4f}")
    
    # Print comparison table
    print_comparison_table(results)
    
    # Baseline comparison (coin flip baselines for imbalanced data)
    baseline_rows = []
    if results:
        logger.info(f"\n{'='*60}")
        logger.info("BASELINE COMPARISON")
        logger.info(f"{'='*60}")
        
        baseline = create_imbalance_baselines()
        baseline.fit(y_train)
        baseline_eval = baseline.evaluate(y_test, class_names=CLASS_NAMES)
        
        # Calculate class priors for biased baseline ROC curves
        n_classes = len(CLASS_NAMES)
        class_counts = np.bincount(y_train, minlength=n_classes)
        class_priors = class_counts / class_counts.sum()
        
        # Random generator for baseline probabilities
        rng = np.random.RandomState(42)
        n_test = len(y_test)
        
        for baseline_name, baseline_metrics in baseline_eval.items():
            metrics = baseline_metrics.metrics
            baseline_row = {
                "model": baseline_name,
                "is_baseline": True,
                "accuracy": metrics.get("accuracy", 0),
                "f1_macro": metrics.get("f1_macro", 0),
                "f1_micro": metrics.get("f1_micro", 0),
                "f1_weighted": metrics.get("f1_weighted", 0),
                "precision_macro": metrics.get("precision_macro", 0),
                "recall_macro": metrics.get("recall_macro", 0),
            }
            # Add per-class recall if available
            for cls in CLASS_NAMES:
                col = f"recall_{cls}"
                if col in metrics:
                    baseline_row[col] = metrics[col]
            
            # Generate random probabilities for ROC curves
            # DummyClassifier.predict_proba returns CONSTANT probs, which is wrong for ROC
            # We need varying random probabilities per sample
            if baseline_name == "coin_flip":
                # Uniform random: each class gets random probability, then normalize
                raw_proba = rng.uniform(0, 1, size=(n_test, n_classes))
                baseline_row["y_proba"] = raw_proba / raw_proba.sum(axis=1, keepdims=True)
            else:  # biased_coin_flip
                # Biased random: Dirichlet distribution centered on class priors
                # concentration = priors * scale gives random probs centered around priors
                baseline_row["y_proba"] = rng.dirichlet(class_priors * 10 + 0.1, size=n_test)
            
            baseline_rows.append(baseline_row)
        
        # Compare best model against both baselines
        best_model = max(results, key=lambda x: x["f1_macro"])
        best_model_name = best_model["model"]  # Fixed: was "model_name"
        best_model_metrics = {
            "accuracy": best_model["accuracy"],
            "f1_macro": best_model["f1_macro"],
            "f1_weighted": best_model.get("f1_weighted", best_model["f1_macro"]),
        }
        
        print_dual_baseline_comparison(
            model_metrics=best_model_metrics,
            baseline_results=baseline_eval,
            model_name=best_model_name,
            primary_metric="f1_macro",
        )
    
    # Generate comparison plots
    if results:
        logger.info("\nGenerating comparison plots...")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
            # Combine models and baselines for plotting
            all_results_for_plot = results.copy()
            all_results_for_plot.extend(baseline_rows)
            
            df = pd.DataFrame(all_results_for_plot)
            
            # 1. F1 Macro comparison bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            df_sorted = df.sort_values("f1_macro", ascending=True)
            
            colors = ["#94a3b8" if is_base else "#3b82f6" for is_base in df_sorted["is_baseline"]]
            bars = ax.barh(range(len(df_sorted)), df_sorted["f1_macro"], color=colors, edgecolor="black", alpha=0.8)
            
            for bar, val in zip(bars, df_sorted["f1_macro"]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")
            
            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels(df_sorted["model"])
            ax.set_xlabel("F1 Macro Score")
            ax.set_title("Model Comparison - F1 Macro Score")
            ax.set_xlim(0, df_sorted["f1_macro"].max() * 1.15)
            plt.tight_layout()
            
            f1_path = OUTPUT_DIR / "comparison_f1_macro.png"
            plt.savefig(f1_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"F1 comparison plot saved to {f1_path}")
            
            # 2. Per-class recall comparison (models only)
            recall_cols = [f"recall_{cls}" for cls in CLASS_NAMES if f"recall_{cls}" in df.columns]
            if recall_cols:
                models_df = df[df["is_baseline"] == False].copy()
                
                if len(models_df) > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    x = np.arange(len(models_df))
                    width = 0.25
                    
                    for i, col in enumerate(recall_cols):
                        if col in models_df.columns:
                            cls_name = col.replace("recall_", "")
                            offset = (i - len(recall_cols) / 2 + 0.5) * width
                            ax.bar(x + offset, models_df[col].fillna(0), width, label=cls_name, alpha=0.8)
                    
                    ax.set_xticks(x)
                    ax.set_xticklabels(models_df["model"], rotation=45, ha="right")
                    ax.set_ylabel("Recall")
                    ax.set_title("Per-Class Recall by Model")
                    ax.legend(title="Class")
                    ax.set_ylim(0, 1.0)
                    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    
                    recall_path = OUTPUT_DIR / "comparison_recall_by_class.png"
                    plt.savefig(recall_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"Recall comparison plot saved to {recall_path}")
            
            # 3. Metrics heatmap (models only)
            models_df = df[df["is_baseline"] == False].copy()
            metric_cols = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
            metric_cols = [c for c in metric_cols if c in models_df.columns]
            
            if len(models_df) > 0 and metric_cols:
                fig, ax = plt.subplots(figsize=(10, max(6, len(models_df) * 0.5)))
                
                heatmap_data = models_df.set_index("model")[metric_cols]
                sns.heatmap(
                    heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Score"},
                )
                ax.set_title("Metrics Heatmap by Model")
                plt.tight_layout()
                
                heatmap_path = OUTPUT_DIR / "comparison_metrics_heatmap.png"
                plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Metrics heatmap saved to {heatmap_path}")
            
            # 4. Model improvement over baselines
            if baseline_rows:
                best_baseline_f1 = max(b["f1_macro"] for b in baseline_rows)
                models_df = df[df["is_baseline"] == False].copy()
                
                if len(models_df) > 0:
                    models_df["improvement"] = ((models_df["f1_macro"] - best_baseline_f1) / best_baseline_f1 * 100)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    models_sorted = models_df.sort_values("improvement", ascending=True)
                    
                    colors = ["#22c55e" if imp > 0 else "#ef4444" for imp in models_sorted["improvement"]]
                    ax.barh(range(len(models_sorted)), models_sorted["improvement"], color=colors, edgecolor="black", alpha=0.8)
                    
                    for i, (_, row) in enumerate(models_sorted.iterrows()):
                        ax.text(
                            row["improvement"] + (2 if row["improvement"] >= 0 else -2),
                            i,
                            f"{row['improvement']:+.1f}%",
                            va="center",
                            ha="left" if row["improvement"] >= 0 else "right",
                        )
                    
                    ax.set_yticks(range(len(models_sorted)))
                    ax.set_yticklabels(models_sorted["model"])
                    ax.set_xlabel("% Improvement over Best Baseline")
                    ax.set_title(f"Model Improvement over Best Baseline (F1 = {best_baseline_f1:.3f})")
                    ax.axvline(x=0, color="black", linewidth=1)
                    plt.tight_layout()
                    
                    improvement_path = OUTPUT_DIR / "comparison_improvement.png"
                    plt.savefig(improvement_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"Improvement plot saved to {improvement_path}")
            
        except ImportError as e:
            logger.warning(f"Could not generate comparison plots (missing dependencies): {e}")
        except Exception as e:
            logger.warning(f"Error generating comparison plots: {e}")
    
    # Generate ROC comparison plots (include baselines too)
    if results:
        try:
            all_results_for_roc = results + baseline_rows
            generate_roc_comparison(all_results_for_roc, y_test)
        except Exception as e:
            logger.warning(f"Error generating ROC comparison: {e}")
    
    # Save comparison with baselines included
    save_comparison(results, baseline_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all trained models")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for faster iteration",
    )
    parser.add_argument(
        "--feature-filter",
        type=str,
        choices=["none", "drop-low", "drop-review"],
        default="drop-low",
        help="Feature filtering mode. MUST match the filter used during training! Default: drop-low",
    )
    
    args = parser.parse_args()
    main(sample_size=args.sample, feature_filter=args.feature_filter)
