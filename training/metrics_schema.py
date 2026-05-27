"""
Unified metrics schema for training results and baseline comparisons.

Defines standard column names and structures for CSV exports across all training pipelines.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from utils.csv_filename_generator import generate_csv_filename


# Standard columns for classification metrics CSV
CLASSIFICATION_METRICS_COLUMNS = [
    "model",
    "dataset",
    "timestamp",
    "is_baseline",
    "accuracy",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "recall_class_0",
    "recall_class_1",
    "recall_class_2",
    "recall_class_3",
    "recall_class_4",
    "auc_injury",
    "auc_severe",
    "auc_macro",
    "confusion_matrix_json",
    "sample_size",
    "n_classes",
    "class_distribution_json",
    "duration_seconds",
    "hyperparameters_json",
    "notes",
]

# Standard columns for regression metrics CSV
REGRESSION_METRICS_COLUMNS = [
    "model",
    "dataset",
    "timestamp",
    "is_baseline",
    "mae",
    "rmse",
    "mape",
    "smape",
    "r2",
    "mae_per_zone_json",
    "sample_size",
    "n_zones",
    "duration_seconds",
    "hyperparameters_json",
    "notes",
]

# Standard columns for model vs baseline comparison CSV
BASELINE_COMPARISON_COLUMNS = [
    "model",
    "baseline",
    "metric",
    "model_value",
    "baseline_value",
    "improvement_pct",
    "improvement_absolute",
    "timestamp",
    "dataset",
]

# Standard columns for training history CSV
TRAINING_HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_metric",
    "val_metric",
    "metric_name",
    "learning_rate",
    "timestamp",
]


@dataclass
class ClassificationMetrics:
    """Container for classification metrics with export functionality."""

    model: str
    dataset: str = "default"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_baseline: bool = False

    # Core metrics
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    f1_weighted: float = 0.0
    precision_macro: float = 0.0
    precision_weighted: float = 0.0
    recall_macro: float = 0.0
    recall_weighted: float = 0.0

    # Per-class recall (up to 5 classes for hierarchical)
    recall_class_0: Optional[float] = None
    recall_class_1: Optional[float] = None
    recall_class_2: Optional[float] = None
    recall_class_3: Optional[float] = None
    recall_class_4: Optional[float] = None

    # AUC metrics
    auc_injury: Optional[float] = None
    auc_severe: Optional[float] = None
    auc_macro: Optional[float] = None

    # Additional info
    confusion_matrix: Optional[List[List[int]]] = None
    sample_size: int = 0
    n_classes: int = 0
    class_distribution: Optional[Dict[str, int]] = None
    duration_seconds: float = 0.0
    hyperparameters: Optional[Dict[str, Any]] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        d = asdict(self)
        # Convert nested structures to JSON strings
        if self.confusion_matrix is not None:
            d["confusion_matrix_json"] = json.dumps(self.confusion_matrix)
        else:
            d["confusion_matrix_json"] = None
        del d["confusion_matrix"]

        if self.class_distribution is not None:
            d["class_distribution_json"] = json.dumps(self.class_distribution)
        else:
            d["class_distribution_json"] = None
        del d["class_distribution"]

        if self.hyperparameters is not None:
            d["hyperparameters_json"] = json.dumps(self.hyperparameters)
        else:
            d["hyperparameters_json"] = None
        del d["hyperparameters"]

        return d

    @classmethod
    def from_sklearn_report(
        cls,
        model: str,
        y_true,
        y_pred,
        dataset: str = "default",
        duration_seconds: float = 0.0,
        is_baseline: bool = False,
        **kwargs,
    ) -> "ClassificationMetrics":
        """Create metrics from sklearn predictions."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            confusion_matrix,
        )
        import numpy as np

        cm = confusion_matrix(y_true, y_pred)
        n_classes = len(np.unique(y_true))

        # Per-class recall
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = {
            f"recall_class_{i}": float(per_class_recall[i])
            for i in range(min(len(per_class_recall), 5))
        }

        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        class_dist = {str(u): int(c) for u, c in zip(unique, counts)}

        return cls(
            model=model,
            dataset=dataset,
            is_baseline=is_baseline,
            accuracy=float(accuracy_score(y_true, y_pred)),
            f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            f1_micro=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            f1_weighted=float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            precision_macro=float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            precision_weighted=float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            recall_macro=float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            recall_weighted=float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            confusion_matrix=cm.tolist(),
            sample_size=len(y_true),
            n_classes=n_classes,
            class_distribution=class_dist,
            duration_seconds=duration_seconds,
            **recall_per_class,
            **kwargs,
        )


@dataclass
class RegressionMetrics:
    """Container for regression metrics with export functionality."""

    model: str
    dataset: str = "default"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_baseline: bool = False

    # Core metrics
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    smape: float = 0.0
    r2: float = 0.0

    # Per-zone metrics
    mae_per_zone: Optional[Dict[str, float]] = None

    # Additional info
    sample_size: int = 0
    n_zones: int = 0
    duration_seconds: float = 0.0
    hyperparameters: Optional[Dict[str, Any]] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        d = asdict(self)

        if self.mae_per_zone is not None:
            d["mae_per_zone_json"] = json.dumps(self.mae_per_zone)
        else:
            d["mae_per_zone_json"] = None
        del d["mae_per_zone"]

        if self.hyperparameters is not None:
            d["hyperparameters_json"] = json.dumps(self.hyperparameters)
        else:
            d["hyperparameters_json"] = None
        del d["hyperparameters"]

        return d


@dataclass
class BaselineComparison:
    """Container for model vs baseline comparison."""

    model: str
    baseline: str
    metric: str
    model_value: float
    baseline_value: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset: str = "default"

    @property
    def improvement_pct(self) -> float:
        """Calculate percentage improvement over baseline."""
        if self.baseline_value == 0:
            return 0.0
        return ((self.model_value - self.baseline_value) / abs(self.baseline_value)) * 100

    @property
    def improvement_absolute(self) -> float:
        """Calculate absolute improvement over baseline."""
        return self.model_value - self.baseline_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        d = asdict(self)
        d["improvement_pct"] = self.improvement_pct
        d["improvement_absolute"] = self.improvement_absolute
        return d


@dataclass
class TrainingHistoryEntry:
    """Single entry in training history."""

    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_metric: Optional[float] = None
    val_metric: Optional[float] = None
    metric_name: str = "accuracy"
    learning_rate: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def export_metrics_to_csv(
    metrics_list: List[ClassificationMetrics | RegressionMetrics],
    output_dir: str,
    model_name: str | None = None,
    base_name: str = "metrics",
    append: bool = False,
) -> Path:
    """
    Export a list of metrics to CSV with timestamped filename.

    Args:
        metrics_list: List of metrics objects
        output_dir: Directory for output CSV file
        model_name: Model name to include in filename
        base_name: Base filename (default: "metrics")
        append: If True, append to existing file (uses non-timestamped name)
        
    Returns:
        Path to the created CSV file
    """
    import pandas as pd
    from pathlib import Path

    if not metrics_list:
        return None

    records = [m.to_dict() for m in metrics_list]
    df = pd.DataFrame(records)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if append:
        # For append mode, use fixed filename to accumulate results
        output_path = output_dir / f"{base_name}.csv"
        if output_path.exists():
            existing_df = pd.read_csv(output_path)
            df = pd.concat([existing_df, df], ignore_index=True)
    else:
        # Use timestamped filename
        filename = generate_csv_filename(base_name, model_name)
        output_path = output_dir / filename

    df.to_csv(output_path, index=False)
    return output_path


def export_baseline_comparisons_to_csv(
    comparisons: List[BaselineComparison],
    output_dir: str,
    model_name: str | None = None,
    base_name: str = "baseline_comparisons",
) -> Path:
    """
    Export baseline comparisons to CSV with timestamped filename.

    Args:
        comparisons: List of BaselineComparison objects
        output_dir: Directory for output CSV file
        model_name: Model name to include in filename
        base_name: Base filename (default: "baseline_comparisons")
        
    Returns:
        Path to the created CSV file
    """
    import pandas as pd
    from pathlib import Path

    if not comparisons:
        return None

    records = [c.to_dict() for c in comparisons]
    df = pd.DataFrame(records)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = generate_csv_filename(base_name, model_name)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


def export_training_history_to_csv(
    history: List[TrainingHistoryEntry],
    output_dir: str,
    model_name: str | None = None,
    base_name: str = "training_history",
) -> Path:
    """
    Export training history to CSV with timestamped filename.

    Args:
        history: List of TrainingHistoryEntry objects
        output_dir: Directory for output CSV file
        model_name: Model name to include in filename
        base_name: Base filename (default: "training_history")
        
    Returns:
        Path to the created CSV file
    """
    import pandas as pd
    from pathlib import Path

    if not history:
        return None

    records = [asdict(h) for h in history]
    df = pd.DataFrame(records)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = generate_csv_filename(base_name, model_name)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    return output_path


def export_model_vs_baselines_csv(
    model_name: str,
    model_metrics: Dict[str, float],
    baseline_results: Dict[str, Any],
    output_dir: str,
    base_name: str = "baseline_comparison",
) -> Path:
    """
    Export model vs baseline comparison to CSV with timestamped filename.
    
    This helper function converts the typical training script output format
    to the structured BaselineComparison format.
    
    Args:
        model_name: Name of the trained model
        model_metrics: Dict of metric_name -> model_value
        baseline_results: Dict of baseline_name -> BaselineResult 
                         (with .metrics attribute containing metric_name -> baseline_value)
        output_dir: Directory for output CSV file
        base_name: Base filename (default: "baseline_comparison")
        
    Returns:
        Path to the created CSV file
    """
    import pandas as pd
    from pathlib import Path

    rows = []
    
    for metric_name, model_value in model_metrics.items():
        for baseline_name, baseline_result in baseline_results.items():
            # Handle both dict and object with .metrics attribute
            if hasattr(baseline_result, "metrics"):
                baseline_metrics = baseline_result.metrics
            else:
                baseline_metrics = baseline_result
                
            baseline_value = baseline_metrics.get(metric_name)
            if baseline_value is None:
                continue
                
            improvement = model_value - baseline_value
            improvement_pct = (
                (improvement / baseline_value * 100)
                if baseline_value != 0
                else 0.0
            )
            
            rows.append({
                "model": model_name,
                "baseline": baseline_name,
                "metric": metric_name,
                "model_value": model_value,
                "baseline_value": baseline_value,
                "improvement": improvement,
                "improvement_pct": round(improvement_pct, 2),
            })
    
    df = pd.DataFrame(rows)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = generate_csv_filename(base_name, model_name)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    return output_path
