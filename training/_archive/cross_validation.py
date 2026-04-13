"""Stratified K-Fold cross-validation utilities.

Research emphasizes using Stratified K-Fold Cross-Validation to ensure
fair distribution of classes across training and validation sets,
especially important for imbalanced crash data.
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Any

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


@dataclass
class CrossValidationResult:
    """Results from cross-validation.

    Attributes:
        n_folds: Number of folds used.
        accuracy_scores: Accuracy for each fold.
        precision_scores: Precision (macro) for each fold.
        recall_scores: Recall (macro) for each fold.
        f1_scores: F1 (macro) for each fold.
        auc_scores: AUC-ROC for each fold (if computed).
        mean_accuracy: Mean accuracy across folds.
        std_accuracy: Std dev of accuracy.
        mean_precision: Mean precision across folds.
        std_precision: Std dev of precision.
        mean_recall: Mean recall across folds.
        std_recall: Std dev of recall.
        mean_f1: Mean F1 across folds.
        std_f1: Std dev of F1.
        mean_auc: Mean AUC across folds (if computed).
        std_auc: Std dev of AUC (if computed).
    """

    n_folds: int
    accuracy_scores: list[float] = field(default_factory=list)
    precision_scores: list[float] = field(default_factory=list)
    recall_scores: list[float] = field(default_factory=list)
    f1_scores: list[float] = field(default_factory=list)
    auc_scores: list[float] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float:
        return np.mean(self.accuracy_scores) if self.accuracy_scores else 0.0

    @property
    def std_accuracy(self) -> float:
        return np.std(self.accuracy_scores) if self.accuracy_scores else 0.0

    @property
    def mean_precision(self) -> float:
        return np.mean(self.precision_scores) if self.precision_scores else 0.0

    @property
    def std_precision(self) -> float:
        return np.std(self.precision_scores) if self.precision_scores else 0.0

    @property
    def mean_recall(self) -> float:
        return np.mean(self.recall_scores) if self.recall_scores else 0.0

    @property
    def std_recall(self) -> float:
        return np.std(self.recall_scores) if self.recall_scores else 0.0

    @property
    def mean_f1(self) -> float:
        return np.mean(self.f1_scores) if self.f1_scores else 0.0

    @property
    def std_f1(self) -> float:
        return np.std(self.f1_scores) if self.f1_scores else 0.0

    @property
    def mean_auc(self) -> float | None:
        return np.mean(self.auc_scores) if self.auc_scores else None

    @property
    def std_auc(self) -> float | None:
        return np.std(self.auc_scores) if self.auc_scores else None


def print_cv_results(result: CrossValidationResult, title: str = "Cross-Validation Results") -> None:
    """Print cross-validation results in a formatted table.

    Args:
        result: CrossValidationResult to print.
        title: Title for the output.
    """
    print(f"\n{title}")
    print("=" * 60)
    print(f"  Folds: {result.n_folds}")
    print("-" * 60)
    print(f"  {'Metric':<15} {'Mean':>10} {'± Std':>10}")
    print("-" * 60)
    print(f"  {'Accuracy':<15} {result.mean_accuracy:>10.4f} {result.std_accuracy:>10.4f}")
    print(f"  {'Precision':<15} {result.mean_precision:>10.4f} {result.std_precision:>10.4f}")
    print(f"  {'Recall':<15} {result.mean_recall:>10.4f} {result.std_recall:>10.4f}")
    print(f"  {'F1 (macro)':<15} {result.mean_f1:>10.4f} {result.std_f1:>10.4f}")

    if result.mean_auc is not None:
        print(f"  {'AUC-ROC':<15} {result.mean_auc:>10.4f} {result.std_auc:>10.4f}")

    print("=" * 60)


def stratified_kfold_cv(
    X: NDArray,
    y: NDArray,
    model_factory: Callable[[], Any],
    n_folds: int = 5,
    compute_auc: bool = True,
    verbose: bool = True,
    random_state: int = 42,
) -> CrossValidationResult:
    """Perform stratified k-fold cross-validation.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Labels array of shape (n_samples,).
        model_factory: Callable that returns a fresh model instance.
            Example: lambda: RandomForestClassifier(n_estimators=100)
        n_folds: Number of CV folds.
        compute_auc: Whether to compute AUC-ROC (requires predict_proba).
        verbose: Whether to print progress.
        random_state: Random seed for reproducible splits.

    Returns:
        CrossValidationResult containing metrics for all folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    result = CrossValidationResult(n_folds=n_folds)

    if verbose:
        print(f"\nRunning {n_folds}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        if verbose:
            print(f"  Fold {fold}/{n_folds}...", end=" ", flush=True)

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model = model_factory()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_val)

        # Compute metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        result.accuracy_scores.append(acc)
        result.precision_scores.append(prec)
        result.recall_scores.append(rec)
        result.f1_scores.append(f1)

        # Compute AUC if possible
        if compute_auc and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_val)
                n_classes = len(np.unique(y))

                if n_classes == 2:
                    auc = roc_auc_score(y_val, y_proba[:, 1])
                else:
                    auc = roc_auc_score(
                        y_val, y_proba, multi_class="ovr", average="macro"
                    )
                result.auc_scores.append(auc)
            except Exception:
                # AUC computation can fail in edge cases
                pass

        if verbose:
            print(f"F1={f1:.4f}")

    if verbose:
        print_cv_results(result)

    return result


def compare_models_cv(
    X: NDArray,
    y: NDArray,
    models: dict[str, Callable[[], Any]],
    n_folds: int = 5,
    verbose: bool = True,
) -> dict[str, CrossValidationResult]:
    """Compare multiple models using cross-validation.

    Args:
        X: Feature array.
        y: Labels array.
        models: Dictionary mapping model name to model factory.
        n_folds: Number of CV folds.
        verbose: Whether to print progress.

    Returns:
        Dictionary mapping model name to CrossValidationResult.
    """
    results = {}

    for name, model_factory in models.items():
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Model: {name}")
            print("=" * 60)

        result = stratified_kfold_cv(
            X, y, model_factory, n_folds=n_folds, verbose=verbose
        )
        results[name] = result

    # Print comparison summary
    if verbose:
        print(f"\n{'=' * 60}")
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(f"  {'Model':<20} {'F1 Mean':>10} {'F1 Std':>10} {'AUC Mean':>10}")
        print("-" * 60)

        for name, result in sorted(results.items(), key=lambda x: -x[1].mean_f1):
            auc_str = f"{result.mean_auc:.4f}" if result.mean_auc else "N/A"
            print(f"  {name:<20} {result.mean_f1:>10.4f} {result.std_f1:>10.4f} {auc_str:>10}")

    return results
