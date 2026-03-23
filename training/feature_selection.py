"""Feature selection utilities.

This module implements hybrid feature selection combining:
- Correlation-based Feature Selection (CFS): Remove highly correlated features
- Recursive Feature Elimination (RFE): Iteratively remove least important features

Research shows that removing noise and redundant features can significantly
boost accuracy and computational efficiency.
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


@dataclass
class FeatureSelectionResult:
    """Result of feature selection process.

    Attributes:
        selected_mask: Boolean mask of selected features.
        selected_indices: Indices of selected features.
        selected_names: Names of selected features (if provided).
        n_features_original: Number of features before selection.
        n_features_selected: Number of features after selection.
        feature_ranking: Ranking of all features (1 = best).
        correlation_removed: Features removed by correlation filter.
    """

    selected_mask: NDArray[np.bool_]
    selected_indices: NDArray[np.int64]
    selected_names: list[str] | None
    n_features_original: int
    n_features_selected: int
    feature_ranking: NDArray[np.int64] | None = None
    correlation_removed: list[str] | None = None


def correlation_filter(
    X: NDArray,
    feature_names: list[str] | None = None,
    threshold: float = 0.95,
    verbose: bool = True,
) -> tuple[NDArray, list[int], list[str]]:
    """Remove highly correlated features.

    For each pair of features with correlation > threshold, removes the one
    that has higher average correlation with all other features.

    Args:
        X: Feature array of shape (n_samples, n_features).
        feature_names: Names of features. Uses indices if None.
        threshold: Correlation threshold (0-1). Features with correlation
            above this are considered redundant.
        verbose: Whether to print progress.

    Returns:
        Tuple of (filtered_X, kept_indices, removed_feature_names).
    """
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)

    # Handle NaN correlations (constant features)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Find highly correlated pairs
    to_remove = set()
    removed_names = []

    for i in range(n_features):
        if i in to_remove:
            continue
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                # Remove the one with higher mean correlation
                mean_corr_i = np.mean(np.abs(corr_matrix[i, :]))
                mean_corr_j = np.mean(np.abs(corr_matrix[j, :]))

                if mean_corr_i > mean_corr_j:
                    to_remove.add(i)
                    removed_names.append(feature_names[i])
                else:
                    to_remove.add(j)
                    removed_names.append(feature_names[j])

    # Keep features not in to_remove
    kept_indices = [i for i in range(n_features) if i not in to_remove]
    filtered_X = X[:, kept_indices]

    if verbose and removed_names:
        print(f"\nCorrelation Filter (threshold={threshold}):")
        print(f"  Removed {len(removed_names)} highly correlated features:")
        for name in removed_names[:10]:  # Show first 10
            print(f"    - {name}")
        if len(removed_names) > 10:
            print(f"    ... and {len(removed_names) - 10} more")

    return filtered_X, kept_indices, removed_names


def recursive_feature_elimination(
    X: NDArray,
    y: NDArray,
    feature_names: list[str] | None = None,
    n_features_to_select: int | float | None = None,
    use_cv: bool = True,
    cv_folds: int = 5,
    step: int | float = 1,
    verbose: bool = True,
) -> FeatureSelectionResult:
    """Perform Recursive Feature Elimination.

    Uses a Random Forest estimator to rank features and iteratively
    removes the least important ones.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Labels array of shape (n_samples,).
        feature_names: Names of features. Uses indices if None.
        n_features_to_select: Number of features to select. If None with CV,
            automatically finds optimal number. If float, proportion of features.
        use_cv: Whether to use cross-validation to find optimal number of features.
        cv_folds: Number of CV folds.
        step: Number/proportion of features to remove at each iteration.
        verbose: Whether to print progress.

    Returns:
        FeatureSelectionResult containing selected features and rankings.
    """
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Convert proportion to number
    if isinstance(n_features_to_select, float) and n_features_to_select < 1.0:
        n_features_to_select = int(n_features * n_features_to_select)

    if verbose:
        print(f"\nRecursive Feature Elimination:")
        print(f"  Starting features: {n_features}")

    # Use Random Forest as the estimator
    estimator = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    if use_cv:
        # RFECV automatically finds optimal number of features
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        selector = RFECV(
            estimator=estimator,
            step=step,
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            min_features_to_select=max(1, n_features // 10),  # At least 10% of features
        )
        if verbose:
            print(f"  Using {cv_folds}-fold CV to find optimal features")
    else:
        # Standard RFE with fixed number of features
        if n_features_to_select is None:
            n_features_to_select = n_features // 2  # Default: keep half
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features_to_select,
            step=step,
        )
        if verbose:
            print(f"  Target features: {n_features_to_select}")

    # Fit selector
    selector.fit(X, y)

    # Get results
    selected_mask = selector.support_
    selected_indices = np.where(selected_mask)[0]
    selected_names = [feature_names[i] for i in selected_indices]
    feature_ranking = selector.ranking_

    if verbose:
        print(f"  Selected features: {len(selected_indices)}")
        print(f"\n  Top selected features:")
        for name in selected_names[:10]:
            print(f"    + {name}")

    return FeatureSelectionResult(
        selected_mask=selected_mask,
        selected_indices=selected_indices,
        selected_names=selected_names,
        n_features_original=n_features,
        n_features_selected=len(selected_indices),
        feature_ranking=feature_ranking,
    )


def hybrid_feature_selection(
    X: NDArray,
    y: NDArray,
    feature_names: list[str] | None = None,
    correlation_threshold: float = 0.95,
    use_rfe: bool = True,
    rfe_use_cv: bool = True,
    n_features_target: int | float | None = None,
    verbose: bool = True,
) -> tuple[NDArray, FeatureSelectionResult]:
    """Perform hybrid feature selection: CFS + RFE.

    First removes highly correlated features, then applies RFE to
    find the most important remaining features.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Labels array of shape (n_samples,).
        feature_names: Names of features.
        correlation_threshold: Threshold for correlation filter.
        use_rfe: Whether to apply RFE after correlation filter.
        rfe_use_cv: Whether to use CV in RFE.
        n_features_target: Target number of features for RFE.
        verbose: Whether to print progress.

    Returns:
        Tuple of (selected_X, FeatureSelectionResult).
    """
    n_original = X.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_original)]

    if verbose:
        print("=" * 60)
        print("Hybrid Feature Selection (CFS + RFE)")
        print("=" * 60)

    # Step 1: Correlation filter
    X_filtered, kept_indices, removed_names = correlation_filter(
        X, feature_names, threshold=correlation_threshold, verbose=verbose
    )
    filtered_names = [feature_names[i] for i in kept_indices]

    if not use_rfe:
        # Return after correlation filter only
        selected_mask = np.zeros(n_original, dtype=bool)
        selected_mask[kept_indices] = True

        return X_filtered, FeatureSelectionResult(
            selected_mask=selected_mask,
            selected_indices=np.array(kept_indices),
            selected_names=filtered_names,
            n_features_original=n_original,
            n_features_selected=len(kept_indices),
            correlation_removed=removed_names,
        )

    # Step 2: RFE on filtered features
    rfe_result = recursive_feature_elimination(
        X_filtered,
        y,
        feature_names=filtered_names,
        n_features_to_select=n_features_target,
        use_cv=rfe_use_cv,
        verbose=verbose,
    )

    # Map RFE selection back to original indices
    final_indices = np.array(kept_indices)[rfe_result.selected_indices]
    final_mask = np.zeros(n_original, dtype=bool)
    final_mask[final_indices] = True

    # Select final features
    X_selected = X[:, final_indices]

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Final: {n_original} → {len(final_indices)} features")
        print(f"{'=' * 60}")

    return X_selected, FeatureSelectionResult(
        selected_mask=final_mask,
        selected_indices=final_indices,
        selected_names=rfe_result.selected_names,
        n_features_original=n_original,
        n_features_selected=len(final_indices),
        feature_ranking=None,  # Ranking relative to filtered set
        correlation_removed=removed_names,
    )


def apply_feature_selection(
    X: NDArray,
    result: FeatureSelectionResult,
) -> NDArray:
    """Apply previously computed feature selection to new data.

    Args:
        X: Feature array to filter.
        result: FeatureSelectionResult from previous selection.

    Returns:
        Filtered feature array.
    """
    return X[:, result.selected_indices]
