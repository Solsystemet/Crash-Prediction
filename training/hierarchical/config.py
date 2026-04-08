"""Configuration classes for hierarchical classification.

Provides dataclasses for configuring:
- Base hierarchical parameters (thresholds, splits)
- Tree-based model parameters (XGBoost, RandomForest, ExtraTrees)
- Neural network parameters (epochs, learning rate, architecture)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class HierarchicalConfig:
    """Base configuration for hierarchical classification.

    These parameters are shared across all model types.

    Attributes:
        sample_size: Optional limit on dataset size (None = use all).
        test_size: Fraction of data for testing.
        val_size: Fraction of training data for validation (threshold tuning).
        random_state: Random seed for reproducibility.
        l1_threshold: Probability threshold for injury detection.
        l2_threshold: Probability threshold for severity detection.
        l25_threshold: Probability threshold for fatal detection.
        l3_threshold: Probability threshold for reported detection.
    """

    # Data parameters
    sample_size: int | None = None
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

    # Classification thresholds (defaults, will be optimized)
    l1_threshold: float = 0.3
    l2_threshold: float = 0.2
    l25_threshold: float = 0.15
    l3_threshold: float = 0.5


@dataclass
class TreeHierarchicalConfig(HierarchicalConfig):
    """Configuration for tree-based hierarchical classifiers.

    Extends base config with tree-specific parameters and SMOTE settings.

    Attributes:
        l1_model: Model type for Level 1 ('rf', 'xgb', 'et').
        l2_model: Model type for Level 2.
        l25_model: Model type for Level 2.5.
        l3_model: Model type for Level 3.
        l1_sampling_strategy: SMOTE ratio for L1 (minority/majority).
        l2_sampling_strategy: SMOTE ratio for L2.
        l25_sampling_strategy: SMOTE ratio for L2.5.
        l3_sampling_strategy: SMOTE ratio for L3.
        l2_weight_multiplier: Extra weight multiplier for L2 positive class.
        l25_weight_multiplier: Extra weight multiplier for L2.5 positive class.
        n_estimators: Number of trees in ensemble.
        max_depth: Maximum tree depth.
        n_jobs: Parallel jobs (-1 for all cores).
    """

    # Model types per level
    l1_model: Literal["rf", "xgb", "et", "lgbm"] = "lgbm"
    l2_model: Literal["rf", "xgb", "et", "lgbm"] = "lgbm"
    l25_model: Literal["rf", "xgb", "et", "lgbm"] = "lgbm"
    l3_model: Literal["rf", "xgb", "et", "lgbm"] = "lgbm"

    # SMOTE sampling strategies (ratio of minority to majority)
    l1_sampling_strategy: float = 0.5
    l2_sampling_strategy: float = 0.7
    l25_sampling_strategy: float = 1.0  # Full balance for rare FATAL class
    l3_sampling_strategy: float = 0.5

    # Class weight multipliers for imbalanced levels
    l2_weight_multiplier: float = 1.5
    l25_weight_multiplier: float = 5.0  # Aggressive weighting for FATAL

    # Tree hyperparameters
    n_estimators: int = 200
    max_depth: int = 15
    n_jobs: int = -1


@dataclass
class NeuralHierarchicalConfig(HierarchicalConfig):
    """Configuration for neural network hierarchical classifiers.

    Extends base config with neural network-specific parameters.

    Attributes:
        epochs: Maximum training epochs per level.
        learning_rate: Adam optimizer learning rate.
        batch_size: Training batch size.
        hidden_sizes: Tuple of hidden layer sizes for each binary classifier.
        dropout: Dropout probability.
        early_stopping_patience: Stop if val loss doesn't improve for N epochs.
        use_class_weights: Whether to weight loss by inverse class frequency.
        device: Training device ('cuda' or 'cpu').
    """

    # Training parameters
    epochs: int = 30
    learning_rate: float = 1e-3
    batch_size: int = 256
    early_stopping_patience: int = 5

    # Architecture
    hidden_sizes: tuple[int, int] = (128, 64)
    dropout: float = 0.3

    # Class balancing (alternative to SMOTE)
    use_class_weights: bool = True

    # Device
    device: str = "auto"  # "auto", "cuda", or "cpu"

    def get_device(self) -> str:
        """Get actual device string, resolving 'auto'."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


@dataclass
class SimplifiedTreeConfig:
    """Configuration for simplified 2-level tree classifiers.

    Simplified 3-class system: SEVERE, MINOR, NO_INJURY
    Only uses L1 (injury) and L2 (severity) classifiers.

    Attributes:
        sample_size: Optional limit on dataset size (None = use all).
        test_size: Fraction of data for testing.
        val_size: Fraction of training data for validation.
        random_state: Random seed for reproducibility.
        l1_threshold: Probability threshold for injury detection.
        l2_threshold: Probability threshold for severity detection.
        l1_model: Model type for L1 ('rf', 'xgb', 'et', 'lgbm').
        l2_model: Model type for L2.
        l1_sampling_strategy: SMOTE ratio for L1.
        l2_sampling_strategy: SMOTE ratio for L2.
        l2_weight_multiplier: Extra weight for severe class.
        n_estimators: Number of trees in ensemble.
        max_depth: Maximum tree depth.
        n_jobs: Parallel jobs (-1 for all cores).
    """

    # Data parameters
    sample_size: int | None = None
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

    # Classification thresholds
    l1_threshold: float = 0.3
    l2_threshold: float = 0.3

    # Model types per level
    l1_model: Literal["rf", "xgb", "et", "lgbm"] = "lgbm"
    l2_model: Literal["rf", "xgb", "et", "lgbm"] = "lgbm"

    # SMOTE sampling strategies
    l1_sampling_strategy: float = 0.5
    l2_sampling_strategy: float = 0.7

    # Class weight multiplier for severe class
    l2_weight_multiplier: float = 1.5

    # Tree hyperparameters
    n_estimators: int = 200
    max_depth: int = 15
    n_jobs: int = -1
