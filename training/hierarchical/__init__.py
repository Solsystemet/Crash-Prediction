"""Hierarchical classification package for crash severity prediction.

This package provides a model-agnostic framework for hierarchical classification
with a 4-level structure:
- Level 1: INJURY vs NO_INJURY
- Level 2: SEVERE vs MINOR (among injuries)
- Level 2.5: FATAL vs INCAPACITATING (among severe)
- Level 3: REPORTED vs VISIBLE (among minor injuries)

Usage:
    from training.hierarchical import (
        HierarchicalConfig,
        HierarchicalTargets,
        HierarchicalTreeClassifier,
        HierarchicalNeuralClassifier,
        prepare_hierarchical_targets,
        evaluate_hierarchical,
    )
"""

from training.hierarchical.config import (
    HierarchicalConfig,
    NeuralHierarchicalConfig,
    TreeHierarchicalConfig,
)
from training.hierarchical.structure import (
    HierarchyLevel,
    HierarchicalTargets,
    SEVERITY_CLASS_ORDER,
    prepare_hierarchical_targets,
    get_multiclass_indices,
)
from training.hierarchical.base import HierarchicalClassifierBase
from training.hierarchical.tree_classifier import HierarchicalTreeClassifier
from training.hierarchical.neural_classifier import HierarchicalNeuralClassifier
from training.hierarchical.evaluation import evaluate_hierarchical

__all__ = [
    # Configs
    "HierarchicalConfig",
    "TreeHierarchicalConfig",
    "NeuralHierarchicalConfig",
    # Structure
    "HierarchyLevel",
    "HierarchicalTargets",
    "SEVERITY_CLASS_ORDER",
    "prepare_hierarchical_targets",
    "get_multiclass_indices",
    # Classifiers
    "HierarchicalClassifierBase",
    "HierarchicalTreeClassifier",
    "HierarchicalNeuralClassifier",
    # Evaluation
    "evaluate_hierarchical",
]
