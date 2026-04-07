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
from training.hierarchical.tree_classifier import (
    HierarchicalTreeClassifier,
    save_hierarchical_model,
    load_hierarchical_model,
)
from training.hierarchical.neural_classifier import HierarchicalNeuralClassifier
from training.hierarchical.evaluation import evaluate_hierarchical

# Simplified 3-class system
from training.hierarchical.config import SimplifiedTreeConfig
from training.hierarchical.simplified_targets import (
    SimplifiedSeverity,
    SimplifiedTargets,
    SIMPLIFIED_CLASS_NAMES,
    prepare_simplified_targets,
    map_simplified_predictions,
)
from training.hierarchical.simplified_classifier import (
    SimplifiedClassifierBase,
    SimplifiedTreeClassifier,
    save_simplified_model,
    load_simplified_model,
)

__all__ = [
    # Configs
    "HierarchicalConfig",
    "TreeHierarchicalConfig",
    "NeuralHierarchicalConfig",
    "SimplifiedTreeConfig",
    # Structure (5-class hierarchical)
    "HierarchyLevel",
    "HierarchicalTargets",
    "SEVERITY_CLASS_ORDER",
    "prepare_hierarchical_targets",
    "get_multiclass_indices",
    # Structure (3-class simplified)
    "SimplifiedSeverity",
    "SimplifiedTargets",
    "SIMPLIFIED_CLASS_NAMES",
    "prepare_simplified_targets",
    "map_simplified_predictions",
    # Classifiers (hierarchical)
    "HierarchicalClassifierBase",
    "HierarchicalTreeClassifier",
    "HierarchicalNeuralClassifier",
    # Classifiers (simplified)
    "SimplifiedClassifierBase",
    "SimplifiedTreeClassifier",
    # Save/Load utilities
    "save_simplified_model",
    "load_simplified_model",
    "save_hierarchical_model",
    "load_hierarchical_model",
    # Evaluation
    "evaluate_hierarchical",
]
