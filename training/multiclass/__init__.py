"""Multiclass neural network classification module.

This module provides a direct 5-class neural network classifier for crash
severity prediction, as an alternative to the hierarchical binary cascade.

Classes:
    MulticlassNeuralConfig: Configuration for the multiclass classifier.
    MulticlassMLP: PyTorch neural network architecture.
    FocalLoss: Class-imbalance-aware loss function.
    MulticlassNeuralClassifier: Training and inference wrapper.
"""

from training.multiclass.config import MulticlassNeuralConfig
from training.multiclass.neural_network import MulticlassMLP
from training.multiclass.losses import FocalLoss, weighted_cross_entropy
from training.multiclass.classifier import MulticlassNeuralClassifier

__all__ = [
    "MulticlassNeuralConfig",
    "MulticlassMLP",
    "FocalLoss",
    "weighted_cross_entropy",
    "MulticlassNeuralClassifier",
]
