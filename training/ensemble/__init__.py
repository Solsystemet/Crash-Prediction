"""Ensemble methods for crash severity prediction.

Implements stacking and voting ensembles combining multiple base models.
"""

from training.ensemble.stacking import StackingEnsemble, train_stacking_ensemble

__all__ = ["StackingEnsemble", "train_stacking_ensemble"]
