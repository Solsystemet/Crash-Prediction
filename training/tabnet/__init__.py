"""TabNet training module for crash severity prediction.

TabNet is an attention-based tabular learning architecture that provides
built-in feature selection and interpretability.
"""

from training.tabnet.train import train_tabnet, TabNetConfig

__all__ = ["train_tabnet", "TabNetConfig"]
