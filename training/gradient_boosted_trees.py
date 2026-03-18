from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

from data_preparation.tensor_dataset import CrashTensorDataset


@dataclass
class GradientBoostingConfig:
    objective: str = "multi:softmax"
    max_depth: int = 3
    learning_rate: float = 0.1
    n: int = 50
    base_score: float = 0.5
    num_class: int | None = None  # Set automatically based on num_classes

    def to_xgboost_params(self, num_classes: int) -> dict:
        """Convert config to XGBoost parameters, excluding training-specific fields.

        Args:
            num_classes: Number of classes for multi-class classification.
        """
        params = {
            "objective": self.objective,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "base_score": self.base_score,
        }

        # Add num_class for multi-class objectives
        if "multi:" in self.objective:
            params["num_class"] = num_classes

        return params


def train_gradient_boosted_trees(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_features: int,
    num_classes: int,
    config: GradientBoostingConfig | None = None,
    verbose: bool = True,
    train_dataset: CrashTensorDataset | None = None,  # noqa: F821
):
    if config is None:
        config = GradientBoostingConfig()

    features, labels = next(iter(train_loader))
    val_features, val_labels = next(iter(val_loader))
    train_data = xgb.DMatrix(
        features.numpy(), label=labels.numpy(), enable_categorical=True
    )
    val_data = xgb.DMatrix(
        val_features.numpy(), label=val_labels.numpy(), enable_categorical=True
    )
    model = xgb.train(
        params=config.to_xgboost_params(num_classes),
        dtrain=train_data,
        num_boost_round=config.n,
    )
    preds = model.predict(val_data)

    if "softprob" in config.objective:
        preds = preds.argmax(axis=1)

    preds = preds.astype(np.int64)
    accuracy = accuracy_score(val_labels.numpy(), preds)

    if verbose:
        print("Accuracy of the model is:", accuracy * 100)

    return model, accuracy


def save_gradient_boosted_trees(model: xgb.Booster, path: Path | str) -> None:
    """Save a trained XGBoost model to disk.

    Args:
        model: Trained XGBoost model.
        path: Path where the model should be saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
