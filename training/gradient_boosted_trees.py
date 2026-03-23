from dataclasses import dataclass
from pathlib import Path
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader
import torch
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
    early_stopping_rounds: int | None = 20
    eval_metric: str | None = None
    seed: int = 42

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
            "seed": self.seed,
        }

        # Add num_class for multi-class objectives
        if "multi:" in self.objective:
            params["num_class"] = num_classes

        if self.eval_metric is not None:
            params["eval_metric"] = self.eval_metric
        else:
            # Sensible defaults depending on whether the objective yields probabilities.
            if "multi:" in self.objective:
                params["eval_metric"] = (
                    "mlogloss" if "softprob" in self.objective else "merror"
                )

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

    def loader_to_numpy(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        feature_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []
        for batch_features, batch_labels in loader:
            feature_batches.append(batch_features.detach().cpu())
            label_batches.append(batch_labels.detach().cpu())

        if not feature_batches:
            raise ValueError("DataLoader produced no batches")

        all_features = torch.cat(feature_batches, dim=0)
        all_labels = torch.cat(label_batches, dim=0)
        return all_features.numpy(), all_labels.numpy()

    X_train, y_train = loader_to_numpy(train_loader)
    X_val, y_val = loader_to_numpy(val_loader)

    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    y_train_sm = y_train_sm.astype(np.int64, copy=False)
    y_val = y_val.astype(np.int64, copy=False)

    train_data = xgb.DMatrix(
        X_train_sm,
        label=y_train_sm,
        enable_categorical=True,
    )
    val_data = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    evals = [(train_data, "train"), (val_data, "val")]
    model = xgb.train(
        params=config.to_xgboost_params(num_classes),
        dtrain=train_data,
        num_boost_round=config.n,
        evals=evals,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=False,
    )

    iteration_range = None
    if (
        config.early_stopping_rounds is not None
        and getattr(model, "best_iteration", None) is not None
    ):
        iteration_range = (0, model.best_iteration + 1)

    preds = model.predict(val_data, iteration_range=iteration_range)

    if "softprob" in config.objective:
        preds = preds.argmax(axis=1)

    preds = preds.astype(np.int64)
    accuracy = accuracy_score(y_val, preds)

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
