from dataclasses import dataclass
from pathlib import Path
from sklearn.utils import compute_sample_weight
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
    class_weight: str | None = "balanced"
    class_weight_clip: float | None = None
    early_stopping_rounds: int | None = 20
    eval_metric: str | None = None
    subsample: float | None = None
    colsample_bytree: float | None = None
    min_child_weight: float | None = None
    reg_alpha: float | None = None
    reg_lambda: float | None = None
    gamma: float | None = None
    max_delta_step: float | None = None
    tree_method: str | None = None
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

        if self.subsample is not None:
            params["subsample"] = self.subsample
        if self.colsample_bytree is not None:
            params["colsample_bytree"] = self.colsample_bytree
        if self.min_child_weight is not None:
            params["min_child_weight"] = self.min_child_weight
        if self.reg_alpha is not None:
            params["reg_alpha"] = self.reg_alpha
        if self.reg_lambda is not None:
            params["reg_lambda"] = self.reg_lambda
        if self.gamma is not None:
            params["gamma"] = self.gamma
        if self.max_delta_step is not None:
            params["max_delta_step"] = self.max_delta_step
        if self.tree_method is not None:
            params["tree_method"] = self.tree_method

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

    y_train = y_train.astype(np.int64, copy=False)
    y_val = y_val.astype(np.int64, copy=False)

    sample_weights: np.ndarray | None = None
    val_weights: np.ndarray | None = None
    if config.class_weight is not None:
        sample_weights = compute_sample_weight(
            class_weight=config.class_weight,
            y=y_train,
        ).astype(np.float32, copy=False)
        val_weights = compute_sample_weight(
            class_weight=config.class_weight,
            y=y_val,
        ).astype(np.float32, copy=False)

        if config.class_weight_clip is not None:
            sample_weights = np.clip(sample_weights, 0.0, config.class_weight_clip)
            val_weights = np.clip(val_weights, 0.0, config.class_weight_clip)

    train_data = xgb.DMatrix(
        X_train,
        label=y_train,
        weight=sample_weights,
        enable_categorical=True,
    )
    val_data = xgb.DMatrix(
        X_val,
        label=y_val,
        weight=val_weights,
        enable_categorical=True,
    )

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
