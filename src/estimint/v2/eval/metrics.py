import numpy as np
from grain.python import DataLoader
from jax import Array
from estimint.utils import mse, r2, rmse, mae, bias
from dataclasses import dataclass

@dataclass
class Metrics:
    mse: float
    r2: float
    rmse: float
    mae: float
    bias: float

def compute_metrics(
    model_bundle,
    loader: DataLoader,
):
    preds, targets = get_preds_targets(model_bundle, loader)

    return Metrics(
        mse=mse(targets, preds),
        r2=r2(targets, preds),
        rmse=rmse(targets, preds),
        mae=mae(targets, preds),
        bias=bias(targets, preds)
    )


def get_preds_targets(model_bundle, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    all_preds, all_targets = [], []
    for batch in data_loader:
        preds = model_bundle.predict(batch["x_raw"])
        all_preds.append(preds)
        all_targets.append(batch["y_raw"])
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_preds, all_targets

