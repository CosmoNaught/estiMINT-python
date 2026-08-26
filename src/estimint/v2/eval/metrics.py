import numpy as np
from grain.python import DataLoader
from estimint.utils import mse, r2, rmse, mae, bias, medape
from dataclasses import dataclass
from estimint.v2.common.types import ModelArtifact

@dataclass
class Metrics:
    mse: float
    r2: float
    rmse: float
    log10_mse: float
    mae: float
    bias: float
    medape: float

def compute_metrics(
    model_artifact: ModelArtifact,
    loader: DataLoader,
):
    preds, targets = get_preds_targets(model_artifact, loader)

    return Metrics(
        mse=mse(targets, preds),
        r2=r2(targets, preds),
        rmse=rmse(targets, preds),
        mae=mae(targets, preds),
        bias=bias(targets, preds),
        medape=medape(targets, preds),
        log10_mse=mse(np.log10(targets), np.log10(preds))
    )


def get_preds_targets(model_artifact: ModelArtifact, data_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    all_preds, all_targets = [], []
    for batch in data_loader:
        preds = model_artifact.predict(batch["x_raw"])
        all_preds.append(preds)
        all_targets.append(batch["y_raw"])

    all_preds = np.concat(all_preds)
    all_targets = np.concat(all_targets)

    return all_preds, all_targets

