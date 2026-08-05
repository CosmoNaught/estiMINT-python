import logging
from pathlib import Path

import duckdb
import hydra
import jax
from omegaconf import DictConfig, OmegaConf

from .data.preprocess import PreparedData, prepare_data
from .data.dataset import make_loader
import wandb
from .data.features import get_features
from flax import nnx
from .training.train_step import train_model
import numpy as np
from estimint.v2.eval.metrics import compute_metrics
from .models.rqs import ConditionalRQS, rqs_loss, RQSArtifact
from .training.calibrate import conformal_offset

log = logging.getLogger(__name__)

def train_rqs(cfg: DictConfig, prepared_data: PreparedData):
    features = get_features(cfg.predictor)
    model = ConditionalRQS.from_cfg(cfg, n_context=len(features))
    model = train_model(model, cfg, prepared_data, rqs_loss, name="RQS", use_standardized_y=True)

    rqs_artifact = RQSArtifact(model, prepared_data.feature_scaler, prepared_data.target_scaler, features=features)

    # ------------ calibration -------------------
    calib_loader = make_loader(
        data=prepared_data.calib_data,
        batch_size=len(prepared_data.calib_data), # load all at once
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
    for batch in calib_loader:
        calib_x_raw, calib_y_raw = batch["x_raw"], batch["y_raw"]
        lower, upper = rqs_artifact.quantile(calib_x_raw, 0.05), rqs_artifact.quantile(calib_x_raw, 0.95)
        rqs_artifact.conformal[0.10] = conformal_offset(lower, upper, calib_y_raw, alpha=0.10)

    # ------------ test evaluation ----------------
    test_loader = make_loader(
        data=prepared_data.test_data,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
    metrics = compute_metrics(rqs_artifact, test_loader)
    log.info(f"test R2={metrics.r2:.4f}  RMSE={metrics.rmse:.2f}  MAE={metrics.mae:.2f} MSE={metrics.mse:.2f} Bias={metrics.bias:.2f}")
    if cfg.use_wandb:
        wandb.log({"test/r2": metrics.r2, "test/rmse": metrics.rmse, "test/mae": metrics.mae, "test/mse": metrics.mse, "test/bias": metrics.bias})

    # confidence interval evaluation
    test_loader = make_loader(
        data=prepared_data.test_data,
        batch_size=len(prepared_data.test_data), # load all at once
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
    for batch in test_loader:
        test_x_raw, test_y_raw = batch["x_raw"], batch["y_raw"]
        raw_lower, raw_upper = rqs_artifact.quantile(test_x_raw, 0.05), rqs_artifact.quantile(test_x_raw, 0.95)
        conformal_lower, conformal_upper = rqs_artifact.interval(test_x_raw, alpha=0.10)

        coverage_raw = np.mean((test_y_raw >= raw_lower) & (test_y_raw <= raw_upper))
        coverage_conformal = np.mean((test_y_raw >= conformal_lower) & (test_y_raw <= conformal_upper))
        log.info(f"Raw 90% interval coverage: {coverage_raw:.4f}")
        log.info(f"Conformal 90% interval coverage: {coverage_conformal:.4f}")
        if cfg.use_wandb:
            wandb.log({"test/raw_coverage": coverage_raw, "test/conformal_coverage": coverage_conformal})
    return rqs_artifact

@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("JAX devices: %s", jax.devices())
    if cfg.use_wandb:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.name,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
                settings=wandb.Settings(start_method="thread"),
            )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    raw_df = duckdb.read_parquet(cfg.data_file).df()
    prepared_data = prepare_data(raw_df, cfg, calib_frac=cfg.calib_frac)

    train_rqs(cfg, prepared_data)

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()