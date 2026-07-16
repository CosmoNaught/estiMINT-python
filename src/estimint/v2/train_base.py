import logging
import time
from pathlib import Path

import duckdb
import hydra
import jax
import jax.numpy as jnp
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from estimint.v2.training.checkpoint import save_checkpoint
from .models.umnn import MonotoneUMNN, umnn_loss, UMNNArtifact
from .data.preprocess import PreparedData, prepare_data
from .data.dataset import make_loader
import wandb
from grain.python import DataLoader
from .data.features import FEATURES_BASE
from flax import nnx
from .training.train_step import create_optimizer, make_train_step, make_eval_step
import numpy as np
from orbax.checkpoint import v1 as ocp
from etils import epath
from estimint.utils import r2, rmse, mae, mse
from estimint.v2.eval.metrics import compute_metrics
from typing import Callable
from jaxtyping import Array
from .models.rqs import ConditionalRQS, rqs_loss, RQSArtifact, conformal_offset

logging.getLogger("absl").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

def get_total_params(model: nnx.Module) -> int:
    """
    Get the total number of parameters in the model.

    Args:
        model: Flax module.

    Returns:
        Total parameter count.
    """
    params = nnx.state(model, nnx.Param)
    return sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))


def train_model(
    model: nnx.Module,
    cfg: DictConfig,
    prepared_data: PreparedData,
    loss_fn: Callable[[nnx.Module, Array, Array, Array], Array],
    name: str,
    use_standardized_y: bool = False,
    ) -> nnx.Module:
    train_step = make_train_step(loss_fn)
    eval_step = make_eval_step(loss_fn)
    target_key = "y_std" if use_standardized_y else "y"
    val_loader = make_loader(
        data=prepared_data.val_data,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
    total_steps = cfg.num_epochs * len(prepared_data.train_data) // cfg.batch_size
    optimizer = create_optimizer(model, cfg.lr, total_steps, weight_decay=cfg.weight_decay)

    # ---- training loop ----
    patience_n = 0
    best_val_loss = float("inf")
    best_model = nnx.state(model)
    epoch_pbar = tqdm(range(cfg.num_epochs), desc="Epoch")
    for epoch in epoch_pbar:
        # remake train loader each epoch to reshuffle with new seed
        train_loader = make_loader(
            data=prepared_data.train_data,
            batch_size=cfg.batch_size,
            seed=cfg.seed + epoch,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_remainder=True,
        )
        model.train()
        train_losses: list[jax.Array] = [train_step(model, optimizer, batch["x"], batch[target_key], batch["w"]) for batch in train_loader]

        model.eval()
        val_losses: list[jax.Array] = [eval_step(model, batch["x"], batch[target_key], batch["w"]) for batch in val_loader]

        avg_train_loss = float(jnp.mean(jnp.stack(train_losses)))
        avg_val_loss = float(jnp.mean(jnp.stack(val_losses)))
        epoch_pbar.set_postfix(
                train=f"{avg_train_loss:.6f}",
                val=f"{avg_val_loss:.6f}",
                patience=f"{patience_n}/{cfg.patience}",
            )
        if cfg.use_wandb:
            wandb.log({"train/loss": avg_train_loss, "val/loss": avg_val_loss, "epoch": epoch})
        if epoch < cfg.min_epochs:
            continue
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = nnx.state(model)
            patience_n = 0
        else:
            patience_n += 1
            if patience_n >= cfg.patience:
                log.info(f"Early stopping at epoch {epoch} with best val loss {best_val_loss:.6f}")
                break

    nnx.update(model, best_model)
    save_checkpoint(cfg.output_dir, name, model)

    return model


def train_umnn(cfg: DictConfig, prepared_data: PreparedData) -> UMNNArtifact:
    model = MonotoneUMNN(len(FEATURES_BASE) - 1, rngs=nnx.Rngs(cfg.seed), width=cfg.width, depth=cfg.depth, n_quad=cfg.n_quad, mlp_residual=cfg.mlp_residual, dropout_rate=cfg.dropout_rate)
    log.info(f"Total parameters: {get_total_params(model) / 1e6:.2f}M")
    model = train_model(model, cfg, prepared_data, umnn_loss, name="UMNN")

    umnn_artifact = UMNNArtifact(model, prepared_data.feature_scaler, FEATURES_BASE)

    # ------------ test evaluation ----------------
    umnn_artifact.model.eval()
    test_loader = make_loader(
        data=prepared_data.test_data,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
    metrics = compute_metrics(umnn_artifact, test_loader)
    log.info(f"test R2={metrics.r2:.4f}  RMSE={metrics.rmse:.2f}  MAE={metrics.mae:.2f} MSE={metrics.mse:.2f} Bias={metrics.bias:.2f}")
    if cfg.use_wandb:
        wandb.log({"test/r2": metrics.r2, "test/rmse": metrics.rmse, "test/mae": metrics.mae, "test/mse": metrics.mse, "test/bias": metrics.bias})
    return umnn_artifact

def train_rqs(cfg: DictConfig, prepared_data: PreparedData):
    model = ConditionalRQS(len(FEATURES_BASE), rngs=nnx.Rngs(cfg.seed), width=cfg.width, depth=cfg.depth, n_bins=cfg.n_bins, bounds=cfg.rqs_bounds, residual=cfg.mlp_residual, dropout_rate=cfg.dropout_rate)
    log.info(f"Total parameters: {get_total_params(model) / 1e6:.2f}M")
    model = train_model(model, cfg, prepared_data, rqs_loss, name="RQS", use_standardized_y=True)

    rqs_artifact = RQSArtifact(model, prepared_data.feature_scaler, prepared_data.target_scaler, FEATURES_BASE)

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
    # -------- data loading and preprocessing --------
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    raw_df = duckdb.read_parquet(cfg.data_file).df()

    prepared_data = prepare_data(raw_df, cfg, calib_frac=cfg.calib_frac)

    # umnn_bundle = train_umnn(cfg, prepared_data)
    rqs_artifact = train_rqs(cfg, prepared_data)

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()