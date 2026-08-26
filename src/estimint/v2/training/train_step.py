import logging

from flax import nnx
import optax
from typing import Callable
from jaxtyping import Array
import jax
import numpy as np
from ..data.preprocess import PreparedData
from omegaconf import DictConfig
import tqdm
import jax.numpy as jnp
import wandb
from ..data.dataset import make_loader
from tqdm import tqdm
from .checkpoint import save_checkpoint
from ..common.types import LossFn


log = logging.getLogger(__name__)


def create_optimizer(
    model: nnx.Module, learning_rate: float, total_steps: int, weight_decay: float = 1e-4
) -> nnx.Optimizer:
    """
    Create the training optimizer.

    Args:
        model: Model to optimize.
        learning_rate: Peak learning rate.
        total_steps: Total training steps.

    Returns:
        Configured optimizer.
    """
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=int(0.03 * total_steps),  # warmup for 3% of training
        decay_steps=total_steps,
        end_value=0.01 * learning_rate,  # decay to 1% of initial LR
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=scheduler, weight_decay=weight_decay))
    return nnx.Optimizer(model, tx, wrt=nnx.Param)

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

def _weighted_mean(terms: list[tuple[Array, Array]]) -> float:
    """Combine per-batch (total_losses, normalizations) pairs into the loss over all records."""
    total_losses, normalizations = (jnp.stack(t) for t in zip(*terms))
    return float(jnp.sum(total_losses) / jnp.sum(normalizations))

def make_train_step(loss_fn: LossFn):
    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, x: Array, y: Array, w: Array):
        def compute_total_loss(model: nnx.Module, x: Array, y: Array, w: Array):
            loss_sum, normalization = loss_fn(model, x, y, w)
            return loss_sum / normalization, (loss_sum, normalization)

        (_, (loss_sum, normalization)), grads = nnx.value_and_grad(compute_total_loss, has_aux=True)(model, x, y, w)
        optimizer.update(model, grads)
        return loss_sum, normalization

    return train_step

def make_eval_step(loss_fn: LossFn):
    @nnx.jit
    def eval_step(model: nnx.Module, x: Array, y: Array, w: Array):
        return loss_fn(model, x, y, w)

    return eval_step


def train_model(
    model: nnx.Module,
    cfg: DictConfig,
    prepared_data: PreparedData,
    loss_fn: LossFn,
    name: str,
    use_standardized_y: bool = False,
    ) -> nnx.Module:

    log.info(f"Total parameters: {get_total_params(model) / 1e6:.2f}M")

    train_step = make_train_step(loss_fn)
    eval_step = make_eval_step(loss_fn)
    target_key = "y_std" if use_standardized_y else "y"
    val_loader = make_loader(
        data=prepared_data.val_data,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=False,
    )
    total_steps = cfg.num_epochs * len(prepared_data.train_data) // cfg.batch_size
    optimizer = create_optimizer(model, cfg.lr, total_steps, weight_decay=cfg.weight_decay)

    # ---- training loop ----
    patience_n = 0
    best_val_loss = float("inf")
    best_state = jax.tree.map(lambda x: x, nnx.state(model))
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
        train_terms = [train_step(model, optimizer, batch["x"], batch[target_key], batch["w"]) for batch in train_loader]

        model.eval()
        val_terms = [eval_step(model, batch["x"], batch[target_key], batch["w"]) for batch in val_loader]

        avg_train_loss = _weighted_mean(train_terms)
        avg_val_loss = _weighted_mean(val_terms)
        epoch_pbar.set_postfix(
                train=f"{avg_train_loss:.6f}",
                val=f"{avg_val_loss:.6f}",
                patience=f"{patience_n}/{cfg.patience}",
            )
        if cfg.use_wandb:
            wandb.log({"train/loss": avg_train_loss, "val/loss": avg_val_loss, "epoch": epoch})
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = jax.tree.map(lambda x: x, nnx.state(model))
            patience_n = 0
        else:
            patience_n += 1
            if epoch >= cfg.min_epochs and patience_n >= cfg.patience:
                log.info(f"Early stopping at epoch {epoch} with best val loss {best_val_loss:.6f}")
                break

    nnx.update(model, best_state)
    save_checkpoint(cfg.checkpoint_dir, name, model)

    return model