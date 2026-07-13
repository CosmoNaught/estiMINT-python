from flax import nnx
import optax
from typing import Callable
from jaxtyping import Array
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

def make_train_step(loss_fn: Callable):
    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, x: Array, y: Array, w: Array):
        loss, grads = nnx.value_and_grad(loss_fn)(model, x, y, w)
        optimizer.update(model, grads)
        return loss

    return train_step

def make_eval_step(loss_fn: Callable):
    @nnx.jit
    def eval_step(model: nnx.Module, x: Array, y: Array, w: Array):
        loss = loss_fn(model, x, y, w)
        return loss

    return eval_step