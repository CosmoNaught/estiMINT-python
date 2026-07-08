import logging
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from typing import Any, Iterator

import flax.nnx as nnx
from orbax.checkpoint import v1 as ocp
from etils import epath

log = logging.getLogger(__name__)

# Orbax checkpointing logs verbosely via the absl logger; silence INFO-level noise.
logging.getLogger("absl").setLevel(logging.WARNING)


def restore_model(
    ckptr: ocp.training.Checkpointer,
    model: nnx.Module,
    step: int | None = None,
) -> nnx.Module:
    """
    Restore model from checkpoint.

    Args:
        ckptr: Orbax checkpointer.
        model: Model to restore.
        step: Checkpoint step to restore. If None, restore the latest checkpoint.

    Returns:
        Restored model.
    """
    loaded = ckptr.load_checkpointables(
        step,
        abstract_checkpointables={"model": nnx.state(model)},
    )
    nnx.update(model, loaded["model"])
    return model


def init_or_restore_last(
    ckptr: ocp.training.Checkpointer,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    restore_checkpoint: bool = False,
) -> tuple[nnx.Module, nnx.Optimizer, int, float]:
    """
    Initialize or restore model and optimizer from checkpoint.

    Args:
        ckptr: Orbax checkpointer.
        model: Model to restore.
        optimizer: Optimizer to restore.
        restore_checkpoint: Whether to restore from checkpoint.

    Returns:
        Model, optimizer, start epoch, and validation loss from the checkpoint.
    """
    if not restore_checkpoint or not ckptr.latest:
        log.info("Initializing model and optimizer from scratch.")
        return model, optimizer, 0, float("inf")

    log.info(f"Restoring model and optimizer from checkpoint: {ckptr.latest.step}")
    loaded_state = ckptr.load_checkpointables(
        abstract_checkpointables={
            "model": nnx.state(model),
            "optimizer": nnx.state(optimizer),
        }
    )
    nnx.update(model, loaded_state["model"])
    nnx.update(optimizer, loaded_state["optimizer"])
    metadata = ckptr.metadata()
    metrics: dict[str, Any] = metadata.metrics if isinstance(metadata.metrics, dict) else {}
    val_loss = float(metrics.get("val/loss", float("inf")))

    return model, optimizer, ckptr.latest.step + 1, val_loss


@dataclass
class CheckpointSession:
    ckptr: ocp.training.Checkpointer
    model: nnx.Module
    optimizer: nnx.Optimizer
    start_epoch: int
    best_val_loss: float

    def save_if_best(self, epoch: int, val_loss: float) -> bool:
        """
        Save a checkpoint when validation loss improves.

        Args:
            epoch: Current epoch.
            val_loss: Current validation loss.

        Returns:
            Whether a checkpoint was saved.
        """
        if val_loss >= self.best_val_loss:
            return False

        self.best_val_loss = val_loss
        self.ckptr.save_checkpointables_async(
            epoch,
            {
                "model": nnx.state(self.model),
                "optimizer": nnx.state(self.optimizer),
            },
            metrics={"val/loss": self.best_val_loss},
            overwrite=True,
        )
        return True


@contextmanager
def checkpoint_session(
    checkpoint_dir: str | PathLike[str],
    max_checkpoints_to_keep: int,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    restore_checkpoint: bool = False,
) -> Iterator[CheckpointSession]:
    """
    Open a checkpointing session.

    Args:
        checkpoint_dir: Checkpoint directory.
        max_checkpoints_to_keep: Number of checkpoints to keep.
        model: Model to checkpoint.
        optimizer: Optimizer to checkpoint.
        restore_checkpoint: Whether to restore existing state.

    Returns:
        Checkpoint session iterator.
    """
    ckpt_dir = epath.Path(checkpoint_dir).resolve()
    with ocp.training.Checkpointer(
        ckpt_dir,
        preservation_policy=ocp.training.preservation_policies.LatestN(max_checkpoints_to_keep),  # type: ignore[arg-type]
    ) as ckptr:
        model, optimizer, start_epoch, best_val_loss = init_or_restore_last(ckptr, model, optimizer, restore_checkpoint)
        yield CheckpointSession(
            ckptr=ckptr,
            model=model,
            optimizer=optimizer,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
        )
