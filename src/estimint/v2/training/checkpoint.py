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


def _resolve_checkpoint_dir(checkpoint_dir: str, model_name: str) -> epath.Path:
    return (epath.Path(checkpoint_dir) / model_name).resolve()

def restore_model(
    checkpoint_dir: str,
    model_name: str,
    model: nnx.Module,
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
    ckpt_dir = _resolve_checkpoint_dir(checkpoint_dir, model_name)
    with ocp.training.Checkpointer(ckpt_dir) as ckptr:
        loaded = ckptr.load_checkpointables(
            abstract_checkpointables={"model": nnx.state(model)},
        )
        nnx.update(model, loaded["model"])
        return model


def save_checkpoint(checkpoint_dir: str, model_name: str, model: nnx.Module):
    ckpt_dir = _resolve_checkpoint_dir(checkpoint_dir, model_name)
    preservation_policy = ocp.training.preservation_policies.LatestN(n=1)
    with ocp.training.Checkpointer(ckpt_dir, preservation_policy=preservation_policy) as ckptr: # type: ignore[arg-type]
        ckptr.save_checkpointables(
            0,
            {
                "model": nnx.state(model),
            },
            overwrite=True
        )
