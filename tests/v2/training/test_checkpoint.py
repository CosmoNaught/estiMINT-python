"""
Tests for Orbax checkpoint saving and restoring.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from estimint.v2.models.mlp import MLP
from estimint.v2.training.checkpoint import _resolve_checkpoint_dir, restore_model, save_checkpoint

MODEL_NAME = "RQS"


def make_model(seed):
    return MLP(3, 1, width=4, depth=1, dropout_rate=0.0, rngs=nnx.Rngs(seed))


def weights(model):
    return [np.asarray(w) for w in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))]


@pytest.fixture
def ckpt_dir(tmp_path):
    return str(tmp_path / "ckpts")


def test_resolve_checkpoint_dir_appends_the_model_name(tmp_path):
    assert _resolve_checkpoint_dir(str(tmp_path), MODEL_NAME).name == MODEL_NAME


def test_save_creates_the_checkpoint_directory(ckpt_dir):
    save_checkpoint(ckpt_dir, MODEL_NAME, make_model(0))
    assert _resolve_checkpoint_dir(ckpt_dir, MODEL_NAME).exists()


def test_restore_recovers_the_saved_weights(ckpt_dir):
    saved = make_model(0)
    save_checkpoint(ckpt_dir, MODEL_NAME, saved)

    restored = restore_model(ckpt_dir, MODEL_NAME, make_model(1))  # different init
    for got, want in zip(weights(restored), weights(saved)):
        np.testing.assert_allclose(got, want)


def test_restored_model_reproduces_predictions(ckpt_dir):
    saved = make_model(0)
    save_checkpoint(ckpt_dir, MODEL_NAME, saved)
    x = jnp.ones((2, 3))

    restored = restore_model(ckpt_dir, MODEL_NAME, make_model(1))
    np.testing.assert_allclose(restored(x), saved(x), rtol=1e-6)


def test_saving_twice_keeps_the_latest_weights(ckpt_dir):
    save_checkpoint(ckpt_dir, MODEL_NAME, make_model(0))
    latest = make_model(2)
    save_checkpoint(ckpt_dir, MODEL_NAME, latest)

    restored = restore_model(ckpt_dir, MODEL_NAME, make_model(1))
    for got, want in zip(weights(restored), weights(latest)):
        np.testing.assert_allclose(got, want)


def test_models_are_namespaced_by_name(ckpt_dir):
    first, second = make_model(0), make_model(2)
    save_checkpoint(ckpt_dir, "first", first)
    save_checkpoint(ckpt_dir, "second", second)

    restored = restore_model(ckpt_dir, "first", make_model(1))
    for got, want in zip(weights(restored), weights(first)):
        np.testing.assert_allclose(got, want)
