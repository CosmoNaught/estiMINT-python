"""
Tests for the optimizer, train/eval steps, and the training loop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from estimint.v2.data.features import StandardScaler
from estimint.v2.data.preprocess import PreparedData
from estimint.v2.models.mlp import MLP
from estimint.v2.training.checkpoint import _resolve_checkpoint_dir
from estimint.v2.training.train_step import (
    create_optimizer,
    get_total_params,
    make_eval_step,
    make_train_step,
    train_model,
)

N_FEATURES = 3


def mse_loss(model, x, y, w):
    """Weighted MSE against a scalar target — stands in for rqs_loss."""
    preds = model(x)[:, 0]
    return jnp.sum(w * (preds - y) ** 2) / jnp.sum(w)


def make_model(seed=0, width=8, depth=1):
    return MLP(N_FEATURES, 1, width=width, depth=depth, dropout_rate=0.0, rngs=nnx.Rngs(seed))


def make_records(n=32, seed=0):
    """Records shaped like preprocess output: y_std is a linear function of x."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    y = X.sum(axis=1).astype(np.float32)
    return [{"x": X[i], "y": y[i], "y_std": y[i], "w": np.float32(1.0)} for i in range(n)]


def weights(model):
    return [np.asarray(w) for w in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))]


@pytest.fixture
def batch():
    records = make_records(16)
    return (
        jnp.stack([r["x"] for r in records]),
        jnp.array([r["y_std"] for r in records]),
        jnp.ones(len(records)),
    )


class TestGetTotalParams:
    def test_counts_every_weight_and_bias(self):
        # inp: 3*8 + 8, hidden: 8*8 + 8, out: 8*1 + 1
        assert get_total_params(make_model(width=8, depth=1)) == 32 + 72 + 9

    def test_deeper_models_have_more_parameters(self):
        assert get_total_params(make_model(depth=3)) > get_total_params(make_model(depth=1))


class TestCreateOptimizer:
    def apply_updates(self, model, optimizer, batch, n):
        for _ in range(n):
            _, grads = nnx.value_and_grad(mse_loss)(model, *batch)
            optimizer.update(model, grads)

    def test_first_step_is_a_no_op_because_warmup_starts_at_zero(self, batch):
        model = make_model()
        optimizer = create_optimizer(model, learning_rate=1e-2, total_steps=100)
        before = weights(model)

        self.apply_updates(model, optimizer, batch, 1)

        for a, b in zip(before, weights(model)):
            np.testing.assert_allclose(a, b)

    def test_optimizer_updates_the_model_once_warmup_ramps_up(self, batch):
        model = make_model()
        optimizer = create_optimizer(model, learning_rate=1e-2, total_steps=100)
        before = weights(model)

        self.apply_updates(model, optimizer, batch, 5)  # warmup is 3% of total_steps

        assert any(not np.allclose(a, b) for a, b in zip(before, weights(model)))


class TestTrainStep:
    def test_training_reduces_the_loss(self, batch):
        model = make_model()
        optimizer = create_optimizer(model, learning_rate=1e-2, total_steps=100)
        train_step = make_train_step(mse_loss)

        first = float(train_step(model, optimizer, *batch))
        for _ in range(50):
            last = float(train_step(model, optimizer, *batch))
        assert last < first

    def test_train_step_returns_the_pre_update_loss(self, batch):
        model = make_model()
        optimizer = create_optimizer(model, learning_rate=1e-2, total_steps=100)
        eval_step = make_eval_step(mse_loss)

        expected = float(eval_step(model, *batch))
        assert float(make_train_step(mse_loss)(model, optimizer, *batch)) == pytest.approx(expected, rel=1e-5)

    def test_eval_step_leaves_the_model_unchanged(self, batch):
        model = make_model()
        before = weights(model)
        make_eval_step(mse_loss)(model, *batch)
        for a, b in zip(before, weights(model)):
            np.testing.assert_array_equal(a, b)


@pytest.fixture
def cfg(tmp_path):
    return OmegaConf.create(
        {
            "batch_size": 8,
            "seed": 0,
            "num_workers": 0,
            "num_epochs": 8,
            "min_epochs": 0,
            "patience": 3,
            "lr": 1e-2,
            "weight_decay": 1e-4,
            "use_wandb": False,
            "checkpoint_dir": str(tmp_path / "ckpts"),
        }
    )


@pytest.fixture
def prepared_data():
    scaler = StandardScaler().fit(np.zeros((2, N_FEATURES)) + np.arange(N_FEATURES))
    return PreparedData(
        train_data=make_records(64, seed=0),
        val_data=make_records(32, seed=1),
        test_data=[],
        input_size=N_FEATURES,
        feature_scaler=scaler,
        target_scaler=scaler,
    )


class TestTrainModel:
    def test_training_improves_validation_loss(self, cfg, prepared_data):
        model = make_model()
        eval_step = make_eval_step(mse_loss)
        val = (
            jnp.stack([r["x"] for r in prepared_data.val_data]),
            jnp.array([r["y_std"] for r in prepared_data.val_data]),
            jnp.ones(len(prepared_data.val_data)),
        )
        before = float(eval_step(model, *val))

        trained = train_model(model, cfg, prepared_data, mse_loss, name="RQS", use_standardized_y=True)
        assert float(eval_step(trained, *val)) < before

    def test_training_writes_a_checkpoint(self, cfg, prepared_data):
        train_model(make_model(), cfg, prepared_data, mse_loss, name="RQS")
        assert _resolve_checkpoint_dir(cfg.checkpoint_dir, "RQS").exists()

    def test_returns_the_same_model_instance_updated_in_place(self, cfg, prepared_data):
        model = make_model()
        assert train_model(model, cfg, prepared_data, mse_loss, name="RQS") is model

    def test_early_stopping_ends_training_when_validation_stalls(self, cfg, prepared_data, caplog):
        cfg.num_epochs = 50
        cfg.patience = 2
        constant_loss = lambda model, x, y, w: jnp.sum(jnp.zeros_like(y)) + 1.0

        with caplog.at_level("INFO"):
            train_model(make_model(), cfg, prepared_data, constant_loss, name="RQS")

        assert "Early stopping" in caplog.text

    def test_min_epochs_defers_early_stopping(self, cfg, prepared_data, caplog):
        cfg.num_epochs = 4
        cfg.min_epochs = 4  # never eligible to stop or checkpoint a best model
        cfg.patience = 1
        constant_loss = lambda model, x, y, w: jnp.sum(jnp.zeros_like(y)) + 1.0

        with caplog.at_level("INFO"):
            train_model(make_model(), cfg, prepared_data, constant_loss, name="RQS")

        assert "Early stopping" not in caplog.text
