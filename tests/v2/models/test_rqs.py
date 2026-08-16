"""
Tests for the v2 conditional rational-quadratic spline (RQS) flow model.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from estimint.v2.data.features import StandardScaler
from estimint.v2.models.rqs import ConditionalRQS, RQSArtifact, _rqs, rqs_loss

BOUNDS = 6
N_BINS = 8
N_CONTEXT = 4


@pytest.fixture
def raw_params():
    """Unconstrained (widths, heights, derivatives) for a batch of splines."""
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    n = 16
    return (
        jax.random.normal(k1, (n, N_BINS)),
        jax.random.normal(k2, (n, N_BINS)),
        jax.random.normal(k3, (n, N_BINS + 1)),
    )


@pytest.fixture
def model():
    return ConditionalRQS(N_CONTEXT, width=16, depth=2, n_bins=N_BINS, bounds=BOUNDS, rngs=nnx.Rngs(0))


@pytest.fixture
def context():
    return jax.random.normal(jax.random.key(1), (16, N_CONTEXT))


class TestSpline:
    """The bare spline transform, independent of any network."""

    def test_forward_inverse_round_trip(self, raw_params):
        x = jnp.linspace(-BOUNDS + 0.1, BOUNDS - 0.1, 16)
        z, _ = _rqs(x, *raw_params, BOUNDS, inverse=False)
        x_back, _ = _rqs(z, *raw_params, BOUNDS, inverse=True)
        np.testing.assert_allclose(x_back, x, atol=1e-4)

    def test_identity_outside_bounds(self, raw_params):
        x = jnp.full((16,), BOUNDS + 2.0)
        z, log_det = _rqs(x, *raw_params, BOUNDS, inverse=False)
        np.testing.assert_allclose(z, x)
        np.testing.assert_allclose(log_det, 0.0)

    def test_forward_is_monotonic(self, raw_params):
        widths, heights, derivatives = raw_params
        # Evaluate one spline (row 0) on an increasing grid of inputs.
        grid = jnp.linspace(-BOUNDS + 0.1, BOUNDS - 0.1, 64)
        row = lambda p: jnp.repeat(p[:1], grid.shape[0], axis=0)
        z, _ = _rqs(grid, row(widths), row(heights), row(derivatives), BOUNDS, inverse=False)
        assert jnp.all(jnp.diff(z) > 0)

    def test_stays_within_bounds(self, raw_params):
        x = jnp.linspace(-BOUNDS + 0.1, BOUNDS - 0.1, 16)
        z, _ = _rqs(x, *raw_params, BOUNDS, inverse=False)
        assert jnp.all(jnp.abs(z) <= BOUNDS)

    def test_log_det_matches_numerical_derivative(self, raw_params):
        x = jnp.linspace(-BOUNDS + 0.5, BOUNDS - 0.5, 16)
        eps = 1e-3
        z_hi, _ = _rqs(x + eps, *raw_params, BOUNDS, inverse=False)
        z_lo, _ = _rqs(x - eps, *raw_params, BOUNDS, inverse=False)
        _, log_det = _rqs(x, *raw_params, BOUNDS, inverse=False)
        np.testing.assert_allclose(jnp.exp(log_det), (z_hi - z_lo) / (2 * eps), rtol=1e-2)


class TestConditionalRQS:
    def test_log_prob_shape_and_finite(self, model, context):
        y0 = jax.random.normal(jax.random.key(2), (context.shape[0],))
        log_prob = model.log_prob(y0, context)
        assert log_prob.shape == (context.shape[0],)
        assert jnp.all(jnp.isfinite(log_prob))

    def test_density_integrates_to_one(self, model, context):
        """The flow is a normalized density in the standardized target space."""
        grid = jnp.linspace(-12, 12, 4001)
        one_row = jnp.repeat(context[:1], grid.shape[0], axis=0)
        density = jnp.exp(model.log_prob(grid, one_row))
        assert jnp.trapezoid(density, grid) == pytest.approx(1.0, abs=1e-3)

    def test_quantiles_increase_with_probability(self, model, context):
        probs = jnp.array([0.05, 0.25, 0.5, 0.75, 0.95])
        y0 = model.quantiles(context, probs)  # (Q, B)
        assert y0.shape == (probs.shape[0], context.shape[0])
        assert jnp.all(jnp.diff(y0, axis=0) > 0)

    def test_quantiles_matches_single_quantile(self, model, context):
        probs = jnp.array([0.1, 0.5, 0.9])
        batched = model.quantiles(context, probs)
        for i, q in enumerate(probs):
            np.testing.assert_allclose(batched[i], model.quantile(context, float(q)), atol=1e-5)

    def test_median_maps_back_to_base_zero(self, model, context):
        """quantile(0.5) is the target value the flow maps to z = 0."""
        y0 = model.quantile(context, 0.5)
        widths, heights, derivatives = model._params(context)
        z, _ = _rqs(y0, widths, heights, derivatives, model.bounds, inverse=False)
        np.testing.assert_allclose(z, 0.0, atol=1e-4)

    def test_from_cfg(self):
        cfg = OmegaConf.create(
            {
                "seed": 0,
                "width": 16,
                "depth": 2,
                "n_bins": N_BINS,
                "rqs_bounds": BOUNDS,
                "mlp_residual": False,
                "dropout_rate": 0.0,
            }
        )
        model = ConditionalRQS.from_cfg(cfg, n_context=N_CONTEXT)
        assert model.K == N_BINS
        assert model.bounds == BOUNDS
        # net emits K widths + K heights + (K + 1) derivatives
        assert model.net(jnp.zeros((2, N_CONTEXT))).shape == (2, 3 * N_BINS + 1)

    def test_residual_variant_runs(self, context):
        model = ConditionalRQS(
            N_CONTEXT, width=16, depth=2, n_bins=N_BINS, bounds=BOUNDS, residual=True, rngs=nnx.Rngs(0)
        )
        assert jnp.all(jnp.isfinite(model.log_prob(jnp.zeros(context.shape[0]), context)))


class TestLoss:
    def test_loss_is_finite(self, model, context):
        y0 = jax.random.normal(jax.random.key(3), (context.shape[0],))
        w = jnp.ones_like(y0)
        total_loss, normalization = rqs_loss(model, context, y0, w)
        assert jnp.isfinite(total_loss / normalization)

    def test_loss_is_gradable(self, model, context):
        y0 = jax.random.normal(jax.random.key(3), (context.shape[0],))
        w = jnp.ones_like(y0)
        def objective(model, X, y0, w):
            total_loss, normalization = rqs_loss(model, X, y0, w)
            return total_loss / normalization

        grads = nnx.grad(objective)(model, context, y0, w)
        leaves = jax.tree_util.tree_leaves(grads)
        assert leaves and all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_zero_weight_rows_are_ignored(self, model, context):
        y0 = jax.random.normal(jax.random.key(3), (context.shape[0],))
        w = jnp.ones_like(y0).at[8:].set(0.0)
        masked_total, masked_normalization = rqs_loss(model, context, y0, w)
        kept_total, kept_normalization = rqs_loss(model, context[:8], y0[:8], jnp.ones(8))
        assert masked_total / masked_normalization == pytest.approx(float(kept_total / kept_normalization), rel=1e-5)


def make_scaler(mean, scale):
    scaler = StandardScaler()
    scaler.mean_ = np.array(mean, dtype=np.float32)
    scaler.scale_ = np.array(scale, dtype=np.float32)
    return scaler


@pytest.fixture
def features():
    return ["prev_y9", "dn0_use", "Q0", "phi_bednets"]


@pytest.fixture
def artifact(model, features):
    model.eval()
    return RQSArtifact(
        model=model,
        feature_scaler=make_scaler(np.zeros(len(features)), np.ones(len(features))),
        target_scaler=make_scaler([0.0], [1.0]),
        features=features,
    )


class TestRQSArtifact:
    def test_rejects_scaler_with_wrong_feature_count(self, model, features):
        with pytest.raises(ValueError, match="features"):
            RQSArtifact(
                model=model,
                feature_scaler=make_scaler(np.zeros(2), np.ones(2)),
                target_scaler=make_scaler([0.0], [1.0]),
                features=features,
            )

    def test_predict_shape_and_non_negative(self, artifact, features):
        X = np.random.default_rng(0).normal(size=(5, len(features))).astype(np.float32)
        preds = artifact.predict(X)
        assert preds.shape == (5,)
        assert np.all(preds >= 0)

    def test_predict_accepts_single_row(self, artifact, features):
        X = np.zeros(len(features), dtype=np.float32)
        assert artifact.predict(X).shape == (1,)

    def test_dict_input_matches_array_input(self, artifact, features):
        values = [0.4, 0.1, 0.9, 0.5]
        row = dict(zip(features, values))
        np.testing.assert_allclose(artifact.predict(row), artifact.predict(np.array(values, dtype=np.float32)))

    def test_dict_input_is_order_independent(self, artifact, features):
        row = {f: v for f, v in zip(features, [0.4, 0.1, 0.9, 0.5])}
        shuffled = dict(reversed(list(row.items())))
        np.testing.assert_allclose(artifact.predict(shuffled), artifact.predict(row))

    def test_list_of_dicts_is_batched(self, artifact, features):
        rows = [dict.fromkeys(features, 0.1), dict.fromkeys(features, 0.2)]
        assert artifact.predict(rows).shape == (2,)

    @pytest.mark.parametrize(
        "bad_row",
        [
            {"prev_y9": 0.1},  # missing features
            {"prev_y9": 0.1, "dn0_use": 0.1, "Q0": 0.1, "phi_bednets": 0.1, "nope": 0.1},  # unexpected feature
        ],
    )
    def test_rejects_malformed_dict_rows(self, artifact, bad_row):
        with pytest.raises(KeyError):
            artifact.predict(bad_row)

    def test_rejects_empty_input(self, artifact):
        with pytest.raises(ValueError, match="empty"):
            artifact.predict([])

    def test_quantiles_are_ordered(self, artifact, features):
        X = np.zeros((3, len(features)), dtype=np.float32)
        low, mid, high = (artifact.quantile(X, q) for q in (0.1, 0.5, 0.9))
        assert np.all(low <= mid) and np.all(mid <= high)

    def test_interval_brackets_the_prediction(self, artifact, features):
        X = np.zeros((3, len(features)), dtype=np.float32)
        lower, upper = artifact.interval(X, alpha=0.10)
        preds = artifact.predict(X)
        assert np.all(lower >= 0)
        assert np.all(lower <= preds) and np.all(preds <= upper)

    def test_conformal_offset_widens_the_interval(self, artifact, features):
        X = np.zeros((3, len(features)), dtype=np.float32)
        lower, upper = artifact.interval(X, alpha=0.10)
        artifact.conformal[0.10] = 1.0
        wide_lower, wide_upper = artifact.interval(X, alpha=0.10)
        assert np.all(wide_upper > upper)
        assert np.all(wide_lower <= lower)
