from flax import nnx

from estimint.v2.data.features import StandardScaler
from .mlp import MLP
import numpy as np
import jax.numpy as jnp


class MonotoneUMNN(nnx.Module):
    def __init__(self, n_context, *, width=128, depth=4, n_quad=48, mlp_residual=False, dropout_rate=0.1, rngs: nnx.Rngs):
        self.bias = MLP(n_context, 1, width=width, depth=depth, residual=mlp_residual, dropout_rate=dropout_rate, rngs=rngs)
        self.integrand = MLP(n_context + 1, 1, width=width, depth=depth, residual=mlp_residual, dropout_rate=dropout_rate, rngs=rngs)
        # leggauss(n) returns nodes/weights for integrating on [-1, 1]. We'll
        # rescale them to [0, m] at call time.
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        self.nodes = jnp.array(nodes) # (Q, )
        self.weights = jnp.array(weights) # (Q, )

    def _integral(self, m, c):
        # m: (B,) monotone feature; c: (B, C) context. Compute ∫₀ᵐ g dt per row.
        m = m[:, None] # (B, 1)
        half = 0.5 * m
        # map [-1, 1] -> [0, m] for each row
        t = half * (self.nodes[None, :]  + 1.0) # (B, Q)
        cE = jnp.broadcast_to(c[:, None, :], (c.shape[0], t.shape[1], c.shape[1])) # (B, Q, C)
        inp = jnp.concatenate([t[..., None], cE], axis=-1) # (B, Q, C+1) : [t, context]
        # compute g(t, c). g(t, c) >= 0
        g = nnx.softplus(self.integrand(inp))[..., 0] # (B, Q)
        return half[:, 0] * (self.weights[None, :] * g).sum(axis=1) # (B,)

    def __call__(self, m, c):
        # m: (B,) monotone feature; c: (B, C) context. Compute f(m, c) = b(c) + ∫₀ᵐ g dt
        return self.bias(c)[:, 0] + self._integral(m, c)

@nnx.jit
def _forward(models: list[MonotoneUMNN], m: jnp.ndarray, c: jnp.ndarray):
    return [model(m, c) for model in models]

class UMNNBundle:
    def __init__(self, models: list[MonotoneUMNN], scaler: StandardScaler, features: list[str]):
        self.models = models
        self.scaler = scaler
        self.features = features

    def set_models_to_eval(self):
        for model in self.models:
            model.eval()

    def _transform_inputs(self, X_raw) -> tuple[jnp.ndarray, jnp.ndarray]:
        X_scaled = self.scaler.transform(X_raw)
        context = X_scaled[:, 1:] # all but first column
        monotone_feature = X_scaled[:, 0] # first column
        return jnp.array(monotone_feature), jnp.array(context)


    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        m, c = self._transform_inputs(X_raw)
        preds = _forward(self.models, m, c)
        mean_pred = jnp.mean(jnp.stack(preds), axis=0)
        return np.power(10, mean_pred) # return in original scale


def umnn_loss(model, X, y, w):
    # X is arranged as [monotone_feature, *context]; split it back out.
    m, c = X[:, 0], X[:, 1:]
    pred = model(m, c)
    # Weighted mean-squared error on log10(EIR). Dividing by sum(w) makes it a
    # proper weighted average.
    return jnp.sum(w * (pred - y) ** 2) / jnp.sum(w)

