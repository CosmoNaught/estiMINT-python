from flax import nnx
from .mlp import MLP
import jax.numpy as jnp
import jax
from estimint.v2.data.features import StandardScaler
import numpy as np


class ConditionalRQS(nnx.Module):
    """Conditional rational-quadratic spline flow.

    Note: The inverse is flipped as compaerd to standard normalizing flow convention.
    Usually it is as x = T(z) where z ~ N(0, 1). Here, we have defined z = T(x) where z ~ N(0, 1).
    And x = T^{-1}(z)


    """
    def __init__(self, n_context, *, width=128, depth=4, n_bins=12, bounds=6, residual=False, dropout_rate=0.0, rngs: nnx.Rngs):
        self.K = n_bins
        self.bounds = bounds
        # The net maps context -> all spline params: K widths + K heights +
        # (K+1) derivatives = 3K+1 numbers.
        self.net = MLP(n_context, 3 * n_bins + 1, width=width, depth=depth, residual=residual, dropout_rate=dropout_rate, rngs=rngs)

    def _params(self, context):
        params = self.net(context) # (B, 3K + 1)
        return jnp.split(params, [self.K, 2 * self.K], axis=-1)  # widths(K), heights(K), derivatives(K+1)

    def log_prob(self, y0, context):
        """standardized target y0 -> base z; density = Normal(z) * |dT/dy|."""
        widths, heights, derivatives = self._params(context)
        z, log_det = _rqs(y0, widths, heights, derivatives, self.bounds, inverse=False)
        return jax.scipy.stats.norm.logpdf(z) + log_det

    def quantile(self, context, quantile):
        """Flow base z -> target y.The q-quantile of y is flow T^{-1}(Phi^{-1}(q))"""
        widths, heights, derivatives = self._params(context)
        z = jax.scipy.stats.norm.ppf(quantile) # Phi^{-1}(q)
        y0, _ = _rqs(z, widths, heights, derivatives, self.bounds, inverse=True)
        return y0

@nnx.jit
def _forward(models: list[ConditionalRQS], context: jnp.ndarray, quantile: float):
    return [model.quantile(context, quantile) for model in models]

class RQSBundle:
    def __init__(self, models: list[ConditionalRQS], scaler_x: StandardScaler, scaler_y: StandardScaler, features: list[str]):
        self.models = models
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.features = features
        self.conformal = {} # alpha -> offset Q (fit on calibration set)

    def _quantile(self, X_raw: np.ndarray, quantile: float) -> np.ndarray:
        context = jnp.array(self.scaler_x.transform(X_raw))
        y0 = np.mean(_forward(self.models, context, quantile), axis=0)
        return np.maximum(0, np.power(10, self.scaler_y.inverse_transform(y0))) # return in original scale


def _rqs(x: jax.Array, raw_widths: jax.Array, raw_heights: jax.Array, raw_derivatives: jax.Array, bounds: int, inverse=False):
    """Evaluate the monotone rational-quadratic spline

    Args:
        x               : (B,) points to transform. Will be z for inverse=False and y for inverse=True.
        raw_widths      : (B,K)   unconstrained bin WIDTHS (softmax'd to sum to 2B).
        raw_heights     : (B,K)   unconstrained bin HEIGHTS (softmax'd to sum to 2B).
        raw_derivatives : (B,K+1) unconstrained knot DERIVATIVES (softplus'd to be positive).
        inverse         : False computes z = T(x); True computes x = T^{-1}(z).
        inverse : False computes z = T(x); True computes x = T^{-1}(z).

    Returns
        (transformed, log|derivative|).
    """
    B, K = raw_widths.shape
    # unconstrained widths/heights/derivatives -> +ve and sum to 2*bounds
    widths = jax.nn.softmax(raw_widths, axis=-1) * (2 * bounds)
    heights = jax.nn.softmax(raw_heights, axis=-1) * (2 * bounds)
    derivatives = jax.nn.softplus(raw_derivatives) + 1e-3
    # cumulative sums to get knot locations. starting at -bounds.
    knot_x = jnp.concatenate([jnp.full((B, 1), -bounds), - bounds + jnp.cumsum(widths, axis=-1)], axis=-1) # (B, K+1)
    knot_y = jnp.concatenate([jnp.full((B, 1), -bounds), - bounds + jnp.cumsum(heights, axis=-1)], axis=-1) # (B, K+1)

    in_domain = (x > -bounds) & (x < bounds)
    x_clamped = jnp.clip(x, -bounds + 1e-6, bounds - 1e-6)
    # forward search for knot_x for input x. inverse searches for knot_y for input z
    knot_coords_to_search = knot_y if inverse else knot_x
    bin_idx = jnp.sum((x_clamped[..., None] >= knot_coords_to_search[:,:-1]).astype(jnp.int32), axis=-1) - 1 # (B,) which bin each x is in
    bin_idx = jnp.clip(bin_idx, 0, K - 1)

    def take_per_row(knot, bin_idx):
        """Select knot[i, bin_idx[i]] for each row i in knot."""
        return jnp.take_along_axis(knot, bin_idx[:, None], axis=1)[:, 0]

    # Left/right knot values bracketing each point's bin.
    x_lo, x_hi = take_per_row(knot_x, bin_idx), take_per_row(knot_x, bin_idx + 1)
    y_lo, y_hi = take_per_row(knot_y, bin_idx), take_per_row(knot_y, bin_idx + 1)
    deriv_lo, deriv_hi = take_per_row(derivatives, bin_idx), take_per_row(derivatives, bin_idx + 1)
    bin_slope = (y_hi - y_lo) / (x_hi - x_lo)

    if not inverse:
        # forward transform: z = T(x)
        theta = (x_clamped - x_lo) / (x_hi - x_lo)
        theta_comp = 1.0 - theta
        numerator = (y_hi - y_lo) * (bin_slope * theta**2 + deriv_lo * theta * theta_comp)
        denominator = bin_slope + (deriv_hi + deriv_lo - 2 * bin_slope) * theta * theta_comp
        z = y_lo + numerator / denominator # z = T(x)

        deriv_numerator = bin_slope**2 * (
            deriv_hi * theta**2 + 2 * bin_slope * theta * theta_comp + deriv_lo * theta_comp**2
        )
        log_abs_det = jnp.log(deriv_numerator) - 2 * jnp.log(denominator)

        return jnp.where(in_domain, z, x), jnp.where(in_domain, log_abs_det, 0.0)
    else:
        # inverse transform: x = t^{-1}(z)
        # Solve for theta theta via the quadratic a*theta^2 + b*theta + c = 0
        y_offest = x_clamped - y_lo
        slope_term = deriv_hi + deriv_lo - 2 * bin_slope
        a = (y_hi - y_lo) * (bin_slope - deriv_lo) + y_offest * slope_term
        b = (y_hi - y_lo) * deriv_lo - y_offest * slope_term
        c = -bin_slope * y_offest

        theta = 2 * c / (-b - jnp.sqrt(jnp.maximum(b**2 - 4 * a * c, 0.0))) # quadratic formula
        theta_comp = 1.0 - theta
        x_out = theta * (x_hi - x_lo) + x_lo # x = T^{-1}(z)

        denominator = bin_slope + slope_term * theta * theta_comp
        deriv_numerator = bin_slope**2 * (
            deriv_hi * theta**2 + 2 * bin_slope * theta * theta_comp + deriv_lo * theta_comp**2
        )
        # d(T^{-1})/dz = 1 / (dT/dx).
        log_abs_det = -(jnp.log(deriv_numerator) - 2 * jnp.log(denominator))
        return jnp.where(in_domain, x_out, x), jnp.where(in_domain, log_abs_det, 0.0)

def rqs_loss(model, X, y0, w):
    """
    Compute the negative log-likelihood loss for the conditional rational-quadratic spline flow model.

    Args:
        model: An instance of the ConditionalRQS model.
        X: Input context data (features).
        y0: Standardized target data (labels).
        w: Sample weights for each data point.
    Returns:
        The average negative log-likelihood loss, weighted by the sample weights.
    """
    log_prob = model.log_prob(y0, X)
    return -jnp.sum(w * log_prob) / jnp.sum(w)