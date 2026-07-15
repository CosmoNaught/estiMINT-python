from flax import nnx
from .mlp import MLP
import jax.numpy as jnp
import jax
from estimint.v2.data.features import StandardScaler
from estimint.utils import fit_qmap_w, predict_qmap_w, scale_pos
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

    def quantiles(self, context, probs):
        """Evaluate many quantile levels at once, reusing the per-row spline params.

        Args:
            context: (B, C) conditioning features.
            probs:   (Q,) probability levels in (0, 1).

        Returns:
            (Q, B) standardized targets y0, one row per probability level.
        """
        widths, heights, derivatives = self._params(context)  # (B,K),(B,K),(B,K+1)
        zs = jax.scipy.stats.norm.ppf(probs)  # (Q,)

        def _invert(z_scalar):
            z_col = jnp.full((context.shape[0],), z_scalar)
            y0, _ = _rqs(z_col, widths, heights, derivatives, self.bounds, inverse=True)
            return y0

        return jax.vmap(_invert)(zs)  # (Q, B)

@nnx.jit
def _forward(model: nnx.Module, context: jnp.ndarray, quantile: float):
    return model.quantile(context, quantile) # type: ignore

@nnx.jit
def _forward_quantiles(model: nnx.Module, context: jnp.ndarray, probs: jnp.ndarray):
    return model.quantiles(context, probs) # type: ignore

class RQSArtifact:
    def __init__(self, model: nnx.Module, feature_scaler: StandardScaler, target_scaler: StandardScaler, features: list[str]):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.features = features
        self.conformal = {} # alpha -> offset Q

    def _quantile(self, X_raw: np.ndarray, quantile: float) -> np.ndarray:
        context = jnp.array(self.feature_scaler.transform(X_raw))
        y0 = _forward(self.model, context, quantile)
        return np.maximum(0, np.power(10, self.target_scaler.inverse_transform(y0)))

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        return self._quantile(X_raw, 0.5) # median prediction

    def quantile(self, X_raw: np.ndarray, quantile: float) -> np.ndarray:
        return self._quantile(X_raw, quantile)

    def interval(self, X_raw: np.ndarray, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
        """Conformal (1-alpha) band with guaranteed coverage on calibration set. Returns (lower, upper) bounds."""
        lower = self._quantile(X_raw, alpha / 2)
        upper  = self._quantile(X_raw, 1 - alpha / 2)
        Q = self.conformal.get(alpha, 0.0)
        return np.maximum(0, lower - Q), upper + Q

def conformal_offset(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray, alpha: float = 0.10) -> float:
    """Split-conformal (CQR, Romano 2019) width correction on a HELD-OUT split.

    Returns offset Q so that widening to [lo-Q, hi+Q] gives >= (1-alpha)
    marginal coverage. Fit this on the CALIB split (disjoint from the val split
    used for early stopping) so the guarantee is honest.
    """
    scores = np.maximum(lower - y_true, y_true - upper)
    n = len(scores)
    k = np.ceil((n + 1) * (1 - alpha)).astype(int)
    return float(np.sort(scores)[k - 1])

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


# ------------ RQS utils ----------------
def _spline_knots(raw_widths: jax.Array, raw_heights: jax.Array, raw_derivatives: jax.Array, bounds: int, n_points: int):
    """Map unconstrained net outputs to positive bin sizes/derivatives and knot coordinates.

    Widths and heights are softmax'd to sum to 2*bounds, so their cumulative sums
    (starting at -bounds) land exactly on +bounds. Derivatives are softplus'd to
    stay positive.
    """
    widths = jax.nn.softmax(raw_widths, axis=-1) * (2 * bounds)
    heights = jax.nn.softmax(raw_heights, axis=-1) * (2 * bounds)
    derivatives = jax.nn.softplus(raw_derivatives) + 1e-3

    knot_x = jnp.concatenate([jnp.full((n_points, 1), -bounds), -bounds + jnp.cumsum(widths, axis=-1)], axis=-1)  # (n_points, K+1)
    knot_y = jnp.concatenate([jnp.full((n_points, 1), -bounds), -bounds + jnp.cumsum(heights, axis=-1)], axis=-1)  # (n_points, K+1)
    return knot_x, knot_y, derivatives


def _locate_bin(x: jax.Array, knots: jax.Array, n_bins: int) -> jax.Array:
    """Index k of the bin containing x, i.e. knots[k] <= x < knots[k+1]."""
    bin_idx = jnp.sum((x[..., None] >= knots[:, :-1]).astype(jnp.int32), axis=-1) - 1
    return jnp.clip(bin_idx, 0, n_bins - 1)


def _gather_bin(knot: jax.Array, bin_idx: jax.Array):
    """Return (knot[i, bin_idx[i]], knot[i, bin_idx[i] + 1]) for every row i."""
    lo = jnp.take_along_axis(knot, bin_idx[:, None], axis=1)[:, 0]
    hi = jnp.take_along_axis(knot, bin_idx[:, None] + 1, axis=1)[:, 0]
    return lo, hi


def _rqs_logdet(theta: jax.Array, s: jax.Array, d_lo: jax.Array, d_hi: jax.Array):
    """log|dz/dx| at spline-local parameter theta in [0, 1] (Durkan et al. 2019, eq. 5).

    Also returns the shared denominator and (1 - theta), which the forward pass reuses.
    """
    theta_comp = 1.0 - theta
    denom = s + (d_hi + d_lo - 2 * s) * theta * theta_comp
    deriv_numer = s**2 * (d_hi * theta**2 + 2 * s * theta * theta_comp + d_lo * theta_comp**2)
    log_abs_det = jnp.log(deriv_numer) - 2 * jnp.log(denom)
    return log_abs_det, denom, theta_comp


def _solve_theta(z: jax.Array, y_lo: jax.Array, dy: jax.Array, s: jax.Array, d_lo: jax.Array, d_hi: jax.Array):
    """Invert eq. for theta given a target z: solve a*theta^2 + b*theta + c = 0.

    """
    dz = z - y_lo
    slope_term = d_hi + d_lo - 2 * s
    a = dy * (s - d_lo) + dz * slope_term
    b = dy * d_lo - dz * slope_term
    c = -s * dz
    return 2 * c / (-b - jnp.sqrt(jnp.maximum(b**2 - 4 * a * c, 0.0)))


def _rqs(x: jax.Array, raw_widths: jax.Array, raw_heights: jax.Array, raw_derivatives: jax.Array, bounds: int, inverse=False):
    """Evaluate the monotone rational-quadratic spline (Durkan et al. 2019, "Neural Spline Flows").

    Args:
        x               : (B,) points to transform. Will be z for inverse=False and y for inverse=True.
        raw_widths      : (B,K)   unconstrained bin widths.
        raw_heights     : (B,K)   unconstrained bin heights.
        raw_derivatives : (B,K+1) unconstrained knot derivatives.
        bounds          : the spline is the identity outside [-bounds, bounds].
        inverse         : False computes z = T(x); True computes x = T^{-1}(z).

    Returns:
        (transformed, log|d(transformed)/dx|).
    """
    n_points, n_bins = raw_widths.shape
    knot_x, knot_y, derivatives = _spline_knots(raw_widths, raw_heights, raw_derivatives, bounds, n_points)

    in_domain = (x > -bounds) & (x < bounds)
    x_clamped = jnp.clip(x, -bounds + 1e-6, bounds - 1e-6)

    # forward looks up knot_x for x; inverse looks up knot_y for z.
    bin_idx = _locate_bin(x_clamped, knot_y if inverse else knot_x, n_bins)
    x_lo, x_hi = _gather_bin(knot_x, bin_idx)
    y_lo, y_hi = _gather_bin(knot_y, bin_idx)
    d_lo, d_hi = _gather_bin(derivatives, bin_idx)
    dx, dy = x_hi - x_lo, y_hi - y_lo
    s = dy / dx  # bin slope

    if inverse:
        theta = _solve_theta(x_clamped, y_lo, dy, s, d_lo, d_hi)
        log_dzdx, _, _ = _rqs_logdet(theta, s, d_lo, d_hi)
        out = theta * dx + x_lo  # x = T^{-1}(z)
        log_abs_det = -log_dzdx  # d(T^{-1})/dz = 1 / (dz/dx)
    else:
        theta = (x_clamped - x_lo) / dx
        log_dzdx, denom, theta_comp = _rqs_logdet(theta, s, d_lo, d_hi)
        numer = dy * (s * theta**2 + d_lo * theta * theta_comp)
        out = y_lo + numer / denom  # z = T(x)
        log_abs_det = log_dzdx

    return jnp.where(in_domain, out, x), jnp.where(in_domain, log_abs_det, 0.0)

