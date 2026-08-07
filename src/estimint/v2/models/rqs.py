from flax import nnx
from .mlp import MLP
import jax.numpy as jnp
import jax
from estimint.v2.data.features import StandardScaler, FeatureScaler
import numpy as np
from omegaconf import DictConfig
from ..common.types import PredictorType, TargetType

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

    @classmethod
    def from_cfg(cls, cfg: DictConfig, n_context: int) -> "ConditionalRQS":
        return  cls(
        n_context,
        rngs=nnx.Rngs(cfg.get("seed", 0)),
        width=cfg.width,
        depth=cfg.depth,
        n_bins=cfg.n_bins,
        bounds=cfg.rqs_bounds,
        residual=cfg.mlp_residual,
        dropout_rate=cfg.dropout_rate)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str,
        predictor: PredictorType,
        target: TargetType,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_dir: str | None = None,
    ) -> "RQSArtifact":
        """
        Load a pretrained RQS artifact from a local folder or Hugging Face repo.

        Example usage:
        ```
        ConditionalRQS.from_pretrained("dide-ic/estiMINT", predictor="prev_y9", target="eir")
        ConditionalRQS.from_pretrained("dide-ic/estiMINT", predictor="prev_y9", target="eir", revision="v0.1.0")
        ```

        Args:
            path_or_repo_id: Hugging Face repo ID or local folder path.
            predictor: Predictor type.
            target: Target type.
            revision: Optional revision of the model to load from the repo.
            cache_dir: Optional cache directory for the Hugging Face repo.
            local_dir: Optional local directory to download the repo into.

        Returns:
            RQSArtifact containing the restored model and fitted scalers.
        """
        from .hub import load_model_artifact


        return load_model_artifact(
            path_or_repo_id, predictor, target, revision=revision, cache_dir=cache_dir, local_dir=local_dir,
        )

@nnx.jit
def _forward(model: nnx.Module, context: jnp.ndarray, quantile: float):
    return model.quantile(context, quantile) # type: ignore

FeatureInput = np.ndarray | dict[str, float] | list[dict[str, float]]
class RQSArtifact:
    def __init__(self, model: nnx.Module, feature_scaler: FeatureScaler | StandardScaler, target_scaler: StandardScaler, features: list[str], conformal: dict[float, float] = dict()):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.conformal = conformal # alpha -> offset Q
        self.feature_names = features

        if self.feature_scaler.mean_.shape[0] != len(self.feature_names):
            raise ValueError(f"Feature scaler has {self.feature_scaler.mean_.shape[0]} features, but expected {len(self.feature_names)} features for features {self.feature_names}.")

    def _prepare_inputs(self, X_raw: FeatureInput) -> np.ndarray:
        """Normalize user input to a (B, C) float32 array in training feature order."""
        if isinstance(X_raw, dict):
            X_raw = [X_raw]

        if isinstance(X_raw, list):
            if not X_raw:
                raise ValueError("Input list is empty.")

            names = self.feature_names
            rows = []
            for i, row in enumerate(X_raw):
                missing = [f for f in names if f not in row]
                extra = [f for f in row if f not in names]
                if missing or extra:
                    raise KeyError(f"row {i}: missing={missing}, unexpected={extra}. Expected exactly {names}.")
                rows.append([row[f] for f in names]) # preserve feature order
            X = np.array(rows, dtype=np.float32)
        else:
            X = np.asarray(X_raw, dtype=np.float32)
            if X.ndim == 1:
                X = X[None, :] # add batch dimension
        return X


    def _quantile(self, X_raw: FeatureInput, quantile: float) -> np.ndarray:
        X = self._prepare_inputs(X_raw)
        context = jnp.array(self.feature_scaler.transform(X))
        y0 = _forward(self.model, context, quantile)
        return np.maximum(0, np.power(10, self.target_scaler.inverse_transform(y0)))

    def predict(self, X_raw: FeatureInput) -> np.ndarray:
        return self._quantile(X_raw, 0.5) # median prediction

    def quantile(self, X_raw: FeatureInput, quantile: float) -> np.ndarray:
        return self._quantile(X_raw, quantile)

    def interval(self, X_raw: FeatureInput, alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
        """Conformal (1-alpha) band with guaranteed coverage on calibration set. Returns (lower, upper) bounds."""
        lower = self._quantile(X_raw, alpha / 2)
        upper  = self._quantile(X_raw, 1 - alpha / 2)
        Q = self.conformal.get(alpha, 0.0)
        return np.maximum(0, lower - Q), upper + Q

# ------------ RQS loss ----------------
def rqs_loss(model, X, y0, w):
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

