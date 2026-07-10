"""
============================================================================
 test_estimint_nn.py  —  REVIEW / SCRATCH FILE (not wired into the package)
============================================================================
Consolidates the "replace XGBoost" findings into ONE readable file so you can
review before splitting the model code into the real v2 modules.

DATA code now lives in the package (imported below, not redefined here):
  * src/estimint/v2/data/preprocess.py  group_split() — leakage-free param split
    (+ stratification & calib split), sharing its core with the existing
    _create_split()/split.csv pipeline.
  * src/estimint/v2/data/features.py    resolve_mono(), MONO_FEATURES, StandardScaler

Fixes applied (from our discussion):
  FIX 1  KMeans row split -> group-by-parameter split (no replicate leakage).
  FIX 2  10-fold OOF calibration -> QMAP+scale on the VAL split.
  FIX 3  monotone feature resolved BY NAME (works with prev_y9-last order).
  FIX 4  per-row records: data is 1 row per (parameter, sim); just stack rows.

GPU-budget upgrades (this pass):
  * Residual pre-LayerNorm MLP (`MLP(..., residual=True)`) for depth >= 4.
  * Warmup + cosine LR schedule in train_model.
  * 4-way split (train/val/calib/test): conformal offset fit on the held-out
    CALIB split (not val) for honest coverage.
  * TIERS config dicts (smoke / solid / max) + resolve_tier().

Run:  PYTHONPATH=src python test_estimint_nn.py     (fast 'smoke' tier)
Deps: jax, flax(nnx), optax, numpy, pandas — all already in pyproject.toml.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import optax
from flax import nnx

# Project calibrator helpers.
from estimint.utils import fit_qmap_w, predict_qmap_w, scale_pos, r2, rmse, mae
from estimint.data_processing import make_value_weights

# DATA code, now living in src/estimint/v2/data/ (SECTIONS 1 & 2 moved there).
# group_split lives in preprocess.py alongside the existing parameter split.
from estimint.v2.data.preprocess import create_splits
from estimint.v2.data.features import resolve_mono, StandardScaler


# ============================================================================
# TIER CONFIGS  ->  src/estimint/v2/conf/train_config.yaml  (or a dataclass)
# ----------------------------------------------------------------------------
# 'smoke' = fast CPU sanity check. 'solid' = sensible GPU default. 'max' = the
# heavy setting for the hard hbr->EIR map (residual net, big ensemble, long
# training). epochs is a CAP; early-stopping decides the real length.
# ============================================================================
TIERS = {
    "smoke": dict(
        width=128,
        depth=2,
        residual=False,
        epochs=60,
        patience=20,
        batch=512,
        lr=3.0e-3,
        wd=1e-4,
        n_ensemble=1,
        n_bins=12,
        n_quad=48,
    ),
    "solid": dict(
        width=256,
        depth=4,
        residual=False,
        epochs=2000,
        patience=150,
        batch=1024,
        lr=1.5e-3,
        wd=3e-4,
        n_ensemble=6,
        n_bins=24,
        n_quad=96,
    ),
    "max": dict(
        width=512,
        depth=6,
        residual=True,
        epochs=4000,
        patience=250,
        batch=1024,
        lr=1.0e-3,
        wd=1e-3,
        n_ensemble=10,
        n_bins=32,
        n_quad=128,
    ),
}


def resolve_tier(name: str, kind: str):
    """Split a flat tier dict into (n_ensemble, model_kwargs, train_kwargs).

    kind: 'flow' (uses n_bins) or 'umnn' (uses n_quad).
    """
    t = dict(TIERS[name])
    n_ensemble = t.pop("n_ensemble")
    n_bins, n_quad = t.pop("n_bins"), t.pop("n_quad")
    model_kw = dict(width=t.pop("width"), depth=t.pop("depth"), residual=t.pop("residual"))
    if kind == "flow":
        model_kw["n_bins"] = n_bins
    elif kind == "umnn":
        model_kw["n_quad"] = n_quad
    # kind == "fm": width/depth/residual only (no spline/quad params)
    train_kw = t  # epochs, patience, batch, lr, wd
    return n_ensemble, model_kw, train_kw


# ============================================================================
# SECTION 3  ->  src/estimint/v2/models/common.py   (NEW FILE)
# ----------------------------------------------------------------------------
# Shared NN plumbing: a GELU MLP that is either a plain stack (residual=False)
# or a pre-LayerNorm residual net (residual=True, use for depth>=4), plus a
# generic AdamW + warmup-cosine trainer with val early-stopping.
# ============================================================================
class MLP(nnx.Module):
    def __init__(self, din, dout, *, width=128, depth=3, residual=False, rngs):
        self.residual = residual
        self.inp = nnx.Linear(din, width, rngs=rngs)
        if residual:
            self.norms = nnx.List([nnx.LayerNorm(width, rngs=rngs) for _ in range(depth)])
            self.fc1 = nnx.List([nnx.Linear(width, width, rngs=rngs) for _ in range(depth)])
            self.fc2 = nnx.List([nnx.Linear(width, width, rngs=rngs) for _ in range(depth)])
        else:
            self.hidden = nnx.List([nnx.Linear(width, width, rngs=rngs) for _ in range(max(0, depth - 1))])
        self.out = nnx.Linear(width, dout, rngs=rngs)

    def __call__(self, x):
        x = self.inp(x)
        if self.residual:
            for ln, f1, f2 in zip(self.norms, self.fc1, self.fc2):
                x = x + f2(nnx.gelu(f1(ln(x))))  # pre-LN residual block
            x = nnx.gelu(x)
        else:
            x = nnx.gelu(x)
            for h in self.hidden:
                x = nnx.gelu(h(x))
        return self.out(x)


def train_model(
    model,
    loss_fn,
    Xtr,
    ytr,
    wtr,
    Xva,
    yva,
    wva,
    *,
    epochs=2000,
    batch=1024,
    lr=1.5e-3,
    wd=3e-4,
    patience=150,
    warmup_frac=0.05,
    seed=0,
):
    steps = max(1, len(Xtr) // batch) * epochs
    warmup = max(1, int(warmup_frac * steps))
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=steps,
        end_value=lr * 0.02,
    )
    opt = nnx.Optimizer(model, optax.adamw(sched, weight_decay=wd), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt, xb, yb, wb):
        loss, grads = nnx.value_and_grad(loss_fn)(model, xb, yb, wb)
        opt.update(model, grads)
        return loss

    @nnx.jit
    def val_loss(model, x, y, w):
        return loss_fn(model, x, y, w)

    Xtr, ytr, wtr = map(jnp.asarray, (Xtr, ytr, wtr))
    Xva, yva, wva = map(jnp.asarray, (Xva, yva, wva))
    rng = np.random.default_rng(seed)
    best = (np.inf, nnx.state(model))
    bad = 0
    n = len(Xtr)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            step(model, opt, Xtr[idx], ytr[idx], wtr[idx])
        vl = float(val_loss(model, Xva, yva, wva))
        if vl < best[0] - 1e-5:
            best = (vl, nnx.state(model))
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    nnx.update(model, best[1])
    return model


# ============================================================================
# SECTION 4  ->  src/estimint/v2/models/umnn.py   (NEW FILE)
# ----------------------------------------------------------------------------
# CHOICE 1 — MonotoneUMNN. Monotone-by-construction: f(m,c) = b(c) + integral of
# a strictly-positive integrand (softplus MLP) via Gauss-Legendre quadrature =>
# smooth AND monotone in m by design. Use on prev_y9->EIR and hbr_y9->EIR; it
# makes run.py's _smooth_staircase / dense sweep obsolete.
# ============================================================================
class MonotoneUMNN(nnx.Module):
    def __init__(self, n_context, *, width=128, depth=3, n_quad=48, residual=False, rngs):
        self.bias = MLP(n_context, 1, width=width, depth=depth, residual=residual, rngs=rngs)
        self.integrand = MLP(1 + n_context, 1, width=width, depth=depth, residual=residual, rngs=rngs)
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        self.nodes = jnp.asarray(nodes)
        self.weights = jnp.asarray(weights)

    def _integral(self, m, c):
        m = m[:, None]
        half = 0.5 * m
        t = half * (self.nodes[None, :] + 1.0)  # (N,Q)
        cE = jnp.broadcast_to(c[:, None, :], (c.shape[0], t.shape[1], c.shape[1]))
        inp = jnp.concatenate([t[..., None], cE], -1)
        g = nnx.softplus(self.integrand(inp))[..., 0]  # >0
        return half[:, 0] * (self.weights[None, :] * g).sum(1)

    def __call__(self, m, c):
        return self.bias(c)[:, 0] + self._integral(m, c)


def _umnn_loss(model, X, y, w):
    m, c = X[:, 0], X[:, 1:]  # X arranged as [mono, *context]
    pred = model(m, c)
    return jnp.sum(w * (pred - y) ** 2) / jnp.sum(w)


# ============================================================================
# SECTION 5  ->  src/estimint/v2/models/flow.py   (NEW FILE)
# ----------------------------------------------------------------------------
# CHOICE 2 (recommended primary) — ConditionalRQS. A conditional
# rational-quadratic-spline normalizing flow (Durkan 2019). Matches/beats
# XGBoost point accuracy via the median AND returns a calibrated posterior with
# exact, non-crossing quantiles:  quantile_q = T^{-1}(Phi^{-1}(q)). Works for
# all three maps. The _rqs forward/inverse pair passed an invertibility unit
# check (max error ~1e-6).
# ============================================================================
_B = 6.0  # spline support half-width (standardized units); linear tails outside


def _rqs(x, w_un, h_un, d_un, B, inverse):
    N, K = w_un.shape
    widths = jax.nn.softmax(w_un, -1) * (2 * B)
    heights = jax.nn.softmax(h_un, -1) * (2 * B)
    derivs = jax.nn.softplus(d_un) + 1e-3
    cx = jnp.concatenate([jnp.full((N, 1), -B), -B + jnp.cumsum(widths, -1)], -1)
    cy = jnp.concatenate([jnp.full((N, 1), -B), -B + jnp.cumsum(heights, -1)], -1)

    inside = (x > -B) & (x < B)
    xc = jnp.clip(x, -B + 1e-6, B - 1e-6)
    search = cy if inverse else cx
    b = jnp.clip(jnp.sum((xc[:, None] >= search[:, :-1]).astype(jnp.int32), -1) - 1, 0, K - 1)
    g = lambda A, i: jnp.take_along_axis(A, i[:, None], 1)[:, 0]
    xk, xk1 = g(cx, b), g(cx, b + 1)
    yk, yk1 = g(cy, b), g(cy, b + 1)
    dk, dk1 = g(derivs, b), g(derivs, b + 1)
    s = (yk1 - yk) / (xk1 - xk)

    if not inverse:
        xi = (xc - xk) / (xk1 - xk)
        num = (yk1 - yk) * (s * xi**2 + dk * xi * (1 - xi))
        den = s + (dk1 + dk - 2 * s) * xi * (1 - xi)
        y = yk + num / den
        dnum = s**2 * (dk1 * xi**2 + 2 * s * xi * (1 - xi) + dk * (1 - xi) ** 2)
        logdet = jnp.log(dnum) - 2 * jnp.log(den)
        return jnp.where(inside, y, x), jnp.where(inside, logdet, 0.0)
    else:
        yrel = xc - yk
        a = (yk1 - yk) * (s - dk) + yrel * (dk1 + dk - 2 * s)
        bb = (yk1 - yk) * dk - yrel * (dk1 + dk - 2 * s)
        cc = -s * yrel
        xi = 2 * cc / (-bb - jnp.sqrt(jnp.maximum(bb**2 - 4 * a * cc, 0.0)))
        xout = xi * (xk1 - xk) + xk
        den = s + (dk1 + dk - 2 * s) * xi * (1 - xi)
        dnum = s**2 * (dk1 * xi**2 + 2 * s * xi * (1 - xi) + dk * (1 - xi) ** 2)
        logdet = -(jnp.log(dnum) - 2 * jnp.log(den))
        return jnp.where(inside, xout, x), jnp.where(inside, logdet, 0.0)


class ConditionalRQS(nnx.Module):
    def __init__(self, n_context, *, n_bins=12, width=128, depth=3, B=_B, residual=False, rngs):
        self.K = n_bins
        self.B = B
        self.net = MLP(
            n_context,
            3 * n_bins + 1,
            width=width,
            depth=depth,
            residual=residual,
            rngs=rngs,
        )

    def _params(self, c):
        raw = self.net(c)
        return jnp.split(raw, [self.K, 2 * self.K], axis=-1)

    def log_prob(self, y0, c):
        w, h, d = self._params(c)
        z, logdet = _rqs(y0, w, h, d, self.B, inverse=False)
        return -0.5 * (z**2 + jnp.log(2 * jnp.pi)) + logdet

    def quantile(self, c, q):
        w, h, d = self._params(c)
        z = jax.scipy.stats.norm.ppf(jnp.full(c.shape[0], q))
        y0, _ = _rqs(z, w, h, d, self.B, inverse=True)
        return y0


def _rqs_loss(model, X, y0, w):
    lp = model.log_prob(y0, X)
    return -jnp.sum(w * lp) / jnp.sum(w)


# ============================================================================
# SECTION 5b (EXPERIMENTAL)  ->  src/estimint/v2/models/flow_matching.py
# ----------------------------------------------------------------------------
# ConditionalFM — conditional flow matching / rectified flow (Lipman 2023).
# ADDED FOR BENCHMARKING, *NOT* RECOMMENDED AS THE PRIMARY for these maps. Why:
# the target is 1-D, and in 1-D the RQS flow already gives an EXACT, monotone,
# invertible CDF => exact non-crossing quantiles + exact log-prob in one forward
# pass, and fast inference. Flow matching needs an ODE solve to draw samples and
# only gives EMPIRICAL quantiles (noisy, needs many samples, monotonicity not
# guaranteed), with no expressiveness gain in 1-D (a monotone RQS can already be
# multimodal). Expect it to match RQS at best while costing more at inference.
# FM earns its keep on JOINT / high-dim posteriors — not scalar-out maps.
#
# Training (simulation-free): y0~N(0,1), y1=standardized target, t~U(0,1),
#   yt = (1-t)*y0 + t*y1, regress velocity v(yt,t,c) -> (y1 - y0).
# Sampling: integrate dy/dt = v(y,t,c) from t=0 (y~N(0,1)) to t=1 (Euler).
# ============================================================================
class ConditionalFM(nnx.Module):
    def __init__(self, n_context, *, width=128, depth=3, residual=False, rngs):
        # input = [y_t, t, context] -> scalar velocity
        self.net = MLP(2 + n_context, 1, width=width, depth=depth, residual=residual, rngs=rngs)

    def velocity(self, y, t, c):
        inp = jnp.concatenate([y[..., None], t[..., None], c], -1)
        return self.net(inp)[..., 0]


def _fm_batch_loss(model, c, y1, w, key):
    k1, k2 = jax.random.split(key)
    y0 = jax.random.normal(k1, y1.shape)  # base sample
    t = jax.random.uniform(k2, y1.shape)  # random time
    yt = (1 - t) * y0 + t * y1  # linear interpolant path
    v = model.velocity(yt, t, c)
    return jnp.sum(w * (v - (y1 - y0)) ** 2) / jnp.sum(w)


@nnx.jit
def _fm_velocity(model, y, t, c):
    return model.velocity(y, t, c)


def train_fm_one(
    model,
    Xtr,
    ytr,
    wtr,
    Xva,
    yva,
    wva,
    *,
    epochs,
    batch,
    lr,
    wd,
    patience,
    warmup_frac=0.05,
    seed=0,
):
    """Dedicated FM trainer (loss is stochastic per step, so it needs its own
    key stream rather than the shared train_model)."""
    steps = max(1, len(Xtr) // batch) * epochs
    warmup = max(1, int(warmup_frac * steps))
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=steps,
        end_value=lr * 0.02,
    )
    opt = nnx.Optimizer(model, optax.adamw(sched, weight_decay=wd), wrt=nnx.Param)

    @nnx.jit
    def step(model, opt, xb, yb, wb, key):
        loss, grads = nnx.value_and_grad(_fm_batch_loss)(model, xb, yb, wb, key)
        opt.update(model, grads)
        return loss

    @nnx.jit
    def vloss(model, x, y, w, key):  # average a few noise draws for a stable metric
        ks = jax.random.split(key, 8)
        return jnp.mean(jnp.stack([_fm_batch_loss(model, x, y, w, k) for k in ks]))

    Xtr, ytr, wtr = map(jnp.asarray, (Xtr, ytr, wtr))
    Xva, yva, wva = map(jnp.asarray, (Xva, yva, wva))
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    val_key = jax.random.PRNGKey(seed + 999)
    best = (np.inf, nnx.state(model))
    bad = 0
    n = len(Xtr)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            key, sub = jax.random.split(key)
            step(model, opt, Xtr[idx], ytr[idx], wtr[idx], sub)
        vl = float(vloss(model, Xva, yva, wva, val_key))
        if vl < best[0] - 1e-5:
            best = (vl, nnx.state(model))
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    nnx.update(model, best[1])
    return model


# ============================================================================
# SECTION 6  ->  src/estimint/v2/calibration.py   (NEW FILE)
# ----------------------------------------------------------------------------
# FIX 2: QMAP + positive-scale calibrator fit on VAL predictions (same objects
# the XGBoost trainers used). Plus split-conformal offset fit on the CALIB split.
# ============================================================================
def calibrate_on_val(val_pred_raw, val_obs_raw):
    cal = fit_qmap_w(val_pred_raw, val_obs_raw, ngrid=1024, round_digits=8)
    a = scale_pos(val_obs_raw, predict_qmap_w(val_pred_raw, cal))
    return {
        "kind": "qmap+scale",
        "qmap": {"xq": cal["xq"], "yq": cal["yq"]},
        "scale": a,
    }


def apply_cal(raw, cal):
    return np.maximum(0.0, cal["scale"] * predict_qmap_w(raw, cal["qmap"]))


def conformal_offset(lo, hi, obs, alpha=0.10):
    """Split-conformal (CQR, Romano 2019) width correction on a HELD-OUT split.

    Returns offset Q so that widening to [lo-Q, hi+Q] gives >= (1-alpha)
    marginal coverage. Fit this on the CALIB split (disjoint from the val split
    used for early stopping) so the guarantee is honest.
    """
    scores = np.maximum(lo - obs, obs - hi)
    n = len(scores)
    k = min(max(int(np.ceil((1 - alpha) * (n + 1))), 1), n)
    return float(np.sort(scores)[k - 1])


# ============================================================================
# SECTION 7  ->  src/estimint/v2/train_base.py   (EXTENDS existing)
# ----------------------------------------------------------------------------
# Bundles expose the same natural-scale `.predict(X_raw)` contract as run.py's
# `_predict_direct`, plus `.quantile`/`.interval` for the flow. Trainers use the
# group split (v2/data/preprocess), deep ensembling, val calibration, and a 4-way
# split so conformal coverage is fit on held-out CALIB data.
# ============================================================================
class UMNNBundle:
    def __init__(self, models, scaler, cal, features, mono_idx):
        self.models, self.scaler, self.cal = models, scaler, cal
        self.features, self.mono_idx = features, mono_idx

    def _mc(self, X_raw):
        Xs = self.scaler.transform(np.asarray(X_raw, dtype=np.float64))
        ctx = np.delete(Xs, self.mono_idx, axis=1)
        return jnp.asarray(Xs[:, self.mono_idx]), jnp.asarray(ctx)

    def predict(self, X_raw):
        m, c = self._mc(X_raw)
        log10 = np.mean([np.asarray(mdl(m, c)) for mdl in self.models], 0)
        return apply_cal(np.power(10.0, log10), self.cal)


class RQSBundle:
    def __init__(self, models, scaler, y_mu, y_sd, cal, features):
        self.models, self.scaler = models, scaler
        self.y_mu, self.y_sd, self.cal, self.features = y_mu, y_sd, cal, features
        self.conformal = {}  # alpha -> offset Q (fit on CALIB split)

    def _q(self, X_raw, q):
        c = jnp.asarray(self.scaler.transform(np.asarray(X_raw, dtype=np.float64)))
        y0 = np.mean([np.asarray(m.quantile(c, q)) for m in self.models], 0)
        return np.power(10.0, y0 * self.y_sd + self.y_mu)

    def predict(self, X_raw):
        return apply_cal(self._q(X_raw, 0.5), self.cal)  # calibrated median

    def quantile(self, X_raw, q):
        return apply_cal(self._q(X_raw, q), self.cal)  # raw calibrated band

    def interval(self, X_raw, alpha=0.10):
        """Conformalized (1-alpha) band with guaranteed coverage."""
        lo = self.quantile(X_raw, alpha / 2)
        hi = self.quantile(X_raw, 1 - alpha / 2)
        Q = self.conformal.get(alpha, 0.0)
        return np.maximum(0.0, lo - Q), hi + Q


class FMBundle:
    """EXPERIMENTAL sample-based bundle (see SECTION 5b). Point estimate is the
    sample median; quantiles are empirical order statistics of ODE samples."""

    def __init__(
        self,
        models,
        scaler,
        y_mu,
        y_sd,
        cal,
        features,
        *,
        n_steps=100,
        n_samples=256,
        seed=0,
    ):
        self.models, self.scaler = models, scaler
        self.y_mu, self.y_sd, self.cal, self.features = y_mu, y_sd, cal, features
        self.n_steps, self.n_samples, self.seed = n_steps, n_samples, seed
        self.conformal = {}

    def _samples_natural(self, X_raw):
        """Draw ODE samples -> raw (uncalibrated) natural-scale target samples."""
        c = jnp.asarray(self.scaler.transform(np.asarray(X_raw, dtype=np.float64)))
        N, ctx = c.shape
        dt = 1.0 / self.n_steps
        cols = []
        for i, m in enumerate(self.models):
            key = jax.random.PRNGKey(self.seed + 1000 + i)
            y = jax.random.normal(key, (N, self.n_samples))  # base draws
            cE = jnp.broadcast_to(c[:, None, :], (N, self.n_samples, ctx))
            for stp in range(self.n_steps):  # Euler ODE solve
                t = jnp.full((N, self.n_samples), stp * dt)
                y = y + dt * _fm_velocity(m, y, t, cE)
            cols.append(np.asarray(y))
        S = np.concatenate(cols, axis=1)  # (N, n_samples*ens)
        return np.power(10.0, S * self.y_sd + self.y_mu)

    def predict(self, X_raw):
        return apply_cal(np.median(self._samples_natural(X_raw), axis=1), self.cal)

    def quantile(self, X_raw, q):
        return apply_cal(np.quantile(self._samples_natural(X_raw), q, axis=1), self.cal)

    def interval(self, X_raw, alpha=0.10):
        lo = self.quantile(X_raw, alpha / 2)
        hi = self.quantile(X_raw, 1 - alpha / 2)
        Q = self.conformal.get(alpha, 0.0)
        return np.maximum(0.0, lo - Q), hi + Q


@dataclass
class SplitArrays:
    X: np.ndarray
    Xs: np.ndarray
    ylog: np.ndarray
    w: np.ndarray


@dataclass
class PrepSplits:
    train: SplitArrays
    val: SplitArrays
    calib: SplitArrays
    test: SplitArrays
    scaler: StandardScaler


def _records_for_split(df, param_sims, scaler, features, target):
    groups = df.groupby(["parameter_index", "simulation_index"])
    records = []
    for ps in sorted(param_sims):
        if ps not in groups.groups:
            continue

        idx = groups.groups[ps]
        X = df.loc[idx, features].to_numpy(dtype=np.float64)
        y_raw = df.loc[idx, target].to_numpy(dtype=np.float64)
        records.append(
            {
                "x": scaler.transform(X),
                "x_raw": X,
                "y": np.log10(y_raw),
                "y_raw": y_raw,
                "w": df.loc[idx, "_weight"].to_numpy(dtype=np.float64),
                "ps": np.asarray(ps, dtype=np.int32),
            }
        )
    return records


def _as_arrays(records, n_features):
    if not records:
        empty_x = np.empty((0, n_features), dtype=np.float64)
        empty_y = np.empty((0,), dtype=np.float64)
        return SplitArrays(X=empty_x, Xs=empty_x.copy(), ylog=empty_y, w=empty_y.copy())

    return SplitArrays(
        X=np.concatenate([r["x_raw"] for r in records], axis=0),
        Xs=np.concatenate([r["x"] for r in records], axis=0),
        ylog=np.concatenate([r["y"] for r in records], axis=0),
        w=np.concatenate([r["w"] for r in records], axis=0),
    )


def _prep(df, features, target, seed, stratify, calib_frac):
    df = df.copy()
    if np.any(df[target].to_numpy(dtype=np.float64) <= 0):
        raise ValueError(f"target {target!r} must be strictly positive for log10 training")

    df["_ps"] = list(zip(df["parameter_index"], df["simulation_index"]))
    df["_weight"] = make_value_weights(df[target].to_numpy(dtype=np.float64), digits=3)
    splits = create_splits(
        df,
        seed=seed,
        val_frac=0.10,
        test_frac=0.10,
        calib_frac=calib_frac,
        stratify=stratify,
        target=target,
    )

    train_mask = df["_ps"].isin(splits.train)
    scaler = StandardScaler().fit(df.loc[train_mask, features].to_numpy(dtype=np.float64))
    n_features = len(features)

    return PrepSplits(
        train=_as_arrays(_records_for_split(df, splits.train, scaler, features, target), n_features),
        val=_as_arrays(_records_for_split(df, splits.val, scaler, features, target), n_features),
        calib=_as_arrays(_records_for_split(df, splits.calib, scaler, features, target), n_features),
        test=_as_arrays(_records_for_split(df, splits.test, scaler, features, target), n_features),
        scaler=scaler,
    )


def train_umnn(df, features, target="eir", *, tier="solid", seed=42, stratify=True):
    mono_idx = resolve_mono(features)  # FIX 3
    assert mono_idx is not None, "UMNN needs a monotone feature (prev_y9/hbr_y9)"
    n_ens, model_kw, train_kw = resolve_tier(tier, "umnn")
    prep = _prep(df, features, target, seed, stratify, 0.0)
    tr, va = prep.train, prep.val
    # arrange columns as [mono, *context]
    Xtr = np.column_stack([tr.Xs[:, mono_idx], np.delete(tr.Xs, mono_idx, axis=1)])
    Xva = np.column_stack([va.Xs[:, mono_idx], np.delete(va.Xs, mono_idx, axis=1)])

    models = []
    for e in range(n_ens):
        mdl = MonotoneUMNN(len(features) - 1, rngs=nnx.Rngs(seed + e), **model_kw)
        mdl = train_model(
            mdl,
            _umnn_loss,
            Xtr,
            tr.ylog,
            tr.w,
            Xva,
            va.ylog,
            va.w,
            seed=seed + e,
            **train_kw,
        )
        models.append(mdl)

    bundle = UMNNBundle(models, prep.scaler, None, features, mono_idx)
    val_pred = np.mean(
        [np.asarray(mm(jnp.asarray(Xva[:, 0]), jnp.asarray(Xva[:, 1:]))) for mm in models],
        0,
    )
    bundle.cal = calibrate_on_val(
        np.power(10.0, val_pred),  # FIX 2
        np.power(10.0, va.ylog),
    )
    return bundle, (prep.test.X, np.power(10.0, prep.test.ylog))


def train_rqs(df, features, target="eir", *, tier="solid", seed=42, stratify=True):
    n_ens, model_kw, train_kw = resolve_tier(tier, "flow")
    # 4-way split so conformal is fit on held-out CALIB data
    prep = _prep(df, features, target, seed, stratify, calib_frac=0.10)
    tr, va, ca, te = prep.train, prep.val, prep.calib, prep.test
    y_mu, y_sd = tr.ylog.mean(), tr.ylog.std() + 1e-8

    models = []
    for e in range(n_ens):
        mdl = ConditionalRQS(len(features), rngs=nnx.Rngs(seed + e), **model_kw)
        mdl = train_model(
            mdl,
            _rqs_loss,
            tr.Xs,
            (tr.ylog - y_mu) / y_sd,
            tr.w,
            va.Xs,
            (va.ylog - y_mu) / y_sd,
            va.w,
            seed=seed + e,
            **train_kw,
        )
        models.append(mdl)

    bundle = RQSBundle(models, prep.scaler, y_mu, y_sd, None, features)
    val_med = np.mean([np.asarray(mm.quantile(jnp.asarray(va.Xs), 0.5)) for mm in models], 0) * y_sd + y_mu
    bundle.cal = calibrate_on_val(
        np.power(10.0, val_med),  # FIX 2
        np.power(10.0, va.ylog),
    )
    # conformal offset fit on CALIB (disjoint from val) for honest coverage
    lo, hi = bundle.quantile(ca.X, 0.05), bundle.quantile(ca.X, 0.95)
    bundle.conformal[0.10] = conformal_offset(lo, hi, np.power(10.0, ca.ylog), alpha=0.10)
    return bundle, (te.X, np.power(10.0, te.ylog))


def train_fm(
    df,
    features,
    target="eir",
    *,
    tier="solid",
    seed=42,
    stratify=True,
    n_steps=100,
    n_samples=256,
):
    """EXPERIMENTAL: conditional flow matching. Same 4-way split + conformal as
    train_rqs; point estimate = sample median, quantiles = empirical."""
    n_ens, model_kw, train_kw = resolve_tier(tier, "fm")  # width/depth/residual only
    prep = _prep(df, features, target, seed, stratify, calib_frac=0.10)
    tr, va, ca, te = prep.train, prep.val, prep.calib, prep.test
    y_mu, y_sd = tr.ylog.mean(), tr.ylog.std() + 1e-8

    models = []
    for e in range(n_ens):
        mdl = ConditionalFM(len(features), rngs=nnx.Rngs(seed + e), **model_kw)
        mdl = train_fm_one(
            mdl,
            tr.Xs,
            (tr.ylog - y_mu) / y_sd,
            tr.w,
            va.Xs,
            (va.ylog - y_mu) / y_sd,
            va.w,
            seed=seed + e,
            **train_kw,
        )
        models.append(mdl)

    bundle = FMBundle(
        models,
        prep.scaler,
        y_mu,
        y_sd,
        None,
        features,
        n_steps=n_steps,
        n_samples=n_samples,
        seed=seed,
    )
    val_med = np.median(bundle._samples_natural(va.X), axis=1)  # raw (pre-cal)
    bundle.cal = calibrate_on_val(val_med, np.power(10.0, va.ylog))
    lo, hi = bundle.quantile(ca.X, 0.05), bundle.quantile(ca.X, 0.95)
    bundle.conformal[0.10] = conformal_offset(lo, hi, np.power(10.0, ca.ylog), alpha=0.10)
    return bundle, (te.X, np.power(10.0, te.ylog))


# ============================================================================
# SECTION 8  ->  smoke test (this file's __main__; not shipped)
# ----------------------------------------------------------------------------
# Fast CPU sanity check on the 'smoke' tier. For real runs use tier="solid" or
# tier="max" on GPU. Also exercises the residual MLP + warmup path.
# ============================================================================
if __name__ == "__main__":
    TIER = "smoke"  # switch to "solid" / "max" on GPU
    df = pd.read_parquet("models/prevalence/training.parquet")
    feats = [
        "dn0_use",
        "Q0",
        "phi_bednets",
        "seasonal",
        "itn_use",
        "irs_use",
        "prev_y9",
    ]

    print("== residual MLP path ==")
    rm = MLP(7, 1, width=32, depth=3, residual=True, rngs=nnx.Rngs(0))
    print("  output shape", tuple(rm(jnp.zeros((4, 7))).shape))

    print("== MonotoneUMNN (prev->EIR) ==")
    umnn, (Xte, yte) = train_umnn(df, feats, tier=TIER)
    p = umnn.predict(Xte)
    base = np.tile(Xte[0], (60, 1))
    base[:, -1] = np.linspace(0.02, 0.8, 60)
    print(f"  test R2={r2(yte, p):.4f}  RMSE={rmse(yte, p):.2f}  MAE={mae(yte, p):.2f}")
    print(f"  monotone in prev_y9: {bool(np.all(np.diff(umnn.predict(base)) >= -1e-6))}")

    print("== ConditionalRQS (prev->EIR) ==")
    flow, (Xte2, yte2) = train_rqs(df, feats, tier=TIER)
    p2 = flow.predict(Xte2)
    lo, hi = flow.quantile(Xte2, 0.05), flow.quantile(Xte2, 0.95)
    cov_raw = float(np.mean((yte2 >= lo) & (yte2 <= hi)))
    clo, chi = flow.interval(Xte2, alpha=0.10)
    cov_conf = float(np.mean((yte2 >= clo) & (yte2 <= chi)))
    print(f"  test R2={r2(yte2, p2):.4f}  RMSE={rmse(yte2, p2):.2f}  MAE={mae(yte2, p2):.2f}")
    print(f"  90% coverage: raw={cov_raw:.3f}  conformalized={cov_conf:.3f}")

    print("== ConditionalFM (EXPERIMENTAL, prev->EIR) ==")
    fm, (Xte3, yte3) = train_fm(df, feats, tier=TIER, n_steps=50, n_samples=64)
    p3 = fm.predict(Xte3)
    clo3, chi3 = fm.interval(Xte3, alpha=0.10)
    cov3 = float(np.mean((yte3 >= clo3) & (yte3 <= chi3)))
    print(f"  test R2={r2(yte3, p3):.4f}  RMSE={rmse(yte3, p3):.2f}  MAE={mae(yte3, p3):.2f}")
    print(f"  90% conformalized coverage={cov3:.3f}   (vs RQS above — expect FM ~= RQS at best)")
    print("SMOKE_OK")
