"""
Utility functions for estiMINT package.

Equivalent to: utils.R
"""

import sys
from datetime import datetime
from typing import  Dict, Any

import numpy as np
from numpy.typing import ArrayLike


def ts(*args) -> None:
    """
    Print timestamped message to console.

    Equivalent to R's ts() function.

    Parameters
    ----------
    *args : str
        Format string and arguments (like sprintf)
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    if len(args) == 1:
        message = args[0]
    else:
        message = args[0] % args[1:]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


def r2(y: ArrayLike, yhat: ArrayLike) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Equivalent to R's r2() function.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        R-squared value
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    ssr = np.sum((y - yhat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return 1 - ssr / sst


def rmse(y: ArrayLike, yhat: ArrayLike) -> float:
    """
    Calculate Root Mean Squared Error.

    Equivalent to R's rmse() function.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        RMSE value
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.sqrt(np.mean((y - yhat) ** 2))


def mse(y: ArrayLike, yhat: ArrayLike) -> float:
    """
    Calculate Mean Squared Error.

    Equivalent to R's mse() function.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        MSE value
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.mean((y - yhat) ** 2)


def mae(y: ArrayLike, yhat: ArrayLike) -> float:
    """
    Calculate Mean Absolute Error.

    Equivalent to R's mae() function.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        MAE value
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.mean(np.abs(y - yhat))

def medape(y: ArrayLike, yhat: ArrayLike) -> float:
    """
    Calculate Median Absolute Percentage Error.

    Equivalent to R's medape() function.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        Median APE value
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.median(np.abs((y - yhat) / np.maximum(1, y))) * 100

def bias(y: ArrayLike, yhat: ArrayLike) -> float:
    """
    Calculate bias (mean error).

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        Bias value (mean of yhat - y)
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.mean(yhat - y)

def median_ae(y: ArrayLike, yhat: ArrayLike) -> float:
    """
    Calculate Median Absolute Error.

    Equivalent to R's median_ae() function.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        Median AE value
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.median(np.abs(y - yhat))


def mae_rel(y: ArrayLike, yhat: ArrayLike) -> float:
    """
    Calculate Relative Median Absolute Error.

    Equivalent to R's mae_rel() function.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        Relative MAE value (median of abs(yhat-y) / max(1, y))
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.median(np.abs(yhat - y) / np.maximum(1, y))


def rmsle(y: ArrayLike, yhat: ArrayLike) -> float:
    """
    Calculate Root Mean Squared Log Error.

    Equivalent to R's rmsle() function.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        RMSLE value
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.sqrt(np.mean((np.log1p(yhat) - np.log1p(y)) ** 2))


def safe_div(num: ArrayLike, den: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """
    Safe division with epsilon floor on denominator.

    Equivalent to R's safe_div() function.

    Parameters
    ----------
    num : array-like
        Numerator
    den : array-like
        Denominator
    eps : float, optional
        Minimum value for denominator (default: 1e-12)

    Returns
    -------
    np.ndarray
        Result of num / max(eps, den)
    """
    num = np.asarray(num)
    den = np.asarray(den)
    return num / np.maximum(eps, den)


def smape(y: ArrayLike, yhat: ArrayLike, eps: float = 1e-12) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    Equivalent to R's smape() function.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values
    eps : float, optional
        Epsilon for numerical stability (default: 1e-12)

    Returns
    -------
    float
        sMAPE value
    """
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return np.mean(2 * np.abs(yhat - y) / np.maximum(eps, np.abs(y) + np.abs(yhat)))


def fit_qmap_w(
    pred_raw: ArrayLike,
    obs_raw: ArrayLike,
    ngrid: int = 1024,
    round_digits: int = 8
) -> Dict[str, Any]:
    """
    Fit weighted quantile mapping calibration.

    Equivalent to R's fit_qmap_w() function.

    Parameters
    ----------
    pred_raw : array-like
        Raw predictions
    obs_raw : array-like
        Observed values
    ngrid : int, optional
        Number of grid points for quantile mapping (default: 1024)
    round_digits : int, optional
        Digits for rounding observed values (default: 8)

    Returns
    -------
    dict
        Dictionary with keys: 'kind', 'xq', 'yq'
    """
    pred_raw = np.asarray(pred_raw)
    obs_raw = np.asarray(obs_raw)

    # Keep only finite values
    keep = np.isfinite(pred_raw) & np.isfinite(obs_raw)
    x = pred_raw[keep]
    y = obs_raw[keep]

    # Sort predictions and compute empirical CDF
    o1 = np.argsort(x)
    x1 = x[o1]
    F1 = (np.arange(1, len(x1) + 1) - 0.5) / len(x1)

    # Weighted CDF for observations
    y_key = np.round(y, round_digits)
    unique_y, counts = np.unique(y_key, return_counts=True)

    o2 = np.argsort(unique_y)
    y2 = unique_y[o2]
    w2 = counts[o2]
    F2 = np.cumsum(w2) / np.sum(w2)

    # Interpolate quantiles
    q = np.linspace(0, 1, ngrid)
    xq = np.interp(q, F1, x1)
    yq = np.interp(q, F2, y2)

    # Enforce strict monotonicity to prevent flat spots in the mapping
    rel_eps = 1e-10
    for i in range(1, len(xq)):
        if xq[i] <= xq[i - 1]:
            xq[i] = xq[i - 1] + rel_eps * max(1.0, abs(xq[i - 1]))
    for i in range(1, len(yq)):
        if yq[i] <= yq[i - 1]:
            yq[i] = yq[i - 1] + rel_eps * max(1.0, abs(yq[i - 1]))

    return {"kind": "qmap", "xq": xq, "yq": yq}


def predict_qmap_w(newx_raw: ArrayLike, cal: Dict[str, Any]) -> np.ndarray:
    """
    Apply quantile mapping calibration to new predictions.

    Equivalent to R's predict_qmap_w() function.

    Parameters
    ----------
    newx_raw : array-like
        New raw predictions to calibrate
    cal : dict
        Calibration object from fit_qmap_w()

    Returns
    -------
    np.ndarray
        Calibrated predictions
    """
    newx_raw = np.asarray(newx_raw, dtype=np.float64)
    xq = np.array(cal["xq"], dtype=np.float64)
    yq = np.array(cal["yq"], dtype=np.float64)

    # np.interp clamps out-of-range values to the boundary, which destroys
    # ordering for predictions below xq[0] or above xq[-1].  Instead,
    # linearly extrapolate using the slope of the first/last grid segment
    # so that distinct raw predictions always yield distinct calibrated values.
    result = np.interp(newx_raw, xq, yq)

    # Left extrapolation: values below xq[0]
    below = newx_raw < xq[0]
    if np.any(below):
        slope_left = (yq[1] - yq[0]) / (xq[1] - xq[0]) if xq[1] != xq[0] else 0.0
        result[below] = yq[0] + slope_left * (newx_raw[below] - xq[0])

    # Right extrapolation: values above xq[-1]
    above = newx_raw > xq[-1]
    if np.any(above):
        slope_right = (yq[-1] - yq[-2]) / (xq[-1] - xq[-2]) if xq[-1] != xq[-2] else 0.0
        result[above] = yq[-1] + slope_right * (newx_raw[above] - xq[-1])

    return result


def scale_pos(obs: ArrayLike, pred: ArrayLike) -> float:
    """
    Calculate positive scaling factor.

    Equivalent to R's scale_pos() function.

    Parameters
    ----------
    obs : array-like
        Observed values
    pred : array-like
        Predicted values

    Returns
    -------
    float
        Scaling factor a = sum(obs * pred) / sum(pred^2)
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    a = np.sum(obs * pred) / np.sum(pred ** 2)
    if not np.isfinite(a) or a <= 0:
        a = 1.0
    return a
