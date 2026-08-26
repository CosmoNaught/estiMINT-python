import numpy as np

def conformal_offset(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray, alpha: float = 0.10) -> float:
    """Split-conformal (CQR, Romano 2019) width correction.

    Returns offset Q so that widening to [lo-Q, hi+Q] gives >= (1-alpha)
    marginal coverage.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    scores = np.maximum(lower - y_true, y_true - upper)
    n = len(scores)
    k = np.ceil((n + 1) * (1 - alpha)).astype(int)
    return float(np.sort(scores)[k - 1])
