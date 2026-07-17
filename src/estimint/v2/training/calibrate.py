import numpy as np

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
