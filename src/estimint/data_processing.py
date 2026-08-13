import numpy as np

def make_value_weights(eir_raw: np.ndarray, digits: int = 3) -> np.ndarray:
    """
    Create inverse-frequency weights based on EIR values.

    Equivalent to R's make_value_weights() function.

    Parameters
    ----------
    eir_raw : array-like
        Raw EIR values
    digits : int, optional
        Number of digits for rounding (default: 3)

    Returns
    -------
    np.ndarray
        Normalized weights (mean = 1)
    """
    eir_raw = np.asarray(eir_raw)
    key = np.round(eir_raw, digits)

    # Count frequency of each rounded value
    unique_vals, counts = np.unique(key, return_counts=True)
    freq_dict = dict(zip(unique_vals, counts))

    # Inverse frequency weights
    w = np.array([1.0 / freq_dict[k] for k in key])

    # Normalize to mean = 1
    w = w / np.mean(w)

    return w

