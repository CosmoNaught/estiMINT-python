import numpy as np
from ..common.types import PredictorType

FEATURES_BASE = ["dn0_use", "Q0", "phi_bednets", "seasonal", "itn_use", "irs_use"]

def get_features(predictor: PredictorType) -> list[str]:
    """
    Get the list of features based on the predictor and target.
    The predictor is is inserted at the beginning of the list of features.

    Args:
        predictor: The predictor type.
    Returns:
        A list of feature names.
    """
    return [predictor] + FEATURES_BASE


class StandardScaler:
    def __init__(self):
        """
        Initialize an unfitted scaler.

        Returns:
            None.
        """
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None and self.scale_ is not None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Fit feature means and scales.

        Args:
            X: Feature matrix.

        Returns:
            Fitted scaler.
        """
        self.mean_ = np.mean(X, axis=0)
        # To avoid division by zero, set scale to 1.0 for any feature with zero variance
        scale = np.std(X, axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features.

        Args:
            X: Feature matrix.

        Returns:
            Standardized features.
        """
        if not self.is_fitted:
            raise ValueError("StandardScaler instance is not fitted yet.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and standardize features.

        Args:
            X: Feature matrix.

        Returns:
            Standardized features.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Restore standardized features.

        Args:
            X: Standardized feature matrix.

        Returns:
            Features in the original scale.
        """
        if not self.is_fitted:
            raise ValueError("StandardScaler instance is not fitted yet.")
        return X * self.scale_ + self.mean_
