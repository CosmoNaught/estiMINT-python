"""
Global variables and constants for estiMINT package.

Equivalent to: globals.R

In R, globalVariables() silences CMD check warnings for NSE columns.
In Python, we define these as module-level constants for documentation
and type-checking purposes.
"""

from typing import List

# Column names used throughout the package (equivalent to R's globalVariables)
COLUMN_NAMES: List[str] = [
    "row_id",
    "true_value", 
    "case_range",
    "model",
    "prediction",
    "error",
    "true",
    "pred",
    "bin",
    "year",
]

# Feature column names
FEATURE_COLUMNS: List[str] = [
    "Feature",
    "Gain_scaled",
    "feature",
    "importance_scaled",
]

# Metric column names
METRIC_COLUMNS: List[str] = [
    "Model",
    "Quantile",
    "RMSE",
]

# Default features for EIR prediction
DEFAULT_FEATURES: List[str] = [
    "dn0_use",
    "Q0",
    "phi_bednets",
    "seasonal",
    "itn_use",
    "irs_use",
    "prev_y9",
]

# Default thresholds
DEFAULT_THR_LO: float = 0.02
DEFAULT_THR_HI: float = 0.95

# Default k-means strata
DEFAULT_K_STRATA: int = 16

# Default CV folds
DEFAULT_K_FOLDS: int = 10

# Default random seed
DEFAULT_SEED: int = 42
