"""
Model inference functions for estiMINT package.

Equivalent to: run.R
"""

from typing import Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import xgboost as xgb

from .utils import predict_qmap_w


# Global model storage (equivalent to R's .GlobalEnv)
_global_model: Optional[Dict[str, Any]] = None


def set_global_model(model: Dict[str, Any]) -> None:
    """
    Set the global estiMINT model.
    
    Parameters
    ----------
    model : dict
        An 'estiMINT_model' object
    """
    global _global_model
    _global_model = model


def get_global_model() -> Optional[Dict[str, Any]]:
    """
    Get the global estiMINT model.
    
    Returns
    -------
    dict or None
        The global model if set, None otherwise
    """
    return _global_model


def run_xgb_model(
    new_data: Union[pd.DataFrame, Dict[str, Any]],
    model: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Run XGBoost model with initial conditions.
    
    Equivalent to R's run_xgb_model() function.
    
    Parameters
    ----------
    new_data : pd.DataFrame or dict
        Data frame with columns: prevalence (or prev_y9), dn0_use, Q0, 
        phi_bednets, seasonal, itn_use, irs_use
    model : dict, optional
        An 'estiMINT_model' object; if None, tries global 'estiMINT_model'
        
    Returns
    -------
    np.ndarray
        Numeric array of calibrated EIR predictions
        
    Raises
    ------
    ValueError
        If no model is provided and no global model exists
    ValueError
        If required columns are missing from new_data
    """
    # Get model
    if model is None:
        model = get_global_model()
        if model is None:
            raise ValueError(
                "No model provided and 'estiMINT_model' not found in the global context. "
                "Either pass a model or call set_global_model() first."
            )
    
    # Get required features
    req = model["features"]
    
    # Convert to DataFrame if necessary
    if isinstance(new_data, dict):
        nd = pd.DataFrame(new_data)
    else:
        nd = new_data.copy()
    
    # Handle prevalence -> prev_y9 alias
    if "prevalence" in nd.columns and "prev_y9" not in nd.columns:
        nd["prev_y9"] = nd["prevalence"]
    
    # Check for missing columns
    missing = set(req) - set(nd.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    
    # Extract feature matrix
    X = nd[req].values.astype(np.float64)
    
    # Create DMatrix and predict
    dnew = xgb.DMatrix(X)
    pred_log10 = model["booster"].predict(dnew)
    pred_raw = np.power(10, pred_log10)
    
    # Apply calibration
    pred_cal = predict_qmap_w(pred_raw, model["calibrator"]["qmap"])
    pred_final = np.maximum(0, model["calibrator"]["scale"] * pred_cal)
    
    return pred_final.astype(np.float64)
