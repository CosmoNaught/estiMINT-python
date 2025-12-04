"""
Model training functions for estiMINT package.

Equivalent to: models.R
"""

from typing import Dict, Any, Optional, Callable, List
import numpy as np
import xgboost as xgb
from numpy.typing import ArrayLike


def train_eir_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    tune_params: bool = True
) -> Dict[str, Any]:
    """
    Train XGBoost model for EIR prediction.
    
    Equivalent to R's train_eir_xgboost() function.
    
    Parameters
    ----------
    X_train : np.ndarray
        Numeric matrix of training features
    y_train : np.ndarray
        Numeric vector of training targets
    X_val : np.ndarray, optional
        Numeric matrix of validation features (default: None)
    y_val : np.ndarray, optional
        Numeric vector of validation targets (default: None)
    tune_params : bool, optional
        Whether to tune hyperparameters (default: True)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'model': trained XGBoost model
        - 'params': best parameters
        - 'nrounds': number of training rounds
        - 'importance': feature importance DataFrame
        - 'transform': function to transform target (log10(y + 1))
        - 'inverse_transform': inverse transform function (10^y - 1)
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    
    # Transform target
    y_train_log = np.log10(y_train + 1)
    dtrain = xgb.DMatrix(data=X_train, label=y_train_log)
    
    # Build watchlist
    evals = [(dtrain, "train")]
    if X_val is not None and y_val is not None:
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        dval = xgb.DMatrix(data=X_val, label=np.log10(y_val + 1))
        evals.append((dval, "eval"))
    
    base_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "gamma": 0.1,
        "alpha": 0.1,
        "lambda": 1.0,
        "seed": 42,
    }
    
    if tune_params:
        best_rmse = float("inf")
        best_params = base_params.copy()
        best_nrounds = 100
        
        for depth in [3, 4, 5]:
            for eta in [0.01, 0.05, 0.1]:
                for subsample in [0.6, 0.7, 0.8]:
                    params = base_params.copy()
                    params.update({
                        "max_depth": depth,
                        "eta": eta,
                        "subsample": subsample,
                    })
                    
                    cv_results = xgb.cv(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=500,
                        nfold=5,
                        early_stopping_rounds=20,
                        verbose_eval=False,
                        seed=42,
                    )
                    
                    # Get best iteration metrics
                    best_iter = len(cv_results) - 1
                    cv_rmse = cv_results["test-rmse-mean"].iloc[best_iter]
                    
                    if cv_rmse < best_rmse:
                        best_rmse = cv_rmse
                        best_params = params.copy()
                        best_nrounds = best_iter + 1
        
        params = best_params
        nrounds = best_nrounds
        print(
            f"Best XGBoost: depth={params['max_depth']}, "
            f"eta={params['eta']:.3f}, subsample={params['subsample']:.2f}, "
            f"nrounds={nrounds}, CV-RMSE={best_rmse:.4f}"
        )
    else:
        cv_results = xgb.cv(
            params=base_params,
            dtrain=dtrain,
            num_boost_round=500,
            nfold=5,
            early_stopping_rounds=20,
            verbose_eval=False,
            seed=42,
        )
        params = base_params
        nrounds = len(cv_results)
    
    # Train final model
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=nrounds,
        evals=evals,
        verbose_eval=False,
    )
    
    # Get feature importance
    importance = model.get_score(importance_type="gain")
    
    # Define transform functions
    def transform(y: ArrayLike) -> np.ndarray:
        return np.log10(np.asarray(y) + 1)
    
    def inverse_transform(y: ArrayLike) -> np.ndarray:
        return np.power(10, np.asarray(y)) - 1
    
    return {
        "model": model,
        "params": params,
        "nrounds": nrounds,
        "importance": importance,
        "transform": transform,
        "inverse_transform": inverse_transform,
    }
