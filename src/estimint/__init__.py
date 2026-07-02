"""
estiMINT - EIR Estimation using Machine learning INTerventions

This package provides tools for training and running XGBoost models
to predict Entomological Inoculation Rate (EIR) from malaria intervention data.

Dependencies
------------
Core (inference): numpy, pandas, xgboost, scipy.
Optional extras:
- train: duckdb, scikit-learn, pyarrow  (data prep + model training)
- viz:   matplotlib                      (plotting)
- download: requests, appdirs            (fetch published models)
"""

__package_name__ = "estiMINT"

# Public API exports
from .utils import (
    ts,
    r2,
    rmse,
    mse,
    mae,
    median_ae,
    mae_rel,
    rmsle,
    safe_div,
    smape,
    fit_qmap_w,
    predict_qmap_w,
    scale_pos,
)

from .data_processing import (
    load_and_filter,
    make_value_weights,
    strata_and_split,
)

from .models import train_eir_xgboost

from .train import train_xgb_model

from .plotting import plot_obs_pred

from .storage import (
    load_xgb_model,
    save_xgb_model,
    bundle_model,
)

from .run import run_xgb_model, set_global_model, get_global_model

from .hbr import estimate_eir_with_mosquito_delta

from .bednet import calculate_dn0, net_types, DN0Result

from .scenarios import run_scenarios, preload_models
from .types import Scenario, EirTarget

__all__ = [
    # utils
    "ts",
    "r2",
    "rmse",
    "mse",
    "mae",
    "median_ae",
    "mae_rel",
    "rmsle",
    "safe_div",
    "smape",
    "fit_qmap_w",
    "predict_qmap_w",
    "scale_pos",
    # data_processing
    "load_and_filter",
    "make_value_weights",
    "strata_and_split",
    # models
    "train_eir_xgboost",
    # train
    "train_xgb_model",
    # plotting
    "plot_obs_pred",
    # storage
    "load_xgb_model",
    "save_xgb_model",
    "bundle_model",
    # run
    "run_xgb_model",
    "set_global_model",
    "get_global_model",
    # hbr
    "estimate_eir_with_mosquito_delta",
    # bednet
    "calculate_dn0",
    "net_types",
    "DN0Result",
    # scenarios
    "run_scenarios",
    "preload_models",
    # types
    "Scenario",
    "EirTarget",
]
