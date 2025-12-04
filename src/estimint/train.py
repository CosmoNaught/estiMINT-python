"""
Main training pipeline for estiMINT package.

Equivalent to: train.R
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import xgboost as xgb

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
from .plotting import plot_obs_pred


def train_xgb_model(
    in_parquet: str,
    out_dir: str,
    thr_lo: float = 0.02,
    thr_hi: float = 0.95,
    k_strata: int = 16,
    K: int = 10,
    seed: int = 42,
    xgb_params: Optional[Dict[str, Any]] = None,
    nrounds_max: int = 5000,
    early_stopping_rounds: int = 100,
    save_pkl: bool = True,
    export_onnx: bool = False,
    save_plots: bool = True,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Train XGBoost with K-fold CV, QMAP+scale calibration, and optional artifacts.
    
    Equivalent to R's train_xgb_model() function.
    
    Parameters
    ----------
    in_parquet : str
        Path to input parquet file
    out_dir : str
        Base output directory (models/plots/metrics/predictions will be created)
    thr_lo : float, optional
        Lower prevalence filter, inclusive (default: 0.02)
    thr_hi : float, optional
        Upper prevalence filter, inclusive (default: 0.95)
    k_strata : int, optional
        Number of strata for k-means on log10(EIR) (default: 16)
    K : int, optional
        Number of CV folds (default: 10)
    seed : int, optional
        Random seed for reproducibility (default: 42)
    xgb_params : dict, optional
        XGBoost parameters (default: see function body)
    nrounds_max : int, optional
        Max rounds per fold for early-stopped training (default: 5000)
    early_stopping_rounds : int, optional
        Early stopping patience (default: 100)
    save_pkl : bool, optional
        Save a pickle bundle with model, calibrator, metadata (default: True)
    export_onnx : bool, optional
        Attempt ONNX export (default: False)
    save_plots : bool, optional
        Save diagnostic plots (default: True)
    save_artifacts : bool, optional
        Save CSV metrics and fold stats (default: True)
        
    Returns
    -------
    dict
        An 'estiMINT_model' object with booster, calibrator, features, metadata
    """
    # Validate inputs
    assert isinstance(in_parquet, str), "in_parquet must be a string"
    assert isinstance(out_dir, str), "out_dir must be a string"
    
    # Default XGBoost parameters
    if xgb_params is None:
        xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "lambda": 1.0,
            "seed": seed,
        }
    
    # Create output directories
    out_dir = Path(out_dir)
    dir_models = out_dir / "models"
    dir_plots = out_dir / "plots"
    dir_metric = out_dir / "metrics"
    dir_pred = out_dir / "predictions"
    
    for d in [dir_models, dir_plots, dir_metric, dir_pred]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load and filter data
    ts("Reading parquet & applying prevalence filters ...")
    lf = load_and_filter(in_parquet, thr_lo=thr_lo, thr_hi=thr_hi)
    DT = lf["DT"]
    DT_excluded = lf["DT_excluded"]
    
    # Save excluded and kept data
    DT_excluded.to_csv(
        dir_metric / f"excluded_prev_outside_0p{int(thr_lo*100):02d}_0p{int(thr_hi*100):02d}.csv",
        index=False
    )
    DT.to_csv(
        dir_metric / f"kept_after_prev_filters_0p{int(thr_lo*100):02d}_0p{int(thr_hi*100):02d}.csv",
        index=False
    )
    
    # Define features
    features = ["dn0_use", "Q0", "phi_bednets", "seasonal", "itn_use", "irs_use", "prev_y9"]
    
    # Transform EIR to log10
    DT["eir_log10"] = np.log10(DT["eir"])
    
    assert len(DT) > 0, "No data remaining after filters"
    
    # Create strata and split
    np.random.seed(seed)
    ts("Creating %d strata on log10(EIR) and 70/15/15 split ...", k_strata)
    DT = strata_and_split(DT, k_strata=k_strata, seed=seed)
    
    # Hold-out test set
    DT_test = DT[DT["split"] == "test"]
    X_test = DT_test[features].values.astype(np.float64)
    y_test = DT_test["eir_log10"].values
    obs_eir_test = np.power(10, y_test)
    
    assert np.all(np.isfinite(X_test)), "X_test contains non-finite values"
    assert np.all(np.isfinite(y_test)), "y_test contains non-finite values"
    
    # CV folds on train+val
    ts("Assigning %d-fold CV within TRAIN+VAL strata ...", K)
    DTcv = DT[DT["split"] != "test"].copy()
    
    np.random.seed(seed + 1)
    
    # Assign folds within each stratum
    DTcv["fold"] = 0
    for b in DTcv["strat_bin"].unique():
        mask = DTcv["strat_bin"] == b
        n_b = mask.sum()
        idx = DTcv.index[mask].tolist()
        np.random.shuffle(idx)
        folds = np.tile(np.arange(1, K + 1), int(np.ceil(n_b / K)))[:n_b]
        np.random.shuffle(folds)
        DTcv.loc[idx, "fold"] = folds
    
    # K-fold CV training
    ts("Running %d-fold CV with early stopping ...", K)
    oof_pred_raw = np.full(len(DTcv), np.nan)
    best_iters = np.zeros(K, dtype=int)
    
    for k in range(1, K + 1):
        ts(" Fold %d / %d", k, K)
        
        idx_val = DTcv["fold"] == k
        idx_tr = DTcv["fold"] != k
        
        X_tr = DTcv.loc[idx_tr, features].values.astype(np.float64)
        y_tr = DTcv.loc[idx_tr, "eir_log10"].values
        X_va = DTcv.loc[idx_val, features].values.astype(np.float64)
        y_va = DTcv.loc[idx_val, "eir_log10"].values
        
        # Compute weights
        w_tr = make_value_weights(np.power(10, y_tr), digits=3)
        w_va = make_value_weights(np.power(10, y_va), digits=3)
        
        # Create DMatrix objects
        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dva = xgb.DMatrix(X_va, label=y_va, weight=w_va)
        
        # Train model
        mdl = xgb.train(
            params=xgb_params,
            dtrain=dtr,
            num_boost_round=nrounds_max,
            evals=[(dtr, "train"), (dva, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        
        best_iters[k - 1] = mdl.best_iteration
        
        # Predict on validation fold
        pred_log10_va = mdl.predict(dva)
        oof_pred_raw[idx_val.values] = np.power(10, pred_log10_va)
    
    assert np.all(np.isfinite(oof_pred_raw)), "OOF predictions contain non-finite values"
    obs_cv_raw = np.power(10, DTcv["eir_log10"].values)
    
    # Save fold statistics
    if save_artifacts:
        fold_stats = pd.DataFrame({
            "fold": np.arange(1, K + 1),
            "best_iteration": best_iters
        })
        fold_stats.to_csv(dir_metric / f"cv_fold_best_iterations_K{K}.csv", index=False)
    
    # Fit calibrator on OOF predictions
    ts("Fitting final calibrator (QMAP + positive scale) on OOF ...")
    cal_oof = fit_qmap_w(oof_pred_raw, obs_cv_raw, ngrid=1024, round_digits=8)
    oof_pred_cal = predict_qmap_w(oof_pred_raw, cal_oof)
    a_oof = scale_pos(obs_cv_raw, oof_pred_cal)
    oof_pred_final = np.maximum(0, a_oof * oof_pred_cal)
    
    # Save OOF metrics
    if save_artifacts:
        oof_metrics = pd.DataFrame({
            "set": ["OOF_uncalibrated", "OOF_calibrated"],
            "R2": [r2(obs_cv_raw, oof_pred_raw), r2(obs_cv_raw, oof_pred_final)],
            "bias": [np.mean(oof_pred_raw - obs_cv_raw), np.mean(oof_pred_final - obs_cv_raw)],
            "MSE": [mse(obs_cv_raw, oof_pred_raw), mse(obs_cv_raw, oof_pred_final)],
            "RMSE": [rmse(obs_cv_raw, oof_pred_raw), rmse(obs_cv_raw, oof_pred_final)],
            "MAE": [mae(obs_cv_raw, oof_pred_raw), mae(obs_cv_raw, oof_pred_final)],
            "MedianAE": [median_ae(obs_cv_raw, oof_pred_raw), median_ae(obs_cv_raw, oof_pred_final)],
            "MAE_rel": [mae_rel(obs_cv_raw, oof_pred_raw), mae_rel(obs_cv_raw, oof_pred_final)],
            "RMSLE": [rmsle(obs_cv_raw, oof_pred_raw), rmsle(obs_cv_raw, oof_pred_final)],
            "NRMSE_mean": [
                safe_div(rmse(obs_cv_raw, oof_pred_raw), np.mean(obs_cv_raw)),
                safe_div(rmse(obs_cv_raw, oof_pred_final), np.mean(obs_cv_raw))
            ],
            "RelRMSE_p1": [
                np.sqrt(np.mean(safe_div(oof_pred_raw - obs_cv_raw, np.maximum(1, obs_cv_raw)) ** 2)),
                np.sqrt(np.mean(safe_div(oof_pred_final - obs_cv_raw, np.maximum(1, obs_cv_raw)) ** 2))
            ],
            "sMAPE": [smape(obs_cv_raw, oof_pred_raw), smape(obs_cv_raw, oof_pred_final)],
        })
        oof_metrics.to_csv(dir_metric / f"eir_OOF_metrics_K{K}CV.csv", index=False)
    
    # Train final model on TRAIN+VAL
    ts("Training final model on TRAIN+VAL with nrounds = median(best_iteration) ...")
    best_nrounds = int(np.round(np.median(best_iters)))
    
    DT_trcv = DT[DT["split"] != "test"]
    X_trcv = DT_trcv[features].values.astype(np.float64)
    y_trcv = DT_trcv["eir_log10"].values
    w_trcv = make_value_weights(np.power(10, y_trcv), digits=3)
    
    dtrcv = xgb.DMatrix(X_trcv, label=y_trcv, weight=w_trcv)
    
    xgb_cvfit = xgb.train(
        params=xgb_params,
        dtrain=dtrcv,
        num_boost_round=best_nrounds,
        verbose_eval=False,
    )
    xgb_cvfit.save_model(str(dir_models / "eir_xgb_KCV.model"))
    
    # Predict on TEST and calibrate
    dtest = xgb.DMatrix(X_test, label=y_test)
    pred_log10_test_raw = xgb_cvfit.predict(dtest)
    pred_raw_test = np.power(10, pred_log10_test_raw)
    pred_eir_test = predict_qmap_w(pred_raw_test, cal_oof)
    pred_eir_test = np.maximum(0, a_oof * pred_eir_test)
    
    # Save test predictions
    test_preds = pd.DataFrame({
        "obs": obs_eir_test,
        "pred_xgb": pred_eir_test
    })
    test_preds.to_csv(dir_pred / "eir_test_predictions_xgb_QMAP_SCALE.csv", index=False)
    
    # Save diagnostic plots
    if save_plots:
        plot_obs_pred(
            obs_eir_test, pred_eir_test,
            f"EIR — Observed vs Predicted (XGBoost, K={K} CV, QMAP+Scale, test)",
            str(dir_plots / "eir_obs_vs_pred_xgb_QMAP_SCALE_test.png"),
            xlab="Observed EIR", ylab="Predicted EIR"
        )
        plot_obs_pred(
            y_test, np.log10(np.maximum(1e-12, pred_eir_test)),
            f"EIR (log10) — Observed vs Predicted (XGBoost, K={K} CV after QMAP+Scale, test)",
            str(dir_plots / "eir_log10_obs_vs_pred_xgb_after_QMAP_SCALE_test.png"),
            xlab="Observed log10(EIR)", ylab="Predicted log10(EIR)"
        )
    
    # Calculate range-based metrics
    bins = [0, 10, 50, 100, 200, np.inf]
    labels = ["[0,10]", "(10,50]", "(50,100]", "(100,200]", "(200,Inf]"]
    DTm = pd.DataFrame({
        "range": pd.cut(obs_eir_test, bins=bins, labels=labels, include_lowest=True),
        "obs": obs_eir_test,
        "pred": pred_eir_test,
        "err": pred_eir_test - obs_eir_test
    })
    
    per_range_list = []
    for rng in labels:
        subset = DTm[DTm["range"] == rng]
        if len(subset) == 0:
            continue
        obs_s = subset["obs"].values
        pred_s = subset["pred"].values
        err_s = subset["err"].values
        
        per_range_list.append({
            "range": rng,
            "N": len(subset),
            "obs_mean": np.mean(obs_s),
            "obs_median": np.median(obs_s),
            "obs_sd": np.std(obs_s, ddof=1) if len(obs_s) > 1 else np.nan,
            "pred_mean": np.mean(pred_s),
            "bias": np.mean(err_s),
            "MAE": mae(obs_s, pred_s),
            "MedianAE": median_ae(obs_s, pred_s),
            "RMSE": rmse(obs_s, pred_s),
            "RMSLE": rmsle(obs_s, pred_s),
            "NRMSE_mean": safe_div(rmse(obs_s, pred_s), np.mean(obs_s)),
            "RelRMSE_p1": np.sqrt(np.mean(safe_div(err_s, np.maximum(1, obs_s)) ** 2)),
            "sMAPE": smape(obs_s, pred_s),
        })
    
    per_range = pd.DataFrame(per_range_list)
    
    if save_artifacts:
        per_range[["range", "RMSE"]].assign(model="xgboost_KCV").to_csv(
            dir_metric / "eir_RMSE_by_range_test_QMAP_SCALE.csv", index=False
        )
        per_range.assign(model="xgboost_KCV").to_csv(
            dir_metric / "eir_metrics_by_range_test_QMAP_SCALE.csv", index=False
        )
    
    # Train deployment model on ALL filtered data
    ts("Training deployment booster on ALL filtered data ...")
    X_all = DT[features].values.astype(np.float64)
    y_all = DT["eir_log10"].values
    dall = xgb.DMatrix(X_all, label=y_all)
    
    xgb_final = xgb.train(
        params=xgb_params,
        dtrain=dall,
        num_boost_round=best_nrounds,
        verbose_eval=False,
    )
    xgb_final.save_model(str(dir_models / "eir_xgb_FINAL.model"))
    
    # Create calibration bundle
    cal_bundle = {
        "kind": "qmap+scale",
        "qmap": {"xq": cal_oof["xq"], "yq": cal_oof["yq"]},
        "scale": a_oof
    }
    
    # Create preprocessing metadata
    preprocess = {
        "features": features,
        "target": "eir",
        "transform": "log10",
        "inverse": "pow10",
        "prevalence_filter": {
            "min_prev_input": thr_lo,
            "avg_prev_years_1_to_8_ge": thr_lo,
            "year9_prev_le": thr_hi
        },
        "reweighting": {
            "scheme": "inverse_frequency_by_raw_EIR_value",
            "digits": 3,
            "applied_to": ["train", "val", "cv_folds"]
        },
        "cv": {
            "K": K,
            "stratify_by": f"strat_bin (k-means on log10(EIR), centers={k_strata})",
            "best_iteration_median": best_nrounds
        },
        "calibration": {
            "final": "QMAP then positive scale",
            "final_pred": "pmax(0, a * QMAP(10^pred_log10))"
        }
    }
    
    # Create model bundle
    model_bundle = {
        "class": "estiMINT_model",
        "booster": xgb_final,
        "calibrator": cal_bundle,
        "features": features,
        "best_nrounds": best_nrounds,
        "preprocess": preprocess,
        "artifacts": {
            "dir_models": str(dir_models),
            "dir_plots": str(dir_plots),
            "dir_metric": str(dir_metric),
            "dir_pred": str(dir_pred)
        }
    }
    
    # Save pickle bundle
    if save_pkl:
        with open(dir_models / "estiMINT_model.pkl", "wb") as f:
            pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # ONNX export not supported
    if export_onnx:
        raise NotImplementedError(
            "ONNX export is not implemented in this version. "
            "Use onnxmltools or skl2onnx externally to convert the model."
        )
    
    ts("Done. Artifacts saved under: %s", out_dir)
    
    return model_bundle
