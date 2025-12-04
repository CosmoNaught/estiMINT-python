"""
Data processing functions for estiMINT package.

Equivalent to: data_processing.R
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import duckdb
from sklearn.cluster import KMeans


def load_and_filter(
    in_parquet: str,
    thr_lo: float = 0.02,
    thr_hi: float = 0.95
) -> Dict[str, pd.DataFrame]:
    """
    Load parquet file and apply prevalence filters.
    
    Equivalent to R's load_and_filter() function.
    
    Parameters
    ----------
    in_parquet : str
        Path to input parquet file
    thr_lo : float, optional
        Lower prevalence threshold (inclusive, default: 0.02)
    thr_hi : float, optional
        Upper prevalence threshold (inclusive, default: 0.95)
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'DT': DataFrame with rows passing filters
        - 'DT_excluded': DataFrame with rows failing filters
    """
    con = duckdb.connect(database=":memory:")
    
    try:
        # Set DuckDB parameters
        con.execute("PRAGMA threads=8; PRAGMA memory_limit='16GB';")
        
        qry = f"""
            WITH base AS (SELECT * FROM read_parquet('{in_parquet}')),
            avg_prev AS (
                SELECT parameter_index,
                       AVG(CASE WHEN year BETWEEN 1 AND 8 THEN prevalence_annual_mean END) AS prev_avg_1_8
                FROM base GROUP BY parameter_index
            ),
            y9 AS (
                SELECT b.parameter_index,
                       b.dn0_use, b.Q0, b.phi_bednets, b.seasonal, b.itn_use, b.irs_use,
                       b.prevalence_annual_mean AS prev_y9,
                       b.eir
                FROM base b WHERE b.year = 9
            )
            SELECT y9.*, avg_prev.prev_avg_1_8
            FROM y9 JOIN avg_prev USING (parameter_index);
        """
        
        df_all = con.execute(qry).fetchdf()
        
    finally:
        con.close()
    
    # Remove rows with any NaN
    DT0 = df_all.dropna()
    
    # Apply prevalence filters
    excluded_mask = (
        (DT0["prev_avg_1_8"] < thr_lo) | 
        (DT0["prev_avg_1_8"] > thr_hi) |
        (DT0["prev_y9"] < thr_lo) | 
        (DT0["prev_y9"] > thr_hi)
    )
    
    DT_excluded = DT0[excluded_mask].copy().reset_index(drop=True)
    DT = DT0[~excluded_mask].copy().reset_index(drop=True)
    
    return {"DT": DT, "DT_excluded": DT_excluded}


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


def strata_and_split(
    DT: pd.DataFrame,
    k_strata: int = 16,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create strata using k-means on log10(EIR) and perform stratified train/val/test split.
    
    Equivalent to R's strata_and_split() function.
    
    Parameters
    ----------
    DT : pd.DataFrame
        Input DataFrame (must have 'eir_log10' column)
    k_strata : int, optional
        Number of strata for k-means (default: 16)
    seed : int, optional
        Random seed (default: 42)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'strat_bin' and 'split' columns
    """
    DT = DT.copy()
    np.random.seed(seed)
    
    # K-means clustering on log10(EIR)
    eir_log10 = DT["eir_log10"].values.reshape(-1, 1)
    km = KMeans(n_clusters=k_strata, n_init=50, max_iter=5000, random_state=seed)
    km.fit(eir_log10)
    
    # Reorder cluster IDs by center value (ascending)
    centers = km.cluster_centers_.flatten()
    ord_idx = np.argsort(centers)
    id_map = {old_id: new_id + 1 for new_id, old_id in enumerate(ord_idx)}
    
    DT["strat_bin"] = np.array([id_map[c] for c in km.labels_])
    
    # Initialize split column
    DT["split"] = None
    
    # Stratified split within each bin
    for b in sorted(DT["strat_bin"].unique()):
        idx = DT[DT["strat_bin"] == b].index.tolist()
        n_b = len(idx)
        n_tr = int(np.floor(0.70 * n_b))
        n_val = int(np.floor(0.15 * n_b))
        
        # Shuffle indices
        np.random.shuffle(idx)
        
        # Assign splits
        tr_idx = idx[:n_tr] if n_tr > 0 else []
        val_idx = idx[n_tr:n_tr + n_val] if n_val > 0 else []
        te_idx = idx[n_tr + n_val:]
        
        DT.loc[tr_idx, "split"] = "train"
        DT.loc[val_idx, "split"] = "val"
        DT.loc[te_idx, "split"] = "test"
    
    # Fill any remaining NaN splits with 'train'
    DT["split"] = DT["split"].fillna("train")
    
    return DT
