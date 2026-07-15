

from omegaconf import DictConfig
import random
import pandas as pd
import logging
from pathlib import Path
import numpy as np
from .features import StandardScaler, FEATURES_BASE
from estimint.data_processing import make_value_weights
import pickle
from dataclasses import dataclass, field
from typing import cast
log = logging.getLogger(__name__)

@dataclass
class PreparedData:
    train_data: list
    val_data: list
    test_data: list
    input_size: int
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    calib_data: list = field(default_factory=list)
    train_param_sims: set[tuple[int, int]] = field(default_factory=set)
    val_param_sims: set[tuple[int, int]] = field(default_factory=set)
    test_param_sims: set[tuple[int, int]] = field(default_factory=set)
    calib_param_sims: set[tuple[int, int]] = field(default_factory=set)

@dataclass
class SplitParamSims:
    train: set[tuple[int, int]] = field(default_factory=set)
    val: set[tuple[int, int]] = field(default_factory=set)
    calib: set[tuple[int, int]] = field(default_factory=set)
    test: set[tuple[int, int]] = field(default_factory=set)

def _filter_by_threshold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter parameter-simulation pairs where the mean target value is below the threshold.

    Args:
        df: Input dataframe.

    Returns:
        Filtered dataframe.
    """
    prev_threshold = 0.01  # TODO: make this configurable
    group_means = df.groupby(["parameter_index", "simulation_index"])["prev_y9"].mean()
    valid = set(map(tuple, group_means[group_means >= prev_threshold].index.tolist()))
    df["_ps"] = list(zip(df["parameter_index"], df["simulation_index"]))

    log.info(
        f"Filtering with prev_threshold {prev_threshold} on {"prevalence"}: {len(valid)} valid parameter-simulation pairs out of {len(group_means)}"
    )

    return df[df["_ps"].isin(valid)]


def _load_split(
    split_file: str, df: pd.DataFrame
) -> SplitParamSims:
    """
    Load an existing train/val/test split.

    Args:
        split_file: Split CSV path.
        df: Filtered dataframe.

    Returns:
        Train, validation, and test parameter-simulation sets.
    """
    split_df = pd.read_csv(split_file)
    present = set(df[["parameter_index", "simulation_index"]].itertuples(index=False, name=None))

    def _pairs(df: pd.DataFrame, split_name: str) -> set[tuple[int, int]]:
        return {

            (r.parameter_index, r.simulation_index) for r in df[df["split"] == split_name].itertuples()
        } & present # type: ignore

    return SplitParamSims(
        train=_pairs(split_df, "train"),
        val=_pairs(split_df, "validate"),
        test=_pairs(split_df, "test"),
        calib=_pairs(split_df, "calibrate")
    )

# TODO: check if need to strata or not.
def _parameter_strata(
    df: pd.DataFrame,
    target: str,
    n_bins: int,
    stratify: bool,
) -> list[np.ndarray]:
    """
    Build parameter-index groups used for stratified split assignment.

    Args:
        df: Input dataframe containing ``parameter_index`` and ``target`` columns.
        target: Column whose per-parameter mean defines the strata.
        n_bins: Maximum number of quantile bins to create when stratifying.
        stratify: If False, return all parameters as a single group.

    Returns:
        A list of parameter-index arrays. Each array is assigned to train,
        validation, calibration, and test splits independently.
    """
    param_target = df.groupby("parameter_index")[target].mean()
    param_indices = param_target.index.to_numpy()
    if not stratify:
        return [param_indices]

    target_values = np.log10(param_target.to_numpy(dtype=np.float64))
    unique_targets = np.unique(target_values)

    quantile_bins = pd.qcut(
        target_values,
        q=min(n_bins, len(unique_targets)),
        labels=False,
        duplicates="drop",
    )

    return [param_indices[quantile_bins == bucket] for bucket in np.unique(quantile_bins)]


def _param_sims_from_assignment(df: pd.DataFrame, assign: dict[int, str]) -> SplitParamSims:
    """Build parameter-simulation sets from a parameter-level split assignment."""
    all_ps = set(df[["parameter_index", "simulation_index"]].itertuples(index=False, name=None))
    split_params = {name: {p for p, split in assign.items() if split == name} for name in ["train", "val", "calib", "test"]}
    return SplitParamSims(
        train={ps for ps in all_ps if ps[0] in split_params["train"]},
        val={ps for ps in all_ps if ps[0] in split_params["val"]},
        calib={ps for ps in all_ps if ps[0] in split_params["calib"]},
        test={ps for ps in all_ps if ps[0] in split_params["test"]},
    )

def _assign_param_group(
    params: np.ndarray,
    rng: np.random.Generator,
    train_frac: float,
    val_frac: float,
    calib_frac: float,
) -> dict[int, str]:
    shuffled = np.array(params, copy=True)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end, val_end, calib_end = np.cumsum([np.array([train_frac, val_frac, calib_frac]) * n]).astype(int)

    assigned: dict[int, str] = {}
    for split_name, split_param in (
        ("train", shuffled[:train_end]),
        ("val", shuffled[train_end: val_end]),
        ("calib", shuffled[val_end: calib_end]),
        ("test", shuffled[calib_end:]),
    ):
        assigned.update({param: split_name for param in split_param})
    return assigned

def _assign_param_splits(
    df: pd.DataFrame,
    *,
    seed: int,
    calib_frac: float = 0.0,
    stratify: bool = True,
    target: str = "eir",
    n_bins: int = 10,
) -> dict[int, str]:
    """Assign each parameter_index to a split, grouping whole parameters.
    If stratify is True, the assignment is balanced across quantile bins of the mean target value per parameter.
    """
    rng = np.random.default_rng(seed)
    train_frac = 0.7
    val_frac = (1.0 - train_frac - calib_frac) / 2.0
    if val_frac <= 0:
        raise ValueError("Validation fraction must be positive; check calib_frac.")

    assign: dict[int, str] = {}
    for params in _parameter_strata(df, target, n_bins, stratify):
        assign.update(_assign_param_group(params, rng, train_frac, val_frac, calib_frac))
    return assign

def _create_split(
    df: pd.DataFrame,
    seed: int,
    *,
    stratify: bool = True,
    target: str = "eir",
    n_bins: int = 10,
    calib_frac: float = 0.0
) -> SplitParamSims:
    """
    Create grouped parameter-simulation splits.

    This is the pair-set version of :func:`group_split`. Both functions share
    the same parameter-level assignment, so train/val/calib/test never contain
    different simulations from the same ``parameter_index``.

    Args:
        df: Filtered dataframe.
        seed: Shuffle seed.
        calib_frac: Fraction to split calibration
        stratify: Balance the split across target-magnitude quantile bins.
        target: Column used for stratification.
        n_bins: Number of quantile strata when ``stratify`` is True.

    Returns:
        Grouped train, validation, optional calibration, and test
        parameter-simulation sets.
    """
    assign = _assign_param_splits(
        df,
        seed=seed,
        calib_frac=calib_frac,
        stratify=stratify,
        target=target,
        n_bins=n_bins,
    )
    return _param_sims_from_assignment(df, assign)


def _save_split(path, split_ps: SplitParamSims):
    """
    Save parameter-simulation split assignments.

    Args:
        path: Output CSV path.
        split_ps: Split parameter-simulation sets.
        df: Source dataframe.

    Returns:
        None.
    """
    rows = [
        (param_idx, sim_idx, split)
        for split, ps in (
            ("train", split_ps.train),
            ("validate", split_ps.val),
            ("test", split_ps.test),
            ("calibrate", split_ps.calib),
        )
        for param_idx, sim_idx in ps
    ]
    pd.DataFrame(rows, columns=["parameter_index", "simulation_index", "split"]).to_csv(path, index=False)
    log.info(f"Split saved to {path}")

def _fit_scaler(df: pd.DataFrame, train_ps: set[tuple[int, int]], output_dir: str, features: list[str] = FEATURES_BASE) -> StandardScaler:
    """
    Fit and save the static covariate scaler.

    Args:
        df: Filtered dataframe.
        train_ps: Training pairs.
        output_dir: Directory for scaler output.
        features: List of feature columns to use.

    Returns:
        Fitted scaler.
    """
    train_mask = df["_ps"].isin(train_ps)
    train_static = (
        df.loc[train_mask, ["_ps"] + features]
        .drop_duplicates(subset=["_ps"])[features]
        .to_numpy(dtype=np.float32)
    )
    scaler = StandardScaler()
    scaler.fit(train_static)

    save_path = Path(output_dir) / "static_scaler.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)

    return scaler

def _fit_target_scaler(df: pd.DataFrame, train_ps: set[tuple[int, int]], output_dir: str, target: str = "eir") -> StandardScaler:
    """
    Fit and save the target scaler.

    Args:
        df: Filtered dataframe.
        train_ps: Training pairs.
        output_dir: Directory for scaler output.
        target: Target column to scale.
    Returns:
        Fitted scaler.
    """
    train_mask = df["_ps"].isin(train_ps)
    train_y = (
        df.loc[train_mask, ["_ps", target]]
        .drop_duplicates(subset=["_ps"])[target]
        .to_numpy(dtype=np.float32)
    )
    scaler = StandardScaler()
    scaler.fit(np.log10(train_y)[:, None])  # Fit on log10 of target

    save_path = Path(output_dir) / "target_scaler.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)
    return scaler

def _build_data(
    df: pd.DataFrame,
    param_sims: set[tuple[int, int]],
    scaler: StandardScaler,
    target_scaler: StandardScaler,
    features: list[str] = FEATURES_BASE,
    target: str = "eir"
) -> list[dict[str, np.ndarray]]:
    """
    Build scaled per-parameter-simulation training records.

    Args:
        df: Filtered dataframe with feature, target, and ``_weight`` columns.
        param_sims: Parameter-simulation pairs to include.
        scaler: Fitted static feature scaler.
        target_scaler: Fitted target scaler.
        features: Feature columns to scale and include as model inputs.
        target: Positive target column to log-transform.

    Returns:
        A list of sequence dictionaries containing scaled features, raw
        features, log10 targets, raw targets, sample weights, and pair IDs.
    """
    # One row per (parameter_index, simulation_index): features/target are
    # static per simulation, so any row in the group carries the same values.
    rows = df.groupby(["parameter_index", "simulation_index"]).first()
    data = []

    for ps in param_sims:
        if ps not in rows.index:
            continue

        row = cast(pd.Series, rows.loc[ps])
        X_raw = row[features].to_numpy(dtype=np.float32)
        X = scaler.transform(X_raw)
        Y_raw = np.float32(row[target])
        Y = np.log10(Y_raw)
        Y_std = target_scaler.transform(np.array([[Y]], dtype=np.float32))[0, 0]
        W = np.float32(row["_weight"])
        data.append(
            {
                "x_raw": X_raw,
                "x": X,
                "y_raw": Y_raw,
                "y": Y,
                "y_std": Y_std,
                "w": W,
                "ps": np.asarray(ps, dtype=np.int32),  # (2,) parameter_index, simulation_index
            }
        )

    return data

# TODO: sort out exisitng splits and files with different models etc
def prepare_data(df: pd.DataFrame, cfg: DictConfig, calib_frac: float = 0.0) -> PreparedData:
    """
    Split and transform raw simulation data.

    Filters out low-signal parameter-simulation pairs, creates or loads the
    train/val/test split, fits static covariate scaling on the train split only,
    and builds per-sequence records for each split.

    Note: each malariasimulation run covers TOTAL_DAYS days: a MODEL_START_DAY's warmup followed by
    TOTAL_DAYS - MODEL_START_DAY days of actual simulation. Only the latter are used here; the
    warmup has already been discarded in the input `df` parameter.
    The intervention is applied at INTERVENTION_DAY.

    Args:
        df: Raw simulation dataframe.
        cfg: Data preparation config.

    Returns:
        Prepared train, validation, and test data.
    """
    random.seed(cfg.seed)

    # Filter by threshold
    df = _filter_by_threshold(df)
    df["_weight"] = make_value_weights(df[cfg.target].to_numpy(dtype=np.float32))
    # split data
    if cfg.use_existing_split and Path(cfg.split_file).exists():
        log.info(f"Loading existing split from {cfg.split_file}")
        split_ps = _load_split(cfg.split_file, df)
    else:
        log.info("Creating new train/val/calib/test split")
        split_ps = _create_split(df, cfg.seed, stratify=cfg.stratify, target=cfg.target, calib_frac=calib_frac)
        if cfg.split_file:
            log.info(f"Saving split to {cfg.split_file}")
            _save_split(cfg.split_file, split_ps)

    log.info(
        "Split — train: %s, val: %s, calib: %s, test: %s",
        len(split_ps.train),
        len(split_ps.val),
        len(split_ps.calib),
        len(split_ps.test),
    )

    scaler = _fit_scaler(df, split_ps.train, cfg.output_dir) # TODO fix scaling
    target_scaler = _fit_target_scaler(df, split_ps.train, cfg.output_dir, target=cfg.target)

    train_data = _build_data(df, split_ps.train, scaler, target_scaler, target=cfg.target)
    val_data = _build_data(df, split_ps.val, scaler, target_scaler, target=cfg.target)
    test_data = _build_data(df, split_ps.test, scaler, target_scaler, target=cfg.target)
    calib_data = _build_data(df, split_ps.calib, scaler, target_scaler, target=cfg.target)

    return PreparedData(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        calib_data=calib_data,
        input_size=len(FEATURES_BASE),
        feature_scaler=scaler,
        target_scaler=target_scaler,
        train_param_sims=split_ps.train,
        val_param_sims=split_ps.val,
        test_param_sims=split_ps.test,
        calib_param_sims=split_ps.calib
    )

