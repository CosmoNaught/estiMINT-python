

from omegaconf import DictConfig
import random
import pandas as pd
import logging
from pathlib import Path
import numpy as np
from .features import StandardScaler, FEATURES_BASE
import pickle
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

@dataclass
class PreparedData:
    train_data: list
    val_data: list
    test_data: list
    input_size: int
    scaler: StandardScaler
    train_param_sims: set[tuple[int, int]] = field(default_factory=set)
    val_param_sims: set[tuple[int, int]] = field(default_factory=set)
    test_param_sims: set[tuple[int, int]] = field(default_factory=set)

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
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
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
    train_ps = {
        (r.parameter_index, r.simulation_index) for r in split_df[split_df["split"] == "train"].itertuples()
    } & present
    val_ps = {
        (r.parameter_index, r.simulation_index) for r in split_df[split_df["split"] == "validate"].itertuples()
    } & present
    test_ps = {
        (r.parameter_index, r.simulation_index) for r in split_df[split_df["split"] == "test"].itertuples()
    } & present
    return train_ps, val_ps, test_ps  # type: ignore


def _create_split(
    df: pd.DataFrame, seed: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Create a train/val/test split by parameter.

    Args:
        df: Filtered dataframe.
        seed: Shuffle seed.

    Returns:
        Train, validation, and test parameter-simulation sets.
    """
    random.seed(seed)
    params = list(df["parameter_index"].unique())
    random.shuffle(params)
    n = len(params)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_p = set(params[:n_train])
    val_p = set(params[n_train : n_train + n_val])
    test_p = set(params[n_train + n_val :])
    all_ps = set(df[["parameter_index", "simulation_index"]].itertuples(index=False, name=None))
    return (
        {ps for ps in all_ps if ps[0] in train_p},
        {ps for ps in all_ps if ps[0] in val_p},
        {ps for ps in all_ps if ps[0] in test_p},
    )


def _save_split(path, train_ps, val_ps, test_ps, df):
    """
    Save parameter-simulation split assignments.

    Args:
        path: Output CSV path.
        train_ps: Training pairs.
        val_ps: Validation pairs.
        test_ps: Test pairs.
        df: Source dataframe.

    Returns:
        None.
    """
    rows = []
    for ps, split in (
        [(p, "train") for p in train_ps] + [(p, "validate") for p in val_ps] + [(p, "test") for p in test_ps]
    ):
        rows.append(
            {"parameter_index": ps[0], "simulation_index": ps[1], "split": split}
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"Split saved to {path}")

def _fit_scaler(df: pd.DataFrame, train_ps: set[tuple[int, int]], output_dir: str) -> StandardScaler:
    """
    Fit and save the static covariate scaler.

    Args:
        df: Filtered dataframe.
        train_ps: Training pairs.
        output_dir: Directory for scaler output.

    Returns:
        Fitted scaler.
    """
    train_mask = df["_ps"].isin(train_ps)
    train_static = (
        df.loc[train_mask, ["_ps"] + FEATURES_BASE]
        .drop_duplicates(subset=["_ps"])[FEATURES_BASE]
        .astype(np.float32)
        .values
    )
    scaler = StandardScaler()
    scaler.fit(train_static)

    save_path = Path(output_dir) / "static_scaler.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)

    return scaler

def _build_data(
    df: pd.DataFrame,
    param_sims: set[tuple[int, int]],
    scaler: StandardScaler,
    cfg: DictConfig,
) -> list[dict[str, np.ndarray]]:
    """

    """
    groups = df.groupby(["parameter_index", "simulation_index"])
    data = []

    for ps in param_sims:
        if ps not in groups.groups:
            continue

        df["eir_log10"] = np.log10(df["eir"], dtype=np.float32) # TODO: do we need to log10?
        X = (
            df.loc[groups.groups[ps], FEATURES_BASE]
            .astype(np.float32)
            .values
        )
        Y = df.loc[groups.groups[ps], "eir_log10"].values
        data.append(
            {
                "x": X,
                "y": Y,
                "ps": np.asarray(ps, dtype=np.int32),  # (2,) parameter_index, simulation_index
            }
        )

    return data

def prepare_data(df: pd.DataFrame, cfg: DictConfig):
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


    # split data
    if cfg.use_existing_split and Path(cfg.split_file).exists():
        log.info(f"Loading existing split from {cfg.split_file}")
        train_ps, val_ps, test_ps = _load_split(cfg.split_file, df)
    else:
        log.info("Creating new train/val/test split (70/15/15)")
        train_ps, val_ps, test_ps = _create_split(df, cfg.seed)
        if cfg.split_file:
            log.info(f"Saving split to {cfg.split_file}")
            _save_split(cfg.split_file, train_ps, val_ps, test_ps, df)

    log.info(f"Split — train: {len(train_ps)}, val: {len(val_ps)}, test: {len(test_ps)}")

    scaler = _fit_scaler(df, train_ps, cfg.output_dir) # TODO fix scaling

    train_data = _build_data(df, train_ps, scaler, cfg)
    val_data = _build_data(df, val_ps, scaler, cfg)
    test_data = _build_data(df, test_ps, scaler, cfg)

    return PreparedData(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        input_size=len(FEATURES_BASE),
        scaler=scaler,
        train_param_sims=train_ps,
        val_param_sims=val_ps,
        test_param_sims=test_ps,
    )

