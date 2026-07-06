import logging
import time
from pathlib import Path

import duckdb
import hydra
import jax
import jax.numpy as jnp
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .data.preprocess import PreparedData, prepare_data

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("JAX devices: %s", jax.devices())

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    raw_df = duckdb.read_parquet(cfg.data_file).df()

    prepared_data = prepare_data(raw_df, cfg)

    print(f"train data len: {len(prepared_data.train_data)}")

if __name__ == "__main__":
    main()