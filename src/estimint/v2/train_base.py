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
from .data.dataset import make_loader

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("JAX devices: %s", jax.devices())

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    raw_df = duckdb.read_parquet(cfg.data_file).df()

    prepared_data = prepare_data(raw_df, cfg)

    val_loader = make_loader(
        data=prepared_data.val_data,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
    test_loader = make_loader(
        data=prepared_data.test_data,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_remainder=True,
    )
if __name__ == "__main__":
    main()