import logging
import hydra
import pickle
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from .models.rqs import ConditionalRQS
from .training.checkpoint import restore_model, save_checkpoint
from .data.preprocess import StandardScaler, get_features
import json

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="export_config")
def main(cfg: DictConfig):
    """
    Export the trained model for sharing to other users.

    Args:
        cfg: Hydra config for exporting the model.
    """
    log.info(OmegaConf.to_yaml(cfg))
    artifact_dir = Path(cfg.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.features_scaler_file, "rb") as f:
        feature_scaler: StandardScaler = pickle.load(f)
    with open(cfg.target_scaler_file, "rb") as f:
        target_scaler: StandardScaler = pickle.load(f)
    if not feature_scaler.is_fitted or not target_scaler.is_fitted:
        raise ValueError("Feature or target scaler is not fitted. Please fit the scalers before exporting the model.")

    features = get_features(cfg.predictor)
    model = ConditionalRQS.from_cfg(cfg, n_context=len(features))
    model = restore_model(cfg.checkpoint_dir, cfg.model_name, model)
    model.eval()
    save_checkpoint(f"{cfg.artifact_dir}/checkpoint", cfg.model_name, model)


    config = dict(
        model_name=cfg.model_name,
        predictor=cfg.predictor,
        target=cfg.target,
        width=cfg.width,
        depth=cfg.depth,
        n_bins=cfg.n_bins,
        rqs_bounds=cfg.rqs_bounds,
        mlp_residual=cfg.mlp_residual,
        dropout_rate=cfg.dropout_rate,
        features=features,
        feature_scalar_mean=feature_scaler.mean_.tolist(),
        feature_scalar_scale=feature_scaler.scale_.tolist(),
        target_scalar_mean=target_scaler.mean_.tolist(),
        target_scalar_scale=target_scaler.scale_.tolist(),
    )

    with (artifact_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()