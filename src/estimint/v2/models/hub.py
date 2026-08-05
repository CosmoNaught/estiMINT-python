import json
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from ..common.types import PredictorType

from ..data.features import StandardScaler
from .rqs import ConditionalRQS, RQSArtifact
from ..training.checkpoint import restore_model


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _load_scaler(mean: list[float], scale: list[float]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.array(mean, dtype=np.float32)
    scaler.scale_ = np.array(scale, dtype=np.float32)
    return scaler


def _download_from_hf(
    repo_id: str,
    name: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> Path:
    root = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{name}/*", f"{name}/**"],
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )
    return Path(root) / name


def load_model_artifact(
    path_or_repo_id: str,
    predictor: PredictorType,
    target: PredictorType,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> RQSArtifact:
    """
    Load an RQS inference artifact from a local folder or Hugging Face repo.

    Expected artifact layout, matching estimint.v2.model_export:
        <name>/config.json
        <name>/checkpoint/

    Args:
        path_or_repo_id: Hugging Face repo ID or local folder path.
        name: Artifact subfolder name, e.g. "prev_y9-eir".
        revision: Optional revision of the model to load from the repo.
        cache_dir: Optional cache directory for the Hugging Face repo.
        local_dir: Optional local directory to download the repo into.

    Returns:
        RQSArtifact wrapping the restored model and fitted scalers.
    """
    if Path(path_or_repo_id).exists():
        artifact_dir = Path(path_or_repo_id)
    else:
        artifact_dir = _download_from_hf(
            path_or_repo_id, f"{predictor}-{target}", revision=revision, cache_dir=cache_dir, local_dir=local_dir
        )

    config = _load_json(artifact_dir / "config.json")

    features = config["features"]
    model = ConditionalRQS.from_cfg(OmegaConf.create(config), n_context=len(features))
    model = restore_model(str(artifact_dir / "checkpoint"), config["model_name"], model)
    model.eval()

    feature_scaler = _load_scaler(config["feature_scalar_mean"], config["feature_scalar_scale"])
    target_scaler = _load_scaler(config["target_scalar_mean"], config["target_scalar_scale"])

    return RQSArtifact(model=model, feature_scaler=feature_scaler, target_scaler=target_scaler, features=features)

