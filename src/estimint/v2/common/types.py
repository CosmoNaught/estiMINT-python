
from typing import Literal, Protocol

from flax import nnx
from omegaconf import DictConfig
import numpy as np


class ModelFactory(Protocol):
    @classmethod
    def from_cfg(cls, cfg: DictConfig, input_size: int) -> nnx.Module: ...

class ModelArtifact(Protocol):
    def predict(self, X_raw: np.ndarray) -> np.ndarray: ...

PredictorType = Literal["prev_y9", "eir", "hbr_y9"]
TargetType = Literal["eir", "hbr_y9"]

