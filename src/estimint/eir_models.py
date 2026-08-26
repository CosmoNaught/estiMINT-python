"""Loading and inference for the estiMINT EIR models.

Three conditional RQS flows are published on the Hugging Face hub. Each takes the
shared intervention covariates plus one measurement, and is keyed here by the name
of that measurement:

    prev_y9 -> eir       baseline prevalence to EIR
    hbr_y9  -> eir       human biting rate to EIR
    eir     -> hbr_y9    EIR to human biting rate
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from estimint.v2.common.types import PredictorType, TargetType
from estimint.v2.models.rqs import ConditionalRQS, RQSArtifact

from .types import Input_Mode, PreparedScenario

ESTIMINT_HF_REPO = "dide-ic/estiMINT"

# What each model predicts from the measurement it is named after.
EIR_MODEL_TARGETS: dict[PredictorType, TargetType] = {
    "prev_y9": "eir",
    "hbr_y9": "eir",
    "eir": "hbr_y9",
}

# The measurement a scenario supplies, per input mode.
INPUT_MODE_TO_PREDICTOR: dict[Input_Mode, PredictorType] = {
    "prevalence": "prev_y9",
    "hbr": "hbr_y9",
    "eir": "eir",
}

EirModels = dict[PredictorType, RQSArtifact]

_MODEL_CACHE: dict[str, EirModels] = {}


def load_eir_models(hf_repo: str = ESTIMINT_HF_REPO) -> EirModels:
    """Load the three estiMINT RQS artifacts, caching them per repo.

    Args:
        hf_repo: HuggingFace repo ID (or local folder) holding the model artifacts.

    Returns:
        The artifacts keyed by the measurement each one takes as input.
    """
    if hf_repo not in _MODEL_CACHE:
        _MODEL_CACHE[hf_repo] = {
            predictor: ConditionalRQS.from_pretrained(hf_repo, predictor, target)
            for predictor, target in EIR_MODEL_TARGETS.items()
        }
    return _MODEL_CACHE[hf_repo]


def predict_from_measurements(
    eir_models: EirModels,
    predictor: PredictorType,
    prepared_scenarios: Sequence[PreparedScenario],
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Predict ``EIR_MODEL_TARGETS[predictor]`` for a batch of scenarios.

    Each model row is the scenario's intervention covariates plus the supplied
    measurement; the artifact reorders them into the training feature order.

    Args:
        eir_models: Artifacts from :func:`load_eir_models`.
        predictor: The measurement carried by *values*, e.g. ``"prev_y9"``.
        prepared_scenarios: Scenarios supplying the intervention covariates.
        values: One *predictor* measurement per scenario, in the same order.

    Returns:
        Median predictions, one per scenario, in input order.
    """
    records = [
        {**prepared_scenario.eir_model_features, predictor: float(value)}
        for prepared_scenario, value in zip(prepared_scenarios, values, strict=True)
    ]
    return eir_models[predictor].predict(records)
