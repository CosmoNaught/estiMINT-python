from typing import Any, Literal, Dict
from dataclasses import dataclass

from estimint.v2.models.hub import repo_id

Input_Mode = Literal["prevalence", "eir", "hbr"]


@dataclass
class EirTarget:
    input_value: float
    input_mode: Input_Mode = "prevalence"


@dataclass
class Scenario:
    name: str
    res_use: float
    Q0: float
    phi: float
    seasonal: float
    irs: float
    eir_target: EirTarget
    py_only: float = 0.0
    py_pbo: float = 0.0
    py_pyrrole: float = 0.0
    py_ppf: float = 0.0
    mosquito_delta: float = 0.0
    itn_future: float = 0.0
    net_type_future: str | None = None
    irs_future: float = 0.0
    routine: float = 0.0
    lsm: float = 0.0

@dataclass
class PreparedScenario:
    eir_target: EirTarget
    mosquito_density_change: float
    eir_model_features: Dict[str, float]
    summary_values: dict[str, Any]
    emulator_covariates: dict[str, float]

INPUT_MODE_TO_REPO_IDS: Dict[Input_Mode, str] = {
    "prevalence": repo_id("prev_y9", "eir"),
    "hbr": repo_id("hbr_y9", "eir"),
    "eir": repo_id("eir", "hbr_y9"),
}