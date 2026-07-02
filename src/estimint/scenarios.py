from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from .bednet import DN0Result, calculate_dn0
from .hbr import estimate_eir_with_mosquito_delta
from .run import run_xgb_model
from .storage import load_xgb_model
from .types import EirTarget, Scenario
from collections import defaultdict

####################### Constants and global storage ###################
HF_REPO = "dide-ic/stateMINT"
# 157 windows of 14 days from day 2190; intervention at day 3285.
_ABS_TIME = 2190 + 14 * np.arange(157)
_IDX_Y9 = int(np.argmin(np.abs(_ABS_TIME - 3285)))

_EIR_MODEL_CACHE: Dict[str, Any] = {}
_EMULATOR_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}

_NET_KEYS = (
    "py_only",
    "py_pbo",
    "py_pyrrole",
    "py_ppf",
)

_REQUIRED_EIR_MODEL_NAMES = ("prevalence", "hbr", "eir_to_hbr")


@dataclass(frozen=True)
class _EirInputModelConfig:
    feature_column: str
    model_name: str


_EIR_INPUT_MODEL_CONFIG = {
    "prevalence": _EirInputModelConfig(feature_column="prev_y9", model_name="prevalence"),
    "hbr": _EirInputModelConfig(feature_column="hbr_y9", model_name="hbr"),
}


######################## Internal helpers  ########################
def _load_eir_hbr_models() -> Dict[str, Any]:
    if not _EIR_MODEL_CACHE:
        for model_name in _REQUIRED_EIR_MODEL_NAMES:
            _EIR_MODEL_CACHE[model_name] = load_xgb_model(model_name)
    return _EIR_MODEL_CACHE


def _load_emulators(hf_repo: str) -> Dict[str, Any]:
    if hf_repo not in _EMULATOR_MODEL_CACHE:
        try:
            from stateMINT.model import Mamba2Regressor
        except ImportError as error:
            raise ImportError(
                "run_scenarios needs stateMINT. Install it with: "
                "uv sync --extra scenarios (or pip install "
                '"git+https://github.com/mrc-ide/stateMINT.git@mamba2-train").'
            ) from error
        _EMULATOR_MODEL_CACHE[hf_repo] = {
            outcome_name: Mamba2Regressor.from_pretrained(hf_repo, predictor=outcome_name)
            for outcome_name in ("prevalence", "cases")
        }
    return _EMULATOR_MODEL_CACHE[hf_repo]


@dataclass(frozen=True)
class _BedNetEffects:
    current_coverages: dict[str, float]
    current: DN0Result
    future: DN0Result


def _calculate_bednet_effects(scenario: Scenario) -> _BedNetEffects:
    """Calculate current and future bed-net coverage and transmission effects."""
    current_coverages = {net_type: getattr(scenario, net_type) for net_type in _NET_KEYS if getattr(scenario, net_type)}

    current_effect = calculate_dn0(scenario.res_use, **current_coverages) if current_coverages else DN0Result(0.0, 0.0)

    future_net_type = scenario.net_type_future
    if scenario.itn_future == 0.0 or not future_net_type:
        future_effect = DN0Result(0.0, 0.0)
    else:
        future_effect = calculate_dn0(scenario.res_use, **{future_net_type: scenario.itn_future})

    return _BedNetEffects(
        current_coverages=current_coverages,
        current=current_effect,
        future=future_effect,
    )


@dataclass
class _PreparedScenario:
    eir_target: EirTarget
    mosquito_density_change: float
    eir_model_features: Dict[str, float]
    summary_values: dict[str, Any]
    emulator_covariates: dict[str, float]


def _prepare_scenario_inputs(scenario: Scenario) -> _PreparedScenario:
    """Compute the model inputs and initial output values for one scenario."""
    bednet_effects = _calculate_bednet_effects(scenario)
    current_dn0 = bednet_effects.current.dn0
    current_itn_use = bednet_effects.current.itn_use
    future_dn0 = bednet_effects.future.dn0
    future_itn_use = bednet_effects.future.itn_use

    adjusted_lsm_coverage = scenario.lsm
    if scenario.py_ppf > 0:
        adjusted_lsm_coverage = min(scenario.py_ppf * 0.248 + adjusted_lsm_coverage, 1.0)

    eir_model_features = {
        "dn0_use": current_dn0,
        "Q0": scenario.Q0,
        "phi_bednets": scenario.phi,
        "seasonal": scenario.seasonal,
        "itn_use": current_itn_use,
        "irs_use": scenario.irs,
    }

    emulator_covariates = {
        "eir": np.nan,
        "dn0_use": current_dn0,
        "dn0_future": future_dn0,
        "Q0": scenario.Q0,
        "phi_bednets": scenario.phi,
        "seasonal": scenario.seasonal,
        "routine": scenario.routine,
        "itn_use": current_itn_use,
        "irs_use": scenario.irs,
        "itn_future": future_itn_use,
        "irs_future": scenario.irs_future,
        "lsm": adjusted_lsm_coverage,
    }

    summary_values = {
        "name": scenario.name,
        "input_mode": scenario.eir_target.input_mode,
        "net": "+".join(bednet_effects.current_coverages) or "none",
        "net_future": scenario.net_type_future or "none",
        "dn0_use": current_dn0,
        "itn_use": current_itn_use,
        "irs_use": scenario.irs,
        "dn0_future": future_dn0,
        "itn_future": future_itn_use,
        "irs_future": scenario.irs_future,
        "routine": scenario.routine,
        "lsm": adjusted_lsm_coverage,
        "seasonal": scenario.seasonal,
        "eir_baseline": np.nan,
        "mosquito_delta": scenario.mosquito_delta,
        "eir_final": np.nan,
        "hbr_baseline": np.nan,
        "hbr_new": np.nan,
    }

    return _PreparedScenario(
        eir_target=scenario.eir_target,
        mosquito_density_change=scenario.mosquito_delta,
        eir_model_features=eir_model_features,
        summary_values=summary_values,
        emulator_covariates=emulator_covariates,
    )


def _record_eir_estimate(
    prepared_scenario: _PreparedScenario,
    *,
    eir_baseline: float,
    eir_final: float,
    hbr_baseline: float = np.nan,
    hbr_new: float = np.nan,
) -> None:
    prepared_scenario.summary_values["eir_baseline"] = float(eir_baseline)
    prepared_scenario.summary_values["eir_final"] = float(eir_final)
    prepared_scenario.summary_values["hbr_baseline"] = float(hbr_baseline)
    prepared_scenario.summary_values["hbr_new"] = float(hbr_new)
    prepared_scenario.emulator_covariates["eir"] = float(eir_final)


def _predict_eir_from_measurements(
    prepared_scenarios: list[_PreparedScenario], *, input_mode: str, eir_models: Dict[str, Any]
) -> None:
    """Predict EIR from baseline prevalence or HBR measurements."""
    model_config = _EIR_INPUT_MODEL_CONFIG[input_mode]
    model_input_records = [
        {
            **prepared_scenario.eir_model_features,
            model_config.feature_column: prepared_scenario.eir_target.input_value,
        }
        for prepared_scenario in prepared_scenarios
    ]

    eir_predictions = run_xgb_model(pd.DataFrame(model_input_records), eir_models[model_config.model_name])

    for prepared_scenario, eir_prediction in zip(prepared_scenarios, eir_predictions):
        _record_eir_estimate(
            prepared_scenario,
            eir_baseline=eir_prediction,
            eir_final=eir_prediction,
        )


def _classify_prepared_scenario(prepared_scenario: _PreparedScenario) -> str:
    """Return the EIR estimation method for a prepared scenario."""
    if prepared_scenario.eir_target.input_mode == "eir":
        return "eir"
    if prepared_scenario.eir_target.input_mode == "prevalence" and prepared_scenario.mosquito_density_change != 0.0:
        return "mosquito_delta"
    return prepared_scenario.eir_target.input_mode  # "prevalence" or "hbr"


def _apply_mosquito_delta_batch(prepared_scenarios: list[_PreparedScenario], eir_models: Dict[str, Any]) -> None:
    inputs = pd.DataFrame(
        [
            {
                "prevalence": prepared_scenario.eir_target.input_value,
                "mosquito_delta": prepared_scenario.mosquito_density_change,
                **prepared_scenario.eir_model_features,
            }
            for prepared_scenario in prepared_scenarios
        ]
    )
    estimates = estimate_eir_with_mosquito_delta(inputs, models=eir_models).to_dict(orient="records")
    for prepared_scenario, estimate in zip(prepared_scenarios, estimates):
        _record_eir_estimate(
            prepared_scenario,
            eir_baseline=estimate["eir_baseline"],
            eir_final=estimate["eir_new"],
            hbr_baseline=estimate["hbr_baseline"],
            hbr_new=estimate["hbr_new"],
        )


def _estimate_eir(scenarios: list[Scenario], eir_models: Dict[str, Any]) -> list[_PreparedScenario]:
    """Estimate EIR for many scenarios, dispatching each to one of three paths:
    - "eir": supplied directly, passed through unchanged
    - "prevalence" / "hbr": predicted from baseline measurements via XGBoost
    - "mosquito_delta": prevalence input with a projected mosquito-density change
    """
    if any(scenario.eir_target.input_mode not in {"prevalence", "eir", "hbr"} for scenario in scenarios):
        raise ValueError("All scenarios must have input_mode in {'prevalence', 'eir', 'hbr'}")

    prepared_scenarios = [_prepare_scenario_inputs(scenario) for scenario in scenarios]

    scenario_groups: dict[str, list[_PreparedScenario]] = defaultdict(list)
    for prepared_scenario in prepared_scenarios:
        scenario_groups[_classify_prepared_scenario(prepared_scenario)].append(prepared_scenario)

    for prepared_scenario in scenario_groups["eir"]:
        supplied_eir = prepared_scenario.eir_target.input_value
        _record_eir_estimate(prepared_scenario, eir_baseline=supplied_eir, eir_final=supplied_eir)

    for input_mode in ("prevalence", "hbr"):
        if scenario_groups[input_mode]:
            _predict_eir_from_measurements(scenario_groups[input_mode], input_mode=input_mode, eir_models=eir_models)

    if scenario_groups["mosquito_delta"]:
        _apply_mosquito_delta_batch(scenario_groups["mosquito_delta"], eir_models)

    return prepared_scenarios


######################### Public API  ########################
def preload_models(*, hf_repo: str = HF_REPO) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Preload the models used by run_scenarios."""
    eir_models = _load_eir_hbr_models()
    emulator_models = _load_emulators(hf_repo)

    return eir_models, emulator_models


def run_scenarios(
    scenarios: list[Scenario],
    *,
    hf_repo: str = HF_REPO,
) -> pd.DataFrame:
    """Run a list of scenarios through the estiMINT -> stateMINT pipeline.

    For each scenario, estimates the entomological inoculation rate (EIR) from
    the given input (prevalence, EIR, or HBR), then feeds the resulting
    covariate vector into emulator models to predict malaria prevalence and
    case burden over time.

    Args:
        scenarios: Scenarios to evaluate. Each ``Scenario`` describes
            intervention coverages (ITN, IRS, LSM, etc.) and an
            ``EirTarget`` specifying the baseline transmission intensity.
        hf_repo: HuggingFace repo ID from which emulator model weights are
            downloaded. Defaults to the package-level ``HF_REPO`` constant.

    Returns:
        A ``pd.DataFrame`` with one row per scenario containing:

        - ``name``, ``input_mode``, ``net``, ``net_future`` — scenario labels
        - ``dn0_use``, ``itn_use``, ``irs_use``, ``dn0_future``,
          ``itn_future``, ``irs_future``, ``routine``, ``lsm``,
          ``seasonal``, ``mosquito_delta`` — intervention covariates
        - ``eir_baseline``, ``eir_final`` — estimated EIR before and after
          interventions
        - ``hbr_baseline``, ``hbr_new`` — human biting rate (populated for
          prevalence inputs with a mosquito-density change, otherwise ``NaN``)
        - ``prev_y9`` — prevalence at year 9 (≈3285 days)
        - ``prev_endline`` — prevalence at the final time step
        - ``cases_endline`` — cases at the final time step
        - ``prevalence`` — full prevalence time series (numpy array)
        - ``cases`` — full cases time series (numpy array, floored at 0)

        Returns an empty ``DataFrame`` if ``scenarios`` is empty.

    Example:
        >>> from estimint.scenarios import run_scenarios
        >>> from estimint.types import EirTarget, Scenario
        >>>
        >>> scenarios = [
        ...     Scenario(
        ...         name="baseline",
        ...         res_use=0.0,
        ...         Q0=0.9,
        ...         phi=0.85,
        ...         seasonal=0.5,
        ...         irs=0.0,
        ...         eir_target=EirTarget(input_value=50.0, input_mode="eir"),
        ...     ),
        ...     Scenario(
        ...         name="itn_campaign",
        ...         res_use=0.0,
        ...         Q0=0.9,
        ...         phi=0.85,
        ...         seasonal=0.5,
        ...         irs=0.0,
        ...         eir_target=EirTarget(input_value=50.0, input_mode="eir"),
        ...         py_only=0.6,
        ...         net_type_future="pyrethroid-only",
        ...         itn_future=0.6,
        ...     ),
        ... ]
        >>>
        >>> results = run_scenarios(scenarios)
        >>> results[["name", "prevalence", "cases"]]
    """
    if not scenarios:
        return pd.DataFrame()

    eir_models, emulator_models = preload_models(hf_repo=hf_repo)

    scenario_estimates = _estimate_eir(scenarios, eir_models)
    emulator_covariates = [estimate.emulator_covariates for estimate in scenario_estimates]
    prevalence_timeseries = emulator_models["prevalence"].predict(emulator_covariates)
    case_timeseries = np.maximum(emulator_models["cases"].predict(emulator_covariates), 0.0)

    return pd.DataFrame(
        [
            {
                **scenario_estimate.summary_values,
                "prev_y9": float(prevalence_series[_IDX_Y9]),
                "prev_endline": float(prevalence_series[-1]),
                "cases_endline": float(case_series[-1]),
                "prevalence": prevalence_series,
                "cases": case_series,
            }
            for scenario_estimate, prevalence_series, case_series in zip(
                scenario_estimates,
                prevalence_timeseries,
                case_timeseries,
            )
        ]
    )
