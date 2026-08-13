"""
Human Biting Rate (HBR) pipeline for estiMINT.

Answers the question: "What happens to EIR if mosquito density changes by X%?"

Pipeline:
1. prev_y9 + interventions -> EIR_baseline       (prevalence model)
2. EIR_baseline + interventions -> HBR_baseline   (EIR-to-HBR model)
3. HBR_new = HBR_baseline * (1 + mosquito_delta)  (user's mosquito change, pos or neg)
4. HBR model predicts EIR at both HBR values      (ratio approach)
5. EIR_new = EIR_baseline * (EIR_scaled / EIR_roundtrip)
"""
import numpy as np
from estimint.types import PreparedScenario, INPUT_MODE_TO_REPO_IDS
from estimint.v2.models.rqs import RQSArtifact


def estimate_eir_with_mosquito_delta(prepared_scenarios: list[PreparedScenario], *, eir_models: dict[str, RQSArtifact]) -> list[dict[str, float]]:
    """
    Estimate new EIR after a change in mosquito density for multiple scenarios.

    The HBR model predicts EIR at both the
    baseline and scaled HBR, then applies the relative multiplier to the clean
    baseline EIR from the prevalence model.

    Parameters
    ----------
    inputs : pd.DataFrame
        One row per scenario. Required columns:

        - ``prevalence`` : baseline malaria prevalence (prev_y9), e.g. 0.30 for 30%.
        - ``mosquito_delta`` : fractional change in mosquito density, e.g. 0.10
          for +10%, -0.50 for -50%. Must be > -1 per row.
        - ``dn0_use`` : bednet contact reduction parameter.
        - ``Q0`` : human blood index.
        - ``phi_bednets`` : proportion of bites on humans while in bed.
        - ``seasonal`` : seasonality flag (0.0 or 1.0).
        - ``itn_use`` : ITN coverage (0-1).
        - ``irs_use`` : IRS coverage (0-1).

    models : dict
        Pre-loaded model dictionary with keys ``"prevalence"``, ``"hbr"``,
        and ``"eir_to_hbr"``.

    Returns
    -------
    pd.DataFrame
        Same index as *inputs*, with columns:

        - ``eir_baseline`` : baseline EIR from prevalence.
        - ``eir_new`` : new EIR after mosquito density change.
        - ``eir_multiplier`` : ratio of new EIR to baseline.
        - ``hbr_baseline`` : estimated baseline HBR.
        - ``hbr_new`` : HBR after mosquito density change.

    Examples
    --------
    >>> import pandas as pd
    >>> from estimint import estimate_eir_with_mosquito_delta
    >>> inputs = pd.DataFrame([
    ...     {"prevalence": 0.30, "mosquito_delta": 0.25,
    ...      "dn0_use": 0.33, "Q0": 0.87, "phi_bednets": 0.82,
    ...      "seasonal": 0.0, "itn_use": 0.6, "irs_use": 0.0},
    ... ])
    >>> result = estimate_eir_with_mosquito_delta(inputs, models=models)
    >>> print(result[["eir_baseline", "eir_new"]])
    """
    # Step 1: prevalence -> EIR baseline
    prev_eir_artifact = eir_models[INPUT_MODE_TO_REPO_IDS["prevalence"]]
    prev_eir_records = [
        {
            **prepared_scenario.eir_model_features,
            "prev_y9": prepared_scenario.eir_target.input_value,
        }
        for prepared_scenario in prepared_scenarios
    ]
    eir_baselines = prev_eir_artifact.predict(prev_eir_records)

    # 2: EIR -> HBR baseline
    eir_hbr_artifact = eir_models[INPUT_MODE_TO_REPO_IDS["eir"]]
    eir_hbr_records = [
        {
            **prepared_scenario.eir_model_features,
            "eir": eir_value,
        }
        for prepared_scenario, eir_value in zip(prepared_scenarios, eir_baselines)
    ]
    hbr_baselines = eir_hbr_artifact.predict(eir_hbr_records)

    # Step 3: apply mosquito delta (positive or negative)
    mosquito_deltas = [prepared_scenario.mosquito_density_change for prepared_scenario in prepared_scenarios]
    hbr_adjusted = hbr_baselines * (1 + np.array(mosquito_deltas))

    # Step 4: ratio approach — batch both HBR values in one call so they
    # share the same smooth PCHIP curve
    hbr_eir_artifact = eir_models[INPUT_MODE_TO_REPO_IDS["hbr"]]
    hbr_eir_records = [
        {
            **prepared_scenario.eir_model_features,
            "hbr_y9": hbr_value,
        }
        for hbr_values in (hbr_baselines, hbr_adjusted)
        for prepared_scenario, hbr_value in zip(prepared_scenarios, hbr_values)
    ]
    eir_from_hbr = hbr_eir_artifact.predict(hbr_eir_records)

    count = len(prepared_scenarios)
    eir_rt = eir_from_hbr[:count]
    eir_new_raw = eir_from_hbr[count:]

    # Step 5: multiplier applied to clean baseline
    multiplier = eir_new_raw / eir_rt
    eir_new = eir_baselines * multiplier

    return [
        {
            "eir_baseline": eir_baseline,
            "eir_new": eir_new,
            "eir_multiplier": multiplier,
            "hbr_baseline": hbr_baseline,
            "hbr_new": hbr_adjusted,
        }
        for  eir_baseline, eir_new, multiplier, hbr_baseline, hbr_adjusted in zip(
            eir_baselines, eir_new, multiplier, hbr_baselines, hbr_adjusted
        )
    ]
