"""
Human Biting Rate (HBR) pipeline for estiMINT.

Answers the question: "What happens to EIR if mosquito density changes by X%?"

Pipeline:
1. prev_y9 + interventions -> EIR_baseline         (prev_y9 -> eir model)
2. EIR_baseline + interventions -> HBR_baseline    (eir -> hbr_y9 model)
3. HBR_new = HBR_baseline * (1 + mosquito_delta)   (user's mosquito change, pos or neg)
4. EIR predicted at both HBR values                (hbr_y9 -> eir model)
5. EIR_new = EIR_baseline * (EIR_new_raw / EIR_roundtrip)

Step 5 applies the *ratio* of the two step-4 predictions rather than using EIR_new_raw
directly, so the result stays anchored to the cleaner step-1 baseline and any bias in
the EIR -> HBR -> EIR round trip cancels out.
"""

import numpy as np

from .eir_models import EirModels, predict_from_measurements
from .types import PreparedScenario


def estimate_eir_with_mosquito_delta(
    prepared_scenarios: list[PreparedScenario], *, eir_models: EirModels
) -> list[dict[str, float]]:
    """
    Estimate the new EIR after a change in mosquito density, for a batch of scenarios.

    Parameters
    ----------
    prepared_scenarios : list[PreparedScenario]
        Scenarios with ``input_mode == "prevalence"``. Each supplies:

        - ``eir_target.input_value`` : baseline malaria prevalence (prev_y9), e.g. 0.30 for 30%.
        - ``mosquito_density_change`` : fractional change in mosquito density, e.g. 0.10
          for +10%, -0.50 for -50%. Must be > -1.
        - ``eir_model_features`` : the intervention covariates shared by all three models
          (``dn0_use``, ``Q0``, ``phi_bednets``, ``seasonal``, ``itn_use``, ``irs_use``).

    eir_models : EirModels
        Artifacts from :func:`estimint.eir_models.load_eir_models`.

    Returns
    -------
    list[dict[str, float]]
        One entry per scenario, in input order, with keys:

        - ``eir_baseline`` : baseline EIR from prevalence.
        - ``eir_new`` : new EIR after the mosquito density change.
        - ``eir_multiplier`` : ratio of new EIR to baseline.
        - ``hbr_baseline`` : estimated baseline HBR.
        - ``hbr_new`` : HBR after the mosquito density change.

    Examples
    --------
    >>> from estimint.eir_models import load_eir_models
    >>> from estimint.hbr import estimate_eir_with_mosquito_delta
    >>> results = estimate_eir_with_mosquito_delta(prepared_scenarios, eir_models=load_eir_models())
    >>> results[0]["eir_new"]
    """
    # Step 1: prevalence -> EIR baseline
    prevalences = [prepared_scenario.eir_target.input_value for prepared_scenario in prepared_scenarios]
    eir_baselines = predict_from_measurements(eir_models, "prev_y9", prepared_scenarios, prevalences)

    # Step 2: EIR -> HBR baseline
    hbr_baselines = predict_from_measurements(eir_models, "eir", prepared_scenarios, eir_baselines)

    # Step 3: apply mosquito delta (positive or negative)
    mosquito_deltas = np.array(
        [prepared_scenario.mosquito_density_change for prepared_scenario in prepared_scenarios]
    )
    hbr_new = hbr_baselines * (1 + mosquito_deltas)

    # Step 4: both HBR values go back through the HBR -> EIR model in one batched call
    eir_from_hbr = predict_from_measurements(
        eir_models,
        "hbr_y9",
        [*prepared_scenarios, *prepared_scenarios],
        np.concatenate([hbr_baselines, hbr_new]),
    )
    eir_roundtrip, eir_new_raw = np.split(eir_from_hbr, 2)

    # Step 5: multiplier applied to the clean baseline
    eir_multipliers = eir_new_raw / eir_roundtrip
    eir_news = eir_baselines * eir_multipliers

    return [
        {
            "eir_baseline": float(eir_baseline),
            "eir_new": float(eir_new),
            "eir_multiplier": float(eir_multiplier),
            "hbr_baseline": float(hbr_baseline),
            "hbr_new": float(hbr_adjusted),
        }
        for eir_baseline, eir_new, eir_multiplier, hbr_baseline, hbr_adjusted in zip(
            eir_baselines, eir_news, eir_multipliers, hbr_baselines, hbr_new, strict=True
        )
    ]
