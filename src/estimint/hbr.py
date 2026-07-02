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

from typing import Any

import pandas as pd

from .run import run_xgb_model


def estimate_eir_with_mosquito_delta(inputs: pd.DataFrame, *, models: dict[str, Any]) -> pd.DataFrame:
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
    features = [
        "dn0_use",
        "Q0",
        "phi_bednets",
        "seasonal",
        "itn_use",
        "irs_use",
    ]
    intervention_data = inputs[features]

    # Step 1: prevalence -> EIR baseline
    prevalence_data = intervention_data.assign(prev_y9=inputs["prevalence"].to_numpy())
    eir_baseline = run_xgb_model(prevalence_data, models["prevalence"])

    # Step 2: EIR -> HBR baseline
    eir_data = intervention_data.assign(eir=eir_baseline)
    hbr_baseline = run_xgb_model(eir_data, models["eir_to_hbr"])

    # Step 3: apply mosquito delta (positive or negative)
    hbr_new = hbr_baseline * (1 + inputs["mosquito_delta"].to_numpy())

    # Step 4: ratio approach — batch both HBR values in one call so they
    # share the same smooth PCHIP curve
    hbr_data = pd.concat(
        [
            intervention_data.assign(hbr_y9=hbr_baseline),
            intervention_data.assign(hbr_y9=hbr_new),
        ],
        ignore_index=True,
    )
    eir_from_hbr = run_xgb_model(hbr_data, models["hbr"])

    count = len(inputs)
    eir_rt = eir_from_hbr[:count]
    eir_new_raw = eir_from_hbr[count:]

    # Step 5: multiplier applied to clean baseline
    multiplier = eir_new_raw / eir_rt
    eir_new = eir_baseline * multiplier

    return pd.DataFrame(
        {
            "eir_baseline": eir_baseline,
            "eir_new": eir_new,
            "eir_multiplier": multiplier,
            "hbr_baseline": hbr_baseline,
            "hbr_new": hbr_new,
        },
        index=inputs.index,
    )
