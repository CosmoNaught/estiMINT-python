"""Tests for the run_scenarios pipeline.

The estiMINT half (_estimate_eir_many) is tested offline against the bundled
models. The full run_scenarios call also runs the stateMINT emulator, so it is
skipped unless stateMINT is installed.
"""

from typing import Any

import numpy as np
import pytest

from estimint.scenarios import (
    _PreparedScenario,
    _apply_mosquito_delta_batch,
    _classify_prepared_scenario,
    _estimate_eir,
    _load_eir_hbr_models,
    _prepare_scenario_inputs,
    run_scenarios,
)
from estimint.types import EirTarget, Scenario

INTV: dict[str, Any] = dict(Q0=0.87, phi=0.82, seasonal=0.0, irs=0.0)


def mk(**kwargs: Any) -> Scenario:
    defaults: dict[str, Any] = dict(name="t", res_use=0.0, **INTV)
    input_mode = kwargs.pop("input")
    input_value = kwargs.pop("value")
    defaults.update(kwargs)
    return Scenario(eir_target=EirTarget(input_value, input_mode), **defaults)


def _estimate_eir_single(scenario: Scenario, eir_models: dict[str, Any]) -> _PreparedScenario:
    """Estimate EIR for a single scenario via the batch estimator."""
    return _estimate_eir([scenario], eir_models)[0]


@pytest.fixture(scope="module")
def est():
    return _load_eir_hbr_models()


class TestEstimateEir:
    def test_prevalence_input(self, est):
        out = _estimate_eir_single(mk(input="prevalence", value=0.30), est)
        assert out.summary_values["eir_baseline"] > 0
        assert out.emulator_covariates["eir"] == out.summary_values["eir_final"]
        # no nets, no delta
        assert out.summary_values["dn0_use"] == 0.0
        assert np.isnan(out.summary_values["hbr_baseline"])

    def test_eir_input_passes_through(self, est):
        out = _estimate_eir_single(mk(input="eir", value=20.0), est)
        assert out.summary_values["eir_baseline"] == 20.0
        assert out.summary_values["eir_final"] == 20.0

    def test_hbr_input(self, est):
        out = _estimate_eir_single(mk(input="hbr", value=250000.0), est)
        assert out.summary_values["eir_baseline"] > 0

    def test_bednet_mix_matches_minte(self, est):
        # net-type usage mix feeds calculate_dn0 directly, same as minte; itn_use
        # is the sum of pyrethroid shares (NOT rescaled by coverage again).
        out = _estimate_eir_single(mk(input="prevalence", value=0.30, py_only=0.70, py_pbo=0.30, res_use=0.30), est)
        assert out.summary_values["dn0_use"] > 0
        assert out.summary_values["itn_use"] == pytest.approx(1.00)
        assert out.summary_values["net"] == "py_only+py_pbo"

    def test_future_net_switch_is_separate_leg(self, est):
        # net_type_future/itn_future drive the future leg, not current nets;
        # the future leg shares the same res_use as current (no res_future field).
        scenario_estimate = _estimate_eir_single(
            mk(
                input="prevalence",
                value=0.30,
                py_only=0.50,
                res_use=0.30,
                net_type_future="pyrethroid_pbo",
                itn_future=0.70,
            ),
            est,
        )
        summary_values = scenario_estimate.summary_values
        assert summary_values["itn_use"] == pytest.approx(0.50)  # current: py_only=0.50
        assert summary_values["itn_future"] == pytest.approx(0.70)  # future: pbo=0.70
        assert summary_values["dn0_use"] != summary_values["dn0_future"]
        assert summary_values["net"] == "py_only" and summary_values["net_future"] == "pyrethroid_pbo"

    def test_future_without_net_type_is_zeroed(self, est):
        # no net_type_future named -> future leg is zeroed, even if itn_future is set
        # (no carry-forward of the current mix; this is intentional, not a default)
        summary_values = _estimate_eir_single(
            mk(input="prevalence", value=0.30, py_pbo=0.80, res_use=0.30, itn_future=0.70), est
        ).summary_values
        assert summary_values["dn0_future"] == 0.0 and summary_values["itn_future"] == 0.0
        assert summary_values["net_future"] == "none"

    def test_future_nets_removed(self, est):
        # itn_future == 0 removes nets in the future leg
        summary_values = _estimate_eir_single(
            mk(input="prevalence", value=0.30, py_only=0.60, res_use=0.30, itn_future=0.0), est
        ).summary_values
        assert summary_values["itn_use"] == pytest.approx(0.60)
        assert summary_values["dn0_future"] == 0.0 and summary_values["itn_future"] == 0.0

    def test_ppf_boosts_lsm(self, est):
        # PPF nets add larviciding to LSM (minte: py_ppf * 0.248)
        out = _estimate_eir_single(mk(input="eir", value=15.0, py_ppf=0.50, res_use=0.30, lsm=0.10), est)
        assert out.emulator_covariates["lsm"] == pytest.approx(0.50 * 0.248 + 0.10)

    def test_irs_future_and_routine_are_inputs(self, est):
        # irs_future and routine are separate scenario inputs, like minte
        emulator_covariates = _estimate_eir_single(
            mk(input="eir", value=15.0, irs=0.40, irs_future=0.10, routine=0.25), est
        ).emulator_covariates
        assert emulator_covariates["irs_use"] == 0.40 and emulator_covariates["irs_future"] == 0.10
        assert emulator_covariates["routine"] == 0.25

    def test_irs_future_and_routine_defaults(self, est):
        # irs_future is a static dataclass default (0.0); it does NOT carry irs
        # forward automatically, even when irs is nonzero.
        emulator_covariates = _estimate_eir_single(mk(input="eir", value=15.0, irs=0.40), est).emulator_covariates
        assert emulator_covariates["irs_future"] == 0.0
        assert emulator_covariates["irs_use"] == 0.40
        assert emulator_covariates["routine"] == 0.0

    def test_mosquito_delta_direction(self, est):
        up = _estimate_eir_single(mk(input="prevalence", value=0.30, mosquito_delta=0.25), est)
        down = _estimate_eir_single(mk(input="prevalence", value=0.30, mosquito_delta=-0.50), est)
        assert up.summary_values["eir_final"] > up.summary_values["eir_baseline"]
        assert down.summary_values["eir_final"] < down.summary_values["eir_baseline"]
        assert up.summary_values["hbr_new"] > up.summary_values["hbr_baseline"]

    def test_covariate_dict_keys(self, est):
        out = _estimate_eir_single(mk(input="eir", value=15.0, lsm=0.3), est)
        assert set(out.emulator_covariates) == {
            "eir",
            "dn0_use",
            "dn0_future",
            "Q0",
            "phi_bednets",
            "seasonal",
            "routine",
            "itn_use",
            "irs_use",
            "itn_future",
            "irs_future",
            "lsm",
        }
        assert out.emulator_covariates["lsm"] == 0.3
        assert out.emulator_covariates["dn0_future"] == out.emulator_covariates["dn0_use"]

    def test_bad_input_raises(self, est):
        with pytest.raises(ValueError, match="input_mode"):
            _estimate_eir_single(mk(input="nope", value=1.0), est)

    def test_batch_mixed_input_modes(self, est):
        # one batched call must estimate every input mode and preserve order
        scenarios = [
            mk(name="prev", input="prevalence", value=0.30),
            mk(name="eir", input="eir", value=20.0),
            mk(name="hbr", input="hbr", value=250000.0),
            mk(name="prev_delta", input="prevalence", value=0.30, mosquito_delta=0.25),
        ]
        outs = _estimate_eir(scenarios, est)
        assert [out.summary_values["name"] for out in outs] == ["prev", "eir", "hbr", "prev_delta"]
        assert all(out.summary_values["eir_baseline"] > 0 for out in outs)
        assert outs[1].summary_values["eir_final"] == 20.0  # explicit eir passes through
        # mosquito-delta scenario shifts eir away from baseline and fills hbr
        assert outs[3].summary_values["eir_final"] != outs[3].summary_values["eir_baseline"]
        assert outs[3].summary_values["hbr_new"] > 0


class TestClassifyPreparedScenario:
    def _make_prepared(self, input_mode: str, mosquito_density_change: float = 0.0) -> _PreparedScenario:
        from typing import cast
        from estimint.types import Input_Mode

        return _PreparedScenario(
            eir_target=EirTarget(input_value=10.0, input_mode=cast(Input_Mode, input_mode)),
            mosquito_density_change=mosquito_density_change,
            eir_model_features={},
            summary_values={},
            emulator_covariates={},
        )

    def test_eir_mode(self):
        assert _classify_prepared_scenario(self._make_prepared("eir")) == "eir"

    def test_eir_mode_ignores_mosquito_delta(self):
        # eir input always passes through, even if mosquito_delta is set
        assert _classify_prepared_scenario(self._make_prepared("eir", mosquito_density_change=0.5)) == "eir"

    def test_prevalence_without_delta(self):
        assert _classify_prepared_scenario(self._make_prepared("prevalence")) == "prevalence"

    def test_prevalence_with_delta(self):
        assert (
            _classify_prepared_scenario(self._make_prepared("prevalence", mosquito_density_change=0.25))
            == "mosquito_delta"
        )

    def test_hbr_mode(self):
        assert _classify_prepared_scenario(self._make_prepared("hbr")) == "hbr"


class TestApplyMosquitoDeltaBatch:
    def test_fills_eir_and_hbr_estimates(self, est):
        prepared = _prepare_scenario_inputs(mk(input="prevalence", value=0.30, mosquito_delta=0.25))
        _apply_mosquito_delta_batch([prepared], est)
        assert prepared.summary_values["eir_baseline"] > 0
        assert prepared.summary_values["eir_final"] > prepared.summary_values["eir_baseline"]
        assert prepared.summary_values["hbr_new"] > prepared.summary_values["hbr_baseline"]
        assert prepared.emulator_covariates["eir"] == prepared.summary_values["eir_final"]

    def test_negative_delta_lowers_eir(self, est):
        prepared = _prepare_scenario_inputs(mk(input="prevalence", value=0.30, mosquito_delta=-0.50))
        _apply_mosquito_delta_batch([prepared], est)
        assert prepared.summary_values["eir_final"] < prepared.summary_values["eir_baseline"]


class TestRunScenariosFullPipeline:
    def test_end_to_end(self):
        pytest.importorskip("stateMINT", reason="stateMINT not installed")
        df = run_scenarios(
            [
                Scenario(
                    name="prev+delta",
                    eir_target=EirTarget(0.30, "prevalence"),
                    py_only=0.60,
                    res_use=0.55,
                    net_type_future="pyrethroid_pbo",
                    itn_future=0.85,
                    Q0=0.90,
                    phi=0.85,
                    seasonal=1,
                    irs=0.40,
                    mosquito_delta=0.60,
                ),
                Scenario(
                    name="eir", eir_target=EirTarget(20.0, "eir"), res_use=0.0, Q0=0.88, phi=0.78, seasonal=1, irs=0.60
                ),
            ]
        )
        assert len(df) == 2
        assert {"eir_final", "prev_y9", "prevalence", "cases"} <= set(df.columns)
        assert len(df.iloc[0]["prevalence"]) == 157
        assert (df["cases"].iloc[0] >= 0).all()
