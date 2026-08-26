"""End-to-end flow tests: prevalence -> EIR, and the mosquito-delta HBR pipeline.

The prevalence -> EIR flow exercises the bundled XGBoost models in src/estimint/data,
so it runs offline. The mosquito-delta pipeline uses the published RQS models, which
are downloaded (and then cached) from the estiMINT HuggingFace repo.
"""

import pytest

from estimint import (
    estimate_eir_with_mosquito_delta,
)
from estimint.eir_models import load_eir_models
from estimint.types import EirTarget, PreparedScenario

INTERVENTIONS = dict(
    dn0_use=0.33, Q0=0.87, phi_bednets=0.82,
    seasonal=0.0, itn_use=0.6, irs_use=0.0,
)

def _prepared(delta: float, prevalence: float = 0.30) -> PreparedScenario:
    """A prevalence-input scenario carrying a mosquito-density change."""
    return PreparedScenario(
        eir_target=EirTarget(prevalence, "prevalence"),
        mosquito_density_change=delta,
        eir_model_features=dict(INTERVENTIONS),
        summary_values={},
        emulator_covariates={},
    )


class TestMosquitoDelta:
    @pytest.fixture(scope="class")
    def models(self):
        return load_eir_models()

    def _run(self, models, delta):
        return estimate_eir_with_mosquito_delta([_prepared(delta)], eir_models=models)[0]

    def test_returns_expected_keys(self, models):
        res = self._run(models, 0.25)
        assert set(res) == {
            "eir_baseline", "eir_new", "eir_multiplier", "hbr_baseline", "hbr_new",
        }
        assert all(isinstance(value, float) for value in res.values())

    def test_zero_delta_is_identity(self, models):
        res = self._run(models, 0.0)
        assert res["eir_new"] == pytest.approx(res["eir_baseline"])
        assert res["eir_multiplier"] == pytest.approx(1.0)

    def test_more_mosquitoes_raises_eir(self, models):
        res = self._run(models, 0.25)
        assert res["eir_new"] > res["eir_baseline"]
        assert res["eir_multiplier"] > 1.0
        assert res["hbr_new"] > res["hbr_baseline"]

    def test_fewer_mosquitoes_lowers_eir(self, models):
        res = self._run(models, -0.50)
        assert res["eir_new"] < res["eir_baseline"]
        assert res["hbr_new"] < res["hbr_baseline"]

    def test_batch_is_monotonic_in_delta(self, models):
        # a single batched call handles every row and preserves input order
        deltas = [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
        results = estimate_eir_with_mosquito_delta([_prepared(d) for d in deltas], eir_models=models)
        assert len(results) == len(deltas)
        eir_new = [result["eir_new"] for result in results]
        assert eir_new == sorted(eir_new)
