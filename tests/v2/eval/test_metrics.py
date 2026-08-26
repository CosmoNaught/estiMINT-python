"""
Tests for evaluation metric aggregation over a data loader.
"""

import numpy as np
import pytest

from estimint.v2.data.dataset import make_loader
from estimint.v2.eval.metrics import Metrics, compute_metrics, get_preds_targets


class ConstantOffset:
    """Stand-in ModelArtifact: predicts the first raw feature plus a fixed offset."""

    def __init__(self, offset=0.0):
        self.offset = offset

    def predict(self, X_raw):
        return X_raw[:, 0] + self.offset


@pytest.fixture
def records():
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 100, size=12).astype(np.float32)
    return [{"x_raw": np.array([v, 0.0], dtype=np.float32), "y_raw": v} for v in y]


@pytest.fixture
def loader(records):
    return make_loader(records, batch_size=4, shuffle=False)


class TestGetPredsTargets:
    def test_returns_one_value_per_record(self, loader, records):
        preds, targets = get_preds_targets(ConstantOffset(), loader)
        assert preds.shape == targets.shape == (len(records),)

    def test_targets_follow_loader_order(self, loader, records):
        _, targets = get_preds_targets(ConstantOffset(), loader)
        np.testing.assert_allclose(targets, [r["y_raw"] for r in records])

    def test_predictions_come_from_raw_features(self, loader, records):
        preds, _ = get_preds_targets(ConstantOffset(offset=2.0), loader)
        np.testing.assert_allclose(preds, [r["x_raw"][0] + 2.0 for r in records], rtol=1e-6)

    def test_dropped_remainder_shortens_the_result(self, records):
        loader = make_loader(records, batch_size=5, drop_remainder=True)
        preds, _ = get_preds_targets(ConstantOffset(), loader)
        assert preds.shape == (10,)


class TestComputeMetrics:
    def test_perfect_predictions_score_perfectly(self, loader):
        metrics = compute_metrics(ConstantOffset(), loader)
        assert isinstance(metrics, Metrics)
        assert metrics.mse == pytest.approx(0.0, abs=1e-8)
        assert metrics.rmse == pytest.approx(0.0, abs=1e-8)
        assert metrics.mae == pytest.approx(0.0, abs=1e-8)
        assert metrics.r2 == pytest.approx(1.0)

    def test_constant_offset_shows_up_as_bias(self, loader):
        metrics = compute_metrics(ConstantOffset(offset=3.0), loader)
        assert metrics.bias == pytest.approx(3.0, rel=1e-4)
        assert metrics.mae == pytest.approx(3.0, rel=1e-4)
        assert metrics.rmse == pytest.approx(3.0, rel=1e-4)

    def test_worse_predictions_lower_r2(self, records):
        better = compute_metrics(ConstantOffset(offset=1.0), make_loader(records, batch_size=4))
        worse = compute_metrics(ConstantOffset(offset=50.0), make_loader(records, batch_size=4))
        assert worse.r2 < better.r2
        assert worse.mse > better.mse
