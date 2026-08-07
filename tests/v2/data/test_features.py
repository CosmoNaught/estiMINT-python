"""
Tests for the v2 feature naming and standardization helpers.
"""

import numpy as np
import pytest

from estimint.v2.data.features import FEATURES_BASE, StandardScaler, get_features


class TestGetFeatures:
    @pytest.mark.parametrize("predictor", ["prev_y9", "eir", "hbr_y9"])
    def test_predictor_leads_the_feature_list(self, predictor):
        features = get_features(predictor)
        assert features[0] == predictor
        assert features[1:] == FEATURES_BASE

    def test_feature_names_are_unique(self):
        features = get_features("prev_y9")
        assert len(set(features)) == len(features)


class TestStandardScaler:
    @pytest.fixture
    def X(self):
        return np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    def test_unfitted_scaler_raises(self, X):
        scaler = StandardScaler()
        assert not scaler.is_fitted
        with pytest.raises(ValueError, match="not fitted"):
            scaler.transform(X)
        with pytest.raises(ValueError, match="not fitted"):
            scaler.inverse_transform(X)

    def test_fit_transform_standardizes_columns(self, X):
        Z = StandardScaler().fit_transform(X)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(Z.std(axis=0), 1.0)

    def test_inverse_transform_round_trips(self, X):
        scaler = StandardScaler().fit(X)
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(X)), X)

    def test_constant_column_does_not_divide_by_zero(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaler = StandardScaler().fit(X)
        assert scaler.scale_[0] == 1.0
        assert np.all(np.isfinite(scaler.transform(X)))

    def test_fit_returns_self(self, X):
        scaler = StandardScaler()
        assert scaler.fit(X) is scaler
        assert scaler.is_fitted

    def test_transform_uses_train_statistics(self, X):
        scaler = StandardScaler().fit(X)
        # a row equal to the training mean maps to zero
        np.testing.assert_allclose(scaler.transform(X.mean(axis=0)[None, :]), 0.0, atol=1e-12)
