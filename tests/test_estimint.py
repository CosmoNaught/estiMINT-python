"""
Basic tests for estiMINT package.
"""

import numpy as np
import pytest


class TestUtils:
    """Test utility functions."""
    
    def test_r2(self):
        from estimint import r2
        y = np.array([1, 2, 3, 4, 5])
        yhat = np.array([1.1, 2.0, 2.9, 4.0, 5.1])
        result = r2(y, yhat)
        assert 0.95 < result <= 1.0
    
    def test_rmse(self):
        from estimint import rmse
        y = np.array([1, 2, 3])
        yhat = np.array([1, 2, 3])
        assert rmse(y, yhat) == 0.0
        
        yhat2 = np.array([2, 3, 4])
        assert rmse(y, yhat2) == 1.0
    
    def test_mse(self):
        from estimint import mse
        y = np.array([1, 2, 3])
        yhat = np.array([2, 3, 4])
        assert mse(y, yhat) == 1.0
    
    def test_mae(self):
        from estimint import mae
        y = np.array([1, 2, 3])
        yhat = np.array([2, 3, 4])
        assert mae(y, yhat) == 1.0
    
    def test_median_ae(self):
        from estimint import median_ae
        y = np.array([1, 2, 3])
        yhat = np.array([2, 3, 4])
        assert median_ae(y, yhat) == 1.0
    
    def test_safe_div(self):
        from estimint import safe_div
        result = safe_div(np.array([1, 2]), np.array([0, 2]))
        assert result[1] == 1.0
        assert result[0] > 0  # Should not be inf
    
    def test_fit_predict_qmap(self):
        from estimint import fit_qmap_w, predict_qmap_w
        pred = np.array([1, 2, 3, 4, 5])
        obs = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        
        cal = fit_qmap_w(pred, obs)
        assert "xq" in cal
        assert "yq" in cal
        
        calibrated = predict_qmap_w(pred, cal)
        assert len(calibrated) == len(pred)
    
    def test_scale_pos(self):
        from estimint import scale_pos
        obs = np.array([1, 2, 3])
        pred = np.array([0.5, 1, 1.5])
        
        a = scale_pos(obs, pred)
        assert a > 0


class TestDataProcessing:
    """Test data processing functions."""
    
    def test_make_value_weights(self):
        from estimint import make_value_weights
        eir = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
        weights = make_value_weights(eir, digits=3)
        
        assert len(weights) == len(eir)
        assert np.isclose(np.mean(weights), 1.0)  # Normalized to mean=1


class TestRun:
    """Test model inference functions."""
    
    def test_global_model_functions(self):
        from estimint.run import set_global_model, get_global_model
        
        assert get_global_model() is None
        
        dummy_model = {"class": "estiMINT_model", "features": ["a", "b"]}
        set_global_model(dummy_model)
        
        assert get_global_model() is not None
        assert get_global_model()["class"] == "estiMINT_model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
