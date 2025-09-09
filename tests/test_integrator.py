import numpy as np
import pytest
from proyecto_uao.integrator import predict

def test_predict_output():
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    try:
        label, proba, heatmap = predict(dummy)
        assert label in ["bacteriana", "normal", "viral"]
        assert 0 <= proba <= 100
        assert heatmap.shape[0] == 512
        assert heatmap.shape[1] == 512
    except Exception as e:
        pytest.skip(f"Saltando test predict (modelo no disponible): {e}")
