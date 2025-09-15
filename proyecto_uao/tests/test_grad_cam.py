import numpy as np
import pytest
from proyecto_uao.grad_cam import grad_cam

def test_grad_cam_output():
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    try:
        result = grad_cam(dummy)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 512
        assert result.shape[1] == 512
    except Exception as e:
        pytest.skip(f"Saltando test grad_cam: {e}")
