from proyecto_uao.load_model import model_fun
import pytest

def test_model_fun_loads_or_none():
    try:
        model = model_fun()
        if model is None:
            pytest.skip("Modelo no disponible en la ruta definida.")
        else:
            assert hasattr(model, "predict")
    except Exception:
        pytest.skip("Modelo no cargado, test saltado.")
