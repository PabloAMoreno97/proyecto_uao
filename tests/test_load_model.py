from src.models.load_model import model_fun


def test_model_fun_loads():
    model = model_fun()
    assert hasattr(model, "predict")
