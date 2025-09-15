import numpy as np
from proyecto_uao.preprocess_img import preprocess

def test_preprocess_output_shape():
    dummy = np.zeros((600, 800, 3), dtype=np.uint8)  # Imagen de prueba
    processed = preprocess(dummy)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (1, 512, 512, 1)  # batch, h, w, channel

def test_preprocess_normalization():
    dummy = np.ones((512, 512, 3), dtype=np.uint8) * 255
    processed = preprocess(dummy)
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0
