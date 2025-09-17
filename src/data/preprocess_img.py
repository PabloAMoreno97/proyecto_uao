import cv2
import numpy as np


def preprocess(array):
    """
    Preprocesa una imagen para el modelo de detección de neumonía.
    Pasos:
    1. Redimensiona la imagen a 512x512 píxeles.
    2. Convierte a escala de grises.
    3. Aplica ecualización adaptativa de histograma (CLAHE).
    4. Normaliza los valores a rango [0,1].
    5. Expande dimensiones para que sea compatible con TensorFlow/Keras.
    Parámetros
    ----------
    array : numpy.ndarray
        Imagen de entrada cargada con OpenCV.
    Retorna
    -------
    numpy.ndarray
        Imagen preprocesada con forma (1, 512, 512, 1).
    """
    # Redimensionar
    array = cv2.resize(array, (512, 512))

    # Convertir a escala de grises
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    # Aplicar ecualización adaptativa de histograma (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)

    # Normalizar a [0,1]
    array = array / 255.0

    # Expandir dimensiones para que el modelo lo acepte (batch, alto, ancho, canales)
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)

    return array