import cv2
import numpy as np


def preprocess(array):

    """
    Preprocesa una imagen para su uso en un modelo de inferencia.
    Redimensiona la imagen a 512×512 píxeles, la convierte de BGR a
    escala de grises, aplica CLAHE para realzar el contraste, normaliza
    los valores de píxel a [0.0, 1.0] y expande las dimensiones para
    producir un tensor 4D con forma (1, 512, 512, 1).

    Parámetros
    ----------
    array : numpy.ndarray
        Imagen de entrada en formato BGR, con forma (alto, ancho, 3)
        y tipo uint8.

    Retorna
    -------
    numpy.ndarray
        Imagen preprocesada con forma (1, 512, 512, 1) y tipo float32.
    """

    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array
