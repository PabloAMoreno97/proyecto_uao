import cv2
import numpy as np
from PIL import Image

def read_jpg_file(path):
    """
    Lee un archivo JPG y devuelve dos representaciones de la imagen.
    
    Parámetros
    ----------
    path : str
        Ruta al archivo de imagen en formato JPG.
    
    Retorna
    -------
    img2 : numpy.ndarray
        Imagen en formato NumPy normalizada a valores enteros de 0 a 255.
    img2show : PIL.Image.Image
        Objeto de imagen de PIL listo para ser mostrado o manipulado.
    """
    img = cv2.imread(path)  # Leer la imagen desde el archivo
    img_array = np.asarray(img)  # Convertir la imagen a un arreglo NumPy
    img2show = Image.fromarray(img_array)  # Crear versión PIL para mostrar
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)  # Normalizar entre 0–255 y regresar a uint8
    return img2, img2show


def read_dicom_file(path):
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show
