import pydicom
import numpy as np
import cv2  # Agregado
from PIL import Image  # Agregado


def read_jpg_file(path):
    """
    Lee un archivo de imagen JPG o PNG y lo convierte a un formato compatible.
    
    Args:
        path (str): La ruta del archivo de imagen.
        
    Returns:
        tuple: Una tupla con el array de la imagen y un objeto de imagen para mostrar.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer el archivo de imagen: {path}")
    
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show


def read_dicom_file(path):
    """
    Lee un archivo DICOM y extrae el array de píxeles.
    
    Args:
        path (str): La ruta del archivo DICOM.
        
    Returns:
        tuple: Una tupla con el array de píxeles y un objeto de imagen para mostrar.
    """
    # Corregido: 'dicom' se cambió a 'pydicom'
    img = pydicom.dcmread(path)
    img_array = img.pixel_array
    
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    
    return img_RGB, img2show