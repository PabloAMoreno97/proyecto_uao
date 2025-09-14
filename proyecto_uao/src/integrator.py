import os
import numpy as np
import pydicom

# Importamos las funciones individuales
from src.preprocess_img import preprocess
from src.load_model import model_fun
from src.grad_cam import grad_cam
from src.read_img import read_dicom_file


def predict_pneumonia(filepath):
    """
    Función principal que integra el flujo completo:
    1. Lee la imagen.
    2. Carga y predice con el modelo.
    3. Genera el heatmap de Grad-CAM.
    4. Devuelve los resultados.
    """
    try:
        # Se lee el archivo de imagen (DICOM o JPG/PNG)
        array, _ = read_dicom_file(filepath)

        # Preprocesar la imagen para que sea compatible con el modelo
        preprocessed_array = preprocess(array)

        # Cargar el modelo desde load_model.py
        model = model_fun()
        if model is None:
            raise RuntimeError("No se pudo cargar el modelo entrenado.")

        # Realizar la predicción
        predictions = model.predict(preprocessed_array)
        
        # Ahora, usamos np.argmax para obtener la clase con mayor probabilidad de las 3 salidas
        predicted_class_index = np.argmax(predictions[0])
        probabilities = predictions[0]

        # Mapeamos el índice a la etiqueta de texto
        # Asumiendo el orden de las etiquetas: [bacteriana, normal, viral]
        class_labels = ["Bacteriana", "Normal", "Viral"]
        label = class_labels[predicted_class_index]
        proba = probabilities[predicted_class_index] * 100

        # Generar heatmap con la imagen original
        heatmap = grad_cam(array)

        return label, proba, heatmap

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en la ruta: {filepath}")
        return None, None, None
    except Exception as e:
        print(f"❌ Ocurrió un error al procesar la imagen: {e}")
        return None, None, None