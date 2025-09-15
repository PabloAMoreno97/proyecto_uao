import numpy as np

from grad_cam import grad_cam
from load_model import model_fun
from preprocess_img import preprocess


def predict(array):
    """
    Procesa la imagen, hace la predicción con el modelo y genera el Grad-CAM.
    
    Args:
        array (np.ndarray): Imagen procesada en formato NumPy.
    
    Returns:
        tuple: (label, proba, heatmap)
    """
    # 1. Preprocesar la imagen (convertir a batch)
    batch_array_img = preprocess(array)

    # Aseguramos que sea un numpy array
    batch_array_img = np.array(batch_array_img)

    # Depuración: ver tipo y forma del input al modelo
    print("DEBUG - tipo preprocess:", type(batch_array_img))
    print("DEBUG - shape preprocess:", getattr(batch_array_img, "shape", None))

    # 2. Cargar modelo
    model = model_fun()
    if model is None:
        print("Error: el modelo no se pudo cargar")
        return None, None, None

    # 3. Hacer predicción con el modelo
    preds = model.predict(batch_array_img)   # <<--- ESTA LÍNEA ERA LA QUE FALTABA
    preds = np.array(preds)  # convertir explícitamente a ndarray

    # Depuración: inspeccionar resultados
    print("DEBUG - preds type:", type(preds))
    print("DEBUG - preds shape:", preds.shape)
    print("DEBUG - preds:", preds)

    # preds debe ser algo como [[0.1, 0.8, 0.1]]
    if preds.ndim == 2 and preds.shape[0] == 1:
        prediction = int(np.argmax(preds[0]))
        proba = float(np.max(preds[0]) * 100)
    else:
        raise ValueError(f"Formato inesperado de predicción: {preds.shape}")

    # 4. Asignar etiqueta
    labels = ["bacteriana", "normal", "viral"]
    label = labels[prediction] if prediction < len(labels) else "desconocido"

    # 5. Generar Grad-CAM
    heatmap = grad_cam(array)

    return label, proba, heatmap