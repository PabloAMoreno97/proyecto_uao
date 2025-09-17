import cv2
import numpy as np
import tensorflow as tf

from data.preprocess_img import preprocess
from models.load_model import model_fun


def grad_cam(array):
    """
    Genera un heatmap con Grad-CAM para visualizar las regiones de la imagen
    que más influyen en la predicción del modelo.

    Args:
        array (numpy.ndarray): Imagen original cargada.

    Returns:
        numpy.ndarray: Imagen con el heatmap superpuesto.
    """
    # Preprocesar la imagen
    img = preprocess(array)

    # Cargar el modelo
    model = model_fun()

    # Predicciones
    preds = model.predict(img)
    argmax = np.argmax(preds[0])

    # Se crea un modelo intermedio para obtener los tensores necesarios
    grad_model = tf.keras.Model(
        [model.inputs], [model.get_layer("conv10_thisone").output, model.output]
    )
    
    with tf.GradientTape() as tape:
        # Se obtiene el output como una lista y luego se extraen los tensores
        outputs = grad_model(img)
        last_conv_layer_output = outputs[0]
        predictions = outputs[1]
        
        # Corregido: Se asegura que 'predictions' sea un tensor
        if isinstance(predictions, list):
            predictions = predictions[0]

        # Se obtiene el índice de la clase predicha para la imagen actual
        target_class_prediction = predictions[:, argmax]

    # Calcular gradientes de la clase predicha con respecto al output de la última capa conv
    grads = tape.gradient(target_class_prediction, last_conv_layer_output)
    
    # Se obtiene el gradiente promedio por cada canal de la última capa conv
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Se asegura que la capa de salida sea un tensor
    last_conv_layer_output = last_conv_layer_output[0]

    # Multiplicar cada canal por el gradiente promedio
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Ahora sí se puede convertir a numpy porque es un tensor
    heatmap = heatmap.numpy()

    # Crear el heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1  # Normalizar

    # Redimensionar al tamaño de la imagen original
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superponer el heatmap sobre la imagen original
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)

    return superimposed_img[:, :, ::-1]
