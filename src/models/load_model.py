import os

import tensorflow as tf


def model_fun():
    """
    Carga el modelo de detección de neumonía desde un archivo.
    """
    try:
        # Volvemos a usar el modelo que ha demostrado ser funcional
        model_path = os.path.join('src/models', 'conv_MLP_84_converted.h5')
        print(f"✅ Modelo cargado correctamente desde: {model_path}")
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return None
