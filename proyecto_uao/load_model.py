from keras.models import load_model, Model
import os

def model_fun() -> Model:
    try:
        # Usar el modelo convertido para evitar incompatibilidades
        modelo_path = os.path.join("modelo", "conv_MLP_84_converted.h5")
        modelo = load_model(modelo_path, compile=False)  # compile=False = más seguro
        return modelo
    except Exception as e:
        print("Error al importar el modelo:", e)
        return None
