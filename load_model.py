from keras.models import load_model, Model
import os

def model_fun() -> Model:
    try:
        modelo_path = os.path.join("modelo", "conv_MLP_84.h5")
        modelo = load_model(modelo_path)
        return modelo
    except Exception as e:
        print("Error al importar el modelo:", e)
        return None

