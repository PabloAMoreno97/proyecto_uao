from keras.models import load_model, Model
import os

def model_fun() -> Model:
    """
    Carga el modelo de neumonía desde la carpeta 'modelo'.

    Se utiliza os.path.join para asegurar compatibilidad
    tanto en Windows como en Linux/Docker.
    """
    try:
        # Construir la ruta del archivo del modelo de forma relativa y multiplataforma
        modelo_path = os.path.join("modelo", "conv_MLP_84.h5")

        # Cargar el modelo desde la ruta indicada
        modelo = load_model(modelo_path)

        return modelo
    except Exception as e:
        # Capturar y mostrar cualquier error en la carga del modelo
        print("Error al importar el modelo:", e)
        return None
