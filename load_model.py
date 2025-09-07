from keras.models import load_model, Model


def model_fun() -> Model:
    try:
        modelo = load_model("modelo\conv_MLP_84.h5")
        return modelo
    except Exception as e:
        print("Error al importar el modelo:", e)
