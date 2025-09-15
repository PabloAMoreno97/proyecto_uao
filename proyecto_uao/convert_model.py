from tensorflow.keras.models import load_model
import os

# Ruta al modelo viejo
old_model_path = os.path.join("modelo", "conv_MLP_84.h5")

print(f"Cargando modelo desde {old_model_path}...")
# 🚀 Cargar sin compilar para evitar el error "reduction=auto"
model = load_model(old_model_path, compile=False)

# Nueva ruta para guardar el modelo convertido
new_model_path = os.path.join("modelo", "conv_MLP_84_converted.h5")

# Guardar el modelo en versión compatible
model.save(new_model_path)

print(f"✅ Modelo convertido y guardado en {new_model_path}")
