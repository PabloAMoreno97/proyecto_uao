from src.integrator import predict_pneumonia
import matplotlib.pyplot as plt
import cv2

# Reemplaza 'imagenes_prueba/normal.dcm' con la ruta de tu imagen de prueba
test_image_path = "imagenes_prueba/viral.dcm" 

print(f"Iniciando prueba de la función predict_pneumonia con: {test_image_path}")

# Llama a la función y captura los resultados
label, proba, heatmap = predict_pneumonia(test_image_path)

if label and proba is not None and heatmap is not None:
    print("\n✅ La función se ejecutó correctamente. Resultados:")
    print(f"   Etiqueta de la predicción: {label}")
    print(f"   Probabilidad: {proba:.2f}%")

    # Muestra la imagen con el mapa de calor
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title(f"Resultado: {label} ({proba:.2f}%)")
    plt.axis('off')  # Oculta los ejes
    plt.show()
else:
    print("\n❌ La función falló. Revisa los errores en la consola.")