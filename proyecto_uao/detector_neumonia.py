#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import tensorflow as tf
from integrator import predict
from read_img import read_dicom_file

# Deshabilitar la ejecución ávida para compatibilidad con TensorFlow 1.x
# tf.compat.v1.disable_eager_execution()

def process_image(filepath):
    """
    Procesa una imagen DICOM o JPG/PNG y predice la neumonía.
    
    Args:
        filepath (str): La ruta del archivo de imagen a procesar.
    
    Returns:
        tuple: Una tupla con el resultado (etiqueta, probabilidad, heatmap).
    """
    try:
        # Leer el archivo de imagen
        array, _ = read_dicom_file(filepath)
        
        # Realizar la predicción
        label, proba, heatmap = predict(array)
        
        # Guardar el resultado en un archivo CSV
        with open("historial.csv", "a", newline="") as csvfile:
            w = csv.writer(csvfile, delimiter="-")
            # En un entorno de línea de comandos, no hay ID de paciente, así que se usa "N/A"
            w.writerow(["N/A", label, "{:.2f}".format(proba) + "%"])
            
        return label, proba, heatmap
        
    except FileNotFoundError:
        print(f"Error: El archivo no fue encontrado en la ruta especificada: {filepath}")
        return None, None, None
    except Exception as e:
        print(f"Ocurrió un error al procesar la imagen: {e}")
        return None, None, None

if __name__ == "__main__":
    # Configurar el analizador de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Herramienta para la detección de neumonía desde la línea de comandos.")
    parser.add_argument("filepath", type=str, help="La ruta completa al archivo de imagen DICOM, JPG o PNG.")
    
    # Analizar los argumentos
    args = parser.parse_args()
    
    print("Iniciando el proceso de detección de neumonía...")
    
    # Llamar a la función principal con la ruta del archivo proporcionada
    label, proba, heatmap = process_image(args.filepath)
    
    if label:
        print("\n--- Resultados de la predicción ---")
        print(f"Resultado: {label}")
        print(f"Probabilidad: {proba:.2f}%")
    else:
        print("El proceso de predicción ha fallado. Por favor, revisa los errores anteriores.")
