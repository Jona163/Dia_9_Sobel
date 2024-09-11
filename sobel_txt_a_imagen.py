# Autor: Jonathan Hernández
# Fecha: 10 Septiembre 2024
# Descripción: Código para procesamiento de imagenes con Sobel.
# GitHub: https://github.com/Jona163

import cv2
import numpy as np
import matplotlib.pyplot as plt

def leer_matriz_desde_txt(ruta_txt):
    """
    Lee una matriz desde un archivo de texto y la convierte en un array de NumPy.
    
    Parámetros:
    - ruta_txt (str): Ruta del archivo de texto que contiene la matriz.
    
    Retorna:
    - matriz (numpy array): Matriz cargada desde el archivo de texto.
    """
    matriz = np.loadtxt(ruta_txt, dtype=np.uint8)  # Cargar la matriz como un array de enteros
    return matriz
