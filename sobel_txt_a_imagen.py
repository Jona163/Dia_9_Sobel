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

def aplicar_sobel(imagen):
    """
    Aplica el filtro de Sobel a la imagen en ambas direcciones, X y Y.
    También calcula la magnitud y la fase (dirección) del gradiente.
    
    Parámetros:
    - imagen (numpy array): Imagen en escala de grises.
    
    Retorna:
    - sobel_x (numpy array): Gradiente de la imagen en la dirección X.
    - sobel_y (numpy array): Gradiente de la imagen en la dirección Y.
    - magnitud (numpy array): Magnitud combinada de los gradientes X y Y.
    """
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)  # Filtro Sobel en dirección X
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)  # Filtro Sobel en dirección Y
    
    # Magnitud del gradiente combinando los valores de sobel_x y sobel_y
    magnitud = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitud = np.uint8(255 * magnitud / np.max(magnitud))  # Normalizar a un rango entre 0 y 255
    
    return sobel_x, sobel_y, magnitud
