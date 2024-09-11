# Autor: Jonathan Hernández
# Fecha: 10 Septiembre 2024
# Descripción: Código para procesamiento de imagenes con Sobel.
# GitHub: https://github.com/Jona163

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cargar_imagen(ruta):
    """
    Carga una imagen desde la ruta especificada y la convierte en escala de grises.
    
    Parámetros:
    - ruta (str): Ruta del archivo de imagen.
    
    Retorna:
    - imagen (numpy array): Imagen cargada en escala de grises.
    
    Lanza:
    - FileNotFoundError: Si el archivo no existe.
    - ValueError: Si el archivo no es una imagen válida o no se puede cargar.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"La ruta '{ruta}' no existe.")
    
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    
    if imagen is None:
        raise ValueError("No se pudo cargar la imagen. Asegúrate de que el archivo es una imagen válida.")
    
    return imagen

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
    - fase (numpy array): Fase o dirección del gradiente (en radianes).
    """
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)  # Filtro Sobel en dirección X
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)  # Filtro Sobel en dirección Y
    
    # Magnitud del gradiente combinando los valores de sobel_x y sobel_y
    magnitud = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitud = np.uint8(255 * magnitud / np.max(magnitud))  # Normalizar a un rango entre 0 y 255
    
    # Fase del gradiente (dirección de los bordes)
    fase = np.arctan2(sobel_y, sobel_x)
    
    return sobel_x, sobel_y, magnitud, fase

def mostrar_resultados(imagen, sobel_x, sobel_y, magnitud, fase):
    """
    Muestra la imagen original junto con los resultados del filtro Sobel (gradientes y magnitud) en una cuadrícula.
    
    Parámetros:
    - imagen (numpy array): Imagen original en escala de grises.
    - sobel_x (numpy array): Gradiente en la dirección X.
    - sobel_y (numpy array): Gradiente en la dirección Y.
    - magnitud (numpy array): Magnitud combinada del gradiente.
    - fase (numpy array): Fase o dirección del gradiente.
    """
