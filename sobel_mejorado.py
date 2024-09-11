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
