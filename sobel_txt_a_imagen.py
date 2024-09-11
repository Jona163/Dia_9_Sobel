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


def mostrar_imagenes(imagen_original, sobel_x, sobel_y, magnitud):
    """
    Muestra la imagen original reconstruida y los resultados del filtro Sobel en una cuadrícula.
    
    Parámetros:
    - imagen_original (numpy array): Imagen original reconstruida desde el archivo.
    - sobel_x (numpy array): Gradiente en la dirección X.
    - sobel_y (numpy array): Gradiente en la dirección Y.
    - magnitud (numpy array): Magnitud combinada del gradiente.
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(imagen_original, cmap='gray')
    plt.title('Imagen Original Reconstruida')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(sobel_y, cmap='gray')
    plt.title('Sobel Y')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(magnitud, cmap='gray')
    plt.title('Magnitud del Gradiente (Sobel)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
