# Autor: Jonathan Hernández
# Fecha: 10 Septiembre 2024
# Descripción: Código para procesamiento de imagenes con Sobel.
# GitHub: https://github.com/Jona163

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cargar_imagen(ruta):
    """Carga una imagen en escala de grises y gestiona errores."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"La ruta '{ruta}' no existe.")
    
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    
    if imagen is None:
        raise ValueError("No se pudo cargar la imagen. Asegúrate de que el archivo es una imagen válida.")
    
    return imagen

def aplicar_sobel(imagen):
    """Aplica los filtros de Sobel en las direcciones X, Y y calcula la magnitud y fase."""
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitud del gradiente
    magnitud = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitud = np.uint8(255 * magnitud / np.max(magnitud))  # Normalizar
    
    # Fase del gradiente (dirección del borde)
    fase = np.arctan2(sobel_y, sobel_x)
    
    return sobel_x, sobel_y, magnitud, fase

def mostrar_resultados(imagen, sobel_x, sobel_y, magnitud, fase):
    """Muestra la imagen original y los resultados del filtro Sobel en una cuadrícula."""
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(sobel_y, cmap='gray')
    plt.title('Sobel Y')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(magnitud, cmap='gray')
    plt.title('Magnitud del Gradiente')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(fase, cmap='hsv')
    plt.title('Fase del Gradiente (Dirección)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def guardar_resultado(matriz, ruta_salida):
    """Guarda la matriz resultante de Sobel en un archivo .txt."""
    np.savetxt(ruta_salida, matriz, fmt='%d')
    print(f"Matriz guardada en: {ruta_salida}")

def analisis_imagen_sobel(ruta_imagen, ruta_salida=None):
    """Pipeline completo de análisis de imagen con el filtro de Sobel."""
    try:
        # Cargar la imagen
        imagen = cargar_imagen(ruta_imagen)
        
        # Aplicar el filtro Sobel
        sobel_x, sobel_y, magnitud, fase = aplicar_sobel(imagen)
        
        # Mostrar los resultados
        mostrar_resultados(imagen, sobel_x, sobel_y, magnitud, fase)
        
        # Guardar la matriz si se proporciona una ruta de salida
        if ruta_salida:
            guardar_resultado(magnitud, ruta_salida)

    except Exception as e:
        print(f"Error: {e}")

# Ejemplo de uso
ruta_imagen = 'chems.jpeg'  # ruta de tu imagen
ruta_salida = 'resultado_sobel.txt'  # guardar la matriz
analisis_imagen_sobel(ruta_imagen, ruta_salida)
