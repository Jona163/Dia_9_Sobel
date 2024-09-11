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
    plt.figure(figsize=(14, 10))  # Establece el tamaño de la figura
    
    # Mostrar la imagen original
    plt.subplot(2, 3, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    # Mostrar el gradiente en la dirección X (Sobel X)
    plt.subplot(2, 3, 2)
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Sobel X')
    plt.axis('off')

    # Mostrar el gradiente en la dirección Y (Sobel Y)
    plt.subplot(2, 3, 3)
    plt.imshow(sobel_y, cmap='gray')
    plt.title('Sobel Y')
    plt.axis('off')

    # Mostrar la magnitud del gradiente (combinación de Sobel X y Y)
    plt.subplot(2, 3, 4)
    plt.imshow(magnitud, cmap='gray')
    plt.title('Magnitud del Gradiente')
    plt.axis('off')

    # Mostrar la fase del gradiente (dirección de los bordes) en un mapa de colores
    plt.subplot(2, 3, 5)
    plt.imshow(fase, cmap='hsv')  # El colormap 'hsv' ayuda a visualizar direcciones
    plt.title('Fase del Gradiente (Dirección)')
    plt.axis('off')

    plt.tight_layout()  # Ajusta el espacio entre las subplots
    plt.show()  # Muestra todas las imágenes en una ventana

def guardar_resultado(matriz, ruta_salida):
    """
    Guarda la matriz de la imagen procesada (magnitud del gradiente) en un archivo de texto.
    
    Parámetros:
    - matriz (numpy array): Matriz de la imagen procesada.
    - ruta_salida (str): Ruta donde se guardará el archivo.
    """
    np.savetxt(ruta_salida, matriz, fmt='%d')  # Guardar la matriz en un archivo de texto
    print(f"Matriz guardada en: {ruta_salida}")

def analisis_imagen_sobel(ruta_imagen, ruta_salida=None):
    """
    Ejecuta todo el pipeline de procesamiento de imagen con el filtro de Sobel.
    Incluye la carga de imagen, la aplicación del filtro y la visualización de resultados.
    
    Parámetros:
    - ruta_imagen (str): Ruta de la imagen a procesar.
    - ruta_salida (str, opcional): Ruta donde se guardará la matriz de la magnitud del gradiente (si se proporciona).
    """
    try:
        # 1. Cargar la imagen
        imagen = cargar_imagen(ruta_imagen)
        
        # 2. Aplicar el filtro Sobel
        sobel_x, sobel_y, magnitud, fase = aplicar_sobel(imagen)
        
        # 3. Mostrar los resultados
        mostrar_resultados(imagen, sobel_x, sobel_y, magnitud, fase)
        
        # 4. Guardar la matriz de la magnitud del gradiente si se proporciona una ruta de salida
        if ruta_salida:
            guardar_resultado(magnitud, ruta_salida)

    except Exception as e:
        print(f"Error: {e}")

# Ejemplo de uso
ruta_imagen = 'chems.jpeg'  # ruta de tu imagen
ruta_salida = 'resultado_sobel.txt'  # guardar la matriz
analisis_imagen_sobel(ruta_imagen, ruta_salida)
