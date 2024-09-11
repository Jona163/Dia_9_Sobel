# Filtro Sobel y Conversión Matricial de Imágenes
Este proyecto contiene un pipeline completo para procesar imágenes utilizando el filtro de Sobel y realizar análisis de bordes. El código permite cargar una imagen, aplicar el filtro Sobel para detectar bordes en las direcciones X e Y, y visualizar los resultados. Además, incluye la funcionalidad para guardar y cargar representaciones matriciales de la imagen.

Características:
Carga y Procesamiento de Imágenes:

Carga de imágenes en escala de grises desde una ruta especificada.
Aplicación del filtro Sobel en las direcciones X y Y.
Cálculo de la magnitud y dirección del gradiente para detectar bordes.
Guardado de Matriz:

Guarda la matriz de la magnitud del gradiente resultante en un archivo de texto.
Lectura de Matriz desde un Archivo de Texto:

Carga una matriz desde un archivo de texto y la convierte en una imagen.
Aplica el filtro Sobel a la imagen reconstruida para detectar los bordes.
Visualización y Guardado de Resultados:

Visualiza la imagen original, los gradientes en X e Y, la magnitud del gradiente, y la dirección de los bordes.
Guarda la imagen procesada si se especifica una ruta de salida.
