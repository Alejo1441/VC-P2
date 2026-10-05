## Práctica 2. Funciones básicas de OpenCV

### Contenidos

[Tarea 1](#tarea-1)  
[Tarea 2](#tarea-2)  
[Tarea 3](#tarea-3)

### Tarea 1
Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil. Resalta con alguna primitiva gráfica en la imagen de Canny las filas que cumplen dicha condición.

Para esta tarea se va a realizar lo mismo que el ejercicio anterior, que estaba en el notebook pero esta vez para las filas y para ello he realizado una serie de cambios

```python
fil_counts = cv2.reduce(canny, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)   # 0 columnas, 1 filas
```
el primer cambio realizado es que en vez de ser un 0, en el reduce, es un 1 para que se realice por filas en vez de columnas

```python
filas_cuentas = fil_counts[:, 0] / (255 * canny.shape[1])
maxfil = np.max(filas_cuentas)
umbral = 0.9 * maxfil

filas_selecionadas = np.where(filas_cuentas >= umbral)[0] # Devuelve las filas que cumplen la condición

canny_marcada = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR) # Convierte a BGR para poder dibujar en color   
```
lo siguiente que se realiza es el conteo de las filas, y la creacion de una variable llamada canny_marcada para pintar sobre esta las lineas que nos pide la tarea

```python
for fila in filas_selecionadas:
    cv2.line(canny_marcada, (0, fila), (canny.shape[1]-1, fila), (0, 0, 255), 1) 
```
por ultimo aqui lo que se realiza es dibujar lineas horizontales, en la imagen, exactamente donde se supera el umbral


### Tarea 2
TAREA: Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. Visualiza los resultados obtenidos para la imagen (o una de tu elección) con Canny y Sobe tras umbralizar ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?




Respuesta:


### Tarea 3
TAREA: Tras ver los vídeos [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy), [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared) y [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared) proponer un demostrador reinterpretando la parte de procesamiento de la imagen, tomando como punto de partida alguna de dichas instalaciones.

Para esta tarea, lo que se he ha realizado, es un selector de diferentes modos, y teniendo en mente "ocultar la cara", como en el video de My little piece of privacy, y es que aunque no he conseguido que la cara solo se "oculte", se aplican los conocimientos de esta practica, como umbralizar la imagen utilizando sobel; aparte he puesto otros modos, como por ejemplo resaltar las filas y columnas de que cumplan con las condiciones de la tarea 2 ( es decir he implementado la tarea2 aqui, y la he puesto como un modo mas) y el otro modo seria simplemente alterar los colores de como se ve la camara.



Bajo licencia de Creative Commons Reconocimiento - No Comercial 4.0 Internacional
