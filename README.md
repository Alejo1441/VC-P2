## Práctica 2. Funciones básicas de OpenCV

### Contenidos

[Tarea 1](#tarea-1)  
[Tarea 2](#tarea-2)  
[Tarea 3](#tarea-3)

### Tarea 1
Realiza la cuenta de píxeles blancos por filas (en lugar de por columnas). Determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90*maxfil. Resalta con alguna primitiva gráfica en la imagen de Canny las filas que cumplen dicha condición.

Para esta tarea se va a realizar lo mismo que el ejercicio anterior que ya estaba resuelto, que estaba en el notebook pero esta vez para las filas y para ello he realizado una serie de cambios

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
lo siguiente que se realiza es obtener el porcentaje de píxeles blancos de cada fila, calcular el máximo y establecer el umbral del 90%

```python
for fila in filas_selecionadas:
    cv2.line(canny_marcada, (0, fila), (canny.shape[1]-1, fila), (0, 0, 255), 1) 
```
por último aquí lo que se realiza es dibujar líneas horizontales, en la imagen, exactamente donde se supera el umbral


### Tarea 2
TAREA: Aplica umbralizado a la imagen resultante de Sobel (convertida a 8 bits), y posteriormente realiza el conteo por filas y columnas similar al realizado en el ejemplo con la salida de Canny de píxeles no nulos. Calcula el valor máximo de la cuenta por filas y columnas, y determina las filas y columnas por encima del 0.90*máximo. Remarca con alguna primitiva gráfica dichas filas y columnas sobre la imagen del mandril. Visualiza los resultados obtenidos para la imagen (o una de tu elección) con Canny y Sobe tras umbralizar ¿Cómo se comparan los resultados obtenidos a partir de Sobel y Canny?

```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # x
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)  # y

sobel = cv2.add(sobelx, sobely)
sobel8 = cv2.convertScaleAbs(sobel)

```
En esta parte lo que se realiza es el cálculo los gradientes en los ejes x e y, utilizando sobel, sumamos ambas direcciones, para obtener los bordes combinados y los convertimos en 8 bits

```python

_, sobel_thresh = cv2.threshold(sobel8, 100, 255, cv2.THRESH_BINARY) 

canny = cv2.Canny(gray, 100, 200)  # Detección de bordes con Canny
```
Aquí aplicamos un umbralizado binario a la imagen de Sobel para que los píxeles de los bordes destaquen,cabe destacar que se puede cambiar el umbral, cambiando el valor del segundo argumento de la función **threshold** para que la imagen se vea más clara o más oscura, y con esto más tarde, a la hora de pintar las líneas, saldrán más, o menos; también  aplicamos Canny para pintarlo y comparar

```python

conteo_filas = cv2.reduce(sobel_thresh, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)   
conteo_columnas = cv2.reduce(sobel_thresh, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)   
```

hacemos lo mismo de antes, solo que en vez de usar Canny, usamos Sobel, y tenemos que usarlo tanto para las filas y las columnas


```python

max_filas = np.max(conteo_filas)
max_columnas = np.max(conteo_columnas)

filas_destacadas = np.where(conteo_filas >= 0.9 * max_filas)[0]  # Filas con al menos el 90% del máximo
columnas_destacadas = np.where(conteo_columnas >= 0.9 * max_columnas)[1]  # Columnas con al menos el 90% del máximo


for f in filas_destacadas:
    cv2.line(img_rgb, (0, f), (img_rgb.shape[1]-1, f), (255, 0, 0), 1)  # Línea roja para filas
for c in columnas_destacadas:
    cv2.line(img_rgb, (c, 0), (c, img_rgb.shape[0]-1), (0, 255, 0), 1)  # Línea verde para columnas
```
En este último bloque se realiza, lo mismo que en el ejercicio anterior, solo que ahora se va a realizar tanto por filas y columnas, y por eso también se utilizan colores diferentes, para una mejor visualización

Respuesta: Se observan diferencias apreciables, al menos con el umbral escogido (100), y es que Sobel, dependiendo del valor del umbral si es demasiado bajo hará que se detecten más píxeles como bordes, mientras que un umbral mayor hace que solamente se conserven los bordes que sean más fuertes; mientras tanto Canny realiza un procesamiento más elaborado y produce bordes más definidos, y por eso en este caso Sobel genera bordes más gruesos dependiendo del umbral seleccionado


### Tarea 3
TAREA: Tras ver los vídeos [My little piece of privacy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy), [Messa di voce](https://youtu.be/GfoqiyB1ndE?feature=shared) y [Virtual air guitar](https://youtu.be/FIAmyoEpV5c?feature=shared) proponer un demostrador reinterpretando la parte de procesamiento de la imagen, tomando como punto de partida alguna de dichas instalaciones.

Para esta tarea, se ha realizado un selector de diferentes modos teniendo en mente la idea de "ocultar la cara", como en el vídeo de My little piece of privacy, y es que aunque mi idea principal era aplicar el procesamiento de Sobel únicamente sobre la cara, para detectar y aislar únicamente la cara habría sido necesario utilizar técnicas de detección de objetos o reconocimiento, que todavía no hemos visto en prácticas. Por este motivo, decidí aplicar Sobel sobre toda la imagen y utilizar diferentes modos de visualización, uno de los otros modos consiste en resaltar las filas y columnas que cumplan con las condiciones de la tarea 2 ( es decir he implementado la tarea 2 aquí, y la he puesto como un modo más) y el otro modo consiste simplemente en alterar los colores de como se ve la cámara.

para ello se procede a comentar las partes más importantes de esta tarea

```python
while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('1'):
            modo = 1
        elif tecla == ord('2'):
            modo = 2
        elif tecla == ord('3'):
            modo = 3
        elif tecla == 27:  # Tecla 'ESC' para salir
            break
```
en esta sección de código lo que se realiza es que dependiendo del botón pulsado, en este caso 1, 2 o 3, se entra en un modo diferente

```python
 # modo 1
        if modo == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ggris = cv2.GaussianBlur(gray, (5, 5), 0)

            sobelx = cv2.Sobel(ggris, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(ggris, cv2.CV_64F, 0, 1)
            sobel = cv2.add(sobelx, sobely)
            sobel8 = cv2.convertScaleAbs(sobel)
            _, sobel_thresh = cv2.threshold(sobel8, 100, 255, cv2.THRESH_BINARY)

            resultado = cv2.cvtColor(sobel_thresh, cv2.COLOR_GRAY2BGR)
            cv2.putText(resultado, "Modo 1: Sobel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```
en el primer modo, que sería el que me inspiró para la tarea 3, es aplicar Sobel a la cámara y que me lo muestre, se puede ver la explicación en la tarea 2, solo que ahora aplicado a la cámara web y con un texto que ponga, en que modo se encuentra


```python
       elif modo == 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(gray, 100, 200)

            conteo_filas = cv2.reduce(canny, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
            conteo_columnas = cv2.reduce(canny, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)

            max_filas = np.max(conteo_filas)
            max_columnas = np.max(conteo_columnas)

            resultado = frame.copy()

            if max_filas > 0 and max_columnas > 0:
                filas_destacadas = np.where(conteo_filas >= 0.9 * max_filas)[0]
                columnas_destacadas = np.where(conteo_columnas >= 0.9 * max_columnas)[1]

                for f in filas_destacadas:
                    cv2.line(resultado, (0, f), (resultado.shape[1]-1, f), (255, 0, 0), 1)
                for c in columnas_destacadas:
                    cv2.line(resultado, (c, 0), (c, resultado.shape[0]-1), (0, 255, 0), 1)

            cv2.putText(resultado, "Modo 2: Canny", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 

```
este modo está de prueba y lo que se realiza es igual que en la tarea 1, solo que con filas y columnas, utilizando Canny

```python
 elif modo == 3:
            resultado = np.zeros_like(frame)

            r= frame[:, :, 2]
            g= frame[:, :, 1]
            b = frame[:, :, 0]

            resultado[:, :, 2] = 255- r
            resultado[:, :, 1] = 255- b
            resultado[:, :, 0] =  g

            cv2.putText(resultado, "Modo 3: Color alterado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
```
en este último modo, quise rescatar la ultima tarea y aplicar un filtro, y asi poder tener varios modos en la misma practica, lo que se realice en el es un cambio en los colores RGB, para que sea más difícil de reconocer a la persona

Autor: Alejandro Alejo Santana


