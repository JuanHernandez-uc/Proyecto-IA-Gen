# Scripts para extraer información de WikiData

## IMPORTANTE

Para correr el código hay que cambiar el identificador "QXXX" en la línea 28. Los identificadores de los países se pueden obtener en el archivo paises.txt

Además, hay que cambiar en la línea 237 el nombre del dataset que se producirá.

## Otros

Desde la línea 252 en adelante está lo importante del código:
- El primer argumento de la función `query` es el LIMIT, que es cuántas entidades se procesan por batch.
- El segundo argumento de la función `query` es el OFFSET, que es cuántas lineas se omite la consulta (a veces las respuestas de la query son deterministas, entonces el offset sirve para evitar las primeras filas)

Si se cae, hay que cambiar el inicio del for i in range (por lo que se explicaba del OFFSET)

