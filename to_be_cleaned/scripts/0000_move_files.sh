#!/bin/bash

# Obtener la ruta absoluta del directorio donde está el script
CARPETA_MADRE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Recorre cada subcarpeta dentro de la carpeta madre
for carpeta in "$CARPETA_MADRE"/*/; do
    # Mueve todos los archivos al directorio madre
    mv "$carpeta"* "$CARPETA_MADRE"/

    # Borra la carpeta vacía
    rmdir "$carpeta"
done
