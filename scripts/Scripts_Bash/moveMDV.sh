#!/bin/bash

# Usar el directorio actual como ruta base
RUTA=$(pwd)
cd "$RUTA" || { echo "No se pudo acceder al directorio actual"; exit 1; }

# Contar archivos .mdv
NUM_FILES=$(ls *.mdv 2>/dev/null | wc -l)
if [ "$NUM_FILES" -eq 0 ]; then
    echo "No se encontraron archivos .mdv en el directorio actual"
    exit 1
fi

# Obtener nombre base y carpetas
BASE=$(basename "$RUTA")
CARPETAS=($(ls -d ${BASE}[0-9]* 2>/dev/null | sort))
if [ ${#CARPETAS[@]} -eq 0 ]; then
    echo "No se encontraron carpetas destino"
    exit 1
fi

# Mover archivos en orden
COUNT=0
for FILE in $(ls *.mdv | sort); do
    FOLDER_IDX=$((COUNT / 7))
    if [ $FOLDER_IDX -ge ${#CARPETAS[@]} ]; then
        echo "No hay suficientes carpetas. Quedan archivos sin mover."
        break
    fi
    
    DEST="${CARPETAS[$FOLDER_IDX]}"
    mv "$FILE" "$DEST/"
    echo "Movido $FILE a $DEST"
    ((COUNT++))
done

echo "Proceso completado!"
