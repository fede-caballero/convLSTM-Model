#!/bin/bash

# Directorio actual como base
RUTA=$(pwd)
cd "$RUTA" || { echo "No se pudo acceder al directorio actual"; exit 1; }

# Archivos .mdv
FILES=($(ls *.mdv 2>/dev/null | sort))
NUM_FILES=${#FILES[@]}
if [ "$NUM_FILES" -eq 0 ]; then
    echo "No se encontraron archivos .mdv en el directorio actual"
    exit 1
fi

# Carpetas destino ordenadas
BASE=$(basename "$RUTA")
CARPETAS=($(find . -maxdepth 1 -type d -name "${BASE}[0-9]*" | sed 's|^\./||' | sort))
if [ ${#CARPETAS[@]} -eq 0 ]; then
    echo "No se encontraron carpetas destino"
    exit 1
fi

# Mover archivos en grupos de 7
for ((i=0; i<NUM_FILES; i++)); do
    FOLDER_IDX=$((i / 7))
    if [ $FOLDER_IDX -ge ${#CARPETAS[@]} ]; then
        echo "No hay suficientes carpetas. Quedan archivos sin mover."
        break
    fi
    FILE=${FILES[$i]}
    DEST="${CARPETAS[$FOLDER_IDX]}"
    mv "$FILE" "$DEST/"
    echo "Movido $FILE a $DEST"
done

echo "Proceso completado!"

