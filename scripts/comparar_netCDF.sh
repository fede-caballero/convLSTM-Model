#!/bin/bash

ORIG1="/home/f-caballero/UM/netCDF_Big_sample"
ORIG2="/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample"
DEST="/home/f-caballero/UM/netCDF_diferencias"

mkdir -p "$DEST"

echo "Buscando carpetas en $ORIG2 que no están en $ORIG1"
echo "Copiando al destino: $DEST"
echo "-------------------------------------------"

# Crear un array con los nombres de carpeta en ORIG1
mapfile -t ORIG1_DIRS < <(ls "$ORIG1")

# Recorrer las carpetas de ORIG2
for dir in "$ORIG2"/*; do
    folder_name=$(basename "$dir")

    # Chequear si está en ORIG1
    if [[ ! " ${ORIG1_DIRS[*]} " =~ " ${folder_name} " ]]; then
        echo "➕ Carpeta $folder_name no está en ORIG1 → Copiando..."
        cp -r "$dir" "$DEST/"
    else
        echo "✅ Carpeta $folder_name ya existe en ORIG1 → Ignorada"
    fi
done

echo "🚀 Proceso completado."
