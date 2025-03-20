# 1. Instalar y cargar librerías necesarias
import os
import glob
import numpy as np
import pyart
import matplotlib.pyplot as plt

# 2. Función para leer .mdv y calcular el composite
def get_composite_from_mdv(filename):
    # Leer el archivo .mdv
    radar = pyart.io.read_grid_mdv(filename)
    reflectivity = radar.fields['reflectivity']['data'].filled(np.nan)  # (36, 1000, 1000)
    
    # Calcular el composite (máximo a lo largo de la dimensión vertical, eje 0)
    composite = np.nanmax(reflectivity, axis=0)  # (1000, 1000)
    
    # Depuración: Ver valores extremos y NaN
    print(f"Archivo: {filename}")
    print(f"Mínimo en composite: {np.nanmin(composite)} dBZ")
    print(f"Máximo en composite: {np.nanmax(composite)} dBZ")
    print(f"NaN count en composite: {np.isnan(composite).sum()}")
    
    return composite

# 3. Función para visualizar el composite
def plot_composite(composite, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(composite, cmap='jet', vmin=-30, vmax=70)  # Rango típico de dBZ
    plt.colorbar(label='Reflectividad (dBZ)')
    plt.title(f"Composite - {os.path.basename(filename)}")
    plt.xlabel('X (píxeles)')
    plt.ylabel('Y (píxeles)')
    plt.show()

# 4. Recorrer archivos .mdv directamente en la carpeta
ruta_base = 'D:\\Fede Facultad\\tesis\\MDV\\20211011'  # Ruta ajustada para Windows

print(f"Buscando en: {ruta_base}")
if os.path.exists(ruta_base):
    print("Carpeta encontrada. Contenido:", os.listdir(ruta_base))
else:
    print("Error: Carpeta no encontrada. Verifica la ruta.")
    exit()

# Listar todos los archivos .mdv en la carpeta
mdv_files = sorted(glob.glob(os.path.join(ruta_base, "*.mdv")))
print(f"Archivos .mdv encontrados: {mdv_files}")

# Procesar y visualizar cada archivo .mdv
for mdv_file in mdv_files:
    try:
        composite = get_composite_from_mdv(mdv_file)
        plot_composite(composite, mdv_file)
    except Exception as e:
        print(f"Error al procesar {mdv_file}: {e}")

print("Procesamiento completado.")