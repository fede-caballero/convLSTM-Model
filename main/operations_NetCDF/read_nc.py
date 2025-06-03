# Este script lee archivos NetCDF, extrae el campo de reflectividad (DBZ),
# calcula el composite máximo y lo grafica. Se asegura de que los valores NaN se manejen correctamente.

import matplotlib.pyplot as plt
import numpy as np
import os
import netCDF4 as nc
import glob

def read_netcdf(filename):
    """Lee un archivo NetCDF y extrae el campo DBZ."""
    try:
        data = nc.Dataset(filename)
        # Asumimos que DBZ está en la variable 'DBZ' con forma (time, z, y, x)
        reflectivity = data.variables['DBZ'][:]  # Shape: (1, 18, 256, 256) o (1, 18, 500, 500)
        data.close()
        # Eliminar la dimensión time (índice 0)
        reflectivity = reflectivity[0]  # Shape: (18, 256, 256) o (18, 500, 500)
        # Convertir a numpy con NaNs
        reflectivity = np.array(reflectivity, dtype=np.float32)
        reflectivity = np.where(np.isfinite(reflectivity), reflectivity, np.nan)
        return reflectivity
    except Exception as e:
        print(f"Error al leer {filename}: {e}")
        return None

def plot_composite(composite, filename):
    """Grafica el composite de reflectividad."""
    plt.figure(figsize=(10, 8))
    
    # Corregir orientación (N arriba, E derecha)
    composite_corrected = np.flipud(composite)
    
    plt.imshow(composite_corrected, cmap='jet', interpolation='none', vmin=-30, vmax=70)
    plt.title(f'Composite - {os.path.basename(filename)}', fontsize=14)
    plt.colorbar(label='Reflectividad (dBZ)')
    
    # Añadir etiquetas de orientación
    plt.text(0.02, 0.98, 'N', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.98, 0.02, 'S', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right')
    plt.text(0.02, 0.02, 'W', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.98, 0.98, 'E', transform=plt.gca().transAxes, fontsize=12, 
             horizontalalignment='right', verticalalignment='top')

    plt.show()

def process_netcdf_file(filename, threshold=0):
    """Procesa un archivo NetCDF y genera el composite."""
    reflectivity = read_netcdf(filename)
    if reflectivity is None:
        return None
    
    # Crear composite (máximo a lo largo de z)
    masked_reflectivity = np.where(reflectivity < threshold, np.nan, reflectivity)
    composite = np.nanmax(masked_reflectivity, axis=0)  # Shape: (256, 256) o (500, 500)
    
    return composite

def process_folder(folder_path):
    """Procesa todos los archivos NetCDF en una carpeta."""
    nc_files = sorted(glob.glob(os.path.join(folder_path, '*.nc')))
    if not nc_files:
        print(f"No se encontraron archivos .nc en {folder_path}")
        return
    
    for file in nc_files:
        print(f"Procesando archivo: {file}")
        composite = process_netcdf_file(file, threshold=0)
        if composite is not None:
            plot_composite(composite, file)

# Uso del script
folder_path = '/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/' 
process_folder(folder_path)