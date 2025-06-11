# plot_predictions_corrected.py
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr  # <<< Usamos xarray para la lectura
import glob

def read_netcdf_with_xarray(filename):
    """
    Lee un archivo NetCDF usando xarray, que aplica automáticamente
    _FillValue, scale_factor y add_offset.
    """
    try:
        # mask_and_scale=True es el comportamiento por defecto y hace todo el trabajo
        with xr.open_dataset(filename, mask_and_scale=True) as ds:
            # El array ya está en valores físicos (dBZ) y con NaNs donde había _FillValue
            reflectivity = ds['DBZ'].values
        
        # Quitar la dimensión de tiempo si existe y tiene tamaño 1
        if reflectivity.ndim == 4 and reflectivity.shape[0] == 1:
            reflectivity = reflectivity[0]
            
        return reflectivity
    except Exception as e:
        print(f"Error al leer {filename} con xarray: {e}")
        return None

def plot_composite(composite, filename):
    """Grafica el composite de reflectividad (sin cambios, ya estaba bien)."""
    plt.figure(figsize=(10, 8))
    
    # Corregir orientación para que el Norte esté arriba
    composite_corrected = np.flipud(composite)
    
    # Usamos vmin y vmax para mantener la escala de colores consistente
    plt.imshow(composite_corrected, cmap='jet', interpolation='none', vmin=-30, vmax=75)
    plt.title(f'Composite - {os.path.basename(filename)}', fontsize=14)
    cbar = plt.colorbar()
    cbar.set_label('Reflectividad (dBZ)')
    
    # Etiquetas de orientación
    plt.text(0.02, 0.98, 'N', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='white')
    plt.text(0.98, 0.02, 'S', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', color='white')
    plt.text(0.02, 0.02, 'W', transform=plt.gca().transAxes, fontsize=12, color='white')
    plt.text(0.98, 0.98, 'E', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', verticalalignment='top', color='white')

    plt.show()

def process_netcdf_file(filename, threshold=0):
    """Procesa un archivo NetCDF y genera el composite."""
    # Ahora usamos la nueva función de lectura
    reflectivity = read_netcdf_with_xarray(filename)
    if reflectivity is None:
        return None
    
    # La reflectividad ya está en dBZ, por lo que el threshold se aplica correctamente.
    # Los valores nulos ya son NaN, por lo que no necesitamos enmascararlos.
    composite = np.nanmax(reflectivity, axis=0)
    
    return composite

def process_folder(folder_path):
    """Procesa todos los archivos NetCDF en una carpeta."""
    nc_files = sorted(glob.glob(os.path.join(folder_path, '*.nc')))
    if not nc_files:
        print(f"No se encontraron archivos .nc en {folder_path}")
        return
    
    for file in nc_files:
        print(f"Procesando archivo: {file}")
        composite = process_netcdf_file(file) # Ya no necesitamos el threshold aquí
        if composite is not None:
            plot_composite(composite, file)

# --- USO DEL SCRIPT ---
# Apunta a la carpeta donde tu modelo está guardando las predicciones
folder_path = '/home/f-caballero/UM/TIF3/convLSTM-project/Modificaciones_modelo/Modelo_080625/New_model-8'
process_folder(folder_path)