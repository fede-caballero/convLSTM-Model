# Este script lee y compara archivos NetCDF de dos fuentes (archivos o carpetas).
# Para cada par de archivos, extrae el campo de reflectividad (DBZ), calcula el
# composite máximo y grafica el original, el procesado y su diferencia.

import matplotlib.pyplot as plt
import numpy as np
import os
import netCDF4 as nc
import glob

def read_netcdf(filename):
    """
    Lee un archivo NetCDF y extrae el campo de reflectividad (DBZ).
    Maneja los valores no finitos convirtiéndolos a NaN.
    """
    try:
        with nc.Dataset(filename) as data:
            # Asumimos que DBZ está en la variable 'DBZ' con forma (time, z, y, x)
            reflectivity = data.variables['DBZ'][:]  # Shape: (1, 18, 256, 256) o similar
        
        # Eliminar la dimensión de tiempo si existe y tiene tamaño 1
        if reflectivity.shape[0] == 1:
            reflectivity = reflectivity[0]  # Shape: (18, 256, 256)
            
        # Asegurar que el tipo de dato es float para poder usar np.nan
        reflectivity = np.array(reflectivity, dtype=np.float32)
        reflectivity = np.where(np.isfinite(reflectivity), reflectivity, np.nan)
        return reflectivity
    
    except Exception as e:
        print(f"Error al leer el archivo {filename}: {e}")
        return None

def process_netcdf_file(filename, threshold=0):
    """
    Procesa un único archivo NetCDF para generar su composite de máxima reflectividad.
    Se aplica un umbral para filtrar valores bajos antes del cálculo.
    """
    reflectivity = read_netcdf(filename)
    if reflectivity is None:
        return None
    
    # Enmascarar valores por debajo del umbral y calcular el máximo en la vertical (eje z)
    masked_reflectivity = np.where(reflectivity < threshold, np.nan, reflectivity)
    composite = np.nanmax(masked_reflectivity, axis=0)
    
    return composite

def plot_comparison(composite1, composite2, filename1, filename2):
    """
    Grafica dos composites de reflectividad y su diferencia para una fácil comparación.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
    
    base_name1 = os.path.basename(filename1)
    base_name2 = os.path.basename(filename2)
    fig.suptitle(f'Comparación de Reflectividad\nOriginal: {base_name1} vs. Procesado: {base_name2}', fontsize=16)

    # --- Plot 1: Composite del primer archivo (Original) ---
    # Corregir orientación (Norte arriba, Este derecha)
    comp1_corrected = np.flipud(composite1)
    im1 = axes[0].imshow(comp1_corrected, cmap='jet', interpolation='nearest', vmin=-30, vmax=70)
    axes[0].set_title('Composite Original')
    axes[0].set_xlabel('Píxel X')
    axes[0].set_ylabel('Píxel Y')
    fig.colorbar(im1, ax=axes[0], label='Reflectividad (dBZ)', orientation='horizontal', pad=0.15)
    
    # --- Plot 2: Composite del segundo archivo (Normalizado/Procesado) ---
    comp2_corrected = np.flipud(composite2)
    im2 = axes[1].imshow(comp2_corrected, cmap='jet', interpolation='nearest', vmin=-30, vmax=70)
    axes[1].set_title('Composite Procesado')
    axes[1].set_xlabel('Píxel X')
    fig.colorbar(im2, ax=axes[1], label='Reflectividad (dBZ)', orientation='horizontal', pad=0.15)

    # --- Plot 3: Diferencia entre los dos composites ---
    difference = composite1 - composite2
    diff_corrected = np.flipud(difference)
    
    # Usar un colormap divergente para la diferencia y centrarlo en 0
    max_abs_diff = np.nanmax(np.abs(difference))
    if max_abs_diff == 0: max_abs_diff = 1 # Evitar vmin=vmax=0

    im3 = axes[2].imshow(diff_corrected, cmap='coolwarm', interpolation='nearest', vmin=-max_abs_diff, vmax=max_abs_diff)
    axes[2].set_title('Diferencia (Original - Procesado)')
    axes[2].set_xlabel('Píxel X')
    fig.colorbar(im3, ax=axes[2], label='Diferencia de Reflectividad (dBZ)', orientation='horizontal', pad=0.15)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Ajustar para el título y colorbars
    plt.show()

def compare_data(path1, path2):
    """
    Compara archivos NetCDF de dos rutas, que pueden ser archivos individuales o carpetas.
    """
    # --- Caso 1: Ambas rutas son archivos ---
    if os.path.isfile(path1) and os.path.isfile(path2):
        print(f"Comparando archivos:\n1: {path1}\n2: {path2}\n")
        composite1 = process_netcdf_file(path1)
        composite2 = process_netcdf_file(path2)
        
        if composite1 is not None and composite2 is not None:
            plot_comparison(composite1, composite2, path1, path2)

    # --- Caso 2: Ambas rutas son carpetas ---
    elif os.path.isdir(path1) and os.path.isdir(path2):
        print(f"Comparando carpetas:\n1: {path1}\n2: {path2}\n")
        files1 = sorted(glob.glob(os.path.join(path1, '*.nc')))
        files2 = sorted(glob.glob(os.path.join(path2, '*.nc')))

        if not files1 or not files2:
            print("Error: No se encontraron archivos .nc en una o ambas carpetas.")
            return
        if len(files1) != len(files2):
            print("Advertencia: Las carpetas tienen un número diferente de archivos. Se compararán los pares correspondientes.")

        # Iterar y comparar archivos correspondientes
        for f1, f2 in zip(files1, files2):
            print(f"--- Procesando par ---\nOriginal: {os.path.basename(f1)}\nProcesado: {os.path.basename(f2)}")
            composite1 = process_netcdf_file(f1)
            composite2 = process_netcdf_file(f2)
            
            if composite1 is not None and composite2 is not None:
                plot_comparison(composite1, composite2, f1, f2)
    
    # --- Caso 3: Las rutas no son válidas o no coinciden en tipo ---
    else:
        print("Error: Ambas rutas deben ser archivos o ambas deben ser carpetas válidas.")
        if not os.path.exists(path1): print(f"La ruta no existe: {path1}")
        if not os.path.exists(path2): print(f"La ruta no existe: {path2}")

# --- MODO DE USO ---
if __name__ == "__main__":
    # ⬇️⬇️⬇️ MODIFICA ESTAS RUTAS PARA APUNTAR A TUS DATOS ⬇️⬇️⬇️

    # --- Ejemplo 1: Comparar dos carpetas ---
    # Asegúrate de que los archivos dentro de las carpetas estén ordenados de la misma manera.
    path_carpeta_original = '/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/netcdf_500x500x18/20100102/201001021'
    path_carpeta_normalizada = '/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/netcdf_500x500x18/standardized/20100102/201001021'
    
    # Descomenta la siguiente línea para usar la comparación de carpetas
    compare_data(path_carpeta_original, path_carpeta_normalizada)

    # --- Ejemplo 2: Comparar dos archivos específicos ---
    path_archivo_original = '/ruta/a/tu/archivo_original.nc'
    path_archivo_normalizado = '/ruta/a/tu/archivo_normalizado.nc'
    
    # Descomenta la siguiente línea para usar la comparación de archivos
    # compare_data(path_archivo_original, path_archivo_normalizado)

    print("Script listo. Edita las rutas en la sección '__main__' para ejecutar la comparación.")
    # Pequeña verificación para guiar al usuario
    if not (os.path.exists(path_carpeta_original) and os.path.exists(path_carpeta_normalizada)) and \
       not (os.path.exists(path_archivo_original) and os.path.exists(path_archivo_normalizado)):
        print("\nADVERTENCIA: Las rutas de ejemplo no existen. Por favor, actualízalas con tus rutas reales.")