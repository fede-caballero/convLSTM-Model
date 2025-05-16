# Este script lee archivos NetCDF y extrae la variable DBZ.
# Se espera que la variable DBZ tenga la forma (1, 18, 500, 500).

import netCDF4 as nc
import numpy as np
import glob
import os

def read_netcdf_dbz(file_path):
    """Lee la variable DBZ de un archivo NetCDF y devuelve forma, min y max."""
    try:
        ds = nc.Dataset(file_path)
        if 'DBZ' not in ds.variables:
            print(f"Error: No se encontró la variable 'DBZ' en {file_path}")
            ds.close()
            return None
        
        dbz = ds.variables['DBZ'][:]  # Shape: (1, 18, 500, 500) esperado
        shape = dbz.shape
        dbz_min = np.nanmin(dbz)
        dbz_max = np.nanmax(dbz)
        ds.close()
        
        return shape, dbz_min, dbz_max
    except Exception as e:
        print(f"Error al procesar {file_path}: {e}")
        return None

def process_netcdf_folder(folder_path):
    """Procesa todos los archivos NetCDF en una carpeta."""
    # Buscar archivos .nc
    nc_files = sorted(glob.glob(os.path.join(folder_path, '*.nc')))
    if not nc_files:
        print(f"No se encontraron archivos .nc en {folder_path}")
        return
    
    print(f"Procesando {len(nc_files)} archivos en {folder_path}")
    for file_path in nc_files:
        print(f"\nAbriendo {os.path.basename(file_path)}")
        result = read_netcdf_dbz(file_path)
        if result is not None:
            shape, dbz_min, dbz_max = result
            print(f"Forma: {shape}")
            print(f"Min: {dbz_min}, Max: {dbz_max}")
        else:
            print(f"No se pudo procesar {file_path}")

# Uso del script
folder_path = '/home/f-caballero/UM/Backup-convLSTM/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/201702255'
process_netcdf_folder(folder_path)
#folder_path_1 = '/home/f-caballero/UM/netCDF_Big_sample/2015101410'
#process_netcdf_folder(folder_path_1)

# Comparando las salidas del normalize_netcdf_recover.py con los archivos originales, voy a tener que volver a procesar los datos para que el rango esté entre -30 y 70 dBZ.