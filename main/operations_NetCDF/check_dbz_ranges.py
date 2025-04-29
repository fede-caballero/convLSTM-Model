# Este script sirve para comprobar si los archivos NetCDF en una carpeta tienen valores de DBZ fuera del rango [-30, 70].
# Se asume que los archivos NetCDF están organizados en carpetas y que cada archivo tiene una variable llamada 'DBZ'.
# Se utiliza la librería netCDF4 para leer los archivos NetCDF y numpy para manejar los datos.
# Se requiere instalar las librerías netCDF4 y numpy si no están instaladas.

import netCDF4 as nc
import numpy as np
import glob
import os

def check_dbz_range(file_path, min_dbz=-30, max_dbz=70):
    try:
        ds = nc.Dataset(file_path)
        if 'DBZ' not in ds.variables:
            ds.close()
            return None
        dbz = ds.variables['DBZ'][:]
        dbz_min = np.nanmin(dbz)
        dbz_max = np.nanmax(dbz)
        ds.close()
        if dbz_min < min_dbz or dbz_max > max_dbz:
            return dbz_min, dbz_max
        return None
    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return None

def scan_netcdf_folders(base_path, min_dbz=-30, max_dbz=70):
    outliers = []
    for folder in sorted(glob.glob(os.path.join(base_path, '*'))):
        if not os.path.isdir(folder):
            continue
        nc_files = sorted(glob.glob(os.path.join(folder, '*.nc')))
        for file_path in nc_files:
            result = check_dbz_range(file_path, min_dbz, max_dbz)
            if result is not None:
                dbz_min, dbz_max = result
                outliers.append((folder, os.path.basename(file_path), dbz_min, dbz_max))
    
    if outliers:
        print("Archivos con DBZ fuera de [-30, 70]:")
        for folder, file_name, dbz_min, dbz_max in outliers:
            print(f"Carpeta: {os.path.basename(folder)}, Archivo: {file_name}, Min: {dbz_min}, Max: {dbz_max}")
    else:
        print("No se encontraron archivos con DBZ fuera de [-30, 70].")

# Uso
base_path = '/home/f-caballero/UM/netCDF_Big_sample'
scan_netcdf_folders(base_path, min_dbz=-30, max_dbz=70)