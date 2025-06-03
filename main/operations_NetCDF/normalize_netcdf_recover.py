# Con este script se normalizan los archivos NetCDF de la carpeta de entrada y se guardan en la carpeta de salida.
# Se espera que los archivos de entrada tengan la forma (1, 36, 1000, 1000) y se normalizan a (1, 18, 500, 500).
# ********************Tendré que hacer otra pasada de este archivo para que el rango de reflectividad sea entre -30 y 70 dBZ.*********************
# El script también maneja errores y crea un archivo de log para registrar el progreso y los problemas encontrados.
import os
import glob
import netCDF4 as nc
import numpy as np
from scipy.ndimage import zoom

# Directorios
input_base_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/operations_MDV/MDV_Operations_tcsh/netCDF"
output_base_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/operations_MDV/MDV_Operations_tcsh/netCDF/normalized_output"
log_file = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/logs/normalize_log_new_range.txt"

# Crear archivo de log y directorio de salida
os.makedirs(os.path.dirname(log_file), exist_ok=True)
os.makedirs(output_base_dir, exist_ok=True)
with open(log_file, 'w') as f:
    f.write("Registro de normalización - Inicio: " + str(np.datetime64('now')) + "\n")

# Iterar sobre subcarpetas
print(f"Buscando subcarpetas en {input_base_dir}")
for folder in glob.glob(os.path.join(input_base_dir, "*")):
    if not os.path.isdir(folder):
        continue
    folder_name = os.path.basename(folder)
    input_dir = os.path.join(input_base_dir, folder_name)
    output_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Procesando carpeta: {folder_name}")

    # Procesar cada NetCDF
    print(f"Buscando archivos .nc en {input_dir}")
    for nc_file in glob.glob(os.path.join(input_dir, "*.nc")):
        base_name = os.path.basename(nc_file)
        output_file = os.path.join(output_dir, base_name)
        print(f"Procesando archivo: {nc_file}")

        # Verificar si el archivo de salida ya existe
        if os.path.exists(output_file):
            try:
                ds_out = nc.Dataset(output_file, 'r')
                if 'DBZ' in ds_out.variables and ds_out.variables['DBZ'].shape == (1, 18, 500, 500):
                    print(f"Ignorando {nc_file}: salida ya normalizada en {output_file}")
                    with open(log_file, 'a') as f:
                        f.write(f"Ignorado: {nc_file} (salida ya normalizada en {output_file})\n")
                    ds_out.close()
                    continue
                ds_out.close()
            except Exception as e:
                print(f"Error al verificar {output_file}: {e}, procesando de nuevo")
                with open(log_file, 'a') as f:
                    f.write(f"Error al verificar {output_file}: {e}, procesando de nuevo\n")

        # Leer NetCDF
        try:
            ds = nc.Dataset(nc_file, 'r')
            if 'DBZ' not in ds.variables:
                raise ValueError("Variable 'DBZ' no encontrada")
            dbz = ds.variables['DBZ'][:]  # Shape: (1, 36, 1000, 1000) o similar
            dbz_min, dbz_max = np.nanmin(dbz), np.nanmax(dbz)
            print(f"Archivo {nc_file} leído, rango: [{dbz_min:.2f}, {dbz_max:.2f}]")
            ds.close()
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"Error al leer {nc_file}: {e}\n")
            print(f"Error al leer {nc_file}: {e}")
            continue

        # Manejar dimensiones
        if len(dbz.shape) == 4 and dbz.shape[0] == 1 and dbz.shape[1:] == (36, 1000, 1000):
            print(f"Normalizando {nc_file} de (1, 36, 1000, 1000) a (1, 18, 500, 500)")
            # Extraer volumen 3D
            dbz_3d = dbz[0]  # Shape: (36, 1000, 1000)
            # Redimensionar a 18x500x500
            dbz_resized = zoom(dbz_3d, (18/36, 500/1000, 500/1000), order=1)
            # Estandarizar rango
            dbz_normalized = np.clip(dbz_resized, -30, 70)
            # Reemplazar valores no finitos
            dbz_normalized = np.where(np.isfinite(dbz_normalized), dbz_normalized, -30)
            # Agregar dimensión 1
            dbz_normalized = dbz_normalized[np.newaxis, :]  # Shape: (1, 18, 500, 500)

            # Guardar NetCDF normalizado
            try:
                ds_out = nc.Dataset(output_file, 'w', format='NETCDF4')
                ds_out.createDimension('time', 1)
                ds_out.createDimension('z', 18)
                ds_out.createDimension('y', 500)
                ds_out.createDimension('x', 500)
                dbz_var = ds_out.createVariable('DBZ', 'f4', ('time', 'z', 'y', 'x'), zlib=True, complevel=4)
                dbz_var[:] = dbz_normalized
                ds_out.close()
                with open(log_file, 'a') as f:
                    f.write(f"Normalizado: {nc_file} -> {output_file}, Rango original: [{dbz_min:.2f}, {dbz_max:.2f}]\n")
                print(f"Guardado: {output_file}")
            except Exception as e:
                with open(log_file, 'a') as f:
                    f.write(f"Error al guardar {output_file}: {e}\n")
                print(f"Error al guardar {output_file}: {e}")
        elif dbz.shape == (1, 18, 500, 500):
            print(f"Ignorando {nc_file}: ya es (1, 18, 500, 500)")
            with open(log_file, 'a') as f:
                f.write(f"Ignorado: {nc_file} (ya es 1, 18, 500, 500), Rango: [{dbz_min:.2f}, {dbz_max:.2f}]\n")
        else:
            with open(log_file, 'a') as f:
                f.write(f"Dimensiones desconocidas en {nc_file}: {dbz.shape}\n")
            print(f"Dimensiones desconocidas en {nc_file}: {dbz.shape}")