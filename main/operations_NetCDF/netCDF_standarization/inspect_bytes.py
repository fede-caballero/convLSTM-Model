import netCDF4 as nc
import numpy as np
import sys

# Pasamos la ruta al archivo como argumento en la línea de comandos
if len(sys.argv) < 2:
    print("Error: Debes proporcionar la ruta a un archivo NetCDF.")
    print("Ejemplo: python inspect_bytes.py /ruta/a/tu/archivo.nc")
    sys.exit(1)

file_path = sys.argv[1]
print(f"--- Inspeccionando los bytes del archivo: {file_path} ---")

try:
    with nc.Dataset(file_path, 'r') as ds_in:
        if 'DBZ' not in ds_in.variables:
            raise ValueError("Variable 'DBZ' no encontrada.")
            
        dbz_var = ds_in.variables['DBZ']
        
        # MUY IMPORTANTE: Leemos los datos como bytes, sin aplicar scale/offset
        dbz_var.set_auto_mask(False)
        dbz_var.set_auto_scale(False)
        raw_data_bytes = dbz_var[:]
        
        # Obtenemos los valores de byte únicos y su frecuencia
        unique_values, counts = np.unique(raw_data_bytes, return_counts=True)
        
        # Leemos el FillValue definido en los metadatos
        fill_value_in_metadata = getattr(dbz_var, '_FillValue', 'No definido')

        print(f"\nEl _FillValue definido en los metadatos es: {fill_value_in_metadata}")
        print("\nValores de byte únicos encontrados en los datos y su frecuencia:")
        
        # Creamos un diccionario para fácil lectura
        value_counts = dict(zip(unique_values, counts))
        
        for value, count in value_counts.items():
            print(f"  Byte: {value:<5} | Frecuencia: {count:<10}")

        if fill_value_in_metadata in value_counts:
            print(f"\nVEREDICTO: El _FillValue ({fill_value_in_metadata}) SÍ se encontró en los datos.")
        else:
            print(f"\nVEREDICTO: El _FillValue ({fill_value_in_metadata}) NO se encontró en los datos.")
            print("Esto confirma que el archivo es no estándar.")

except Exception as e:
    print(f"\nOcurrió un error: {e}")