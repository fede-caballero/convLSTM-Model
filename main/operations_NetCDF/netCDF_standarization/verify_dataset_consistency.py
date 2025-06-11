# verify_dataset_consistency.py
import os
import netCDF4 as nc
import numpy as np
import logging

# --- CONFIGURACIÓN ---
# 1. Directorio raíz que contiene todas tus carpetas de secuencias (ej: 200610183, etc.)
DATASET_ROOT_DIR = "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2008"

# 2. Los valores de metadatos que deben cumplir TODOS los archivos.
TARGET_SCALE = 0.5
TARGET_OFFSET = 33.5

# 3. Número de archivos que esperamos encontrar en una carpeta de secuencia válida.
EXPECTED_FILES_PER_SEQUENCE = 17

# Configuración del logging para un reporte claro.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- INICIO DEL SCRIPT ---
logging.info(f"Iniciando la verificación de consistencia en: {DATASET_ROOT_DIR}")

# Contadores para el resumen
total_sequences_found = 0
valid_sequences = 0
problematic_sequences = []

# os.walk es perfecto para recorrer la estructura de carpetas
for dirpath, _, filenames in os.walk(DATASET_ROOT_DIR):
    
    # Filtramos para quedarnos solo con los archivos .nc
    nc_files = [f for f in filenames if f.endswith('.nc')]
    
    # Solo procesamos directorios que contengan el número exacto de archivos de una secuencia
    if len(nc_files) == EXPECTED_FILES_PER_SEQUENCE:
        total_sequences_found += 1
        is_sequence_valid = True
        logging.info(f"Verificando secuencia en: {dirpath}")

        # Iteramos sobre cada archivo DENTRO de la secuencia
        for filename in nc_files:
            file_path = os.path.join(dirpath, filename)
            try:
                with nc.Dataset(file_path, 'r') as ds:
                    if 'DBZ' not in ds.variables:
                        logging.error(f"  -> PROBLEMA en '{dirpath}': El archivo '{filename}' no tiene variable DBZ.")
                        is_sequence_valid = False
                        break # No necesitamos seguir revisando esta secuencia

                    dbz_var = ds.variables['DBZ']
                    scale = getattr(dbz_var, 'scale_factor', None)
                    offset = getattr(dbz_var, 'add_offset', None)

                    # Comparamos los valores usando np.isclose para seguridad con floats
                    if not (scale is not None and offset is not None and \
                            np.isclose(scale, TARGET_SCALE) and np.isclose(offset, TARGET_OFFSET)):
                        
                        logging.error(f"  -> PROBLEMA en '{dirpath}': El archivo '{filename}' tiene metadatos incorrectos (scale: {scale}, offset: {offset}).")
                        is_sequence_valid = False
                        break # No necesitamos seguir revisando esta secuencia
            
            except Exception as e:
                logging.error(f"  -> PROBLEMA en '{dirpath}': No se pudo leer el archivo '{filename}'. Error: {e}")
                is_sequence_valid = False
                break
        
        if is_sequence_valid:
            valid_sequences += 1
            logging.info(f"  -> SECUENCIA VÁLIDA.")
        else:
            problematic_sequences.append(dirpath)

# --- Resumen Final ---
print("\n" + "="*50)
logging.info("VERIFICACIÓN COMPLETADA")
print("="*50)
logging.info(f"Total de secuencias de {EXPECTED_FILES_PER_SEQUENCE} archivos encontradas: {total_sequences_found}")
logging.info(f"Secuencias VÁLIDAS y consistentes: {valid_sequences}")
logging.info(f"Secuencias con PROBLEMAS: {len(problematic_sequences)}")

if problematic_sequences:
    logging.warning("\nLas siguientes carpetas de secuencia contienen al menos un archivo con errores y deben ser revisadas o eliminadas:")
    for path in problematic_sequences:
        print(f" - {path}")