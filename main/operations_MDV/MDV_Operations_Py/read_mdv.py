import pyart
import os

# Ruta a uno de los archivos .mdv
file_path = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/MDV_samples/big_sample/201001011/170906.mdv"  # Reemplaza con la ruta a uno de tus archivos

# Leer el archivo con Py-ART
try:
    mdv_file = pyart.io.mdv_common.MdvFile(pyart.io.prepare_for_read(file_path))
    print("Field headers:", mdv_file.field_headers)
    print("Master header:", mdv_file.master_header)
    print("Projection type:", mdv_file.field_headers[0]["proj_type"])
except Exception as e:
    print("Error al leer el archivo:", e)