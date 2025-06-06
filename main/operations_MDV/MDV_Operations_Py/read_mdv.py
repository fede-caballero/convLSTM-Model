import pyart
import os

# Ruta a uno de los archivos .mdv
file_path = "/home/f-caballero/UM/TIF3/convLSTM-project/pruebas_conversiones/prueba_netcdf2mdv/NcGeneric2Mdv/output_mdv_roundtrip_float/20100101/20100101_170906.mdv"
# Leer el archivo con Py-ART
try:
    mdv_file = pyart.io.mdv_common.MdvFile(pyart.io.prepare_for_read(file_path))
    print("Field headers:", mdv_file.field_headers)
    print("Master header:", mdv_file.master_header)
    print("Projection type:", mdv_file.field_headers[0]["proj_type"])
except Exception as e:
    print("Error al leer el archivo:", e)