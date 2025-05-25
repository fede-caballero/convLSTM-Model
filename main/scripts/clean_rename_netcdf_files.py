import os
import glob

# Directorio base
output_base_dir = "/home/f-caballero/UM/netCDF_Big_sample"

# Iterar sobre subcarpetas (e.g., 201001011)
for folder in glob.glob(os.path.join(output_base_dir, "*")):
    if os.path.isdir(folder):
        folder_name = os.path.basename(folder)
        output_dir = os.path.join(output_base_dir, folder_name)

        # Eliminar archivos _latest_data_info*
        for file in glob.glob(os.path.join(output_dir, "_latest_data_info*")):
            os.remove(file)
            print(f"Eliminado: {file}")

        # Renombrar NetCDFs (quitar ncfdata%Y%m%d_)
        for file in glob.glob(os.path.join(output_dir, "ncfdata*.nc")):
            old_name = os.path.basename(file)
            # Extraer el timestamp (e.g., 170906 de ncfdata20100101_170906.nc)
            timestamp = old_name.split("_")[-1].replace(".nc", "")
            new_name = os.path.join(output_dir, f"{timestamp}.nc")
            if os.path.exists(new_name):
                os.remove(new_name)
            os.rename(file, new_name)
            print(f"Renombrado: {old_name} -> {timestamp}.nc")
