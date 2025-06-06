import xarray as xr
import numpy as np

# RUTA A UNO DE TUS ARCHIVOS ORIGINALES
file_path = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/netcdf_500x500x18/20100101/201001011/170906.nc"

try:
    print(f"Abriendo {file_path} con xarray...")
    # mask_and_scale=True es el comportamiento por defecto, lo ponemos por claridad
    # xarray debería manejar el _FillValue y el scale/offset automáticamente
    ds = xr.open_dataset(file_path, mask_and_scale=True)
    
    # Extraemos el array de numpy PURO
    dbz_array = ds['DBZ'].values[0] # [0] para quitar la dimension de tiempo

    print("\n--- ¡Array leído con xarray! ---")
    print(f"Tipo de dato: {dbz_array.dtype}")
    print(f"Forma: {dbz_array.shape}")
    
    num_nans = np.isnan(dbz_array).sum()
    total_pixels = dbz_array.size
    
    print(f"Número de NaNs encontrados: {num_nans}")
    print(f"Porcentaje de píxeles nulos: {(num_nans / total_pixels) * 100:.2f}%")

    if num_nans > 0:
        print("\nVEREDICTO: ¡ÉXITO! xarray SÍ pudo leer los datos nulos como NaN.")
    else:
        print("\nVEREDICTO: FRACASO. Incluso xarray no pudo crear los NaNs.")

except Exception as e:
    print(f"Ocurrió un error: {e}")