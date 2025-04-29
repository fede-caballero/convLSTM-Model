import matplotlib.pyplot as plt
import numpy as np
import os
import netCDF4 as nc
import glob

def standardize_netcdf(input_file, output_file, min_dbz=-29, max_dbz=55):
    """Lee un NetCDF, estandariza DBZ, y guarda el resultado."""
    try:
        # Leer el archivo NetCDF
        data = nc.Dataset(input_file)
        reflectivity = data.variables['DBZ'][:]  # Shape: (1, 18, 500, 500)
        data.close()

        # Truncar valores de DBZ
        reflectivity = np.clip(reflectivity, min_dbz, max_dbz)  # [-29, 55]
        reflectivity = np.where(np.isfinite(reflectivity), reflectivity, np.nan)

        # Crear directorio de salida si no existe
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Crear nuevo archivo NetCDF
        ds_out = nc.Dataset(output_file, 'w', format='NETCDF4')
        ds_out.createDimension('time', 1)
        ds_out.createDimension('z', 18)
        ds_out.createDimension('y', reflectivity.shape[2])  # 500
        ds_out.createDimension('x', reflectivity.shape[3])  # 500
        dbz_var = ds_out.createVariable('DBZ', 'f4', ('time', 'z', 'y', 'x'))
        dbz_var[:] = reflectivity
        ds_out.close()

        return reflectivity[0]  # Shape: (18, 500, 500)
    except Exception as e:
        print(f"Error al procesar {input_file}: {e}")
        return None

def plot_composite(composite, filename, min_dbz=-29, max_dbz=55):
    """Grafica el composite de reflectividad."""
    plt.figure(figsize=(10, 8))
    
    # Corregir orientación (N arriba, E derecha)
    composite_corrected = np.flipud(composite)
    
    plt.imshow(composite_corrected, cmap='jet', interpolation='none', vmin=min_dbz, vmax=max_dbz)
    plt.title(f'Composite - {os.path.basename(filename)}', fontsize=14)
    plt.colorbar(label='Reflectividad (dBZ)')
    
    # Añadir etiquetas de orientación
    plt.text(0.02, 0.98, 'N', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.98, 0.02, 'S', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right')
    plt.text(0.02, 0.02, 'W', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.98, 0.98, 'E', transform=plt.gca().transAxes, fontsize=12, 
             horizontalalignment='right', verticalalignment='top')

    plt.show()

def process_netcdf_file(input_file, output_file, threshold=0, min_dbz=-29, max_dbz=55):
    """Procesa un NetCDF, estandariza, y genera el composite."""
    reflectivity = standardize_netcdf(input_file, output_file, min_dbz, max_dbz)
    if reflectivity is None:
        return None
    
    # Crear composite (máximo a lo largo de z)
    masked_reflectivity = np.where(reflectivity < threshold, np.nan, reflectivity)
    composite = np.nanmax(masked_reflectivity, axis=0)  # Shape: (500, 500)
    
    return composite

def process_netcdf_folder(input_folder, output_dir, threshold=0, min_dbz=-29, max_dbz=55):
    """Procesa todos los archivos NetCDF en una carpeta."""
    # Buscar archivos .nc
    nc_files = sorted(glob.glob(os.path.join(input_folder, '*.nc')))
    if not nc_files:
        print(f"No se encontraron archivos .nc en {input_folder}")
        return
    
    print(f"Procesando {len(nc_files)} archivos en {input_folder}")
    for input_file in nc_files:
        # Generar nombre de archivo de salida
        base_name = os.path.basename(input_file).replace('.nc', '_standardized.nc')
        output_file = os.path.join(output_dir, base_name)
        
        print(f"\nProcesando archivo: {input_file}")
        composite = process_netcdf_file(input_file, output_file, threshold, min_dbz, max_dbz)
        if composite is not None:
            print(f"Archivo estandarizado guardado en: {output_file}")
            plot_composite(composite, output_file, min_dbz, max_dbz)
        else:
            print(f"No se pudo generar el composite para {input_file}")

# Uso del script
input_folder = '/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/201501181'
output_dir = '/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/operations_NetCDF/output'

# Procesar la carpeta
process_netcdf_folder(input_folder, output_dir, threshold=0, min_dbz=-29, max_dbz=55)