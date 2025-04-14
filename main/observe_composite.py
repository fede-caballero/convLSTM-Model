import matplotlib.pyplot as plt
import pyart
import numpy as np
import os
import netCDF4
import glob
import geopandas as gpd

def read_mdv_to_nc(filename):
    radar = pyart.io.read_grid_mdv(filename)
    return radar

def convert_mdv_nc(radar, output_folder, output_filename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    full_path = os.path.join(output_folder, output_filename)
    if not os.path.isfile(full_path):
        radar.write(full_path, format='NETCDF4')

    data = netCDF4.Dataset(full_path)
    return data

def process_mdv_file(filename):
    radar = read_mdv_to_nc(filename=filename)
    
    output_folder = './nc_files'
    output_filename = os.path.basename(filename).replace('.mdv', '.nc')
    
    data = convert_mdv_nc(radar, output_folder, output_filename)
    
    reflectivity = data['reflectivity'][0,:,:,:]  # Extraer datos y eliminar dimensión extra
    reflectivity = reflectivity.filled(np.nan)  # Convertir a array con NaNs
    
    return reflectivity

def plot_composite(composite, filename, gis_overlay):
    plt.figure(figsize=(10, 8))
    
    composite_corrected = np.flipud(composite)
    
    plt.imshow(composite_corrected, cmap='jet', interpolation='none', vmin=-30, vmax=70)
    plt.title(f'Composite - {os.path.basename(filename)}', fontsize=14)
    plt.colorbar(label='Reflectividad (dBZ)')
    
    if gis_overlay is not None:
        gis_overlay.plot(ax=plt.gca(), color='black', linewidth=1)

    plt.text(0.02, 0.98, 'N', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.98, 0.02, 'S', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right')
    plt.text(0.02, 0.02, 'W', transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.98, 0.98, 'E', transform=plt.gca().transAxes, fontsize=12, 
        horizontalalignment='right', verticalalignment='top')

    plt.show()

def process_folder(folder_path, gis_file):
    gis_overlay = None
    if gis_file:
        gis_overlay = gpd.read_file(gis_file)
    
    mdv_files = sorted(glob.glob(os.path.join(folder_path, '*.mdv')))

    
    for file in mdv_files:
        print(f"Procesando archivo: {file}")
        reflectivity = process_mdv_file(file)
        
        # Crear y mostrar el composite
        threshold = 0  # Ajusta este valor según sea necesario
        masked_reflectivity = np.where(reflectivity < threshold, np.nan, reflectivity)
        composite = np.nanmax(masked_reflectivity, axis=0)
        plot_composite(composite, file, gis_overlay)

# Uso del script
folder_path = '/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main'
gis_file = '/home/f-caballero/UM/TIF3/Argentina.kml'
process_folder(folder_path, gis_file)
