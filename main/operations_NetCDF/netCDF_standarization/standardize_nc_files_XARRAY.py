# standardize_nc_files_XARRAY_FINAL_CORREGIDO.py
import os
import glob
import netCDF4 as nc
import xarray as xr
import numpy as np
from scipy.ndimage import zoom
from datetime import datetime, timezone
import pyproj
import logging

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
input_root_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/netcdf_500x500x18/"
output_root_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/netcdf_500x500x18/standardized/"

# --- Constantes Físicas y de Formato ---
MIN_RELEVANT_DBZ = 5.0
CLIP_PHYSICAL_MIN_DBZ = -29.0
CLIP_PHYSICAL_MAX_DBZ = 65.0
OUTPUT_NC_SCALE_FACTOR = np.float32(0.5)
OUTPUT_NC_ADD_OFFSET = np.float32(33.5)
OUTPUT_NC_FILL_BYTE = np.int8(-128)
PHYSICAL_FILL_VALUE = (float(OUTPUT_NC_FILL_BYTE) * OUTPUT_NC_SCALE_FACTOR) + OUTPUT_NC_ADD_OFFSET

TARGET_DIM_X0 = 500; TARGET_DIM_Y0 = 500; TARGET_DIM_Z0 = 18
X0_COORD_VALUES = np.arange(-249.5, 250.5, 1.0, dtype=np.float32)[:TARGET_DIM_X0]
Y0_COORD_VALUES = np.arange(-249.5, 250.5, 1.0, dtype=np.float32)[:TARGET_DIM_Y0]
Z0_COORD_VALUES = np.arange(1.0, 19.0, 1.0, dtype=np.float32)[:TARGET_DIM_Z0]

PROJ_ORIGIN_LON = -68.0169982910156; PROJ_ORIGIN_LAT = -34.6479988098145
PROJ_EARTH_RADIUS_M = 6378137.0; PROJ_FALSE_EASTING = 0.0; PROJ_FALSE_NORTHING = 0.0
proj_aeqd = pyproj.Proj(proj="aeqd", lon_0=PROJ_ORIGIN_LON, lat_0=PROJ_ORIGIN_LAT, R=PROJ_EARTH_RADIUS_M)
x0_grid_m, y0_grid_m = np.meshgrid(X0_COORD_VALUES * 1000.0, Y0_COORD_VALUES * 1000.0)
lon0_grid, lat0_grid = proj_aeqd(x0_grid_m, y0_grid_m, inverse=True)

search_path = os.path.join(input_root_dir, '**', '*.nc')
all_files = glob.glob(search_path, recursive=True)
logging.info(f"Se encontraron {len(all_files)} archivos para procesar.")

for nc_file_in_path in all_files:
    relative_path = os.path.relpath(nc_file_in_path, input_root_dir)
    output_file_path = os.path.join(output_root_dir, relative_path)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    if os.path.exists(output_file_path):
        logging.info(f"Omitiendo archivo ya existente: {output_file_path}")
        continue
    
    logging.info(f"Procesando: {nc_file_in_path}")
    
    try:
        # --- PASO 1: LECTURA ROBUSTA CON XARRAY ---
        ds = xr.open_dataset(nc_file_in_path, mask_and_scale=True, decode_times=False)
        dbz_physical_input = ds['DBZ'].values
        
        # --- PASO 2: LIMPIEZA Y PROCESAMIENTO ---
        dbz_working = dbz_physical_input[0].copy()
        irrelevant_data_mask = (dbz_working < MIN_RELEVANT_DBZ) | np.isnan(dbz_working)
        dbz_working[irrelevant_data_mask] = np.nan
        
        # --- PASO 3: REDIMENSIONAMIENTO (si es necesario) ---
        if dbz_working.shape != (TARGET_DIM_Z0, TARGET_DIM_Y0, TARGET_DIM_X0):
            valid_mask = ~np.isnan(dbz_working)
            filled_for_zoom = np.nan_to_num(dbz_working, nan=0.0)
            factors = (TARGET_DIM_Z0/dbz_working.shape[0], TARGET_DIM_Y0/dbz_working.shape[1], TARGET_DIM_X0/dbz_working.shape[2])
            resized_data = zoom(filled_for_zoom, factors, order=0, mode='mirror', prefilter=False)
            resized_mask_float = zoom(valid_mask.astype(float), factors, order=0, mode='mirror', prefilter=False)
            dbz_final_physical = resized_data
            dbz_final_physical[resized_mask_float <= 0.5] = np.nan
        else:
            dbz_final_physical = dbz_working

        # --- PASO 4: EMPAQUETADO FINAL ---
        dbz_clipped = np.clip(dbz_final_physical, CLIP_PHYSICAL_MIN_DBZ, CLIP_PHYSICAL_MAX_DBZ)
        dbz_for_packing = np.where(np.isnan(dbz_clipped), PHYSICAL_FILL_VALUE, dbz_clipped)
        dbz_packed_byte = np.round((dbz_for_packing - OUTPUT_NC_ADD_OFFSET) / OUTPUT_NC_SCALE_FACTOR).astype(np.int8)
        dbz_packed_byte[np.isclose(dbz_for_packing, PHYSICAL_FILL_VALUE)] = OUTPUT_NC_FILL_BYTE
        dbz_final_packed = dbz_packed_byte[np.newaxis, ...]

        # --- PASO 5: ESCRITURA ---
        with nc.Dataset(output_file_path, 'w', format='NETCDF3_CLASSIC') as ds_out:
            ds_out.setncatts({k: v for k, v in ds.attrs.items() if k not in ['_NCProperties']})
            ds_out.history = f"Standardized on {datetime.now(timezone.utc).isoformat()}"
            ds_out.createDimension('time', None); ds_out.createDimension('bounds', 2); ds_out.createDimension('x0', TARGET_DIM_X0); ds_out.createDimension('y0', TARGET_DIM_Y0); ds_out.createDimension('z0', TARGET_DIM_Z0)
            
            # <<< CORRECCIÓN CLAVE: Se leen las variables de tiempo desde el objeto 'ds' de xarray >>>
            for var_name in ['time', 'start_time', 'stop_time', 'time_bounds']:
                if var_name in ds:
                    var_out = ds_out.createVariable(var_name, ds[var_name].dtype, ds[var_name].dims)
                    var_out.setncatts(ds[var_name].attrs)
                    var_out[:] = ds[var_name].values
            
            # Creación de las nuevas coordenadas estandarizadas
            for var_name, dim_name, coord_vals, attrs in [('x0', 'x0', X0_COORD_VALUES, {'standard_name':"projection_x_coordinate", 'units':"km", 'axis':"X"}), ('y0', 'y0', Y0_COORD_VALUES, {'standard_name':"projection_y_coordinate", 'units':"km", 'axis':"Y"}), ('z0', 'z0', Z0_COORD_VALUES, {'standard_name':"altitude", 'units':"km", 'axis':"Z", 'positive':"up", 'long_name':"constant altitude levels"})]: 
                v = ds_out.createVariable(var_name, 'f4', (dim_name,)); v.setncatts(attrs); v[:] = coord_vals
            
            # Creación de las variables de georreferenciación
            lat0_v = ds_out.createVariable('lat0', 'f4', ('y0', 'x0',)); lat0_v.standard_name = "latitude"; lat0_v.units = "degrees_north"; lat0_v[:] = lat0_grid; 
            lon0_v = ds_out.createVariable('lon0', 'f4', ('y0', 'x0',)); lon0_v.standard_name = "longitude"; lon0_v.units = "degrees_east"; lon0_v[:] = lon0_grid; 
            gm_v = ds_out.createVariable('grid_mapping_0', 'i4'); gm_v.grid_mapping_name = "azimuthal_equidistant"; gm_v.longitude_of_projection_origin = PROJ_ORIGIN_LON; gm_v.latitude_of_projection_origin = PROJ_ORIGIN_LAT; gm_v.false_easting = PROJ_FALSE_EASTING; gm_v.false_northing = PROJ_FALSE_NORTHING; gm_v.earth_radius = PROJ_EARTH_RADIUS_M
            
            # Creación y escritura de la variable DBZ final
            dbz_out_var = ds_out.createVariable('DBZ', 'i1', ('time', 'z0', 'y0', 'x0'), fill_value=OUTPUT_NC_FILL_BYTE)
            dbz_out_var.setncatts({'scale_factor': OUTPUT_NC_SCALE_FACTOR, 'add_offset': OUTPUT_NC_ADD_OFFSET, 'valid_min': np.int8(-127), 'valid_max': np.int8(127), 'min_value': np.float32(CLIP_PHYSICAL_MIN_DBZ), 'max_value': np.float32(CLIP_PHYSICAL_MAX_DBZ), 'standard_name': "DBZ", 'long_name': "DBZ", 'units': "dBZ", 'coordinates': "lon0 lat0", 'grid_mapping': "grid_mapping_0"})
            dbz_out_var[:] = dbz_final_packed

        logging.info(f"Archivo estandarizado guardado en: {output_file_path}")

    except Exception as e:
        logging.error(f"Error CRÍTICO procesando {nc_file_in_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())