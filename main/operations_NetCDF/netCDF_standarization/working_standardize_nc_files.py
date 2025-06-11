# standardize_nc_files_FINAL.py
import os
import glob
import netCDF4 as nc
import numpy as np
from scipy.ndimage import zoom
from datetime import datetime, timezone
import pyproj
import logging

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

input_root_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/500x500x18_diff_scale/20150105/201501051"
output_root_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/netcdf_500x500x18/standardized/201501051"

# --- Constantes Físicas y de Formato ---
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
logging.info(f"Se encontraron {len(all_files)} archivos .nc para procesar.")

for nc_file_in_path in all_files:
    relative_path = os.path.relpath(nc_file_in_path, input_root_dir)
    output_file_path = os.path.join(output_root_dir, relative_path)
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    if os.path.exists(output_file_path):
        logging.info(f"Omitiendo archivo ya existente: {output_file_path}")
        continue
    
    logging.info(f"Procesando: {nc_file_in_path}")
    
    try:
        with nc.Dataset(nc_file_in_path, 'r') as ds_in:
            if 'DBZ' not in ds_in.variables:
                logging.warning(f"Variable 'DBZ' no encontrada en {nc_file_in_path}. Omitiendo.")
                continue
            
            # --- Lectura y Desempaquetado ---
            dbz_var_in = ds_in.variables['DBZ']
            data_raw = dbz_var_in[:]
            scale_in = getattr(dbz_var_in, 'scale_factor', 1.0)
            offset_in = getattr(dbz_var_in, 'add_offset', 0.0)
            fill_value_packed_in = getattr(dbz_var_in, '_FillValue', None)
            
            dbz_physical = data_raw.astype(np.float32) * scale_in + offset_in
            if fill_value_packed_in is not None:
                dbz_physical[data_raw == fill_value_packed_in] = np.nan
            
            # --- Limpieza Robusta ---
            # 1. Eliminar ceros, que son una fuente común de error.
            dbz_physical[np.isclose(dbz_physical, 0.0)] = np.nan
            # 2. Eliminar cualquier valor por debajo de nuestro mínimo físico.
            dbz_physical[dbz_physical < CLIP_PHYSICAL_MIN_DBZ] = np.nan
            
            dbz_3d = dbz_physical[0]

            # --- Redimensionamiento (si es necesario) con manejo de NaN ---
            if dbz_3d.shape != (TARGET_DIM_Z0, TARGET_DIM_Y0, TARGET_DIM_X0):
                valid_data_mask = ~np.isnan(dbz_3d)
                dbz_filled_for_zoom = np.nan_to_num(dbz_3d, nan=0.0)
                factors = (TARGET_DIM_Z0 / dbz_3d.shape[0], TARGET_DIM_Y0 / dbz_3d.shape[1], TARGET_DIM_X0 / dbz_3d.shape[2])
                
                resized_data = zoom(dbz_filled_for_zoom, factors, order=0, mode='mirror', prefilter=False)
                resized_mask_float = zoom(valid_data_mask.astype(float), factors, order=0, mode='mirror', prefilter=False)
                resized_mask = resized_mask_float > 0.5
                
                resized_data[~resized_mask] = np.nan
                dbz_final_physical = resized_data
            else:
                dbz_final_physical = dbz_3d

            # --- Clippeo y Empaquetado Final ---
            dbz_clipped = np.clip(dbz_final_physical, CLIP_PHYSICAL_MIN_DBZ, CLIP_PHYSICAL_MAX_DBZ)
            dbz_for_packing = np.where(np.isnan(dbz_clipped), PHYSICAL_FILL_VALUE, dbz_clipped)
            
            dbz_packed_byte = np.round((dbz_for_packing - OUTPUT_NC_ADD_OFFSET) / OUTPUT_NC_SCALE_FACTOR).astype(np.int8)
            dbz_packed_byte[np.isclose(dbz_for_packing, PHYSICAL_FILL_VALUE)] = OUTPUT_NC_FILL_BYTE
            dbz_final_packed = dbz_packed_byte[np.newaxis, ...]

            # --- Escritura del Archivo NetCDF ---
            with nc.Dataset(output_file_path, 'w', format='NETCDF3_CLASSIC') as ds_out:
                # ... (Lógica de escritura de dimensiones y metadatos) ...
                ds_out.Conventions = "CF-1.6"; ds_out.history = f"Standardized on {datetime.now(timezone.utc).isoformat()}"
                ds_out.createDimension('time', None); ds_out.createDimension('bounds', 2); ds_out.createDimension('x0', TARGET_DIM_X0); ds_out.createDimension('y0', TARGET_DIM_Y0); ds_out.createDimension('z0', TARGET_DIM_Z0)
                for var_info in [{'name': 'time', 'dims': ('time',), 'dtype': 'f8'}, {'name': 'start_time', 'dims': ('time',), 'dtype': 'f8'}, {'name': 'stop_time', 'dims': ('time',), 'dtype': 'f8'}, {'name': 'time_bounds', 'dims': ('time', 'bounds'), 'dtype': 'f8'}]:
                    if var_info['name'] in ds_in.variables: var_in = ds_in.variables[var_info['name']]; var_out = ds_out.createVariable(var_info['name'], var_in.dtype, var_in.dimensions); var_out.setncatts({k:var_in.getncattr(k) for k in var_in.ncattrs()}); var_out[:] = var_in[:]
                for var_name, dim_name, coord_vals, attrs in [('x0', 'x0', X0_COORD_VALUES, {'standard_name':"projection_x_coordinate", 'units':"km", 'axis':"X"}), ('y0', 'y0', Y0_COORD_VALUES, {'standard_name':"projection_y_coordinate", 'units':"km", 'axis':"Y"}), ('z0', 'z0', Z0_COORD_VALUES, {'standard_name':"altitude", 'units':"km", 'axis':"Z", 'positive':"up", 'long_name':"constant altitude levels"})]: v = ds_out.createVariable(var_name, 'f4', (dim_name,)); v.setncatts(attrs); v[:] = coord_vals
                lat0_v = ds_out.createVariable('lat0', 'f4', ('y0', 'x0',)); lat0_v.standard_name = "latitude"; lat0_v.units = "degrees_north"; lat0_v[:] = lat0_grid; lon0_v = ds_out.createVariable('lon0', 'f4', ('y0', 'x0',)); lon0_v.standard_name = "longitude"; lon0_v.units = "degrees_east"; lon0_v[:] = lon0_grid; gm_v = ds_out.createVariable('grid_mapping_0', 'i4'); gm_v.grid_mapping_name = "azimuthal_equidistant"; gm_v.longitude_of_projection_origin = PROJ_ORIGIN_LON; gm_v.latitude_of_projection_origin = PROJ_ORIGIN_LAT; gm_v.false_easting = PROJ_FALSE_EASTING; gm_v.false_northing = PROJ_FALSE_NORTHING; gm_v.earth_radius = PROJ_EARTH_RADIUS_M
                dbz_out_var = ds_out.createVariable('DBZ', 'i1', ('time', 'z0', 'y0', 'x0'), fill_value=OUTPUT_NC_FILL_BYTE)
                dbz_out_var.scale_factor = OUTPUT_NC_SCALE_FACTOR; dbz_out_var.add_offset = OUTPUT_NC_ADD_OFFSET; dbz_out_var.valid_min = np.int8(-127); dbz_out_var.valid_max = np.int8(127); dbz_out_var.min_value = np.float32(CLIP_PHYSICAL_MIN_DBZ); dbz_out_var.max_value = np.float32(CLIP_PHYSICAL_MAX_DBZ)
                # CORRECCIÓN DEL TYPO:
                dbz_out_var.standard_name = "DBZ"; dbz_out_var.long_name = "DBZ"; dbz_out_var.units = "dBZ"; dbz_out_var.coordinates = "lon0 lat0"; dbz_out_var.grid_mapping = "grid_mapping_0"
                dbz_out_var[:] = dbz_final_packed

            logging.info(f"Archivo estandarizado guardado en: {output_file_path}")

    except Exception as e:
        logging.error(f"Error CRÍTICO procesando {nc_file_in_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())