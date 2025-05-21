import pyart
import netCDF4
import numpy as np
import os

# --- Configuración ---
input_netcdf_file = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/operations_nc_mdv/new_try_pyart/170906.nc"
output_mdv_dir = "./mdv_output_pyart"
output_mdv_filename = "170906_pyart_manual_grid.mdv"
output_mdv_filepath = os.path.join(output_mdv_dir, output_mdv_filename)

if not os.path.exists(output_mdv_dir):
    os.makedirs(output_mdv_dir)
    print(f"Directorio de salida creado: {output_mdv_dir}")

print(f"Intentando procesar archivo NetCDF: {input_netcdf_file}")

try:
    with netCDF4.Dataset(input_netcdf_file, 'r') as ncfile:
        print("NetCDF leído con netCDF4. Extrayendo datos...")

        dbz_data_raw = ncfile.variables['DBZ'][:]
        if dbz_data_raw.ndim == 4 and dbz_data_raw.shape[0] == 1:
            dbz_data_squeezed = dbz_data_raw.squeeze(axis=0)
        else:
            dbz_data_squeezed = dbz_data_raw
        dbz_field_data = dbz_data_squeezed.astype(np.float32)

        x_coords_nc = ncfile.variables['x']
        y_coords_nc = ncfile.variables['y']
        z_coords_nc = ncfile.variables['z']

        x_coords_data = x_coords_nc[:] * 1000.0
        y_coords_data = y_coords_nc[:] * 1000.0
        z_coords_data = z_coords_nc[:] * 1000.0

        time_var = ncfile.variables['time']
        time_data_raw = time_var[:]
        time_units_from_nc = time_var.units # Guardar las unidades originales
        time_data_for_pyart = np.atleast_1d(time_data_raw)

        origin_latitude_scalar = np.array(ncfile.variables['origin_latitude'][:]).item()
        origin_longitude_scalar = np.array(ncfile.variables['origin_longitude'][:]).item()
        origin_altitude_scalar = np.array(ncfile.variables['origin_altitude'][:]).item()

        grid_mapping_variable_name_in_nc = ncfile.variables['DBZ'].grid_mapping
        grid_mapping_var = ncfile.variables[grid_mapping_variable_name_in_nc]
        projection_params = {}
        for attr_name in grid_mapping_var.ncattrs():
            if attr_name == 'grid_mapping_name':
                if grid_mapping_var.getncattr(attr_name) == 'azimuthal_equidistant':
                    projection_params['proj'] = 'aeqd'
                else:
                    projection_params['proj'] = grid_mapping_var.getncattr(attr_name)
            elif attr_name == 'longitude_of_projection_origin':
                projection_params['lon_0'] = grid_mapping_var.getncattr(attr_name)
            elif attr_name == 'latitude_of_projection_origin':
                projection_params['lat_0'] = grid_mapping_var.getncattr(attr_name)
            elif attr_name == 'false_easting':
                projection_params['x_0'] = grid_mapping_var.getncattr(attr_name)
            elif attr_name == 'false_northing':
                projection_params['y_0'] = grid_mapping_var.getncattr(attr_name)
            elif attr_name == 'earth_radius':
                projection_params['R'] = grid_mapping_var.getncattr(attr_name)
            else:
                projection_params[attr_name] = grid_mapping_var.getncattr(attr_name)
        
        if 'proj' not in projection_params:
            raise ValueError("No se pudo determinar el 'proj' para la proyección desde grid_mapping.")
        if 'lat_0' not in projection_params:
            projection_params['lat_0'] = origin_latitude_scalar
        if 'lon_0' not in projection_params:
            projection_params['lon_0'] = origin_longitude_scalar

        fields_dict = {
            'DBZ': {
                'data': dbz_field_data,
                '_FillValue': ncfile.variables['DBZ']._FillValue if hasattr(ncfile.variables['DBZ'], '_FillValue') else pyart.config.get_fillvalue(),
                'units': ncfile.variables['DBZ'].units if hasattr(ncfile.variables['DBZ'], 'units') else 'dBZ',
                'standard_name': ncfile.variables['DBZ'].standard_name if hasattr(ncfile.variables['DBZ'], 'standard_name') else 'equivalent_reflectivity_factor',
                'long_name': ncfile.variables['DBZ'].long_name if hasattr(ncfile.variables['DBZ'], 'long_name') else 'Reflectivity',
                'coordinates': ncfile.variables['DBZ'].coordinates if hasattr(ncfile.variables['DBZ'], 'coordinates') else "time z y x",
                'grid_mapping': grid_mapping_variable_name_in_nc if hasattr(ncfile.variables['DBZ'], 'grid_mapping') else None
            }
        }

        # --- CORRECCIÓN PRINCIPAL AQUÍ para time, origin_*, x, y, z ---
        # Deben ser diccionarios Python estándar directamente.
        grid = pyart.core.Grid(
            time={'data': time_data_for_pyart, 'units': time_units_from_nc}, # Usar unidades leídas del NC
            fields=fields_dict,
            metadata={
                'instrument_name': ncfile.title if hasattr(ncfile, 'title') else 'UnknownRadar',
                'Conventions': ncfile.Conventions if hasattr(ncfile, 'Conventions') else 'CF-1.6',
                'source': ncfile.source if hasattr(ncfile, 'source') else 'Unknown',
                'institution': ncfile.institution if hasattr(ncfile, 'institution') else 'Unknown',
                'comment': ncfile.comment if hasattr(ncfile, 'comment') else '',
                'history': ncfile.history if hasattr(ncfile, 'history') else ''
            },
            origin_latitude={'data': np.array([origin_latitude_scalar], dtype=np.float64), 'units': 'degrees_north'},
            origin_longitude={'data': np.array([origin_longitude_scalar], dtype=np.float64), 'units': 'degrees_east'},
            origin_altitude={'data': np.array([origin_altitude_scalar], dtype=np.float64), 'units': 'm'},
            x={
                'data': x_coords_data.astype(np.float32),
                'long_name': x_coords_nc.long_name if hasattr(x_coords_nc, 'long_name') else 'Distance from radar in x direction',
                'standard_name': x_coords_nc.standard_name if hasattr(x_coords_nc, 'standard_name') else 'projection_x_coordinate',
                'units': 'm',
                'axis': x_coords_nc.axis if hasattr(x_coords_nc, 'axis') else 'X'
            },
            y={
                'data': y_coords_data.astype(np.float32),
                'long_name': y_coords_nc.long_name if hasattr(y_coords_nc, 'long_name') else 'Distance from radar in y direction',
                'standard_name': y_coords_nc.standard_name if hasattr(y_coords_nc, 'standard_name') else 'projection_y_coordinate',
                'units': 'm',
                'axis': y_coords_nc.axis if hasattr(y_coords_nc, 'axis') else 'Y'
            },
            z={
                'data': z_coords_data.astype(np.float32),
                'long_name': z_coords_nc.long_name if hasattr(z_coords_nc, 'long_name') else 'Height above mean sea level',
                'standard_name': z_coords_nc.standard_name if hasattr(z_coords_nc, 'standard_name') else 'altitude',
                'units': 'm',
                'positive': z_coords_nc.positive if hasattr(z_coords_nc, 'positive') else 'up',
                'axis': z_coords_nc.axis if hasattr(z_coords_nc, 'axis') else 'Z'
            },
            projection=projection_params
        )
        # --- FIN DE CORRECCIÓN PRINCIPAL ---

        print("Objeto Grid de Py-ART creado manualmente.")
        # Ahora sí, esta línea debería funcionar:
        print(f"  Dimensiones del Grid (Time, Z, Y, X): {grid.time['data'].shape[0]} x {grid.nz} x {grid.ny} x {grid.nx}")
        print(f"  Proyección del Grid: {grid.projection}")
        print(f"  Campos disponibles: {list(grid.fields.keys())}")

        print(f"Escribiendo a archivo MDV: {output_mdv_filepath}")
        pyart.io.write_grid_mdv(output_mdv_filepath, grid)
        print("Escritura a MDV exitosa.")
        print(f"Archivo MDV generado: {output_mdv_filepath}")

except FileNotFoundError:
    print(f"ERROR: Archivo no encontrado en la ruta: {input_netcdf_file}")
except Exception as e:
    print(f"Ocurrió un error: {e}")
    import traceback
    traceback.print_exc()

print("Proceso completado.")