import netCDF4
import numpy as np

# --- Configuración ---
# Asegúrate que esta es la ruta al archivo que quieres modificar
netcdf_filepath = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/operations_nc_mdv/new_try_pyart/170906.nc"

# Nombres que Py-ART espera para las variables de coordenadas de la malla
pyart_x_coord_name = 'x'
pyart_y_coord_name = 'y'
pyart_z_coord_name = 'z'

# Nombres actuales en tu NetCDF
current_x_coord_name = 'x0'
current_y_coord_name = 'y0'
current_z_coord_name = 'z0'

# Nombres que Py-ART podría estar buscando como variables escalares de origen
pyart_origin_lat_var_name = 'origin_latitude'
pyart_origin_lon_var_name = 'origin_longitude'
pyart_origin_alt_var_name = 'origin_altitude' # En metros

grid_mapping_var_name = 'grid_mapping_0'
nc_origin_lat_attr = 'latitude_of_projection_origin'
nc_origin_lon_attr = 'longitude_of_projection_origin'
radar_altitude_km = 0.550 # Altitud del radar en km, ajusta este valor

try:
    print(f"Abriendo archivo NetCDF en modo lectura/escritura: {netcdf_filepath}")
    with netCDF4.Dataset(netcdf_filepath, 'r+') as ncfile:
        print("Archivo abierto exitosamente.")

        # --- Parte 1: Añadir variables de origen (como antes) ---
        grid_mapping_present = grid_mapping_var_name in ncfile.variables
        if not grid_mapping_present:
            print(f"ADVERTENCIA: La variable de mapeo de la malla '{grid_mapping_var_name}' no se encuentra.")
        else:
            grid_map_var = ncfile.variables[grid_mapping_var_name]
            # origin_latitude
            if pyart_origin_lat_var_name not in ncfile.variables and hasattr(grid_map_var, nc_origin_lat_attr):
                origin_lat_val = grid_map_var.getncattr(nc_origin_lat_attr)
                lat_var = ncfile.createVariable(pyart_origin_lat_var_name, 'f4')
                lat_var[:] = origin_lat_val
                lat_var.units = "degrees_north"; lat_var.long_name = "Latitude of grid origin"
                print(f"Variable '{pyart_origin_lat_var_name}' creada con valor: {origin_lat_val}")
            # origin_longitude
            if pyart_origin_lon_var_name not in ncfile.variables and hasattr(grid_map_var, nc_origin_lon_attr):
                origin_lon_val = grid_map_var.getncattr(nc_origin_lon_attr)
                lon_var = ncfile.createVariable(pyart_origin_lon_var_name, 'f4')
                lon_var[:] = origin_lon_val
                lon_var.units = "degrees_east"; lon_var.long_name = "Longitude of grid origin"
                print(f"Variable '{pyart_origin_lon_var_name}' creada con valor: {origin_lon_val}")
            # origin_altitude
            if pyart_origin_alt_var_name not in ncfile.variables:
                origin_alt_val_meters = radar_altitude_km * 1000.0
                alt_var = ncfile.createVariable(pyart_origin_alt_var_name, 'f4')
                alt_var[:] = origin_alt_val_meters
                alt_var.units = "m"; alt_var.long_name = "Altitude of grid origin above mean sea level"; alt_var.positive = "up"
                print(f"Variable '{pyart_origin_alt_var_name}' creada con valor: {origin_alt_val_meters} m")

        # --- Parte 2: Renombrar variables de coordenadas de la malla ---
        # Renombrar x0 a x (y su dimensión asociada si es necesario y diferente)
        if current_x_coord_name in ncfile.variables and pyart_x_coord_name not in ncfile.variables:
            if current_x_coord_name in ncfile.dimensions and pyart_x_coord_name not in ncfile.dimensions:
                 ncfile.renameDimension(current_x_coord_name, pyart_x_coord_name)
                 print(f"Dimensión '{current_x_coord_name}' renombrada a '{pyart_x_coord_name}'.")
            ncfile.renameVariable(current_x_coord_name, pyart_x_coord_name)
            print(f"Variable '{current_x_coord_name}' renombrada a '{pyart_x_coord_name}'.")
        elif pyart_x_coord_name in ncfile.variables:
            print(f"Variable '{pyart_x_coord_name}' ya existe.")

        # Renombrar y0 a y
        if current_y_coord_name in ncfile.variables and pyart_y_coord_name not in ncfile.variables:
            if current_y_coord_name in ncfile.dimensions and pyart_y_coord_name not in ncfile.dimensions:
                ncfile.renameDimension(current_y_coord_name, pyart_y_coord_name)
                print(f"Dimensión '{current_y_coord_name}' renombrada a '{pyart_y_coord_name}'.")
            ncfile.renameVariable(current_y_coord_name, pyart_y_coord_name)
            print(f"Variable '{current_y_coord_name}' renombrada a '{pyart_y_coord_name}'.")
        elif pyart_y_coord_name in ncfile.variables:
            print(f"Variable '{pyart_y_coord_name}' ya existe.")

        # Renombrar z0 a z
        if current_z_coord_name in ncfile.variables and pyart_z_coord_name not in ncfile.variables:
            if current_z_coord_name in ncfile.dimensions and pyart_z_coord_name not in ncfile.dimensions:
                ncfile.renameDimension(current_z_coord_name, pyart_z_coord_name)
                print(f"Dimensión '{current_z_coord_name}' renombrada a '{pyart_z_coord_name}'.")
            ncfile.renameVariable(current_z_coord_name, pyart_z_coord_name)
            print(f"Variable '{current_z_coord_name}' renombrada a '{pyart_z_coord_name}'.")
        elif pyart_z_coord_name in ncfile.variables:
            print(f"Variable '{pyart_z_coord_name}' ya existe.")

        # Es posible que también necesites renombrar las dimensiones en la variable DBZ
        if 'DBZ' in ncfile.variables:
            dbz_var = ncfile.variables['DBZ']
            new_dims = []
            changed_dims = False
            for dim_name in dbz_var.dimensions:
                if dim_name == current_x_coord_name and pyart_x_coord_name in ncfile.dimensions:
                    new_dims.append(pyart_x_coord_name)
                    changed_dims = True
                elif dim_name == current_y_coord_name and pyart_y_coord_name in ncfile.dimensions:
                    new_dims.append(pyart_y_coord_name)
                    changed_dims = True
                elif dim_name == current_z_coord_name and pyart_z_coord_name in ncfile.dimensions:
                    new_dims.append(pyart_z_coord_name)
                    changed_dims = True
                else:
                    new_dims.append(dim_name)
            
            if changed_dims:
                 # Esto es más complejo, porque no puedes cambiar las dimensiones de una variable existente directamente.
                 # La forma más segura sería crear una nueva variable DBZ con las dimensiones correctas,
                 # copiar los datos y atributos, y luego eliminar la antigua y renombrar la nueva.
                 # O, si SOLO los nombres de las dimensiones cambiaron y no su orden o significado:
                 print(f"Los nombres de las dimensiones de DBZ podrían necesitar actualización si las dimensiones fueron renombradas.")
                 print(f"Dimensiones actuales de DBZ: {dbz_var.dimensions}")
                 print(f"Dimensiones esperadas podrían ser: {tuple(new_dims)}")
                 # Si pyart.io.read_grid es suficientemente inteligente, podría manejar esto si las variables de coordenadas
                 # ahora se llaman 'x', 'y', 'z' y tienen los atributos 'axis' correctos.

        print("Proceso de modificación de NetCDF completado.")

except FileNotFoundError:
    print(f"ERROR: Archivo no encontrado en la ruta: {netcdf_filepath}")
except Exception as e:
    print(f"Ocurrió un error al intentar modificar el archivo NetCDF: {e}")

print("Script de modificación de NetCDF finalizado.")