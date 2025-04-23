import netCDF4 as nc
import numpy as np

# Path al NetCDF
nc_file = "/home/f-caballero/UM/netCDF/ncfdata20100101_170906.nc"

# Abrir el archivo
ds = nc.Dataset(nc_file, "r")

# Inspeccionar variables y dimensiones
print("Variables:", ds.variables.keys())
print("Dimensiones:", ds.dimensions.keys())
print("Atributos globales:", ds.__dict__)

# Leer el campo DBZ
dbz = ds.variables["DBZ"][:]  # Shape: (1, 18, 500, 500)
dbz = np.squeeze(dbz, axis=0)  # Eliminar dimensión time, Shape: (18, 500, 500)
print("Shape de DBZ ajustado:", dbz.shape)

# Verificar rango de valores (debería ser -29.0 a 60.5, según MDV)
print("Min DBZ:", np.min(dbz), "Max DBZ:", np.max(dbz))

# Generar coordenadas x, y, z usando metadatos del MDV
grid_dx = 1.0  # km, de grid_dx en MDV
grid_dy = 1.0
grid_dz = 1.0
grid_minx = -249.5  # km, de grid_minx en MDV
grid_miny = -249.5
grid_minz = 1.0  # km

x = np.arange(grid_minx, grid_minx + 500 * grid_dx, grid_dx)  # 500 puntos
y = np.arange(grid_miny, grid_miny + 500 * grid_dy, grid_dy)
z = np.arange(grid_minz, grid_minz + 18 * grid_dz, grid_dz)

print("x shape:", x.shape, "y shape:", y.shape, "z shape:", z.shape)
print("x range:", x[0], "to", x[-1])
print("y range:", y[0], "to", y[-1])
print("z range:", z[0], "to", z[-1])

# Cerrar el archivo
ds.close()

# Opcional: Guardar NetCDF ajustado sin dimensión time
output_file = "/home/f-caballero/UM/netCDF/ncfdata20100101_170906_adjusted.nc"
ds_out = nc.Dataset(output_file, "w", format="NETCDF4")
ds_out.createDimension("z0", 18)
ds_out.createDimension("y0", 500)
ds_out.createDimension("x0", 500)

# Variables de coordenadas
x_var = ds_out.createVariable("x0", np.float32, ("x0",))
y_var = ds_out.createVariable("y0", np.float32, ("y0",))
z_var = ds_out.createVariable("z0", np.float32, ("z0",))
x_var[:] = x
y_var[:] = y
z_var[:] = z
x_var.units = "km"
y_var.units = "km"
z_var.units = "km"

# Variable DBZ
dbz_var = ds_out.createVariable("DBZ", np.float32, ("z0", "y0", "x0"))
dbz_var[:] = dbz
dbz_var.units = "dBZ"

# Metadatos
ds_out.Conventions = "CF-1.6"
ds_out.grid_dx = 1.0
ds_out.grid_dy = 1.0
ds_out.grid_dz = 1.0
ds_out.grid_minx = -249.5
ds_out.grid_miny = -249.5
ds_out.grid_minz = 1.0
ds_out.proj_type = 8
ds_out.proj_origin_lat = -34.64799880981445
ds_out.proj_origin_lon = -68.01699829101562

ds_out.close()
print(f"NetCDF ajustado guardado en: {output_file}")