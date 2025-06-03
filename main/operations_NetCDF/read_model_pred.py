import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import os # Añadido para os.path.exists si es necesario

# --- Rutas a los archivos (¡ASEGÚRATE QUE SEAN CORRECTAS!) ---
# Archivo de predicción del modelo
# Este es el que genera tu función generate_prediction_netcdf
pred_path = "/home/f-caballero/UM/TIF3/convLSTM-project/Modificaciones_modelo/WorkinModel_30-05-25/020625-Second_try/pred_DBZ_20250602_184053_sample1.nc"

# Archivo objetivo real (el 7mo archivo de la secuencia de validación/prueba correspondiente)
# DEBES ASEGURARTE QUE ESTE SEA EL ARCHIVO CORRECTO QUE CORRESPONDE A LA PREDICCIÓN
# Si tu predicción es para el paso t+5 (usando 20 de entrada), y la secuencia de entrada
# termina en el archivo X, entonces el objetivo es el archivo X+5.
target_path = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/2010010112/182534.nc"

# Nombre de la variable de reflectividad en los archivos
var_name_pred_in_nc = 'DBZ' # El nombre que le pusiste en generate_prediction_netcdf (antes era DBZ_forecast)
var_name_true_in_nc = 'DBZ' # El nombre en tus archivos de datos originales/objetivo

# Parámetros de desempaquetado para la PREDICCIÓN
# Estos DEBEN COINCIDIR con los que usaste para GUARDAR el NetCDF de predicción
# (los output_nc_scale_factor y output_nc_add_offset de tu config)
pred_scale_factor = 0.5  # Ejemplo, toma el valor de tu config['output_nc_scale_factor']
pred_add_offset = 33.5 # Ejemplo, toma el valor de tu config['output_nc_add_offset']
pred_fill_value_byte = -128 # El _FillValue byte que usaste al guardar

# Parámetros de desempaquetado para el archivo REAL (OBJETIVO)
# Si tus archivos NetCDF "reales" (objetivo) también están empaquetados, necesitas sus parámetros.
# Si son flotantes directos, pon scale=1.0 y offset=0.0.
# Asumiré que son como los NetCDF generados por Mdv2NetCDF:
true_scale_factor = 0.5  # Ejemplo, ajústalo si es diferente en tus archivos objetivo
true_add_offset = 33.5 # Ejemplo, ajústalo
true_fill_value_byte = -128

# Valor físico para los datos faltantes después del desempaquetado
# Esto es (-128 * 0.5) + 33.5 = -30.5
physical_fill_value_pred = (pred_fill_value_byte * pred_scale_factor) + pred_add_offset
physical_fill_value_true = (true_fill_value_byte * true_scale_factor) + true_add_offset


# --- Función para cargar, desempaquetar (si es necesario) y hacer composite ---
def load_and_composite_dbz(file_path, var_name, is_prediction_format=False, scale=1.0, offset=0.0, fill_byte=None, physical_fill=None):
    """
    Carga datos de reflectividad, los desempaqueta si son byte, y calcula el composite.
    Si is_prediction_format es True, usa los parámetros de escala/offset de la predicción.
    """
    if not os.path.exists(file_path):
        print(f"ADVERTENCIA: No se encontró el archivo en {file_path}")
        return None
        
    try:
        with netCDF4.Dataset(file_path, 'r') as nc_file:
            if var_name not in nc_file.variables:
                print(f"ADVERTENCIA: Variable '{var_name}' no encontrada en {file_path}")
                return None

            dbz_var = nc_file.variables[var_name]
            
            # Asumimos que la variable tiene al menos forma (time, z, y, x)
            # o (z,y,x) si la dimensión de tiempo no existe o es implícita.
            # Tomamos el primer (o único) tiempo.
            if dbz_var.ndim == 4:
                dbz_data_packed_or_float = dbz_var[0, :, :, :] 
            elif dbz_var.ndim == 3: # Si solo es (z,y,x)
                dbz_data_packed_or_float = dbz_var[:, :, :]
            else:
                print(f"ADVERTENCIA: Forma de variable inesperada {dbz_var.shape} para '{var_name}' en {file_path}")
                return None

            # Desempaquetar si es byte y tiene scale_factor y add_offset
            if dbz_data_packed_or_float.dtype == np.int8 or dbz_data_packed_or_float.dtype == 'i1':
                # Si los atributos scale_factor y add_offset están en el archivo, úsalos.
                # Si no, usa los pasados como argumento (que vienen de la config).
                s = dbz_var.scale_factor if hasattr(dbz_var, 'scale_factor') else scale
                o = dbz_var.add_offset if hasattr(dbz_var, 'add_offset') else offset
                fb = dbz_var._FillValue if hasattr(dbz_var, '_FillValue') else fill_byte
                
                print(f"Desempaquetando {var_name} de {file_path} con scale={s}, offset={o}, fill_byte={fb}")
                
                # Convertir a float ANTES de la operación para evitar overflow/underflow con bytes
                dbz_data_float = dbz_data_packed_or_float.astype(np.float32)
                
                # Identificar dónde estaba el fill value original (byte)
                is_fill_value_location = (dbz_data_packed_or_float == fb)
                
                # Aplicar escala y offset
                dbz_data_physical = dbz_data_float * s + o
                
                # Donde estaba el fill value original, poner el fill value físico deseado
                if physical_fill is not None:
                    dbz_data_physical[is_fill_value_location] = physical_fill
                
            elif dbz_data_packed_or_float.dtype == np.float32 or dbz_data_packed_or_float.dtype == np.float64:
                print(f"Datos de {var_name} en {file_path} ya son float.")
                dbz_data_physical = dbz_data_packed_or_float.astype(np.float32)
                # Si ya es float, y tiene un _FillValue, reemplazarlo por un NaN o un valor físico de fill estándar
                if hasattr(dbz_var, '_FillValue') and physical_fill is not None:
                    dbz_data_physical[dbz_data_physical == dbz_var._FillValue] = physical_fill
            else:
                print(f"ADVERTENCIA: Tipo de dato no manejado {dbz_data_packed_or_float.dtype} para '{var_name}' en {file_path}")
                return None

            # Calcular el máximo a lo largo del eje Z (eje 0 en este array 3D)
            dbz_composite = np.max(dbz_data_physical, axis=0)
            print(f"Forma del composite de '{var_name}' de {file_path}: {dbz_composite.shape}")
            print(f"Min/Max del composite: {np.min(dbz_composite)}, {np.max(dbz_composite)}")
            return dbz_composite
            
    except Exception as e:
        print(f"Error al cargar o procesar el archivo {file_path}: {e}")
        return None

# --- Cargar y calcular composites ---
print("--- Procesando Predicción ---")
dbz_pred_composite = load_and_composite_dbz(pred_path, var_name_pred_in_nc, 
                                            is_prediction_format=True, # Para indicar que use los params de predicción
                                            scale=pred_scale_factor, 
                                            offset=pred_add_offset,
                                            fill_byte=pred_fill_value_byte,
                                            physical_fill=physical_fill_value_pred)

print("\n--- Procesando Real (Objetivo) ---")
dbz_true_composite = load_and_composite_dbz(target_path, var_name_true_in_nc,
                                            scale=true_scale_factor,
                                            offset=true_add_offset,
                                            fill_byte=true_fill_value_byte,
                                            physical_fill=physical_fill_value_true)


# --- Graficar ---
num_plots = 0
if dbz_pred_composite is not None: num_plots += 1
if dbz_true_composite is not None: num_plots += 1

if num_plots > 0:
    plt.figure(figsize=(7 * num_plots, 6)) 
    plot_index = 1

    cmap_to_use = 'jet' 
    vmin_val = -30   # Ajustado para ver valores negativos (como tu -28.5 dBZ o -30.5 dBZ)
    vmax_val = 75    # Ajustado para ver hasta 75 dBZ

    if dbz_pred_composite is not None:
        plt.subplot(1, num_plots, plot_index)
        plt.imshow(dbz_pred_composite, origin='lower', cmap=cmap_to_use, vmin=vmin_val, vmax=vmax_val)
        plt.title("Composite Predicción")
        plt.colorbar(label="Max dBZ")
        plot_index += 1

    if dbz_true_composite is not None:
        plt.subplot(1, num_plots, plot_index)
        plt.imshow(dbz_true_composite, origin='lower', cmap=cmap_to_use, vmin=vmin_val, vmax=vmax_val)
        plt.title("Composite Real (Objetivo)")
        plt.colorbar(label="Max dBZ")
        plot_index += 1
    
    plt.tight_layout()
    # Guardar la figura
    output_figure_path = "comparacion_prediccion_real.png"
    plt.savefig(output_figure_path)
    print(f"\nFigura guardada en: {output_figure_path}")
    plt.show()
else:
    print("No hay datos para graficar.")