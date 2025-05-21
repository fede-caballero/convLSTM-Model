import netCDF4
import matplotlib.pyplot as plt
import numpy as np # Necesitarás numpy para np.max

# --- Rutas a los archivos (¡ASEGÚRATE QUE SEAN CORRECTAS!) ---
# Archivo de predicción del modelo
pred_path = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/prediction_output_20250516_182532.nc"

# Archivo objetivo real (el 7mo archivo de la secuencia de validación/prueba correspondiente)
# Debes identificar cuál es. Ejemplo:
# Si 'pred_..._sample1.nc' corresponde a la SEGUNDA secuencia en tu 'val_subdirs'
# y esa subcarpeta es, digamos, '2020020322', y el 7mo archivo es '051030.nc':
target_path = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/Big-Sample-New/201511164/122913.nc" # <--- ¡CAMBIA ESTO AL REAL!

# Nombre de la variable de reflectividad en los archivos
var_name_pred = 'DBZ_forecast' # En tu archivo de predicción
var_name_true = 'DBZ'          # En tu archivo real (original)

# --- Cargar y calcular composite de la PREDICCIÓN ---
try:
    with netCDF4.Dataset(pred_path, 'r') as nc_pred:
        # dbz_pred_all_layers tiene forma (time, z, y, x) -> tomamos time=0
        # y luego todas las capas z. Resultado: (z, y, x)
        dbz_pred_all_layers = nc_pred.variables[var_name_pred][0, :, :, :]
        # Calcular el máximo a lo largo del eje Z (eje 0 en este array 3D)
        dbz_pred_composite = np.max(dbz_pred_all_layers, axis=0)
        print(f"Forma del composite de predicción: {dbz_pred_composite.shape}")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de predicción en {pred_path}")
    dbz_pred_composite = None
except Exception as e:
    print(f"Error al cargar o procesar el archivo de predicción: {e}")
    dbz_pred_composite = None

# --- Cargar y calcular composite del archivo REAL (OBJETIVO) ---
try:
    with netCDF4.Dataset(target_path, 'r') as nc_true:
        # dbz_true_all_layers tiene forma (time, z, y, x) -> tomamos time=0
        # y luego todas las capas z. Resultado: (z, y, x)
        dbz_true_all_layers = nc_true.variables[var_name_true][0, :, :, :]
        # Calcular el máximo a lo largo del eje Z (eje 0 en este array 3D)
        dbz_true_composite = np.max(dbz_true_all_layers, axis=0)
        print(f"Forma del composite real: {dbz_true_composite.shape}")
except FileNotFoundError:
    print(f"Advertencia: No se encontró el archivo objetivo en {target_path}")
    dbz_true_composite = None
except Exception as e:
    print(f"Error al cargar o procesar el archivo objetivo: {e}")
    dbz_true_composite = None


# --- Graficar ---
# Determinar cuántos subplots se necesitan
num_plots = 0
if dbz_pred_composite is not None:
    num_plots += 1
if dbz_true_composite is not None:
    num_plots += 1

if num_plots > 0:
    plt.figure(figsize=(6 * num_plots, 5)) # Ajustar tamaño de figura
    plot_index = 1

    # Colormap y rango para la barra de colores (ajusta vmin y vmax según tus datos)
    cmap_to_use = 'jet' # o 'gist_ncar', 'viridis', etc.
    vmin_val = -20   # Valor mínimo para la escala de colores
    vmax_val = 70    # Valor máximo para la escala de colores

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
    plt.show()
else:
    print("No hay datos para graficar (ni predicción ni objetivo encontrados/procesados).")