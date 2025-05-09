# Ejemplo rápido para visualizar una capa
import netCDF4
import matplotlib.pyplot as plt

# Carga tu predicción
pred_path = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/jupyter_notebooks/GeminiModels/Salidas modelo andando/salidas/pred_DBZ_forecast_20250509_033057_sample0.nc" # Asegúrate que este path sea correcto en tu sistema local
with netCDF4.Dataset(pred_path, 'r') as nc_pred:
    # Verifica el nombre exacto de la variable en tu archivo NC de predicción.
    # En el script de entrenamiento, se llama config.get('dbz_variable_name_pred', 'DBZ_predicted')
    # que por defecto es 'DBZ_forecast'
    dbz_pred = nc_pred.variables['DBZ_forecast'][0, 10, :, :] # time=0, level=10 (ejemplo)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# --- CAMBIO AQUÍ ---
plt.imshow(dbz_pred, origin='lower', cmap='jet') # Cambiado a 'jet' o prueba 'viridis', 'gist_ncar'
# --- FIN DEL CAMBIO ---
plt.title("Predicción (Capa 10)")
plt.colorbar(label="dBZ")

# Si quieres comparar con el real, asegúrate de tener el path correcto y descomentar
# target_path = "/ruta/a/tu/first_try_nc/SUBDIR_VALIDACION_0/ARCHIVO_OBJETIVO_REAL.nc"
# try:
#     with netCDF4.Dataset(target_path, 'r') as nc_true:
#         dbz_true = nc_true.variables['DBZ'][0, 10, :, :] # Asumiendo que la variable original se llama 'DBZ'
#     plt.subplot(1, 2, 2)
#     plt.imshow(dbz_true, origin='lower', cmap='jet') # Usa el mismo cmap para comparar
#     plt.title("Real (Capa 10)")
#     plt.colorbar(label="dBZ")
# except FileNotFoundError:
#     print(f"Advertencia: No se encontró el archivo objetivo en {target_path}")
# except Exception as e:
#     print(f"Error al cargar o graficar el archivo objetivo: {e}")

plt.tight_layout() # Ajusta el espaciado
plt.show()