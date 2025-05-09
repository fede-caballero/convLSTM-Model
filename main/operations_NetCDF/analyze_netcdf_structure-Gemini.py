import netCDF4
import numpy as np
import os

# Asegúrate de que la biblioteca netCDF4 está instalada.
# Puedes instalarla con: pip install netCDF4

def analyze_nc_file(file_path):
    """
    Analiza y muestra la estructura detallada de un archivo NetCDF.
    """
    print(f"--- Iniciando análisis del archivo: {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: El archivo '{file_path}' no existe o la ruta es incorrecta.")
        print(f"--- Fin del análisis (ERROR) de: {file_path} ---\n")
        return
    if not os.access(file_path, os.R_OK):
        print(f"Error: No se tienen permisos de lectura para el archivo '{file_path}'.")
        print(f"--- Fin del análisis (ERROR) de: {file_path} ---\n")
        return

    try:
        with netCDF4.Dataset(file_path, 'r') as nc_file:
            # 1. Dimensiones Globales
            print("\n[Dimensiones Globales]")
            if not nc_file.dimensions:
                print("  No se encontraron dimensiones globales.")
            for dim_name, dim_obj in nc_file.dimensions.items():
                print(f"  Nombre de Dimensión: {dim_name}")
                print(f"    Tamaño: {len(dim_obj)}")
                print(f"    ¿Ilimitada?: {dim_obj.isunlimited()}")

            # 2. Atributos Globales
            print("\n[Atributos Globales]")
            global_attrs = nc_file.ncattrs()
            if not global_attrs:
                print("  No se encontraron atributos globales.")
            for attr_name in global_attrs:
                try:
                    print(f"  {attr_name}: {nc_file.getncattr(attr_name)}")
                except Exception as e:
                    print(f"  Error al leer atributo global '{attr_name}': {e}")


            # 3. Variables y sus detalles
            print("\n[Variables]")
            if not nc_file.variables:
                print("  No se encontraron variables.")
            for var_name, var_obj in nc_file.variables.items():
                print(f"\n  Variable: {var_name}")
                print(f"    Dimensiones que usa: {var_obj.dimensions}")
                print(f"    Forma (Shape): {var_obj.shape}")
                print(f"    Tipo de Dato: {var_obj.dtype}")

                print("    Atributos de la Variable:")
                var_attrs = var_obj.ncattrs()
                if not var_attrs:
                    print("      No tiene atributos.")
                for attr_name in var_attrs:
                    try:
                        print(f"      {attr_name}: {var_obj.getncattr(attr_name)}")
                    except Exception as e:
                        print(f"      Error al leer atributo '{attr_name}' de la variable '{var_name}': {e}")


                # Mostrar algunos valores de ejemplo
                print("    Valores de Ejemplo (primeros elementos si es un array):")
                if var_obj.ndim == 0: # Escalar
                    try:
                        print(f"      Valor: {var_obj[:]}")
                    except Exception as e:
                        print(f"      No se pudo leer el valor escalar: {e}")
                elif var_obj.size > 0 : # Array no vacío
                    try:
                        # Intentar tomar un slice. Para arrays grandes, tomar solo el inicio.
                        # Ajusta estos slices según la dimensionalidad esperada de tus variables
                        if var_obj.ndim == 1:
                            print(f"      {var_obj[:min(5, var_obj.shape[0])]}")
                        elif var_obj.ndim == 2: # e.g., (altura, tiempo) o (y, x)
                            print(f"      {var_obj[:min(3, var_obj.shape[0]), :min(3, var_obj.shape[1])]}")
                        elif var_obj.ndim == 3: # e.g., (tiempo, y, x) o (z, y, x)
                            print(f"      {var_obj[:min(2, var_obj.shape[0]), :min(2, var_obj.shape[1]), :min(3, var_obj.shape[2])]}")
                        elif var_obj.ndim == 4: # e.g., (batch, z, y, x) o (tiempo, z, y, x)
                             print(f"      {var_obj[:min(1, var_obj.shape[0]), :min(2, var_obj.shape[1]), :min(2, var_obj.shape[2]), :min(3, var_obj.shape[3])]}")
                        else: # Para dimensiones mayores, solo los primeros elementos aplanados
                            flat_data = var_obj[:].flatten()
                            print(f"      {flat_data[:min(5, flat_data.size)]}")
                    except Exception as e:
                        print(f"      No se pudieron mostrar valores de ejemplo para '{var_name}': {e}")
                else:
                    print(f"      La variable '{var_name}' está vacía o no se pudieron leer sus datos.")

    except Exception as e:
        print(f"Error CRÍTICO al procesar el archivo NetCDF '{file_path}': {e}")
    finally:
        print(f"--- Fin del análisis de: {file_path} ---\n")


if __name__ == "__main__":
    print("--- Script de análisis de NetCDF para VSCode ---")

    # --------------------------------------------------------------------
    # --- MODIFICA AQUÍ LA RUTA AL ARCHIVO QUE QUIERES ANALIZAR ---
    # --------------------------------------------------------------------
    # Asegúrate de que la ruta sea correcta y accesible desde donde ejecutas el script.
    # Ejemplo para Linux/macOS:
    FILE_TO_ANALYZE = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/Model_Outputs-Gemini/pred_DBZ_forecast_20250509_033057_sample0.nc"
    # Ejemplo para Windows (usa raw strings r"..." o barras dobles \\):
    # FILE_TO_ANALYZE = r"C:\ruta\a\tus\datos\archivo.nc"
    # FILE_TO_ANALYZE = "C:\\ruta\\a\\tus\\datos\\archivo.nc"
    # --------------------------------------------------------------------

    if FILE_TO_ANALYZE:
        analyze_nc_file(FILE_TO_ANALYZE)
    else:
        print("Por favor, establece la variable 'FILE_TO_ANALYZE' en el script con la ruta al archivo NetCDF.")

    print("--- Script finalizado ---")