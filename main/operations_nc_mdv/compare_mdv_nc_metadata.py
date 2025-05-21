import subprocess
import json # Para imprimir los diccionarios de Py-ART de forma más legible
import argparse
import os

# Intenta importar Py-ART, si no está disponible, algunas funciones no se ejecutarán.
try:
    import pyart
    PYART_AVAILABLE = True
except ImportError:
    PYART_AVAILABLE = False
    print("Advertencia: Py-ART no está instalado. No se podrán analizar archivos MDV.")

# Intenta importar netCDF4
try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False
    print("Advertencia: La biblioteca netCDF4 no está instalada. No se podrán analizar archivos NetCDF con ella directamente.")


def analyze_mdv_file_pyart(mdv_file_path):
    """
    Analiza un archivo MDV usando Py-ART y devuelve sus headers.
    """
    if not PYART_AVAILABLE:
        print(f"  Py-ART no disponible, no se puede analizar {mdv_file_path}")
        return None, None
    
    if not os.path.exists(mdv_file_path):
        print(f"  Error: Archivo MDV no encontrado en {mdv_file_path}")
        return None, None
        
    print(f"\n--- Análisis del Archivo MDV (con Py-ART): {mdv_file_path} ---")
    try:
        mdv_file = pyart.io.mdv_common.MdvFile(pyart.io.prepare_for_read(mdv_file_path))
        master_header = mdv_file.master_header
        field_headers = mdv_file.field_headers
        
        print("\n  [Master Header del MDV]")
        # Usamos json.dumps para una impresión más bonita de los diccionarios
        print(json.dumps(master_header, indent=4, default=str)) # default=str para manejar tipos no serializables

        print("\n  [Field Headers del MDV (mostrando el primero si hay múltiples campos)]")
        if field_headers:
            # Imprimir todos los field headers si son pocos, o solo el primero si son muchos
            for i, f_header in enumerate(field_headers):
                print(f"\n    Field Header [{i}]:")
                print(json.dumps(f_header, indent=4, default=str))
        else:
            print("    No se encontraron field headers.")
            
        print("--- Fin del Análisis MDV (Py-ART) ---")
        return master_header, field_headers
    except Exception as e:
        print(f"  Error al leer o analizar el archivo MDV '{mdv_file_path}' con Py-ART: {e}")
        print("--- Fin del Análisis MDV (Py-ART) con ERROR ---")
        return None, None

def analyze_nc_file_ncdump(nc_file_path):
    """
    Analiza un archivo NetCDF usando ncdump -h y devuelve la salida.
    """
    if not os.path.exists(nc_file_path):
        print(f"  Error: Archivo NetCDF no encontrado en {nc_file_path}")
        return None
        
    print(f"\n--- Análisis del Archivo NetCDF (con ncdump -h): {nc_file_path} ---")
    try:
        # Ejecutar ncdump -h como un subproceso
        ncdump_executable = "/usr/bin/ncdump"
        if not os.path.exists(ncdump_executable):
            ncdump_executable = "ncdump"  # Asumir que está en el PATH
            print(f"Advertencia: ncdump no encontrado en {ncdump_executable}. Usando el comando 'ncdump' directamente.")
        result = subprocess.run(['ncdump', '-h', nc_file_path], capture_output=True, text=True, check=True)
        ncdump_output = result.stdout
        print(ncdump_output)
        print("--- Fin del Análisis NetCDF (ncdump) ---")
        return ncdump_output
    except FileNotFoundError:
        print("  Error: El comando 'ncdump' no se encontró. Asegúrate de que las herramientas de NetCDF estén instaladas y en tu PATH.")
        print("--- Fin del Análisis NetCDF (ncdump) con ERROR ---")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  Error al ejecutar 'ncdump -h' en '{nc_file_path}': {e}")
        print(f"  Salida del error de ncdump:\n{e.stderr}")
        print("--- Fin del Análisis NetCDF (ncdump) con ERROR ---")
        return None
    except Exception as e:
        print(f"  Un error inesperado ocurrió al analizar NetCDF con ncdump '{nc_file_path}': {e}")
        print("--- Fin del Análisis NetCDF (ncdump) con ERROR ---")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Compara metadatos de un archivo MDV y su (potencial) equivalente NetCDF."
    )
    parser.add_argument(
        "--mdv",
        type=str,
        required=False, # Hacerlo opcional por si solo quieres analizar un NC
        help="Ruta al archivo MDV original."
    )
    parser.add_argument(
        "--nc",
        type=str,
        required=True, # Requerir al menos un archivo NC para analizar
        help="Ruta al archivo NetCDF (ya sea convertido desde el MDV o generado por el modelo)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        help="Opcional: Archivo donde guardar toda la salida del análisis."
    )
    
    args = parser.parse_args()

    all_output = []

    if args.mdv:
        master_h, field_hs = analyze_mdv_file_pyart(args.mdv)
        if master_h: # Si el análisis del MDV fue exitoso
            all_output.append(f"\n=== METADATOS MDV: {args.mdv} ===\n")
            all_output.append("[Master Header del MDV]\n")
            all_output.append(json.dumps(master_h, indent=4, default=str) + "\n")
            if field_hs:
                for i, f_h in enumerate(field_hs):
                    all_output.append(f"\n[Field Header [{i}] del MDV]\n")
                    all_output.append(json.dumps(f_h, indent=4, default=str) + "\n")
            all_output.append("=== FIN METADATOS MDV ===\n")


    if args.nc:
        ncdump_str = analyze_nc_file_ncdump(args.nc)
        if ncdump_str: # Si el análisis del NC fue exitoso
            all_output.append(f"\n=== METADATOS NetCDF (ncdump -h): {args.nc} ===\n")
            all_output.append(ncdump_str)
            all_output.append("=== FIN METADATOS NetCDF ===\n")

    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                for line in all_output:
                    f.write(line)
            print(f"\nLa salida completa del análisis ha sido guardada en: {args.output_file}")
        except Exception as e:
            print(f"Error al guardar la salida en {args.output_file}: {e}")

if __name__ == "__main__":
    main()