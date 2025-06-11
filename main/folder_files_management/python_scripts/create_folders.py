import os
import math
import re

def crear_carpetas_mdv(ruta_base):
    if not os.path.exists(ruta_base):
        print(f"La ruta {ruta_base} no existe")
        return

    # Nombre base de la carpeta madre (ej: "20180206")
    nombre_base = os.path.basename(os.path.normpath(ruta_base))

    # Buscar subcarpetas ya creadas con el patrón base+numero
    subcarpetas_existentes = [
        nombre for nombre in os.listdir(ruta_base)
        if os.path.isdir(os.path.join(ruta_base, nombre)) and re.match(rf"^{re.escape(nombre_base)}\d+$", nombre)
    ]

    # Extraer los números finales y obtener el mayor
    numeros_existentes = [
        int(re.search(rf"{re.escape(nombre_base)}(\d+)", carpeta).group(1))
        for carpeta in subcarpetas_existentes
    ] if subcarpetas_existentes else []

    ultimo_numero = max(numeros_existentes) if numeros_existentes else 0

    # Buscar archivos .nc sueltos (no dentro de subcarpetas)
    archivos_mdv = [
        f for f in os.listdir(ruta_base)
        if f.endswith('.nc') and os.path.isfile(os.path.join(ruta_base, f))
    ]

    total_archivos_sueltos = len(archivos_mdv)

    if total_archivos_sueltos == 0:
        print("No se encontraron archivos .nc sueltos en la carpeta")
        return

    # Calcular la cantidad de carpetas nuevas necesarias
    nuevas_carpetas = math.ceil(total_archivos_sueltos / 17)

    print(f"Se encontraron {total_archivos_sueltos} archivos .nc sueltos")
    print(f"Creando {nuevas_carpetas} nuevas carpetas...")

    for i in range(nuevas_carpetas):
        nuevo_numero = ultimo_numero + i + 1
        nueva_carpeta = f"{nombre_base}{nuevo_numero}"
        ruta_nueva_carpeta = os.path.join(ruta_base, nueva_carpeta)
        os.makedirs(ruta_nueva_carpeta, exist_ok=True)
        print(f"Creada carpeta {nueva_carpeta}")

def main():
    campaña = "2007"
    carpeta_base = "20071201"
    ruta_especifica = f"/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/{campaña}/{carpeta_base}"


    print(f"Procesando carpeta: {ruta_especifica}")
    print("Buscando archivos .nc sueltos...")

    crear_carpetas_mdv(ruta_especifica)
    

    print("Proceso completado!")

if __name__ == "__main__":
    main()

