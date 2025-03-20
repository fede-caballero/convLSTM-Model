import os
import math

def crear_carpetas_mdv(ruta_base):
    # Verificar si la ruta existe
    if not os.path.exists(ruta_base):
        print(f"La ruta {ruta_base} no existe")
        return
    
    # Obtener la lista de archivos .mdv en la carpeta
    archivos_mdv = [f for f in os.listdir(ruta_base) if f.endswith('.mdv')]
    total_archivos = len(archivos_mdv)
    
    if total_archivos == 0:
        print("No se encontraron archivos .mdv en la carpeta")
        return
    
    # Calcular el número de carpetas necesarias (grupos de 7)
    num_carpetas = math.ceil(total_archivos / 7)
    
    # Obtener el nombre de la carpeta base (solo el último componente)
    nombre_base = os.path.basename(os.path.normpath(ruta_base))
    
    # Crear solo las carpetas
    for i in range(num_carpetas):
        # Crear nombre de la nueva carpeta usando el nombre base + número
        nueva_carpeta = f"{nombre_base}{i+1}"
        ruta_nueva_carpeta = os.path.join(ruta_base, nueva_carpeta)
        
        # Crear la carpeta si no existe
        os.makedirs(ruta_nueva_carpeta, exist_ok=True)
        
        print(f"Creada carpeta {nueva_carpeta}")

def main():
    # Ruta específica de tu carpeta
    campaña = "2016-2017"
    carpeta_base = "20170329"
    ruta_especifica = f"/home/f-caballero/UM/TIF3/Tesis/MDV/Selected/{campaña}/{carpeta_base}"
    
    print(f"Procesando carpeta: {ruta_especifica}")
    print(f"Buscando archivos .mdv...")
    
    crear_carpetas_mdv(ruta_especifica)
    
    print("Proceso completado!")

if __name__ == "__main__":
    main()