import pandas as pd
import os

# Nombre del archivo y rutas
file_name = "Observaciones 23-24 SUR.xlsx"
file_filtered = file_name.replace(".xlsx", "") + "-filtered.xlsx"  # Nombre del archivo filtrado
base_path = r"D:\Fede Facultad\tesis\data_analysis\Observations"  # Ruta base
file_path = os.path.join(base_path, file_name)  # Ruta completa del archivo original

# Crear la carpeta 'filtered' si no existe
filtered_folder = os.path.join(base_path, "filtered")
os.makedirs(filtered_folder, exist_ok=True)  # Crea la carpeta, no da error si ya existe

# Leer el archivo Excel
df = pd.read_excel(file_path)

print("Datos originales: ")
print(df.head())
print("\n")

# Filtrar las filas donde 'Observados' no sea nulo
df_filtered = df.dropna(subset=['Observados'])

print("Datos filtrados: ")
print(df_filtered.head())
print("\n")

# Ruta de salida en la carpeta 'filtered'
out_path = os.path.join(filtered_folder, file_filtered)
df_filtered.to_excel(out_path, index=False)
print(f"Archivo filtrado guardado como: {out_path}")