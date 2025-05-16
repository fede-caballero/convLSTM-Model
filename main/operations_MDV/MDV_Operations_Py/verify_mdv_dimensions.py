import pyart
import os

file_path = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/214452.mdv"
print(f"Buscando archivo en: {os.path.abspath(file_path)}")

try:
    radar = pyart.io.read_mdv(file_path)
    print("Archivo cargado con éxito")
    print("Fields disponibles:", radar.fields.keys())
    print("Reflectivity shape:", radar.fields.get("reflectivity", {}).get("data", "No reflectivity").shape)
except NotImplementedError as e:
    print(f"Error de proyección: {e}")
except Exception as e:
    print(f"Otro error: {e}")