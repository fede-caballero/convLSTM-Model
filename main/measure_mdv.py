import os
import numpy as np
import struct

class AdvancedMDVReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_header_details(self):
        """
        Lee detalles específicos de la cabecera MDV
        """
        with open(self.file_path, 'rb') as file:
            # Leer cabecera completa
            header = file.read(1024)
            
            # Extraer texto plano de la cabecera
            text_sections = [
                section.decode('utf-8', errors='ignore').strip() 
                for section in header.split(b'\x00') 
                if section and len(section) > 3
            ]
            
            return {
                'raw_text_sections': text_sections,
                'header_length': len(header)
            }

    def explore_data_structure(self):
        """
        Explora diferentes estrategias de lectura de datos
        """
        with open(self.file_path, 'rb') as file:
            # Saltar cabecera
            file.seek(1024)
            
            # Estrategias de lectura
            strategies = [
                ('float32', np.float32),
                ('float64', np.float64),
                ('int16', np.int16),
                ('int32', np.int32)
            ]
            
            results = {}
            for name, dtype in strategies:
                try:
                    # Volver al inicio de los datos
                    file.seek(1024)
                    
                    # Leer datos
                    data = np.fromfile(file, dtype=dtype)
                    
                    results[name] = {
                        'total_elements': data.size,
                        'min': data.min(),
                        'max': data.max(),
                        'mean': data.mean()
                    }
                except Exception as e:
                    results[name] = {'error': str(e)}
            
            return results

def main():
    file_path = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/214452.mdv"
    reader = AdvancedMDVReader(file_path)
    
    print("Detalles de Cabecera:")
    header_details = reader.read_header_details()
    for detail in header_details['raw_text_sections']:
        print(detail)
    
    print("\nEstructuras de Datos Exploradas:")
    data_structures = reader.explore_data_structure()
    for strategy, result in data_structures.items():
        print(f"\nEstrategia: {strategy}")
        print(result)

if __name__ == "__main__":
    main()