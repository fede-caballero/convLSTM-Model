# Instalar librerías necesarias (compatibles con A100 y TensorFlow 2.12)
!pip install numpy==1.23.5
!pip install scipy==1.10.1
!pip install arm_pyart
!pip install netCDF4
!pip install tensorflow==2.12.0



# Verificar GPU
!nvidia-smi

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print("GPUs disponibles:", gpus)

if not gpus:
    print("TensorFlow no detecta GPU. Verifica CUDA/cuDNN o contacta al soporte de Vast.ai.")
else:
    print("TensorFlow está usando GPU.")

print("Librerías instaladas y GPU verificada.")


# 1. Procesar y cargar datos .mdv (1000x1000) con depuración y normalización
import os
import glob
import numpy as np
import pyart
import netCDF4
from scipy.ndimage import zoom
import gc

def process_mdv_file(filename):
    radar = pyart.io.read_grid_mdv(filename)
    reflectivity = radar.fields['reflectivity']['data'].filled(np.nan)  # (36, 1000, 1000)
    
    # Depuración
    print(f"Mínimo en {filename}: {np.nanmin(reflectivity)}")
    print(f"Máximo en {filename}: {np.nanmax(reflectivity)}")
    print(f"NaN count: {np.isnan(reflectivity).sum()}")
    
    # Limpiar NaN
    reflectivity = np.nan_to_num(reflectivity, nan=-9999)  # Reemplaza NaN por -9999 (ajusta según tus datos)
    
    # Normalizar reflectividad (dBZ) a [0, 1] o estandarizar
    reflectivity = np.clip(reflectivity, -30, 70)  # Limita valores extremos
    reflectivity = (reflectivity + 30) / 100  # Normaliza a [0, 1] (ajusta si necesitas otro rango)
    
    return reflectivity.astype(np.float32)  # Mantener resolución original (36, 1000, 1000)

# Crear secuencias desde la carpeta MDV
ruta_base = '/home/MDV'
secuencias_finales = []

print(f"Buscando en: {ruta_base}")
if os.path.exists(ruta_base):
    print("Carpeta encontrada. Contenido:", os.listdir(ruta_base))
else:
    print("Error: Carpeta no encontrada. Verifica la ruta.")

for set_carpeta in sorted(os.listdir(ruta_base)):
    ruta_set = os.path.join(ruta_base, set_carpeta)
    print(f"Revisando: {ruta_set}")
    if os.path.isdir(ruta_set):
        mdv_files = sorted(glob.glob(os.path.join(ruta_set, "*.mdv")))
        print(f"Archivos .mdv en {ruta_set}: {mdv_files}")
        if len(mdv_files) == 7:
            secuencias_finales.append(mdv_files)
            print(f"Secuencia añadida desde {ruta_set}")
        else:
            print(f"Advertencia: {ruta_set} no contiene exactamente 7 .mdv (encontrados {len(mdv_files)}).")
    else:
        print(f"Ignorando {ruta_set} (no es una carpeta)")

print(f"Total de secuencias: {len(secuencias_finales)}")

# 2. Generador de datos optimizado
def generador_datos(secuencias, batch_size=1):
    while True:
        for i in range(0, len(secuencias), batch_size):
            batch = secuencias[i:i+batch_size]
            X_batch = []
            y_batch = []
            for seq in batch:
                entradas = [process_mdv_file(f) for f in seq[:6]]
                X_batch.append(np.stack(entradas))  # (6, 36, 1000, 1000)
                del entradas  # Liberar memoria
                salida = process_mdv_file(seq[6])
                y_batch.append(salida)  # (36, 1000, 1000)
            X_batch = np.expand_dims(np.array(X_batch), axis=-1)  # (batch, 6, 36, 1000, 1000, 1)
            y_batch = np.expand_dims(np.array(y_batch), axis=-1)  # (batch, 36, 1000, 1000, 1)
            yield X_batch, y_batch
            del X_batch, y_batch  # Liberar memoria
            gc.collect()  # Forzar recolección de basura

# 3. Modelo ConvLSTM3D (ajustado a 1000x1000) con optimización
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM3D, Conv3D, Dropout
from tensorflow.keras.optimizers import Adam

# Desactivar XLA si hay problemas de ciclos
tf.config.optimizer.set_jit(False)

modelo = Sequential()
modelo.add(ConvLSTM3D(8, (3, 3, 3), padding='same', return_sequences=False,
                      input_shape=(6, 36, 1000, 1000, 1)))
modelo.add(Dropout(0.2))  # Añadir regularización
modelo.add(Conv3D(1, (3, 3, 3), padding='same', activation='linear'))
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)  # Ajustar optimizador
modelo.compile(optimizer=optimizer, loss='mse')
modelo.summary()

# 4. Entrenar
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Optimización para memoria fragmentada

batch_size = 1  # Empezar con 1, subir a 2-4 si la A100 (80 GB) lo soporta
steps_per_epoch = len(secuencias_finales) // batch_size  # 41 steps
modelo.fit(generador_datos(secuencias_finales, batch_size), steps_per_epoch=steps_per_epoch, epochs=10)

# Monitoreo de recursos
!nvidia-smi
!free -h