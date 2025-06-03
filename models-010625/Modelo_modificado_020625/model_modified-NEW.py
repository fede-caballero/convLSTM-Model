import os
import glob
import random
import time
from datetime import datetime, timedelta # Asegúrate de importar timedelta
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from netCDF4 import Dataset as NCDataset # Renombrar para evitar conflicto con la clase Dataset

import matplotlib.pyplot as plt
import torchmetrics # Para métricas adicionales como SSIM
from torch.utils.checkpoint import checkpoint # <--- IMPORTANTE PARA GRADIENT CHECKPOINTING
import torch.amp # <--- IMPORTANTE PARA APIS MODERNAS DE AMP

import torch.optim as optim
import pyproj #


# Configuración del Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración para reproducibilidad y rendimiento
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Necesario si usas múltiples GPUs
        # La siguiente configuración es un balance entre reproducibilidad y rendimiento
        torch.backends.cudnn.deterministic = False # Para reproducibilidad estricta, sería True
        torch.backends.cudnn.benchmark = True    # Para reproducibilidad estricta, sería False. Acelera si los tamaños de entrada no cambian.
    logging.info(f"Semillas configuradas con valor: {seed}")

class RadarDataset(Dataset):
    def __init__(self, data_dir, subdirs_list, seq_len=6, pred_len=1,
                 min_dbz_norm=-30.0, max_dbz_norm=70.0, # Cambiado para usar estos nombres y valores por defecto consistentes
                 expected_shape=(18, 500, 500), variable_name='DBZ'):
        self.data_dir = data_dir
        self.subdirs_list = subdirs_list
        self.seq_len = seq_len
        self.pred_len = pred_len
        # Usar los parámetros pasados para normalización
        self.min_dbz_norm = min_dbz_norm
        self.max_dbz_norm = max_dbz_norm
        self.expected_z, self.expected_h, self.expected_w = expected_shape
        self.variable_name = variable_name
        self.valid_sequences = self._validate_subdirs()
        if not self.valid_sequences:
            logging.error("No se encontraron secuencias válidas. Verifica los datos y la estructura de carpetas.")
            raise ValueError("No se encontraron secuencias válidas.")
        logging.info(f"RadarDataset inicializado con {len(self.valid_sequences)} secuencias válidas.")

    def _validate_subdirs(self):
        valid_sequences = []
        for subdir_name in self.subdirs_list:
            subdir_path = os.path.join(self.data_dir, subdir_name)
            if not os.path.isdir(subdir_path):
                logging.warning(f"Subdirectorio {subdir_name} no encontrado...")
                continue
            if ".ipynb_checkpoints" in subdir_name: #
                logging.debug(f"Omitiendo directorio de checkpoints: {subdir_name}") #
                continue
            files = sorted(glob.glob(os.path.join(subdir_path, "*.nc")))
            if len(files) >= self.seq_len + self.pred_len:
                output_files = files[self.seq_len:self.seq_len + self.pred_len] #
                valid_output = True
                for f_path in output_files: # Cambiado 'f' por 'f_path' para evitar confusión con la f-string
                    try:
                        with NCDataset(f_path, 'r') as nc_file:
                            if self.variable_name not in nc_file.variables: #
                                valid_output = False
                                logging.warning(f"Variable {self.variable_name} no encontrada en archivo de salida {f_path}") #
                                break
                            dbz_var = nc_file.variables[self.variable_name]
                            data_raw = dbz_var[0, ...] if dbz_var.ndim == 4 else dbz_var[...] #
                            
                            # Aplicar scale/offset para obtener valores físicos para el log
                            dbz_physical_for_log = data_raw # Asumir que ya es físico si no hay scale/offset
                            if hasattr(dbz_var, 'scale_factor') and hasattr(dbz_var, 'add_offset'):
                                scale = dbz_var.scale_factor
                                offset = dbz_var.add_offset
                                dbz_physical_for_log = data_raw.astype(np.float32) * scale + offset
                                if hasattr(dbz_var, '_FillValue'):
                                    fill_val_packed = getattr(dbz_var, '_FillValue')
                                    # Convertir fill_val_packed al dtype de data_raw para comparación segura
                                    is_fill = (data_raw == np.array(fill_val_packed, dtype=data_raw.dtype))
                                    dbz_physical_for_log[is_fill] = np.nan

                            # Loguear min/max de valores físicos (ignorando NaN)
                            logging.info(f"Archivo validación {os.path.basename(f_path)}: Físico Min={np.nanmin(dbz_physical_for_log):.2f}, Max={np.nanmax(dbz_physical_for_log):.2f}") #
                    except Exception as e:
                        logging.warning(f"Error leyendo archivo de salida {f_path}: {e}") #
                        valid_output = False
                        break
                if valid_output:
                    valid_sequences.append((files, subdir_name))
            else:
                logging.warning(f"Subdirectorio {subdir_name} tiene {len(files)} archivos, se necesitan {self.seq_len + self.pred_len}...") #
        return valid_sequences

    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        sequence_files, subdir_name = self.valid_sequences[idx]
        input_data_list = []
        output_data_list = []

        all_files_for_sequence = sequence_files[:self.seq_len + self.pred_len]

        for i, file_path in enumerate(all_files_for_sequence):
            try:
                with NCDataset(file_path, 'r') as nc_file:
                    if self.variable_name not in nc_file.variables:
                        logging.warning(f"Variable '{self.variable_name}' no encontrada en {file_path}. Intentando siguiente muestra.")
                        return self.__getitem__((idx + 1) % len(self))

                    dbz_var = nc_file.variables[self.variable_name]

                    # 1. Cargar datos raw (byte/short) SIN convertirlos a float todavía
                    if dbz_var.ndim == 4 and dbz_var.shape[0] == 1:
                        data_raw_original_type = dbz_var[0, ...] 
                    elif dbz_var.ndim == 3:
                        data_raw_original_type = dbz_var[...]
                    else:
                        # ... (manejo de error) ...
                        return self.__getitem__((idx + 1) % len(self))
                    
                    # 2. Crear la máscara de fill_value A PARTIR DE LOS DATOS RAW ORIGINALES
                    is_fill_location = None
                    if hasattr(dbz_var, '_FillValue'):
                        fill_value_packed = dbz_var._FillValue 
                        logging.debug(f"FILE: {os.path.basename(file_path)} - data_raw_original_type.dtype: {data_raw_original_type.dtype}, fill_value_packed: {fill_value_packed} (type: {type(fill_value_packed)})")
                        # Comparación directa con el tipo original de data_raw_original_type
                        is_fill_location = (data_raw_original_type == np.array(fill_value_packed, dtype=data_raw_original_type.dtype))
                        if is_fill_location is not None:
                            logging.info(f"FILE: {os.path.basename(file_path)} - Num True en is_fill_location: {np.sum(is_fill_location)}")
                        else: # Esto no debería ocurrir si hasattr es True
                            logging.info(f"FILE: {os.path.basename(file_path)} - is_fill_location es None.")
                    
                    # 3. Convertir data_raw a float32 para cálculos físicos
                    dbz_physical = data_raw_original_type.astype(np.float32)

                    # 4. Aplicar scale/offset para obtener valores físicos
                    if hasattr(dbz_var, 'scale_factor') and hasattr(dbz_var, 'add_offset'):
                        scale = dbz_var.scale_factor
                        offset = dbz_var.add_offset
                        dbz_physical = dbz_physical * scale + offset # Aplicado a la copia float32

                    # 5. Aplicar la máscara de NaN AHORA a los datos físicos (dbz_physical)
                    if is_fill_location is not None:
                        dbz_physical[is_fill_location] = np.nan # Los NaNs se introducen aquí

                    # dbz_physical ahora tiene valores físicos, con NaN donde había _FillValue

                    # Logging para depuración (NIVEL DEBUG)
                    logging.debug(f"FILE: {os.path.basename(file_path)} - dbz_physical (con NaNs): "
                                 f"NanMin={np.nanmin(dbz_physical):.2f}, NanMax={np.nanmax(dbz_physical):.2f}, "
                                 f"NumNaNs={np.isnan(dbz_physical).sum()}")
                    
                    # 6. Clipping y Normalización, preservando NaN
                    dbz_clipped = np.clip(dbz_physical, self.min_dbz_norm, self.max_dbz_norm) # np.clip propaga NaNs
                    logging.debug(f"FILE: {os.path.basename(file_path)} - dbz_clipped (con NaNs): "
                                 f"NanMin={np.nanmin(dbz_clipped):.2f}, NanMax={np.nanmax(dbz_clipped):.2f}, "
                                 f"NumNaNs={np.isnan(dbz_clipped).sum()}")
                    

                    # Denominador para la normalización
                    range_dbz = self.max_dbz_norm - self.min_dbz_norm
                    if range_dbz == 0: # Evitar división por cero
                        logging.warning(f"Rango de normalización es cero (min_dbz_norm == max_dbz_norm) en {file_path}. Asignando 0 a los valores no NaN.")
                        dbz_normalized = np.where(np.isnan(dbz_clipped), np.nan, 0.0)
                    else:
                        dbz_normalized = (dbz_clipped - self.min_dbz_norm) / range_dbz


                    logging.debug(f"FILE: {os.path.basename(file_path)} - dbz_normalized (esperado [0,1] y NaN): "
                                 f"NanMin={np.nanmin(dbz_normalized):.4f}, NanMax={np.nanmax(dbz_normalized):.4f}, "
                                 f"NanMean={np.nanmean(dbz_normalized):.4f}, NumNaNs={np.isnan(dbz_normalized).sum()}")
                    
                    if dbz_normalized.shape != (self.expected_z, self.expected_h, self.expected_w):
                        logging.warning(f"Forma inesperada {dbz_normalized.shape} en {file_path}. Intentando siguiente muestra.")
                        return self.__getitem__((idx + 1) % len(self))

                    dbz_final = dbz_normalized[..., np.newaxis]

                    if i < self.seq_len:
                        input_data_list.append(dbz_final)
                    else:
                        output_data_list.append(dbz_final)

            except Exception as e:
                logging.error(f"Error procesando archivo {file_path} en __getitem__: {e}") #
                return self.__getitem__((idx + 1) % len(self)) #

        if len(input_data_list) != self.seq_len or len(output_data_list) != self.pred_len: #
            logging.warning(f"No se pudieron cargar suficientes frames para la secuencia {subdir_name}. Intentando siguiente muestra.") #
            return self.__getitem__((idx + 1) % len(self)) #

        input_tensor = np.stack(input_data_list, axis=1)
        output_tensor = np.stack(output_data_list, axis=1)
        logging.info(f"RadarDataset: output_tensor (ANTES de from_numpy): "
                    f"NanMin={np.nanmin(output_tensor):.4f}, NanMax={np.nanmax(output_tensor):.4f}, "
                    f"NanMean={np.nanmean(output_tensor):.4f}, NumNaNs={np.isnan(output_tensor).sum()}, "
                    f"Shape={output_tensor.shape}, Dtype={output_tensor.dtype}")

        # logging.info(f"Output tensor (antes de nan_to_num): Min={np.nanmin(output_tensor):.4f}, Max={np.nanmax(output_tensor):.4f}, Mean={np.nanmean(output_tensor):.4f}")

        # **AJUSTE CLAVE PARA MANEJO DE NaN EN LA PÉRDIDA**
        # Para x (entrada del modelo), los NaN se pueden convertir a 0 (o un valor de imputación)
        x = torch.from_numpy(np.nan_to_num(input_tensor, nan=0.0)).float() #
        # Para y (objetivo), MANTENER los NaN. La función de pérdida se encargará de enmascararlos.
        y = torch.from_numpy(output_tensor).float() # ANTES: np.nan_to_num(output_tensor, nan=0.0)        
        # --- Lógica para devolver Timestamps (DEBES IMPLEMENTAR LA EXTRACCIÓN REAL) ---
        # last_input_file_path = sequence_files[self.seq_len - 1]
        # filename_no_ext = os.path.splitext(os.path.basename(last_input_file_path))[0]
        # last_input_dt_utc_placeholder = datetime.utcnow() # ¡ESTO ES SOLO UN PLACEHOLDER!
        # try:import torch.optim as optim
        #     # Intenta parsear el timestamp del nombre del archivo o del subdirectorio
        #     # Ejemplo: parts = filename_no_ext.split('_'); timestamp_str = parts[0][-8:] + parts[1]
        #     # last_input_dt_utc = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        #     pass # Implementa tu lógica de parseo aquí
        # except Exception as e_time:
        #     logging.warning(f"No se pudo parsear el timestamp de {last_input_file_path} en dataset. Usando placeholder. Error: {e_time}")
        #     # last_input_dt_utc = last_input_dt_utc_placeholder # Mantener el placeholder si falla
    
        # return x, y, last_input_dt_utc_placeholder # Si devuelves timestamp

        logging.debug(f"RadarDataset: output_tensor (con NaNs) stats: "
              f"NanMin={np.nanmin(output_tensor) if np.isnan(output_tensor).any() else np.min(output_tensor):.4f}, "
              f"NanMax={np.nanmax(output_tensor) if np.isnan(output_tensor).any() else np.max(output_tensor):.4f}, "
              f"NanMean={np.nanmean(output_tensor):.4f}, "
              f"NumNaNs={np.isnan(output_tensor).sum()}")
        
        return x, y # Si NO devuelves timestamp por ahora

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # Asegura "same" padding
        self.bias = bias
        # La convolución combina la entrada y el estado oculto anterior.
        # Genera las 4 compuertas/candidatos de una vez.
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim, # Para i, f, o, g
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        # Inicialización de pesos y sesgos
        nn.init.xavier_uniform_(self.conv.weight) # Buena práctica para pesos
        if self.bias:
            nn.init.zeros_(self.conv.bias) # Práctica estándar para sesgos
            # Opcional: Inicializar el sesgo de la compuerta de olvido (forget gate) a un valor pequeño positivo
            # Esto puede ayudar a que la celda recuerde información por defecto al inicio del entrenamiento.
            # La compuerta de olvido es el segundo bloque de canales.
            # self.conv.bias.data[hidden_dim : 2 * hidden_dim].fill_(1.0) 

    def forward(self, input_tensor, cur_state): # input_tensor: (B, C_in, H, W)
        h_cur, c_cur = cur_state # Estados oculto y de celda previos

        combined = torch.cat([input_tensor, h_cur], dim=1) # Concatena entrada y estado oculto
        combined_conv = self.conv(combined) # Aplica la convolución
        
        # Divide en las 4 partes para las compuertas y el candidato a celda
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        
        i = torch.sigmoid(cc_i) # Compuerta de entrada (input gate)
        f = torch.sigmoid(cc_f) # Compuerta de olvido (forget gate)
        o = torch.sigmoid(cc_o) # Compuerta de salida (output gate)
        g = torch.tanh(cc_g)    # Candidato a celda (cell candidate)
        
        c_next = f * c_cur + i * g # Nuevo estado de celda
        h_next = o * torch.tanh(c_next) # Nuevo estado oculto
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        # Inicializa los estados oculto y de celda a ceros
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class ConvLSTM2DLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, return_all_layers=False):
        super(ConvLSTM2DLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.return_all_layers = return_all_layers # Importante para capas apiladas
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias) #

    def forward(self, input_tensor, hidden_state=None): # input_tensor: (B, T_in, C_in, H, W)
        # B: Batch size, T_in: Sequence length, C_in: Input channels, H: Height, W: Width
        b, seq_len, _, h, w = input_tensor.size() # _ es C_in
        device = input_tensor.device # Obtener el dispositivo del tensor de entrada
        
        # Inicializar estado oculto si no se proporciona
        if hidden_state is None:
            hidden_state = self.cell.init_hidden(b, (h, w), device) #

        layer_output_list = []
        h_cur, c_cur = hidden_state # Desempaquetar estados actuales

        # Iterar a través de la secuencia de tiempo
        for t in range(seq_len):
            # input_tensor[:, t, :, :, :] tiene forma (B, C_in, H, W)
            h_cur, c_cur = self.cell(input_tensor=input_tensor[:, t, :, :, :], cur_state=[h_cur, c_cur]) #
            layer_output_list.append(h_cur) # Guardar el estado oculto de este paso de tiempo

        if self.return_all_layers:
            # Apilar todos los estados ocultos a lo largo de la dimensión de tiempo
            layer_output = torch.stack(layer_output_list, dim=1) # Forma: (B, T_in, C_hidden, H, W)
        else:
            # Devolver solo el último estado oculto, manteniendo una dimensión de tiempo de tamaño 1
            layer_output = h_cur.unsqueeze(1) # Forma: (B, 1, C_hidden, H, W)

        return layer_output, (h_cur, c_cur) # Devuelve la salida de la capa y el último estado (oculto y de celda)

class ConvLSTM3D_Enhanced(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[32, 64], kernel_sizes=[(3,3), (3,3)],
                 num_layers=2, pred_steps=1, use_layer_norm=True, use_residual=False, # use_residual no se implementa actualmente
                 img_height=500, img_width=500,
                 # Añadir para gradient checkpointing
                 use_gradient_checkpointing=False): # Nuevo parámetro
        super(ConvLSTM3D_Enhanced, self).__init__()
        # Asegurar que hidden_dims y kernel_sizes sean listas de la longitud correcta
        if isinstance(hidden_dims, int): hidden_dims = [hidden_dims] * num_layers # [cite: 010625_seq_size_7_2010_2015.ipynb]
        if isinstance(kernel_sizes, tuple): kernel_sizes = [kernel_sizes] * num_layers # [cite: 010625_seq_size_7_2010_2015.ipynb]
        assert len(hidden_dims) == num_layers and len(kernel_sizes) == num_layers, \
               "hidden_dims y kernel_sizes deben tener una longitud igual a num_layers"

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.pred_steps = pred_steps # Número de pasos de tiempo a predecir
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual # Actualmente no implementado en el forward
        self.img_height = img_height
        self.img_width = img_width
        self.use_gradient_checkpointing = use_gradient_checkpointing # Guardar para usar en forward

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        current_dim = input_dim
        for i in range(num_layers):
            self.layers.append(
                ConvLSTM2DLayer(input_dim=current_dim, hidden_dim=hidden_dims[i],
                                kernel_size=kernel_sizes[i], bias=True,
                                # La última capa ConvLSTM2D devuelve solo el último estado oculto (con T=1)
                                # Las capas intermedias devuelven toda la secuencia de estados ocultos
                                return_all_layers=True if i < num_layers - 1 else False) # [cite: 010625_seq_size_7_2010_2015.ipynb]
            )
            if use_layer_norm:
                # LayerNorm se aplica a (C, H, W)
                self.layer_norms.append(nn.LayerNorm([hidden_dims[i], img_height, img_width])) # [cite: 010625_seq_size_7_2010_2015.ipynb]
            current_dim = hidden_dims[i]

        # Capa convolucional de salida para mapear el último estado oculto a la predicción
        # El kernel_size=(1, 3, 3) opera sobre la dimensión temporal (que es 1 para la salida de la última ConvLSTM2DLayer)
        # y espacialmente.
        self.output_conv = nn.Conv3d(in_channels=hidden_dims[-1],
                                     out_channels=input_dim * pred_steps, # Predice `pred_steps` para cada canal de entrada
                                     kernel_size=(1, 3, 3), padding=(0, 1, 1)) # [cite: 010625_seq_size_7_2010_2015.ipynb]
        self.sigmoid = nn.Sigmoid() # Para asegurar salida en [0,1] (normalizada) [cite: 010625_seq_size_7_2010_2015.ipynb]

        # Inicialización de pesos
        nn.init.xavier_uniform_(self.output_conv.weight) # [cite: 010625_seq_size_7_2010_2015.ipynb]
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias) # [cite: 010625_seq_size_7_2010_2015.ipynb]

        logging.info(f"Modelo ConvLSTM3D_Enhanced creado: {num_layers} capas, Hidden dims: {hidden_dims}, LayerNorm: {use_layer_norm}, PredSteps: {pred_steps}") # [cite: 010625_seq_size_7_2010_2015.ipynb]

    def forward(self, x_volumetric):  # Espera (Z, B, T_in, H, W, C_in)
        num_z_levels, b, seq_len, h, w, c_in = x_volumetric.shape # [cite: 010625_seq_size_7_2010_2015.ipynb]
        all_level_predictions = []

        # Procesar cada nivel Z de forma independiente
        for z_idx in range(num_z_levels):
            x_level = x_volumetric[z_idx, ...]  # Forma: (B, T_in, H, W, C_in) [cite: 010625_seq_size_7_2010_2015.ipynb]
            # Permutar para ConvLSTM2DLayer: (B, T_in, C_in, H, W)
            x_level_permuted = x_level.permute(0, 1, 4, 2, 3)  # [cite: 010625_seq_size_7_2010_2015.ipynb]
            current_input = x_level_permuted

            # Estados ocultos para este nivel Z (no se comparten entre niveles Z)
            hidden_states_for_level = [None] * self.num_layers # [cite: 010625_seq_size_7_2010_2015.ipynb]

            for i in range(self.num_layers):
                # Aplicar Gradient Checkpointing si está habilitado y en modo entrenamiento
                if self.use_gradient_checkpointing and self.training:
                    # Es importante que los argumentos que no requieren gradiente (como hidden_state si es None)
                    # no causen problemas con checkpoint. PyTorch maneja esto bien.
                    # `use_reentrant=False` es recomendado para versiones más nuevas de PyTorch.
                    layer_output, hidden_state = torch.utils.checkpoint.checkpoint(
                        self.layers[i], current_input, hidden_states_for_level[i],
                        use_reentrant=False 
                    )
                else:
                    layer_output, hidden_state = self.layers[i](current_input, hidden_states_for_level[i]) # [cite: 010625_seq_size_7_2010_2015.ipynb]
                
                hidden_states_for_level[i] = hidden_state # [cite: 010625_seq_size_7_2010_2015.ipynb]

                if self.use_layer_norm and self.layer_norms:
                    # LayerNorm espera (N, C, H, W) o (N, ..., C)
                    # La salida de ConvLSTM2DLayer es (B, T, C_hidden, H, W)
                    B_ln, T_ln, C_ln, H_ln, W_ln = layer_output.shape # [cite: 010625_seq_size_7_2010_2015.ipynb]
                    # Reshape para LayerNorm: (B*T, C_hidden, H, W)
                    output_reshaped_for_ln = layer_output.contiguous().view(B_ln * T_ln, C_ln, H_ln, W_ln) # [cite: 010625_seq_size_7_2010_2015.ipynb]
                    normalized_output = self.layer_norms[i](output_reshaped_for_ln) # [cite: 010625_seq_size_7_2010_2015.ipynb]
                    layer_output = normalized_output.view(B_ln, T_ln, C_ln, H_ln, W_ln) # [cite: 010625_seq_size_7_2010_2015.ipynb]
                
                # Aquí se podría añadir la conexión residual si self.use_residual es True
                # if self.use_residual and current_input.shape == layer_output.shape:
                #    current_input = current_input + layer_output 
                # else: (manejar si las dimensiones no coinciden, ej. con una conv 1x1 en current_input)
                current_input = layer_output # [cite: 010625_seq_size_7_2010_2015.ipynb]

            # La salida de la última capa ConvLSTM2DLayer (si return_all_layers=False) es (B, 1, C_hidden, H, W)
            # Permutar para Conv3D: (B, C_hidden, T=1, H, W)
            output_for_conv3d = current_input.permute(0, 2, 1, 3, 4) # [cite: 010625_seq_size_7_2010_2015.ipynb]
            raw_conv_output = self.output_conv(output_for_conv3d) # Salida: (B, C_out*T_pred, 1, H, W) [cite: 010625_seq_size_7_2010_2015.ipynb]
            
            # Eliminar la dimensión temporal espuria (T=1) de la salida de Conv3D
            prediction_features = raw_conv_output.squeeze(2) # Forma: (B, C_out*T_pred, H, W) [cite: 010625_seq_size_7_2010_2015.ipynb]
            
            # Reshape para separar pred_steps y canales de salida
            # C_out es self.input_dim (ya que predecimos la misma variable de entrada)
            level_prediction = prediction_features.view(b, self.pred_steps, self.input_dim, h, w) # [cite: 010625_seq_size_7_2010_2015.ipynb]
            
            # Permutar a (B, T_pred, H, W, C_out) para consistencia con la forma de y_true
            level_prediction = level_prediction.permute(0, 1, 3, 4, 2) # [cite: 010625_seq_size_7_2010_2015.ipynb]
            level_prediction = self.sigmoid(level_prediction) # Aplicar Sigmoid para normalizar a [0,1] [cite: 010625_seq_size_7_2010_2015.ipynb]
            all_level_predictions.append(level_prediction)

        # Apilar las predicciones de todos los niveles Z: (Z, B, T_pred, H, W, C_out)
        predictions_volumetric = torch.stack(all_level_predictions, dim=0) # [cite: 010625_seq_size_7_2010_2015.ipynb]
        return predictions_volumetric

class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, kernel_size_for_metric=7):
        super(SSIMLoss, self).__init__()
        # torchmetrics.SSIM calcula el Structural Similarity Index Measure.
        # data_range: Rango de los datos de entrada (tus predicciones y objetivos normalizados están en [0,1]).
        # kernel_size: Tamaño del kernel Gaussiano para la ventana.
        # reduction: 'elementwise_mean' promedia los valores SSIM de todas las imágenes en el lote.
        self.ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(
            data_range=data_range, #
            kernel_size=kernel_size_for_metric, #
            reduction='elementwise_mean' # 
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) #

    def forward(self, predictions, targets): 
        # Renombrado img1 a predictions, img2 a targets para mayor claridad.
        # Ambas se esperan con forma (Z, B, T_pred, H, W, C)
        # 'targets' puede contener NaNs (gracias a los cambios en RadarDataset).
        # 'predictions' debería ser la salida del modelo (ya normalizada por Sigmoid, sin NaNs).

        num_z, batch_s, pred_t, height, width, channels = predictions.shape #

        # SSIM se aplica a imágenes 2D (o slices con canales).
        # Aplanamos las dimensiones Z, B, T_pred en una sola dimensión de "lote_aplanado"
        # y permutamos para que la forma sea (lote_aplanado, Canales, H, W), que es lo que espera torchmetrics.
        
        # (Z, B, T, C, H, W) -> (Z*B*T, C, H, W)
        predictions_reshaped = predictions.permute(0, 1, 2, 5, 3, 4).contiguous().view(-1, channels, height, width) #
        targets_reshaped = targets.permute(0, 1, 2, 5, 3, 4).contiguous().view(-1, channels, height, width) #

        # **AJUSTE CLAVE PARA MANEJAR NaN EN TARGETS PARA SSIM**
        # torchmetrics.SSIM no maneja NaNs directamente. Si hay NaNs, el resultado puede ser NaN.
        # Para esta pérdida SSIM, una estrategia es imputar los NaNs en los targets.
        # Usamos 0.0 para la imputación, ya que en el espacio normalizado [0,1], 0.0 
        # corresponde al valor físico mínimo (ej. -30.5 dBZ o el self.min_dbz_norm).
        # Las predicciones del modelo (predictions_reshaped) no deberían tener NaNs si vienen de un Sigmoid.
        
        targets_for_ssim = torch.nan_to_num(targets_reshaped, nan=0.0) 
        # Opcionalmente, si crees que las predicciones podrían tener NaN por alguna razón extrema:
        # predictions_for_ssim = torch.nan_to_num(predictions_reshaped, nan=0.0)
        # Pero generalmente, solo imputar targets es suficiente si la salida del modelo es 'limpia'.

        # Calcular SSIM. Como reduction='elementwise_mean', esto ya devuelve un scalar tensor con el promedio.
        ssim_val_mean = self.ssim_metric(predictions_reshaped, targets_for_ssim) #
        
        # La línea original `ssim_val_mean = ssim_val_elementwise.mean()` es redundante aquí
        # si `reduction` en el constructor ya es 'elementwise_mean'.

        logging.info(f"SSIM Mean (targets con NaNs imputados a 0.0): {ssim_val_mean.item():.4f}") #
        
        # SSIM es una medida de similitud (mayor es mejor, rango [-1, 1] o [0, 1] para imágenes no negativas).
        # Para usarlo como pérdida (menor es mejor), se usa 1 - SSIM.
        return 1.0 - ssim_val_mean #

def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #
    logging.info(f"Usando dispositivo: {device}") #
    model.to(device) #

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-5)) #
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.get('lr_patience', 3), verbose=True) #

    # MSELoss con reducción 'mean' es el estándar. Lo aplicaremos a tensores ya filtrados.
    criterion_mse = nn.MSELoss().to(device) #
    criterion_ssim = None #
    ssim_loss_weight = 0.0 #
    mse_loss_weight = 1.0 #

    if config.get('use_ssim_loss', False): #
        try:
            criterion_ssim = SSIMLoss( #
                data_range=1.0, 
                kernel_size_for_metric=config.get('ssim_kernel_size', 7)
            ).to(device)
            ssim_loss_weight = config.get('ssim_loss_weight', 0.3) #
            mse_loss_weight = 1.0 - ssim_loss_weight #
            logging.info(f"Usando SSIM loss con peso {ssim_loss_weight} y MSE con peso {mse_loss_weight}") #
        except Exception as e:
            logging.error(f"Error al inicializar SSIMLoss: {e}. Se usará solo MSE.") #
            criterion_ssim = None 
            ssim_loss_weight = 0.0
            mse_loss_weight = 1.0

    scaler = torch.amp.GradScaler(enabled=config['use_amp']) #

    best_val_loss = float('inf') #
    train_losses, val_losses = [], [] #
    accumulation_steps = config.get('accumulation_steps', 1) #

    logging.info(f"Iniciando entrenamiento: {config['epochs']} épocas, LR: {config['learning_rate']}, Batch (efectivo): {config['batch_size'] * accumulation_steps}") #

    for epoch in range(config['epochs']): #
        if torch.cuda.is_available(): torch.cuda.empty_cache() #
        epoch_start_time = time.time() #
        model.train() #
        running_train_loss = 0.0 #
        optimizer.zero_grad() #
        
        if torch.cuda.is_available(): #
            logging.info(f"Inicio Época {epoch+1} - Memoria GPU Asignada: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB, Reservada: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB") # [cite: 7]

        for batch_idx, (x, y) in enumerate(train_loader): # y ahora contiene NaNs
            x = x.to(device) #
            y = y.to(device) # y tiene NaNs

            if x.dim() == 6 and y.dim() == 6:  # (B,Z,T_in,H,W,C) y (B,Z,T_out,H,W,C)
                x = x.permute(1, 0, 2, 3, 4, 5) # (Z, B, T_in, H, W, C)
                y = y.permute(1, 0, 2, 3, 4, 5) # (Z, B, T_out, H, W, C)
            else:
                logging.error(f"Formas inesperadas para x o y antes de la permutación: x={x.shape}, y={y.shape}") #
                continue

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config['use_amp']): #
                predictions = model(x) #

                if predictions.shape != y.shape: #
                    logging.error(f"Discrepancia de formas entre predicción {predictions.shape} y objetivo {y.shape}") #
                    continue

                # **AJUSTE CLAVE PARA MSE CON NaN EN Y**
                # 1. Crear una máscara para los píxeles válidos (no-NaN) en el tensor objetivo 'y'
                valid_pixel_mask = ~torch.isnan(y)

                # 2. Calcular MSE solo sobre los píxeles válidos
                if valid_pixel_mask.sum() > 0:  # Asegurarse de que hay al menos un píxel válido
                    # Aplicar la máscara tanto a las predicciones como a los objetivos
                    # Esto crea tensores 1D con solo los elementos válidos
                    loss_mse_val = criterion_mse(predictions[valid_pixel_mask], y[valid_pixel_mask])
                else:
                    # Si (muy improbablemente) todos los píxeles en 'y' son NaN para este lote
                    loss_mse_val = torch.tensor(0.0, device=device, requires_grad=predictions.requires_grad if predictions.requires_grad else False)
                current_loss = loss_mse_val #

                if criterion_ssim is not None: #
                    # SSIMLoss ahora maneja NaNs en 'y' internamente (imputándolos)
                    loss_ssim_component = criterion_ssim(predictions, y) #
                    current_loss = mse_loss_weight * loss_mse_val + ssim_loss_weight * loss_ssim_component #

                if batch_idx == 0 and epoch == 0: #
                    if y.numel() > 0:
                        is_nan_present_y = torch.isnan(y).any()
                        log_nanmin_y, log_nanmax_y, log_nanmean_y = float('nan'), float('nan'), float('nan') # Defaults
                        if is_nan_present_y:
                            valid_y_elements = y[~torch.isnan(y)]
                            if valid_y_elements.numel() > 0:
                                log_nanmin_y = valid_y_elements.min().item()
                                log_nanmax_y = valid_y_elements.max().item()
                                log_nanmean_y = valid_y_elements.mean().item()
                        elif y.numel() > 0: # No NaNs, but tensor is not empty
                            log_nanmin_y = y.min().item()
                            log_nanmax_y = y.max().item()
                            log_nanmean_y = y.mean().item()
                        
                        logging.info(f"  Objetivos y (normalizados, con NaNs): "
                                    f"MinVal={log_nanmin_y:.4f}, MaxVal={log_nanmax_y:.4f}, "
                                    f"MeanVal={log_nanmean_y:.4f}, NumNaNs={torch.isnan(y).sum()}")
                    else:
                        logging.info("  Objetivos y (normalizados, con NaNs): Tensor vacío.")
                    logging.info(f"  MSE (masked): {loss_mse_val.item():.6f}")
                    if criterion_ssim is not None:
                        logging.info(f"  SSIM (1-SSIM, imputed targets): {loss_ssim_component.item():.6f}")


                loss_to_accumulate = current_loss / accumulation_steps #

            scaler.scale(loss_to_accumulate).backward() #

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader): #
                if config.get('clip_grad_norm', None): #
                    scaler.unscale_(optimizer) 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['clip_grad_norm']) #
                scaler.step(optimizer) #
                scaler.update() #
                optimizer.zero_grad() #

            running_train_loss += current_loss.item() #

            if (batch_idx + 1) % config.get('log_interval', 1) == 0: #
                logging.info(f"Época {epoch+1}/{config['epochs']} [{batch_idx+1}/{len(train_loader)}] - Pérdida (batch): {current_loss.item():.6f}") #

        avg_train_loss = running_train_loss / len(train_loader) #
        train_losses.append(avg_train_loss) #

        # Validación
        if val_loader and len(val_loader) > 0: #
            model.eval() #
            running_val_loss = 0.0 #
            with torch.no_grad(): #
                for x_val, y_val in val_loader: # y_val ahora contiene NaNs
                    x_val = x_val.to(device) #
                    y_val = y_val.to(device) #
                    if x_val.dim() == 6 and y_val.dim() == 6: #
                        x_val = x_val.permute(1, 0, 2, 3, 4, 5) #
                        y_val = y_val.permute(1, 0, 2, 3, 4, 5) #
                    else:
                        logging.error(f"Formas inesperadas (val) x={x_val.shape}, y={y_val.shape}") #
                        continue

                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config['use_amp']): #
                        predictions_val = model(x_val) #
                        if predictions_val.shape != y_val.shape: #
                            logging.error(f"Discrepancia de formas (val) entre predicción {predictions_val.shape} y objetivo {y_val.shape}") #
                            continue
                        
                        # **AJUSTE CLAVE PARA MSE CON NaN EN Y_VAL**
                        valid_pixel_mask_val = ~torch.isnan(y_val)
                        if valid_pixel_mask_val.sum() > 0:
                            val_loss_mse_val = criterion_mse(predictions_val[valid_pixel_mask_val], y_val[valid_pixel_mask_val])
                        else:
                            val_loss_mse_val = torch.tensor(0.0, device=device)

                        current_val_loss = val_loss_mse_val #
                        if criterion_ssim is not None: #
                            val_loss_ssim_component = criterion_ssim(predictions_val, y_val) #
                            current_val_loss = mse_loss_weight * val_loss_mse_val + ssim_loss_weight * val_loss_ssim_component #
                    running_val_loss += current_val_loss.item() #

            if len(val_loader) > 0: #
                avg_val_loss = running_val_loss / len(val_loader) #
                val_losses.append(avg_val_loss) #
                scheduler.step(avg_val_loss) #
                epoch_duration = time.time() - epoch_start_time #
                logging.info(f"Época {epoch+1} completada en {epoch_duration:.2f}s. Pérdida (train): {avg_train_loss:.6f}, Pérdida (val): {avg_val_loss:.6f}") # [cite: 15]

                if avg_val_loss < best_val_loss: #
                    best_val_loss = avg_val_loss #
                    torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), #
                                'optimizer_state_dict': optimizer.state_dict(), 'loss': best_val_loss}, #
                               os.path.join(config['model_save_dir'], "best_convlstm_model.pth")) #
                    logging.info(f"Mejor modelo guardado (Pérdida Val: {best_val_loss:.6f})") # [cite: 15]
            else: 
                epoch_duration = time.time() - epoch_start_time #
                logging.info(f"Época {epoch+1} completada en {epoch_duration:.2f}s. Pérdida (train): {avg_train_loss:.6f} (Dataset de validación vacío, no se calculó pérdida de validación)") #
        else: 
            epoch_duration = time.time() - epoch_start_time #
            logging.info(f"Época {epoch+1} completada en {epoch_duration:.2f}s. Pérdida (train): {avg_train_loss:.6f} (No hay val_loader)") #

        if (epoch + 1) % config.get('checkpoint_interval', 1) == 0: #
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), #
                        'optimizer_state_dict': optimizer.state_dict(), 'train_losses': train_losses, #
                        'val_losses': val_losses if (val_loader and len(val_loader) > 0) else []}, #
                       os.path.join(config['model_save_dir'], f"checkpoint_epoch_{epoch+1}.pth")) #
            logging.info(f"Checkpoint guardado en la época {epoch+1}") # [cite: 15]

    logging.info("Entrenamiento finalizado.") # [cite: 23]
    if train_loader and len(train_losses) > 0: #
        plt.figure(figsize=(10, 5)) #
        plt.plot(train_losses, label='Pérdida Entrenamiento') #
        if val_loader and len(val_losses) > 0: #
            plt.plot(val_losses, label='Pérdida Validación') #
        plt.xlabel('Épocas') #
        plt.ylabel('Pérdida') #
        plt.legend() #
        plt.title('Curvas de Pérdida del Entrenamiento') #
        plt.savefig(os.path.join(config['model_save_dir'], "loss_curves.png")) #
        plt.close() #
        logging.info(f"Curvas de pérdida guardadas en {os.path.join(config['model_save_dir'], 'loss_curves.png')}") # [cite: 24]

    return model, {'train_losses': train_losses, 'val_losses': val_losses if (val_loader and len(val_loader) > 0) else []} #

def generate_prediction_netcdf(model, data_loader, config, device, num_samples=1):
    model.to(device) #
    model.float() #
    model.eval() #

    output_dir = config['predictions_output_dir'] #

    # **AJUSTE 1: Usar los mismos nombres de config para el rango de (de)normalización**
    # Estos deben ser los mismos valores que se usan en RadarDataset.
    # Asegúrate de que tu dict 'config' tenga 'norm_min_dbz' y 'norm_max_dbz'.
    min_dbz_norm_config = config.get('norm_min_dbz', -30.0) # Valor por defecto si no está en config
    max_dbz_norm_config = config.get('norm_max_dbz', 70.0)  # Valor por defecto si no está en config
    
    output_scale_factor = np.float32(config.get('output_nc_scale_factor', 0.5)) #
    output_add_offset = np.float32(config.get('output_nc_add_offset', 33.5)) #
    output_fill_value_byte = np.int8(config.get('output_nc_fill_value_byte', -128)) #
    
    # Umbral físico para considerar una predicción como "nula" o "sin reflectividad"
    # Este valor debería ser cercano a min_dbz_norm_config.
    # Por ejemplo, si min_dbz_norm_config es -30.5, un umbral de -28 podría ser adecuado.
    # O, si el modelo predice 0.0 dBZ para el fondo, un umbral de 0.1 dBZ.
    physical_null_threshold = config.get('physical_null_threshold', min_dbz_norm_config + 2.0) # Ejemplo: -30.5 + 2.0 = -28.5 dBZ
    logging.info(f"Usando umbral físico para nulos: < {physical_null_threshold:.2f} dBZ")

    # Nombres de dimensiones y variables (ya bien parametrizados desde config)
    time_dim_name = config.get('time_dim_name', 'time') #
    # ... (resto de los nombres de variables y dimensiones) ...
    dbz_var_name = config.get('dbz_variable_name_pred_nc', 'DBZ') #
    grid_mapping_var_name = config.get('projection_variable_name', "grid_mapping_0") #

    proj_origin_lon = config.get('sensor_longitude', -68.0169982910156) #
    proj_origin_lat = config.get('sensor_latitude', -34.6479988098145) #
    earth_radius_m = config.get('earth_radius_m', 6378137) #
    
    if not os.path.exists(output_dir): os.makedirs(output_dir) #

    num_z, num_y, num_x = config['expected_shape'] #

    z_coord_values = np.arange(config['grid_minz_km'], config['grid_minz_km'] + num_z * config['grid_dz_km'], config['grid_dz_km'], dtype=np.float32)[:num_z] #
    x_coord_values = np.arange(config['grid_minx_km'], config['grid_minx_km'] + num_x * config['grid_dx_km'], config['grid_dx_km'], dtype=np.float32)[:num_x] #
    y_coord_values = np.arange(config['grid_miny_km'], config['grid_miny_km'] + num_y * config['grid_dy_km'], config['grid_dy_km'], dtype=np.float32)[:num_y] #
    
    proj = pyproj.Proj(proj="aeqd", lon_0=proj_origin_lon, lat_0=proj_origin_lat, R=earth_radius_m) #
    x_grid, y_grid = np.meshgrid(x_coord_values, y_coord_values) #
    lon0, lat0 = proj(x_grid * 1000, y_grid * 1000, inverse=True) # pyproj espera metros para x,y en proyecciones
                                                                 # si x_coord_values están en km.

    sample_count = 0
    with torch.no_grad(): #
        for batch_idx, data_batch in enumerate(data_loader): #
            if sample_count >= num_samples: break #
            
            # Asumimos que data_batch puede contener (x, y_con_nans) o (x, y_con_nans, timestamp_del_ultimo_input)
            # Por ahora, la lógica del timestamp sigue siendo placeholder si no se modifica RadarDataset
            if len(data_batch) == 2:
                x_input_volume, _ = data_batch # y_true_volume no se usa aquí para la predicción
                # Placeholder para timestamp si no viene del DataLoader
                last_input_dt_actual = datetime.utcnow() - timedelta(minutes=config['seq_len'] * config['prediction_interval_minutes'])
            elif len(data_batch) == 3: # Suponiendo que el tercer elemento es el timestamp
                x_input_volume, _, last_input_dt_actual = data_batch
                if not isinstance(last_input_dt_actual, datetime): # Tomar el primero del batch si es una lista/tensor
                    last_input_dt_actual = last_input_dt_actual[0] 
            else:
                logging.error("Formato de data_batch inesperado.")
                continue

            x_permuted = x_input_volume.permute(1, 0, 2, 3, 4, 5) #
            x_to_model = x_permuted.to(device) #
            
            current_x_to_model = x_to_model[:, 0:1, ...] # Procesar solo el primer ítem del batch si batch_size > 1

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config['use_amp']): #
                prediction_norm_all_steps = model(current_x_to_model) #
            
            # Asumimos pred_steps = 1 para la predicción guardada
            pred_data_np = prediction_norm_all_steps[:, 0, 0, :, :, 0].cpu().numpy()  # (Z, H, W)
            
            # Desnormalizar usando el rango CONSISTENTE
            pred_data_desnorm_float = np.where(
                np.isnan(pred_data_np), 
                np.nan,
                pred_data_np * (max_dbz_norm_config - min_dbz_norm_config) + min_dbz_norm_config
            ) #
            
            # **AJUSTE 2: Clipping físico explícito (Opcional pero recomendado para robustez)**
            pred_data_desnorm_float = np.clip(pred_data_desnorm_float, min_dbz_norm_config, max_dbz_norm_config)
            
            logging.info(f"Predicción Física Desnormalizada y Clipeada (muestra {sample_count}): Min={np.nanmin(pred_data_desnorm_float):.2f}, Max={np.nanmax(pred_data_desnorm_float):.2f}, Mean={np.nanmean(pred_data_desnorm_float):.2f}") #
            
            # **AJUSTE 3: Marcar áreas sin reflectividad usando el umbral físico configurado**
            pred_data_for_packing = np.where(
                pred_data_desnorm_float < physical_null_threshold, # Usar el umbral configurable
                np.nan, 
                pred_data_desnorm_float
            ) #
            
            pred_data_byte = np.where(
                np.isnan(pred_data_for_packing), 
                output_fill_value_byte,
                np.clip(((pred_data_for_packing - output_add_offset) / output_scale_factor), -127, 127).round().astype(np.int8)
            ) #
            pred_data_final_for_nc = np.expand_dims(pred_data_byte, axis=0) #

            # Lógica de Timestamps (usando last_input_dt_actual)
            # Asegúrate que last_input_dt_actual sea un objeto datetime sin tzinfo para estos cálculos
            if last_input_dt_actual.tzinfo is not None:
                 last_input_dt_actual = last_input_dt_actual.replace(tzinfo=None)

            forecast_lead_seconds = (0 + 1) * config['prediction_interval_minutes'] * 60 #
            actual_forecast_datetime_utc = last_input_dt_actual + timedelta(seconds=forecast_lead_seconds) #

            epoch_time = datetime(1970, 1, 1, 0, 0, 0) #
            time_value_seconds = (actual_forecast_datetime_utc - epoch_time).total_seconds() #
            
            # Estos start/stop time para el NetCDF deberían reflejar el intervalo del dato PREDICHO.
            # Por ejemplo, si la predicción es instantánea, start y stop podrían ser iguales al time_value.
            # Si representa un intervalo, ajústalo. El original parece usar un intervalo de scan.
            # Para una predicción a t+3min, podríamos decir que el intervalo es corto.
            time_begin_calc_seconds = time_value_seconds # O un poco antes si representa un intervalo
            time_end_calc_seconds = time_value_seconds   #
            
            file_timestamp_str = actual_forecast_datetime_utc.strftime("%Y%m%d_%H%M%S") #
            output_filename = os.path.join(output_dir, f"pred_{dbz_var_name}_{file_timestamp_str}_sample{sample_count}.nc") #

            with NCDataset(output_filename, 'w', format='NETCDF3_CLASSIC') as ncfile: #
                # --- ESCRITURA DE METADATOS Y VARIABLES NetCDF ---
                # (Esta parte es extensa y parece mayormente correcta, solo ajustaré los atributos min/max de la variable DBZ)
                # ... (copiar toda la sección de creación de dimensiones y variables de tu código original) ...
                ncfile.Conventions = "CF-1.6" #
                ncfile.title = f"{config.get('radar_name', 'SAN_RAFAEL')} - Forecast t+{config['prediction_interval_minutes']}min" #
                # ... (otros atributos globales) ...
                ncfile.references = f"Tesis de {config.get('author_name', 'Federico Caballero')}, {config.get('author_institution', 'Universidad de Mendoza')}" #

                ncfile.createDimension(time_dim_name, None) #
                ncfile.createDimension(config.get('bounds_dim_name', 'bounds'), 2) #
                ncfile.createDimension(config.get('x_dim_name', 'x0'), num_x) #
                ncfile.createDimension(config.get('y_dim_name', 'y0'), num_y) #
                ncfile.createDimension(config.get('z_dim_name', 'z0'), num_z) #

                time_v = ncfile.createVariable(config.get('time_var_name', 'time'), 'f8', (time_dim_name,)) #
                time_v.standard_name = "time" #
                time_v.long_name = "Data time" #
                time_v.units = "seconds since 1970-01-01T00:00:00Z" #
                time_v.axis = "T" #
                time_v.bounds = config.get('time_bounds_var_name', 'time_bounds') #
                time_v.comment = actual_forecast_datetime_utc.strftime("%Y-%m-%dT%H:%M:%SZ") #
                time_v[:] = [time_value_seconds] #

                time_bnds_v = ncfile.createVariable(config.get('time_bounds_var_name', 'time_bounds'), 'f8', (time_dim_name, config.get('bounds_dim_name', 'bounds'))) #
                time_bnds_v.comment = "Time bounds for data interval" #
                time_bnds_v.units = "seconds since 1970-01-01T00:00:00Z" #
                time_bnds_v[:] = [[time_begin_calc_seconds, time_end_calc_seconds]] #
                
                # ... (definición de start_time_v, stop_time_v, x_v, y_v, z_v, lat0_v, lon0_v, gm_v como en tu código)
                # (Asegúrate que estos también se llenen correctamente)
                start_time_v = ncfile.createVariable(config.get('start_time_var_name', 'start_time'), 'f8', (time_dim_name,)) #
                start_time_v.long_name = "start_time" #
                start_time_v.units = "seconds since 1970-01-01T00:00:00Z" #
                start_time_v.comment = datetime.fromtimestamp(time_begin_calc_seconds).strftime("%Y-%m-%dT%H:%M:%SZ") #
                start_time_v[:] = [time_begin_calc_seconds] #

                stop_time_v = ncfile.createVariable(config.get('stop_time_var_name', 'stop_time'), 'f8', (time_dim_name,)) #
                stop_time_v.long_name = "stop_time" #
                stop_time_v.units = "seconds since 1970-01-01T00:00:00Z" #
                stop_time_v.comment = datetime.fromtimestamp(time_end_calc_seconds).strftime("%Y-%m-%dT%H:%M:%SZ") #
                stop_time_v[:] = [time_end_calc_seconds] #

                x_v = ncfile.createVariable(config.get('x_coord_var_name', 'x0'), 'f4', (config.get('x_dim_name', 'x0'),)) #
                x_v.standard_name = "projection_x_coordinate" #
                x_v.units = "km" #
                x_v.axis = "X" #
                x_v[:] = x_coord_values #
                
                y_v = ncfile.createVariable(config.get('y_coord_var_name', 'y0'), 'f4', (config.get('y_dim_name', 'y0'),)) #
                y_v.standard_name = "projection_y_coordinate" #
                y_v.units = "km" #
                y_v.axis = "Y" #
                y_v[:] = y_coord_values #
                
                z_v = ncfile.createVariable(config.get('z_coord_var_name', 'z0'), 'f4', (config.get('z_dim_name', 'z0'),)) #
                z_v.standard_name = "altitude" #
                z_v.long_name = "constant altitude levels" #
                z_v.units = "km" #
                z_v.positive = "up" #
                z_v.axis = "Z" #
                z_v[:] = z_coord_values #

                lat0_v = ncfile.createVariable('lat0', 'f4', (config.get('y_dim_name', 'y0'), config.get('x_dim_name', 'x0'))) #
                lat0_v.standard_name = "latitude" #
                lat0_v.units = "degrees_north" #
                lat0_v[:] = lat0 #
                
                lon0_v = ncfile.createVariable('lon0', 'f4', (config.get('y_dim_name', 'y0'), config.get('x_dim_name', 'x0'))) #
                lon0_v.standard_name = "longitude" #
                lon0_v.units = "degrees_east" #
                lon0_v[:] = lon0 #

                gm_v = ncfile.createVariable(grid_mapping_var_name, 'i4') #
                gm_v.grid_mapping_name = "azimuthal_equidistant" #
                gm_v.longitude_of_projection_origin = proj_origin_lon #
                gm_v.latitude_of_projection_origin = proj_origin_lat #
                gm_v.false_easting = 0.0 #
                gm_v.false_northing = 0.0 #
                gm_v.earth_radius = earth_radius_m #

                dbz_v = ncfile.createVariable(dbz_var_name, 'i1', 
                             (config.get('time_dim_name', 'time'), 
                              config.get('z_dim_name', 'z0'), # <<-- OBTENER DE CONFIG
                              config.get('y_dim_name', 'y0'), 
                              config.get('x_dim_name', 'x0')),
                             fill_value=output_fill_value_byte) #
                dbz_v.units = 'dBZ' #
                dbz_v.long_name = 'DBZ' #
                dbz_v.standard_name = 'DBZ' #
                dbz_v.coordinates = "lon0 lat0" #
                dbz_v.grid_mapping = grid_mapping_var_name #
                dbz_v.scale_factor = output_scale_factor #
                dbz_v.add_offset = output_add_offset #
                dbz_v.valid_min = np.int8(-127) #
                dbz_v.valid_max = np.int8(127) #
                # **AJUSTE 4: Atributos min/max_value consistentes**
                dbz_v.min_value = np.float32(min_dbz_norm_config) # Usar el mismo min de (de)normalización
                dbz_v.max_value = np.float32(max_dbz_norm_config) # Usar el mismo max de (de)normalización
                dbz_v[:] = pred_data_final_for_nc #

                print(f"NetCDF predicción t+{config['prediction_interval_minutes']}min guardado: {output_filename}") #
            sample_count += 1 #

def main():
    set_seed(42) #
    config = {
        'data_dir': "/home/sample", 
        'model_save_dir': "/home/model_output_final_v_ckpt",
        'predictions_output_dir': "/home/predictions_final_v_ckpt",

        'seq_len': 6, 
        'pred_len': 1, 
        'pred_steps_model': 1, 

        # **AJUSTE 1: Estandarizar nombres y valores para normalización/desnormalización**
        # Estos serán usados por RadarDataset y generate_prediction_netcdf
        'norm_min_dbz': -30.5,  # Valor físico del _FillValue original (-128 * 0.5 + 33.5), o un poco menos (-31).
        'norm_max_dbz': 70.0,   # Máximo físico esperado/deseado (ej. 65.0, 70.0, o 75.0)
        
        # **AJUSTE 2: Umbral físico para considerar una predicción como nula (para _FillValue)**
        # Debe ser un poco mayor que norm_min_dbz para capturar predicciones que son "nulas".
        # Si norm_min_dbz es -30.5, un umbral de -28.0 podría ser un buen inicio.
        # Si el modelo predice 0.0 dBZ para el fondo, un umbral de 0.1 dBZ sería adecuado.
        'physical_null_threshold': -28.0, # dBZ. Ajusta según lo que el modelo aprenda a predecir para nulos.

        # 'fill_value': -9999.0, # Este fill_value en config no se usa activamente si RadarDataset maneja NaNs
        'expected_shape': (18, 500, 500), 
        'dbz_variable_name': 'DBZ', # Para leer de los NC de entrada

        # Parámetros del Sensor y Grilla (mayormente para metadatos NetCDF)
        'sensor_latitude': -34.64799880981445,   
        'sensor_longitude': -68.01699829101562,  
        'sensor_altitude_km': 0.550000011920929,
        'grid_minz_km': 1.0, 'grid_dz_km': 1.0,
        'grid_minx_km': -249.5, 'grid_dx_km': 1.0,
        'grid_miny_km': -249.5, 'grid_dy_km': 1.0,
        'radar_name': "SAN_RAFAEL", 'institution_name': "UCAR", 
        'author_name': "Federico Caballero", 'author_institution': "Universidad de Mendoza",
        'data_source_name': "Gobierno de Mendoza", 

        # Parámetros para la SALIDA NetCDF
        'dbz_variable_name_pred_nc': 'DBZ', 
        'output_nc_scale_factor': 0.5, #
        'output_nc_add_offset': 33.5, #
        'output_nc_fill_value_byte': -128, # El byte para _FillValue
        'projection_variable_name': "grid_mapping_0", 
        'earth_radius_m': 6378137.0, 

        'prediction_interval_minutes': 3, 

        # Parámetros del Modelo ConvLSTM
        'model_input_dim': 1,
        'model_hidden_dims': [32, 32], #
        'model_kernel_sizes': [(3,3), (3,3)], #
        'model_num_layers': 2, #
        'model_use_layer_norm': True, 'model_use_residual': False, #
        'use_gradient_checkpointing': True, # **AJUSTE 3: Habilitar/Deshabilitar Gradient Checkpointing**

        # Parámetros de Entrenamiento
        'batch_size': 1, #
        'epochs': 2, # Mantener bajo para pruebas iniciales
        'learning_rate': 1e-3, 'weight_decay': 1e-4, 'lr_patience': 3, #
        'use_amp': True, 'accumulation_steps': 1, #
        'clip_grad_norm': 1.0, #
        'log_interval': 1, 
        'checkpoint_interval': 1, #

        'use_ssim_loss': False, 'ssim_kernel_size': 7, 'ssim_loss_weight': 0.3, #

        'train_val_split_ratio': 0.8, #
        'max_sequences_to_use': 10, # Para pruebas rápidas
    }

    os.makedirs(config['model_save_dir'], exist_ok=True) #
    os.makedirs(config['predictions_output_dir'], exist_ok=True) #

    all_subdirs_available = sorted([ #
        d for d in os.listdir(config['data_dir'])
        if os.path.isdir(os.path.join(config['data_dir'], d)) and not d.startswith('.')
    ])
    if not all_subdirs_available: logging.error(f"No subdirs in {config['data_dir']}"); return #

    if config.get('max_sequences_to_use') and config['max_sequences_to_use'] < len(all_subdirs_available): # Usar .get para seguridad
        logging.info(f"Usando muestra aleatoria de {config['max_sequences_to_use']} secuencias.") #
        random.shuffle(all_subdirs_available) #
        subdirs_to_use = all_subdirs_available[:config['max_sequences_to_use']] #
    else: subdirs_to_use = all_subdirs_available #
    
    logging.info(f"Total secuencias a usar: {len(subdirs_to_use)}.") #
    if not subdirs_to_use : logging.error("No hay secuencias para procesar."); return #

    split_idx = int(len(subdirs_to_use) * config['train_val_split_ratio']) #
    train_subdirs, val_subdirs = subdirs_to_use[:split_idx], subdirs_to_use[split_idx:] #
    if not train_subdirs: logging.info("No hay secuencias de entrenamiento, usando todas para validación si existen."); train_subdirs = [] #
    logging.info(f"Entrenamiento: {len(train_subdirs)} sec. Validación: {len(val_subdirs)} sec.") #

    train_loader = None #
    if train_subdirs: #
        # **AJUSTE 4: Pasar los parámetros de normalización correctos a RadarDataset**
        train_dataset = RadarDataset(config['data_dir'], train_subdirs, 
                                     seq_len=config['seq_len'], pred_len=config['pred_len'],
                                     min_dbz_norm=config['norm_min_dbz'], # Usar nuevo nombre de config
                                     max_dbz_norm=config['norm_max_dbz'], # Usar nuevo nombre de config
                                     expected_shape=config['expected_shape'], 
                                     variable_name=config['dbz_variable_name'])
        if len(train_dataset) > 0: #
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True) #
        else:
            logging.info("Dataset de entrenamiento vacío después de filtrar.") #

    val_loader = None; val_dataset_len = 0 #
    if val_subdirs: #
        val_dataset = RadarDataset(config['data_dir'], val_subdirs, 
                                   seq_len=config['seq_len'], pred_len=config['pred_len'],
                                   min_dbz_norm=config['norm_min_dbz'], # Usar nuevo nombre de config
                                   max_dbz_norm=config['norm_max_dbz'], # Usar nuevo nombre de config
                                   expected_shape=config['expected_shape'], 
                                   variable_name=config['dbz_variable_name'])
        val_dataset_len = len(val_dataset) #
        if val_dataset_len > 0: #
             val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True) #
        else: logging.info("Dataset de validación vacío.") #
    else: logging.info("No subdirectorios para validación.") #

    if not val_loader and not train_loader: #
        logging.error("No hay datos de validación ni de entrenamiento para generar predicciones.") #
        return

    model = ConvLSTM3D_Enhanced( #
        input_dim=config['model_input_dim'], hidden_dims=config['model_hidden_dims'],
        kernel_sizes=config['model_kernel_sizes'], num_layers=config['model_num_layers'],
        pred_steps=config['pred_steps_model'], use_layer_norm=config['model_use_layer_norm'],
        use_residual=config['model_use_residual'],
        img_height=config['expected_shape'][1], img_width=config['expected_shape'][2],
        use_gradient_checkpointing=config.get('use_gradient_checkpointing', False) # **AJUSTE 5**
    )
    model.float() #

    logging.info(f"Arquitectura del modelo:\n{model}") #
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) #
    logging.info(f"Número total de parámetros entrenables: {total_params:,}") #

    device_for_execution = torch.device("cuda" if torch.cuda.is_available() else "cpu") #

    model_path = os.path.join(config['model_save_dir'], "best_convlstm_model.pth") #
    if os.path.exists(model_path) and config.get('load_existing_model_if_available', True): # Nuevo flag para controlar carga
        logging.info(f"Cargando modelo pre-entrenado desde: {model_path}") #
        try:
            # Cargar a CPU primero para evitar problemas de GPU si el modelo se guardó en una GPU diferente
            checkpoint_data = torch.load(model_path, map_location='cpu') 
            model.load_state_dict(checkpoint_data['model_state_dict'])
            model.float() 
            # Opcional: Cargar estado del optimizador si vas a continuar el entrenamiento
            # if 'optimizer_state_dict' in checkpoint_data and config.get('continue_training', False):
            #     optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            logging.info(f"Modelo cargado. Dtype parámetros: {next(model.parameters()).dtype}") #
        except Exception as e:
            logging.error(f"Error al cargar el modelo pre-entrenado: {e}. Entrenando desde cero.")
            if not train_loader:
                logging.error("No hay datos de entrenamiento y el modelo pre-entrenado no se pudo cargar. Saliendo.")
                return
            model, history = train_model(model, train_loader, val_loader, config)
    else:
        if os.path.exists(model_path):
             logging.info(f"Modelo pre-entrenado en {model_path} existe, pero load_existing_model_if_available es False. Entrenando desde cero.")
        else:
             logging.info("No se encontró modelo pre-entrenado. Entrenando desde cero...") #
        
        if not train_loader: #
            logging.error("No hay datos de entrenamiento y no se encontró/cargó modelo pre-entrenado. Saliendo.") #
            return
        model, history = train_model(model, train_loader, val_loader, config) #

    # 'model' ya es 'trained_model' en este punto.
    model.to(device_for_execution) #
    model.float() #
    logging.info(f"Modelo listo para predicción. Dtype: {next(model.parameters()).dtype}, Dispositivo: {next(model.parameters()).device}") #

    prediction_loader = val_loader if val_loader and val_dataset_len > 0 else train_loader #
    # Ajustar num_prediction_samples para usar el tamaño del dataset si es menor que 5
    if prediction_loader:
        dataset_size_for_pred = len(prediction_loader.dataset)
        num_prediction_samples = min(config.get('num_prediction_samples', 2), dataset_size_for_pred) # Pide 2 por defecto
    else:
        num_prediction_samples = 0

    if prediction_loader and num_prediction_samples > 0: #
        logging.info(f"Generando {num_prediction_samples} predicciones de ejemplo...") #
        generate_prediction_netcdf(model, prediction_loader, config,
                                   device=device_for_execution,
                                   num_samples=num_prediction_samples) #
    else:
        logging.warning("No hay datos disponibles en val_loader o train_loader para generar predicciones de ejemplo.") #

    logging.info("Proceso completado.") #

if __name__ == '__main__':
    # --- Asegúrate de que todas las clases y funciones necesarias estén definidas arriba ---
    # Ejemplo de cómo podrían estar estructuradas las importaciones si estuvieran en otro archivo
    # from data_utils import RadarDataset 
    # from model_arch import ConvLSTMCell, ConvLSTM2DLayer, ConvLSTM3D_Enhanced 
    # from training_utils import train_model, SSIMLoss
    # from prediction_utils import generate_prediction_netcdf
    # from reproducibility_utils import set_seed
    main()