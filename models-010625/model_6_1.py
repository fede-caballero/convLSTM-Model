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
# from torch.cuda.amp import autocast # Se usará torch.amp.autocast
# from torch.cuda.amp import GradScaler # Se usará torch.amp.GradScaler
import matplotlib.pyplot as plt
import torchmetrics # Para métricas adicionales como SSIM
from torch.utils.checkpoint import checkpoint # <--- IMPORTANTE PARA GRADIENT CHECKPOINTING
import torch.amp # <--- IMPORTANTE PARA APIS MODERNAS DE AMP

# Configuración del Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración para reproducibilidad y rendimiento
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    logging.info(f"Semillas configuradas con valor: {seed}")

class RadarDataset(Dataset):
    def __init__(self, data_dir, subdirs_list, seq_len=6, pred_len=1,
                 min_dbz_norm=-30, max_dbz_norm=70,
                 expected_shape=(18, 500, 500), variable_name='DBZ'):
        self.data_dir = data_dir
        self.subdirs_list = subdirs_list
        self.seq_len = seq_len
        self.pred_len = pred_len
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
            if ".ipynb_checkpoints" in subdir_name:
                logging.debug(f"Omitiendo directorio de checkpoints: {subdir_name}")
                continue
            files = sorted(glob.glob(os.path.join(subdir_path, "*.nc")))
            if len(files) >= self.seq_len + self.pred_len:
                # Verificar que los archivos de salida tengan datos válidos
                output_files = files[self.seq_len:self.seq_len + self.pred_len]
                valid_output = True
                for f in output_files:
                    try:
                        with NCDataset(f, 'r') as nc_file:
                            if self.variable_name not in nc_file.variables:
                                valid_output = False
                                logging.warning(f"Variable {self.variable_name} no encontrada en archivo de salida {f}")
                                break
                            dbz_var = nc_file.variables[self.variable_name]
                            data = dbz_var[0, ...] if dbz_var.ndim == 4 else dbz_var[...]
                            logging.info(f"Archivo {f}: Min={np.min(data):.2f}, Max={np.max(data):.2f}")
                    except Exception as e:
                        logging.warning(f"Error leyendo archivo de salida {f}: {e}")
                        valid_output = False
                        break
                if valid_output:
                    valid_sequences.append((files, subdir_name))
            else:
                logging.warning(f"Subdirectorio {subdir_name} tiene {len(files)} archivos...")
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
                        logging.warning(f"Forma de variable inesperada {dbz_var.shape} en {file_path}. Intentando siguiente muestra.")
                        return self.__getitem__((idx + 1) % len(self))

                    # 2. Crear la máscara de fill_value A PARTIR DE LOS DATOS RAW ORIGINALES
                    is_fill_location = None
                    if hasattr(dbz_var, '_FillValue'):
                        fill_value_packed = dbz_var._FillValue # Este es el _FillValue del tipo de dato original
                        is_fill_location = (data_raw_original_type == np.array(fill_value_packed, dtype=data_raw_original_type.dtype))

                    # 3. Convertir data_raw a float32 para cálculos físicos
                    dbz_physical = data_raw_original_type.astype(np.float32)

                    # 4. Aplicar scale/offset para obtener valores físicos
                    if hasattr(dbz_var, 'scale_factor') and hasattr(dbz_var, 'add_offset'):
                        scale = dbz_var.scale_factor
                        offset = dbz_var.add_offset
                        dbz_physical = dbz_physical * scale + offset

                    # 5. Aplicar la máscara de NaN AHORA a los datos físicos
                    if is_fill_location is not None:
                        dbz_physical[is_fill_location] = np.nan

                    # dbz_physical ahora tiene valores físicos, con NaN donde había _FillValue

                    # Logging para depuración (NIVEL DEBUG)
                    logging.debug(f"FILE: {os.path.basename(file_path)} - dbz_physical (con NaNs): "
                                 f"NanMin={np.nanmin(dbz_physical):.2f}, NanMax={np.nanmax(dbz_physical):.2f}, "
                                 f"NumNaNs={np.isnan(dbz_physical).sum()}")
                    
                    # 6. Clipping y Normalización, preservando NaN
                    dbz_clipped = np.clip(dbz_physical, self.min_dbz_norm, self.max_dbz_norm)
                    
                    dbz_normalized = (dbz_clipped - self.min_dbz_norm) / (self.max_dbz_norm - self.min_dbz_norm)
                    # NaNs en dbz_physical -> NaNs en dbz_clipped -> NaNs en dbz_normalized

                    logging.debug(f"FILE: {os.path.basename(file_path)} - dbz_clipped (con NaNs): "
                                 f"NanMin={np.nanmin(dbz_clipped):.2f}, NanMax={np.nanmax(dbz_clipped):.2f}, "
                                 f"NumNaNs={np.isnan(dbz_clipped).sum()}")
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
                logging.error(f"Error procesando archivo {file_path}...: {e}")
                return self.__getitem__((idx + 1) % len(self))

        if len(input_data_list) != self.seq_len or len(output_data_list) != self.pred_len:
            logging.warning(f"No se pudieron cargar suficientes frames...")
            return self.__getitem__((idx + 1) % len(self))

        input_tensor = np.stack(input_data_list, axis=1)
        output_tensor = np.stack(output_data_list, axis=1)
        logging.debug(f"RadarDataset: output_tensor (ANTES de from_numpy): "
                    f"NanMin={np.nanmin(output_tensor):.4f}, NanMax={np.nanmax(output_tensor):.4f}, "
                    f"NanMean={np.nanmean(output_tensor):.4f}, NumNaNs={np.isnan(output_tensor).sum()}, "
                    f"Shape={output_tensor.shape}, Dtype={output_tensor.dtype}")

        
        x = torch.from_numpy(np.nan_to_num(input_tensor, nan=0.0)).float()
        y = torch.from_numpy(output_tensor).float()
        
        # --- Lógica para devolver Timestamps (DEBES IMPLEMENTAR LA EXTRACCIÓN REAL) ---
        # last_input_file_path = sequence_files[self.seq_len - 1]
        # filename_no_ext = os.path.splitext(os.path.basename(last_input_file_path))[0]
        # last_input_dt_utc_placeholder = datetime.utcnow() # ¡ESTO ES SOLO UN PLACEHOLDER!
        # try:
        #     # Intenta parsear el timestamp del nombre del archivo o del subdirectorio
        #     # Ejemplo: parts = filename_no_ext.split('_'); timestamp_str = parts[0][-8:] + parts[1]
        #     # last_input_dt_utc = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        #     pass # Implementa tu lógica de parseo aquí
        # except Exception as e_time:
        #     logging.warning(f"No se pudo parsear el timestamp de {last_input_file_path} en dataset. Usando placeholder. Error: {e_time}")
        #     # last_input_dt_utc = last_input_dt_utc_placeholder # Mantener el placeholder si falla
    
        # return x, y, last_input_dt_utc_placeholder # Si devuelves timestamp
        return x, y # Si NO devuelves timestamp por ahora

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        nn.init.xavier_uniform_(self.conv.weight)
        if self.bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next
    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class ConvLSTM2DLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, return_all_layers=False):
        super(ConvLSTM2DLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)

    def forward(self, input_tensor, hidden_state=None): # input_tensor: (B, T_in, C_in, H, W)
        b, seq_len, _, h, w = input_tensor.size() # _ es C_in
        device = input_tensor.device
        if hidden_state is None:
            hidden_state = self.cell.init_hidden(b, (h, w), device)

        layer_output_list = []
        h_cur, c_cur = hidden_state
        for t in range(seq_len):
            h_cur, c_cur = self.cell(input_tensor=input_tensor[:, t, :, :, :], cur_state=[h_cur, c_cur])
            layer_output_list.append(h_cur)

        if self.return_all_layers:
            layer_output = torch.stack(layer_output_list, dim=1) # (B, T_in, C_hidden, H, W)
        else:
            # Solo el último estado oculto como salida de la capa, pero manteniendo la dim de tiempo
            layer_output = h_cur.unsqueeze(1) # (B, 1, C_hidden, H, W)

        return layer_output, (h_cur, c_cur)

class ConvLSTM3D_Enhanced(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[32, 64], kernel_sizes=[(3,3), (3,3)],
                 num_layers=2, pred_steps=1, use_layer_norm=True, use_residual=False,
                 img_height=500, img_width=500):
        super(ConvLSTM3D_Enhanced, self).__init__()
        if isinstance(hidden_dims, int): hidden_dims = [hidden_dims] * num_layers
        if isinstance(kernel_sizes, tuple): kernel_sizes = [kernel_sizes] * num_layers
        assert len(hidden_dims) == num_layers and len(kernel_sizes) == num_layers

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.pred_steps = pred_steps
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.img_height = img_height
        self.img_width = img_width

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        current_dim = input_dim
        for i in range(num_layers):
            self.layers.append(
                ConvLSTM2DLayer(input_dim=current_dim, hidden_dim=hidden_dims[i],
                                kernel_size=kernel_sizes[i], bias=True,
                                return_all_layers=True if i < num_layers - 1 else False)
            )
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm([hidden_dims[i], img_height, img_width]))
            current_dim = hidden_dims[i]

        self.output_conv = nn.Conv3d(in_channels=hidden_dims[-1],
                                     out_channels=input_dim * pred_steps,
                                     kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

        logging.info(f"Modelo ConvLSTM3D_Enhanced creado: {num_layers} capas, Hidden dims: {hidden_dims}, LayerNorm: {use_layer_norm}, PredSteps: {pred_steps}")

    def forward(self, x_volumetric):  # Espera (Z, B, T_in, H, W, C_in)
        num_z_levels, b, seq_len, h, w, c_in = x_volumetric.shape
        all_level_predictions = []

        for z_idx in range(num_z_levels):
            x_level = x_volumetric[z_idx, ...]  # (B, T_in, H, W, C_in)
            x_level_permuted = x_level.permute(0, 1, 4, 2, 3)  # (B, T_in, C_in, H, W)
            current_input = x_level_permuted

            hidden_states_for_level = [None] * self.num_layers

            for i in range(self.num_layers):
                layer_output, hidden_state = self.layers[i](current_input, hidden_states_for_level[i])
                hidden_states_for_level[i] = hidden_state

                if self.use_layer_norm and self.layer_norms:
                    B_ln, T_ln, C_ln, H_ln, W_ln = layer_output.shape
                    output_reshaped_for_ln = layer_output.contiguous().view(B_ln * T_ln, C_ln, H_ln, W_ln)
                    normalized_output = self.layer_norms[i](output_reshaped_for_ln)
                    layer_output = normalized_output.view(B_ln, T_ln, C_ln, H_ln, W_ln)
                current_input = layer_output

            output_for_conv3d = current_input.permute(0, 2, 1, 3, 4)
            raw_conv_output = self.output_conv(output_for_conv3d)
            prediction_features = raw_conv_output.squeeze(2)
            level_prediction = prediction_features.view(b, self.pred_steps, self.input_dim, h, w)
            level_prediction = level_prediction.permute(0, 1, 3, 4, 2)
            level_prediction = self.sigmoid(level_prediction)
            all_level_predictions.append(level_prediction)

        predictions_volumetric = torch.stack(all_level_predictions, dim=0)
        return predictions_volumetric


class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, kernel_size_for_metric=7):
        super(SSIMLoss, self).__init__()
        # No necesitas try-except aquí si estás usando una versión de torchmetrics que lo soporta
        self.ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(
            data_range=data_range,
            kernel_size=kernel_size_for_metric,
            reduction='elementwise_mean' # Común, o None y luego .mean()
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, img1, img2): # Espera (Z, B, T_pred, H, W, C)
        num_z, batch_s, pred_t, height, width, channels = img1.shape

        # SSIM se aplica típicamente a imágenes (o slices 2D/3D con un canal)
        # Aplanar Z, B, T_pred en la dimensión de batch para SSIM
        # Permutar para tener (Batch_flat, Canales, H, W)
        img1_reshaped = img1.permute(0, 1, 2, 5, 3, 4).contiguous().view(-1, channels, height, width)
        img2_reshaped = img2.permute(0, 1, 2, 5, 3, 4).contiguous().view(-1, channels, height, width)

        ssim_val_elementwise = self.ssim_metric(img1_reshaped, img2_reshaped) # Esto dará un valor por imagen en el batch aplanado
        ssim_val_mean = ssim_val_elementwise.mean() # Tomar la media sobre todos los elementos del batch aplanado
        logging.info(f"SSIM Mean: {ssim_val_mean.item():.4f}")
        return 1.0 - ssim_val_mean # Queremos maximizar SSIM, así que minimizamos 1-SSIM


def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")
    model.to(device)  # Mover el modelo al dispositivo ANTES de crear el optimizador

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.get('lr_patience', 3), verbose=True)

    criterion_mse = nn.MSELoss().to(device)
    criterion_ssim = None
    ssim_loss_weight = 0.0
    mse_loss_weight = 1.0

    if config.get('use_ssim_loss', False):
        try:
            criterion_ssim = SSIMLoss(
                data_range=1.0,  # Ya que los datos están normalizados a [0,1]
                kernel_size_for_metric=config.get('ssim_kernel_size', 7)
            ).to(device)  # SSIMLoss ya se mueve al device en su __init__
            ssim_loss_weight = config.get('ssim_loss_weight', 0.3)
            mse_loss_weight = 1.0 - ssim_loss_weight
            logging.info(f"Usando SSIM loss con peso {ssim_loss_weight} y MSE con peso {mse_loss_weight}")
        except Exception as e:
            logging.error(f"Error al inicializar SSIMLoss: {e}. Se usará solo MSE.")
            criterion_ssim = None  # Reasegurar
            ssim_loss_weight = 0.0
            mse_loss_weight = 1.0

    scaler = torch.amp.GradScaler(enabled=config['use_amp'])

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    accumulation_steps = config.get('accumulation_steps', 1)

    logging.info(f"Iniciando entrenamiento: {config['epochs']} épocas, LR: {config['learning_rate']}, Batch (efectivo): {config['batch_size'] * accumulation_steps}")

    for epoch in range(config['epochs']):
        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad()  # Mover zero_grad aquí, antes del bucle de acumulación
        # Monitoreo de uso de GPU
        if torch.cuda.is_available():
            logging.info(f"Inicio Época {epoch+1} - Memoria GPU Asignada: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB, Reservada: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

        for batch_idx, (x, y) in enumerate(train_loader):  # x, y de RadarDataset son (Z, T, H, W, C)
            # DataLoader añade B: (B, Z, T, H, W, C)
            x = x.to(device)
            y = y.to(device)

            # Permutar para el modelo: (Z, B, T, H, W, C)
            if x.dim() == 6 and y.dim() == 6:  # (B,Z,T_in,H,W,C) y (B,Z,T_out,H,W,C)
                x = x.permute(1, 0, 2, 3, 4, 5)
                y = y.permute(1, 0, 2, 3, 4, 5)
            else:
                logging.error(f"Formas inesperadas para x o y antes de la permutación: x={x.shape}, y={y.shape}")
                continue

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config['use_amp']):
                predictions = model(x)  # Espera (Z,B,T_in,H,W,C) -> Sale (Z,B,T_pred,H,W,C)

                # Asegúrate que 'y' (objetivo) tenga la misma forma que 'predictions'
                if predictions.shape != y.shape:
                    logging.error(f"Discrepancia de formas entre predicción {predictions.shape} y objetivo {y.shape}")
                    continue

                loss_mse_val = criterion_mse(predictions, y)
                current_loss = loss_mse_val

                if criterion_ssim is not None:
                    loss_ssim_component = criterion_ssim(predictions, y)
                    current_loss = mse_loss_weight * loss_mse_val + ssim_loss_weight * loss_ssim_component

                # Loguear métricas del primer batch de la primera época
                if batch_idx == 0 and epoch == 0:
                    logging.info(f"  Predicciones (normalizadas): Min={predictions.min().item():.4f}, Max={predictions.max().item():.4f}, Mean={predictions.mean().item():.4f}")
                    logging.info(f"  Objetivos y (normalizados): Min={y.min().item():.4f}, Max={y.max().item():.4f}, Mean={y.mean().item():.4f}")

                loss_to_accumulate = current_loss / accumulation_steps

            scaler.scale(loss_to_accumulate).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if config.get('clip_grad_norm', None):
                    scaler.unscale_(optimizer)  # Unscale antes de clip_grad_norm
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['clip_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Zero grad después de step y update

            running_train_loss += current_loss.item()  # Usar current_loss (no dividida por accumulation_steps) para el log

            if (batch_idx + 1) % config.get('log_interval', 1) == 0:
                logging.info(f"Época {epoch+1}/{config['epochs']} [{batch_idx+1}/{len(train_loader)}] - Pérdida (batch): {current_loss.item():.6f}")

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validación
        if val_loader and len(val_loader) > 0:  # Asegurar que val_loader no sea None y tenga datos
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    if x_val.dim() == 6 and y_val.dim() == 6:
                        x_val = x_val.permute(1, 0, 2, 3, 4, 5)
                        y_val = y_val.permute(1, 0, 2, 3, 4, 5)
                    else:
                        logging.error(f"Formas inesperadas (val) x={x_val.shape}, y={y_val.shape}")
                        continue

                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config['use_amp']):
                        predictions_val = model(x_val)
                        if predictions_val.shape != y_val.shape:
                            logging.error(f"Discrepancia de formas (val) entre predicción {predictions_val.shape} y objetivo {y_val.shape}")
                            continue
                        val_loss_mse_val = criterion_mse(predictions_val, y_val)
                        current_val_loss = val_loss_mse_val
                        if criterion_ssim is not None:
                            val_loss_ssim_component = criterion_ssim(predictions_val, y_val)
                            current_val_loss = mse_loss_weight * val_loss_mse_val + ssim_loss_weight * val_loss_ssim_component
                    running_val_loss += current_val_loss.item()

            if len(val_loader) > 0:  # Evitar división por cero
                avg_val_loss = running_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)
                epoch_duration = time.time() - epoch_start_time
                logging.info(f"Época {epoch+1} completada en {epoch_duration:.2f}s. Pérdida (train): {avg_train_loss:.6f}, Pérdida (val): {avg_val_loss:.6f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(), 'loss': best_val_loss},
                               os.path.join(config['model_save_dir'], "best_convlstm_model.pth"))
                    logging.info(f"Mejor modelo guardado (Pérdida Val: {best_val_loss:.6f})")
            else:  # Si len(val_loader) es 0
                epoch_duration = time.time() - epoch_start_time
                logging.info(f"Época {epoch+1} completada en {epoch_duration:.2f}s. Pérdida (train): {avg_train_loss:.6f} (Dataset de validación vacío, no se calculó pérdida de validación)")
        else:  # Si no hay val_loader
            epoch_duration = time.time() - epoch_start_time
            logging.info(f"Época {epoch+1} completada en {epoch_duration:.2f}s. Pérdida (train): {avg_train_loss:.6f} (No hay val_loader)")

        # Guardar checkpoint de época
        if (epoch + 1) % config.get('checkpoint_interval', 1) == 0:
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'train_losses': train_losses,
                        'val_losses': val_losses if (val_loader and len(val_loader) > 0) else []},
                       os.path.join(config['model_save_dir'], f"checkpoint_epoch_{epoch+1}.pth"))
            logging.info(f"Checkpoint guardado en la época {epoch+1}")

    logging.info("Entrenamiento finalizado.")
    if train_loader and len(train_losses) > 0:  # Solo plotear si hubo entrenamiento y pérdidas
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Pérdida Entrenamiento')
        if val_loader and len(val_losses) > 0:
            plt.plot(val_losses, label='Pérdida Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.title('Curvas de Pérdida del Entrenamiento')
        plt.savefig(os.path.join(config['model_save_dir'], "loss_curves.png"))
        plt.close()
        logging.info(f"Curvas de pérdida guardadas en {os.path.join(config['model_save_dir'], 'loss_curves.png')}")

    return model, {'train_losses': train_losses, 'val_losses': val_losses if (val_loader and len(val_loader) > 0) else []}

def generate_prediction_netcdf(model, data_loader, config, device, num_samples=1):
    model.to(device)
    model.float()
    model.eval()

    output_dir = config['predictions_output_dir']
    min_dbz_model_output = config['min_dbz']  # e.g., -29
    max_dbz_model_output = config['max_dbz']  # e.g., 60.5
    
    output_scale_factor = np.float32(config.get('output_nc_scale_factor', 0.5))
    output_add_offset = np.float32(config.get('output_nc_add_offset', 33.5))
    output_fill_value_byte = np.int8(-128)
    
    time_dim_name = config.get('time_dim_name', 'time')
    bounds_dim_name = config.get('bounds_dim_name', 'bounds')
    x_dim_name = config.get('x_dim_name', 'x0')
    y_dim_name = config.get('y_dim_name', 'y0')
    z_dim_name = config.get('z_dim_name', 'z0')
    
    time_var_name = config.get('time_var_name', 'time')
    time_bounds_var_name = config.get('time_bounds_var_name', 'time_bounds')
    start_time_var_name = config.get('start_time_var_name', 'start_time')
    stop_time_var_name = config.get('stop_time_var_name', 'stop_time')
    x_coord_var_name = config.get('x_coord_var_name', 'x0')
    y_coord_var_name = config.get('y_coord_var_name', 'y0')
    z_coord_var_name = config.get('z_coord_var_name', 'z0')
    
    grid_mapping_var_name = config.get('projection_variable_name', "grid_mapping_0")
    dbz_var_name = config.get('dbz_variable_name_pred_nc', 'DBZ')

    proj_origin_lon = config.get('sensor_longitude', -68.0169982910156)
    proj_origin_lat = config.get('sensor_latitude', -34.6479988098145)
    earth_radius_m = config.get('earth_radius_m', 6378137)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_z = config['expected_shape'][0]
    num_y = config['expected_shape'][1]
    num_x = config['expected_shape'][2]

    z_coord_values = np.arange(config['grid_minz_km'], config['grid_minz_km'] + num_z * config['grid_dz_km'], config['grid_dz_km'], dtype=np.float32)[:num_z]
    x_coord_values = np.arange(config['grid_minx_km'], config['grid_minx_km'] + num_x * config['grid_dx_km'], config['grid_dx_km'], dtype=np.float32)[:num_x]
    y_coord_values = np.arange(config['grid_miny_km'], config['grid_miny_km'] + num_y * config['grid_dy_km'], config['grid_dy_km'], dtype=np.float32)[:num_y]
    
    # Calcular lat0 y lon0
    import pyproj
    proj = pyproj.Proj(proj="aeqd", lon_0=proj_origin_lon, lat_0=proj_origin_lat, R=earth_radius_m)
    x_grid, y_grid = np.meshgrid(x_coord_values, y_coord_values)
    lon0, lat0 = proj(x_grid, y_grid, inverse=True)

    sample_count = 0
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(data_loader):
            if sample_count >= num_samples:
                break
            x_input_volume, y_true_volume = data_batch
            last_input_dt_from_loader = datetime.utcnow() - timedelta(minutes=config['seq_len'] * config['prediction_interval_minutes'])

            x_permuted = x_input_volume.permute(1, 0, 2, 3, 4, 5)
            x_to_model = x_permuted.to(device)
            
            if x_to_model.shape[1] > 1:
                logging.warning(f"Procesando solo el primer item de un batch de tamaño {x_to_model.shape[1]}")
            current_x_to_model = x_to_model[:, 0:1, ...]

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config['use_amp']):
                prediction_norm_all_steps = model(current_x_to_model)
            
            pred_data_np = prediction_norm_all_steps[:, 0, 0, :, :, 0].cpu().numpy()  # (Z, H, W)
            
            # Desnormalizar, preservando NaN
            pred_data_desnorm_float = np.where(np.isnan(pred_data_np), np.nan,
                                              pred_data_np * (max_dbz_model_output - min_dbz_model_output) + min_dbz_model_output)
            
            logging.info(f"Predicción Física Desnormalizada (muestra {sample_count}): Min={np.nanmin(pred_data_desnorm_float):.2f}, Max={np.nanmax(pred_data_desnorm_float):.2f}, Mean={np.nanmean(pred_data_desnorm_float):.2f}")
            
            # Marcar áreas sin reflectividad como _FillValue
            pred_data_for_packing = np.where(pred_data_desnorm_float < 0, np.nan, pred_data_desnorm_float)
            pred_data_byte = np.where(np.isnan(pred_data_for_packing), output_fill_value_byte,
                                     np.clip(((pred_data_for_packing - output_add_offset) / output_scale_factor), -127, 127).round().astype(np.int8))
            pred_data_final_for_nc = np.expand_dims(pred_data_byte, axis=0)

            last_input_frame_datetime_utc = last_input_dt_from_loader.replace(tzinfo=None)
            forecast_lead_seconds = (0 + 1) * config['prediction_interval_minutes'] * 60
            actual_forecast_datetime_utc = last_input_frame_datetime_utc + timedelta(seconds=forecast_lead_seconds)

            epoch_time = datetime(1970, 1, 1, 0, 0, 0)
            time_value_seconds = (actual_forecast_datetime_utc - epoch_time).total_seconds()
            time_begin_calc_seconds = time_value_seconds - (2 * 60 + 48)
            time_end_calc_seconds = time_value_seconds
            
            file_timestamp_str = actual_forecast_datetime_utc.strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(output_dir, f"pred_{dbz_var_name}_{file_timestamp_str}_sample{sample_count}.nc")

            with NCDataset(output_filename, 'w', format='NETCDF3_CLASSIC') as ncfile:
                ncfile.Conventions = "CF-1.6"
                ncfile.title = f"{config.get('radar_name', 'SAN_RAFAEL')} - Forecast t+{config['prediction_interval_minutes']}min"
                ncfile.institution = config.get('institution_name', "UCAR")
                ncfile.source = config.get('data_source_name', "Gobierno de Mendoza")
                ncfile.history = f"Created {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} by ConvLSTM prediction script."
                ncfile.comment = f"Forecast data from ConvLSTM model for lead time +{forecast_lead_seconds/60.0:.0f} min."
                ncfile.references = f"Tesis de {config.get('author_name', 'Federico Caballero')}, {config.get('author_institution', 'Universidad de Mendoza')}"

                ncfile.createDimension(time_dim_name, None)
                ncfile.createDimension(bounds_dim_name, 2)
                ncfile.createDimension(x_dim_name, num_x)
                ncfile.createDimension(y_dim_name, num_y)
                ncfile.createDimension(z_dim_name, num_z)

                time_v = ncfile.createVariable(time_var_name, 'f8', (time_dim_name,))
                time_v.standard_name = "time"
                time_v.long_name = "Data time"
                time_v.units = "seconds since 1970-01-01T00:00:00Z"
                time_v.axis = "T"
                time_v.bounds = time_bounds_var_name
                time_v.comment = actual_forecast_datetime_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                time_v[:] = [time_value_seconds]

                time_bnds_v = ncfile.createVariable(time_bounds_var_name, 'f8', (time_dim_name, bounds_dim_name))
                time_bnds_v.comment = "time_bounds also stored the start and stop times, provided the time variable value lies within the start_time to stop_time interval"
                time_bnds_v.units = "seconds since 1970-01-01T00:00:00Z"
                time_bnds_v[:] = [[time_begin_calc_seconds, time_end_calc_seconds]]

                start_time_v = ncfile.createVariable(start_time_var_name, 'f8', (time_dim_name,))
                start_time_v.long_name = "start_time"
                start_time_v.units = "seconds since 1970-01-01T00:00:00Z"
                start_time_v.comment = datetime.fromtimestamp(time_begin_calc_seconds).strftime("%Y-%m-%dT%H:%M:%SZ")
                start_time_v[:] = [time_begin_calc_seconds]

                stop_time_v = ncfile.createVariable(stop_time_var_name, 'f8', (time_dim_name,))
                stop_time_v.long_name = "stop_time"
                stop_time_v.units = "seconds since 1970-01-01T00:00:00Z"
                stop_time_v.comment = datetime.fromtimestamp(time_end_calc_seconds).strftime("%Y-%m-%dT%H:%M:%SZ")
                stop_time_v[:] = [time_end_calc_seconds]

                x_v = ncfile.createVariable(x_coord_var_name, 'f4', (x_dim_name,))
                x_v.standard_name = "projection_x_coordinate"
                x_v.units = "km"
                x_v.axis = "X"
                x_v[:] = x_coord_values
                
                y_v = ncfile.createVariable(y_coord_var_name, 'f4', (y_dim_name,))
                y_v.standard_name = "projection_y_coordinate"
                y_v.units = "km"
                y_v.axis = "Y"
                y_v[:] = y_coord_values
                
                z_v = ncfile.createVariable(z_coord_var_name, 'f4', (z_dim_name,))
                z_v.standard_name = "altitude"
                z_v.long_name = "constant altitude levels"
                z_v.units = "km"
                z_v.positive = "up"
                z_v.axis = "Z"
                z_v[:] = z_coord_values

                # Agregar lat0 y lon0
                lat0_v = ncfile.createVariable('lat0', 'f4', (y_dim_name, x_dim_name))
                lat0_v.standard_name = "latitude"
                lat0_v.units = "degrees_north"
                lat0_v[:] = lat0
                
                lon0_v = ncfile.createVariable('lon0', 'f4', (y_dim_name, x_dim_name))
                lon0_v.standard_name = "longitude"
                lon0_v.units = "degrees_east"
                lon0_v[:] = lon0

                gm_v = ncfile.createVariable(grid_mapping_var_name, 'i4')
                gm_v.grid_mapping_name = "azimuthal_equidistant"
                gm_v.longitude_of_projection_origin = proj_origin_lon
                gm_v.latitude_of_projection_origin = proj_origin_lat
                gm_v.false_easting = 0.0
                gm_v.false_northing = 0.0
                gm_v.earth_radius = earth_radius_m

                dbz_v = ncfile.createVariable(dbz_var_name, 'i1', (time_dim_name, z_dim_name, y_dim_name, x_dim_name),
                                             fill_value=output_fill_value_byte)
                dbz_v.units = 'dBZ'
                dbz_v.long_name = 'DBZ'
                dbz_v.standard_name = 'DBZ'
                dbz_v.coordinates = "lon0 lat0"
                dbz_v.grid_mapping = grid_mapping_var_name
                dbz_v.scale_factor = output_scale_factor
                dbz_v.add_offset = output_add_offset
                dbz_v.valid_min = np.int8(-127)
                dbz_v.valid_max = np.int8(127)
                dbz_v.min_value = np.float32(config.get('template_min_value', -29.0))
                dbz_v.max_value = np.float32(config.get('template_max_value', 60.5))
                dbz_v[:] = pred_data_final_for_nc

                print(f"NetCDF predicción t+{config['prediction_interval_minutes']}min guardado: {output_filename}")
            sample_count += 1


def main():
    set_seed(42)
    config = {
        'data_dir': "/home/sample", # AJUSTADO para tu prueba con el tar.gz
        'model_save_dir': "/home/model_output_final_v_ckpt",
        'predictions_output_dir': "/home/predictions_final_v_ckpt",

        'seq_len': 6,      # MODIFICADO
        'pred_len': 1,      # MODIFICADO (RadarDataset usará esto para y_true_volume)
        'pred_steps_model': 1, # MODIFICADO (El modelo generará esta cantidad de pasos)

        'min_dbz': -40.0,
        'max_dbz': 70.0,
        'fill_value': -9999.0, # Fill value para los datos flotantes del modelo ANTES de empaquetar
        'expected_shape': (18, 500, 500), # nz, ny, nx
        'dbz_variable_name': 'DBZ',
        # 'dbz_variable_name_pred': 'DBZ_forecast', # Usaremos dbz_variable_name_pred_nc para el nombre final

        # --- Parámetros del Sensor y Grilla (para coincidir con la plantilla MDV/NetCDF) ---
        'sensor_latitude': -34.64799880981445,    # De la plantilla
        'sensor_longitude': -68.01699829101562,   # De la plantilla
        'sensor_altitude_km': 0.550000011920929, # De la plantilla

        # Grilla de la plantilla (18, 500, 500)
        'grid_minz_km': 1.0,
        'grid_dz_km': 1.0,
        'grid_minx_km': -249.5,
        'grid_dx_km': 1.0,
        'grid_miny_km': -249.5,
        'grid_dy_km': 1.0,

        'radar_name': "SAN_RAFAEL",             # De la plantilla
        'institution_name': "UCAR",             # De la plantilla
        'author_name': "Federico Caballero",
        'author_institution': "Universidad de Mendoza",
        'data_source_name': "Gobierno de Mendoza", # De la plantilla (para atributo global 'source')

        # --- Parámetros para la SALIDA NetCDF (para que se parezca a la plantilla) ---
        'dbz_variable_name_pred_nc': 'DBZ', # Nombre de la variable en el NetCDF de SALIDA
        # Parámetros de empaquetado para la variable DBZ de SALIDA (byte)
        # Tomados de la plantilla ncfdata20100101_204855_ncdump.txt
        'output_nc_scale_factor': 0.5,
        'output_nc_add_offset': 33.5,
        # Parámetros de la proyección AZIMUTHAL_EQUIDISTANT (de la plantilla)
        'projection_variable_name': "grid_mapping_0",
        'earth_radius_m': 6378137.0,

        'prediction_interval_minutes': 3, # Intervalo entre los archivos de entrada/salida
                                          # Si tus archivos de radar son cada 3 min, este es el valor.

        'model_input_dim': 1,
        'model_hidden_dims': [32, 32],
        'model_kernel_sizes': [(3,3), (3,3)],
        'model_num_layers': 2,
        'model_use_layer_norm': True, 'model_use_residual': False,

        'batch_size': 1,
        'epochs': 2, # AJUSTADO para una prueba rápida de predicción
        'learning_rate': 1e-3, 'weight_decay': 1e-4, 'lr_patience': 3,
        'use_amp': True, 'accumulation_steps': 1,
        'clip_grad_norm': 1.0,
        'log_interval': 1, # Loguear cada batch para prueba
        'checkpoint_interval': 1,

        'use_ssim_loss': False, 'ssim_kernel_size': 7, 'ssim_loss_weight': 0.3,

        'train_val_split_ratio': 0.8,
        'max_sequences_to_use': 10, # AJUSTADO para una prueba rápida con tus datos de muestra
    }

    os.makedirs(config['model_save_dir'], exist_ok=True)
    os.makedirs(config['predictions_output_dir'], exist_ok=True)

    all_subdirs_available = sorted([
        d for d in os.listdir(config['data_dir'])
        if os.path.isdir(os.path.join(config['data_dir'], d)) and not d.startswith('.')
    ])
    if not all_subdirs_available: logging.error(f"No subdirs in {config['data_dir']}"); return

    if config['max_sequences_to_use'] and config['max_sequences_to_use'] < len(all_subdirs_available):
        logging.info(f"Usando muestra aleatoria de {config['max_sequences_to_use']} secuencias.")
        random.shuffle(all_subdirs_available)
        subdirs_to_use = all_subdirs_available[:config['max_sequences_to_use']]
    else: subdirs_to_use = all_subdirs_available
    logging.info(f"Total secuencias a usar: {len(subdirs_to_use)}.")
    if not subdirs_to_use : logging.error("No hay secuencias para procesar."); return

    split_idx = int(len(subdirs_to_use) * config['train_val_split_ratio'])
    train_subdirs, val_subdirs = subdirs_to_use[:split_idx], subdirs_to_use[split_idx:]
    if not train_subdirs: logging.info("No hay secuencias de entrenamiento, usando todas para validación si existen."); train_subdirs = [] # Permitir correr solo con validación para predicción

    logging.info(f"Entrenamiento: {len(train_subdirs)} sec. Validación: {len(val_subdirs)} sec.")

    train_loader = None
    if train_subdirs:
        train_dataset = RadarDataset(config['data_dir'], train_subdirs, seq_len=config['seq_len'], pred_len=config['pred_len'],
                                     min_dbz_norm=config['min_dbz'], max_dbz_norm=config['max_dbz'],
                                     expected_shape=config['expected_shape'], variable_name=config['dbz_variable_name'])
        if len(train_dataset) > 0:
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        else:
            logging.info("Dataset de entrenamiento vacío después de filtrar.")
            # Si no hay datos de entrenamiento, no podemos entrenar. Podrías querer cargar un modelo directamente.

    val_loader = None; val_dataset_len = 0
    if val_subdirs:
        val_dataset = RadarDataset(config['data_dir'], val_subdirs, seq_len=config['seq_len'], pred_len=config['pred_len'],
                                   min_dbz_norm=config['min_dbz'], max_dbz_norm=config['max_dbz'],
                                   expected_shape=config['expected_shape'], variable_name=config['dbz_variable_name'])
        val_dataset_len = len(val_dataset)
        if val_dataset_len > 0:
             val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        else: logging.info("Dataset de validación vacío.")
    else: logging.info("No subdirectorios para validación.")

    # Necesitamos datos para generar predicciones, al menos de validación o entrenamiento
    if not val_loader and not train_loader:
        logging.error("No hay datos de validación ni de entrenamiento para generar predicciones.")
        return

    model = ConvLSTM3D_Enhanced(
        input_dim=config['model_input_dim'], hidden_dims=config['model_hidden_dims'],
        kernel_sizes=config['model_kernel_sizes'], num_layers=config['model_num_layers'],
        pred_steps=config['pred_steps_model'], use_layer_norm=config['model_use_layer_norm'],
        use_residual=config['model_use_residual'],
        img_height=config['expected_shape'][1], img_width=config['expected_shape'][2]
    )
    model.float() # Asegurar que el modelo se inicialice en float32

    logging.info(f"Arquitectura del modelo:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Número total de parámetros entrenables: {total_params:,}")

    device_for_execution = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(config['model_save_dir'], "best_convlstm_model.pth")
    if os.path.exists(model_path):
        logging.info(f"Cargando modelo pre-entrenado desde: {model_path}")
        # Cargar a CPU, luego asegurar .float(), luego mover a device
        checkpoint_data = torch.load(model_path, map_location='cpu', weights_only=True) #weights_only=True por seguridad
        model.load_state_dict(checkpoint_data['model_state_dict'])
        model.float() # Asegurar float32 después de cargar
        logging.info(f"Modelo cargado. Dtype parámetros: {next(model.parameters()).dtype}")
        trained_model = model
    else:
        logging.info("No se encontró modelo pre-entrenado. Entrenando desde cero...")
        if not train_loader:
            logging.error("No hay datos de entrenamiento y no se encontró modelo pre-entrenado. Saliendo.")
            return
        trained_model, history = train_model(model, train_loader, val_loader, config) # train_model se encarga de .to(device)

    trained_model.to(device_for_execution) # Mover el modelo final al dispositivo
    trained_model.float() # Re-asegurar float32 después de mover (por si acaso)
    logging.info(f"Modelo listo para predicción. Dtype: {next(trained_model.parameters()).dtype}, Dispositivo: {next(trained_model.parameters()).device}")

    # Priorizar val_loader para predicciones, si no, usar train_loader
    prediction_loader = val_loader if val_loader and val_dataset_len > 0 else train_loader
    num_prediction_samples = min(5, val_dataset_len if val_loader and val_dataset_len > 0 else (len(train_loader.dataset) if train_loader else 0))

    if prediction_loader and num_prediction_samples > 0:
        logging.info("Generando predicciones de ejemplo...")
        generate_prediction_netcdf(trained_model, prediction_loader, config,
                                   device=device_for_execution,
                                   num_samples=num_prediction_samples)
    else:
        logging.warning("No hay datos disponibles en val_loader o train_loader para generar predicciones de ejemplo.")

    logging.info("Proceso completado.")

if __name__ == '__main__':
    main()