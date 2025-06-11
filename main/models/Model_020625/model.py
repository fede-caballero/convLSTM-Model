import os
import glob
import random
import time
from datetime import datetime, timedelta, timezone
import logging
import xarray as xr # <<< MEJORA: Usar xarray para una lectura robusta
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchmetrics
import torch.amp
import matplotlib.pyplot as plt
import pyproj
from netCDF4 import Dataset as NCDataset


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
    def __init__(self, sequence_paths, seq_len=12, pred_len=5, 
                 min_dbz_norm=-29.0, max_dbz_norm=65.0):
        self.sequence_paths = sequence_paths
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.min_dbz_norm = min_dbz_norm
        self.max_dbz_norm = max_dbz_norm
        logging.info(f"RadarDataset inicializado con {len(self.sequence_paths)} secuencias.")

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        sequence_files = self.sequence_paths[idx]
        data_list = []

        for file_path in sequence_files:
            try:
                # <<< MEJORA: Lectura automática y segura con xarray >>>
                # xarray maneja _FillValue, scale_factor y add_offset automáticamente
                with xr.open_dataset(file_path, mask_and_scale=True, decode_times=False) as ds:
                    # .values extrae un array de numpy puro con NaNs donde corresponde
                    dbz_physical = ds['DBZ'].values
                
                # Normalización, preservando NaNs
                dbz_clipped = np.clip(dbz_physical, self.min_dbz_norm, self.max_dbz_norm)
                dbz_normalized = (dbz_clipped - self.min_dbz_norm) / (self.max_dbz_norm - self.min_dbz_norm)
                
                data_list.append(dbz_normalized[0, ..., np.newaxis]) # [0] para quitar dim de tiempo, newaxis para canal

            except Exception as e:
                logging.error(f"Error procesando archivo {file_path}. Omitiendo secuencia. Error: {e}")
                # Devolver la siguiente muestra si esta falla
                return self.__getitem__((idx + 1) % len(self))
        
        full_sequence = np.stack(data_list, axis=1) # Forma: (Z, T, H, W, C)
        
        input_tensor = full_sequence[:, :self.seq_len, ...]
        output_tensor = full_sequence[:, self.seq_len:, ...]

        # Reemplazar NaNs con 0 para la entrada (X)
        x = torch.from_numpy(np.nan_to_num(input_tensor, nan=0.0)).float()
        # Mantener NaNs para el objetivo (Y) para la pérdida enmascarada
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
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.get('lr_patience', 3), verbose=True)

    criterion_mse = nn.MSELoss().to(device)
    criterion_ssim = None
    if config.get('use_ssim_loss', False):
        criterion_ssim = SSIMLoss(data_range=1.0, kernel_size_for_metric=config.get('ssim_kernel_size', 7)).to(device)
        ssim_loss_weight = config.get('ssim_loss_weight', 0.3)
        mse_loss_weight = 1.0 - ssim_loss_weight
        logging.info(f"Usando SSIM loss con peso {ssim_loss_weight} y MSE con peso {mse_loss_weight}")
    else:
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
        optimizer.zero_grad()

        if torch.cuda.is_available():
            logging.info(f"Inicio Época {epoch+1} - Memoria GPU Asignada: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device).permute(1, 0, 2, 3, 4, 5)
            y = y.to(device).permute(1, 0, 2, 3, 4, 5)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config['use_amp']):
                predictions = model(x)

                # Asegúrate que 'y' (objetivo) tenga la misma forma que 'predictions'
                if predictions.shape != y.shape:
                    logging.error(f"Discrepancia de formas: Pred {predictions.shape}, Target {y.shape}")
                    continue

                valid_mask = ~torch.isnan(y)
                
                loss_mse_val = criterion_mse(predictions[valid_mask], y[valid_mask])
                current_loss = loss_mse_val

                if criterion_ssim is not None:
                    preds_for_ssim = torch.nan_to_num(predictions, nan=0.0)
                    y_for_ssim = torch.nan_to_num(y, nan=0.0)
                    loss_ssim_component = criterion_ssim(preds_for_ssim, y_for_ssim)
                    current_loss = mse_loss_weight * loss_mse_val + ssim_loss_weight * loss_ssim_component
                # <<< FIN DE LA CORRECCIÓN CRÍTICA >>>

                loss_to_accumulate = current_loss / accumulation_steps

            scaler.scale(loss_to_accumulate).backward()


            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if config.get('clip_grad_norm', None):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['clip_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_train_loss += current_loss.item()

            if (batch_idx + 1) % config.get('log_interval', 1) == 0:
                logging.info(f"Época {epoch+1}/{config['epochs']} [{batch_idx+1}/{len(train_loader)}] - Pérdida (batch): {current_loss.item():.6f}")

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validación
        if val_loader and len(val_loader) > 0:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device).permute(1, 0, 2, 3, 4, 5)
                    y_val = y_val.to(device).permute(1, 0, 2, 3, 4, 5)

                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config['use_amp']):
                        predictions_val = model(x_val)

                        valid_mask_val = ~torch.isnan(y_val)
                        
                        val_loss_mse_val = criterion_mse(predictions_val[valid_mask_val], y_val[valid_mask_val])
                        current_val_loss = val_loss_mse_val
                        
                        if criterion_ssim is not None:
                            preds_val_for_ssim = torch.nan_to_num(predictions_val, nan=0.0)
                            y_val_for_ssim = torch.nan_to_num(y_val, nan=0.0)
                            val_loss_ssim_component = criterion_ssim(preds_val_for_ssim, y_val_for_ssim)
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

    return model, {'train_losses': train_losses, 'val_losses': val_losses}


def generate_prediction_netcdf(model, data_loader, config, device, num_samples=1):
    model.to(device).eval()
    
    output_dir = config['predictions_output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # --- Parámetros de la Salida ---
    # --- Parámetros de la Salida (leídos desde tu config) ---
    min_dbz, max_dbz = config['min_dbz'], config['max_dbz']
    scale_out = np.float32(config['output_nc_scale_factor'])
    offset_out = np.float32(config['output_nc_add_offset'])
    fill_byte_out = np.int8(config['output_nc_fill_value'])
    fill_physical_out = (float(fill_byte_out) * scale_out) + offset_out

    # --- Preparación de la Grilla (leído desde tu config) ---
    num_z, num_y, num_x = config['expected_shape']
    z_coords = np.arange(1.0, 1.0 + num_z * 1.0, 1.0, dtype=np.float32)
    x_coords = np.arange(-249.5, -249.5 + num_x * 1.0, 1.0, dtype=np.float32)
    y_coords = np.arange(-249.5, -249.5 + num_y * 1.0, 1.0, dtype=np.float32)
    
    # Pre-calcular grillas de lat/lon para los metadatos
    proj = pyproj.Proj(proj="aeqd", lon_0=config['sensor_longitude'], lat_0=config['sensor_latitude'], R=config['earth_radius_m'])
    x_grid_m, y_grid_m = np.meshgrid(x_coords * 1000.0, y_coords * 1000.0)
    lon0_grid, lat0_grid = proj(x_grid_m, y_grid_m, inverse=True)

    sample_count = 0
    with torch.no_grad():
        # El DataLoader ahora debe devolver x, y, y el path del último input
        for x_input_volume, _, last_input_filepath_batch in data_loader:
            if sample_count >= num_samples: break
            
            x_to_model = x_input_volume[0:1].permute(1, 0, 2, 3, 4, 5).to(device)
            last_input_filepath = last_input_filepath_batch[0]

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=config['use_amp']):
                predictions_norm = model(x_to_model)

            # --- Bucle para iterar sobre cada uno de los 5 pasos de la predicción ---
            for pred_step_idx in range(config['pred_len']):
                
                pred_norm_step = predictions_norm[:, 0, pred_step_idx, :, :, 0].cpu().numpy()

                # --- Lógica de Desnormalización y Empaquetado ---
                # --- Proceso de Desnormalización y Empaquetado ---
                # 1. Desnormalizar a valores físicos (dBZ)
                pred_physical_dbz_raw = pred_norm_step * (max_dbz - min_dbz) + min_dbz

                # <<< LÍNEA DE SEGURIDAD CRÍTICA AÑADIDA >>>
                pred_physical_dbz_clipped = np.clip(pred_physical_dbz_raw, min_dbz, max_dbz)
                
                # 2. Aplicar umbral físico: todo lo irrelevante se convierte en NaN
                pred_physical_dbz_cleaned = pred_physical_dbz_clipped
                pred_physical_dbz_cleaned[pred_physical_dbz_cleaned < config.get('MIN_RELEVANT_DBZ', 5.0)] = np.nan
                
                # 3. Preparar para empaquetado: Reemplazar NaNs con el valor físico de relleno
                dbz_for_packing = np.where(np.isnan(pred_physical_dbz_cleaned), fill_physical_out, pred_physical_dbz_cleaned)
                
                # 4. Empaquetar a byte
                dbz_packed_byte = np.round((dbz_for_packing - offset_out) / scale_out).astype(np.int8)
                dbz_packed_byte[np.isclose(dbz_for_packing, fill_physical_out)] = fill_byte_out
                dbz_final_packed = dbz_packed_byte[np.newaxis, ...]

                # --- Cálculo de Timestamps para ESTE paso de predicción ---
                try:
                    parts = last_input_filepath.split('/')
                    date_str, time_str = parts[-2][:8], os.path.splitext(parts[-1])[0]
                    last_input_dt_utc = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
                except Exception:
                    last_input_dt_utc = datetime.utcnow()

                lead_time_minutes = (pred_step_idx + 1) * config['prediction_interval_minutes']
                forecast_dt_utc = last_input_dt_utc + timedelta(minutes=lead_time_minutes)
                
                # --- Escritura del Archivo NetCDF Completo para este paso ---
                file_ts = forecast_dt_utc.strftime("%Y%m%d_%H%M%S")
                output_filename = os.path.join(output_dir, f"pred_t+{lead_time_minutes}min_{file_ts}_sample{sample_count}.nc")

                with NCDataset(output_filename, 'w', format='NETCDF3_CLASSIC') as ds_out:
                        # <<< INICIO DE TU CÓDIGO DE ESCRITURA DETALLADO (ADAPTADO) >>>
                        
                        # --- Atributos Globales ---
                        ds_out.Conventions = "CF-1.6"
                        ds_out.title = f"{config.get('radar_name', 'SAN_RAFAEL')} - Forecast t+{lead_time_minutes}min"
                        ds_out.institution = config.get('institution_name', "UCAR")
                        ds_out.source = config.get('data_source_name', "ConvLSTM Model Prediction")
                        ds_out.history = f"Created {datetime.now(timezone.utc).isoformat()} by ConvLSTM prediction script."
                        ds_out.comment = f"Forecast data from model. Lead time: {lead_time_minutes} min."

                        # --- Dimensiones ---
                        ds_out.createDimension('time', None)
                        ds_out.createDimension('bounds', 2)
                        ds_out.createDimension('x0', num_x)
                        ds_out.createDimension('y0', num_y)
                        ds_out.createDimension('z0', num_z)

                        # --- Variables de Tiempo ---
                        time_value = (forecast_dt_utc.replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds()
                        
                        time_v = ds_out.createVariable('time', 'f8', ('time',))
                        time_v.standard_name = "time"; time_v.long_name = "Data time"
                        time_v.units = "seconds since 1970-01-01T00:00:00Z"; time_v.axis = "T"
                        time_v.bounds = "time_bounds"; time_v.comment = forecast_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                        time_v[:] = [time_value]

                        # Puedes ajustar la lógica de start/stop time si lo necesitas, o simplificarla
                        start_time_v = ds_out.createVariable('start_time', 'f8', ('time',))
                        start_time_v[:] = [time_value - 180] # Ejemplo: 3 minutos antes
                        stop_time_v = ds_out.createVariable('stop_time', 'f8', ('time',))
                        stop_time_v[:] = [time_value]
                        time_bnds_v = ds_out.createVariable('time_bounds', 'f8', ('time', 'bounds',))
                        time_bnds_v[:] = [[time_value - 180, time_value]]

                        # --- Variables de Coordenadas ---
                        x_v = ds_out.createVariable('x0', 'f4', ('x0',)); x_v.setncatts({'standard_name':"projection_x_coordinate", 'units':"km", 'axis':"X"}); x_v[:] = x_coords
                        y_v = ds_out.createVariable('y0', 'f4', ('y0',)); y_v.setncatts({'standard_name':"projection_y_coordinate", 'units':"km", 'axis':"Y"}); y_v[:] = y_coords
                        z_v = ds_out.createVariable('z0', 'f4', ('z0',)); z_v.setncatts({'standard_name':"altitude", 'units':"km", 'axis':"Z", 'positive':"up"}); z_v[:] = z_coords
                        
                        # --- Variables de Georreferenciación ---
                        lat0_v = ds_out.createVariable('lat0', 'f4', ('y0', 'x0',)); lat0_v.setncatts({'standard_name':"latitude", 'units':"degrees_north"}); lat0_v[:] = lat0_grid
                        lon0_v = ds_out.createVariable('lon0', 'f4', ('y0', 'x0',)); lon0_v.setncatts({'standard_name':"longitude", 'units':"degrees_east"}); lon0_v[:] = lon0_grid
                        
                        gm_v = ds_out.createVariable('grid_mapping_0', 'i4'); gm_v.setncatts({'grid_mapping_name':"azimuthal_equidistant", 'longitude_of_projection_origin':config['sensor_longitude'], 'latitude_of_projection_origin':config['sensor_latitude'], 'false_easting':0.0, 'false_northing':0.0, 'earth_radius':config['earth_radius_m']})

                        # --- Variable Principal DBZ ---
                        dbz_v = ds_out.createVariable('DBZ', 'i1', ('time', 'z0', 'y0', 'x0'), fill_value=fill_byte_out)
                        dbz_v.setncatts({'units':'dBZ', 'long_name':'DBZ', 'standard_name':'DBZ', 'coordinates':"lon0 lat0", 'grid_mapping':'grid_mapping_0', 'scale_factor':scale_out, 'add_offset':offset_out, 'valid_min':np.int8(-127), 'valid_max':np.int8(127), 'min_value':np.float32(min_dbz), 'max_value':np.float32(max_dbz)})
                        dbz_v[:] = dbz_final_packed

                logging.info(f"Predicción t+{lead_time_minutes}min guardada en: {output_filename}")

            sample_count += 1


def prepare_and_split_data(root_dir, train_ratio, total_seq_len, seq_stride=1):
    """
    Escanea un directorio donde cada subdirectorio es un evento, ordena los
    eventos cronológicamente, y genera secuencias de entrenamiento y validación.
    """
    # 1. Encontrar todos los directorios de eventos
    try:
        all_event_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
        ])
    except FileNotFoundError:
        logging.error(f"El directorio del dataset no fue encontrado en: {root_dir}")
        return [], []

    if not all_event_dirs:
        logging.warning(f"No se encontraron directorios de eventos en {root_dir}")
        return [], []

    logging.info(f"Encontrados {len(all_event_dirs)} directorios de eventos para procesar.")

    # 2. Dividir la LISTA DE DIRECTORIOS cronológicamente
    split_idx = int(len(all_event_dirs) * train_ratio)
    train_dirs = all_event_dirs[:split_idx]
    val_dirs = all_event_dirs[split_idx:]
    
    logging.info(f"División de eventos - Entrenamiento: {len(train_dirs)} directorios, Validación: {len(val_dirs)} directorios")

    def create_sliding_windows(event_dir_list, base_path):
        """Función interna para generar secuencias con ventanas deslizantes."""
        all_sequences = []
        for event_dir in event_dir_list:
            dir_path = os.path.join(base_path, event_dir)
            files = sorted(glob.glob(os.path.join(dir_path, "*.nc")))
            
            if len(files) >= total_seq_len:
                for i in range(0, len(files) - total_seq_len + 1, seq_stride):
                    sequence = files[i : i + total_seq_len]
                    all_sequences.append(sequence)
        return all_sequences

    # 3. Generar las listas de secuencias para cada conjunto
    train_sequences = create_sliding_windows(train_dirs, root_dir)
    val_sequences = create_sliding_windows(val_dirs, root_dir)

    logging.info(f"Generadas {len(train_sequences)} secuencias de entrenamiento y {len(val_sequences)} de validación.")
    
    return train_sequences, val_sequences

def main():
    set_seed(42)
    config = {
        # --- Rutas y Configuración del Dataset ---
        'dataset_dir': "/home/sample", # <<< ¡AJUSTA ESTA RUTA!
        'model_save_dir': "/home/model",     # <<< ¡AJUSTA ESTA RUTA!
        'predictions_output_dir': "/home/predictions", # <<< ¡AJUSTA ESTA RUTA!

        # --- Parámetros de la Estrategia (12 -> 5) ---
        'seq_len': 12,
        'pred_len': 5,
        'total_seq_len': 17,
        'seq_stride': 1,

        # --- Parámetros de División del Dataset ---
        'train_val_split_ratio': 0.8,
        'max_sequences_to_use': 10, # Perfecto para una prueba rápida

        # --- Parámetros de Normalización y Físicos ---
        'min_dbz': -29.0,
        'max_dbz': 65.0,
        'MIN_RELEVANT_DBZ': 5.0,
        'expected_shape': (18, 500, 500),

        # --- Parámetros de la Salida NetCDF (para compatibilidad con TITAN) ---
        'output_nc_scale_factor': 0.5,
        'output_nc_add_offset': 33.5,
        'output_nc_fill_value': -128,

        # --- Hiperparámetros del Modelo y Entrenamiento ---
        # <<< INICIO DE LAS CLAVES FALTANTES >>>
        'model_input_dim': 1,           # 1 canal de entrada (reflectividad)
        'pred_steps_model': 5,          # El modelo debe saber que predice 5 pasos
        'model_hidden_dims': [32, 32],  # Ejemplo: 2 capas de ConvLSTM
        'model_kernel_sizes': [(3,3), (3,3)],
        'model_num_layers': 2,
        'model_use_layer_norm': True,
        'model_use_residual': False,
        # <<< FIN DE LAS CLAVES FALTANTES >>>
        
        'batch_size': 4,
        'epochs': 5,
        'learning_rate': 1e-4,
        'use_amp': True,
        'use_ssim_loss': False, # Empezar solo con MSE es más simple
        'ssim_loss_weight': 0.5,
        'ssim_kernel_size': 7,
        'accumulation_steps': 1,
        'clip_grad_norm': 1.0,
        'log_interval': 10,
        'checkpoint_interval': 1,
        'lr_patience': 3,
        'weight_decay': 1e-5,

        # --- Otros parámetros para la generación de NetCDF ---
        'sensor_latitude': -34.64799880981445,
        'sensor_longitude': -68.01699829101562,
        'earth_radius_m': 6378137.0,
        'prediction_interval_minutes': 3
    }

    os.makedirs(config['model_save_dir'], exist_ok=True)
    os.makedirs(config['predictions_output_dir'], exist_ok=True)

    # 1. Usar nuestra nueva función para preparar los datos de forma robusta
    train_seq_paths, val_seq_paths = prepare_and_split_data(
        root_dir=config['dataset_dir'],
        train_ratio=config['train_val_split_ratio'],
        total_seq_len=config['total_seq_len'],
        seq_stride=config.get('seq_stride', 1)
    )

    if not train_seq_paths and not val_seq_paths:
        logging.error("No se generaron secuencias de entrenamiento ni de validación. Revisa la ruta del dataset y su contenido.")
        return
        
    # 2. Limitar el número de secuencias para una prueba rápida (si está configurado)
    if config.get('max_sequences_to_use'):
        logging.info(f"Usando una muestra aleatoria de {config['max_sequences_to_use']} secuencias para esta ejecución.")
        # Mezclamos las listas para que la muestra sea variada
        random.shuffle(train_seq_paths)
        random.shuffle(val_seq_paths)
        
        num_train = int(config['max_sequences_to_use'] * config['train_val_split_ratio'])
        num_val = config['max_sequences_to_use'] - num_train
        
        train_seq_paths = train_seq_paths[:num_train]
        val_seq_paths = val_seq_paths[:num_val]
        logging.info(f"Muestra final -> Entrenamiento: {len(train_seq_paths)}, Validación: {len(val_seq_paths)}")

    # 3. Crear los Datasets y DataLoaders
    # El constructor de RadarDataset ahora es mucho más simple, solo necesita la lista de secuencias
    train_dataset = RadarDataset(train_seq_paths, seq_len=config['seq_len'], pred_len=config['pred_len'],
                                 min_dbz_norm=config['min_dbz'], max_dbz_norm=config['max_dbz'])

    val_dataset = RadarDataset(val_seq_paths, seq_len=config['seq_len'], pred_len=config['pred_len'],
                               min_dbz_norm=config['min_dbz'], max_dbz_norm=config['max_dbz'])

    # num_workers > 0 es ideal para acelerar la carga, pero puede dar problemas en algunos notebooks.
    # Si tienes errores, prueba poniendo num_workers=0.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    val_dataset_len = len(val_dataset) # Guardamos el largo para usarlo más adelante

    # 4. Verificar que tenemos datos para continuar
    if val_dataset_len == 0 and len(train_dataset) == 0:
        logging.error("Los datasets de entrenamiento y validación están vacíos. No se puede continuar.")
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