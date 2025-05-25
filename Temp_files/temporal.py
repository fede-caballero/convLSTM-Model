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
    """Configura las semillas para reproducibilidad."""
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
                 min_dbz=-30, max_dbz=70,
                 expected_shape=(18, 500, 500), variable_name='DBZ'):
        self.data_dir = data_dir
        self.subdirs_list = subdirs_list
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.min_dbz = min_dbz
        self.max_dbz = max_dbz
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
                logging.warning(f"Subdirectorio {subdir_name} no encontrado en {self.data_dir}. Omitiendo.")
                continue
            if ".ipynb_checkpoints" in subdir_name:
                logging.debug(f"Omitiendo directorio de checkpoints: {subdir_name}")
                continue
            files = sorted(glob.glob(os.path.join(subdir_path, "*.nc")))
            if len(files) >= self.seq_len + self.pred_len:
                valid_sequences.append((files, subdir_name))
            else:
                logging.warning(f"Subdirectorio {subdir_name} tiene {len(files)} archivos, necesita {self.seq_len + self.pred_len}. Omitiendo.")
        return valid_sequences

    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        sequence_files, subdir_name = self.valid_sequences[idx]
        input_data_list = []
        output_data_list = []

        for i in range(self.seq_len):
            file_path = sequence_files[i]
            try:
                with NCDataset(file_path, 'r') as nc_file:
                    dbz = nc_file.variables[self.variable_name][0, ...].astype(np.float32)
                    if dbz.shape != (self.expected_z, self.expected_h, self.expected_w):
                        logging.warning(f"Forma inesperada {dbz.shape} en {file_path} para {subdir_name}. Omitiendo muestra.")
                        return self.__getitem__((idx + 1) % len(self))
                    dbz = np.clip(dbz, self.min_dbz, self.max_dbz)
                    dbz = (dbz - self.min_dbz) / (self.max_dbz - self.min_dbz)
                    dbz = dbz[..., np.newaxis]
                    input_data_list.append(dbz)
            except Exception as e:
                logging.error(f"Error cargando entrada {file_path} para {subdir_name}: {e}. Omitiendo muestra.")
                return self.__getitem__((idx + 1) % len(self))

        for i in range(self.seq_len, self.seq_len + self.pred_len):
            file_path = sequence_files[i]
            try:
                with NCDataset(file_path, 'r') as nc_file:
                    dbz = nc_file.variables[self.variable_name][0, ...].astype(np.float32)
                    if dbz.shape != (self.expected_z, self.expected_h, self.expected_w):
                        logging.warning(f"Forma inesperada {dbz.shape} en {file_path} para {subdir_name}. Omitiendo muestra.")
                        return self.__getitem__((idx + 1) % len(self))
                    dbz = np.clip(dbz, self.min_dbz, self.max_dbz)
                    dbz = (dbz - self.min_dbz) / (self.max_dbz - self.min_dbz)
                    dbz = dbz[..., np.newaxis]
                    output_data_list.append(dbz)
            except Exception as e:
                logging.error(f"Error cargando salida {file_path} para {subdir_name}: {e}. Omitiendo muestra.")
                return self.__getitem__((idx + 1) % len(self))

        if len(input_data_list) != self.seq_len or len(output_data_list) != self.pred_len:
            logging.warning(f"No se pudieron cargar suficientes archivos para la secuencia {idx} en {subdir_name}. Omitiendo.")
            return self.__getitem__((idx + 1) % len(self))

        input_tensor = np.stack(input_data_list, axis=1)
        output_tensor = np.stack(output_data_list, axis=1)
        x = torch.from_numpy(input_tensor).float()
        y = torch.from_numpy(output_tensor).float()
        return x, y


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
        b, seq_len, _, h, w = input_tensor.size()
        device = input_tensor.device
        if hidden_state is None:
            hidden_state = self.cell.init_hidden(b, (h, w), device)
        layer_output_list = []
        h_cur, c_cur = hidden_state
        for t in range(seq_len):
            h_cur, c_cur = self.cell(input_tensor=input_tensor[:, t, :, :, :], cur_state=[h_cur, c_cur])
            layer_output_list.append(h_cur)
        if self.return_all_layers:
            layer_output = torch.stack(layer_output_list, dim=1)
        else:
            layer_output = layer_output_list[-1].unsqueeze(1)
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

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        current_dim = input_dim
        for i in range(num_layers):
            self.layers.append(
                ConvLSTM2DLayer(input_dim=current_dim, hidden_dim=hidden_dims[i],
                                kernel_size=kernel_sizes[i], bias=True, return_all_layers=True)
            )
            if use_layer_norm:
                 self.layer_norms.append(nn.LayerNorm([hidden_dims[i], img_height, img_width]))
            current_dim = hidden_dims[i]

        self.output_conv = nn.Conv3d(in_channels=hidden_dims[-1],
                                     out_channels=input_dim * pred_steps, # Esto será self.input_dim si pred_steps=1
                                     kernel_size=(1, 3, 3), padding=(0, 1, 1))
        logging.info(f"Modelo ConvLSTM3D_Enhanced creado: {num_layers} capas, Hidden dims: {hidden_dims}, LayerNorm: {use_layer_norm}, Residual: {use_residual}, PredSteps: {pred_steps}")

    def forward(self, x_volumetric): # Espera (Z, B, T_in, H, W, C_in)
        num_z_levels, b, seq_len, h, w, c_in = x_volumetric.shape
        all_level_predictions = []

        for z_idx in range(num_z_levels):
            x_level = x_volumetric[z_idx, ...] # (B, T_in, H, W, C_in)
            x_level_permuted = x_level.permute(0, 1, 4, 2, 3) # (B, T_in, C_in, H, W)
            current_input = x_level_permuted
            # residual_input = None # Lógica residual simplificada/omitida por ahora para claridad

            for i in range(self.num_layers):
                # if self.use_residual and i > 0:
                #     residual_input = current_input # Necesitaría lógica de proyección si canales cambian
                
                if self.training: # Aplicar checkpointing solo durante el entrenamiento
                    # El tercer argumento para checkpoint es el hidden_state para ConvLSTM2DLayer.forward
                    # Como nuestra capa lo inicializa si es None, podemos pasar None.
                    layer_output, _ = checkpoint(self.layers[i], current_input, None, use_reentrant=False)
                else:
                    layer_output, _ = self.layers[i](current_input)

                if self.use_layer_norm and self.layer_norms:
                    B_ln, T_ln, C_ln, H_ln, W_ln = layer_output.shape
                    output_reshaped_for_ln = layer_output.contiguous().view(B_ln * T_ln, C_ln, H_ln, W_ln)
                    normalized_output = self.layer_norms[i](output_reshaped_for_ln)
                    layer_output = normalized_output.view(B_ln, T_ln, C_ln, H_ln, W_ln)

                # if self.use_residual and residual_input is not None:
                #     if residual_input.shape == layer_output.shape: # Simplificado
                #         layer_output = layer_output + residual_input
                current_input = layer_output
            
            # current_input ahora es (B, T_in, C_last_hidden, H, W)
            output_for_conv3d = current_input.permute(0, 2, 1, 3, 4) # (B, C_last_hidden, T_in, H, W)
            
            # raw_conv_output tendrá forma (B, self.input_dim*self.pred_steps, T_in_out, H, W)
            # donde T_in_out = T_in debido a kernel_size[0]=1, padding[0]=0 en Conv3d
            # Ejemplo: (1, 1*1, 3, 500, 500) si T_in=3, pred_steps=1, input_dim=1
            raw_conv_output = self.output_conv(output_for_conv3d)

            # Seleccionar el último paso temporal de la salida de Conv3D
            # Esto asume que la predicción para t+1 se basa en la salida en el último paso t de la secuencia de entrada.
            prediction_at_final_step = raw_conv_output[:, :, -1, :, :] # Forma: (B, self.input_dim*self.pred_steps, H, W)
                                                                      # Ejemplo: (1, 1, 500, 500)

            # Remodelar para que coincida con (B, self.pred_steps, self.input_dim, H, W)
            # prediction_at_final_step es (B, C_total_out, H, W) donde C_total_out = self.input_dim * self.pred_steps
            level_prediction = prediction_at_final_step.view(b, self.pred_steps, self.input_dim, h, w)
            
            # Permutar a la forma de salida deseada para apilar sobre Z
            # (B, pred_steps, H, W, C_out=self.input_dim)
            level_prediction = level_prediction.permute(0, 1, 3, 4, 2)
            all_level_predictions.append(level_prediction)

        predictions_volumetric = torch.stack(all_level_predictions, dim=0) # (Z, B, pred_steps, H, W, C_out)
        return predictions_volumetric

class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, kernel_size_for_metric=7):
        super(SSIMLoss, self).__init__()
        try:
            self.ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(
                data_range=data_range, 
                kernel_size=kernel_size_for_metric
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self._has_reduction = hasattr(self.ssim_metric, 'reduction') and self.ssim_metric.reduction in ['elementwise_mean', 'sum']
        except TypeError: 
             self.ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(
                data_range=data_range, 
                kernel_size=kernel_size_for_metric,
                reduction='elementwise_mean' 
            ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
             self._has_reduction = True


    def forward(self, img1, img2):
        num_z, batch_s, pred_t, height, width, channels = img1.shape
        if pred_t != 1 or channels != 1:
            logging.debug(f"SSIMLoss: T_pred={pred_t}, channels={channels}. Se aplanarán estas dimensiones para SSIM.")
        
        img1_reshaped = img1.permute(0, 1, 2, 5, 3, 4).contiguous().view(-1, channels, height, width)
        img2_reshaped = img2.permute(0, 1, 2, 5, 3, 4).contiguous().view(-1, channels, height, width)
        
        ssim_val = self.ssim_metric(img1_reshaped, img2_reshaped)
        if not self._has_reduction and ssim_val.ndim > 0 and ssim_val.numel() > 1:
            ssim_val = ssim_val.mean()
        return 1.0 - ssim_val


def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 1e-5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.get('lr_patience', 3), verbose=True)
    
    criterion_mse = nn.MSELoss().to(device)
    criterion_ssim = None
    ssim_loss_weight = 0.0
    mse_loss_weight = 1.0

    if config.get('use_ssim_loss', False):
        try:
            criterion_ssim = SSIMLoss(
                data_range=1.0, 
                kernel_size_for_metric=config.get('ssim_kernel_size', 7)
            ).to(device)
            ssim_loss_weight = config.get('ssim_loss_weight', 0.3)
            mse_loss_weight = 1.0 - ssim_loss_weight
            logging.info(f"Usando SSIM loss con peso {ssim_loss_weight} y MSE con peso {mse_loss_weight}")
        except Exception as e:
            logging.error(f"Error al inicializar SSIMLoss: {e}. Se usará solo MSE.")
            criterion_ssim = None
            ssim_loss_weight = 0.0
            mse_loss_weight = 1.0

    scaler = torch.amp.GradScaler(enabled=config['use_amp'])
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    accumulation_steps = config.get('accumulation_steps', 1)

    logging.info(f"Iniciando entrenamiento: {config['epochs']} épocas, LR: {config['learning_rate']}, Batch (efectivo): {config['batch_size'] * accumulation_steps}")

    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            if x.dim() == 6 and y.dim() == 6:
                x = x.permute(1, 0, 2, 3, 4, 5) 
                y = y.permute(1, 0, 2, 3, 4, 5)
            else:
                logging.error(f"Formas inesperadas para x o y antes de la permutación: x={x.shape}, y={y.shape}")
                continue

            if batch_idx == 0 and epoch == 0:
                logging.info(f"Forma de entrada al modelo (después de permutar): {x.shape}")
                logging.info(f"Forma objetivo (después de permutar): {y.shape}")

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config['use_amp']):
                predictions = model(x)
                loss_mse_val = criterion_mse(predictions, y)
                current_loss = loss_mse_val

                if criterion_ssim is not None:
                    loss_ssim_component = criterion_ssim(predictions, y)
                    current_loss = mse_loss_weight * loss_mse_val + ssim_loss_weight * loss_ssim_component
                
                loss = current_loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                if config.get('clip_grad_norm', None):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['clip_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_train_loss += loss.item() * accumulation_steps
            if (batch_idx + 1) % config.get('log_interval', 20) == 0:
                logging.info(f"Época {epoch+1}/{config['epochs']} [{batch_idx+1}/{len(train_loader)}] - Pérdida: {loss.item() * accumulation_steps:.6f}")
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        if val_loader:
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
                    
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config['use_amp']):
                        predictions_val = model(x_val)
                        val_loss_mse_val = criterion_mse(predictions_val, y_val)
                        current_val_loss = val_loss_mse_val
                        if criterion_ssim is not None:
                            val_loss_ssim_component = criterion_ssim(predictions_val, y_val)
                            current_val_loss = mse_loss_weight * val_loss_mse_val + ssim_loss_weight * val_loss_ssim_component
                    running_val_loss += current_val_loss.item()
            
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
        else:
            epoch_duration = time.time() - epoch_start_time
            logging.info(f"Época {epoch+1} completada en {epoch_duration:.2f}s. Pérdida (train): {avg_train_loss:.6f} (No hay datos de validación)")

        if (epoch + 1) % config.get('checkpoint_interval', 5) == 0:
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'train_losses': train_losses,
                        'val_losses': val_losses if val_loader else []},
                       os.path.join(config['model_save_dir'], f"checkpoint_epoch_{epoch+1}.pth"))
            logging.info(f"Checkpoint guardado en la época {epoch+1}")

    logging.info("Entrenamiento finalizado.")
    if train_loader: # Solo plotear si hubo entrenamiento
        plt.figure(figsize=(10,5))
        plt.plot(train_losses, label='Pérdida Entrenamiento')
        if val_loader:
            plt.plot(val_losses, label='Pérdida Validación')
        plt.xlabel('Épocas'); plt.ylabel('Pérdida'); plt.legend()
        plt.savefig(os.path.join(config['model_save_dir'], "loss_curves.png")); plt.close()
    return model, {'train_losses': train_losses, 'val_losses': val_losses if val_loader else []}

def generate_prediction_netcdf(model, data_loader, config, device, num_samples=1):
    output_dir = config['predictions_output_dir']
    min_dbz = config['min_dbz']
    max_dbz = config['max_dbz']
    variable_name = config.get('dbz_variable_name_pred', 'DBZ_predicted')

    sensor_latitude = config.get('sensor_latitude', -34.64799880981445)
    sensor_longitude = config.get('sensor_longitude', -68.01699829101562)
    sensor_altitude_km = config.get('sensor_altitude_km', 0.550000011920929)
    grid_minz_km = config.get('grid_minz_km', 1.0)
    grid_dz_km = config.get('grid_dz_km', 0.5)
    grid_minx_km = config.get('grid_minx_km', -249.75)
    grid_dx_km = config.get('grid_dx_km', 0.5)
    grid_miny_km = config.get('grid_miny_km', -249.75)
    grid_dy_km = config.get('grid_dy_km', 0.5)
    radar_name = config.get('radar_name', "La Llave")
    institution_name = config.get('institution_name', "Tu Institucion/Universidad")
    data_source_name = config.get('data_source_name', "Gobierno de Mendoza")
    projection_var_name_in_file = config.get('projection_variable_name', "radar_projection_info")

    model.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_z_output = config['expected_shape'][0]
    height = config['expected_shape'][1]
    width = config['expected_shape'][2]

    z_coord_values = np.arange(grid_minz_km, grid_minz_km + num_z_output * grid_dz_km, grid_dz_km)[:num_z_output]
    x_coord_values = np.arange(grid_minx_km, grid_minx_km + width * grid_dx_km, grid_dx_km)[:width]
    y_coord_values = np.arange(grid_miny_km, grid_miny_km + height * grid_dy_km, grid_dy_km)[:height]
    
    base_dt_for_samples = datetime.utcnow()

    with torch.no_grad():
        for i, (x_input_volume, y_true_volume) in enumerate(data_loader):
            if i >= num_samples: break

            current_pred_datetime = base_dt_for_samples + timedelta(minutes=i * config.get('prediction_interval_minutes', 5))
            
            x_to_model = x_input_volume.to(device)
            if x_to_model.dim() == 6:
                 x_to_model = x_to_model.permute(1, 0, 2, 3, 4, 5)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=config['use_amp']):
                prediction_norm = model(x_to_model) # Espera (Z,B,T,H,W,C) -> Sale (Z,B,T_pred,H,W,C)

            # prediction_norm es (Z, B=1, T_pred=1, H, W, C=1)
            # Quitar B, T_pred, C para obtener (Z, H, W)
            pred_data_np = prediction_norm.squeeze(1).squeeze(1).squeeze(-1).cpu().numpy() 
            pred_data_desnorm = pred_data_np * (max_dbz - min_dbz) + min_dbz
            pred_data_final_for_nc = np.expand_dims(pred_data_desnorm, axis=0) # (1=time, Z, H, W)

            file_timestamp_str = current_pred_datetime.strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(output_dir, f"pred_{variable_name}_{file_timestamp_str}_sample{i}.nc")

            with NCDataset(output_filename, 'w', format='NETCDF4') as ncfile:
                ncfile.Conventions = "CF-1.7"
                ncfile.title = f"Radar Reflectivity Forecast ({variable_name}) from ConvLSTM Model"
                ncfile.institution = institution_name
                ncfile.source_data_description = f"Based on input data from {data_source_name}, Radar: {radar_name}."
                ncfile.source_model_description = "ConvLSTM neural network prediction."
                ncfile.history = f"Created {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} by prediction script."
                ncfile.comment = "Model-generated forecast. Not for operational use without verification."
                ncfile.radar_name = radar_name; ncfile.sensor_latitude = sensor_latitude
                ncfile.sensor_longitude = sensor_longitude; ncfile.sensor_altitude = sensor_altitude_km
                ncfile.references = f"Tesis de {config.get('author_name', '[Tu Nombre]')}, {config.get('author_institution', '[Tu Universidad/Institucion]')}"

                ncfile.createDimension('time', 1); ncfile.createDimension('level', num_z_output)
                ncfile.createDimension('y', height); ncfile.createDimension('x', width)

                time_var = ncfile.createVariable('time', 'f8', ('time',))
                epoch_time = datetime(1970, 1, 1, 0, 0, 0)
                time_value_seconds = (current_pred_datetime.replace(tzinfo=None) - epoch_time).total_seconds()
                time_var[:] = [time_value_seconds]
                time_var.units = "seconds since 1970-01-01 00:00:00 UTC"; time_var.calendar = "gregorian"
                time_var.long_name = "time of forecast"; time_var.standard_name = "time"; time_var.axis = "T"

                z_coord = ncfile.createVariable('level', 'f4', ('level',))
                z_coord[:] = z_coord_values
                z_coord.units = "km"; z_coord.positive = "up"; z_coord.long_name = "altitude"; z_coord.standard_name = "altitude"; z_coord.axis = "Z"

                x_coord = ncfile.createVariable('x', 'f4', ('x',))
                x_coord[:] = x_coord_values
                x_coord.units = "km"; x_coord.long_name = "projection_x_coordinate"; x_coord.standard_name = "projection_x_coordinate"; x_coord.axis = "X"

                y_coord = ncfile.createVariable('y', 'f4', ('y',))
                y_coord[:] = y_coord_values
                y_coord.units = "km"; y_coord.long_name = "projection_y_coordinate"; y_coord.standard_name = "projection_y_coordinate"; y_coord.axis = "Y"
                
                projection_var = ncfile.createVariable(projection_var_name_in_file, 'i4')
                projection_var.grid_mapping_name = "lambert_azimuthal_equal_area"
                projection_var.longitude_of_projection_origin = sensor_longitude
                projection_var.latitude_of_projection_origin = sensor_latitude
                projection_var.false_easting = 0.0; projection_var.false_northing = 0.0

                pred_dbz_var = ncfile.createVariable(variable_name, 'f4', ('time', 'level', 'y', 'x'),
                                                 fill_value=np.float32(config.get('fill_value', -9999.0)))
                pred_dbz_var.units = 'dBZ'; pred_dbz_var.long_name = 'Predicted Radar Reflectivity'
                pred_dbz_var.coordinates = "time level y x"; pred_dbz_var.grid_mapping = projection_var_name_in_file
                pred_dbz_var[:] = pred_data_final_for_nc
            logging.info(f"Predicción de muestra con metadatos CF guardada en: {output_filename}")

def main():
    # Celda 1 del Notebook:
    # import os
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # print(f"PYTORCH_CUDA_ALLOC_CONF set to: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    # Asegúrate de REINICIAR EL KERNEL después de ejecutar esto por primera vez.

    set_seed(42)
    config = {
        'data_dir': "/home/Big-Sample-Livianos-757-carpetas",
        'model_save_dir': "/home/model_output_final_v_ckpt", 
        'predictions_output_dir': "/home/predictions_final_v_ckpt",
        
        'seq_len': 6,
        'pred_len': 1, 'pred_steps_model': 1,
        'min_dbz': -30.0, 'max_dbz': 70.0, 'fill_value': -9999.0,
        'expected_shape': (18, 500, 500),
        'dbz_variable_name': 'DBZ', 'dbz_variable_name_pred': 'DBZ_forecast',

        'sensor_latitude': -34.64799880981445, 'sensor_longitude': -68.01699829101562,
        'sensor_altitude_km': 0.550000011920929, 'grid_minz_km': 1.0, 'grid_dz_km': 0.5,
        'grid_minx_km': -249.75, 'grid_dx_km': 0.5, 'grid_miny_km': -249.75, 'grid_dy_km': 0.5,
        'radar_name': "La Llave", 'institution_name': "Universidad de Mendoza - Federico Caballero",
        'author_name': "Federico Caballero", 'author_institution': "Universidad de Mendoza",
        'data_source_name': "Gobierno de Mendoza", 'projection_variable_name': "lambert_azimuthal_projection",
        'prediction_interval_minutes': 5,

        'model_input_dim': 1,
        'model_hidden_dims': [32, 32], # REDUCIDO
        'model_kernel_sizes': [(3,3), (3,3)], # AJUSTADO
        'model_num_layers': 2, # REDUCIDO
        'model_use_layer_norm': True, 'model_use_residual': False,

        'batch_size': 1, 'epochs': 10, 
        'learning_rate': 1e-4, 'weight_decay': 1e-5, 'lr_patience': 3, # Reducido patience para prueba rápida
        'use_amp': True, 'accumulation_steps': 1, 
        'clip_grad_norm': 1.0, 'log_interval': 1, 
        'checkpoint_interval': 1,

        'use_ssim_loss': True, 'ssim_kernel_size': 7, 'ssim_loss_weight': 0.3,

        'train_val_split_ratio': 0.8,
        'max_sequences_to_use': None, 
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
    if not train_subdirs: logging.error("No hay secuencias de entrenamiento."); return
    logging.info(f"Entrenamiento: {len(train_subdirs)} sec. Validación: {len(val_subdirs)} sec.")

    train_dataset = RadarDataset(config['data_dir'], train_subdirs, seq_len=config['seq_len'], pred_len=config['pred_len'],
                                 min_dbz=config['min_dbz'], max_dbz=config['max_dbz'],
                                 expected_shape=config['expected_shape'], variable_name=config['dbz_variable_name'])
    val_loader = None; val_dataset_len = 0
    if val_subdirs:
        val_dataset = RadarDataset(config['data_dir'], val_subdirs, seq_len=config['seq_len'], pred_len=config['pred_len'],
                                   min_dbz=config['min_dbz'], max_dbz=config['max_dbz'],
                                   expected_shape=config['expected_shape'], variable_name=config['dbz_variable_name'])
        val_dataset_len = len(val_dataset)
        if val_dataset_len > 0:
             val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        else: logging.info("Dataset de validación vacío.")
    else: logging.info("No subdirectorios para validación.")

    if len(train_dataset) == 0: logging.error("Dataset de entrenamiento vacío."); return
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    
    model = ConvLSTM3D_Enhanced(
        input_dim=config['model_input_dim'], hidden_dims=config['model_hidden_dims'],
        kernel_sizes=config['model_kernel_sizes'], num_layers=config['model_num_layers'],
        pred_steps=config['pred_steps_model'], use_layer_norm=config['model_use_layer_norm'],
        use_residual=config['model_use_residual'],
        img_height=config['expected_shape'][1], img_width=config['expected_shape'][2]
    )
    logging.info(f"Arquitectura del modelo:\n{model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Número total de parámetros entrenables: {total_params:,}")

    trained_model, history = train_model(model, train_loader, val_loader, config)

    if val_loader:
        logging.info("Generando predicciones de ejemplo usando el conjunto de validación...")
        generate_prediction_netcdf(trained_model, val_loader, config,
                                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                   num_samples=min(5, val_dataset_len))
    else:
        logging.info("No val_loader, se omiten predicciones de ejemplo.")
    logging.info("Proceso completado.")

if __name__ == '__main__':
    main()