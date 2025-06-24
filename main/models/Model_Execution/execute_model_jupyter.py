import os
import argparse
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
import torch.nn as nn
from netCDF4 import Dataset as NCDataset
import logging
import pyproj
import xarray as xr

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ==================================================================
# DEFINICIÓN DE LAS CLASES DEL MODELO
# (Esta es la arquitectura final y correcta que coincide con tu entrenamiento)
# ==================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True): # <--- BIAS AÑADIDO
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
                              bias=self.bias) # <--- BIAS AÑADIDO

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i); f = torch.sigmoid(cc_f); o = torch.sigmoid(cc_o); g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class ConvLSTM2DLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, use_layer_norm=True, img_size=(500,500), bias=True, return_all_layers=False): # <--- BIAS AÑADIDO
        super(ConvLSTM2DLayer, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.return_all_layers = return_all_layers
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias=bias) # <--- BIAS AÑADIDO
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm([hidden_dim, img_size[0], img_size[1]])

    def forward(self, input_tensor, hidden_state=None):
        b, seq_len, _, h, w = input_tensor.size()
        if hidden_state is None:
            hidden_state = self.cell.init_hidden(b, (h, w), input_tensor.device)
        
        output_list = []
        h_cur, c_cur = hidden_state
        for t in range(seq_len):
            h_cur, c_cur = self.cell(input_tensor=input_tensor[:, t, :, :, :], cur_state=[h_cur, c_cur])
            output_list.append(h_cur)
            
        if self.return_all_layers:
            layer_output = torch.stack(output_list, dim=1)
            if self.use_layer_norm:
                B, T, C, H, W = layer_output.shape
                output_reshaped = layer_output.contiguous().view(B * T, C, H, W)
                normalized_output = self.layer_norm(output_reshaped)
                layer_output = normalized_output.view(B, T, C, H, W)
        else:
            layer_output = h_cur.unsqueeze(1)

        return layer_output, (h_cur, c_cur)

class ConvLSTM3D_Enhanced(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, pred_steps, use_layer_norm, img_height, img_width):
        super(ConvLSTM3D_Enhanced, self).__init__()
        self.input_dim = input_dim
        self.pred_steps = pred_steps
        self.layers = nn.ModuleList()
        
        current_dim = self.input_dim
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            self.layers.append(
                ConvLSTM2DLayer(
                    input_dim=current_dim,
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_sizes[i],
                    use_layer_norm=use_layer_norm,
                    img_size=(img_height, img_width),
                    return_all_layers=not is_last_layer,
                    bias=True # Asumimos bias=True como en el original
                )
            )
            current_dim = hidden_dims[i]

        self.output_conv = nn.Conv3d(
            in_channels=hidden_dims[-1],
            out_channels=self.pred_steps * self.input_dim,
            kernel_size=(1, 3, 3), 
            padding=(0, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()
        
        # --- INICIALIZACIÓN DE PESOS AÑADIDA POR CONSISTENCIA ---
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x):
        b, seq_len, c, h, w = x.shape
        current_input = x
        hidden_states = [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            current_input, hidden_states[i] = layer(current_input, hidden_states[i])
        
        output_for_conv3d = current_input.permute(0, 2, 1, 3, 4)
        raw_conv_output = self.output_conv(output_for_conv3d)
        prediction_features = raw_conv_output.squeeze(2)
        predictions_norm = self.sigmoid(prediction_features.view(b, self.pred_steps, self.input_dim, h, w))
        
        return predictions_norm

# ==================================================================
# FUNCIONES DE AYUDA
# ==================================================================

def find_sequences(root_dir, seq_length):
    sequences = []
    logging.info(f"Buscando secuencias de longitud {seq_length} en: {root_dir}")
    if not os.path.isdir(root_dir):
        logging.error(f"El directorio raíz no existe: {root_dir}"); return sequences
    for seq_folder_name in sorted(os.listdir(root_dir)):
        seq_path = os.path.join(root_dir, seq_folder_name)
        if not os.path.isdir(seq_path): continue
        nc_files = sorted([os.path.join(seq_path, f) for f in os.listdir(seq_path) if f.endswith('.nc')])
        if len(nc_files) >= seq_length:
            sequences.append(nc_files)
    return sequences

def load_and_preprocess_input_sequence(input_file_paths, data_cfg):
    """
    Carga una secuencia de archivos NetCDF, los pre-procesa y los apila
    correctamente para el modelo de inferencia.
    """
    data_list = []
    min_dbz = data_cfg['min_dbz']
    max_dbz = data_cfg['max_dbz']

    for file_path in input_file_paths:
        try:
            # Usamos xarray, igual que en el script de entrenamiento, para manejar
            # automáticamente máscaras, offsets y escalar factores.
            with xr.open_dataset(file_path, mask_and_scale=True, decode_times=False) as ds:
                # .values extrae el array de numpy.
                # La forma esperada es (time=1, z, y, x)
                dbz_physical = ds[data_cfg['variable_name']].values

            # 1. Quitar la dimensión de tiempo (que es 1) para obtener (Z, H, W)
            dbz_physical_squeezed = dbz_physical[0, ...]

            # 2. Normalizar los datos
            dbz_clipped = np.clip(dbz_physical_squeezed, min_dbz, max_dbz)
            dbz_normalized = (dbz_clipped - min_dbz) / (max_dbz - min_dbz)
            
            # 3. Añadir la dimensión del canal al final -> (Z, H, W, C=1)
            # ¡OJO! No se añade la dimensión de tiempo aquí.
            data_list.append(dbz_normalized[..., np.newaxis])

        except Exception as e:
            logging.error(f"Error procesando archivo {file_path}: {e}")
            raise # Relanzamos el error para detener el procesamiento de esta secuencia

    # 4. Apilar a lo largo del eje de tiempo (axis=1) para crear la secuencia
    # El resultado tendrá el shape correcto: (Z, T, H, W, C)
    full_sequence = np.stack(data_list, axis=1)
    
    return torch.from_numpy(np.nan_to_num(full_sequence, nan=0.0)).float()

def save_prediction_as_netcdf(output_dir, pred_sequence_cleaned, data_cfg, start_datetime, seq_identifier):
    num_pred_steps, num_z, num_y, num_x = pred_sequence_cleaned.shape
    
    # --- Preparación de la Grilla (leído desde tu config) ---
    z_coords = np.arange(1.0, 1.0 + num_z * 1.0, 1.0, dtype=np.float32)
    x_coords = np.arange(-249.5, -249.5 + num_x * 1.0, 1.0, dtype=np.float32)
    y_coords = np.arange(-249.5, -249.5 + num_y * 1.0, 1.0, dtype=np.float32)
    
    proj = pyproj.Proj(proj="aeqd", lon_0=data_cfg['sensor_longitude'], lat_0=data_cfg['sensor_latitude'], R=data_cfg['earth_radius_m'])
    x_grid_m, y_grid_m = np.meshgrid(x_coords * 1000.0, y_coords * 1000.0)
    lon0_grid, lat0_grid = proj(x_grid_m, y_grid_m, inverse=True)

    for i in range(num_pred_steps):
        lead_time_minutes = (i + 1) * data_cfg.get('prediction_interval_minutes', 3)
        forecast_dt_utc = start_datetime + timedelta(minutes=lead_time_minutes)
        file_ts = forecast_dt_utc.strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_dir, f"{file_ts}.nc")

        with NCDataset(output_filename, 'w', format='NETCDF3_CLASSIC') as ds_out:
            # --- Atributos Globales ---
            ds_out.Conventions = "CF-1.6"
            ds_out.title = f"{data_cfg.get('radar_name', 'RADAR_PRED')} - Forecast t+{lead_time_minutes}min"
            ds_out.institution = data_cfg.get('institution_name', "Mi Institucion")
            ds_out.source = data_cfg.get('data_source_name', "ConvLSTM Model Prediction")
            ds_out.history = f"Created {datetime.now(timezone.utc).isoformat()} by ConvLSTM prediction script."
            ds_out.comment = f"Forecast data from model. Lead time: {lead_time_minutes} min."

            # --- Dimensiones (¡AQUÍ ESTÁ LA CLAVE!) ---
            ds_out.createDimension('time', None) # La dimensión de tiempo que faltaba
            ds_out.createDimension('bounds', 2)
            ds_out.createDimension('longitude', num_x)
            ds_out.createDimension('latitude', num_y)
            ds_out.createDimension('altitude', num_z)

            # --- Variables de Tiempo ---
            time_value = (forecast_dt_utc.replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds()
            
            time_v = ds_out.createVariable('time', 'f8', ('time',))
            time_v.standard_name = "time"; time_v.long_name = "Data time"
            time_v.units = "seconds since 1970-01-01T00:00:00Z"; time_v.axis = "T"
            time_v[:] = [time_value]

            # --- Variables de Coordenadas ---
            x_v = ds_out.createVariable('longitude', 'f4', ('longitude',)); x_v.setncatts({'standard_name':"projection_x_coordinate", 'units':"km", 'axis':"X"}); x_v[:] = x_coords
            y_v = ds_out.createVariable('latitude', 'f4', ('latitude',)); y_v.setncatts({'standard_name':"projection_y_coordinate", 'units':"km", 'axis':"Y"}); y_v[:] = y_coords
            z_v = ds_out.createVariable('altitude', 'f4', ('altitude',)); z_v.setncatts({'standard_name':"altitude", 'units':"km", 'axis':"Z", 'positive':"up"}); z_v[:] = z_coords
            
            # --- Variables de Georreferenciación ---
            lat0_v = ds_out.createVariable('lat0', 'f4', ('latitude', 'longitude',)); lat0_v.setncatts({'standard_name':"latitude", 'units':"degrees_north"}); lat0_v[:] = lat0_grid
            lon0_v = ds_out.createVariable('lon0', 'f4', ('latitude', 'longitude',)); lon0_v.setncatts({'standard_name':"longitude", 'units':"degrees_east"}); lon0_v[:] = lon0_grid
            gm_v = ds_out.createVariable('grid_mapping_0', 'i4'); gm_v.setncatts({'grid_mapping_name':"azimuthal_equidistant", 'longitude_of_projection_origin':data_cfg['sensor_longitude'], 'latitude_of_projection_origin':data_cfg['sensor_latitude'], 'false_easting':0.0, 'false_northing':0.0, 'earth_radius':data_cfg['earth_radius_m']})

            # --- Variable Principal DBZ (AHORA SIMPLIFICADA) ---
            fill_value_float = np.float32(-999.0)
            dbz_v = ds_out.createVariable('DBZ', 'f4', ('time', 'altitude', 'latitude', 'longitude'), fill_value=fill_value_float)
            dbz_v.setncatts({'units': 'dBZ', 'long_name': 'DBZ', 'standard_name': 'reflectivity', '_FillValue': fill_value_float, 'missing_value': fill_value_float})
            
            # Los datos ya vienen limpios, solo reemplazamos NaN por el valor de relleno
            pred_data_single_step = pred_sequence_cleaned[i]
            dbz_final_to_write = np.nan_to_num(pred_data_single_step, nan=fill_value_float)
            
            dbz_v[0, :, :, :] = dbz_final_to_write

        logging.info(f"  -> Predicción guardada en: {os.path.basename(output_filename)}")

# ==================================================================
# BLOQUE DE EJECUCIÓN PRINCIPAL
# ==================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de inferencia en lote para el modelo ConvLSTM.")
    parser.add_argument('--sequences_dir', type=str, required=True, help='Directorio raíz que contiene las carpetas de secuencias.')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al archivo .pth del modelo entrenado.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directorio donde se guardarán las predicciones.')
    parser.add_argument('--seq_len', type=int, default=12, help='Longitud de la secuencia de entrada.')
    args = parser.parse_args()

    model_config = {
        'input_dim': 1, 'hidden_dims': [128, 128, 128], 'kernel_sizes': [(3, 3), (3, 3), (3, 3)],
        'num_layers': 3, 'pred_steps': 5, 'use_layer_norm': True,
        'img_height': 500, 'img_width': 500
    }
    data_config = {
    'min_dbz': -29.0, 'max_dbz': 65.0, 'variable_name': 'DBZ',
    'prediction_interval_minutes': 3,
    
    # Umbral físico en dBZ. Las predicciones por debajo de este valor se eliminarán.
    'physical_threshold_dbz': 30.0, 

    # --- Parámetros de Georreferenciación y Metadatos ---
    'sensor_latitude': -34.64799880981445,
    'sensor_longitude': -68.01699829101562,
    'earth_radius_m': 6378137.0,
    'radar_name': 'SAN_RAFAEL_PRED',
    'institution_name': 'UM',
    'data_source_name': 'ConvLSTM Model Prediction'
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")
    model = ConvLSTM3D_Enhanced(**model_config)
    
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logging.info(f"Modelo cargado exitosamente desde: {args.model_path}")
    except Exception as e:
        logging.error(f"Error fatal al cargar el modelo: {e}", exc_info=True); exit()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences_to_process = find_sequences(args.sequences_dir, args.seq_len)
    
    if not sequences_to_process:
        logging.warning(f"No se encontraron secuencias válidas en {args.sequences_dir}"); exit()

logging.info(f"Se encontraron {len(sequences_to_process)} secuencias para procesar. Iniciando inferencia...")
    
# --- PARÁMETRO DE CONFIGURACIÓN PARA LA MEMORIA ---
# Reduce este número si sigues teniendo problemas de memoria.
# Valores buenos para probar: 6, 4, 2, 1.
z_batch_size = 2
logging.info(f"Procesando en lotes de Z (altura) de tamaño: {z_batch_size}")

for i, file_list in enumerate(sequences_to_process):
    seq_id = os.path.splitext(os.path.basename(file_list[-1]))[0]
    logging.info(f"--- Procesando Secuencia {i+1}/{len(sequences_to_process)} ({seq_id}) ---")
    try:
        input_volume = load_and_preprocess_input_sequence(file_list, data_config)
        x_to_model_full = input_volume.permute(0, 1, 4, 2, 3).to(device)
        
        num_z_levels = x_to_model_full.shape[0]
        all_predictions_chunks = []

        for z_start in range(0, num_z_levels, z_batch_size):
            z_end = min(z_start + z_batch_size, num_z_levels)
            x_chunk = x_to_model_full[z_start:z_end, ...]

            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                prediction_chunk = model(x_chunk)
            
            all_predictions_chunks.append(prediction_chunk.cpu())

        prediction_norm = torch.cat(all_predictions_chunks, dim=0)
        prediction_norm_physical_layout = prediction_norm.permute(1, 0, 2, 3, 4).squeeze(2)

        # --- INICIO DEL NUEVO BLOQUE DE POST-PROCESAMIENTO ---
        
        # 1. Desnormalizar a valores físicos (dBZ)
        pred_physical_raw = prediction_norm_physical_layout.numpy() * (data_config['max_dbz'] - data_config['min_dbz']) + data_config['min_dbz']

        # 2. Recortar (Clip) al rango válido para evitar valores extraños
        pred_physical_clipped = np.clip(pred_physical_raw, data_config['min_dbz'], data_config['max_dbz'])
        
        # 3. Aplicar el umbral físico para limpieza (la clave del problema)
        pred_physical_cleaned = pred_physical_clipped.copy()
        threshold = data_config.get('physical_threshold_dbz', 30.0) # Usa 30.0 si no está definido
        pred_physical_cleaned[pred_physical_cleaned < threshold] = np.nan
        
        # --- FIN DEL NUEVO BLOQUE DE POST-PROCESAMIENTO ---

        try:
            last_input_filepath = file_list[-1]
            parts = last_input_filepath.split('/')
            date_str = parts[-2][:8]
            time_str = os.path.splitext(parts[-1])[0]
            last_input_dt_utc = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        except Exception as e_time:
            logging.warning(f"No se pudo parsear el timestamp. Usando datetime.now(). Error: {e_time}")
            last_input_dt_utc = datetime.now()

        # Pasamos el array ya limpio a la función de guardado
        save_prediction_as_netcdf(
            output_dir=args.output_dir,
            pred_sequence_cleaned=pred_physical_cleaned, # <-- Pasamos el array limpio
            data_cfg=data_config,
            start_datetime=last_input_dt_utc,
            seq_identifier=seq_id
        )

        del input_volume, x_to_model_full, prediction_norm, pred_physical_cleaned
        torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Error procesando la secuencia {seq_id}: {e}", exc_info=True)
        torch.cuda.empty_cache() 
        continue

logging.info("Proceso completado.")

