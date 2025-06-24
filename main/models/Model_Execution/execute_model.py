import os
import re
import time
from datetime import datetime, timedelta
import logging
import pyproj
import numpy as np
import torch
import torch.nn as nn
from netCDF4 import Dataset as NCDataset
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint


# --- Definición de las Clases del Modelo ---
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
    def __init__(self, input_dim, hidden_dim, kernel_size, use_layer_norm=True, img_size=(500,500), bias=True, return_all_layers=False):
        super(ConvLSTM2DLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.use_layer_norm = use_layer_norm
        self.img_size = img_size
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)
        
        # <<< LÓGICA DE LAYERNORM MOVIDA AQUÍ ADENTRO >>>
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm([hidden_dim, self.img_size[0], self.img_size[1]])

    def forward(self, input_tensor, hidden_state=None):
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
            layer_output = torch.stack(layer_output_list, dim=1) # (B, T, C_hidden, H, W)

            # <<< APLICAMOS LAYERNORM AQUÍ ADENTRO >>>
            if self.use_layer_norm:
                # LayerNorm espera (N, C, H, W) o similar, lo aplicamos a cada paso de tiempo
                B_ln, T_ln, C_ln, H_ln, W_ln = layer_output.shape
                # Reshape para aplicar LayerNorm a todos los frames a la vez
                output_reshaped_for_ln = layer_output.contiguous().view(B_ln * T_ln, C_ln, H_ln, W_ln)
                normalized_output = self.layer_norm(output_reshaped_for_ln)
                layer_output = normalized_output.view(B_ln, T_ln, C_ln, H_ln, W_ln)
        else:
            layer_output = h_cur.unsqueeze(1) # Si solo devolvemos el último, no normalizamos por ahora para simplificar

        return layer_output, (h_cur, c_cur)


class ConvLSTM3D_Enhanced(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[64, 64, 64], kernel_sizes=[(3,3), (3,3), (3,3)],
                 num_layers=3, pred_steps=1, use_layer_norm=True, use_residual=False,
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
        current_dim = self.input_dim

        for i in range(num_layers):
            # <<< INICIO DE LA CORRECCIÓN CLAVE >>>
            # Para todas las capas MENOS la última, devolvemos la secuencia completa.
            # Para la ÚLTIMA capa, devolvemos solo el estado final.
            is_last_layer = (i == num_layers - 1)
            self.layers.append(
                ConvLSTM2DLayer(
                    input_dim=current_dim, 
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_sizes[i],
                    use_layer_norm=use_layer_norm,
                    img_size=(img_height, img_width),
                    return_all_layers=not is_last_layer # Será True para todas menos la última
                )
            )
            # <<< FIN DE LA CORRECCIÓN CLAVE >>>
            current_dim = hidden_dims[i]

        self.output_conv = nn.Conv3d(
            in_channels=hidden_dims[-1],
            out_channels=self.input_dim * self.pred_steps,
            kernel_size=(1, 3, 3), 
            padding=(0, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
        
        logging.info(f"Modelo ConvLSTM3D_Enhanced creado: {num_layers} capas, Hidden dims: {hidden_dims}, LayerNorm: {use_layer_norm}, PredSteps: {pred_steps}")

    def forward(self, x_volumetric):
        num_z_levels, b, seq_len, h, w, c_in = x_volumetric.shape
        all_level_predictions = []

        for z_idx in range(num_z_levels):
            current_input = x_volumetric[z_idx, ...].permute(0, 1, 4, 2, 3)
            hidden_states_for_level = [None] * self.num_layers

            for i in range(self.num_layers):
                layer_input = current_input
                layer_output, hidden_state = checkpoint(
                    self.layers[i],
                    layer_input,
                    hidden_states_for_level[i],
                    use_reentrant=False
                )
                hidden_states_for_level[i] = hidden_state
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

# --- Funciones de Ayuda ---
def load_and_preprocess_input_sequence(input_file_paths, min_dbz, max_dbz, expected_shape, variable_name, seq_len):
    input_data_list = []
    expected_z, expected_h, expected_w = expected_shape
    if len(input_file_paths) != seq_len:
        raise ValueError(f"Se esperan {seq_len} archivos, se recibieron {len(input_file_paths)}")
    for file_path in input_file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        with NCDataset(file_path, 'r') as nc_file:
            dbz_var = nc_file.variables[variable_name]
            scale = getattr(dbz_var, 'scale_factor', 1.0)
            offset = getattr(dbz_var, 'add_offset', 0.0)
            fill_value = getattr(dbz_var, '_FillValue', None)
            dbz = dbz_var[0, ...].astype(np.float32)
            if dbz.shape != (expected_z, expected_h, expected_w):
                raise ValueError(f"Forma inesperada {dbz.shape} en {file_path}")
            dbz_physical = dbz * scale + offset
            if fill_value is not None:
                dbz_physical[dbz == fill_value] = np.nan
            dbz_clipped = np.clip(dbz_physical, min_dbz, max_dbz, out=np.full_like(dbz_physical, np.nan))
            dbz_normalized = np.where(np.isnan(dbz_clipped), np.nan,
                                     (dbz_clipped - min_dbz) / (max_dbz - min_dbz))
            dbz_normalized = dbz_normalized[..., np.newaxis]
            input_data_list.append(dbz_normalized)
    input_sequence_np = np.stack(input_data_list, axis=1)
    x_input_tensor = torch.from_numpy(np.nan_to_num(input_sequence_np, nan=0.0)).float()
    x_input_tensor = x_input_tensor.unsqueeze(0).permute(1, 0, 2, 3, 4, 5)
    return x_input_tensor

def save_prediction_as_netcdf(output_path, pred_data_desnorm, config_params, prediction_datetime):
    
    # Aplicar el umbral para eliminar el ruido de bajo nivel
    pred_data_desnorm[pred_data_desnorm < config_params.get('MIN_RELEVANT_DBZ', 5.0)] = np.nan
    
    # El valor de relleno debe ser un float32
    fill_value_float = np.float32(-999.0)

    # Reemplazar NaNs con el valor de relleno numérico antes de guardar
    pred_data_final = np.nan_to_num(pred_data_desnorm, nan=fill_value_float)
    
    # Añadir las dimensiones de batch y tiempo para el guardado
    pred_data_final_for_nc = np.expand_dims(pred_data_final, axis=0) # Solo batch, el tiempo ya viene en la predicción de 5 pasos
    
    # --- El resto de la lógica de creación de grilla y metadatos ---
    num_pred, num_z, height, width = pred_data_final_for_nc.shape
    
    x_coords = np.arange(-249.5, -249.5 + width * 1.0, 1.0, dtype=np.float32)
    y_coords = np.arange(-249.5, -249.5 + height * 1.0, 1.0, dtype=np.float32)
    z_coords = np.arange(1.0, 1.0 + num_z * 1.0, 1.0, dtype=np.float32)
    
    with NCDataset(output_path, 'w', format='NETCDF4') as ncfile: # Usar NETCDF4 para mejor manejo de fill_value
        ncfile.Conventions = "CF-1.6"
        ncfile.title = "ConvLSTM Nowcasting Prediction"
        # ... (puedes añadir más atributos globales si quieres) ...

        # --- Dimensiones (usando los nombres nuevos y correctos) ---
        ncfile.createDimension('time', None)
        ncfile.createDimension('altitude', num_z)
        ncfile.createDimension('latitude', height)
        ncfile.createDimension('longitude', width)

        # --- Variables de Coordenadas ---
        time_v = ncfile.createVariable('time', 'f8', ('time',))
        # ... (lógica para escribir el timestamp) ...
        
        alt_v = ncfile.createVariable('altitude', 'f4', ('altitude',))
        lat_v = ncfile.createVariable('latitude', 'f4', ('latitude',))
        lon_v = ncfile.createVariable('longitude', 'f4', ('longitude',))
        alt_v[:] = z_coords; lat_v[:] = y_coords; lon_v[:] = x_coords
        
        # --- Variable Principal DBZ (como float32) ---
        dbz_v = ncfile.createVariable('DBZ', 'f4', ('time', 'altitude', 'latitude', 'longitude'), 
                                      fill_value=fill_value_float)
        dbz_v.setncatts({
            'units': 'dBZ',
            'long_name': 'Predicted Reflectivity',
            '_FillValue': fill_value_float,
            'missing_value': fill_value_float
        })
        dbz_v[:] = pred_data_final_for_nc

    print(f"Predicción guardada en: {output_path}")

# --- Configuración Principal para Inferencia ---
if __name__ == "__main__":
    print("Iniciando script de inferencia local...")
    model_architecture_config = {
    'input_dim': 1,
    'hidden_dims': [128, 128, 128], # <-- CAMBIO CLAVE
    'kernel_sizes': [(3,3), (3,3), (3,3)],
    'num_layers': 3,                 # <-- CAMBIO CLAVE
    'pred_steps': 5,                 # <-- CAMBIO CLAVE (para el modelo final)
    'use_layer_norm': True,
    'use_residual': False,
    'img_height': 500,
    'img_width': 500
}
    data_params = {
        'min_dbz': -29.0,
        'max_dbz': 60.5,
        'expected_shape': (18, 500, 500),
        'variable_name': 'DBZ',
        'dbz_variable_name_pred': 'DBZ',
        'fill_value': -128,
        'grid_minz_km': 1.0,
        'grid_dz_km': 1.0,
        'grid_minx_km': -249.5,
        'grid_dx_km': 1.0,
        'grid_miny_km': -249.5,
        'grid_dy_km': 1.0,
        'sensor_longitude': -68.0169982910156,
        'sensor_latitude': -34.6479988098145,
        'earth_radius_m': 6378137,
        'institution_name': "UCAR",
        'data_source_name': "Gobierno de Mendoza",
        'prediction_interval_minutes': 3
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    model = ConvLSTM3D_Enhanced(
        input_dim=model_architecture_config['input_dim'],
        hidden_dims=model_architecture_config['hidden_dims'],
        kernel_sizes=model_architecture_config['kernel_sizes'],
        num_layers=model_architecture_config['num_layers'],
        pred_steps=model_architecture_config['pred_steps'],
        use_layer_norm=model_architecture_config['use_layer_norm'],
        use_residual=model_architecture_config['use_residual'],
        img_height=model_architecture_config['img_height'],
        img_width=model_architecture_config['img_width']
    )
    model_path = "/home/f-caballero/UM/TIF3/convLSTM-project/Modificaciones_modelo/Modelo_080625/New_model-11/model_epoch_1/checkpoint_epoch_2.pth"
    if not os.path.exists(model_path):
        print(f"Error: Archivo de modelo no encontrado en {model_path}")
        exit()
    try:
        checkpoint_data = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint_data['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Modelo cargado exitosamente desde: {model_path}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        exit()
    input_nc_files_for_prediction = [
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/185048.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/185653.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/185944.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/190234.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/190524.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/190815.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/191107.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/191354.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/191646.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/192821.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/193111.nc",
        "/home/f-caballero/UM/TIF3/MDV_para_25_50050018/netCDF/2007/classified_sliding_windows/20071129/2007112910/193402.nc",
    ]
    seq_len_entrenamiento = 12
    if len(input_nc_files_for_prediction) != seq_len_entrenamiento:
        print(f"Error: Se esperaban {seq_len_entrenamiento} archivos, pero se proporcionaron {len(input_nc_files_for_prediction)}.")
        exit()
    print(f"Cargando {len(input_nc_files_for_prediction)} archivos de entrada para la predicción...")
    try:
        x_input = load_and_preprocess_input_sequence(
            input_file_paths=input_nc_files_for_prediction,
            min_dbz=data_params['min_dbz'],
            max_dbz=data_params['max_dbz'],
            expected_shape=data_params['expected_shape'],
            variable_name=data_params['variable_name'],
            seq_len=seq_len_entrenamiento
        )
        x_input = x_input.to(device)
        print(f"Tensor de entrada preparado con forma: {x_input.shape}")
    except Exception as e:
        print(f"Error al preparar los datos de entrada: {e}")
        exit()
    print("Realizando predicción...")
    inference_start_time = time.time()
    with torch.no_grad():
        prediction_normalized = model(x_input)
    inference_end_time = time.time()
    print(f"Predicción completada en {inference_end_time - inference_start_time:.2f} segundos.")
    pred_data_np = prediction_normalized.squeeze(1).squeeze(1).squeeze(-1).cpu().numpy()
    dbz_predicted_desnormalized = pred_data_np * (data_params['max_dbz'] - data_params['min_dbz']) + data_params['min_dbz']
    print(f"Forma de la predicción desnormalizada (Z, H, W): {dbz_predicted_desnormalized.shape}")
    # Usar timestamp actual como placeholder
    current_prediction_datetime = datetime.now()
    output_nc_path = f"./prediction_output_{current_prediction_datetime.strftime('%Y%m%d_%H%M%S')}.nc"
    save_prediction_as_netcdf(output_path=output_nc_path,
                              pred_data_desnorm=dbz_predicted_desnormalized,
                              config_params=data_params,
                              prediction_datetime=current_prediction_datetime)
    layer_to_plot = 10
    if dbz_predicted_desnormalized.shape[0] > layer_to_plot:
        plt.figure(figsize=(8, 7))
        plt.imshow(dbz_predicted_desnormalized[layer_to_plot, :, :], origin='lower', cmap='jet', vmin=0, vmax=60)
        plt.title(f"Predicción (Capa Z={layer_to_plot}) - {current_prediction_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        plt.colorbar(label="dBZ")
        plt.tight_layout()
        plt.show()
    else:
        print(f"No se puede graficar la capa {layer_to_plot}, solo hay {dbz_predicted_desnormalized.shape[0]} capas.")
    print("Script de inferencia finalizado.")