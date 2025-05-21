import os
import glob
import time
from datetime import datetime, timedelta
import logging

import numpy as np
import torch
import torch.nn as nn
from netCDF4 import Dataset as NCDataset
import matplotlib.pyplot as plt
# No necesitas torch.amp para inferencia en CPU generalmente, ni checkpoint
# from torch.utils.checkpoint import checkpoint

# --- Definición de las Clases del Modelo ---
# (Copia aquí las clases: ConvLSTMCell, ConvLSTM2DLayer, ConvLSTM3D_Enhanced
#  tal como están en tu script de entrenamiento final que funcionó)

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
        i = torch.sigmoid(cc_i); f = torch.sigmoid(cc_f); o = torch.sigmoid(cc_o); g = torch.tanh(cc_g)
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
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.kernel_size = kernel_size
        self.bias = bias; self.return_all_layers = return_all_layers
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)
    def forward(self, input_tensor, hidden_state=None):
        b, seq_len, _, h, w = input_tensor.size()
        device = input_tensor.device
        if hidden_state is None: hidden_state = self.cell.init_hidden(b, (h, w), device)
        layer_output_list = []
        h_cur, c_cur = hidden_state
        for t in range(seq_len):
            h_cur, c_cur = self.cell(input_tensor=input_tensor[:, t, :, :, :], cur_state=[h_cur, c_cur])
            layer_output_list.append(h_cur)
        if self.return_all_layers: layer_output = torch.stack(layer_output_list, dim=1)
        else: layer_output = layer_output_list[-1].unsqueeze(1)
        return layer_output, (h_cur, c_cur)

class ConvLSTM3D_Enhanced(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[32, 64], kernel_sizes=[(3,3), (3,3)],
                 num_layers=2, pred_steps=1, use_layer_norm=True, use_residual=False,
                 img_height=500, img_width=500):
        super(ConvLSTM3D_Enhanced, self).__init__()
        if isinstance(hidden_dims, int): hidden_dims = [hidden_dims] * num_layers
        if isinstance(kernel_sizes, tuple): kernel_sizes = [kernel_sizes] * num_layers
        assert len(hidden_dims) == num_layers and len(kernel_sizes) == num_layers

        self.input_dim = input_dim; self.hidden_dims = hidden_dims; self.num_layers = num_layers
        self.pred_steps = pred_steps; self.use_layer_norm = use_layer_norm; self.use_residual = use_residual
        self.layers = nn.ModuleList(); self.layer_norms = nn.ModuleList() if use_layer_norm else None
        current_dim = input_dim
        for i in range(num_layers):
            self.layers.append(
                ConvLSTM2DLayer(input_dim=current_dim, hidden_dim=hidden_dims[i],
                                kernel_size=kernel_sizes[i], bias=True, return_all_layers=True)
            )
            if use_layer_norm: self.layer_norms.append(nn.LayerNorm([hidden_dims[i], img_height, img_width]))
            current_dim = hidden_dims[i]
        self.output_conv = nn.Conv3d(in_channels=hidden_dims[-1], out_channels=input_dim * pred_steps,
                                     kernel_size=(1, 3, 3), padding=(0, 1, 1))
        # No necesitas logging.info aquí para un script de inferencia simple

    def forward(self, x_volumetric): # Espera (Z, B, T_in, H, W, C_in)
        num_z_levels, b, seq_len, h, w, c_in = x_volumetric.shape
        all_level_predictions = []
        for z_idx in range(num_z_levels):
            x_level = x_volumetric[z_idx, ...]; x_level_permuted = x_level.permute(0, 1, 4, 2, 3)
            current_input = x_level_permuted
            for i in range(self.num_layers):
                # No se usa checkpointing durante model.eval()
                layer_output, _ = self.layers[i](current_input)
                if self.use_layer_norm and self.layer_norms:
                    B_ln, T_ln, C_ln, H_ln, W_ln = layer_output.shape
                    output_reshaped_for_ln = layer_output.contiguous().view(B_ln * T_ln, C_ln, H_ln, W_ln)
                    normalized_output = self.layer_norms[i](output_reshaped_for_ln)
                    layer_output = normalized_output.view(B_ln, T_ln, C_ln, H_ln, W_ln)
                current_input = layer_output
            output_for_conv3d = current_input.permute(0, 2, 1, 3, 4)
            raw_conv_output = self.output_conv(output_for_conv3d)
            prediction_at_final_step = raw_conv_output[:, :, -1, :, :]
            if self.pred_steps == 1 and prediction_at_final_step.shape[1] == self.input_dim:
                level_prediction = prediction_at_final_step.unsqueeze(1)
            else:
                level_prediction = prediction_at_final_step.view(b, self.pred_steps, self.input_dim, h, w)
            level_prediction = level_prediction.permute(0, 1, 3, 4, 2)
            all_level_predictions.append(level_prediction)
        predictions_volumetric = torch.stack(all_level_predictions, dim=0)
        return predictions_volumetric

# --- Funciones de Ayuda ---
def load_and_preprocess_input_sequence(input_file_paths, min_dbz, max_dbz, expected_shape, variable_name):
    """Carga 6 archivos NetCDF, los preprocesa y los apila."""
    input_data_list = []
    expected_z, expected_h, expected_w = expected_shape

    if len(input_file_paths) != 6: # O la seq_len que usaste para entrenar
        raise ValueError(f"Se esperan 6 archivos de entrada, se recibieron {len(input_file_paths)}")

    for file_path in input_file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo de entrada no encontrado: {file_path}")
        with NCDataset(file_path, 'r') as nc_file:
            dbz = nc_file.variables[variable_name][0, ...].astype(np.float32)
            if dbz.shape != (expected_z, expected_h, expected_w):
                raise ValueError(f"Forma inesperada {dbz.shape} en {file_path}. Se esperaba {(expected_z, expected_h, expected_w)}")
            dbz = np.clip(dbz, min_dbz, max_dbz)
            dbz_normalized = (dbz - min_dbz) / (max_dbz - min_dbz)
            dbz_normalized = dbz_normalized[..., np.newaxis] # (Z, H, W, C)
            input_data_list.append(dbz_normalized)
    
    input_sequence_np = np.stack(input_data_list, axis=1) # (Z, T_in, H, W, C)
    x_input_tensor = torch.from_numpy(input_sequence_np).float()
    x_input_tensor = x_input_tensor.unsqueeze(0) # Añadir dim de batch -> (B, Z, T_in, H, W, C)
    x_input_tensor = x_input_tensor.permute(1, 0, 2, 3, 4, 5) # (Z, B, T_in, H, W, C)
    return x_input_tensor

def save_prediction_as_netcdf(output_path, pred_data_desnorm, config_params, prediction_datetime):
    """Guarda la predicción desnormalizada como un archivo NetCDF con metadatos."""
    # pred_data_desnorm debe ser (Z, H, W)
    # Lo expandimos a (1, Z, H, W) para el formato de tiempo
    pred_data_final_for_nc = np.expand_dims(pred_data_desnorm, axis=0)

    num_z, height, width = pred_data_desnorm.shape
    
    with NCDataset(output_path, 'w', format='NETCDF4') as ncfile:
        # Atributos Globales
        ncfile.Conventions = "CF-1.7"
        ncfile.title = f"Radar Reflectivity Forecast ({config_params['dbz_variable_name_pred']}) from ConvLSTM Model (Local Inference)"
        ncfile.institution = config_params.get('institution_name', "Desconocida")
        ncfile.history = f"Created {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} by local inference script."
        # ... (puedes añadir más atributos globales de tu config si quieres) ...

        # Dimensiones
        ncfile.createDimension('time', 1)
        ncfile.createDimension('level', num_z)
        ncfile.createDimension('y', height)
        ncfile.createDimension('x', width)

        # Variables de Coordenadas (simplificado, puedes añadir más detalle como en el script de entrenamiento)
        time_var = ncfile.createVariable('time', 'f8', ('time',))
        epoch_time = datetime(1970, 1, 1, 0, 0, 0)
        time_value_seconds = (prediction_datetime.replace(tzinfo=None) - epoch_time).total_seconds()
        time_var[:] = [time_value_seconds]
        time_var.units = "seconds since 1970-01-01 00:00:00 UTC"; time_var.calendar = "gregorian"

        level_var = ncfile.createVariable('level', 'f4', ('level',))
        level_var[:] = np.arange(config_params.get('grid_minz_km',1.0), config_params.get('grid_minz_km',1.0) + num_z * config_params.get('grid_dz_km',0.5), config_params.get('grid_dz_km',0.5))[:num_z]
        level_var.units = "km"

        # Variable de Datos
        pred_var = ncfile.createVariable(config_params['dbz_variable_name_pred'], 'f4', ('time', 'level', 'y', 'x'),
                                         fill_value=np.float32(config_params.get('fill_value', -9999.0)))
        pred_var.units = 'dBZ'
        pred_var.long_name = 'Predicted Radar Reflectivity'
        pred_var[:] = pred_data_final_for_nc
        print(f"Predicción guardada en: {output_path}")


# --- Configuración Principal para Inferencia ---
if __name__ == "__main__":
    print("Iniciando script de inferencia local...")

    # --- 1. Parámetros (DEBEN COINCIDIR CON LOS DEL ENTRENAMIENTO DEL MODELO CARGADO) ---
    # Estos son los parámetros de la arquitectura del modelo que funcionó sin OOM y fue guardado
    model_architecture_config = {
        'input_dim': 1,
        'hidden_dims': [32, 32],    # La última que probaste que funcionó
        'kernel_sizes': [(3,3), (3,3)],
        'num_layers': 2,           # La última que probaste que funcionó
        'pred_steps': 1,
        'use_layer_norm': True,
        'use_residual': False,
        'img_height': 500,
        'img_width': 500
    }

    # Parámetros de datos y normalización (deben ser los mismos del entrenamiento)
    data_params = {
        'min_dbz': -30.0,
        'max_dbz': 70.0,
        'expected_shape': (18, 500, 500), # (Z, H, W)
        'variable_name': 'DBZ', # Nombre de la variable en los archivos NC de ENTRADA
        'dbz_variable_name_pred': 'DBZ_forecast', # Nombre para la variable en el NC de SALIDA
        'fill_value': -9999.0,
        # Parámetros de grilla para metadatos del NC de salida (opcional, pero bueno para consistencia)
        'grid_minz_km': 1.0, 'grid_dz_km': 0.5,
        # ... (puedes añadir más de tu config original si los usas en save_prediction_as_netcdf) ...
        'institution_name': "Tu Nombre/Institucion Local"
    }
    
    # --- 2. Definir Dispositivo y Cargar Modelo ---
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

    # --- CAMBIA ESTA RUTA a donde tengas tu archivo .pth guardado ---
    # model_path = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/jupyter_notebooks/GeminiModels/Salidas modelo andando/best_convlstm_model.pth" # O el checkpoint que quieras
    
    # Modelo con 5 épocas y 80 batches 
    model_path = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/models/Model_5_Epochs_80_sequences/checkpoint_epoch_5.pth"
    if not os.path.exists(model_path):
        print(f"Error: Archivo de modelo no encontrado en {model_path}")
        exit()

    try:
        checkpoint_data = torch.load(model_path, map_location=device) # map_location es crucial para CPU
        model.load_state_dict(checkpoint_data['model_state_dict'])
        model.to(device)
        model.eval() # ¡MUY IMPORTANTE para inferencia!
        print(f"Modelo cargado exitosamente desde: {model_path}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        exit()

    # --- 3. Preparar la Secuencia de Entrada ---
    # --- CAMBIA ESTAS RUTAS a tus 6 archivos NetCDF de entrada para la nueva predicción ---
    # Deben estar en orden cronológico.
    input_nc_files_for_prediction = [
        #Modelo entrenado con 6 archivos de entrada
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/Big-Sample-New/201511164/121051.nc",
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/Big-Sample-New/201511164/121355.nc",
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/Big-Sample-New/201511164/121658.nc",
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/Big-Sample-New/201511164/122002.nc", 
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/Big-Sample-New/201511164/122305.nc", 
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/Big-Sample-New/201511164/122609.nc", 
    ]
    # Ajusta la longitud de esta lista a la 'seq_len' con la que se entrenó el modelo cargado.
    # Si el modelo se entrenó con seq_len=3 (como en la última config de prueba):
    # input_nc_files_for_prediction = input_nc_files_for_prediction[:3]
    
    # Define la seq_len usada para entrenar el modelo que estás cargando
    # (importante para load_and_preprocess_input_sequence)
    seq_len_entrenamiento = 3 # O 6, o la que corresponda a tu 'model_path'
    
    # Asegúrate de que la lista de archivos coincida con seq_len_entrenamiento
    if len(input_nc_files_for_prediction) != seq_len_entrenamiento:
        print(f"Advertencia: Se esperaban {seq_len_entrenamiento} archivos de entrada, pero se proporcionaron {len(input_nc_files_for_prediction)}.")
        print("Asegúrate de que la lista 'input_nc_files_for_prediction' y 'seq_len_entrenamiento' sean correctas.")
        # Podrías añadir un exit() aquí o intentar continuar si estás seguro.
    
    print(f"Cargando {len(input_nc_files_for_prediction)} archivos de entrada para la predicción...")
    try:
        x_input = load_and_preprocess_input_sequence(
            input_nc_files_for_prediction,
            data_params['min_dbz'],
            data_params['max_dbz'],
            data_params['expected_shape'],
            data_params['variable_name']
        )
        x_input = x_input.to(device)
        print(f"Tensor de entrada preparado con forma: {x_input.shape}")
    except Exception as e:
        print(f"Error al preparar los datos de entrada: {e}")
        exit()

    # --- 4. Realizar la Predicción ---
    print("Realizando predicción...")
    inference_start_time = time.time()
    with torch.no_grad():
        prediction_normalized = model(x_input) # Salida: (Z, B, T_pred, H, W, C)
    inference_end_time = time.time()
    print(f"Predicción completada en {inference_end_time - inference_start_time:.2f} segundos.")

    # --- 5. Post-Procesar y Desnormalizar ---
    # Asumiendo B=1, T_pred=1, C=1 en la salida del modelo
    pred_data_np = prediction_normalized.squeeze(1).squeeze(1).squeeze(-1).cpu().numpy() # (Z, H, W)
    dbz_predicted_desnormalized = pred_data_np * (data_params['max_dbz'] - data_params['min_dbz']) + data_params['min_dbz']
    print(f"Forma de la predicción desnormalizada (Z, H, W): {dbz_predicted_desnormalized.shape}")

    # --- 6. Guardar la Predicción (Opcional) ---
    # Necesitas el timestamp para el cual es esta predicción.
    # Deberías determinarlo a partir del último archivo de entrada + el intervalo de predicción.
    # Ejemplo MUY BÁSICO (¡DEBES ADAPTAR ESTO CON TIEMPOS REALES!):
    # Si parseas el nombre del último archivo de entrada:
    # last_input_datetime = parse_datetime_from_filename(input_nc_files_for_prediction[-1])
    # prediction_interval = timedelta(minutes=5) # Asumiendo 5 min
    # current_prediction_datetime = last_input_datetime + prediction_interval
    current_prediction_datetime = datetime.now() # Placeholder, usa un tiempo real
    
    output_nc_path = f"./prediction_output_{current_prediction_datetime.strftime('%Y%m%d_%H%M%S')}.nc"
    save_prediction_as_netcdf(output_nc_path, dbz_predicted_desnormalized, data_params, current_prediction_datetime)

    # --- 7. Visualizar una Capa de la Predicción (Opcional) ---
    layer_to_plot = 10 # Elige una capa Z (0 a 17)
    if dbz_predicted_desnormalized.shape[0] > layer_to_plot:
        plt.figure(figsize=(8, 7))
        plt.imshow(dbz_predicted_desnormalized[layer_to_plot, :, :], origin='lower', cmap='jet') # 'jet' es un cmap estándar
        plt.title(f"Predicción (Capa Z={layer_to_plot}) - {current_prediction_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        plt.colorbar(label="dBZ")
        plt.tight_layout()
        plt.show()
    else:
        print(f"No se puede graficar la capa {layer_to_plot}, solo hay {dbz_predicted_desnormalized.shape[0]} capas.")

    print("Script de inferencia finalizado.")