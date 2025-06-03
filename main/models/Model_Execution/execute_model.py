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
    def forward(self, x_volumetric):
        num_z_levels, b, seq_len, h, w, c_in = x_volumetric.shape
        all_level_predictions = []
        for z_idx in range(num_z_levels):
            x_level = x_volumetric[z_idx, ...]; x_level_permuted = x_level.permute(0, 1, 4, 2, 3)
            current_input = x_level_permuted
            for i in range(self.num_layers):
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
    pred_data_byte = np.where(pred_data_desnorm < 0, -128,
                             np.clip(((pred_data_desnorm - 33.5) / 0.5), -127, 127).round().astype(np.int8))
    pred_data_final_for_nc = np.expand_dims(pred_data_byte, axis=0)
    num_z, height, width = pred_data_desnorm.shape
    grid_minx_km = config_params.get('grid_minx_km', -249.5)
    grid_miny_km = config_params.get('grid_miny_km', -249.5)
    grid_dx_km = config_params.get('grid_dx_km', 1.0)
    grid_dy_km = config_params.get('grid_dy_km', 1.0)
    grid_minz_km = config_params.get('grid_minz_km', 1.0)
    grid_dz_km = config_params.get('grid_dz_km', 0.5)
    x_coord_values = np.arange(grid_minx_km, grid_minx_km + width * grid_dx_km, grid_dx_km, dtype=np.float32)
    y_coord_values = np.arange(grid_miny_km, grid_miny_km + height * grid_dy_km, grid_dy_km, dtype=np.float32)
    z_coord_values = np.arange(grid_minz_km, grid_minz_km + num_z * grid_dz_km, grid_dz_km, dtype=np.float32)
    proj_origin_lon = config_params.get('sensor_longitude', -68.0169982910156)
    proj_origin_lat = config_params.get('sensor_latitude', -34.6479988098145)
    earth_radius_m = config_params.get('earth_radius_m', 6378137)
    proj = pyproj.Proj(proj="aeqd", lon_0=proj_origin_lon, lat_0=proj_origin_lat, R=earth_radius_m)
    x_grid, y_grid = np.meshgrid(x_coord_values, y_coord_values)
    lon0, lat0 = proj(x_grid, y_grid, inverse=True)
    with NCDataset(output_path, 'w', format='NETCDF3_CLASSIC') as ncfile:
        ncfile.Conventions = "CF-1.6"
        ncfile.title = f"SAN_RAFAEL - Forecast t+{config_params.get('prediction_interval_minutes', 3)}min"
        ncfile.institution = config_params.get('institution_name', "UCAR")
        ncfile.source = config_params.get('data_source_name', "Gobierno de Mendoza")
        ncfile.history = f"Created {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} by ConvLSTM prediction script."
        ncfile.comment = f"Forecast data from ConvLSTM model for lead time +{config_params.get('prediction_interval_minutes', 3)} min."
        ncfile.references = "Tesis de Federico Caballero, Universidad de Mendoza"
        ncfile.createDimension('time', None)
        ncfile.createDimension('bounds', 2)
        ncfile.createDimension('x0', width)
        ncfile.createDimension('y0', height)
        ncfile.createDimension('z0', num_z)
        time_v = ncfile.createVariable('time', 'f8', ('time',))
        time_v.standard_name = "time"
        time_v.long_name = "Data time"
        time_v.units = "seconds since 1970-01-01T00:00:00Z"
        time_v.axis = "T"
        time_v.bounds = "time_bounds"
        epoch_time = datetime(1970, 1, 1)
        time_value_seconds = (prediction_datetime.replace(tzinfo=None) - epoch_time).total_seconds()
        time_v.comment = prediction_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        time_v[:] = [time_value_seconds]
        time_bnds_v = ncfile.createVariable('time_bounds', 'f8', ('time', 'bounds'))
        time_bnds_v.comment = "Time bounds for data interval"
        time_bnds_v.units = "seconds since 1970-01-01T00:00:00Z"
        time_begin_calc_seconds = time_value_seconds - (2 * 60 + 48)
        time_end_calc_seconds = time_value_seconds
        time_bnds_v[:] = [[time_begin_calc_seconds, time_end_calc_seconds]]
        start_time_v = ncfile.createVariable('start_time', 'f8', ('time',))
        start_time_v.long_name = "start_time"
        start_time_v.units = "seconds since 1970-01-01T00:00:00Z"
        start_time_v.comment = datetime.fromtimestamp(time_begin_calc_seconds).strftime("%Y-%m-%dT%H:%M:%SZ")
        start_time_v[:] = [time_begin_calc_seconds]
        stop_time_v = ncfile.createVariable('stop_time', 'f8', ('time',))
        stop_time_v.long_name = "stop_time"
        stop_time_v.units = "seconds since 1970-01-01T00:00:00Z"
        stop_time_v.comment = datetime.fromtimestamp(time_end_calc_seconds).strftime("%Y-%m-%dT%H:%M:%SZ")
        stop_time_v[:] = [time_end_calc_seconds]
        x_v = ncfile.createVariable('x0', 'f4', ('x0',))
        x_v.standard_name = "projection_x_coordinate"
        x_v.units = "km"
        x_v.axis = "X"
        x_v[:] = x_coord_values
        y_v = ncfile.createVariable('y0', 'f4', ('y0',))
        y_v.standard_name = "projection_y_coordinate"
        y_v.units = "km"
        y_v.axis = "Y"
        y_v[:] = y_coord_values
        z_v = ncfile.createVariable('z0', 'f4', ('z0',))
        z_v.standard_name = "altitude"
        z_v.long_name = "constant altitude levels"
        z_v.units = "km"
        z_v.positive = "up"
        z_v.axis = "Z"
        z_v[:] = z_coord_values
        lat0_v = ncfile.createVariable('lat0', 'f4', ('y0', 'x0'))
        lat0_v.standard_name = "latitude"
        lat0_v.units = "degrees_north"
        lat0_v[:] = lat0
        lon0_v = ncfile.createVariable('lon0', 'f4', ('y0', 'x0'))
        lon0_v.standard_name = "longitude"
        lon0_v.units = "degrees_east"
        lon0_v[:] = lon0
        gm_v = ncfile.createVariable('grid_mapping_0', 'i4')
        gm_v.grid_mapping_name = "azimuthal_equidistant"
        gm_v.longitude_of_projection_origin = proj_origin_lon
        gm_v.latitude_of_projection_origin = proj_origin_lat
        gm_v.false_easting = 0.0
        gm_v.false_northing = 0.0
        gm_v.earth_radius = earth_radius_m
        dbz_v = ncfile.createVariable('DBZ', 'i1', ('time', 'z0', 'y0', 'x0'),
                                      fill_value=np.int8(-128))
        dbz_v.units = 'dBZ'
        dbz_v.long_name = 'DBZ'
        dbz_v.standard_name = 'DBZ'
        dbz_v.coordinates = "lon0 lat0"
        dbz_v.grid_mapping = "grid_mapping_0"
        dbz_v.scale_factor = np.float32(0.5)
        dbz_v.add_offset = np.float32(33.5)
        dbz_v.valid_min = np.int8(-127)
        dbz_v.valid_max = np.int8(127)
        dbz_v.min_value = np.float32(config_params.get('min_dbz', -29.0))
        dbz_v.max_value = np.float32(config_params.get('max_dbz', 60.5))
        dbz_v[:] = pred_data_final_for_nc
    print(f"Predicción guardada en: {output_path}")

# --- Configuración Principal para Inferencia ---
if __name__ == "__main__":
    print("Iniciando script de inferencia local...")
    model_architecture_config = {
        'input_dim': 1,
        'hidden_dims': [32, 32],
        'kernel_sizes': [(3,3), (3,3)],
        'num_layers': 2,
        'pred_steps': 1,
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
    model_path = "/home/f-caballero/UM/TIF3/convLSTM-project/Modificaciones_modelo/WorkinModel_30-05-25/010625-New_try/Modelo-de-Grok-3/model_out/best_convlstm_model.pth"
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
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/2010010112/180809.nc",
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/2010010112/181058.nc",
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/2010010112/181421.nc",
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/2010010112/181709.nc",
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/2010010112/181958.nc",
        "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/2010010112/182245.nc",
    ]
    seq_len_entrenamiento = 6
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