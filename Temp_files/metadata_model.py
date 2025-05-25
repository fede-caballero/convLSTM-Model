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
