#!/bin/tcsh -f

# --- Variables de Configuración ---
# Prueba con UN SOLO archivo NetCDF "bueno" (el de 6.5MB) primero
set input_nc_file_to_test = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/netCDF_samples/netCDF_Big_sample/201001011/170906.nc"

set output_mdv_base_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/mdv_conversion_test" # Directorio de prueba
set params_file = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/scripts/NCGridCF_to_MDV.params" # <--- TU NUEVO TDRP

# --- Path a LROSE (asumiendo que ya está configurado o en el PATH) ---
# ... (tu lógica de setenv PATH si es necesaria) ...

# --- Verificaciones Previas ---
if (! -f $params_file) then
    echo "Error: No se encontró TDRP: $params_file"
    exit 1
endif
if (! -f $input_nc_file_to_test) then
    echo "Error: No se encontró NetCDF de entrada: $input_nc_file_to_test"
    exit 1
endif
mkdir -p $output_mdv_base_dir
if (! -d $output_mdv_base_dir) then
    echo "Error creando dir salida: $output_mdv_base_dir"
    exit 1
endif

# --- Conversión ---
set base_name = `basename $input_nc_file_to_test .nc`
set output_mdv_file = "$output_mdv_base_dir/${base_name}.mdv"
set log_file = "$output_mdv_base_dir/conversion_log_${base_name}.txt"

echo "Procesando: $input_nc_file_to_test" | tee $log_file
echo "Salida MDV: $output_mdv_file" | tee -a $log_file
echo "Log: $log_file" | tee -a $log_file
date | tee -a $log_file
echo "------------------------------------------------" | tee -a $log_file

echo "\n--- RadxPrint del NetCDF de Entrada ---" | tee -a $log_file
RadxPrint -f "$input_nc_file_to_test" -summary -atts -coords -fields | tee -a $log_file
echo "--- Fin RadxPrint ---" | tee -a $log_file

# No necesitas -start si el NetCDF tiene una variable de tiempo CF correcta
echo "\n--- Ejecutando RadxConvert ---" | tee -a $log_file
RadxConvert -f "$input_nc_file_to_test" \
            -outdir "$output_mdv_base_dir" \
            -outname "$output_mdv_file" \
            -params "$params_file" \
            -v -debug \
            |& tee -a $log_file

if ($status == 0) then
    echo "ÉXITO: Convertido $input_nc_file_to_test -> $output_mdv_file" | tee -a $log_file
    echo "\n--- RadxPrint del MDV de Salida (resumen) ---" | tee -a $log_file
    RadxPrint -f "$output_mdv_file" -summary | tee -a $log_file
else
    echo "ERROR: Falla al convertir $input_nc_file_to_test. Código de salida: $status" | tee -a $log_file
endif
echo "================================================\n" | tee -a $log_file

echo "Conversión completada."
