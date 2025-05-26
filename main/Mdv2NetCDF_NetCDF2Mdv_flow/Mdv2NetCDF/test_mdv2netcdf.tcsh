#!/bin/tcsh

set input_base_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/Mdv2NetCDF_NetCDF2Mdv_flow/Mdv2NetCDF"
set output_base_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/Mdv2NetCDF_NetCDF2Mdv_flow/Mdv2NetCDF/Output_conversion"
# Define la ruta a tu archivo de parámetros de Mdv2NetCDF
set mdv2netcdf_params_file = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/Mdv2NetCDF_NetCDF2Mdv_flow/Mdv2NetCDF/Mdv2NetCDF.params" # <- CAMBIA ESTO

# Iterar sobre carpetas (e.g., 201001011)
foreach folder ($input_base_dir/*)
    if (-d $folder) then
        set folder_name = `basename $folder`
        set output_dir = "$output_base_dir/$folder_name"
        mkdir -p $output_dir

        # Limpiar archivos previos
        rm -f $output_dir/_latest_data_info*
        rm -f $output_dir/*.nc

        # Convertir cada MDV
        foreach mdv_file ($folder/*.mdv) # [cite: 2]
            set base_name = `basename $mdv_file .mdv`
            # El nombre del archivo de salida parece estar bien, lo mantenemos
            set output_file = "$output_dir/ncfdata20100101_$base_name.nc"
            
            echo "Procesando $mdv_file..."
            # Modificación: Añadir la opción -params
            Mdv2NetCDF -params $mdv2netcdf_params_file -f $mdv_file -outdir $output_dir -v # [cite: 2]
            
            if ($status == 0) then
                echo "Convertido: $mdv_file -> $output_file"
            else
                echo "Error al convertir: $mdv_file (status: $status)" # [cite: 3]
            endif
        end
    endif
end

echo "Proceso de conversión completado."

# Este script convierte archivos MDV a formato NetCDF usando la herramienta Mdv2NetCDF. [cite: 4]
# Itera sobre carpetas que contienen archivos MDV, crea un directorio de salida para cada carpeta, [cite: 4]
# y convierte cada archivo MDV a formato NetCDF, utilizando un archivo de parámetros específico. [cite: 4]
# Los archivos de salida son nombrados con un prefijo y el nombre base original. [cite: 5]
# El script también maneja errores durante la conversión y limpia archivos de salida previos. [cite: 6]
# El script está escrito en tcsh y usa el bucle `foreach` para iterar sobre directorios y archivos. [cite: 7]
# El script asume que la herramienta Mdv2NetCDF está disponible en el PATH del sistema. [cite: 8]
# El script también crea el directorio de salida si no existe y elimina cualquier archivo de salida previo antes de la conversión. [cite: 9]
