#!/bin/tcsh
set input_base_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/operationsMDV/MDV_to_analyze"
set output_base_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/operationsMDV/MDV_to_analyze/Output_conversion"

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
        foreach mdv_file ($folder/*.mdv)
            set base_name = `basename $mdv_file .mdv`
            set output_file = "$output_dir/ncfdata20100101_$base_name.nc"
            Mdv2NetCDF -f $mdv_file -outdir $output_dir -v
            if ($status == 0) then
                echo "Convertido: $mdv_file -> $output_file"
            else
                echo "Error al convertir: $mdv_file"
            endif
        end
    endif
end
