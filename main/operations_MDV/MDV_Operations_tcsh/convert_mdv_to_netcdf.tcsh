#!/bin/tcsh
set input_base_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/operations_MDV/MDV_Operations_tcsh"
set output_base_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/operations_MDV/MDV_Operations_tcsh/output_netCDF"
set mdv2netcdf_params_file = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/operations_MDV/MDV_Operations_tcsh/Mdv2NetCDF.params"

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

# This code converts MDV files to NetCDF format using the Mdv2NetCDF tool.
# It iterates over folders containing MDV files, creates an output directory for each folder,
# and converts each MDV file to NetCDF format. The output files are named with a prefix and the original base name.
# The script also handles errors during conversion and cleans up previous output files.
# The script is written in tcsh and uses the `foreach` loop to iterate over directories and files.
# The script assumes that the Mdv2NetCDF tool is available in the system's PATH.
# The script also creates the output directory if it does not exist and removes any previous output files before conversion.
