#!/bin/tcsh
set params_file = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/params/Mdv2NetCDF_params"
set input_file = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/MDV/small_sample/201001011/170906.mdv"
set output_dir = "/home/f-caballero/UM/netCDF/test"
set output_name = "170906.nc"

mkdir -p $output_dir
rm -rf $output_dir/*

# Actualizar output_filename en params_file
sed -i "s/output_filename = .*/output_filename = \"$output_name\";/" $params_file

Mdv2NetCDF -params $params_file -f $input_file -outdir $output_dir -v
rm -f $output_dir/_latest_data_info*
ls -l $output_dir
