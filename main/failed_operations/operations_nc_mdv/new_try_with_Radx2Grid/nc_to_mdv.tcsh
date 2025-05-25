#!/bin/tcsh

# -------------------------------------------------------------------------
# Script para convertir un archivo NetCDF (Cartesiano) a formato MDV
# usando Radx2Grid.
#
# Asume que el archivo Radx2Grid.params ha sido modificado según
# las necesidades para coincidir con la malla del NetCDF de entrada
# y para generar salida MDV.
# -------------------------------------------------------------------------

# --- Variables configurables ---
set IN_NC_FILE    = "170906.nc"             # Tu archivo NetCDF de entrada
set PARAMS_FILE   = "./Radx2Grid.params"    # Tu archivo de parámetros Radx2Grid MODIFICADO
set OUT_MDV_DIR   = "./mdv_output_from_nc"  # Directorio de salida para los MDV

# --- Configuración del entorno LROSE (descomentar y ajustar si es necesario) ---
# echo "INFO: Configurando entorno LROSE (si es necesario)..."
# setenv LROSE_HOME /usr/local/lrose
# set path = ( $LROSE_HOME/bin $path )
# if ( $?LD_LIBRARY_PATH ) then
#   setenv LD_LIBRARY_PATH ${LROSE_HOME}/lib:${LD_LIBRARY_PATH}
# else
#   setenv LD_LIBRARY_PATH ${LROSE_HOME}/lib
# endif

# --- Establecer el formato de escritura de MDV a binario ---
setenv MDV_WRITE_FORMAT FORMAT_MDV
echo "INFO: MDV_WRITE_FORMAT establecido a: $MDV_WRITE_FORMAT"

# --- Crear directorio de salida si no existe ---
if ( ! -d $OUT_MDV_DIR ) then
    echo "INFO: Creando directorio de salida: $OUT_MDV_DIR"
    mkdir -p $OUT_MDV_DIR
    if ( $status != 0 ) then
        echo "ERROR: No se pudo crear el directorio de salida $OUT_MDV_DIR. Saliendo."
        exit 1
    endif
endif

# --- Verificar archivos de entrada ---
if ( ! -f $IN_NC_FILE ) then
    echo "ERROR: Archivo NetCDF de entrada '$IN_NC_FILE' no encontrado. Saliendo."
    exit 1
endif

if ( ! -f $PARAMS_FILE ) then
    echo "ERROR: Archivo de parámetros '$PARAMS_FILE' no encontrado. Saliendo."
    exit 1
endif

# --- Ejecutar Radx2Grid ---
echo "INFO: Convirtiendo '$IN_NC_FILE' a formato MDV..."
echo "INFO: Usando parámetros de: '$PARAMS_FILE'"
echo "INFO: Los archivos MDV se guardarán en: '$OUT_MDV_DIR'"

Radx2Grid -params $PARAMS_FILE -f $IN_NC_FILE -outdir $OUT_MDV_DIR

# --- Verificar estado ---
if ( $status == 0 ) then
    echo "INFO: Radx2Grid finalizó exitosamente."
    echo "INFO: Archivos MDV deberían estar en '$OUT_MDV_DIR'."
else
    echo "ERROR: Radx2Grid finalizó con errores. Código de salida: $status"
endif

echo "INFO: Script finalizado."
