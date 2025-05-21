#!/bin/tcsh

# -------------------------------------------------------------------------
# Script para convertir un archivo NetCDF a formato MDV usando
# una herramienta LROSE (RadxConvert o Radx2Grid).
# -------------------------------------------------------------------------

# --- Herramienta LROSE a usar (cambia a Radx2Grid si quieres usar ese) ---
set LROSE_TOOL = "RadxConvert"
#set LROSE_TOOL = "Radx2Grid"

# --- Variables configurables ---
set IN_NC_FILE    = "170906.nc"
set OUT_MDV_DIR   = "./mdv_output_from_nc"

# Configura el archivo de parámetros según la herramienta elegida
if ( "$LROSE_TOOL" == "RadxConvert" ) then
    set PARAMS_FILE   = "./RadxConvert.params" # Crea este archivo con el contenido de arriba
else if ( "$LROSE_TOOL" == "Radx2Grid" ) then
    set PARAMS_FILE   = "./Radx2Grid.params"   # Tu archivo Radx2Grid.params existente (verifícalo)
else
    echo "ERROR: Herramienta LROSE desconocida: $LROSE_TOOL. Saliendo."
    exit 1
endif

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
    echo "ERROR: Archivo de parámetros '$PARAMS_FILE' para $LROSE_TOOL no encontrado. Saliendo."
    exit 1
endif

# --- Ejecutar la herramienta LROSE ---
echo "INFO: Iniciando la conversión de '$IN_NC_FILE' a formato MDV usando $LROSE_TOOL..."
echo "INFO: Usando parámetros de: '$PARAMS_FILE'"
echo "INFO: Los archivos MDV se guardarán en: '$OUT_MDV_DIR'"

$LROSE_TOOL -params $PARAMS_FILE -f $IN_NC_FILE -outdir $OUT_MDV_DIR

# --- Verificar estado ---
if ( $status == 0 ) then
    echo "INFO: $LROSE_TOOL finalizó exitosamente."
    echo "INFO: Archivos MDV deberían estar en '$OUT_MDV_DIR'."
else
    echo "ERROR: $LROSE_TOOL finalizó con errores. Código de salida: $status"
    echo "       Por favor, revisa la salida de la consola y el archivo de parámetros."
endif

echo "INFO: Script finalizado."
