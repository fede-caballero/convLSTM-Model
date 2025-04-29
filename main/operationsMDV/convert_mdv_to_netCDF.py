import os
import subprocess

# Paths
mdv_file = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/MDV/small_sample/201001011/170906.mdv"
output_dir = "/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/MDV/small_sample/netCDF/201001011"
output_file = os.path.join(output_dir, os.path.basename(mdv_file).replace(".mdv", ".nc"))

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set environment PATH to include LROSE bin directory
lrose_bin = os.path.expanduser("~/lrose/bin")
env = os.environ.copy()
env["PATH"] = f"{lrose_bin}:{env.get('PATH', '')}"

# RadxConvert command
radx_command = [
    "RadxConvert",
    "-f", mdv_file,
    "-outdir", output_dir,
    "-outname", output_file,
    "-netcdf",
    "-v",  # Verbose output
]

try:
    # Run RadxConvert
    result = subprocess.run(radx_command, check=True, capture_output=True, text=True)
    print(f"Conversion successful! NetCDF file saved at: {output_file}")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error during conversion: {e.stderr}")
except FileNotFoundError:
    print("RadxConvert not found. Ensure LROSE is installed and added to PATH.")
