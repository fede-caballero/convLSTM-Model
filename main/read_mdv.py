import pyart

radar = pyart.io.read_mdv("/home/f-caballero/UM/TIF3/convLSTM-project/convLSTM-Model/main/214452.mdv")
reflectivity = radar.fields["reflectivity"]["data"]
print(reflectivity.shape)