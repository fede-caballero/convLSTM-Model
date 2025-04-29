import os
import glob
import numpy as np
import netCDF4 as nc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt

# Configurar variable de entorno para reducir fragmentación
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configuración
data_dir = "/home/first_try_nc"
output_dir = "/home/model_output"
seq_len = 6  # Pasos de entrada
pred_len = 1  # Pasos de salida
batch_size = 1
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resolution = (18, 500, 500)  # z, y, x

# Liberar memoria de la GPU
torch.cuda.empty_cache()

# Implementación personalizada de ConvLSTM2D
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv3d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            padding=(self.padding, self.padding, self.padding),
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_c, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        c_next = f * c_cur + i * torch.tanh(cc_c)
        o = torch.sigmoid(cc_o)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        z, height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, z, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, z, height, width, device=self.conv.weight.device))

class ConvLSTM2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True):
        super(ConvLSTM2D, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                kernel_size,
                bias
            ) for i in range(num_layers)
        ])

    def forward(self, input_tensor, hidden_state=None):
        if self.batch_first:
            input_tensor = input_tensor.permute(0, 1, 5, 2, 3, 4)
        else:
            input_tensor = input_tensor.permute(1, 0, 5, 2, 3, 4)

        batch_size, seq_len, channels, z, height, width = input_tensor.size()
        if hidden_state is None:
            hidden_state = [
                self.cells[l].init_hidden(batch_size, (z, height, width))
                for l in range(self.num_layers)
            ]

        cur_layer_input = input_tensor
        output_inner = []

        for t in range(seq_len):
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                input_t = cur_layer_input[:, t]
                if input_t.dim() == 4:
                    input_t = input_t.unsqueeze(1)
                h, c = self.cells[layer_idx](input_t, (h, c))
                hidden_state[layer_idx] = (h, c)
                cur_layer_input = h.unsqueeze(1) if layer_idx < self.num_layers - 1 else h
            output_inner.append(h)

        output = torch.stack(output_inner, dim=1 if self.batch_first else 0)
        return output, hidden_state

# Dataset
class RadarDataset(Dataset):
    def __init__(self, data_dir, seq_len=6, pred_len=1, min_dbz=-30, max_dbz=70):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.min_dbz = min_dbz
        self.max_dbz = max_dbz
        self.folders = sorted([f for f in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(f)])
        if not self.folders:
            raise ValueError(f"No se encontraron subcarpetas en {data_dir}")
        
        self.sequences = []
        for folder in self.folders:
            files = sorted(glob.glob(os.path.join(folder, "*.nc")))
            if len(files) >= seq_len + pred_len:
                for i in range(len(files) - seq_len - pred_len + 1):
                    self.sequences.append(files[i:i + seq_len + pred_len])
        if not self.sequences:
            raise ValueError(f"No se encontraron secuencias válidas")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_files = self.sequences[idx]
        x = []
        for f in seq_files[:self.seq_len]:
            ds = nc.Dataset(f)
            dbz = ds.variables['DBZ'][:]  # (1, 18, 500, 500)
            ds.close()
            dbz = (dbz - self.min_dbz) / (self.max_dbz - self.min_dbz)  # Normaliza [-30, 70] a [0, 1]
            x.append(dbz[0])
        x = np.stack(x, axis=0)  # (6, 18, 500, 500)
        y = []
        for f in seq_files[self.seq_len:self.seq_len + self.pred_len]:
            ds = nc.Dataset(f)
            dbz = ds.variables['DBZ'][:]  # (1, 18, 500, 500)
            ds.close()
            dbz = (dbz - self.min_dbz) / (self.max_dbz - self.min_dbz)  # Normaliza [-30, 70] a [0, 1]
            y.append(dbz[0])
        y = np.stack(y, axis=0)  # (1, 18, 500, 500)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # (6, 18, 500, 500, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (1, 18, 500, 500, 1)
        return x_tensor, y_tensor

# Modelo ConvLSTM
class ConvLSTM(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=8, num_layers=2, kernel_size=3):
        super(ConvLSTM, self).__init__()
        self.convlstm = ConvLSTM2D(
            input_dim=in_channels,
            hidden_dim=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.conv = nn.Conv3d(hidden_channels, 1, kernel_size=(kernel_size, kernel_size, kernel_size),
                              padding=(1, 1, 1))

    def forward(self, x):
        x, _ = self.convlstm(x)
        x = x[:, -1]
        x = self.conv(x)
        return x

# Crear directorio de salida
os.makedirs(output_dir, exist_ok=True)

# Cargar datos
try:
    dataset = RadarDataset(data_dir, seq_len, pred_len, min_dbz=-30, max_dbz=70)
except ValueError as e:
    print(f"Error en dataset: {e}")
    exit(1)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Inicializar modelo
model = ConvLSTM(hidden_channels=8, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
scaler = GradScaler('cuda')

# Entrenamiento
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with autocast('cuda'):
            y_pred = model(x)
            loss = criterion(y_pred, y[:, 0])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

# Guardar modelo
torch.save(model.state_dict(), os.path.join(output_dir, "convlstm.pth"))

# Generar predicciones (ejemplo)
model.eval()
with torch.no_grad():
    x, y = dataset[0]
    x = x.unsqueeze(0).to(device)  # (1, 6, 18, 500, 500, 1)
    y_pred = model(x)  # (1, 1, 18, 500, 500)
    y_pred = y_pred.cpu().numpy() * (70 - (-30)) + (-30)  # Escala [0, 1] a [-30, 70]
    y = y.numpy() * (70 - (-30)) + (-30)  # Escala [0, 1] a [-30, 70]
    ds_out = nc.Dataset(os.path.join(output_dir, "pred_0.nc"), 'w', format='NETCDF4')
    ds_out.createDimension('time', 1)
    ds_out.createDimension('z', 18)
    ds_out.createDimension('y', 500)
    ds_out.createDimension('x', 500)
    dbz_var = ds_out.createVariable('DBZ', 'f4', ('time', 'z', 'y', 'x'))
    dbz_var[:] = y_pred
    ds_out.close()