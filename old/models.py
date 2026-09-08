import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=0.5)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, 
                 nhead, num_layers, max_len):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        # self.pos_encoder = nn.Embedding(SEQUENCE_SIZE, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_dim)
        # self.relu = nn.ReLU()
        # self.decoder_2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        # x = x+self.pos_encoder(torch.arange(SEQUENCE_SIZE, device=device))
        x = self.transformer_encoder(x)
        # x = self.decoder(x[:, -1, :])
        x = self.decoder(x)
        # x = self.relu(x)
        # x = self.decoder_2(x)
        return x

# model A in RadNet
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(input_dim, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, output_dim)
                                )

    def forward(self, x):
        return self.seq(x)
        
# simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

# ResNet
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)
        
class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=128,
        num_blocks=4
    ):
        super().__init__()
        # Project input → hidden
        self.in_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        # Residual trunk
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_channels) for _ in range(num_blocks)]
        )
        # Project hidden → output
        self.out_proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.res_blocks(h)
        y = self.out_proj(h)
        return y

class GridPointMLPExplicit(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features, 768),
                                 nn.LeakyReLU(0.15),
                                 nn.Linear(768, 640),
                                 nn.LeakyReLU(0.15),
                                 nn.Linear(640, 512),
                                 nn.LeakyReLU(0.15),
                                 nn.Linear(512, 640),
                                 nn.LeakyReLU(0.15),
                                 nn.Linear(640, 640),
                                 nn.LeakyReLU(0.15),
                                 nn.Linear(640, out_features)
                                )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        y = self.mlp(x)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return y


class GridPointMLPExplicit2(nn.Module):
    def __init__(self, in_features, out_features, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features, hidden),
                                 nn.LeakyReLU(0.15),
                                 nn.Linear(hidden, hidden),
                                 nn.LeakyReLU(0.15),
                                 nn.Linear(hidden, out_features)
                                )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        y = self.mlp(x)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return y

class DynamicMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes):
        super().__init__()

        layers = []
        prev_dim = in_features

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LeakyReLU(0.15))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        y = self.net(x)
        y = y.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return y

class DynamicMLP_flatten(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes):
        super().__init__()

        layers = []
        prev_dim = in_features

        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LeakyReLU(0.15))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y



class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(num_groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1),
            nn.GroupNorm(num_groups, dim),
        )

    def forward(self, x):
        return x + self.block(x)

class GridPointResMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, hidden_dim, 1)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.blocks(h)
        return self.out_proj(h)


class GridPointMLPConv1d(nn.Module):
    def __init__(self, in_features, out_features, hidden=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_features, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, out_features, kernel_size=1),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # → [B*H*W, C, 1]
        x = x.permute(0, 2, 3, 1).reshape(-1, C).unsqueeze(-1)
        # MLP via Conv1d
        y = self.net(x)  # [B*H*W, out_features, 1]
        # → [B, out_features, H, W]
        y = y.squeeze(-1).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return y




class GridPointBiLSTM_C1_to_C2(nn.Module):
    def __init__(
        self,
        in_features,      # variables per input level
        hidden_size,
        num_layers,
        C2,               # number of output levels
        out_features=1,   # variables per output level
        dropout=0.1,
    ):
        super().__init__()

        self.C2 = C2
        self.out_features = out_features

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Map column representation → output vertical levels
        self.head = nn.Linear(2 * hidden_size, C2 * out_features)

    def forward(self, x):
        """
        x: [B, C1, H, W]  or [B, C1, H, W, V]
        """
        if x.dim() == 4:
            B, C1, H, W = x.shape
            V = 1
            x = x.unsqueeze(-1)
        else:
            B, C1, H, W, V = x.shape

        # [B*H*W, C1, V]
        x = x.permute(0, 2, 3, 1, 4).reshape(B * H * W, C1, V)

        # Encode vertical profile
        y, _ = self.lstm(x)              # [BHW, C1, 2*hidden]
        y = y.mean(dim=1)                # vertical pooling → [BHW, 2*hidden]

        # Decode to output levels
        y = self.head(y)                 # [BHW, C2*out_features]
        y = y.view(B, H, W, self.C2, self.out_features)

        # [B, C2, H, W] (if out_features=1)
        y = y.permute(0, 3, 1, 2, 4).squeeze(-1)

        return y


import torch
import torch.nn as nn

class LocalTemporalTransformer(nn.Module):
    def __init__(
        self,
        in_channels,     # C
        out_channels,    # C_out
        T,               # time window (e.g. 5 or 10)
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()

        self.T = T

        # Project per-time-step state (vertical profile) to embedding
        self.input_proj = nn.Linear(in_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Predict tendency for CURRENT timestep only
        self.output_proj = nn.Linear(d_model, out_channels)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        returns: [B, C_out, H, W]
        """
        B, T, C, H, W = x.shape
        assert T == self.T

        # Move grid points into batch
        # [B, T, C, H, W] → [B*H*W, T, C]
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)

        # Project features
        x = self.input_proj(x)            # [BHW, T, d_model]

        # Temporal attention
        x = self.transformer(x)           # [BHW, T, d_model]

        # Use LAST timestep representation (causal)
        x = x[:, -1, :]                   # [BHW, d_model]

        # Predict tendency
        y = self.output_proj(x)           # [BHW, C_out]

        # Restore grid
        y = y.view(B, H, W, -1).permute(0, 3, 1, 2)

        return y
