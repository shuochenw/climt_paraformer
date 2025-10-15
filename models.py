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








        