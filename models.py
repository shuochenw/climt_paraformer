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
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y

class DynamicMLP_flatten_v2(nn.Module):
    """
    MLP with unconditional top-layer masking (input + output).

    Input:
      x: [B, C_in]  (normalized)

    Output:
      y: [B, C_out]

    Hard constraints (unconditional):
      (1) INPUT hard mask:
          Zero out T/Q INPUTS for levels >= input_clamp_from_level
          Assumes INPUT ordering: [T(0:nlev), Q(0:nlev), ...optional scalars...]

      (2) OUTPUT hard clamp:
          Zero out dT and dQ outputs for levels >= clamp_from_level
          Assumes OUTPUT ordering: [dT(0:nlev), dQ(0:nlev), ...optional extras...]
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes,
        nlev: int = 28,
        clamp_from_level: int = 19,       # OUTPUT: enforce dT/dQ[clamp_from_level:] = 0
        input_clamp_from_level: int = 19  # INPUT: enforce T/Q[input_clamp_from_level:] = 0
    ):
        super().__init__()
        self.nlev = int(nlev)
        self.clamp_from_level = None if clamp_from_level is None else int(clamp_from_level)
        self.input_clamp_from_level = None if input_clamp_from_level is None else int(input_clamp_from_level)

        layers = []
        prev_dim = int(in_features)
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, int(h)))
            layers.append(nn.ReLU())
            prev_dim = int(h)
        layers.append(nn.Linear(prev_dim, int(out_features)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in]
        if x.ndim != 2:
            raise ValueError(f"Expected x [B, C_in], got {tuple(x.shape)}")

        # ---- (1) INPUT hard mask on T/Q above a given layer ----
        if self.input_clamp_from_level is not None:
            k0 = self.input_clamp_from_level
            if not (0 <= k0 <= self.nlev):
                raise ValueError(f"input_clamp_from_level must be in [0, {self.nlev}], got {k0}")
            if x.shape[1] < 2 * self.nlev:
                raise ValueError(
                    f"Expected x to have at least 2*nlev={2*self.nlev} input features for [T,Q]. "
                    f"Got C_in={x.shape[1]}"
                )

            x = x.clone()
            # T block: [0:nlev]
            x[:, k0:self.nlev] = 0.0
            # Q block: [nlev:2*nlev]
            x[:, self.nlev + k0:self.nlev + self.nlev] = 0.0

        y = self.net(x)  # [B, C_out]

        # ---- (2) OUTPUT hard clamp on dT/dQ above a given layer ----
        if self.clamp_from_level is not None:
            k0 = self.clamp_from_level
            if not (0 <= k0 <= self.nlev):
                raise ValueError(f"clamp_from_level must be in [0, {self.nlev}], got {k0}")
            if y.shape[1] < 2 * self.nlev:
                raise ValueError(
                    f"Expected y to have at least 2*nlev={2*self.nlev} outputs for [dT,dQ]. "
                    f"Got C_out={y.shape[1]}"
                )

            y = y.clone()
            # dT block: [0:nlev]
            y[:, k0:self.nlev] = 0.0
            # dQ block: [nlev:2*nlev]
            y[:, self.nlev + k0:self.nlev + self.nlev] = 0.0

        return y

class DynamicMLP_flatten_maskq(nn.Module):
    """
    Same as DynamicMLP_flatten, but masks dT/dQ outputs in dry layers based on q_raw.
    Assumes output layout:
      y[:, 0:28]   = dT levels
      y[:, 28:56]  = dQ levels
      y[:, 56]     = precip (or whatever your last scalar is)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes,
        q_thresh: float = 5e-5, #1e-8
        nlev: int = 28,
        zero_top_k: int = 0,   # optional: also zero top-k levels if you want
    ):
        super().__init__()
        self.q_thresh = q_thresh
        self.nlev = nlev
        self.zero_top_k = zero_top_k

        layers = []
        prev_dim = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, q_raw: torch.Tensor) -> torch.Tensor:
        """
        x:     [B, in_features]  (your existing normalized input)
        q_raw: [B, nlev]         (RAW specific humidity profile in kg/kg, NOT normalized/logged)
        """
        y = self.net(x)  # [B, out_features]

        # Build moist mask from raw q
        moist = (q_raw >= self.q_thresh).to(dtype=y.dtype)  # [B, nlev]
        # Mask dT and dQ level-wise
        y[:, :self.nlev] *= moist
        y[:, self.nlev:2*self.nlev] *= moist

        # Optional: also zero top-k levels (last k of each block)
        if self.zero_top_k > 0:
            k = self.zero_top_k
            y[:, self.nlev-k:self.nlev] = 0.0
            y[:, 2*self.nlev-k:2*self.nlev] = 0.0

        return y


class FlatTemporalTransformer(nn.Module):
    """
    Works for BOTH:
      1) Spatial case:  x is flattened as [T_total*H*W, C_in]  (time-major, then H,W)
      2) Single-column: set H=W=1 and pass x as [T_total, C_in] (or [T_total*1*1, C_in])

    Returns:
      - Same flattened layout: [T_total*H*W, C_out]
        (for H=W=1 => [T_total, C_out])
    """
    def __init__(
        self,
        C_in,
        C_out,
        T_window,
        H=1,
        W=1,
        d_model=512,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.H = int(H)
        self.W = int(W)
        self.Tw = int(T_window)

        self.input_proj = nn.Linear(C_in, d_model)
        self.output_proj = nn.Linear(d_model, C_out)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.time_emb = nn.Parameter(torch.randn(1, self.Tw, d_model))

    def forward(self, x):
        """
        x: [B, C_in] where B = T_total*H*W  (spatial)
           if H=W=1, B = T_total (single-column)
        returns: [B, C_out]
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x [B, C_in], got {tuple(x.shape)}")

        B, C = x.shape
        H, W, Tw = self.H, self.W, self.Tw
        HW = H * W

        if HW <= 0:
            raise ValueError(f"Invalid H,W: {H},{W}")

        if B % HW != 0:
            raise ValueError(
                f"B ({B}) must be divisible by H*W ({HW}). "
                f"Got H={H}, W={W}."
            )

        T_total = B // HW

        # --------------------------------------------------
        # 1) Flat -> [T_total, C, H, W]
        # --------------------------------------------------
        x = x.view(T_total, H, W, C).permute(0, 3, 1, 2).contiguous()

        # If not enough time steps for one window, just return zeros
        if T_total < Tw:
            return torch.zeros(B, self.output_proj.out_features,
                               device=x.device, dtype=x.dtype)

        # --------------------------------------------------
        # 2) Temporal windows: [T2, Tw, C, H, W]
        # --------------------------------------------------
        x = x.unfold(0, Tw, 1)          # [T2, C, H, W, Tw]
        x = x.permute(0, 4, 1, 2, 3)    # [T2, Tw, C, H, W]

        # --------------------------------------------------
        # 3) Grid points -> batch: [T2*H*W, Tw, C]
        # --------------------------------------------------
        T2, Tw2, C2, H2, W2 = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(T2 * H2 * W2, Tw2, C2)

        # --------------------------------------------------
        # 4) Temporal transformer (causal mask)
        # --------------------------------------------------
        x = self.input_proj(x)          # [T2*H*W, Tw, d_model]
        x = x + self.time_emb           # broadcast [1, Tw, d_model]

        mask = torch.triu(
            torch.ones(Tw2, Tw2, device=x.device, dtype=torch.bool),
            diagonal=1
        )  # True blocks future attention

        x = self.transformer(x, mask=mask)
        x = x[:, -1, :]                 # last token
        x = self.output_proj(x)         # [T2*H*W, C_out]

        # --------------------------------------------------
        # 5) Restore to [T2, C_out, H, W]
        # --------------------------------------------------
        x = x.view(T2, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # --------------------------------------------------
        # 6) Pad early times to length T_total, then flatten back to [B, C_out]
        # --------------------------------------------------
        pad = torch.zeros(
            Tw - 1, x.shape[1], H2, W2,
            device=x.device, dtype=x.dtype
        )
        x = torch.cat([pad, x], dim=0)  # [T_total, C_out, H, W]

        y = x.permute(0, 2, 3, 1).reshape(B, -1)  # [T_total*H*W, C_out]
        return y



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class FlatTemporalTransformer2(nn.Module):
    """
    No spatial dims. Treat B as time.

    Input:
      x: [B, C_in]   where B = T_total (time)

    Output:
      y: [B, C_out]  one output per time step (first Tw-1 are padded with zeros)

    Architecture matches your TransformerModel style:
      Linear encoder -> sinusoidal PE -> TransformerEncoder -> Linear decoder
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        T_window: int,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.Tw = int(T_window)
        self.causal = bool(causal)

        self.encoder = nn.Linear(C_in, d_model)
        # self.in_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.Tw)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in] (B is time)
        returns: [B, C_out]
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x [B(time), C_in], got {tuple(x.shape)}")

        T_total, C_in = x.shape
        Tw = self.Tw

        if T_total < Tw:
            return torch.zeros(T_total, self.decoder.out_features, device=x.device, dtype=x.dtype)

        # 1) Sliding windows: [T2, Tw, C_in]
        xw = x.unfold(0, Tw, 1)                 # [T2, C_in, Tw]
        xw = xw.permute(0, 2, 1).contiguous()   # [T2, Tw, C_in]

        # 2) Encoder + PE
        h = self.encoder(xw)                    # [T2, Tw, d_model]
        # h = self.in_norm(h) # increase error
        h = self.pos_encoder(h)                 # [T2, Tw, d_model]

        # 3) Transformer (+ optional causal mask)
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.ones(Tw, Tw, device=x.device, dtype=torch.bool),
                diagonal=1
            )
        h = self.transformer_encoder(h, mask=attn_mask)  # [T2, Tw, d_model]

        # 4) Decode tokens, then take last token per window
        # y_tokens = self.decoder(h)              # [T2, Tw, C_out]
        # y_core = y_tokens[:, -1, :]             # [T2, C_out]
        y_core = self.decoder(h[:, -1, :])  # [T2, C_out]

        # 5) Pad early times to length T_total
        pad = torch.zeros(Tw - 1, y_core.size(-1), device=x.device, dtype=y_core.dtype)
        y = torch.cat([pad, y_core], dim=0)     # [T_total, C_out]

        return y



class WindowTemporalTransformer(nn.Module):
    """
    Windowed temporal transformer.

    Input:
      x: [B, Tw, C_in]   (each sample is a time window)

    Output:
      y: [B, C_out]      (prediction for the last step of each window)
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        T_window: int,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.Tw = int(T_window)
        self.causal = bool(causal)

        self.encoder = nn.Linear(C_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.Tw)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, C_out)

        # Cache causal mask (Tw x Tw) to avoid re-alloc every forward
        self.register_buffer("_causal_mask", None, persistent=False)

    def _get_causal_mask(self, device):
        if (self._causal_mask is None) or (self._causal_mask.device != device) or (self._causal_mask.shape[0] != self.Tw):
            self._causal_mask = torch.triu(
                torch.ones(self.Tw, self.Tw, device=device, dtype=torch.bool),
                diagonal=1
            )
        return self._causal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Tw, C_in]
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, Tw, C_in], got {tuple(x.shape)}")
        if x.shape[1] != self.Tw:
            raise ValueError(f"Expected Tw={self.Tw}, got x.shape[1]={x.shape[1]}")

        h = self.encoder(x)          # [B, Tw, d_model]
        h = self.pos_encoder(h)      # [B, Tw, d_model]

        attn_mask = self._get_causal_mask(x.device) if self.causal else None
        h = self.transformer_encoder(h, mask=attn_mask)  # [B, Tw, d_model]

        # take last token
        y = self.decoder(h[:, -1, :])  # [B, C_out]
        return y

class WindowTemporalTransformer_v2(nn.Module):
    """
    Windowed temporal transformer.

    Input:
      x: [B, Tw, C_in]   (each sample is a time window)

    Output:
      y: [B, C_out]      (prediction for the last step of each window)

    Hard constraints:
      (1) INPUT hard mask (recommended):
          Zero out T/Q INPUTS above a given vertical index (levels >= input_clamp_from_level),
          so upper-level inputs cannot influence bottom predictions.
          Assumes INPUT ordering: [T(0:nlev), Q(0:nlev), ...optional scalars...]

      (2) OUTPUT hard clamp:
          Zero out dT and dQ outputs above a given vertical index (levels >= clamp_from_level).
          Assumes OUTPUT ordering: [dT(0:nlev), dQ(0:nlev), ...optional extras...]
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        T_window: int,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        causal: bool = True,
        # ---- hard constraint settings ----
        nlev: int = 28,                    # number of vertical levels in T/Q and dT/dQ blocks
        clamp_from_level: int = None,      # OUTPUT: enforce dT/dQ[clamp_from_level:] = 0
        input_clamp_from_level: int = None # INPUT: enforce T/Q[ input_clamp_from_level: ] = 0
    ):
        super().__init__()
        self.Tw = int(T_window)
        self.causal = bool(causal)

        self.nlev = int(nlev)
        self.clamp_from_level = None if clamp_from_level is None else int(clamp_from_level)
        self.input_clamp_from_level = None if input_clamp_from_level is None else int(input_clamp_from_level)

        self.encoder = nn.Linear(C_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.Tw)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, C_out)

        # Cache causal mask (Tw x Tw) to avoid re-alloc every forward
        self.register_buffer("_causal_mask", None, persistent=False)

    def _get_causal_mask(self, device):
        if (self._causal_mask is None) or (self._causal_mask.device != device) or (self._causal_mask.shape[0] != self.Tw):
            self._causal_mask = torch.triu(
                torch.ones(self.Tw, self.Tw, device=device, dtype=torch.bool),
                diagonal=1
            )
        return self._causal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Tw, C_in]
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, Tw, C_in], got {tuple(x.shape)}")
        if x.shape[1] != self.Tw:
            raise ValueError(f"Expected Tw={self.Tw}, got x.shape[1]={x.shape[1]}")

        # ---- NEW (1): INPUT hard mask on T/Q above a given layer ----
        # Assumes input ordering: T[0:nlev], Q[nlev:2*nlev], then optional scalars.
        # We mask in-place on a cloned tensor to avoid surprising side effects upstream.
        if self.input_clamp_from_level is not None:
            k0 = self.input_clamp_from_level
            if not (0 <= k0 <= self.nlev):
                raise ValueError(f"input_clamp_from_level must be in [0, {self.nlev}], got {k0}")
            if x.shape[2] < 2 * self.nlev:
                raise ValueError(
                    f"Expected x to have at least 2*nlev={2*self.nlev} input features for [T,Q]. "
                    f"Got C_in={x.shape[2]}"
                )

            x = x.clone()
            # T block
            x[:, :, k0:self.nlev] = 0.0
            # Q block
            x[:, :, self.nlev + k0:self.nlev + self.nlev] = 0.0

        h = self.encoder(x)          # [B, Tw, d_model]
        h = self.pos_encoder(h)      # [B, Tw, d_model]

        attn_mask = self._get_causal_mask(x.device) if self.causal else None
        h = self.transformer_encoder(h, mask=attn_mask)  # [B, Tw, d_model]

        # take last token
        y = self.decoder(h[:, -1, :])  # [B, C_out]

        # ---- (2): OUTPUT hard constraint on dT/dQ above a given layer ----
        if self.clamp_from_level is not None:
            k0 = self.clamp_from_level
            if not (0 <= k0 <= self.nlev):
                raise ValueError(f"clamp_from_level must be in [0, {self.nlev}], got {k0}")
            if y.shape[1] < 2 * self.nlev:
                raise ValueError(
                    f"Expected y to have at least 2*nlev={2*self.nlev} outputs for [dT,dQ]. "
                    f"Got C_out={y.shape[1]}"
                )

            # dT block: [0:nlev], dQ block: [nlev:2*nlev]
            y[:, k0:self.nlev] = 0.0
            y[:, self.nlev + k0:self.nlev + self.nlev] = 0.0

        return y


class WindowTemporalLSTM(nn.Module):
    """
    Simple windowed LSTM.

    Input:
      x: [B, Tw, C_in]   (each sample is a time window)

    Output:
      y: [B, C_out]      (prediction for the last step of each window)

    Hard constraints:
      (1) INPUT hard mask:
          Zero out T/Q INPUTS above a given vertical index
          (levels >= input_clamp_from_level).
          Assumes INPUT ordering: [T(0:nlev), Q(0:nlev), ...optional scalars...]

      (2) OUTPUT hard clamp:
          Zero out dT and dQ outputs above a given vertical index
          (levels >= clamp_from_level).
          Assumes OUTPUT ordering: [dT(0:nlev), dQ(0:nlev), ...optional extras...]
    """
    def __init__(
        self,
        C_in: int,
        C_out: int,
        T_window: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        # ---- hard constraint settings ----
        nlev: int = 28,
        clamp_from_level: int = 19,
        input_clamp_from_level: int = 19,
    ):
        super().__init__()

        self.Tw = int(T_window)
        self.nlev = int(nlev)
        self.clamp_from_level = None if clamp_from_level is None else int(clamp_from_level)
        self.input_clamp_from_level = None if input_clamp_from_level is None else int(input_clamp_from_level)

        # dropout in nn.LSTM only works when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=C_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.decoder = nn.Linear(hidden_size, C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Tw, C_in]
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, Tw, C_in], got {tuple(x.shape)}")
        if x.shape[1] != self.Tw:
            raise ValueError(f"Expected Tw={self.Tw}, got x.shape[1]={x.shape[1]}")

        # ---- (1) INPUT hard mask on T/Q above a given layer ----
        if self.input_clamp_from_level is not None:
            k0 = self.input_clamp_from_level
            if not (0 <= k0 <= self.nlev):
                raise ValueError(f"input_clamp_from_level must be in [0, {self.nlev}], got {k0}")
            if x.shape[2] < 2 * self.nlev:
                raise ValueError(
                    f"Expected x to have at least 2*nlev={2*self.nlev} input features for [T,Q]. "
                    f"Got C_in={x.shape[2]}"
                )

            x = x.clone()
            # T block
            x[:, :, k0:self.nlev] = 0.0
            # Q block
            x[:, :, self.nlev + k0:2 * self.nlev] = 0.0

        # LSTM over the full window
        out, (h_n, c_n) = self.lstm(x)   # out: [B, Tw, H], h_n: [num_layers, B, H]

        # use the last time step output
        y = self.decoder(out[:, -1, :])  # [B, C_out]

        # ---- (2) OUTPUT hard constraint on dT/dQ above a given layer ----
        if self.clamp_from_level is not None:
            k0 = self.clamp_from_level
            if not (0 <= k0 <= self.nlev):
                raise ValueError(f"clamp_from_level must be in [0, {self.nlev}], got {k0}")
            if y.shape[1] < 2 * self.nlev:
                raise ValueError(
                    f"Expected y to have at least 2*nlev={2*self.nlev} outputs for [dT,dQ]. "
                    f"Got C_out={y.shape[1]}"
                )

            y = y.clone()
            # dT block
            y[:, k0:self.nlev] = 0.0
            # dQ block
            y[:, self.nlev + k0:2 * self.nlev] = 0.0

        return y






