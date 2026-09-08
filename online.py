exp_name = 'gmd_radiative_convective_1420w_10min_v3' #'64x32_4h_400w' 'gmd_radiative_convective_1420w_10min'
exp_folder_name = 'column_code_with_slab' # 'column_code_with_slab' 'gmd_aquaplanet' 'column_rrtmg' 'aquaplanet_RCE_single'
emulate = 'conv' # 'conv' 'rad'
model_name = 'mlp_v2' # 'mlp' 'mlp_v2' 'lstm','trsfm_v2'
best_trial_number = '0'

ml_exp_name = 'best_model'+ '_' + emulate + '_' + model_name
best_model_path = f"./{exp_folder_name}/{ml_exp_name}/trial_{best_trial_number}.pth"
# norm_path = f'/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_{emulate}/normalization.pth'
import climt
from sympl import DataArray, TendencyComponent, AdamsBashforth, ImplicitTendencyComponent
from sympl import (
    PlotFunctionMonitor, NetCDFMonitor,
    TimeDifferencingWrapper, UpdateFrequencyWrapper,
    set_constant, get_constant, initialize_numpy_arrays_with_properties
)
import gfs_dynamical_core
from datetime import timedelta, datetime
import numpy as np
import torch
import torch.nn as nn
import sys, os, pickle, re
from climt import (
    EmanuelConvection, RRTMGShortwave, RRTMGLongwave, SlabSurface,
    DryConvectiveAdjustment, SimplePhysics, get_default_state
)
from datetime import timedelta
import xarray as xr
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
sys.path.append("..")
from models import DynamicMLP_flatten, DynamicMLP_flatten_v2, WindowTemporalLSTM, FlatTemporalTransformer2, WindowTemporalTransformer, WindowTemporalTransformer_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm = torch.load(f'/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_{emulate}/normalization.pth', weights_only=True,map_location=device)
X_mean = norm['X_mean'] # [C]
X_std  = norm['X_std']
X_max  = norm['X_max']
X_min  = norm['X_min']
y_mean = norm['y_mean']
y_std  = norm['y_std']
y_max  = norm['y_max']
y_min  = norm['y_min']

X_std_safe = X_std.clone()
X_std_safe[X_std_safe == 0.0] = 1.0
y_std_safe = y_std.clone()
y_std_safe[y_std_safe == 0.0] = 1.0

X_den = (X_max - X_min).clone()
X_den[X_den == 0.0] = 1.0
y_den = (y_max - y_min).clone()
y_den[y_den == 0.0] = 1.0

def log_transform(x, eps, scale):
    return torch.sign(x) * torch.log1p(torch.abs(x) / eps) / scale
def inv_log_transform(x_n, eps, scale):
    # inverse of sign(x)*log1p(|x|/eps)/10
    return torch.sign(x_n) * torch.expm1(torch.abs(x_n) * scale) * eps

def check_state(state, step):
    for k, v in state.items():

        # Extract raw numpy array
        if hasattr(v, "values"):
            arr = v.values
        elif isinstance(v, np.ndarray):
            arr = v
        else:
            continue  # skip non-numeric entries (time, metadata, etc.)

        # Skip non-numeric dtypes
        if not np.issubdtype(arr.dtype, np.number):
            continue

        if not np.all(np.isfinite(arr)):
            bad = np.sum(~np.isfinite(arr))
            raise RuntimeError(
                f"NaN/Inf in '{k}' at step {step} "
                f"(bad count={bad})"
            )

class NNParameterization(TendencyComponent):
    
    input_properties = {
        'air_temperature': {'dims': ['*', 'mid_levels'], 'units': 'degK'},
        'specific_humidity': {'dims': ['*', 'mid_levels'], 'units': 'kg/kg'},
        'surface_air_pressure': {'dims': ['*'], 'units': 'Pa'},
        'surface_upward_latent_heat_flux': {'dims': ['*'], 'units': 'W m^-2'},
        'surface_upward_sensible_heat_flux': {'dims': ['*'], 'units': 'W m^-2'},
        'cloud_base_mass_flux': {'dims': ['*'], 'units': 'kg m^-2 s^-1'},
    }
    tendency_properties = {}
    if emulate == 'rad':
        tendency_properties['air_temperature'] = {'units': 'degK day^-1'}
    if emulate == 'conv':
        tendency_properties['air_temperature'] = {'units': 'degK s^-1'}
        tendency_properties['specific_humidity'] = {'units': 'kg/kg s^-1'}

    diagnostic_properties = {}
    if emulate == 'rad':
        diagnostic_properties['air_temperature_tendency_from_NN'] = {'dims': ['*', 'mid_levels'],'units': 'degK day^-1'}
    if emulate == 'conv':
        diagnostic_properties['air_temperature_tendency_from_NN'] = {'dims': ['*', 'mid_levels'],'units': 'degK s^-1'}
        diagnostic_properties['specific_humidity_tendency_from_NN'] = {'dims': ['*', 'mid_levels'],'units': 'kg/kg s^-1'}
        diagnostic_properties['convective_precipitation_rate'] = {'dims': ['*'],'units': 'mm day^-1'}
        
    def __init__(self, top_k=5, emulate_layer=28, **kwargs):
        super(NNParameterization, self).__init__(**kwargs)
            
        self.emulate_layer = emulate_layer
        self.top_k = top_k
        
        if emulate == 'rad':
            self.true = RRTMGLongwave()
        # if emulate == 'conv':
        #     self.true = EmanuelConvection(tendencies_in_diagnostics=True)

        if exp_folder_name == 'gmd_aquaplanet' and emulate == 'rad':
            self.IN_FEATURES, self.OUT_FEATURES = 59, 28
        if exp_folder_name == 'gmd_aquaplanet' and emulate == 'conv':
            self.IN_FEATURES, self.OUT_FEATURES = 59-(28-self.emulate_layer)*2, 57-(28-self.emulate_layer)*2
        if exp_folder_name == 'column_code_with_slab' and emulate == 'rad':
            self.IN_FEATURES, self.OUT_FEATURES = 59, 28
        if exp_folder_name == 'column_code_with_slab' and emulate == 'conv':
            self.IN_FEATURES, self.OUT_FEATURES = 59-(28-self.emulate_layer)*2, 57-(28-self.emulate_layer)*2

        ckpt = torch.load(best_model_path, map_location=device)

        if model_name == 'mlp':
            self.model = DynamicMLP_flatten(self.IN_FEATURES,self.OUT_FEATURES,ckpt["hidden_sizes"]).to(device)
            self.Tw = 1
        if model_name == 'mlp_v2':
            self.model = DynamicMLP_flatten_v2(self.IN_FEATURES,self.OUT_FEATURES,ckpt["hidden_sizes"]).to(device)
            self.Tw = 1    
        if 'lstm' in model_name:
            self.Tw = int(ckpt["T_window"])   # <-- store window
            self.model = WindowTemporalLSTM(
                    C_in=self.IN_FEATURES,
                    C_out=self.OUT_FEATURES,
                    T_window=ckpt["T_window"],
                    hidden_size=ckpt["hidden_size"],
                    num_layers=ckpt["num_layers"],
                    dropout=ckpt["dropout"],
                    clamp_from_level=19,
                    input_clamp_from_level=19,
                ).to(device)
    
        if 'trsfm' in model_name:
            self.Tw = int(ckpt["T_window"])   # <-- store window
            self.model = WindowTemporalTransformer_v2(
                    C_in=self.IN_FEATURES,
                    C_out=self.OUT_FEATURES,
                    T_window=ckpt["T_window"],
                    d_model=ckpt["d_model"],
                    nhead=ckpt["nhead"],
                    num_layers=ckpt["num_layers"],
                    dim_feedforward=ckpt["dim_feedforward"],
                    dropout=ckpt["dropout"],
                    causal=True,
                    # load constraint metadata saved in training
                    nlev=28,
                    clamp_from_level=ckpt.get("clamp_from_level", None),
                    input_clamp_from_level=ckpt.get("input_clamp_from_level", None),
                ).to(device)
        
        self._state_buffer = deque(maxlen=self.Tw)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        
    def _snapshot_state(self, state):
        """Store only fields you need as numpy arrays (cheap & safe vs keeping DataArrays)."""
        snap = {}
        for k in self.input_properties.keys():
            # state[k] is usually a DataArray; store numpy copy to freeze time
            v = state[k]
            snap[k] = np.array(v.values, copy=True)
        return snap

    def _to_cols_lev(self, A):
        """
        Convert array to (cols, lev).
        Accepts:
          (lev,) -> (1, lev)
          (cols, lev) -> unchanged
          (lev, lat, lon) -> (lat*lon, lev)
        """
        A = np.asarray(A)
    
        if A.ndim == 1:          # (lev,)
            return A[None, :]
        if A.ndim == 2:          # (cols, lev)
            return A
        if A.ndim == 3:          # (lev, lat, lon)
            lev, lat, lon = A.shape
            return A.reshape(lev, lat * lon).T
        raise ValueError(f"Unsupported shape {A.shape}")
    
    def _to_cols_1d(self, A):
        """
        Convert to (cols,) to match flattened columns.
        Accepts:
          (cols,) -> unchanged
          (lat, lon) -> flatten
          (1,1) -> (1,)
        """
        A = np.asarray(A)
        if A.ndim == 1:
            return A
        return A.reshape(-1)
        
    def __call__(self, state, **kwargs):
        
        self._state_buffer.append(self._snapshot_state(state))
    
        # 1) get "true" convection tendencies (DataArrays)
        if emulate == 'rad':
            true_tend, true_diag = self.true(state)
        # if emulate == 'conv':
        #     true_tend, true_diag = self.true(state, timedelta(minutes=10))
        # 2) get NN tendencies (DataArrays) by running numpy inference on the same state
        nn_tend, nn_diag = super().__call__(state)  # this will call array_call()
    
        # 3) blend: overwrite TOP layers with true values
        if emulate == 'rad':
            nn_tend['air_temperature'].isel(mid_levels=slice(-self.top_k, None))[:] = true_tend['air_temperature'].isel(mid_levels=slice(-self.top_k, None))
        # if emulate == 'conv':       
        #     nn_tend['air_temperature'].isel(mid_levels=slice(self.emulate_layer, None))[:] = \
        #     true_tend['air_temperature'].isel(mid_levels=slice(self.emulate_layer, None))
        #     nn_tend['specific_humidity'].isel(mid_levels=slice(self.emulate_layer, None))[:] = \
        #     true_tend['specific_humidity'].isel(mid_levels=slice(self.emulate_layer, None))

        return nn_tend, nn_diag

    def array_call(self, state):
        # Build time window snapshots (numpy dicts)
        if len(self._state_buffer) == 0:
            snaps = [self._snapshot_state(state)]
        else:
            snaps = list(self._state_buffer)
    
        if len(snaps) < self.Tw:
            snaps = [snaps[0]] * (self.Tw - len(snaps)) + snaps
    
        # Use last snapshot to infer cols/lev
        last = snaps[-1]
        T_last = self._to_cols_lev(last["air_temperature"])   # (cols, lev)
        num_cols, num_levs = T_last.shape
    
        tendencies = initialize_numpy_arrays_with_properties(
            self.tendency_properties, state, self.input_properties
        )
        diagnostics = initialize_numpy_arrays_with_properties(
            self.diagnostic_properties, state, self.input_properties
        )
    
        # ---- build x_seq: [Tw, cols, C_in] ----
        x_list = []
        for s in snaps:
            T = self._to_cols_lev(s["air_temperature"])        # (cols, lev)
            q = self._to_cols_lev(s["specific_humidity"])      # (cols, lev)
    
            if emulate == "conv":
                T = T[:, :self.emulate_layer]
                q = q[:, :self.emulate_layer]
    
            ps = self._to_cols_1d(s["surface_air_pressure"])               # (cols,)
            lh = self._to_cols_1d(s["surface_upward_latent_heat_flux"])    # (cols,)
            sh = self._to_cols_1d(s["surface_upward_sensible_heat_flux"])  # (cols,)
            cld = self._to_cols_1d(s["cloud_base_mass_flux"])  # (cols,)
    
            if exp_folder_name == "gmd_aquaplanet":
                x_t = np.concatenate([T, q, ps[:, None], lh[:, None], sh[:, None], cld[:, None]], axis=-1)
            else:
                x_t = np.concatenate([T, q, lh[:, None], sh[:, None], cld[:, None]], axis=-1)
    
            # safety check
            if x_t.shape[0] != num_cols:
                raise ValueError(f"Column mismatch: x_t has {x_t.shape[0]} cols, expected {num_cols}")
    
            x_list.append(x_t)
    
        x_seq = np.stack(x_list, axis=0)  # (Tw, cols, C_in)
        # torch + normalize
        x = torch.tensor(x_seq, dtype=torch.float32, device=device)
        x = (x - X_min) / X_den
    
        # inference
        with torch.no_grad():
            if "mlp" in model_name:
                y = self.model(x[-1])  # (cols, C_out)
            else:
                # WindowTemporalTransformer expects [B, Tw, C] with B = cols
                x_cols = x.permute(1, 0, 2).contiguous()   # [cols, Tw, C_in]
                y = self.model(x_cols)                     # [cols, C_out]
    
        # denorm output
        y = y * y_std_safe + y_mean 
        y = y.detach().cpu().numpy()
        # ---- map outputs back ----
        if emulate == "rad":
            tendencies["air_temperature"][:] = y[:, :num_levs]
            diagnostics["air_temperature_tendency_from_NN"][:] = y[:, :num_levs]
        else:
            L = self.emulate_layer
            tendencies["air_temperature"][..., :L] = y[:, :L]
            diagnostics["air_temperature_tendency_from_NN"][..., :L] = y[:, :L]
            tendencies["specific_humidity"][..., :L] = y[:, L:2*L]
            diagnostics["specific_humidity_tendency_from_NN"][..., :L] = y[:, L:2*L]
            diagnostics["convective_precipitation_rate"][:] = y[:, -1]
    
        return tendencies, diagnostics

scheme = 'physics' #'nn' 'physics'
if scheme == 'physics':
    model_name = 'physics'
online_path = f'/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_{emulate}'
exp_name_online = '1'
cycle_folder = f'/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_checkpoints/cycle_000019'
# read Tw
ckpt = torch.load(best_model_path, map_location=device)
if "trsfm" in model_name or "lstm" in model_name:
    Tw = int(ckpt["T_window"])
else:
    Tw = 1
    
def list_files_in_folder_sorted(directory_path):
    p = Path(directory_path)
    files = sorted(
        [str(file.resolve()) for file in p.iterdir() if file.is_file()]
    )
    return files
state_path = list_files_in_folder_sorted(cycle_folder)
# choose number of previous states for transformer
state_path = state_path[-Tw:]

fields_to_store_input = ['air_temperature', 'specific_humidity', 'air_pressure', 'surface_temperature',
                   'air_pressure_on_interface_levels','surface_upward_latent_heat_flux',
                   'surface_upward_sensible_heat_flux','surface_air_pressure',
                   'downwelling_shortwave_flux_in_air',
                   'upwelling_shortwave_flux_in_air',
                   'downwelling_longwave_flux_in_air',
                   'upwelling_longwave_flux_in_air','eastward_wind','cloud_base_mass_flux']
if scheme == 'nn':
    fields_to_store_output = ['air_temperature_tendency_from_NN',
                              'specific_humidity_tendency_from_NN',
                              'convective_precipitation_rate']
if scheme == 'physics':
    fields_to_store_output = ['air_temperature_tendency_from_EmanuelConvection',
                              'specific_humidity_tendency_from_EmanuelConvection',
                              'air_temperature_tendency_from_shortwave',
                              'air_temperature_tendency_from_longwave',
                              'convective_precipitation_rate']
netcdf_monitor_input = NetCDFMonitor(
    online_path + f'/online_input_{model_name}_{exp_name_online}.nc',
    write_on_store=True,
    store_names=fields_to_store_input
)
netcdf_monitor_output = NetCDFMonitor(
    online_path + f'/online_output_{model_name}_{exp_name_online}.nc',
    write_on_store=True,
    store_names=fields_to_store_output
)

state_list = []
for checkpoint_path in state_path:
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
        state = data["state"]
        state_list.append(state)
print(f"Loaded {len(state_list)} states. Time: {state['time']}")

state = state_list[-1]

if exp_folder_name == 'column_code_with_slab':
    set_constant('stellar_irradiance', value=1420, units='W m^-2')
    timestep = timedelta(minutes=10)
    convection = EmanuelConvection(tendencies_in_diagnostics=True)
    radiation_sw = RRTMGShortwave()
    radiation_lw = RRTMGLongwave()
    slab = SlabSurface()
    simple_physics = SimplePhysics()
    
    if 'trsfm' in model_name or 'lstm' in model_name:
        # nn component
        nn_component = NNParameterization()
        nn_component._state_buffer.clear()
        Tw = nn_component.Tw
        # Need previous Tw-1 states; current state will be appended inside nn_component(state)
        prev = state_list[-(Tw-1):] if Tw > 1 else []
        # If not enough history, left-pad using the oldest available
        if Tw > 1 and len(prev) < (Tw - 1):
            if len(state_list) == 0:
                raise ValueError("state_list is empty; cannot warm start transformer.")
            pad = [state_list[0]] * ((Tw - 1) - len(prev))
            prev = pad + prev
        for s in prev:
            nn_component._state_buffer.append(nn_component._snapshot_state(s))
        # now call once (this appends current internally)
        tend, diag = nn_component(state)
        
    if 'mlp' in model_name:
        nn_component = NNParameterization()
    
    if scheme == 'physics':
        time_stepper = AdamsBashforth([convection, radiation_lw, radiation_sw, slab])
    if scheme == 'nn':
        if emulate == 'conv':
            time_stepper = AdamsBashforth([nn_component, radiation_lw, radiation_sw, slab])
        if emulate == 'rad':
            time_stepper = AdamsBashforth([convection, nn_component, radiation_sw, slab])

    state['eastward_wind'].attrs['units'] = 'm s^-1'
    # state['northward_wind'].attrs['units'] = 'm s^-1'
    for i in range(525600):

        netcdf_monitor_input.store(state)
        
        diagnostics, state = time_stepper(state, timestep)
        state.update(diagnostics)
        diagnostics, new_state = simple_physics(state, timestep)
        state.update(diagnostics)    
        state.update(new_state)
        
        netcdf_monitor_output.store(state)
        
        state['time'] += timestep
        state['eastward_wind'].values[:] = 3.
        
        if (i+1) % 1000 == 0:
            print(state['surface_temperature'].values)
        
        # netcdf_monitor.store(state)
        check_state(state, i)

# state['ocean_mixed_layer_thickness'].values[:] = 50