exp_name = 'gmd_radiative_convective_1420w_10min_v3'
exp_folder_name = 'column_code_with_slab'
print(exp_name, exp_folder_name)

ckpt_dir = f"/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_checkpoints"

# ---- Cycle + hourly saving settings ----
SAVE_EVERY = 50000            # steps per cycle (new folder every 30000 steps)
DT_MINUTES = 10
SAVE_EVERY_MINUTES = 10
SAVE_EVERY_STEPS = SAVE_EVERY_MINUTES // DT_MINUTES  # 1 for 10-min dt
N_CKPTS_PER_CYCLE = 100                 # only first N ckpts per cycle
# which ckpt to resume
checkpoint_path = ckpt_dir + '/cycle_000010/ckpt_20090704_215000.pkl'

from sympl import (
    DataArray, PlotFunctionMonitor,
    AdamsBashforth, NetCDFMonitor, set_constant
)
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

from climt import (
    EmanuelConvection, RRTMGShortwave, RRTMGLongwave, SlabSurface,
    DryConvectiveAdjustment, SimplePhysics, get_default_state
)
import pickle, os

fields_to_store_input = ['air_temperature', 'specific_humidity', 'air_pressure', 'surface_temperature',
                   'air_pressure_on_interface_levels','surface_upward_latent_heat_flux',
                   'surface_upward_sensible_heat_flux','surface_air_pressure',
                   'downwelling_shortwave_flux_in_air',
                   'upwelling_shortwave_flux_in_air',
                   'downwelling_longwave_flux_in_air',
                   'upwelling_longwave_flux_in_air',
                        'eastward_wind',
                        'cloud_base_mass_flux']
fields_to_store_output = ['convective_precipitation_rate',
                   'air_temperature_tendency_from_EmanuelConvection',
                   'specific_humidity_tendency_from_EmanuelConvection',
                   'air_temperature_tendency_from_shortwave',
                   'air_temperature_tendency_from_longwave']
netcdf_monitor_input = NetCDFMonitor(
    f'/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_input.nc',
    write_on_store=True,
    store_names=fields_to_store_input
)
netcdf_monitor_output = NetCDFMonitor(
    f'/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_output.nc',
    write_on_store=True,
    store_names=fields_to_store_output
)

timestep = timedelta(minutes=DT_MINUTES)

set_constant('stellar_irradiance', value=1420, units='W m^-2')

convection = EmanuelConvection(tendencies_in_diagnostics=True)
radiation_sw = RRTMGShortwave()
radiation_lw = RRTMGLongwave()
slab = SlabSurface()
simple_physics = SimplePhysics()
dry_convection = DryConvectiveAdjustment()

time_stepper = AdamsBashforth([convection, radiation_lw, radiation_sw, slab])

# --------------------------------------------------
# Load or initialize state
# --------------------------------------------------

if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    state, start_i = data["state"], data["i"]
else:
    start_i = 0
    state = get_default_state([simple_physics, convection,
                               radiation_lw, radiation_sw, slab])
    state['air_temperature'].values[:] = 270
    state['surface_albedo_for_direct_shortwave'].values[:] = 0.5
    state['surface_albedo_for_direct_near_infrared'].values[:] = 0.5
    state['surface_albedo_for_diffuse_shortwave'].values[:] = 0.5
    state['zenith_angle'].values[:] = np.pi/2.5
    state['surface_temperature'].values[:] = 280.
    state['ocean_mixed_layer_thickness'].values[:] = 5
    state['area_type'].values[:] = 'sea'
    
    # state['eastward_wind'].values[:] = 3.
    # state['northward_wind'].values[:] = 3.    

# --------------------------------------------------
# Checkpoint utilities
# --------------------------------------------------
def atomic_pickle_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def get_cycle_idx(i):
    return i // SAVE_EVERY

def get_cycle_dir(cycle_idx):
    return os.path.join(ckpt_dir, f"cycle_{cycle_idx:06d}")

def format_time_for_filename(dt):
    # dt is datetime.datetime
    return dt.strftime("%Y%m%d_%H%M%S")

current_cycle_idx = None
current_cycle_dir = None
saved_in_cycle = 0

def rotate_cycle_if_needed(i):
    global current_cycle_idx, current_cycle_dir, saved_in_cycle
    cidx = get_cycle_idx(i)
    if current_cycle_idx != cidx:
        current_cycle_idx = cidx
        current_cycle_dir = get_cycle_dir(cidx)
        os.makedirs(current_cycle_dir, exist_ok=True)
        saved_in_cycle = 0
        print(f"[ckpt] New cycle {cidx} -> {current_cycle_dir}")

def maybe_save_checkpoint(state, i):
    global saved_in_cycle, current_cycle_dir

    if saved_in_cycle >= N_CKPTS_PER_CYCLE:
        return

    model_time = state['time']   # datetime.datetime
    time_str = format_time_for_filename(model_time)

    fname = os.path.join(current_cycle_dir, f"ckpt_{time_str}.pkl")
    atomic_pickle_dump({"state": state, "i": i}, fname)

    saved_in_cycle += 1
    # print(f"[ckpt] saved ({saved_in_cycle}/{N_HOURLY_CKPTS}) {fname}")


os.makedirs(ckpt_dir, exist_ok=True)
rotate_cycle_if_needed(start_i)

# after you load/init state:
state['eastward_wind'].attrs['units'] = 'm s^-1'
# state['northward_wind'].attrs['units'] = 'm s^-1'
# --------------------------------------------------
# Main integration loop
# --------------------------------------------------
toa_history = []

for i in range(start_i, 100000000):

    netcdf_monitor_input.store(state)
    
    if i != start_i and i % SAVE_EVERY == 0:
        rotate_cycle_if_needed(i)
    # save every 10-min step
    if (i + 1) % SAVE_EVERY_STEPS == 0:
        maybe_save_checkpoint(state, i)
    
    diagnostics, state = time_stepper(state, timestep)
    state.update(diagnostics)

    diagnostics, new_state = simple_physics(state, timestep)
    state.update(diagnostics)
    state.update(new_state)
 
    netcdf_monitor_output.store(state)

    state['time'] += timestep
    state['eastward_wind'].values[:] = 3.
    
    # print(state)
    # ---- Equilibrium diagnostics ----
    toa_net = (
        state['downwelling_shortwave_flux_in_air'][-1]
        - state['upwelling_shortwave_flux_in_air'][-1]
        + state['downwelling_longwave_flux_in_air'][-1]
        - state['upwelling_longwave_flux_in_air'][-1]
    )

    toa_history.append(float(toa_net.mean()))

    if (i + 1) % 100 == 0:
        print(
            state["time"],
            f"toa_net: {float(toa_net.mean()):.3f}",
            f"mean_t: {state['air_temperature'].mean().item():.3f}",
            f"mean_sphum: {1000 * state['specific_humidity'].mean().item():.6f}",
            f"e_wind: {state['eastward_wind'].mean().item():.3f}",
            f"cloud_flux: {state['cloud_base_mass_flux'].mean().item():.3f}",
            f"dT/dt (lw,sw,conv): "
            f"{state['air_temperature_tendency_from_longwave'].mean().item():.3f}, "
            f"{state['air_temperature_tendency_from_shortwave'].mean().item():.3f}, "
            f"{state['air_temperature_tendency_from_convection'].mean().item():.3f}, "
            f"Surface_T: {state['surface_temperature'].values.item():.3f}"
        )

