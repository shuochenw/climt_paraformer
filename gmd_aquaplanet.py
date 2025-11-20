exp_name = '64x32'

import climt
from sympl import (
    PlotFunctionMonitor, NetCDFMonitor,
    TimeDifferencingWrapper, UpdateFrequencyWrapper,
    set_constant
)
import numpy as np
from datetime import timedelta
import gfs_dynamical_core
import xarray as xr
import os
import pickle

def plot_function(fig, state):

    ax = fig.add_subplot(2, 2, 1)
    state['specific_humidity'].mean(
        dim='lon').plot.contourf(
            ax=ax, levels=16, robust=True)
    ax.set_title('Specific Humidity')

    ax = fig.add_subplot(2, 2, 3)
    state['eastward_wind'].mean(dim='lon').plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title('Zonal Wind')

    ax = fig.add_subplot(2, 2, 2)
    state['air_temperature_tendency_from_convection'].transpose().mean(
        dim='lon').plot.contourf(
        ax=ax, levels=16, robust=True)
    ax.set_title('Conv. Heating Rate')

    ax = fig.add_subplot(2, 2, 4)
    state['air_temperature'].mean(dim='lon').plot.contourf(
        ax=ax, levels=16)
    ax.set_title('Temperature')
    fig.tight_layout()
    fig.savefig(f'/projects/sds-lab/Shuochen/climt/{exp_name}.png')

# vars
fields_to_store = ['air_temperature', 'specific_humidity', 'air_pressure', 
                   'air_pressure_on_interface_levels','surface_upward_latent_heat_flux',
                   'surface_upward_sensible_heat_flux','surface_air_pressure',
                   'downwelling_shortwave_flux_in_air', #input
                   'convective_precipitation_rate', #diag
                   'air_temperature_tendency_from_convection', #output, but no spec_hum_from_convection?,
                   'upwelling_shortwave_flux_in_air', 'downwelling_longwave_flux_in_air','upwelling_longwave_flux_in_air', # others
                   'latitude', 'longitude']
# Create plotting object
monitor = PlotFunctionMonitor(plot_function)
netcdf_monitor = NetCDFMonitor(f'/projects/sds-lab/Shuochen/climt/{exp_name}.nc',write_on_store=True,store_names=fields_to_store)
set_constant('stellar_irradiance', value=200, units='W m^-2')
model_time_step = timedelta(minutes=10)
# Create components
convection = climt.EmanuelConvection()
simple_physics = TimeDifferencingWrapper(climt.SimplePhysics())
radiation_step = timedelta(hours=1)
radiation_lw = UpdateFrequencyWrapper(
    climt.RRTMGLongwave(), radiation_step)
radiation_sw = UpdateFrequencyWrapper(
    climt.RRTMGShortwave(), radiation_step)
slab_surface = climt.SlabSurface()
dycore = gfs_dynamical_core.GFSDynamicalCore(
    [simple_physics, slab_surface, radiation_sw,
     radiation_lw, convection], number_of_damped_levels=5
)
grid = climt.get_grid(nx=64, ny=32)


# load state from checkpoint
checkpoint_path = f'/projects/sds-lab/Shuochen/climt/{exp_name}_checkpoint.pkl'
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    my_state, start_i = data["state"], data["i"]
else:
    start_i = 0
    # Create model state
    my_state = climt.get_default_state([dycore], grid_state=grid)
    # Set initial/boundary conditions
    latitudes = my_state['latitude'].values
    longitudes = my_state['longitude'].values
    zenith_angle = np.radians(latitudes)
    surface_shape = latitudes.shape
    my_state['zenith_angle'].values = zenith_angle
    my_state['eastward_wind'].values[:] = np.random.randn(
        *my_state['eastward_wind'].shape)
    my_state['ocean_mixed_layer_thickness'].values[:] = 50
    surf_temp_profile = 290 - (40*np.sin(zenith_angle)**2)
    my_state['surface_temperature'].values = surf_temp_profile
# save checkpoint
def save_checkpoint(state, i, filename=checkpoint_path):
    with open(filename, "wb") as f:
        pickle.dump({"state": state, "i": i}, f)
    print(f"--- checkpoint saved at iteration {i} ---")



# loop
toa_history = []
for i in range(start_i, 1500*24*6):
    diag, my_state = dycore(my_state, model_time_step)
    my_state.update(diag)
    my_state['time'] += model_time_step
    
    # ---- EQUILIBRIUM CHECKING ----
    # Compute TOA net radiation
    toa_net = (
        diag['downwelling_shortwave_flux_in_air'][:, -1]
        - diag['upwelling_shortwave_flux_in_air'][:, -1]
        + diag['downwelling_longwave_flux_in_air'][:, -1]
        - diag['upwelling_longwave_flux_in_air'][:, -1]
    )
    toa_history.append(float(toa_net.mean()))
    
    # if i % (6*24) == 0:
    netcdf_monitor.store(my_state)
    monitor.store(my_state)
    save_checkpoint(my_state, i)
    print('max. zonal wind: ', np.amax(my_state['eastward_wind'].values))
    print('max. humidity: ', np.amax(my_state['specific_humidity'].values))
    print('max. surf temp: ', np.amax(my_state['surface_temperature'].values))
        
    print(my_state['time'], float(toa_net.mean()))
