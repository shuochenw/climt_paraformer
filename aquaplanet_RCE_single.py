exp_name = 'test'
exp_folder_name = 'aquaplanet_RCE_single'

from sympl import (
    PlotFunctionMonitor, AdamsBashforth, NetCDFMonitor, set_constant
)

from climt import SimplePhysics, get_default_state
import numpy as np
from datetime import timedelta
from climt import EmanuelConvection, RRTMGShortwave, RRTMGLongwave, SlabSurface
import matplotlib.pyplot as plt
import pickle

def plot_function(fig, state):
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(
        state['air_temperature_tendency_from_EmanuelConvection'].to_units(
            'degK s^-1').values.flatten(),
        state['air_pressure'].to_units('mbar').values.flatten(), '-o')
    ax.set_title('Conv. heating rate')
    ax.set_xlabel('K/s')
    ax.set_ylabel('millibar')
    ax.grid()

    ax.axes.invert_yaxis()
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(
        state['air_temperature'].values.flatten(),
        state['air_pressure'].to_units('mbar').values.flatten(), '-o')
    ax.set_title('Air temperature')
    ax.axes.invert_yaxis()
    ax.set_xlabel('K')
    ax.grid()

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(
        state['air_temperature_tendency_from_longwave'].values.flatten(),
        state['air_pressure'].to_units('mbar').values.flatten(), '-o',
        label='LW')
    ax.plot(
        state['air_temperature_tendency_from_shortwave'].values.flatten(),
        state['air_pressure'].to_units('mbar').values.flatten(), '-o',
        label='SW')
    ax.set_title('LW and SW Heating rates')
    ax.legend()
    ax.axes.invert_yaxis()
    ax.set_xlabel('K/day')
    ax.grid()
    ax.set_ylabel('millibar')

    ax = fig.add_subplot(2, 2, 4)
    net_flux = (state['upwelling_longwave_flux_in_air'] +
                state['upwelling_shortwave_flux_in_air'] -
                state['downwelling_longwave_flux_in_air'] -
                state['downwelling_shortwave_flux_in_air'])
    ax.plot(
        net_flux.values.flatten(),
        state['air_pressure_on_interface_levels'].to_units(
            'mbar').values.flatten(), '-o')
    ax.set_title('Net Flux')
    ax.axes.invert_yaxis()
    ax.set_xlabel('W/m^2')
    ax.grid()
    plt.tight_layout()

# monitor = PlotFunctionMonitor(plot_function)

timestep = timedelta(minutes=10)

convection = EmanuelConvection(tendencies_in_diagnostics=True)
radiation_sw = RRTMGShortwave()
radiation_lw = RRTMGLongwave()
slab = SlabSurface()
simple_physics = SimplePhysics()

store_quantities = ['air_temperature',
                    'air_pressure',
                    'specific_humidity',
                    'air_pressure_on_interface_levels',
                    'surface_upward_latent_heat_flux',
                    'surface_upward_sensible_heat_flux',
                    'surface_air_pressure',
                    'convective_precipitation_rate',
                    'air_temperature_tendency_from_EmanuelConvection', 
                    'specific_humidity_tendency_from_EmanuelConvection',
                    'air_temperature_tendency_from_longwave',
                    'air_temperature_tendency_from_shortwave', 
                    'downwelling_shortwave_flux_in_air',
                    'upwelling_shortwave_flux_in_air', 
                    'downwelling_longwave_flux_in_air',
                    'upwelling_longwave_flux_in_air']

netcdf_monitor = NetCDFMonitor(f'/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}.nc',
                               store_names=store_quantities,
                               write_on_store=True)

convection.current_time_step = timestep

state = get_default_state([simple_physics, convection,
                           radiation_lw, radiation_sw, slab])

set_constant('stellar_irradiance', value=350, units='W m^-2')

state['air_temperature'].values[:] = 270
# state['surface_albedo_for_direct_shortwave'].values[:] = 0.5
# state['surface_albedo_for_direct_near_infrared'].values[:] = 0.5
# state['surface_albedo_for_diffuse_shortwave'].values[:] = 0.5

state['zenith_angle'].values[:] = np.pi/2.5 
state['surface_temperature'].values[:] = 290.
state['ocean_mixed_layer_thickness'].values[:] = 10
# state['area_type'].values[:] = 'sea'

time_stepper = AdamsBashforth([convection, radiation_lw, radiation_sw, slab])

# save checkpoint
checkpoint_path = f'/projects/sds-lab/Shuochen/climt/{exp_folder_name}/{exp_name}_checkpoint.pkl'
def save_checkpoint(state, i, filename=checkpoint_path):
    with open(filename, "wb") as f:
        pickle.dump({"state": state, "i": i}, f)
    print(f"--- checkpoint saved at iteration {i} ---")

for i in range(300000):
    convection.current_time_step = timestep
    diagnostics, state = time_stepper(state, timestep)
    state.update(diagnostics)
    diagnostics, new_state = simple_physics(state, timestep)
    state.update(diagnostics)
    if (i+1) % 20 == 0:
        # monitor.store(state)
        netcdf_monitor.store(state)
    if (i+1) % 200 == 0:
        print(
            f"{state['time']}, "
            f"{state['surface_temperature'].values.item():.2f}, "
            f"{state['surface_upward_sensible_heat_flux'].values.item():.2f}, "
            f"{state['surface_upward_latent_heat_flux'].values.item():.2f}"
        )
        save_checkpoint(state, i)
    state.update(new_state)
    state['time'] += timestep
    state['eastward_wind'].values[:] = 3.

    if not np.isfinite(state['air_temperature'].values).all():
        print("NaNs detected, stopping")
        break


