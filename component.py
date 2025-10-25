# -*- coding: utf-8 -*-
from ..._core import ensure_contiguous_state
from sympl import (
    TendencyComponent, get_constant, initialize_numpy_arrays_with_properties)
import numpy as np
import logging


class ParaformerConvection(TendencyComponent):
    """
    Skeleton code to integrate Paraformer

    """

    input_properties = {
        'air_temperature': {
            'dims': ['*', 'mid_levels'],
            'units': 'degK',
        },
        'specific_humidity': {
            'dims': ['*', 'mid_levels'],
            'units': 'kg/kg',
        },
        # 'eastward_wind': {
        #     'dims': ['*', 'mid_levels'],
        #     'units': 'm s^-1',
        # },
        # 'northward_wind': {
        #     'dims': ['*', 'mid_levels'],
        #     'units': 'm s^-1',
        # },
        'air_pressure': {
            'dims': ['*', 'mid_levels'],
            'units': 'mbar',
        },
        'air_pressure_on_interface_levels': {
            'dims': ['*', 'interface_levels'],
            'units': 'mbar',
        },
        'surface_upward_latent_heat_flux': {
            'dims': ['*'],
            'units': 'W m^-2',
        },
        'surface_upward_sensible_heat_flux': {
            'dims': ['*'],
            'units': 'W m^-2',
        },
        'surface_air_pressure': {
            'dims': ['*'],
            'units': 'Pa',
        },
        'downwelling_shortwave_flux_in_air': { # Insolation
            'dims': ['*', 'interface_levels'],
            'units': 'W m^-2',
        },
        
    }

    diagnostic_properties = {
        'convective_precipitation_rate': { # PRECC
            'dims': ['*'],
            'units': 'mm day^-1',
        },
    }

    tendency_properties = { ''' These below can be added as outputs based on the v2 (Table A2) output variables '''
        'air_temperature': {'units': 'degK s^-1'},
        'specific_humidity': {'units': 'kg/kg s^-1'},
        # 'eastward_wind': {'units': 'm s^-2'},
        # 'northward_wind': {'units': 'm s^-2'},
    }

    def __init__(self, 
                #  minimum_convecting_layer=1,
                #  autoconversion_water_content_threshold=0.0011,
                #  autoconversion_temperature_threshold=-55,
                #  entrainment_mixing_coefficient=1.5,
                #  downdraft_area_fraction=0.05,
                #  precipitation_fraction_outside_cloud=0.12,
                #  speed_water_droplets=50.0,
                #  speed_snow=5.5,
                #  rain_evaporation_coefficient=1.0,
                #  snow_evaporation_coefficient=0.8,
                #  convective_momentum_transfer_coefficient=0.7,
                #  downdraft_surface_velocity_coefficient=10.0,
                #  convection_bouyancy_threshold=0.9,
                #  mass_flux_relaxation_rate=0.1,
                #  mass_flux_damping_rate=0.1,
                #  reference_mass_flux_timescale=300.,
                 **kwargs):
        """

        Based on what parameters are needed during inference to initialize your code, you can pass all such parameters using arguments to __init__
        This could also be, for example, the file name from which the model must be read.
        
        These are the parameters in your training code
        SEQUENCE_SIZE = 5
        LEARNING_RATE = 1e-4
        # set BATCH_SIZE for 32 for demostration, in climsim is 512
        BATCH_SIZE = 32
        # run for 10 epochs for demostration, the actual epoch is 200
        EPOCHS = 10
        IN_FEATURES = 60*4
        OUT_FEATURES = 60
        D_MODEL = 256
        # DROPOUT = 0.2
        N_HEAD = 4
        N_LAYER = 6
        MAX_LEN = SEQUENCE_SIZE

        """

        # if (convective_momentum_transfer_coefficient < 0 or
        #         convective_momentum_transfer_coefficient > 1):
        #     raise ValueError("Momentum transfer coefficient must be between 0 and 1.")

        # if (downdraft_area_fraction < 0 or
        #         downdraft_area_fraction > 1):
        #     raise ValueError("Downdraft fraction must be between 0 and 1.")

        # if (precipitation_fraction_outside_cloud < 0 or
        #         precipitation_fraction_outside_cloud > 1):
        #     raise ValueError("Outside cloud precipitation fraction must be between 0 and 1.")

        # self._con_mom_txfr = convective_momentum_transfer_coefficient
        # self._downdraft_area_frac = downdraft_area_fraction
        # self._precip_frac_outside_cloud = precipitation_fraction_outside_cloud
        # self._min_conv_layer = minimum_convecting_layer
        # self._crit_humidity = autoconversion_water_content_threshold
        # self._crit_temp = autoconversion_temperature_threshold
        # self._entrain_coeff = entrainment_mixing_coefficient
        # self._droplet_speed = speed_water_droplets
        # self._snow_speed = speed_snow
        # self._rain_evap = rain_evaporation_coefficient
        # self._snow_evap = snow_evaporation_coefficient
        # self._beta = downdraft_surface_velocity_coefficient
        # self._dtmax = convection_bouyancy_threshold
        # self._mf_damp = mass_flux_damping_rate
        # self._alpha = mass_flux_relaxation_rate
        # self._mf_timescale = reference_mass_flux_timescale
        # self._ntracers = 0
        # self._set_fortran_constants()

        super(ParaformerConvection, self).__init__(**kwargs)

    @ensure_contiguous_state
    def array_call(self, raw_state, timestep):
        """
        Get convective heating and moistening.

        Args:

            raw_state (dict):
                The state dictionary of numpy arrays satisfying this
                component's input properties.

        Returns:

            tendencies (dict), diagnostics (dict):
                * The heating and moistening tendencies
                * Any diagnostics associated.

        """
        
        # You will get all required arrays in the regular (lon*lat, columns) shape. This needs to be reshaped to fit the model input dimensions
        num_cols, num_levs = raw_state['air_temperature'].shape

        tendencies = initialize_numpy_arrays_with_properties(
            self.tendency_properties, raw_state, self.input_properties
        )
        diagnostics = initialize_numpy_arrays_with_properties(
            self.diagnostic_properties, raw_state, self.input_properties
        )

        
        """
        Call Paraformer code here. Reshape numpy arrays available in raw_state, send it to paraformer, get the outputs, and copy them to the relevant output dictionaries
        """

        diagnostics['convective_precipitation_rate'][:] = precc_from_paraformer #shape -> lon*lat

        tendencies['air_temperature'][:] = dTdt_from_paraformer #shape -> lon*lat, cols

        tendencies['specific_humidity'][:] = dQdt_from_paraformer #shape -> lon*lat, cols

        return tendencies, diagnostics
