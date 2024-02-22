from typing import Literal
import datetime as dt
import numpy as np

from ..base.conservative_process import ConservativeProcess
from ..parameters import Parameters

from pywatershed.base.adapter import adaptable
from pywatershed.base.control import Control
from pywatershed.constants import nan, zero
from pywatershed.utils.time_utils import datetime_doy, datetime_num_days_in_year
import pywatershed.hydrology.swb_functions as swbfn

class SWBRootZone(ConservativeProcess):
    """SWB Root Zone Storage object, first cut.

        This is an attempt to get simple SWB soilzone functionality into pywatershed.
        In order to avoid having to implement SWB snowfall/snowmelt at this time,
        we are using the PRMS variables 'hru_rain' and 'snowmelt' as additions to the
        soil control colume. A future version will implement SWB's simple snowfall and
        snowmelt functionality, which will likely require a change in variable names.

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        hru_ppt: adaptable,
        tminc: adaptable,
        tmaxc: adaptable,
        budget_type: Literal[None, "warn", "error"] = None,
        calc_method: Literal["numba", "numpy"] = None,
        verbose: bool = False,
        load_n_time_batches: int = 1,
    ):
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
        )
        self.name = "SWBRootZone"

        self._set_inputs(locals())
        self._set_options(locals())

        self._set_budget(budget_type)
        # self._init_calc_method()
        self._calc_method = str(calc_method)

    @staticmethod
    def get_dimensions() -> tuple:
        """Get dimensions"""
        return ("nhru",)

    @staticmethod
    def get_parameters() -> tuple:
        """Get root zone reservoir parameters
        Returns:
            parameters: input parameters
        """
        return (
            "base_curve_number",
            "max_net_infiltration_rate",
            "available_water_capacity",
            "rooting_depth",
            "swb_soil_storage_init",
            "swb_snow_storage_init"
        )

    @staticmethod
    def get_inputs() -> tuple:
        """Get root zone reservoir input variables
        Returns:
            variables: input variables
        """
        return (
            "hru_ppt",
            "tmaxc",
            "tminc",
        )

    @staticmethod
    def get_init_values() -> dict:
        """Get SWB rootzone initial values
        Returns:
            dict: initial values for named variables
        """
        return {
            "swb_runoff": zero,
            "swb_actual_et": zero,
            "swb_soil_storage": zero,
            "swb_soil_storage_old": zero,
            "swb_soil_storage_change": zero,
            "swb_net_infiltration": zero,
            "swb_soil_storage_max": zero,
            "swb_snowfall": zero,
            "swb_snowmelt": zero,
            "swb_snow_storage": zero,
            "swb_snow_storage_old": zero,
            "swb_snow_storage_change": zero,            
        }

    @staticmethod
    def get_mass_budget_terms():
        return {
            "inputs": [
                "hru_ppt",
            ],
            "outputs": [
                "swb_runoff",
                "swb_net_infiltration",
            ],
            "storage_changes": [
                "swb_soil_storage_change",
                "swb_snow_storage_change",
            ],
        }

    def _set_initial_conditions(self):
        self.swb_soil_storage = self.swb_soil_storage_init.copy()
        self.swb_soil_storage_max = (
            self.available_water_capacity * self.rooting_depth
        )
        self.curve_number_storage_S = swbfn.calculate_cn_S_inches(self.base_curve_number)
        self.swb_snow_storage = self.swb_snow_storage_init.copy()
        return

    def _advance_variables(self) -> None:
        """Advance the SWBRootZone calculation.
        Returns:
            None
        """
        self.swb_soil_storage_old[:] = self.swb_soil_storage
        self.swb_snow_storage_old[:] = self.swb_snow_storage
        return

    def _calculate(self, simulation_time):
        """Calculate SWBRootZone for a time step

        Args:
            simulation_time: current simulation time

        Returns:
            None

        """

        self._simulation_time = simulation_time

#        self.doy = datetime_doy(self._simulation_time)
#        self.num_days_in_year = datetime_num_days_in_year(self._simulation_time)

        ##############################
        ##############################
        ## DANGER HACK ALERT 
        ##############################
        self.doy = self._simulation_time
        self.num_days_in_year = 365

        (
            self.swb_soil_storage[:],
            self.swb_soil_storage_change[:],
            self.swb_snow_storage[:],
            self.swb_snow_storage_change[:],
            self.swb_snowfall[:],
            self.swb_snowmelt[:],
            self.swb_runoff[:],
            self.swb_actual_et[:],
            self.swb_net_infiltration[:],
        ) = self._calculate_numpy(
            base_curve_number=self.base_curve_number,
            gross_precipitation=self.hru_ppt,
            tmaxc=self.tmaxc,
            tminc=self.tminc,
            doy=self.doy,
            num_days_in_year=self.num_days_in_year,
            soil_storage=self.swb_soil_storage,
            soil_storage_max=self.swb_soil_storage_max,
            snow_storage=self.swb_snow_storage,
        )
        return

    @staticmethod
    def _calculate_numpy(
        base_curve_number,
        gross_precipitation,
        tmaxc,
        tminc,
        doy,
        num_days_in_year,
        soil_storage,
        soil_storage_max,
        snow_storage,
    ):
        actual_et = np.full_like(gross_precipitation, fill_value=0.0)
        runoff = np.full_like(gross_precipitation, fill_value=0.0)
        interception = np.full_like(gross_precipitation, fill_value=0.0)
        net_infiltration = np.full_like(gross_precipitation, fill_value=0.0)
        snowfall = np.full_like(gross_precipitation, fill_value=0.0)
        net_snowfall = np.full_like(gross_precipitation, fill_value=0.0)
        rainfall = np.full_like(gross_precipitation, fill_value=0.0)
        net_rainfall = np.full_like(gross_precipitation, fill_value=0.0)
        snowmelt = np.full_like(gross_precipitation, fill_value=0.0)

        tmaxf = swbfn.c_to_f(tmaxc)
        tminf = swbfn.c_to_f(tminc)
        tmeanc = (tminc + tmaxc / 2.0)

        snowfall, net_snowfall, rainfall, net_rainfall = (
          swbfn.calculate_net_precipitation_and_snowfall(gross_precipitation=gross_precipitation,
                                                         interception=interception, 
                                                         tminf=tminf, 
                                                         tmaxf=tminf)
        )

        snow_storage_old = snow_storage.copy()
        snow_storage = snow_storage + snowfall

        snowmelt = swbfn.calculate_snowmelt(tmeanc, tmaxc, snow_storage)

        snow_storage = snow_storage - snowmelt

        snow_storage_change = snow_storage - snow_storage_old

        inflow = net_rainfall + snowmelt
        curve_number_storage_S = swbfn.calculate_cn_S_inches(base_curve_number)
        
        runoff = swbfn.calculate_cn_runoff(inflow=inflow,
                                           storage_S=curve_number_storage_S,
                                          )
        
    
        ##### HACK ALERT: dividing function values by 25.4 to convert mm to inches of PET
        reference_et = swbfn.calculate_et0_hargreaves_samani(day_of_year=doy,
                                                            number_of_days_in_year=num_days_in_year,
                                                            latitude=43.,
                                                            air_temp_min=tminc, 
                                                            air_temp_max=tmaxc, 
                                                            air_temp_mean=tmeanc) / 25.4

        (p_minus_pet, actual_et) = swbfn.calc_daily_actual_et(rainfall=net_rainfall,
                                                            snowmelt=snowmelt,
                                                            pet=reference_et,
                                                            soil_storage=soil_storage,
                                                            soil_storage_max=soil_storage_max
                                                           )
        #breakpoint()
        soil_storage_old = soil_storage.copy()
        soil_storage = soil_storage + inflow - runoff - actual_et


        cond = soil_storage > soil_storage_max
        soil_storage = np.where(cond, soil_storage_max, soil_storage)
        net_infiltration = np.where(cond, soil_storage - soil_storage_max, zero)

        soil_storage_change = soil_storage - soil_storage_old
        #breakpoint()

        return (soil_storage, soil_storage_change, snow_storage, snow_storage_change,
                snowfall, snowmelt, runoff, actual_et, net_infiltration)
