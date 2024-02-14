from typing import Literal
from ..base.conservative_process import ConservativeProcess
from ..parameters import Parameters

from pywatershed.base.adapter import adaptable
from pywatershed.base.control import Control
from pywatershed.constants import nan, zero
import pywatershed.functions.runoff_curve_number as cn
import pywatershed.functions.actual_et_thornthwaite_mather as aet
import numpy as np


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
        net_rain: adaptable,
        snowmelt: adaptable,
        potet: adaptable,
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
        )

    @staticmethod
    def get_inputs() -> tuple:
        """Get root zone reservoir input variables
        Returns:
            variables: input variables
        """
        return (
            "net_rain",
            "snowmelt",
            "potet",
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
        }

    @staticmethod
    def get_mass_budget_terms():
        return {
            "inputs": [
                "net_rain",
                "snowmelt",
            ],
            "outputs": [
                "swb_runoff",
                "swb_net_infiltration",
            ],
            "storage_changes": [
                "swb_soil_storage_change",
            ],
        }

    def _set_initial_conditions(self):
        self.swb_soil_storage = self.swb_soil_storage_init.copy()
        self.swb_soil_storage_max = (
            self.available_water_capacity * self.rooting_depth
        )
        self.curve_number_storage_S = cn.calculate_cn_S_inches(self.base_curve_number)

        return

    def _advance_variables(self) -> None:
        """Advance the SWBRunoffCurveNumber
        Returns:
            None
        """
        self.swb_soil_storage_old[:] = self.swb_soil_storage
        return

    def _calculate(self, simulation_time):
        """Calculate SWBRunoffCurveNumber for a time step

        Args:
            simulation_time: current simulation time

        Returns:
            None

        """

        self._simulation_time = simulation_time

        (
            self.swb_soil_storage[:],
            self.swb_soil_storage_change[:],
            self.swb_runoff[:],
            self.swb_actual_et[:],
            self.swb_net_infiltration[:],
        ) = self._calculate_numpy(
            base_curve_number=self.base_curve_number,
            net_rain=self.net_rain,
            snowmelt=self.snowmelt,
            reference_et=self.potet,
            soil_storage=self.swb_soil_storage,
            soil_storage_max=self.swb_soil_storage_max,
        )
        return

    @staticmethod
    def _calculate_numpy(
        base_curve_number,
        net_rain,
        snowmelt,
        reference_et,
        soil_storage,
        soil_storage_max,
    ):
        actual_et = np.full_like(net_rain, fill_value=0.0)
        runoff = np.full_like(net_rain, fill_value=0.0)
        net_infiltration = np.full_like(net_rain, fill_value=0.0)

        inflow = net_rain + snowmelt
        curve_number_storage_S = cn.calculate_cn_S_inches(base_curve_number)
        runoff = cn.calculate_cn_runoff(inflow=inflow,
                                        storage_S=curve_number_storage_S,
                                       )

        (p_minus_pet, actual_et) = aet.calc_daily_actual_et(rainfall=net_rain,
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
        # breakpoint()

        return (soil_storage, soil_storage_change, runoff, actual_et, net_infiltration)
