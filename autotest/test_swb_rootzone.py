import pathlib as pl

import pytest

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.hydrology.SWBRootZone import SWBRootZone
from pywatershed.parameters import Parameters
from utils_compare import assert_allclose

import xarray as xr

# compare in memory (faster) or full output files? or both!
do_compare_output_files = False
do_compare_in_memory = True
rtol = atol = 5.0e-6

calc_methods = ("numpy", )
vars_compare = ("swb_soil_storage",)

@pytest.fixture(scope="function")
def control():
    control_file=pl.Path('../test_data/hru_1/swb_control.yaml')
    control = Control.from_yaml(control_file)
    control.options['parameter_file'] = control_file.parent / control.options['parameter_file']
    return control

@pytest.fixture(scope="function")
def parameters(control):
    params = Parameters.from_yaml(control.options['parameter_file'])
    return params

@pytest.fixture(scope="function")
def answers(control):
    answers = {}
    swb_output_dir = pl.Path('../test_data/hru_1/swb_output')
    var_name_map = {'swb_soil_storage': 'soil_storage'}
    for key in vars_compare:
        var_path = (
            pl.Path(swb_output_dir)
            / f"hru_1_5000__{key.removeprefix('swb_')}__1979-01-01_to_2019-12-31__1_by_1.nc"
        )
        ### the following results in a shape mismatch in following steps...
        answers[key] = xr.open_dataset(var_path)[var_name_map[key]].values.squeeze()

    return answers

def test_compare_swb(control, parameters, answers, tmp_path):
    tmp_path = pl.Path(tmp_path)

    # load csv files into dataframes
    prms_output_dir = pl.Path('../test_data/hru_1/output')
    swb_output_dir = pl.Path('../test_data/hru_1/swb_output')
    input_variables = {}

    for key in SWBRootZone.get_inputs():
        nc_path = pl.Path(prms_output_dir) / f"{key}.nc"
        input_variables[key] = nc_path

#    breakpoint()

    swb_rz = SWBRootZone(
        control,
        None,
        parameters,
        **input_variables,
        budget_type="warn",
        calc_method="numpy",
    )

    swb_rz.initialize_netcdf(swb_output_dir)

    for istep in range(control.n_times):
        control.advance()
        swb_rz.advance()
        swb_rz.calculate(float(istep))
        swb_rz.output()
        #for var in vars_compare:
        #    assert_allclose(actual=swb_rz[var],
        #                    desired=answers[var][istep])

    swb_rz.finalize()

    return
