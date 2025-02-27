import pytest

from pywatershed.atmosphere.prms_atmosphere import PRMSAtmosphere
from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.parameters import Parameters
from pywatershed.parameters import PrmsParameters
from utils_compare import compare_in_memory, compare_netcdfs

# compare in memory (faster) or full output files? or both!
do_compare_output_files = False
do_compare_in_memory = True
rtol = 1.0e-5
atol = 1.0e-5  # why is this relatively low accuracy?

params = ["params_sep", "params_one"]


@pytest.fixture(scope="function")
def control(simulation):
    ctl = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    del ctl.options["netcdf_output_dir"]
    return ctl


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function", params=params)
def parameters(simulation, control, request):
    if request.param == "params_one":
        param_file = simulation["dir"] / control.options["parameter_file"]
        params = PrmsParameters.load(param_file)
    else:
        param_file = simulation["dir"] / "parameters_PRMSAtmosphere.nc"
        params = PrmsParameters.from_netcdf(param_file)

    return params


def test_compare_prms(
    simulation, control, discretization, parameters, tmp_path
):
    comparison_var_names = PRMSAtmosphere.get_variables()

    output_dir = simulation["output_dir"]
    cbh_dir = simulation["dir"]
    # cbh_dir = simulation["cbh_inputs"]["prcp"].parent.resolve()

    input_variables = {}
    for key in PRMSAtmosphere.get_inputs():
        dir = ""
        if "soltab" in key:
            dir = "output/"
        nc_pth = cbh_dir / f"{dir}{key}.nc"
        input_variables[key] = nc_pth

    atm = PRMSAtmosphere(
        control=control,
        discretization=discretization,
        parameters=parameters,
        **input_variables,
    )

    atm.initialize_netcdf(output_dir=tmp_path)

    if do_compare_in_memory:
        answers = {}
        for var in comparison_var_names:
            var_pth = output_dir / f"{var}.nc"
            answers[var] = adapter_factory(
                var_pth, variable_name=var, control=control
            )

        # check the advance/calculate the state
        tmaxf_id = id(atm.tmaxf)

        for ii in range(control.n_times):
            control.advance()
            atm.advance()
            if ii == 0:
                atm.output()
            atm.calculate(1.0)

            compare_in_memory(atm, answers, atol=atol, rtol=rtol)
            assert id(atm.tmaxf) == tmaxf_id

    if do_compare_output_files:
        compare_netcdfs(
            comparison_var_names,
            tmp_path,
            output_dir,
            atol=atol,
            rtol=rtol,
            print_var_max_errs=False,
        )

    return
