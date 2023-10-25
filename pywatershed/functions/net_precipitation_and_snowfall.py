#from numpy import nanmax, float_power

def calculate_net_precipitation_and_snowfall(gross_precipitation: float, interception: float, tmin: float, tmax: float):
    """Logic block from SWB for partitioning gross_precipitation into rainfall and snowfall.

    Args:
        gross_precipitation (float): _description_
        interception (float): _description_
        tmin (float): _description_
        tmax (float): _description_
    """

    FREEZING_POINT = 32.0

    if ( (tmin + tmax) / 2.0 - ( tmax - tmin ) / 3.0 ) <= FREEZING_POINT:
        snowfall = gross_precipitation
        net_snowfall = gross_precipitation - interception
        rainfall = 0.0
        net_rainfall = 0.0
    else:
      snowfall = 0.0
      net_snowfall = 0.0
      rainfall = gross_precipitation
      net_rainfall = gross_precipitation - interception

    return snowfall, net_snowfall, rainfall, net_rainfall