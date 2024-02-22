from numpy import (ndarray, 
                   where, clip,
                   exp, arccos, sin, cos, tan, pi, nanmax, float_power)

from pywatershed.constants import nan, zero, DEGREES_TO_RADIANS, RADIANS_TO_DEGREES

"""
Grab bag of functions ripped from SWB.
"""

def calc_daily_actual_et(rainfall: ndarray,
                         snowmelt: ndarray, 
                         pet: ndarray, 
                         soil_storage: ndarray, 
                         soil_storage_max: ndarray):
    """
    Return the current soil moisture (in inches) given the current soil moisture maximum,
    and the difference between (rainfall + snowmelt) minus the daily potential evapotranspiration.

    rainfall(ndarray):                Daily rainfall amount (mm or inches).
    snowmelt(ndarray):                Daily snowmelt amount (mm or inches).
    pet(ndarray):                     Daily potential evapotranspiration (mm or inches).
    soil_storage(ndarray):            
    soil_storage_max                  Maximum moisture content of a soil at field capacity.
    """
    # Alley, W.M., 1984, On the Treatment of Evapotranspiration, Soil Moisture Accounting, and Aquifer Recharge in Monthly
    #     Water Balance Models: Water Resources Research, v. 20, no. 8, p. 1137–1149.

    # Thornthwaite, C.W., and Mather, J.R., 1957, Instructions and tables for computing potential evapotranspiration
    #     and the water balance: Publications in Climatology, v. 10, no. 3, p. 1-104.

    p_minus_pet = rainfall + snowmelt - pet

    cond_p_minus_pet_ge_0 = p_minus_pet >= 0

    # see Alley, 1984, eqns 1 and 2
    temp_soil_storage = where(cond_p_minus_pet_ge_0, zero, soil_storage * exp( p_minus_pet / soil_storage_max ))
    aet = where(cond_p_minus_pet_ge_0, pet, soil_storage - temp_soil_storage)

    return(p_minus_pet, aet)



def calculate_net_precipitation_and_snowfall(gross_precipitation: ndarray,
                                             interception: ndarray, 
                                             tminf: ndarray, 
                                             tmaxf: ndarray):
    """Logic block from SWB for partitioning gross_precipitation into rainfall and snowfall.

    Args:
        gross_precipitation (ndarray): _description_
        interception (ndarray): _description_
        tminf (ndarray): _description_
        tmaxf (ndarray): _description_
    """

    FREEZING_POINT = 32.0

    cond_lt_freezing = ((tminf + tmaxf) / 2.0 - ( tmaxf - tminf ) / 3.0 ) <= FREEZING_POINT

    snowfall = where(cond_lt_freezing, gross_precipitation, zero)
    net_snowfall = where(cond_lt_freezing, gross_precipitation - interception, zero)
    rainfall = where(cond_lt_freezing, zero, gross_precipitation)
    net_rainfall = where(cond_lt_freezing, zero, gross_precipitation - interception)

    return snowfall, net_snowfall, rainfall, net_rainfall


def calculate_snowmelt(tmeanc, tmaxc, snow_storage):

    MELT_INDEX = 1.5  # mm/deg C above freezing

    # temperatures in degrees Celcius, snowmelt in mm
    cond_snowmelt = tmeanc > zero
    potential_snowmelt = where(cond_snowmelt, MELT_INDEX * tmaxc, zero)

    cond_snow_stor = potential_snowmelt > snow_storage
    snowmelt = where(cond_snow_stor, snow_storage, potential_snowmelt)

    return snowmelt


def daylight_hours( omega_s: ndarray ):
    """
    Calculate the number of daylight hours at a location.

    omega_s   Sunset hour angle in Radians.

    Implementation follows equation 34, Allen and others (1998).
    """
    return 24.0 / pi * omega_s



def sunrise_sunset_angle__omega_s(latitude: ndarray,
                                  delta: ndarray):
    """
    Return the sunset angle in radians, given the latitude and solar declination, in radians.

    latitude        Latitude, in Radians
    delta           Solar declination, in Radians

    Implementation follows equation 25, Allen and others (1998).
    """
    omega_s = arccos( - tan(latitude) * tan(delta) )
    return omega_s



def day_angle__Gamma(day_of_year, number_of_days_in_year):
    """
    Return the day angle in Radians.  

    day _of_year             Integer day of the year (January 1 = 1)
    number_of_days_in_year   Number of days in the current year

    Implementation follows equation 1.2.2 in Iqbal (1983)
    """
    day_angle = 2.0 * pi * ( day_of_year - 1.0 ) / number_of_days_in_year
    return day_angle



def solar_declination__delta(day_of_year, number_of_days_in_year):
    """
    Return the solar declination for a given day of the year in Radians.

    day _of_year             Integer day of the year (January 1 = 1)
    number_of_days_in_year   Number of days in the current year

    Implementation follows equation 1.3.1 in Iqbal (1983).
    Iqbal (1983) reports maximum error of 0.0006 radians; if the last two terms are omitted,
      the reported accuracy drops to 0.0035 radians.
    """
    Gamma = day_angle__Gamma( day_of_year, number_of_days_in_year )

    delta =   0.006918                                        \
             - 0.399912 * cos( Gamma )                        \
             + 0.070257 * sin( Gamma )                        \
             - 0.006758 * cos( 2.0 * Gamma )                  \
             + 0.000907 * sin( 2.0 * Gamma )                  \
             - 0.002697 * cos( 3.0 * Gamma )                  \
             + 0.00148  * sin( 3.0 * Gamma )

    return delta


def relative_earth_sun_distance__D_r(day_of_year, number_of_days_in_year):
    """
    Return the inverse relative Earth-Sun distance (unitless) for a given day of the year.

    day_of_year              Integer day of the year (January 1 = 1)
    number_of_days_in_year   Number of days in the current year

    Implementation follows equation 23, Allen and others (1998). 
    See also Equation 1.2.3 in Iqbal, Muhammad (1983-09-28).
    """
    d_r = 1.0 + 0.033                                      \
            * cos( 2.0 * pi * day_of_year )                \
            / number_of_days_in_year      

    return d_r


def extraterrestrial_radiation__Ra(latitude, delta, omega_s, d_r):
    """
    Return the extraterrestrial solar radiation for a point above earth, in MJ / m^2 / day

    latitude        latitude in Radians
    delta           solar declination in Radians
    omega_s         sunset hour angle in Radians
    d_r             Inverse relative sun-earth distance (unitless)

    Implemented as equation 21, Allen and others (1998).
    """
    Gsc = 0.0820   # MJ / m^2 / min

    part_a = omega_s * sin( latitude ) * sin( delta )
    part_b = cos( latitude ) * cos( delta ) * sin( omega_s )

    Ra = 24.0 * 60.0 * Gsc * d_r * ( part_a + part_b ) / pi
    return Ra



def equivalent_evaporation(radiation_energy):
    """
    Returns the equivalent depth of water (in millimeters) that would be evaporated
    per day for a given amount of solar radiation (expressed in MJ/m^2-day).

    radiation_energy        Solar radiation energy expressed in MJ per sq. meter per day

    Implementation follows equation 20, Allen and others (1998).
    """

    return radiation_energy * 0.408


def et0_hargreaves_samani(extraterrestrial_radiation__Ra: ndarray, 
                          air_temp_min: ndarray,
                          air_temp_max: ndarray,
                          air_temp_mean: ndarray,
                          et_slope: float = 0.0023,
                          et_constant: float = 17.8,
                          et_exponent: float =0.5):
    """
    Return the daily reference evapotranspriation in millimeters given extraterrestrial
    radiation (expressed as the number of millimeters of water that could be evaporated by
    the applied radiation) and the minimum and maximum daily air temperatures in Celsius degrees.

    Implemented as equation 4 in Hagreaves and Samani (1985) and as equation 50 in Allen and others (1998).

    """
    air_temp_delta = air_temp_max - air_temp_min

    et0 = nanmax(([0.0],
                  et_slope * extraterrestrial_radiation__Ra * (air_temp_mean + et_constant)          \
                       * float_power(air_temp_delta,et_exponent)) 
             )

    return et0


def calculate_et0_hargreaves_samani(day_of_year, number_of_days_in_year, latitude,
                                        air_temp_min, air_temp_max, air_temp_mean):
    """
    Return the calculated reference evapotranspiration in millimeters per day, 
    given the day number, latitude, and min, max, and mean air temperatures in Celsius.

    day_of_year                 Day number of the current solar year.
    number_of_days_in_year      Number of days in the solar year.
    latitude                    Latitude (in degrees) at which calculation is to be made.
    air_temp_min                Minimum daily air temperature, in degrees Celsius
    air_temp_max                Maximum daily air temperature, in degrees Celsius
    air_temp_mean               Mean daily air temperature, in degrees Celsius
    """    
    latitude_radians = latitude * DEGREES_TO_RADIANS

    d_r     = relative_earth_sun_distance__D_r(day_of_year, number_of_days_in_year)
    delta   = solar_declination__delta(day_of_year, number_of_days_in_year)
    omega_s = sunrise_sunset_angle__omega_s(latitude_radians, delta)

    Ra = equivalent_evaporation(extraterrestrial_radiation__Ra(latitude_radians, delta, omega_s, d_r))
    ref_ET = et0_hargreaves_samani(Ra, air_temp_min, air_temp_max, air_temp_mean)
    print(f'ref_ET: {ref_ET}')
    return ref_ET



def calculate_cn_S_inches(curve_number):
    """
    Return the curve number storage (S) term, in inches. Equation 2-4, Cronshey and others (1986).
    
    """

    S_inches = ( 1000.0 / curve_number ) - 10.0
    return S_inches



def calculate_cn_S_millimeters(curve_number):
    """
    Return the curve number storage (S) term, in millimeters. Equation 2-4, Cronshey and others (1986),
    with constants multiplied by 25.4 (mm to inches).
    
    """

    S_mm = ( 25400.0 / curve_number ) - 254.0
    return S_mm



def calculate_cn_runoff(inflow, storage_S, initial_abstraction_Ia = 0.05):
    """
    Return the runoff value given the inflow (precip), storage, and initial abstraction. 
    Equation 2-3, Cronshey and others (1986).
    """
    Ia = initial_abstraction_Ia
    runoff = where(inflow > Ia,
                   float_power(inflow - Ia * storage_S, 2.0) / (inflow + (1.0 - Ia) * storage_S),
                   0.0
                  )

    return runoff



def calculate_cn_alternative_S_0_05(storage_S):
    """
    Return the curve number storage term, assuming that the initial abstraction is 0.05, rather than 0.2.
    Equation 8, Woodward and others (2003).
    """
    return 1.33 * (storage_S^1.15)



def calculate_cn_arc2_to_arc1(curve_number_arc2):
    """
    Return a curve number corresponding to antecedant runoff condition 1, given a
    curve number corresponding to antecedant runoff condition 2.
    Implemented as equation 3.145 of "SCS Curve Number Methodology", Mishra and Singh (2003),
    and as equation 15 in Ponce and Hawkins (1996).
    
    Resulting curve numbers are clipped to the range 30-100.
    """
    return clip((curve_number_arc2 / (2.281 - 0.01281 * curve_number_arc2 )),      
                30.0,
                100.0
               )


def calculate_cn_arc2_to_arc3(curve_number_arc2):
    """
    Return a curve number corresponding to antecedant runoff condition 3, given a
    curve number corresponding to antecedant runoff condition 2.
    Implemented as equation 3.146 of "SCS Curve Number Methodology", Mishra and Singh (2003), 
    and as equation 16 in Ponce and Hawkins (1996).

    Resulting curve numbers are clipped to the range 30-100.
    """
    return np.clip((curve_number_arc2 / (0.427 - 0.00573 * curve_number_arc2 )),      
                30.0,
                100.0
               )

def adjust_curve_number(curve_number, inflow_5_day_sum, is_growing_season=False, cfgi=0, cfgi_ll=55, cfgi_ul=85):

    ARC_DRY_GROWING = 1.40
    ARC_DRY_DORMANT = 0.50
    ARC_WET_GROWING = 2.10
    ARC_WET_DORMANT = 1.10

    if( cfgi > cfgi_ll ):

        p_er = calculate_probability_of_enhanced_runoff(cfgi, cfgi_ll, cfgi_ul)

        curve_number_adj = curve_number * (1. - p_er) + curve_number_arc3 * p_er

    elif( is_growing_season ):

        curve_number_adj = np.where(inflow_5_day_sum > ARC_WET_GROWING, curve_number_arc3,
                             np.where(inflow_5_day_sum < ARC_DRY_GROWING, curve_number_arc1, curve_number))

    else:

        curve_number_adj = np.where(inflow_5_day_sum > ARC_WET_DORMANT, curve_number_arc3,
                             np.where(inflow_5_day_sum < ARC_DRY_DORMANT, curve_number_arc1, curve_number))

    return curve_number_adj


def c_to_f(temperature_deg_c: ndarray):
    """Return temperature in degrees Fahrenheit.
    

    Args:
        temperature_deg_c (ndarray): temperature in degrees Celsius

    Returns:
        ndarray: temperature in degrees Fahrenheit 
    """

    temperature_deg_f = temperature_deg_c * 1.8 + 32

    return temperature_deg_f