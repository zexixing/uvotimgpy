from astropy.time import Time
import numpy as np
import re
from datetime import datetime
from uvotimgpy.base.file_and_table import parse_date_string
from uvotimgpy.uvot_analysis.activity import get_g_factor
from uvotimgpy.base.math_tools import UnitConverter
from uvotimgpy.uvot_analysis.activity import create_vectorial_model, RatioCalculator_V_UV
from astropy import units as u
from typing import Tuple

def create_magnitude_calculator(coefficients, time_intervals=None):
    """
    Create a function that calculates magnitude.
    Parameters:
        coefficients: tuple or list of tuples
            - Single formula: (H, G), e.g. (12.0, 17.0)
            - Multiple formulas: [(H1, G1), (H2, G2), ...]
        time_intervals: None or list of str
            - None: no segmentation; use a single formula
            - list: time breakpoints; supports multiple formats:
                    ['2025-10-10', '2026-01-08', '2026-06-27']
                    ['2025 Oct. 10', '2026 Jan. 8', '2026 June 27']
                    ['2025 Oct 10', '2026 Jan 8', '2026 Jun 27']
    Returns:
        Function that calculates magnitude.
    """
    # Process input parameters
    if time_intervals is None:
        # Case without segmentation
        if isinstance(coefficients, tuple) and len(coefficients) == 2:
            coeff_list = [coefficients]
        else:
            # raise ValueError("不分段时，coefficients应该是单个tuple (H, G)")
            raise ValueError("When not using segments, coefficients should be a single tuple (H, G)")
        time_boundaries = None
    else:
        # Segmented case
        if not isinstance(coefficients, list):
            # raise ValueError("分段时，coefficients应该是list of tuples")
            raise ValueError("When using segments, coefficients should be a list of tuples")
        
        coeff_list = coefficients
        
        # Parse time strings in various formats
        time_boundaries = []
        for t in time_intervals:
            try:
                parsed_time = parse_date_string(t)
                time_boundaries.append(parsed_time)
            except ValueError as e:
                # raise ValueError(f"解析时间 '{t}' 失败: {e}")
                raise ValueError(f"Failed to parse time '{t}': {e}")
        
        # Check count consistency
        if len(coeff_list) != len(time_boundaries) + 1:
            # raise ValueError(f"需要{len(time_boundaries)+1}个系数组，但提供了{len(coeff_list)}个")
            raise ValueError(f"Expected {len(time_boundaries)+1} coefficient groups, but received {len(coeff_list)}")
    
    def calculate_magnitude(rh_list, delta_list, date_list):
        """
        Calculate magnitude: m = H + 5*log10(delta) + G*log10(rh).
        
        Parameters:
            rh_list: array-like, heliocentric distance r (AU)
            delta_list: array-like, geocentric distance d (AU)
            date_list: list of str, date/time strings
                      supported formats include '2022-01-01T00:00:00.000',
                      '2022-01-01', '2022 Jan 1', etc.
        
        Returns:
            numpy array, calculated magnitude values.
        """
        # Convert to numpy arrays
        rh = np.asarray(rh_list)
        delta = np.asarray(delta_list)
        
        # Parse the date list
        times = []
        for date_str in date_list:
            try:
                # First try direct parsing with Time
                times.append(Time(date_str))
            except:
                # If that fails, use the custom parser
                times.append(parse_date_string(date_str))
        times = Time(times)
        
        # Initialize the result array
        magnitude = np.zeros_like(rh, dtype=float)
        
        if time_boundaries is None:
            # No segmentation; use a single formula
            H, G = coeff_list[0]
            magnitude = H + 5 * np.log10(delta) + G * np.log10(rh)
        else:
            # Segmented calculation
            for i in range(len(times)):
                t = times[i]
                
                # Determine which formula to use
                if t < time_boundaries[0]:
                    H, G = coeff_list[0]
                elif t >= time_boundaries[-1]:
                    H, G = coeff_list[-1]
                else:
                    # In one of the intermediate intervals
                    for j in range(len(time_boundaries) - 1):
                        if time_boundaries[j] <= t < time_boundaries[j + 1]:
                            H, G = coeff_list[j + 1]
                            break
                
                # Calculate magnitude
                magnitude[i] = H + 5 * np.log10(delta[i]) + G * np.log10(rh[i])
        
        return list(magnitude)

    # Helper method
    def get_formula_info():
        """Return formula configuration information."""
        info = []
        if time_boundaries is None:
            H, G = coeff_list[0]
            info.append(f"m = {H} + 5*log(d) + {G}*log(r)  [all times]")
        else:
            for i, (H, G) in enumerate(coeff_list):
                if i == 0:
                    info.append(f"m = {H:4.1f} + 5*log(d) + {G:4.1f}*log(r)  (before {time_boundaries[0].iso[:10]})")
                elif i == len(coeff_list) - 1:
                    info.append(f"m = {H:4.1f} + 5*log(d) + {G:4.1f}*log(r)  (after {time_boundaries[-1].iso[:10]})")
                else:
                    info.append(f"m = {H:4.1f} + 5*log(d) + {G:4.1f}*log(r)  ({time_boundaries[i-1].iso[:10]} to {time_boundaries[i].iso[:10]})")
        return "\n".join(info)
    
    # Add attributes to the returned function
    calculate_magnitude.get_formula_info = get_formula_info
    calculate_magnitude.coefficients = coeff_list
    calculate_magnitude.time_boundaries = time_boundaries
    
    return calculate_magnitude

class ExposureCalculator:
    def __init__(self, target_dict, reference_dict, aperture, snr=3):
        """
        Initialize the exposure time calculator.
        
        Parameters:
        -----------
        reference_dict : dict
            Dictionary for the reference source, containing keys such as 'm_v' and 'cr_v'.
        target_dict : dict
            Dictionary for the target source, containing keys such as 'm_v', 'rh', 'delta', and 'rhv'.
        aperture : float or tuple
            Aperture size (km), either a single value or an (inner radius, outer radius) tuple.
        snr : float
            Required signal-to-noise ratio; default is 3.
        """
        self.reference_dict = reference_dict
        self.target_dict = target_dict
        self.aperture = aperture
        self.snr = snr
        
        # Calculate pixel-related parameters
        km_per_pixel = UnitConverter.arcsec_to_km(1*1.004, target_dict['delta'])
        self.x = 3
        
        if isinstance(aperture, Tuple):
            radius_outer = aperture[1]/km_per_pixel
            radius_inner = aperture[0]/km_per_pixel
            self.pixel_number = (radius_outer**2 - radius_inner**2)*np.pi
        else:
            radius = aperture/km_per_pixel
            self.pixel_number = radius**2*np.pi
        
        # Background count rates
        self.uw1_bkg_countrate = 0.003503 * self.pixel_number
        self.uw1_bkg_countrate_err = 6.475e-04 * np.sqrt(self.pixel_number)
        self.v_bkg_countrate = 0.02226 * self.pixel_number
        self.v_bkg_countrate_err = 0.002021 * np.sqrt(self.pixel_number)
        
        # Add beta as an instance variable, obtained from RatioCalculator
        self.beta = RatioCalculator_V_UV.dust_countrate_ratio_from_reddening(
            reddening=10, 
            obs_time=Time('2025-01-01T00:00:00.000')
        )
    
    def get_qwater(self, m_v, delta):
        """Calculate the water production rate."""
        m_h = m_v - 5*np.log10(delta)
        qwater = np.power(10, 30.675 - 0.2453*m_h)
        return qwater
    
    def qwater_to_oh_number_in_aperture(self, qwater, r_h, delta, aperture):
        """Calculate the number of OH molecules within the aperture."""
        vm = create_vectorial_model(r_h)
        
        if isinstance(aperture, float) or isinstance(aperture, int):
            number_model = vm.total_number(aperture*u.km)
        elif isinstance(aperture, Tuple):
            number_model = vm.total_number(aperture[1]*u.km) - vm.total_number(aperture[0]*u.km)
        
        number_aperture = number_model * qwater/(vm.base_q)
        return number_aperture
    
    def get_oh_countrate(self, oh_number, rhv, rh, delta):
        """Calculate the OH count rate."""
        g_1au_value = get_g_factor(rhv)
        g_value = g_1au_value / np.power(rh, 2)
        luminosity = oh_number * g_value
        delta_cm = UnitConverter.au_to_km(delta) * 1000 * 100
        emission_flux = luminosity / (4 * np.pi * np.power(delta_cm, 2))
        factor = 1/(1.6368359501510164e-12)
        countrate = emission_flux * factor
        return countrate
    
    def get_v_dust_countrate(self, m_v, m_v_ref, cr_v_ref):
        """Calculate the V-band dust count rate."""
        f_to_fref = np.power(10, -0.4*(m_v - m_v_ref))
        cr_to_crref = f_to_fref
        cr_in_aperture = cr_v_ref * cr_to_crref
        return cr_in_aperture
    
    def get_cr_uw1(self, oh_countrate, v_dust_countrate, uw1_bkg_countrate):
        """Calculate the total UW1-band count rate."""
        cr_uw1 = oh_countrate + self.beta*v_dust_countrate + uw1_bkg_countrate
        return cr_uw1
    
    def get_cr_v(self, v_dust_countrate, v_bkg_countrate):
        """Calculate the total V-band count rate."""
        cr_v = v_dust_countrate + v_bkg_countrate
        return cr_v
    
    def get_exposure_time(self, x, cr_uw1, cr_v, cr_oh, cr_uw1_bkg_err, cr_v_bkg_err, snr):
        """Calculate the required exposure time."""
        numerator = cr_uw1/x + (self.beta**2)*cr_v
        a = (cr_uw1_bkg_err**2) + (self.beta**2)*(cr_v_bkg_err**2)
        denominator = (cr_oh/snr)**2 - a
        
        if denominator <= 0:
            print('best SNR = '+f'{cr_oh/np.sqrt(a):.2f}')
            # raise ValueError("无法达到所需的信噪比，分母为负值或零")
            raise ValueError("Cannot reach the required SNR because the denominator is negative or zero")
        
        exposure = numerator/denominator

        return exposure
    
    def calculate_exposure_time(self):
        """Main method for calculating exposure time."""
        target_dict = self.target_dict
        reference_dict = self.reference_dict
        
        # Calculate the water production rate
        qwater = self.get_qwater(target_dict['m_v'], target_dict['delta'])
        
        # Calculate the OH amount within the aperture
        oh_number_in_aperture = self.qwater_to_oh_number_in_aperture(
            qwater, 
            target_dict['rh'], 
            target_dict['delta'], 
            self.aperture
        )
        
        # Calculate the OH count rate
        oh_countrate = self.get_oh_countrate(
            oh_number_in_aperture, 
            target_dict['rhv'], 
            target_dict['rh'], 
            target_dict['delta']
        )
    
        # Calculate the V-band dust count rate
        v_dust_countrate = self.get_v_dust_countrate(
            target_dict['m_v'], 
            reference_dict['m_v'], 
            reference_dict['cr_v']
        )
        
        # Calculate total count rates
        print(qwater, oh_number_in_aperture,
              oh_countrate, v_dust_countrate, 
              self.uw1_bkg_countrate, self.v_bkg_countrate)
        print(self.uw1_bkg_countrate_err, self.v_bkg_countrate_err)
        cr_uw1 = self.get_cr_uw1(oh_countrate, v_dust_countrate, self.uw1_bkg_countrate)
        cr_v = self.get_cr_v(v_dust_countrate, self.v_bkg_countrate)
        
        # Calculate exposure time
        exposure = self.get_exposure_time(
            self.x, 
            cr_uw1, 
            cr_v, 
            oh_countrate, 
            self.uw1_bkg_countrate_err, 
            self.v_bkg_countrate_err, 
            self.snr
        )
        
        return exposure


if __name__ == "__main__":
    #target_j3 = {'m_v': 14.1, 'rh': 4.0, 'delta': 3.47, 'rhv': -3.52, 'date': '2026-08-11'} #12.92
    #
    #target_e1 = {'m_v': 15.8, 'rh': 3.05, 'delta': 2.85, 'rhv': -21.78, 'date': '2025-07-24'}
    #
    #mag_calc_e1 = create_magnitude_calculator((10.5, 10.0))
    #print(mag_calc_e1([target_e1['rh']], [target_e1['delta']], [target_e1['date']]))


    #km = UnitConverter.arcsec_to_km(1, target_e1['delta'])
    #aperture = (36000, 40000) #km
    #print(40000/2067.022189836139)
    #print(36000/2067.022189836139)
    #
    #print((779.62 - 681.45)/196.6)

    target_t5 = {'m_v': 13.67, 'rh': 4.0, 'delta': 3.2, 'rhv': -4.05, 'date': '2027-01-04'} #12.05
    aperture = (36000, 40000) #km
    target_e1 = {'m_v': 14.1, 'rh': 1.9, 'delta': 2.45, 'rhv': 25.5, 'date': '2026-04-28'}
    target_e1 = {'m_v': 17.4, 'rh': 4.0, 'delta': 4.25, 'rhv': 19.5, 'date': '2026-10-09'}
    aperture = 40000
    reference_dict = {'m_v': 15.8, 'cr_v': 0.49934, 'rh':3.05, 'delta':2.85, 'rhv':-21.76, 'date':'2025-07-24'}
    etc = ExposureCalculator(target_dict=target_e1, reference_dict=reference_dict, aperture=aperture, snr=3)
    t_v = etc.calculate_exposure_time()
    t_uw1 = 3*t_v
    print(t_v, t_uw1, (t_v+t_uw1)/1600)
