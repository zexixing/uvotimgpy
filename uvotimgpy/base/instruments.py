from uvotimgpy.config import paths
import astropy.units as u
import numpy as np
from astropy.io import fits
from synphot import SpectralElement, Empirical1D
from typing import Union
import stsynphot as stsyn
from astropy.time import Time


def normalize_filter_name(filter_input, output_format='filename'):
    """
    Normalize Swift/UVOT filter names to standard format.
    
    Parameters:
    -----------
    filter_input : str or iterable
        Input filter name(s). Can be a single string or an iterable of strings
    output_format : str, optional
        Output format, either 'filename' or 'display' (default: 'filename')
        
    Returns:
    --------
    str or list
        Normalized filter name(s). Returns a string if input is a string,
        or a list if input is an iterable
    """
    filter_map = {
        'uvw1': {'filename': 'uw1', 'display': 'UVW1'},
        'uw1': {'filename': 'uw1', 'display': 'UVW1'},
        'uvw2': {'filename': 'uw2', 'display': 'UVW2'},
        'uw2': {'filename': 'uw2', 'display': 'UVW2'},
        'uvm2': {'filename': 'um2', 'display': 'UVM2'},
        'um2': {'filename': 'um2', 'display': 'UVM2'},
        'uuu': {'filename': 'uuu', 'display': 'U'},
        'u': {'filename': 'uuu', 'display': 'U'},
        'uvv': {'filename': 'uvv', 'display': 'V'},
        'v': {'filename': 'uvv', 'display': 'V'},
        'ubb': {'filename': 'ubb', 'display': 'B'},
        'b': {'filename': 'ubb', 'display': 'B'},
        'ugu': {'filename': 'ugu', 'display': 'UV grism'},
        'ugrism': {'filename': 'ugu', 'display': 'UV grism'},
        'uv grism': {'filename': 'ugu', 'display': 'UV grism'},
        'ugv': {'filename': 'ugv', 'display': 'V grism'},
        'vgrism': {'filename': 'ugv', 'display': 'V grism'},
        'v grism': {'filename': 'ugv', 'display': 'V grism'}
    }
    
    # If input is a string, process directly
    if isinstance(filter_input, str):
        return filter_map[filter_input.lower().strip()][output_format]
    
    # If input is iterable, convert each element
    try:
        return [filter_map[f.lower().strip()][output_format] for f in filter_input]
    except AttributeError:
        raise TypeError("Filter input must be a string or an iterable of strings")

def create_bandpass(wave: u.Quantity, thru: Union[u.Quantity, np.ndarray]) -> SpectralElement:
    '''
    Create SpectralElement from wavelength and throughput arrays
    '''
    # 确保thru是无量纲的
    if isinstance(thru, u.Quantity):
        thru = thru.value
    return SpectralElement(Empirical1D, points=wave, lookup_table=thru, fill_value=0)

def format_bandpass(bandpass: Union[str, SpectralElement]):
    if isinstance(bandpass, str):
        bandpass = stsyn.band(bandpass)
    return bandpass

class SensitivityCorrection:
    """
    Calculate UVOT sensitivity correction factors for different filters over time.
    """
    
    # CALDB sensitivity file path
    CALDB_SENSITIVITY_PATH = "/Users/zexixing/Software/caldb/data/swift/uvota/bcf/senscorr/swusenscorr20041120v007.fits"
    
    # All times in the CALDB file are measured in seconds from this date
    UVOT_SENSITIVITY_START_DATE = Time("2005-01-01T00:00")
    
    # Seconds in a year for time conversion
    SECONDS_IN_A_YEAR = 365.2425 * 86400.0
    
    # Map from normalized filter names to CALDB FITS header 'FILTER' values
    FILTER_TO_CALDB_MAP = {
        'uvv': 'V',
        'ubb': 'B',
        'uuu': 'U',
        'uw1': 'UVW1',
        'um2': 'UVM2',
        'uw2': 'UVW2',
        'white': 'WHITE',
        'magnifier': 'MAGNIFIER'
    }
    
    @staticmethod
    def get_correction_factor(filter_name: str, obs_time: Time) -> float:
        """
        Calculate the sensitivity correction factor for a given filter and observation time.
        
        Parameters
        ----------
        filter_name : str
            Name of the filter (e.g., 'uvw1', 'uw1', 'v', 'uvv', etc.)
        obs_time : astropy.time.Time
            Observation time
            
        Returns
        -------
        float
            Sensitivity correction factor. Returns 1.0 if no correction is available.
        """
        # Normalize the filter name to filename format
        normalized_filter = normalize_filter_name(filter_name, output_format='filename')
        
        # Get CALDB filter string
        caldb_filter_string = SensitivityCorrection.FILTER_TO_CALDB_MAP.get(normalized_filter)
        if caldb_filter_string is None:
            print(f"Warning: No CALDB mapping for filter {normalized_filter}")
            return 1.0
        
        # Get sensitivity data for this filter
        sensitivity_data = SensitivityCorrection._get_filter_sensitivity_data(caldb_filter_string)
        if sensitivity_data is None:
            print(f"Warning: No sensitivity data found for filter {caldb_filter_string}")
            return 1.0
        
        # Calculate time since start date
        time_delta_seconds = SensitivityCorrection._seconds_since_start_date(obs_time)
        if time_delta_seconds < 0:
            print(f"Warning: Observation time predates UVOT sensitivity start date")
            return 1.0
        
        # Extract data from table
        times_since_start = sensitivity_data['TIME'].astype(float)
        offsets = sensitivity_data['OFFSET'].astype(float)
        slopes = sensitivity_data['SLOPE'].astype(float)
        
        # Find the latest row with TIME <= obs_time
        idx = np.where(times_since_start <= time_delta_seconds)[0]
        
        # If observation time predates CALDB range, return no correction
        if idx.size == 0:
            return 1.0
        
        # Use the latest applicable time segment
        i = idx.max()
        dt_years = (time_delta_seconds - times_since_start[i]) / SensitivityCorrection.SECONDS_IN_A_YEAR
        
        # Calculate correction factor
        correction_factor = (1.0 + offsets[i]) * (1.0 + slopes[i]) ** dt_years
        
        return float(correction_factor)
    
    @staticmethod
    def _get_filter_sensitivity_data(caldb_filter_string: str):
        """
        Search through the FITS extensions to find data for the specified filter.
        
        Parameters
        ----------
        caldb_filter_string : str
            Filter name as it appears in CALDB FITS headers
            
        Returns
        -------
        astropy.io.fits.FITS_rec or None
            Sensitivity data table for the filter
        """
        try:
            with fits.open(SensitivityCorrection.CALDB_SENSITIVITY_PATH) as hdulist:
                for hdu in hdulist:
                    if hasattr(hdu, 'header') and hdu.header.get('FILTER') == caldb_filter_string:
                        return hdu.data.copy()
        except Exception as e:
            print(f"Error reading CALDB file: {e}")
            return None
        
        return None
    
    @staticmethod
    def _seconds_since_start_date(t: Time) -> float:
        """
        Calculate seconds elapsed since UVOT sensitivity start date.
        
        Parameters
        ----------
        t : astropy.time.Time
            Time to convert
            
        Returns
        -------
        float
            Seconds since start date
        """
        time_delta = t - SensitivityCorrection.UVOT_SENSITIVITY_START_DATE
        return time_delta.to_value(u.s)
    
def get_effective_area(filter_name, transmission=True, bandpass=True, obs_time=None):
    """
    Get the effective area of a filter.

    Parameters
    ----------
    filter_name : str
        The name of the filter.
    transmission : bool, optional
        Whether to return the transmission or the effective area.
    bandpass : bool, optional
        Whether to return the bandpass object or floats.
    obs_time : astropy.time.Time, optional
        Observation time. If provided, the sensitivity correction factor will be applied.

    Returns
    -------
    arf_wave : array_like
        The wavelength in Angstrom.
    arf_area : array_like
        The effective area in cm^2 (transmission=False) or transmission curve in unitless (transmission=True).
    """
    filter_filename = normalize_filter_name(filter_name.lower(), output_format='filename')
    package_path = paths.package_uvotimgpy
    arf_path = paths.get_subpath(package_path, 'auxil', f'arf_{filter_filename}.fits')
    arf_data = fits.getdata(arf_path, ext=1)
    arf_wave = (arf_data['WAVE_MIN']+arf_data['WAVE_MAX'])/2
    arf_area = arf_data['SPECRESP']
    if transmission:
        area = np.pi*15*15
        arf_area = arf_area/area
    if obs_time is not None:
        correction_factor = SensitivityCorrection.get_correction_factor(filter_name, obs_time)
        arf_area = arf_area / correction_factor
    if bandpass:
        bandpass = create_bandpass(arf_wave*u.AA, arf_area)
        return bandpass
    else:
        return arf_wave, arf_area


if __name__ == '__main__':
    start_time = Time('2010-07-31T19:37:00.000')
    end_time = Time('2025-07-31T19:37:00.000')
    time_array = start_time + (end_time - start_time) * np.linspace(0, 1, 100)
    correction_list = []
    for obs_time in time_array:
        correction = SensitivityCorrection.get_correction_factor('uvw1', obs_time)
        correction_list.append(correction)
    #import matplotlib.pyplot as plt
    #plt.plot(time_array.jd, correction_list)
    #plt.show()