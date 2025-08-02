from uvotimgpy.config import paths
import astropy.units as u
import numpy as np
from astropy.io import fits
from synphot import SpectralElement, Empirical1D
from typing import Union
import stsynphot as stsyn

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

def get_effective_area(filter_name, transmission=True, bandpass=True):
    """
    Get the effective area of a filter.

    Parameters
    ----------
    filter_name : str
        The name of the filter.
    transmission : bool, optional
        Whether to return the transmission or the effective area.

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
    if bandpass:
        bandpass = create_bandpass(arf_wave*u.AA, arf_area)
        return bandpass
    else:
        return arf_wave, arf_area