import os
from astropy.io import fits
from datetime import datetime
from astropy.time import Time
import numpy as np
from typing import Union, Tuple, List
from uvotimgpy.base.file_and_table import compress_fits, get_caller_filename, ephemeris_keywords

def save_stacked_fits(images_to_save: dict, 
                      save_path: str, obs_list: list, 
                      target_position: Union[Tuple, List],
                      script_name: str = None, 
                      bkg_list: list = None, bkg_error_list: list = None, 
                      comment: str = None,
                      other_header_info: dict = None,
                      compressed: bool = False,
                      stack_unit: str = 'counts/s',
                      hst: bool = False):
    """
    Save a stacked image as a FITS file.
    
    Parameters
    ----------
    save_path : str
        Save path.
    obs_list : list
        Observation list.
    target_position : Union[Tuple, List]
        Target position, (column, row).
    images_to_save : dict
        Dictionary of images to save; keys are extension names and values are corresponding 2D arrays.
        Example: {'STACKED_IMAGE': stacked_image, 'STACKED_ERROR': stacked_error, 'STACKED_IMAGE_OVERLAP': stacked_image_overlap}
    script_name : str, optional
        Script name.
    bkg_list : list, optional
        List of background values.
    bkg_error_list : list, optional
        List of background error values.
    keyword_dict : dict, optional
        Stores keywords used to look up related information.
        Example: swift: {'file': 'OBSID', 'exp':'EXPOSURE'},
              hst: {'file': 'file_name', 'exp':'EXPTIME'}
    compressed : bool, optional
        Whether to compress the file.
    """
    # Create the primary header
    primary_hdu = fits.PrimaryHDU()
    primary_header = primary_hdu.header
    
    # Get time information and convert it to Time objects
    times_obs = Time([obs['DATE_OBS'] for obs in obs_list])
    times_end = Time([obs['DATE_END'] for obs in obs_list])
    
    # Get the earliest start time and latest end time
    first_date_obs = times_obs.min()
    last_date_end = times_end.max()
    
    # Calculate the midpoint time
    midtime = Time(first_date_obs.jd + (last_date_end.jd - first_date_obs.jd) / 2, format='jd')
    
    # Convert to ISO-format strings for saving in the header
    first_date_obs_str = first_date_obs.iso
    last_date_end_str = last_date_end.iso
    midtime_iso = midtime.iso
    first_date_obs_str = first_date_obs_str.replace(' ', 'T')
    last_date_end_str = last_date_end_str.replace(' ', 'T')
    midtime_iso = midtime_iso.replace(' ', 'T')
    
    # Calculate the total exposure time
    if hst:
        total_exptime = sum(obs['EXPTIME'] for obs in obs_list)
    else:
        total_exptime = sum(obs['EXPOSURE'] for obs in obs_list)
    
    # Get the filter
    filt = obs_list[0]['FILTER']  # Assume all observations use the same filter
    
    # Get the target position
    target_pos = target_position
    
    # Write header information
    if hst:
        primary_header['YEAR'] = (obs_list[0]['date'], 'Observation year')
    primary_header['DATE_OBS'] = (first_date_obs_str, 'Start time of first observation')
    primary_header['DATE_END'] = (last_date_end_str, 'End time of last observation')
    primary_header['MIDTIME'] = (midtime_iso, 'Middle time between first and last observation')
    primary_header['EXPTIME'] = (total_exptime, 'Total exposure time [s]')
    primary_header['BUNIT'] = (stack_unit, 'Physical unit of array values')
    primary_header['COLPIXEL'] = (target_pos[0], 'Target X position in Python coordinates')
    primary_header['ROWPIXEL'] = (target_pos[1], 'Target Y position in Python coordinates')
    primary_header['DS9XPIX'] = (target_pos[0] + 1, 'Target X position in DS9 coordinates')
    primary_header['DS9YPIX'] = (target_pos[1] + 1, 'Target Y position in DS9 coordinates')
    primary_header['FILTER'] = (filt, 'Filter used in observations')

    # Calculate average rh, delta, and elongation
    key_list = obs_list[0].keys()
    ephemeris_keywords_list = ephemeris_keywords()
    ephemeris_keywords_list.extend(['Sky_motion', 'Sky_mot_PA'])
    orbital_keywords = list(set(key_list) & set(ephemeris_keywords_list))
    orbital_keywords = [x for x in orbital_keywords if x not in ['RA', 'DEC', 'ra', 'dec']]
    for key in orbital_keywords:
        avg_value = np.mean([obs[key] for obs in obs_list])
        if key == 'RA*cos(Dec)_rate':
            primary_header['RA_RATE'] = (avg_value, f'Average RA RATE')
        else:
            primary_header[key] = (avg_value, f'Average {key}')

    if bkg_list is not None:
        primary_header['BKG'] = (', '.join(f'{bkg}' for bkg in bkg_list), 'Background values for each image; ctns/s/pixel')
    if bkg_error_list is not None:
        primary_header['BKG_ERR'] = (', '.join(f'{bkg_err}' for bkg_err in bkg_error_list), 'Background error values for each image; ctns/s/pixel')
    
    if other_header_info is not None:
        for key, value in other_header_info.items():
            primary_header[key] = value
    
    primary_header['CREATED'] = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                'File creation time (local)')
    if script_name is None:
        script_name = get_caller_filename()
    primary_header['HISTORY'] = f'Created by {script_name}'
    primary_header['HISTORY'] = f'Created by Zexi Xing'
    
    # Loop through extension information
    for ext_num, ext_name in enumerate(images_to_save.keys(), 1):
        primary_header[f'EXT{ext_num}NAME'] = (ext_name, f'Name of extension {ext_num}')
    
    primary_header.add_comment('Orbital values have the same units as default on JPL Horizons ('+', '.join(orbital_keywords)+').')
    if hst: # HST
        file_list = sorted(set([obs['file_name'] for obs in obs_list]))
        primary_header.add_comment('FILE LIST: '+ ', '.join(f'{file_name}' for file_name in file_list))
    else: # Swift
        primary_header.add_comment('EXP LIST: '+ ', '.join(f'{obs["OBSID"]}'[-4:]+f'_{obs["EXT_NO"]}' for obs in obs_list))
    primary_header.add_comment(comment)
    # Create the HDU list, starting from the primary HDU
    hdul = fits.HDUList([primary_hdu])
    
    # Loop through and create each extension
    for ext_name, image_data in images_to_save.items():
        if image_data is not None:
            if image_data.dtype == bool:
                image_hdu = fits.CompImageHDU(data=image_data.astype(np.uint8), compression_type='RICE_1', name=ext_name)
            else:
                image_hdu = fits.ImageHDU(data=image_data, name=ext_name)
            hdul.append(image_hdu)
    
    # Save the file
    hdul.writeto(save_path, overwrite=True)
    if compressed:
        compress_fits(save_path)

def save_cleaned_fits(images_to_save: dict,
                      save_path: str, obs: dict, 
                      target_position: Union[Tuple, List],
                      script_name: str = None, comment: str = None,
                      other_header_info: dict = None,
                      compressed: bool = True):
    """
    Save a cleaned image as a FITS file.
    
    Parameters
    ----------
    save_path : str
        Save path.
    obs : dict
        Observation information dictionary.
    target_position : Union[Tuple, List]
        Target position.
    images_to_save : dict
        Dictionary of images to save; keys are extension names and values are corresponding 2D arrays.
        Example: {'CLEANED_IMAGE': cleaned_image, 'UNCLEANED_IMAGE': uncleaned_image,
               'CLEANED_ERROR': cleaned_error, 'WHT': wht}
    script_name : str, optional
        Script name.
    compressed : bool, optional
        Whether to compress the file; default is True.
    comment : str, optional
        Comment information.
    """
    primary_hdu = fits.PrimaryHDU()
    primary_header = primary_hdu.header

    for key in obs.keys():
        primary_header[key] = f'{obs[key]}'
    
    target_pos = target_position
    
    primary_header['BUNIT'] = ('COUNTS/S', 'Physical unit of array values')
    primary_header['COLPIXEL'] = (target_pos[0], 'Target column position in Python coordinates')
    primary_header['ROWPIXEL'] = (target_pos[1], 'Target row position in Python coordinates')
    primary_header['DS9XPIX'] = (target_pos[0] + 1, 'Target X position in DS9 coordinates')
    primary_header['DS9YPIX'] = (target_pos[1] + 1, 'Target Y position in DS9 coordinates')

    if other_header_info is not None:
        for key, value in other_header_info.items():
            primary_header[key] = value

    primary_header['CREATED'] = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                'File creation time (local)')
    if script_name is None:
        script_name = get_caller_filename()
    primary_header['HISTORY'] = f'Created by {script_name}'
    primary_header['HISTORY'] = f'Created by Zexi Xing'

    # Loop through extension information
    for ext_num, ext_name in enumerate(images_to_save.keys(), 1):
        primary_header[f'EXT{ext_num}NAME'] = (ext_name, f'Name of extension {ext_num}')
    
    primary_header.add_comment(comment)
    
    # Create the HDU list, starting from the primary HDU
    hdul = fits.HDUList([primary_hdu])
    
    # Loop through and create each extension
    for ext_name, image_data in images_to_save.items():
        if image_data is not None:
            image_hdu = fits.ImageHDU(data=image_data, name=ext_name)
            hdul.append(image_hdu)

    hdul.writeto(save_path, overwrite=True)
    if compressed:
        compress_fits(save_path)
