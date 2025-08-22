from typing import Union, List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
from astropy.io import fits
from uvotimgpy.base.region import create_rectangle_region
from uvotimgpy.base.math_tools import fit_peak_in_region
from uvotimgpy.utils.image_operation import DS9Converter, crop_image

def correct_offset_in_image(img_path: Union[str, Path], 
                            img_extension: str,
                            target_coord: Tuple[float, float], 
                            datatype: Optional[str] = None,
                            box_size: Tuple[int, int] = (41, 41), plot: bool = False,
                            save: bool = False, verbose: bool = False) -> np.ndarray:
    """
    修正offset
    box_size: width, height
    """
    with fits.open(img_path, mode='readonly') as hdul:
        img = hdul[img_extension].data.copy()
        err = hdul['ERROR'].data.copy()
        exp = hdul['EXPOSURE'].data.copy()
        has_starmask = 'STARMASK' in hdul
        if has_starmask:
            mask = hdul['STARMASK'].data.copy()
    region = create_rectangle_region(target_coord, box_size[0], box_size[1])
    col_orig, row_orig, theta = fit_peak_in_region(img, region, plot=plot)
    if verbose:
        print(f'Offset correction parameters: col_orig: {col_orig}, row_orig: {row_orig}, theta: {theta}')
    if save:
        true_coord_py = (round(col_orig), round(row_orig))
        img_shifted, err_shifted = crop_image(img, true_coord_py, target_coord, fill_value=np.nan, image_err=err)
        if datatype == 'image':
            exp_shifted = crop_image(exp, true_coord_py, target_coord, fill_value=0)
        elif datatype == 'Multiple':
            print("Multiple datatypes found! Offset of exposure map is not corrected.")
        if has_starmask:
            mask_shifted = crop_image(mask, true_coord_py, target_coord, fill_value=0)
        with fits.open(img_path, mode='update') as hdul:
            primary_hdu = hdul[0]
            primary_hdu.header['COLOFF'] = round(col_orig) - round(target_coord[0])
            primary_hdu.header['ROWOFF'] = round(row_orig) - round(target_coord[1])
            primary_hdu.header['OFFCORR'] = (True, 'Offset correction applied')
            hdul[img_extension].data = img_shifted
            hdul['ERROR'].data = err_shifted
            if datatype == 'image':
                hdul['EXPOSURE'].data = exp_shifted
            if has_starmask:
                hdul['STARMASK'].data = mask_shifted
        #self.offset_correction = 2
        if verbose:
            print(f'Offset correction applied, saved to {img_path}')