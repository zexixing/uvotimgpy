from typing import Union, List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
from astropy.io import fits
from uvotimgpy.base.region import create_rectangle_region
from uvotimgpy.base.math_tools import fit_peak_in_region
from uvotimgpy.utils.image_operation import DS9Converter, crop_image
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def correct_offset_in_image(img_path: Union[str, Path], 
                            img_extension: Union[str, int],
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

def coi_factor_multfunc(raw):
    # Poole et al. 2008, Eq. 2
    # caldb/data/swift/uvota/bcf/coinc/swucountcor20010101v103.fits, ext=1, data
    a1 = 0.066
    a2 = -0.091
    a3 = 0.029
    a4 = 0.031
    ft = 0.0110329
    x = raw*ft
    y = 1+a1*x+a2*np.power(x,2)+\
        a3*np.power(x,3)+a4*np.power(x,4)
    alp = 0.9842
    z = -np.log(1-alp*raw*ft)/(alp*ft)
    f = y*z/raw
    f = np.nan_to_num(f,nan=1.0,posinf=1.0,neginf=1.0)
    return f

def coi_factor_plinfunc(rate):
    # caldb/data/swift/uvota/bcf/coinc/swucountcor20010101v102.fits, ext=1, data
    # /Users/zexixing/Software/heasoft-6.34/swift/uvot/tasks/uvotcoincidence/ut-uvotcoincidence -> uvotcoincidence, caldb
    # /Users/zexixing/Software/heasoft-6.34/swift/uvot/tasks/uvotcoincidence/uvotcoincidence -> findCoincidenceLossFactors
    # /Users/zexixing/Software/heasoft-6.34/swift/uvot/lib/perl/UVOT/Source.pm -> findCoincidenceLossFactors
    a1 = -0.0663428
    a2 = 0.0900434
    a3 = -0.0237695
    a4 = -0.0336789
    df = 0.01577
    ft = 0.0110329
    raw = rate*(1-df)
    x = raw*ft
    x2 = x*x
    f = (-np.log(1.0-x)/(ft*(1.0-df))) / (1.0+a1*x+a2*x2+a3*x2*x+a4*x2*x2)
    f = f/rate
    f = np.nan_to_num(f,nan=1.0,posinf=1.0,neginf=1.0)
    return f

def get_coi_loss_map(img_data, scale=1.004, func = 'poole2008'):
    """
    修正coincidence loss
    img_data, unit: count/s
    scale: 1.004 or 0.502
    func: 'poole2008' or 'caldb'
    """
    # 创建圆形卷积核
    aper = int(np.ceil(5./scale))
    y, x = np.ogrid[-aper:aper+1, -aper:aper+1]
    kernel = (x**2 + y**2 <= aper**2).astype(float)
    
    # 标记nan位置
    nan_mask = np.isnan(img_data)

    if nan_mask.any():
        img_clean = np.nan_to_num(img_data, nan=0)
        sum_map = convolve(img_clean, kernel, mode='constant', cval=0)
    else:
        sum_map = convolve(img_data, kernel, mode='constant', cval=0)

    if func == 'poole2008':
        coi_map = coi_factor_multfunc(sum_map)
    elif func == 'caldb':
        coi_map = coi_factor_plinfunc(sum_map)
    else:
        raise ValueError(f"Unsupported function: {func}")

    if nan_mask.any():
        nan_propagated = convolve(nan_mask.astype(float), kernel, mode='constant') > 0
        coi_map[nan_propagated] = np.nan

    return coi_map
    

def correct_coi_loss_in_image(img_path: Union[str, Path], 
                              img_extension: Union[str, int], 
                              scale: float, 
                              func: str = 'poole2008', 
                              plot: bool = False,
                              save: bool = False,
                              verbose: bool = False):
    """
    修正coincidence loss
    """
    with fits.open(img_path, mode='readonly') as hdul:
        img = hdul[img_extension].data.copy()
        #err = hdul['ERROR'].data.copy()
        exp = hdul['EXPOSURE'].data.copy()
        coi_loss_map = get_coi_loss_map(img/exp, scale, func)
    if plot:
        plt.imshow(coi_loss_map, origin='lower', vmax=1.08, vmin=1.0)
        plt.show(block=True)
    if save:
        with fits.open(img_path, mode='update') as hdul:
            primary_hdu = hdul[0]
            primary_hdu.header['COICORR'] = (True, 'Coincidence loss correction applied')
            hdul[img_extension].data = img*coi_loss_map
            #hdul['ERROR'].data = err*np.sqrt(coi_loss_map) # ?
        if verbose:
            print(f'Coincidence loss correction applied, saved to {img_path}')
    return img*coi_loss_map

if __name__ == '__main__':
    pass
    #import matplotlib.pyplot as plt
    #img = fits.getdata('/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/projects/C_2025N1/stacked_images/epoch2_uvv_sum_wcscorr.fits', 1)
    #plt.imshow(img, origin='lower', vmin=0, vmax=0.05)
    #plt.show()
    #coi_loss_map = get_coi_loss_map(img, 1.004, func='poole2008')
    #print(coi_loss_map[1000, 1000])
    #plt.imshow(coi_loss_map, origin='lower', vmax=1.08, vmin=1.0)
    #plt.show()
    #
    #coi_loss_map = get_coi_loss_map(img, 1.004, func='caldb')
    #print(coi_loss_map[1000, 1000])
    #plt.imshow(coi_loss_map, origin='lower', vmax=1.08, vmin=1.0)
    #plt.show()

    #hdul = fits.open('/Users/zexixing/Software/caldb/data/swift/uvota/bcf/coinc/swucountcor20010101v102.fits')
    #d = hdul[1].data
    #hdul.close()
    #print(d[0][1])