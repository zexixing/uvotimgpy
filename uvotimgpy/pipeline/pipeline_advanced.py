import os
import shutil
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import math
from regions import PixelRegion
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from sbpy.activity.gas import VectorialModel

import matplotlib.pyplot as plt

from uvotimgpy.config import paths
from uvotimgpy.pipeline.pipeline_basic import is_path_like, load_path, table_to_df, table_to_list, get_obs_path, BasicInfo, DataPreparation
from uvotimgpy.base.math_tools import ErrorPropagation, UnitConverter
from uvotimgpy.utils.image_operation import calc_radial_profile
from uvotimgpy.uvot_analysis.activity import create_vectorial_model, save_vectorial_model_to_csv, \
    countrate_to_emission_flux_for_oh, emission_flux_to_total_number, TotalNumberCalculator, \
    scale_from_total_number, RatioCalculator_V_UV
from uvotimgpy.base.visualizer import smart_float_format
from uvotimgpy.base.instruments import get_effective_area, normalize_filter_name, format_bandpass
from uvotimgpy.utils.spectrum_operation import SolarSpectrum, ReddeningSpectrum, FluxConverter
from uvotimgpy.uvot_analysis.aperture_photometry import BackgroundEstimator, perform_photometry, AfrhoCalculator
from uvotimgpy.base.region import create_circle_region, create_circle_annulus_region, RegionConverter, create_smeared_region_from_obs
from uvotimgpy.base.file_and_table import TableConverter, save_astropy_table

def resolve_profile_arg(arg: Optional[Any], fallback: Optional[Any]):
    """
    Utility to resolve function arguments:
    - If `arg` is not None, convert it to numpy array and return.
    - Else use `fallback`.
    - If both are None, raise ValueError.
    """
    if arg is not None:
        return np.array(arg)
    return fallback


# ===================== 背景和基础测量 =====================
#class BasicParas:
#    def __init__(self, obs_time: Union[Time, str]):
#        self.filt_filename_v = 'uvv'
#        self.filt_filename_uw1 = 'uw1'
#        self.filt_display_v = 'V'
#        self.filt_display_uw1 = 'UVW1'
#        self.obs_time = Time(obs_time) if isinstance(obs_time, str) else obs_time
#        self.effective_wave_uv = 3325.72*u.AA
#        self.effective_wave_v = 5437.83*u.AA
#        self.central_wave = 4381.775*u.AA
#        self.area = np.pi*15*15 * u.cm*u.cm
#        self.bandpass_v = get_effective_area(self.filt_filename_v, transmission=True, bandpass=True, obs_time=self.obs_time)
#        self.bandpass_uw1 = get_effective_area(self.filt_filename_uw1, transmission=True, bandpass=True, obs_time=self.obs_time)
#        self.filt_dict = {
#            'uvv': {
#                'filename': 'uvv',
#                'display': 'V',
#                'bandpass': self.bandpass_v,
#            },
#            'uw1': {
#                'filename': 'uw1',
#                'display': 'UVW1',
#                'bandpass': self.bandpass_uw1,
#            },
#        }
#        self.sun = SolarSpectrum.from_model()
        

class PhotometryAnalysis:
    """基础测量功能"""
    
    def __init__(self,
                 image_path_or_name: Optional[Union[str, Path]],
                 basic_info: BasicInfo,
                 aperture: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray, float, Tuple[float, float]]] = None,
                 aperture_motion: bool = True,
                 elapsed_time: float = None,
                 bkg_region: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray, float, Tuple[float, float]]] = None,
                 obs_paras: Optional[Dict[str, Any]] = {},
                 reddening: Optional[float] = 0,
                 scale: float = 1.004,
                 bad_pixel_mask: Optional[np.ndarray] = None,
                 verbose: bool = True):
        """
        aperture: (inner_radius, outer_radius) in pixel
        bkg_region: (inner_radius, outer_radius) in pixel
        """
        self.basic_info = basic_info
        self.reddening = reddening
        self.scale = scale
        self.image_path = load_path(image_path_or_name, basic_info.stacked_folder_path)
        self.image = fits.getdata(self.image_path, 'IMAGE')
        self.error = fits.getdata(self.image_path, 'ERROR')
        self.exp_map = fits.getdata(self.image_path, 'EXPOSURE')
        try:
            self.starmask = fits.getdata(self.image_path, 'STARMASK')
        except:
            self.starmask = None
        self.header = fits.getheader(self.image_path, 0)
        self.unit = obs_paras.get('BUNIT', self.header['BUNIT'])
        if self.unit == 'count':
            self.image = self.image/self.exp_map
            self.error = self.error/self.exp_map
        elif self.unit == 'count/s':
            pass
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")
        filt = obs_paras.get('FILTER', self.header['FILTER'])
        self.filt_filename = normalize_filter_name(filt, output_format='filename')
        self.midtime = Time(obs_paras.get('MIDTIME', self.header['MIDTIME']))
        self.exptime = obs_paras.get('EXPTIME', self.header['EXPTIME'])
        self.target_coord_py = (obs_paras.get('COLPIXEL', self.header['COLPIXEL']), 
                                obs_paras.get('ROWPIXEL', self.header['ROWPIXEL']))
        self.filt_displayname = basic_info.filt_dict[self.filt_filename]['display']
        self.bandpass = basic_info.filt_dict[self.filt_filename]['bandpass']
        self.sun = basic_info.sun
        self.bkg_for_single_pixel = None
        self.bkg_err_for_single_pixel = None
        #self.bkg_err_for_mean = None # error of the mean of the background
        self.bkg_for_multi_aperture = None
        self.bkg_err_for_multi_aperture = None
        self.sky_motion = obs_paras.get('Sky_motion', self.header['Sky_motion'])
        self.sky_motion_pa = obs_paras.get('Sky_mot_PA', self.header['Sky_mot_PA'])
        self.aperture = aperture
        self.bkg_region = bkg_region
        self.elapsed_time = elapsed_time
        self.aperture_motion = aperture_motion
        self.verbose = verbose
        self.bad_pixel_mask = bad_pixel_mask
        self.radius_inner = None

    def get_regions(self):
        if self.aperture is None:
            raise ValueError("Aperture is not set.")
        elif isinstance(self.aperture, Union[float, int]):
            self.radius_outer = self.aperture
            if self.aperture_motion:
                self.aperture = create_smeared_region_from_obs(self.target_coord_py, self.aperture, 
                                                               self.elapsed_time, 
                                                               self.sky_motion, 
                                                               self.sky_motion_pa, 
                                                               self.scale)
            else:
                self.aperture = create_circle_region(self.target_coord_py, self.aperture)
        elif isinstance(self.aperture, tuple):
            self.radius_inner = self.aperture[0]
            self.radius_outer = self.aperture[1]
            self.aperture = create_circle_annulus_region(self.target_coord_py, self.aperture[0], self.aperture[1])

        aperture_bool = RegionConverter.to_bool_array_general(self.aperture, combine_regions=True, shape=self.image.shape)[0]
        if self.bad_pixel_mask is not None:
            valid_bool = aperture_bool & ~self.bad_pixel_mask
        else:
            valid_bool = aperture_bool
        self.aperture_valid_number = np.sum(valid_bool)

        if self.bkg_region is None:
            raise ValueError("Background region is not set.")
        elif isinstance(self.bkg_region, Union[float, int]):
            self.bkg_radius_outer = self.bkg_region
            self.bkg_region = create_circle_region(self.target_coord_py, self.bkg_region)
        elif isinstance(self.bkg_region, tuple):
            self.bkg_radius_inner = self.bkg_region[0]
            self.bkg_radius_outer = self.bkg_region[1]
            self.bkg_region = create_circle_annulus_region(self.target_coord_py, self.bkg_region[0], self.bkg_region[1])

    def display_regions(self, 
                        xradius: Optional[float] = None,
                        yradius: Optional[float] = None,
                        vrange: Optional[Tuple[float, float]] = None,
                        ):
        # 图示测量区域
        bkg_region = RegionConverter.to_bool_array_general(self.bkg_region, combine_regions=True, shape=self.image.shape)[0]
        aperture = RegionConverter.to_bool_array_general(self.aperture, combine_regions=True, shape=self.image.shape)[0]
        if self.bad_pixel_mask is not None:
            img[self.bad_pixel_mask] = np.nan
        img = self.image.copy()
        if vrange is None:
            vmin = 0
            vmax = np.nanmax(img)*0.1
        else:
            vmin = vrange[0]
            vmax = vrange[1]
        plt.imshow(img, vmin=vmin, vmax=vmax)
        if xradius is None and self.bkg_radius_outer is not None:
            xradius = self.bkg_radius_outer + 20
        if yradius is None and self.bkg_radius_outer is not None:
            yradius = self.bkg_radius_outer + 20
        if xradius is not None:
            plt.xlim(self.target_coord_py[0]-xradius, self.target_coord_py[0]+xradius)
        if yradius is not None:
            plt.ylim(self.target_coord_py[1]-yradius, self.target_coord_py[1]+yradius)
        plt.contour(aperture, levels=[0.5], colors='red', linewidths=0.5)
        plt.contour(bkg_region, levels=[0.5], colors='white', linewidths=0.5)
        plt.show(block=True)
        plt.close()
    
    def get_background(self, 
                           method: str = 'single_pixel', # 'single_pixel', 'multi_region'
                           multi_apertures_params: Optional[dict] = None,
                           mean_or_median: str = 'mean', # 'mean' or 'median'
                           plot: bool = False,
                           plot_bin_num: int = 100,
                           ):
        """
        测量背景亮度
        multi_apertures_params = {
            'region_creation_func': None,
            'radius_inner': None,
            'radius_outer': 100.0,
            'n_samples': 500,
        }
        """
        image = self.image.copy()
        # 实现不同的背景测量方法
        if method == 'single_pixel':
            # 单区域方法
            self.bkg_for_single_pixel, self.bkg_err_for_single_pixel = \
                BackgroundEstimator.for_single_pixel(image = image, regions = self.bkg_region, 
                                                     bad_pixel_mask = self.bad_pixel_mask, 
                                                     method = mean_or_median,
                                                     plot = plot,
                                                     plot_bin_num = plot_bin_num,
                                                     verbose = self.verbose)
            if self.verbose:
                print(f'{self.filt_displayname} image (single pixel): bkg = ({smart_float_format(self.bkg_for_single_pixel)} +/- {smart_float_format(self.bkg_err_for_single_pixel)}) ctns/s/pixel')
        elif method == 'multi_region':
            # 多区域方法
            region_creation_func = multi_apertures_params.get('region_creation_func', None)
            radius_inner = multi_apertures_params.get('radius_inner', self.radius_inner)
            radius_outer = multi_apertures_params.get('radius_outer', self.radius_outer)
            n_samples = multi_apertures_params.get('n_samples', 500)
            region_creation_func = multi_apertures_params.get('region_creation_func', None)
            self.bkg_for_multi_aperture, self.bkg_err_for_multi_aperture = \
                BackgroundEstimator.from_multiple_apertures(image = image, 
                                                            background_center_region = self.bkg_region,
                                                            region_creation_func = region_creation_func,
                                                            radius_inner = radius_inner,
                                                            radius_outer = radius_outer,
                                                            n_samples = n_samples,
                                                            bad_pixel_mask = self.bad_pixel_mask,
                                                            method = mean_or_median,
                                                            plot = plot,
                                                            plot_bin_num = plot_bin_num,
                                                            verbose = self.verbose)
            if self.verbose:
                print(f"{self.filt_displayname} image (multiple apertures): bkg = ({smart_float_format(self.bkg_for_multi_aperture)} +/- {smart_float_format(self.bkg_err_for_multi_aperture)}) ctns/s")
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def get_total_signal(self, 
                         image: np.ndarray,
                         error: np.ndarray,
                         ):
        #aperture = RegionConverter.to_bool_array_general(aperture, combine_regions=True, shape=image.shape)[0]
        signal, signal_error = perform_photometry(image, background=0, regions=self.aperture, 
                                                  mask=self.bad_pixel_mask, image_err=error, background_err=0)
        return signal, signal_error
    
    def do_aperture_photometry(self,
                               method: str = 'single_pixel', # 'single_pixel', 'multi_region'
                               multi_apertures_params: Optional[dict] = None,
                               bkg_mean_or_median: str = 'mean', # 'mean' or 'median'
                               ):
        """
        aperture photometry
        multi_apertures_params = {
            'radius_inner': None,
            'radius_outer': 100.0,
            'n_samples': 500,
            'region_creation_func': None,
        }
        """
        image = self.image.copy()
        error = self.error.copy()
        self.get_background(method=method, multi_apertures_params=multi_apertures_params, 
                                mean_or_median=bkg_mean_or_median)
        if method == 'single_pixel':
            if self.bkg_for_single_pixel is None and self.bkg_err_for_single_pixel is None:
                raise ValueError("Background (single pixel) values are not obtained.")
            background_img_map = np.full_like(image, self.bkg_for_single_pixel)
            background_err_map = np.full_like(image, self.bkg_err_for_single_pixel)
            self.net_image, self.net_error = ErrorPropagation.subtract(image, error, background_img_map, background_err_map)
            self.countrate, self.countrate_err = self.get_total_signal(image = self.net_image, error = self.net_error, )
        elif method == 'multi_region':
            if self.bkg_for_multi_aperture is None and self.bkg_err_for_multi_aperture is None:
                raise ValueError("Background (multiple apertures) values are not obtained.")
            signal, signal_error = self.get_total_signal(image = image, error = error)
            self.countrate, self.countrate_err = ErrorPropagation.subtract(signal, signal_error, 
                                                                           self.bkg_for_multi_aperture, 
                                                                           self.bkg_err_for_multi_aperture)
        if self.verbose:
            print(f'{self.filt_displayname} image: '
                  f'net_countrate = ({smart_float_format(self.countrate)} +/- '
                  f'{smart_float_format(self.countrate_err)}) cnts/s, '
                  f'snr = {smart_float_format(self.countrate/self.countrate_err)}')
    
    def get_reddened_spectrum(self, reddening: Optional[float] = None):
        reddening = self.reddening if reddening is None else reddening
        red_throughput = ReddeningSpectrum.linear_reddening(reddening, wave0 = self.basic_info.central_wave, wave_grid_range=[1000,13000]*u.AA) # to be changed
        self.reddened_spectrum = self.sun * red_throughput

    def get_flux_results(self,
                         countrate: Optional[float] = None,
                         countrate_err: Optional[float] = None,
                         ):
        countrate = self.countrate if countrate is None else countrate
        countrate_err = self.countrate_err if countrate_err is None else countrate_err
        area = self.basic_info.area
        self.flux = FluxConverter.countrate_to_flux(countrate, self.bandpass, self.reddened_spectrum, area=area, result_unit=u.erg/u.s/u.cm**2)
        self.flux_err = FluxConverter.countrate_to_flux(countrate_err, self.bandpass, self.reddened_spectrum, area=area, result_unit=u.erg/u.s/u.cm**2)
        self.vegamag, self.vegamag_err = FluxConverter.flux_to_vegamag(self.flux, self.bandpass, None, self.flux_err)

        if self.verbose:
            print(f'{self.filt_displayname} image: flux = ({smart_float_format(self.flux.value)} +/- {smart_float_format(self.flux_err.value)}) erg/s/cm^2')
            print(f'{self.filt_displayname} image: mag ({self.filt_displayname}) = ({smart_float_format(self.vegamag.value)} +/- {smart_float_format(self.vegamag_err.value)}) mag')
        
        if self.radius_outer is not None and self.radius_inner is None:
            rh = self.header['R'] * u.au
            delta = self.header['DELTA'] * u.au
            from_phase = self.header['ALPHA'] * u.deg
            aper = self.radius_outer
            aper = UnitConverter.arcsec_to_km(aper*self.scale, delta.value) * u.km
            self.afrho, self.afrho_err = AfrhoCalculator.from_flux(flux = self.flux, 
                                                                   bandpass = self.bandpass, 
                                                                   sun = self.sun, 
                                                                   rh = rh, 
                                                                   delta = delta, 
                                                                   aper = aper, 
                                                                   from_phase = from_phase, 
                                                                   area = None, 
                                                                   flux_err = self.flux_err)
            if self.verbose:
                print(f'{self.filt_displayname} image: afrho = ({smart_float_format(self.afrho.value)} +/- {smart_float_format(self.afrho_err.value)}) cm')

    def convert_mag(self, magnitude_filter: str,): # 'johnson,v', 'johnson,r'
        bp2 = format_bandpass(magnitude_filter)
        flux2 = FluxConverter.flux1_to_flux2(self.flux, self.bandpass, bp2, self.reddened_spectrum)
        flux2_err = FluxConverter.flux1_to_flux2(self.flux_err, self.bandpass, bp2, self.reddened_spectrum)
        self.vegamag2, self.vegamag2_err = FluxConverter.flux_to_vegamag(flux2, bp2, None, flux2_err)
        if self.verbose:
            print(f'{self.filt_displayname} image: mag ({magnitude_filter}) = ({smart_float_format(self.vegamag2.value)} +/- {smart_float_format(self.vegamag2_err.value)}) mag')

    def get_profile(self,
                    step: float = 2,
                    profile_center: Optional[Tuple[float, float]] = None,
                    bad_pixel_mask: Optional[np.ndarray] = None,
                    sector_pa: Optional[float] = None,
                    sector_span: Optional[float] = None,
                    start: Optional[float] = None,
                    end: Optional[float] = None,
                    method: str = 'mean', # 'mean' or 'median'
                    median_err_params: Optional[dict] = {'method':'mean', 'mask':True}):
        results = calc_radial_profile(image = self.net_image,
                                      image_err = self.net_error,
                                      center = profile_center,
                                      step = step,
                                      bad_pixel_mask = bad_pixel_mask,
                                      sector_pa = sector_pa,
                                      sector_span = sector_span,
                                      start = start,
                                      end = end,
                                      method = method,
                                      median_err_params = median_err_params)
        if self.error is None:
            self.profile_rho, self.profile_value = results
        else:
            self.profile_rho, self.profile_value, self.profile_err = results


# ===================== Water Production分析 =====================
class OHAnalysis:
    """Water production rate分析相关功能"""
    def __init__(self, basic_info: BasicInfo, reddening: float = 0,
                 netimg_v: Tuple[np.ndarray, np.ndarray] = (None, None), # countrate img
                 netimg_uw1: Tuple[np.ndarray, np.ndarray] = (None, None), # countrate img
                 countrate_results_v: Tuple[float, float] = (None, None),
                 countrate_results_uw1: Tuple[float, float] = (None, None),
                 profile_results_v: Tuple[Union[List, np.ndarray], Union[List, np.ndarray], Union[List, np.ndarray]] = (None, None, None), #countrate
                 profile_results_uw1: Tuple[Union[List, np.ndarray], Union[List, np.ndarray], Union[List, np.ndarray]] = (None, None, None), #countrate
                 radius_inner: float = None, # pixel
                 radius_outer: float = 100.0, # pixel
                 scale: float = 1.004,
                 verbose: bool = True,
                 ):
        if all(x is None for x in (netimg_v, netimg_uw1, countrate_results_v, countrate_results_uw1, profile_results_v, profile_results_uw1)):
            raise ValueError("At least one kind of the input parameters must be provided.")
        self.basic_info = basic_info
        self.observation_info = basic_info.observation_info
        self.km_per_arcsec = basic_info.km_per_arcsec
        self.obs_time = Time(self.observation_info['mid_time'])
        self.rh = self.observation_info['r']
        self.delta = self.observation_info['delta']
        self.rhv = self.observation_info['r_rate']
        self.reddening = reddening
        self.beta = RatioCalculator_V_UV.dust_countrate_ratio_from_reddening(reddening=self.reddening, obs_time=self.obs_time)
        self.g_factor = None
        self.netimg_v, self.netimg_v_err = netimg_v # countrate img
        self.netimg_uw1, self.netimg_uw1_err = netimg_uw1 # countrate img
        self.oh_image = None
        self.oh_image_err = None
        self.target_coord_py = None
        self.verbose = verbose
        # fitting & scale
        self.profile_rho_v, self.profile_value_v, self.profile_err_v = profile_results_v
        self.profile_rho_uw1, self.profile_value_uw1, self.profile_err_uw1 = profile_results_uw1
        self.profile_rho_v = resolve_profile_arg(self.profile_rho_v, None)
        self.profile_value_v = resolve_profile_arg(self.profile_value_v, None)
        self.profile_err_v = resolve_profile_arg(self.profile_err_v, None)
        self.profile_rho_uw1 = resolve_profile_arg(self.profile_rho_uw1, None)
        self.profile_value_uw1 = resolve_profile_arg(self.profile_value_uw1, None)
        self.profile_err_uw1 = resolve_profile_arg(self.profile_err_uw1, None)
        self.oh_profile_value = None
        self.oh_profile_rho = None
        self.oh_profile_err = None
        self.oh_column_density = None
        self.oh_column_density_err = None
        #self.fitting_info = None
        # scale
        #self.aperture_info = None
        self.countrate_v, self.countrate_v_err = countrate_results_v
        self.countrate_uw1, self.countrate_uw1_err = countrate_results_uw1
        self.oh_countrate = None
        self.oh_countrate_err = None
        self.oh_number = None
        self.oh_number_err = None
        self.oh_flux = None
        self.oh_flux_err = None
        self.oh_luminosity = None
        self.oh_luminosity_err = None
        # water
        self.radius_inner = radius_inner # pixel
        self.radius_outer = radius_outer # pixel
        self.scale = scale
        self.base_q = 1e28
        self.vm = None
        self.q_h2o = None
        self.q_h2o_err = None
        self.active_area = None
        self.active_area_err = None

    def get_oh_image(self):
        dust_in_uw1 = self.beta*self.netimg_v
        dust_in_uw1_err = self.beta*self.netimg_v_err
        self.oh_image, self.oh_image_err = ErrorPropagation.subtract(self.netimg_uw1, 
                                                                     self.netimg_uw1_err, 
                                                                     dust_in_uw1, 
                                                                     dust_in_uw1_err)
    
    def get_oh_profile_from_image(self, 
                                  oh_image: Optional[np.ndarray] = None,
                                  oh_image_err: Optional[np.ndarray] = None,
                                  center: Optional[Tuple[float, float]] = None,
                                  step: float = 2,
                                  bad_pixel_mask: Optional[np.ndarray] = None,
                                  sector_pa: Optional[float] = None,
                                  sector_span: Optional[float] = None,
                                  start: Optional[float] = None,
                                  end: Optional[float] = None,
                                  method: str = 'mean', # 'mean' or 'median'
                                  median_err_params: Optional[dict] = {'method':'mean', 'mask':True}):
        if oh_image is None and oh_image_err is None:
            oh_image = self.oh_image.copy()
            oh_image_err = self.oh_image_err.copy()
        center = self.target_coord_py if center is None else center
        results = calc_radial_profile(image = oh_image,
                                      center = self.target_coord_py,
                                      step = step,
                                      image_err = oh_image_err,
                                      bad_pixel_mask = bad_pixel_mask,
                                      sector_pa = sector_pa,
                                      sector_span = sector_span,
                                      start = start,
                                      end = end,
                                      method = method,
                                      median_err_params = median_err_params)
        if oh_image_err is None:
            self.oh_profile_rho, self.oh_profile_value = results
        else:
            self.oh_profile_rho, self.oh_profile_value, self.oh_profile_err = results
    
    def get_oh_profile_from_profiles(self, 
                                     profile_rho_v: Union[List, np.ndarray] = None,
                                     profile_value_v: Union[List, np.ndarray] = None,
                                     profile_err_v: Union[List, np.ndarray] = None,
                                     profile_rho_uw1: Union[List, np.ndarray] = None,
                                     profile_value_uw1: Union[List, np.ndarray] = None,
                                     profile_err_uw1: Union[List, np.ndarray] = None,
                                     step: float = 2,
                                     ):
        profile_rho_v = resolve_profile_arg(profile_rho_v, self.profile_rho_v)
        profile_value_v = resolve_profile_arg(profile_value_v, self.profile_value_v)
        profile_err_v = resolve_profile_arg(profile_err_v, self.profile_err_v)
        profile_rho_uw1 = resolve_profile_arg(profile_rho_uw1, self.profile_rho_uw1)
        profile_value_uw1 = resolve_profile_arg(profile_value_uw1, self.profile_value_uw1)
        profile_err_uw1 = resolve_profile_arg(profile_err_uw1, self.profile_err_uw1)
        if (profile_rho_v is None or profile_value_v is None or \
            profile_rho_uw1 is None or profile_value_uw1 is None):
            raise ValueError("Need full profile information.")
        if not np.array_equal(profile_rho_v, profile_rho_uw1):
            rho_start = max(profile_rho_v.min(), profile_rho_uw1.min())
            rho_end = min(profile_rho_v.max(), profile_rho_uw1.max())
            if rho_start >= rho_end:
                raise ValueError("The range of the two profiles is not overlapped.")
            self.oh_profile_rho = np.arange(rho_start, rho_end+1, step)
            profile_value_v = np.interp(self.oh_profile_rho, profile_rho_v, profile_value_v)
            profile_value_uw1 = np.interp(self.oh_profile_rho, profile_rho_uw1, profile_value_uw1)
            if profile_err_v is not None:
                profile_err_v = np.interp(self.oh_profile_rho, profile_rho_v, profile_err_v)
            if profile_err_uw1 is not None:
                profile_err_uw1 = np.interp(self.oh_profile_rho, profile_rho_uw1, profile_err_uw1)
        dust_in_uw1_profile_value = profile_value_v * self.beta
        if profile_err_v is not None and profile_err_uw1 is not None:
            dust_in_uw1_profile_err = profile_err_v * self.beta
            self.oh_profile_value, self.oh_profile_err = ErrorPropagation.subtract(profile_value_uw1, profile_err_uw1, 
                                                                                   dust_in_uw1_profile_value, dust_in_uw1_profile_err)
        else:
            self.oh_profile_value = profile_value_uw1 - dust_in_uw1_profile_value
            self.oh_profile_err = None
        
    def get_oh_countrate_from_countrates(self, 
                                         countrate_v: Optional[float] = None,
                                         countrate_v_err: Optional[float] = None,
                                         countrate_uw1: Optional[float] = None,
                                         countrate_uw1_err: Optional[float] = None,
                                         ):
        countrate_v = self.countrate_v if countrate_v is None else countrate_v
        countrate_v_err = self.countrate_v_err if countrate_v_err is None else countrate_v_err
        countrate_uw1 = self.countrate_uw1 if countrate_uw1 is None else countrate_uw1
        countrate_uw1_err = self.countrate_uw1_err if countrate_uw1_err is None else countrate_uw1_err
        dust_in_uw1_countrate = countrate_v * self.beta
        if countrate_v_err is not None and countrate_uw1_err is not None:
            dust_in_uw1_countrate_err = countrate_v_err * self.beta
            self.oh_countrate, self.oh_countrate_err = ErrorPropagation.subtract(countrate_uw1, countrate_uw1_err, 
                                                                                 dust_in_uw1_countrate, dust_in_uw1_countrate_err)
        else:
            self.oh_countrate = countrate_uw1 - dust_in_uw1_countrate
            self.oh_countrate_err = None
    
    def get_oh_countrate_from_image(self): # from OH image
        # use circle/ring aperture or smeared circle/ring aperture
        pass

    def get_oh_countrate_from_profile(self): # from OH profile
        pass

    def get_oh_flux(self, oh_countrate: Optional[float] = None, oh_countrate_err: Optional[float] = None):
        if oh_countrate is None and oh_countrate_err is None:
            oh_countrate = self.oh_countrate
            oh_countrate_err = self.oh_countrate_err
        delta_cm = UnitConverter.au_to_km(self.delta) * 1000 * 100
        if oh_countrate_err is None:
            self.oh_flux = countrate_to_emission_flux_for_oh(oh_countrate, oh_countrate_err, obs_time=self.obs_time)
            self.oh_luminosity = self.oh_flux * 4 * np.pi * np.power(delta_cm, 2) * u.erg / u.s
            self.oh_luminosity_err = None
            self.oh_total_number = emission_flux_to_total_number(self.oh_flux, rh=self.rh, delta=self.delta, rhv=self.rhv, emission_flux_err=None)
            self.oh_total_number_err = None
        else:
            self.oh_flux, self.oh_flux_err = countrate_to_emission_flux_for_oh(oh_countrate, oh_countrate_err, obs_time=self.obs_time)
            self.oh_luminosity = self.oh_flux * 4 * np.pi * np.power(delta_cm, 2) * u.erg / u.s
            self.oh_luminosity_err = self.oh_flux_err * 4 * np.pi * np.power(delta_cm, 2) * u.erg / u.s
            self.oh_total_number, self.oh_total_number_err = emission_flux_to_total_number(self.oh_flux, rh=self.rh, delta=self.delta, \
                                                                                           rhv=self.rhv, emission_flux_err=self.oh_flux_err)
        if self.verbose:
            print(f'total OH number = {smart_float_format(self.oh_total_number)} +/- {smart_float_format(self.oh_total_number_err)} molecules')
    
    def get_vectorial_model(self,
                            base_q: Optional[float] = None,
                            save_model: bool = True,
                            save_path_or_name: Optional[Union[str, Path]] = None,
                            parent_folder: Optional[Union[str, Path]] = None,
                            comments: Optional[str] = None,
                            ):
        base_q = self.base_q if base_q is None else base_q
        self.vm = create_vectorial_model(self.rh, base_q=base_q)
        if save_model:
            if save_path_or_name is None:
                save_path_or_name = f'vectorial_model_{self.basic_info.epoch_name}.csv'
            if parent_folder is None:
                parent_folder = self.basic_info.project_docs_path
            save_path = load_path(save_path_or_name, parent_folder)
            self.vm_path = save_path
            save_vectorial_model_to_csv(self.vm, save_path, comments=comments)

    
    def get_water_scale(self, 
                        vm: Optional[VectorialModel] = None,
                        oh_total_number: Optional[float] = None,
                        oh_total_number_err: Optional[float] = None,
                        radius_inner: Optional[float] = None,
                        radius_outer: Optional[float] = None,
                        scale: Optional[float] = 1.004):
        vm = self.vm if vm is None else vm
        oh_total_number = self.oh_total_number if oh_total_number is None else oh_total_number
        oh_total_number_err = self.oh_total_number_err if oh_total_number_err is None else oh_total_number_err
        radius_inner = self.radius_inner if radius_inner is None else radius_inner
        radius_outer = self.radius_outer if radius_outer is None else radius_outer
        scale = self.scale if scale is None else scale
        radius_outer_km = radius_outer*self.scale*self.km_per_arcsec
        if radius_inner is None:
            total_number_inner = 0.0
        else:
            total_number_inner = TotalNumberCalculator.from_model(self.vm, radius_inner*self.scale*self.km_per_arcsec*u.km)
        total_number_outer = TotalNumberCalculator.from_model(self.vm, radius_outer_km*u.km)
        self.total_number_model = total_number_outer - total_number_inner
        self.q_h2o = scale_from_total_number(oh_total_number, self.total_number_model, base_q=self.base_q)
        if oh_total_number_err is not None:
            self.q_h2o_err = scale_from_total_number(oh_total_number_err, self.total_number_model, base_q=self.base_q)
        else:
            self.q_h2o_err = None
        if self.verbose:
            if radius_inner is None:
                inner_arcsec = None
            else:
                inner_arcsec = radius_inner*self.scale
            print(f'total OH number model = {smart_float_format(self.total_number_model)} molecules (within the {inner_arcsec}-{radius_outer*self.scale} arcsec aperture)')
            print(f'aperture size = {1} arcsec <-> {smart_float_format(self.km_per_arcsec)} km')
            print(f'H2O production rate = {smart_float_format(self.q_h2o)} +/- {smart_float_format(self.q_h2o_err)} molecules/s')

    def get_water_fitting(self):
        pass

    def get_effective_area(self):
        pass
    
    def get_water_from_countrates(self):
        self.get_oh_countrate_from_countrates()
        self.get_oh_flux()
        self.get_vectorial_model()
        self.get_water_scale()
    
    #def analyze_multiple_reddenings(self, image: np.ndarray, 
    #                              reddenings: List[float], **kwargs) -> pd.DataFrame:
    #    """多个reddening的water production分析"""
    #    all_results = []
    #    for reddening in reddenings:
    #        result = self.analyze_single_reddening(image, reddening, **kwargs)
    #        result['reddening'] = reddening
    #        all_results.append(result)
    #    
    #    df_results = pd.DataFrame(all_results)
    #    self.water_production_results = df_results
    #    return df_results

# ===================== Reddening依赖分析 =====================
class ReddeningAnalysis:
    """Reddening依赖性分析"""
    
    def __init__(self, basic_info: BasicInfo,  
                 aperture: Tuple[Union[float, None], float] = (None, 10),
                 bkg_region: Tuple[float, float] = (60, 90),
                 aperture_motion_v: bool = True,
                 elapsed_time_v: Optional[float] = None,
                 aperture_motion_uw1: bool = False,
                 elapsed_time_uw1: Optional[float] = None,
                 ):
        self.basic_info = basic_info
        self.epoch_name = basic_info.epoch_name
        self.aperture = aperture
        self.bkg_region = bkg_region
        self.aperture_motion_v = aperture_motion_v
        self.aperture_motion_uw1 = aperture_motion_uw1
        self.elapsed_time_v = elapsed_time_v
        self.elapsed_time_uw1 = elapsed_time_uw1
        self.radius_outer = aperture[1]
        self.reddening_list = np.arange(0, 50+1, 1)
        self.beta_list = []
        self.v_flux_list = []
        self.v_flux_err_list = []
        self.v_vegamag_list = []
        self.v_vegamag_err_list = []
        self.oh_countrate_list = []
        self.oh_countrate_err_list = []
        self.oh_number_list = []
        self.oh_number_err_list = []
        self.luminosity_list = []
        self.luminosity_err_list = []
        self.q_list = []
        self.q_err_list = []
        self.snr_list = []
    
    def prepare(self):
        phot_analysis_uw1 = PhotometryAnalysis(f'{self.epoch_name}_uw1.fits', basic_info = self.basic_info,
                                               aperture = self.radius_outer, 
                                               aperture_motion = self.aperture_motion_uw1, 
                                               elapsed_time = self.elapsed_time_uw1,
                                               bkg_region = self.bkg_region, reddening = None,
                                               verbose = False)
        phot_analysis_uw1.get_regions()

        if self.aperture_motion_uw1:
            def create_motion_region_uw1(center):
                return create_smeared_region_from_obs(center, self.radius_outer, self.elapsed_time_uw1, 
                                                      phot_analysis_uw1.sky_motion, phot_analysis_uw1.sky_motion_pa, scale=1.004)
        else:
            create_motion_region_uw1 = None
        phot_analysis_uw1.do_aperture_photometry(method = 'multi_region',
                                                 multi_apertures_params = {'region_creation_func': create_motion_region_uw1,
                                                                           'n_samples': 2000})
        self.phot_analysis_uw1 = phot_analysis_uw1

        phot_analysis_v = PhotometryAnalysis(f'{self.epoch_name}_uvv.fits', basic_info = self.basic_info,
                                             aperture = self.radius_outer, 
                                             aperture_motion = self.aperture_motion_v, 
                                             elapsed_time = self.elapsed_time_v,
                                             bkg_region = self.bkg_region, reddening = None,
                                             verbose = False)
        phot_analysis_v.get_regions()
        if self.aperture_motion_v:
            def create_motion_region_v(center):
                return create_smeared_region_from_obs(center, self.radius_outer, self.elapsed_time_v, 
                                                      phot_analysis_v.sky_motion, phot_analysis_v.sky_motion_pa, scale=1.004)
        else:
            create_motion_region_v = None
        phot_analysis_v.do_aperture_photometry(method = 'multi_region',
                                               multi_apertures_params = {'region_creation_func': create_motion_region_v,
                                                                         'n_samples': 2000})
        self.phot_analysis_v = phot_analysis_v

        oh_analysis = OHAnalysis(basic_info=self.basic_info,
                                 reddening = 0,
                                 countrate_results_v=(phot_analysis_v.countrate, phot_analysis_v.countrate_err),
                                 countrate_results_uw1=(phot_analysis_uw1.countrate, phot_analysis_uw1.countrate_err),
                                 radius_outer = self.radius_outer,
                                 verbose = False,)
        oh_analysis.get_vectorial_model(save_model=True)
        self.vm = oh_analysis.vm

    def get_results_for_single_reddening(self, reddening: float):
        #self.phot_analysis_uw1.get_reddened_spectrum(reddening)
        #self.phot_analysis_uw1.get_flux_results()
        self.phot_analysis_v.get_reddened_spectrum(reddening)
        self.phot_analysis_v.get_flux_results()
        oh_analysis = OHAnalysis(basic_info=self.basic_info,
                                 reddening = reddening,
                                 countrate_results_v=(self.phot_analysis_v.countrate, self.phot_analysis_v.countrate_err),
                                 countrate_results_uw1=(self.phot_analysis_uw1.countrate, self.phot_analysis_uw1.countrate_err),
                                 radius_outer = self.radius_outer,
                                 verbose = False,
                                )
        oh_analysis.vm = self.vm
        oh_analysis.get_oh_countrate_from_countrates()
        oh_analysis.get_oh_flux()
        oh_analysis.get_water_scale()
        self.beta_list.append(oh_analysis.beta)
        self.v_flux_list.append(self.phot_analysis_v.flux)
        self.v_flux_err_list.append(self.phot_analysis_v.flux_err)
        self.v_vegamag_list.append(self.phot_analysis_v.vegamag)
        self.v_vegamag_err_list.append(self.phot_analysis_v.vegamag_err)
        self.oh_countrate_list.append(oh_analysis.oh_countrate)
        self.oh_countrate_err_list.append(oh_analysis.oh_countrate_err)
        self.oh_number_list.append(oh_analysis.oh_total_number)
        self.oh_number_err_list.append(oh_analysis.oh_total_number_err)
        self.luminosity_list.append(oh_analysis.oh_luminosity)
        self.luminosity_err_list.append(oh_analysis.oh_luminosity_err)
        self.q_list.append(oh_analysis.q_h2o)
        self.q_err_list.append(oh_analysis.q_h2o_err)
        self.snr_list.append(oh_analysis.q_h2o/oh_analysis.q_h2o_err)

    def get_results_for_multiple_reddenings(self):
        for reddening in self.reddening_list:
            self.get_results_for_single_reddening(reddening)
        result_dict = {
            'reddening': self.reddening_list*u.percent,
            'beta': self.beta_list,
            'v_flux': self.v_flux_list,
            'v_flux_err': self.v_flux_err_list,
            'v_vegamag': self.v_vegamag_list,
            'v_vegamag_err': self.v_vegamag_err_list,
            'oh_countrate': np.array(self.oh_countrate_list)*u.count/u.s,
            'oh_countrate_err': np.array(self.oh_countrate_err_list)*u.count/u.s,
            'oh_number': np.array(self.oh_number_list),
            'oh_number_err': np.array(self.oh_number_err_list),
            'oh_luminosity': self.luminosity_list,
            'oh_luminosity_err': self.luminosity_err_list,
            'q': np.array(self.q_list),
            'q_err': np.array(self.q_err_list),
            'snr': self.snr_list,
        }
        self.result_dict = result_dict

    def save_results(self, 
                     save_path_or_name: Optional[Union[str, Path]] = None, 
                     parent_folder: Optional[Union[str, Path]] = None):
        if parent_folder is None:
            parent_folder = self.basic_info.project_docs_path
        if save_path_or_name is None:
            save_path_or_name = f'results_{self.epoch_name}.ecsv'
        save_path = load_path(save_path_or_name, parent_folder)
        result_table = TableConverter.list_dict_to_astropy_table(self.result_dict)
        save_astropy_table(result_table, save_path)

    def plot_results(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        color = '#1f77b4'
        ax.plot(self.reddening_list, self.q_list, '-', markersize=6, color=color,
                 linewidth=1.5, markeredgecolor='white', markeredgewidth=0.5)
        ax.fill_between(self.reddening_list, 
                        np.array(self.q_list) - np.array(self.q_err_list), 
                        np.array(self.q_list) + np.array(self.q_err_list),
                        alpha=0.3, color=color, edgecolor='none')
        ax.axhline(y=0, color=color, linestyle='--', linewidth=0.8)
        ax.set_xlabel('Reddening at 4381.8 Å (%/1000 Å)', fontsize=12)
        ax.set_ylabel('Water production rate (molecules/s)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        ax.set_xlim(0, 50)
        ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=0.8)
        ax.tick_params(axis='both', which='minor', length=2, width=0.6)
        ax.minorticks_on()
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(0.8)
            ax.spines[spine].set_color('black')
        #ax.legend(loc='lower left', fontsize=8)
        ax.set_title(f'Epoch {self.epoch_name} - {self.basic_info.observation_info["mid_time"].split("T")[0]}', fontsize=14)
        plt.show(block=True)
        plt.close()

# ===================== 其他分析（预留） =====================
class ProfileAnalysis:
    """Profile分析"""
    
    def __init__(self,
                 basic_info: BasicInfo,
                 image_path_or_name: Optional[Union[str, Path]] = None,
                 obs_paras: Optional[Dict[str, Any]] = {},
                 step: float = 2,
                 bad_pixel_mask: Optional[np.ndarray] = None,
                 sector_pa: Optional[float] = None,
                 sector_span: Optional[float] = None,
                 start: Optional[float] = None,
                 end: Optional[float] = None,
                 method: str = 'mean', # 'mean' or 'median'
                 median_err_params: Optional[dict] = {'method':'mean', 'mask':True}):
        self.analysis_results = {}
        if image_path_or_name is not None:
            pass
        self.image_path = load_path(image_path_or_name, basic_info.stacked_folder_path)
        self.image = fits.getdata(self.image_path, 'IMAGE')
        self.error = fits.getdata(self.image_path, 'ERROR')
        self.exp_map = fits.getdata(self.image_path, 'EXPOSURE')
        try:
            self.starmask = fits.getdata(self.image_path, 'STARMASK')
        except:
            self.starmask = None
        self.header = fits.getheader(self.image_path, 0)
        self.unit = obs_paras.get('BUNIT', self.header['BUNIT'])
        if self.unit == 'count':
            self.image = self.image/self.exp_map
            self.error = self.error/self.exp_map
        elif self.unit == 'count/s':
            pass
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")
        self.target_coord_py = (obs_paras.get('COLPIXEL', self.header['COLPIXEL']), 
                                obs_paras.get('ROWPIXEL', self.header['ROWPIXEL']))
        self.step = step
        self.bad_pixel_mask = bad_pixel_mask
        self.sector_pa = sector_pa
        self.sector_span = sector_span
        self.start = start
        self.end = end
        self.method = method
        self.median_err_params = median_err_params

    
    def get_profile(self):
        results = calc_radial_profile(image = self.image,
                                      image_err = self.error,
                                      center = self.target_coord_py,
                                      step = self.step,
                                      bad_pixel_mask = self.bad_pixel_mask,
                                      sector_pa = self.sector_pa,
                                      sector_span = self.sector_span,
                                      start = self.start,
                                      end = self.end,
                                      method = self.method,
                                      median_err_params = self.median_err_params)
        if self.error is None:
            self.profile_rho, self.profile_value = results
        else:
            self.profile_rho, self.profile_value, self.oh_profile_err = results
    
    def profile_fitting(self,
                        profile_rho: Optional[np.ndarray] = None,
                        profile_value: Optional[np.ndarray] = None,
                        profile_err: Optional[np.ndarray] = None,
                        ): # 1/rho, vm
        pass

    def display_profile(self):
        pass

class MorphologyAnalysis:
    """形态学分析"""
    
    def __init__(self):
        self.morphology_results = {}
    
    def analyze_morphology(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """形态学分析"""
        pass
    
    def extract_morphology_features(self, image: np.ndarray, **kwargs) -> Dict[str, float]:
        """提取形态学特征"""
        pass

# ===================== 主Pipeline类 =====================
class CometPipeline:
    """彗星观测数据处理主Pipeline，整合所有功能"""
    
    def __init__(self):
        # 初始化所有模块
        pass
    
    # 可以在这里添加一些便捷的组合方法
    def run_standard_analysis(self, **kwargs):
        """运行标准分析流程"""
        pass


#这个设计的优点：
#
#模块化：每个功能类独立，职责清晰
#可独立使用：每个类都可以单独导入和使用
#方法都是公开的：去掉了下划线前缀，所有方法都可以被外部调用
#主Pipeline整合：CometPipeline 类整合所有模块，方便统一管理
#灵活性：既可以用单个模块，也可以用主Pipeline
#使用示例：
# 独立使用某个模块
#from pipeline import WaterProductionAnalysis
#water = WaterProductionAnalysis()
#oh_image = water.get_oh_image(image, reddening=0.3)
#
## 使用主Pipeline
#from pipeline import CometPipeline
#pipeline = CometPipeline()
#pipeline.prep.load_observation_log("observations.csv")
#pipeline.clean.stack_images(image_list)
#pipeline.water.analyze_single_reddening(image, 0.3)