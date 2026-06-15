from typing import Union, List, Optional, Tuple, Callable
import numpy as np
from regions import PixelRegion, CirclePixelRegion, PixCoord
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import astropy.units as u
import synphot.units as su
from synphot import SourceSpectrum, Observation, SpectralElement, Empirical1D
from synphot.units import convert_flux
import stsynphot as stsyn
from sbpy.activity import phase_HalleyMarcus, Afrho
from sbpy.data import Ephem
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits

from uvotimgpy.base.region import RegionStatistics, RegionConverter, create_circle_region, create_circle_annulus_region
from uvotimgpy.utils.image_operation import calc_radial_profile
from uvotimgpy.base.math_tools import ErrorPropagation
from uvotimgpy.utils.spectrum_operation import format_bandpass, SolarSpectrum, calculate_flux, TypicalWaveSfluxd

class BackgroundImageCreator:
    @staticmethod
    def image_mode(filter: str, 
                   ):
        pass
        # rotate_image(image, angle)
        # crop_image(sk_hdu.data, self.sk_coord_py, self.target_coord_py, fill_value=np.nan)

    @staticmethod
    def event_mode():
        pass

class BackgroundEstimator:
    """Background estimation class with static methods for calculating background."""
    
    @staticmethod
    def from_regions(image: np.ndarray,
                     regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
                     bad_pixel_mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None,
                     image_err: Optional[np.ndarray] = None,
                     method: str = 'median',
                     median_err_params: Optional[dict] = {'method':'mean', 'mask':True}) -> Union[float, Tuple[float, float]]:
        """Estimate background using region statistics.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        regions : Union[PixelRegion, List[PixelRegion], np.ndarray]
            Regions used to estimate the background; can be a single region, a list of regions, or a boolean mask.
        bad_pixel_mask : np.ndarray, optional
        image_err : np.ndarray, optional
            Error image; default is None.

        Returns
        -------
        Union[float, Tuple[float, float]]
            If image_err is provided, return (background value, error); otherwise return only the background value.
        """
        #if image_err is None:
        #    return RegionStatistics.median(image, regions, combine_regions=True)
        #else:
        #    # Get pixels in the region and their corresponding errors
        #    regions = RegionConverter.to_bool_array_general(regions, combine_regions=True, shape=image.shape)
        #    region = regions[0]
        #    valid_data = image[region & ~np.isnan(image)]
        #    valid_errors = image_err[region & ~np.isnan(image)]
        #    
        #    # Use ErrorPropagation to compute the median and its error
        #    value, error = ErrorPropagation.median((valid_data, valid_errors), axis=None, method='std')
        #    return value, error
        
        # Get pixels in the region and their corresponding errors
        region = RegionConverter.to_bool_array_general(regions, combine_regions=True, shape=image.shape)[0]
        if bad_pixel_mask is not None:
            bad_pixel_mask = RegionConverter.to_bool_array_general(bad_pixel_mask, combine_regions=True, shape=image.shape)[0]
            valid_region = region & ~np.isnan(image) & ~bad_pixel_mask 
        else:
            valid_region = region & ~np.isnan(image)
        valid_data = image[valid_region]
        if image_err is not None:
            valid_errors = image_err[valid_region]
        else:
            valid_errors = np.zeros_like(valid_data)
        
        if method == 'median':
            value, error = ErrorPropagation.median(valid_data, valid_errors, axis=None, ignore_nan=True,
                                                **median_err_params)
        elif method == 'mean':
            value, error = ErrorPropagation.mean(valid_data, valid_errors, axis=None, ignore_nan=True,)
        else:
            raise ValueError(f"Unsupported method: {method}")

        if image_err is not None:
            return value, error
        else:
            return value

    @staticmethod
    def for_single_pixel(image: np.ndarray,
                         regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
                         bad_pixel_mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None,
                         method: str = 'median', 
                         plot: bool = False,
                         plot_bin_num: int = 100,
                         verbose: bool = False):
        region = RegionConverter.to_bool_array_general(regions, combine_regions=True, shape=image.shape)[0]
        if bad_pixel_mask is not None:
            bad_pixel_mask = RegionConverter.to_bool_array_general(bad_pixel_mask, combine_regions=True, shape=image.shape)[0]
            valid_region = region & ~np.isnan(image) & ~bad_pixel_mask 
        else:
            valid_region = region & ~np.isnan(image)
        valid_data = image[valid_region]
        if verbose:
            print(f'valid pixel number for background = {len(valid_data)}')
        error = np.std(valid_data)
        if method == 'median':
            value = np.median(valid_data)
        elif method == 'mean':
            value = np.mean(valid_data)
        if plot:
            print(f'mean = {np.mean(valid_data)}, median = {np.median(valid_data)}, std = {np.std(valid_data)}')
            plt.hist(valid_data, bins=plot_bin_num)
            plt.axvline(np.median(valid_data), color='blue', linestyle='--', lw=0.5, label='median')
            plt.axvline(np.mean(valid_data), color='red', linestyle='--', lw=0.5, label='mean')
            plt.legend()
            plt.show(block=True)
            plt.close()
        return value, error
        
    @staticmethod
    def from_multiple_apertures(image: np.ndarray,
                                background_center_region: Union[PixelRegion, List[PixelRegion], np.ndarray],
                                region_creation_func: Optional[Callable] = None,
                                radius_inner: Optional[float] = None,
                                radius_outer: float = 100.0,
                                n_samples: int = 500,
                                bad_pixel_mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None,
                                method: str = 'mean',
                                plot: bool = False,
                                plot_bin_num: int = 100,
                                verbose: bool = False,
                                ):
        background_center_region = RegionConverter.to_bool_array_general(background_center_region, combine_regions=True, shape=image.shape)[0]
        if bad_pixel_mask is not None:
            bad_pixel_mask = RegionConverter.to_bool_array_general(bad_pixel_mask, combine_regions=True, shape=image.shape)[0]
            image[bad_pixel_mask] = np.nan
        background_center_region = background_center_region & ~np.isnan(image)
        valid_indices = np.argwhere(background_center_region)

        if len(valid_indices) < n_samples:
            raise ValueError(f"The number of valid pixels is less than the required sample number: {len(valid_indices)} < {n_samples}")
        if verbose:
            print(f'valid pixel number for background = {len(valid_indices)}, n_samples = {n_samples}')
        selected_idx = np.random.choice(len(valid_indices), size=n_samples, replace=False)
        selected_indices = valid_indices[selected_idx]

        background_samples = []
        for row, col in selected_indices:
            if region_creation_func is not None:
                region = region_creation_func((col, row))
            elif radius_inner is None:
                region = create_circle_region((col, row), radius_outer)
            else:
                region = create_circle_annulus_region((col, row), radius_inner, radius_outer)
            region_bool = RegionConverter.region_to_bool_array(region, image.shape)
            valid_pixels_in_region = image[region_bool]
            background_samples.append(np.sum(valid_pixels_in_region))
        error = np.nanstd(background_samples)
        if method == 'mean':
            value = np.nanmean(background_samples)
        elif method == 'median':
            value = np.nanmedian(background_samples)
        else:
            raise ValueError(f"Unsupported method: {method}")
        if plot:
            print(f'mean = {np.nanmean(background_samples)}, median = {np.nanmedian(background_samples)}, std = {np.nanstd(background_samples)}')
            plt.hist(background_samples, bins=plot_bin_num)
            plt.axvline(np.nanmedian(background_samples), color='blue', linestyle='--', lw=0.5, label='median')
            plt.axvline(np.nanmean(background_samples), color='red', linestyle='--', lw=0.5, label='mean')
            plt.legend()
            plt.show(block=True)
            plt.close()
        return value, error
        
    @staticmethod
    def from_profile(image: np.ndarray,
                     center: tuple,
                     fit_range: Tuple[float, float],
                     step: float = 1.0,
                     fit_func: str = 'power_law',
                     p0 = [1, 1, 0],
                     rho_target: Optional[float] = None,
                     image_err: Optional[np.ndarray] = None,
                     bad_pixel_mask: Optional[np.ndarray] = None,
                     median_err_params: Optional[dict] = {'method':'mean', 'mask':True}
                     ) -> Union[float, Tuple[float, float]]:
        """Estimate background using a radial-profile fitting method.

        Parameters
        ----------
        image : np.ndarray
            Input image.
        center : tuple
            Center coordinates (col, row).
        fit_range : tuple
            Distance range used for fitting (start, end).
        step : float, optional
            Step size for the radial profile; default is 1.0.
        fit_func : str, optional
            Fit function type; options are 'linear' or 'power_law', default 'linear'.
        rho_target : float, optional
            Background value at the target distance; if None, use the end of fit_range.
        image_err : np.ndarray, optional
            Error image; default is None.

        Returns
        -------
        Union[float, Tuple[float, float]]
            If image_err is provided, return (background value, error); otherwise return only the background value.
        """
        if image_err is None:
            rho, intensity = calc_radial_profile(image, center, step=step, bad_pixel_mask=bad_pixel_mask,
                                                 start=fit_range[0], end=fit_range[1], method='median')
            sigma = None
        else:
            rho, intensity, errors = calc_radial_profile(image, center, step=step, bad_pixel_mask=bad_pixel_mask,
                                                         start=fit_range[0], end=fit_range[1],
                                                         image_err=image_err, method='median', 
                                                         median_err_params=median_err_params)
            sigma = errors  # Used for weighted fitting
            
        # Define the fitting function
        if fit_func == 'power_law':
            def fit_function(x, a, b, c):
                return a * x**(-b) + c
        else:
            raise ValueError("Unsupported fit function")
            
        # Perform the fit
        try: 
            if fit_func == 'power_law':
                popt, pcov = curve_fit(fit_function, rho, intensity,
                                       p0=p0,
                                       sigma=sigma if sigma is not None else None,
                                       absolute_sigma=True if sigma is not None else False)
                a, b, c = popt
                
                if rho_target is not None:
                    # Use the fitted value at the target distance as the background
                    background = fit_function(rho_target, a, b, c)
                    
                    if image_err is not None:
                        # Use 1D interpolation to compute the error at the target distance
                        error_interp = interp1d(rho, errors, fill_value='extrapolate')
                        background_error = error_interp(rho_target)
                        return background, background_error
                    else:
                        return background
                else:
                    # Use the asymptotic value as the background
                    background = c
                    
                    if image_err is not None:
                        # Use the standard deviation of parameter c as the background error
                        background_error = np.sqrt(pcov[2,2])
                        return background, background_error
                    else:
                        return background
            else:
                raise ValueError("Unsupported fit function")
            
        except RuntimeError:
            raise RuntimeError("Failed to fit the radial profile")

    def from_scattering_map(map_path: Union[str, Path], factor_type: str = 'sum'):
        """
        factor_type: 'sum', 'mean', 'median'
        """
        with fits.open(map_path, mode='readonly') as hdul:
            scattering_map = hdul[0].data
            factor_med = hdul[0].header['FACT_MED']
            factor_ave = hdul[0].header['FACT_AVE']
            factor_sum = hdul[0].header['FACT_SUM']
            if factor_type == 'sum':
                factor = factor_sum
            elif factor_type == 'mean':
                factor = factor_ave
            elif factor_type == 'median':
                factor = factor_med
            else:
                raise ValueError(f"Unsupported factor type: {factor_type}")
        return scattering_map * factor

def perform_photometry(image: np.ndarray,
                      background: float,
                      regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
                      combine_regions: bool = False,
                      mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None,
                      image_err: Optional[np.ndarray] = None,
                      background_err: Optional[float] = None
                      ) -> Union[Union[float, List[float]], Tuple[Union[float, List[float]], Union[float, List[float]]]]:
    """Perform photometric calculation.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    background : float
        Background brightness value, background per pixel.
    regions : Union[PixelRegion, List[PixelRegion], np.ndarray]
        Photometric regions; can be region objects, a list of regions, or a boolean mask.
    mask : Union[PixelRegion, List[PixelRegion], np.ndarray], optional
        Regions to exclude; can be region objects, a list of regions, or a boolean mask.
    image_err : np.ndarray, optional
        Image error array.
    background_err : float, optional
        Background error value.

    Returns
    -------
    Union[Union[float, List[float]], Tuple[Union[float, List[float]], Union[float, List[float]]]]
        If error inputs are provided, return (total flux array, error array); otherwise return only the total flux array.
    """
    background_map = np.full_like(image, background)

    if image_err is not None or background_err is not None:
        if image_err is None:
            image_err = np.zeros_like(image)
        if background_err is None:
            background_err_map = np.zeros_like(image)
        else:
            background_err_map = np.full_like(image, background_err)
        net_flux_map, net_flux_error_map = ErrorPropagation.subtract(image, image_err, background_map, background_err_map)
        net_flux = RegionStatistics.sum(net_flux_map, regions, combine_regions=combine_regions, mask=mask)
        def get_error(err_map):
            return np.sqrt(np.sum(err_map*err_map))
        net_flux_error = RegionStatistics.calculate_stats(net_flux_error_map, regions, func=get_error, combine_regions=False, mask=mask)
        return net_flux, net_flux_error

    else:
        net_flux = RegionStatistics.sum(image-background_map, regions, combine_regions=combine_regions, mask=mask)
        return net_flux

def convert_to_absolute_mag(apparent_mag, r_h, delta, alpha=0, n=2, apparent_mag_err=None,
                            phase_correction_err_percent = None):
    """
    Convert to absolute magnitude R(1,1,0).
    
    Parameters:
    apparent_mag: float, apparent magnitude (AB or Vega)
    r_h: float, heliocentric distance (AU)
    delta: float, geocentric distance (AU)
    alpha: float, phase angle (degrees)
    
    Returns:
    float: absolute magnitude R(1,1,0)
    """
    if isinstance(apparent_mag, u.Quantity):
        apparent_mag = apparent_mag.value
    if isinstance(r_h, u.Quantity):
        r_h = r_h.to(u.au).value
    if isinstance(delta, u.Quantity):
        delta = delta.to(u.au).value
    if isinstance(alpha, u.Quantity):
        alpha = alpha.to(u.deg).value
    if apparent_mag_err is not None and isinstance(apparent_mag_err, u.Quantity):
        apparent_mag_err = apparent_mag_err.value
    
    # Calculate 5log(r_h * delta)
    distance_term = 5 * np.log10(delta) + 2.5 * n * np.log10(r_h)
    
    # Phase-angle correction
    # This uses a linear phase-angle correction as an example
    # More complex phase-angle correction functions may be needed in practice
    phase = phase_HalleyMarcus(alpha * u.deg)    # Simplified phase-angle correction
    phase_correction = 2.5 * np.log10(phase)
    
    # R(1,1,0) = m - 5log(r_h * delta) - φ(α)
    absolute_mag = apparent_mag - distance_term + phase_correction # TODO: check the phase correction
    if apparent_mag_err is not None or phase_correction_err_percent is not None:
        if phase_correction_err_percent is not None:
            phase_correction_err = phase_correction * phase_correction_err_percent / 100
        else:
            phase_correction_err = 0
        if apparent_mag_err is None:
            apparent_mag_err = 0
        _, absolute_mag_err = ErrorPropagation.add(apparent_mag, apparent_mag_err, phase_correction, phase_correction_err)
        return absolute_mag, absolute_mag_err
    else:
        return absolute_mag


class AfrhoCalculator:
    @staticmethod
    def basic_function(sfluxd, sun_sfluxd, rh, delta, aper):
        # delta in cm, rh in au, aper in cm
        # sfluxd and sun_sfluxd have the same unit
        # sun_sfluxd should be that at 1 au
        rh = rh.to(u.au).value
        delta = delta.to(u.cm).value
        aper = aper.to(u.cm).value
        afrho = (4 * delta**2 * rh**2 / aper) * (sfluxd / sun_sfluxd) * u.cm
        return afrho
        
    @staticmethod
    def basic_function_err(sfluxd: u.Quantity, sun_sfluxd: u.Quantity, rh: u.Quantity, delta: u.Quantity, aper: u.Quantity,
                           sfluxd_err: Union[u.Quantity, None], sun_sfluxd_err: Union[u.Quantity, None]):
        """
        Calculate the Afrho error.
        """
        rh = rh.to(u.au).value
        delta = delta.to(u.cm).value
        aper = aper.to(u.cm).value
        factor1 = 4 * delta**2 * rh**2 / aper
        if sfluxd_err is None:
            sfluxd_err = 0*sfluxd
        if sun_sfluxd_err is None:
            sun_sfluxd_err = 0*sun_sfluxd
        factor2, factor2_err = ErrorPropagation.divide(sfluxd, sfluxd_err, sun_sfluxd, sun_sfluxd_err)
        afrho = factor1 * factor2 * u.cm
        afrho_err = factor1 * factor2_err * u.cm
        return afrho, afrho_err
    
    @staticmethod
    def from_sfluxd_at_pivot(pivot_sfluxd: u.Quantity, bandpass: Union[str, SpectralElement],
                                sun: Union[SolarSpectrum, SourceSpectrum],
                                rh: u.Quantity, delta: u.Quantity, aper: u.Quantity,
                                from_phase: u.Quantity = None, to_phase: u.Quantity = 0 * u.deg,
                                pivot_sfluxd_err: Optional[u.Quantity] = None, sun_sfluxd_err: Optional[u.Quantity] = None):
        """
        Convert sfluxd to Afrho.
        """
        bandpass = format_bandpass(bandpass)
        pivot_wave = bandpass.pivot()
        sun_sfluxd = sun(pivot_wave, flux_unit=su.FLAM)
        pivot_sfluxd = convert_flux(pivot_wave, pivot_sfluxd, out_flux_unit=su.FLAM)

        if pivot_sfluxd_err is None and sun_sfluxd_err is None:
            afrho = AfrhoCalculator.basic_function(pivot_sfluxd, sun_sfluxd, rh, delta, aper)
            afrho = Afrho(afrho)
            if from_phase is not None:
                afrho = afrho.to_phase(to_phase, from_phase)
            return afrho
        
        else:
            afrho, afrho_err = AfrhoCalculator.basic_function_err(pivot_sfluxd, sun_sfluxd, rh, delta, aper,
                                                                  pivot_sfluxd_err, sun_sfluxd_err)
            afrho = Afrho(afrho)
            if from_phase is not None:
                afrho = afrho.to_phase(to_phase, from_phase)
            return afrho, afrho_err
    
    @staticmethod
    def from_flux(flux: u.Quantity, bandpass: Union[str, SpectralElement],
                      sun: Union[SolarSpectrum, SourceSpectrum],
                      rh: u.Quantity, delta: u.Quantity, aper: u.Quantity,
                      from_phase: u.Quantity = None, to_phase: u.Quantity = 0 * u.deg,
                      area: u.Quantity = None,
                      flux_err: Optional[u.Quantity] = None,
                      sun_flux_err: Optional[u.Quantity] = None):
        """
        Convert flux to Afrho.
        if flux's unit is erg/s/cm2 (flux), area needs to be None,
        if flux's unit is erg/s (power), area needs to be provided
            area is 1~cm2 for bandpass from effective area (bandpass has no area unit)
        """
        bandpass = format_bandpass(bandpass)
        sun_flux = calculate_flux(sun, bandpass, area=area)

        if flux_err is None and sun_flux_err is None:
            afrho = AfrhoCalculator.basic_function(flux, sun_flux, rh, delta, aper)
            afrho = Afrho(afrho)
            if from_phase is not None:
                afrho = afrho.to_phase(to_phase, from_phase)
            return afrho
        
        else:
            afrho, afrho_err = AfrhoCalculator.basic_function_err(flux, sun_flux, rh, delta, aper,
                                                                  flux_err, sun_flux_err)
            afrho = Afrho(afrho)
            if from_phase is not None:
                afrho = afrho.to_phase(to_phase, from_phase)
            return afrho, afrho_err
        
    #@staticmethod
    #def from_countrate():
    #    pass
