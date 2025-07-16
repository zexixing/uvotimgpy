from typing import Union, List, Optional, Tuple
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

from uvotimgpy.base.region import RegionStatistics, RegionConverter
from uvotimgpy.utils.image_operation import calc_radial_profile
from uvotimgpy.base.math_tools import ErrorPropagation
from uvotimgpy.utils.spectrum_operation import format_bandpass, SolarSpectrum, calculate_flux, TypicalWaveSfluxd


class BackgroundEstimator:
    """背景估计类，提供两种静态方法计算背景"""
    
    @staticmethod
    def estimate_from_regions(image: np.ndarray,
                              regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
                              image_err: Optional[np.ndarray] = None,
                              method: str = 'median',
                              median_err_params: Optional[dict] = None) -> Union[float, Tuple[float, float]]:
        """使用区域统计方法估计背景

        Parameters
        ----------
        image : np.ndarray
            输入图像
        regions : Union[PixelRegion, List[PixelRegion], np.ndarray]
            用于估计背景的区域，可以是单个区域、区域列表或布尔掩模
        image_err : np.ndarray, optional
            误差图像，默认为None

        Returns
        -------
        Union[float, Tuple[float, float]]
            如果提供image_err，返回(背景值, 误差)；否则只返回背景值
        """
        #if image_err is None:
        #    return RegionStatistics.median(image, regions, combine_regions=True)
        #else:
        #    # 获取区域内的像素和对应误差
        #    masks = RegionConverter.to_bool_array_general(regions, combine_regions=True, shape=image.shape)
        #    mask = masks[0]
        #    valid_data = image[mask & ~np.isnan(image)]
        #    valid_errors = image_err[mask & ~np.isnan(image)]
        #    
        #    # 使用ErrorPropagation计算中位数及其误差
        #    value, error = ErrorPropagation.median((valid_data, valid_errors), axis=None, method='std')
        #    return value, error
        
        # 获取区域内的像素和对应误差
        masks = RegionConverter.to_bool_array_general(regions, combine_regions=True, shape=image.shape)
        mask = masks[0]
        valid_data = image[mask & ~np.isnan(image)]
        if image_err is not None:
            valid_errors = image_err[mask & ~np.isnan(image)]
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
    def estimate_from_profile(image: np.ndarray,
                            center: tuple,
                            fit_range: Tuple[float, float],
                            step: float = 1.0,
                            fit_func: str = 'power_law',
                            p0 = [1, 1, 0],
                            rho_target: Optional[float] = None,
                            image_err: Optional[np.ndarray] = None,
                            bad_pixel_mask: Optional[np.ndarray] = None,
                            median_err_params: Optional[dict] = None
                            ) -> Union[float, Tuple[float, float]]:
        """使用径向profile拟合方法估计背景

        Parameters
        ----------
        image : np.ndarray
            输入图像
        center : tuple
            中心点坐标 (col, row)
        fit_range : tuple
            用于拟合的距离范围 (start, end)
        step : float, optional
            径向profile的步长，默认1.0
        fit_func : str, optional
            拟合函数类型，可选 'linear' 或 'power_law'，默认 'linear'
        rho_target : float, optional
            目标距离处的背景值，如果为None则使用fit_range的终点
        image_err : np.ndarray, optional
            误差图像，默认为None

        Returns
        -------
        Union[float, Tuple[float, float]]
            如果提供image_err，返回(背景值, 误差)；否则只返回背景值
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
            sigma = errors  # 用于加权拟合
            
        # 定义拟合函数
        if fit_func == 'power_law':
            def fit_function(x, a, b, c):
                return a * x**(-b) + c
        else:
            raise ValueError("Unsupported fit function")
            
        # 执行拟合
        try: 
            if fit_func == 'power_law':
                popt, pcov = curve_fit(fit_function, rho, intensity,
                                       p0=p0,
                                       sigma=sigma if sigma is not None else None,
                                       absolute_sigma=True if sigma is not None else False)
                a, b, c = popt
                
                if rho_target is not None:
                    # 使用目标距离处的拟合值作为背景
                    background = fit_function(rho_target, a, b, c)
                    
                    if image_err is not None:
                        # 使用一维插值计算目标距离处的误差
                        error_interp = interp1d(rho, errors, fill_value='extrapolate')
                        background_error = error_interp(rho_target)
                        return background, background_error
                    else:
                        return background
                else:
                    # 使用渐近值作为背景
                    background = c
                    
                    if image_err is not None:
                        # 使用参数c的标准差作为背景误差
                        background_error = np.sqrt(pcov[2,2])
                        return background, background_error
                    else:
                        return background
            else:
                raise ValueError("Unsupported fit function")
            
        except RuntimeError:
            raise RuntimeError("Failed to fit the radial profile")
        

def perform_photometry(image: np.ndarray,
                      background: float,
                      regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
                      combine_regions: bool = False,
                      mask: Optional[Union[PixelRegion, List[PixelRegion], np.ndarray]] = None,
                      image_err: Optional[np.ndarray] = None,
                      background_err: Optional[float] = None
                      ) -> Union[Union[float, List[float]], Tuple[Union[float, List[float]], Union[float, List[float]]]]:
    """执行测光计算

    Parameters
    ----------
    image : np.ndarray
        输入图像
    background : float
        背景亮度值
    regions : Union[PixelRegion, List[PixelRegion], np.ndarray]
        测光区域，可以是区域对象、区域列表或布尔掩模
    mask : Union[PixelRegion, List[PixelRegion], np.ndarray], optional
        需要排除的区域，可以是区域对象、区域列表或布尔掩模
    image_err : np.ndarray, optional
        图像误差阵列
    background_err : float, optional
        背景误差值

    Returns
    -------
    Union[Union[float, List[float]], Tuple[Union[float, List[float]], Union[float, List[float]]]]
        如果提供了误差输入，返回(总流量数组, 误差数组)；否则只返回总流量数组
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
    转换为绝对星等R(1,1,0)
    
    参数:
    apparent_mag: float, 视星等(AB或Vega)
    r_h: float, 日距(AU)
    delta: float, 地距(AU)
    alpha: float, 相角(度)
    
    返回:
    float: 绝对星等R(1,1,0)
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
    
    # 计算5log(r_h * delta)
    distance_term = 5 * np.log10(delta) + 2.5 * n * np.log10(r_h)
    
    # 相角改正
    # 这里使用线性相角改正作为示例
    # 实际使用中可能需要更复杂的相角改正函数
    phase = phase_HalleyMarcus(alpha * u.deg)    # 简化的相角改正
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
        计算Afrho的误差
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
        sfluxd转换为Afrho
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
        flux转换为Afrho
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