from typing import Union, List, Optional, Tuple
import numpy as np
from regions import PixelRegion, CirclePixelRegion, PixCoord
from scipy.optimize import curve_fit
from uvotimgpy.base.region import RegionStatistics, RegionConverter
from uvotimgpy.utils.image_operation import calc_radial_profile
from uvotimgpy.base.math_tools import ErrorPropagation
from scipy.interpolate import interp1d
import astropy.units as u
import synphot.units as su
from synphot import SourceSpectrum, Observation, SpectralElement
from synphot.units import convert_flux
import stsynphot as stsyn
from synphot.models import ConstFlux1D


class BackgroundEstimator:
    """背景估计类，提供两种静态方法计算背景"""
    
    @staticmethod
    def estimate_from_regions(image: np.ndarray,
                              regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
                              image_err: Optional[np.ndarray] = None
                              ) -> Union[float, Tuple[float, float]]:
        """使用区域统计方法估计背景

        Parameters
        ----------
        image : np.ndarray
            输入图像
        regions : Union[PixelRegion, List[PixelRegion], np.ndarray]
            用于估计背景的区域，可以是单个区域、区域列表或布尔掩模
        err_image : np.ndarray, optional
            误差图像，默认为None

        Returns
        -------
        Union[float, Tuple[float, float]]
            如果提供err_image，返回(背景值, 误差)；否则只返回背景值
        """
        if image_err is None:
            return RegionStatistics.median(image, regions, combine_regions=True)
        else:
            # 获取区域内的像素和对应误差
            masks = RegionConverter.to_bool_array_general(regions, combine_regions=True, shape=image.shape)
            mask = masks[0]
            valid_data = image[mask & ~np.isnan(image)]
            valid_errors = image_err[mask & ~np.isnan(image)]
            
            # 使用ErrorPropagation计算中位数及其误差
            value, error = ErrorPropagation.median((valid_data, valid_errors), axis=None, method='std')
            return value, error
        
    @staticmethod
    def estimate_from_profile(image: np.ndarray,
                            center: tuple,
                            fit_range: Tuple[float, float],
                            step: float = 1.0,
                            fit_func: str = 'power_law',
                            rho_target: Optional[float] = None,
                            err_image: Optional[np.ndarray] = None,
                            bad_pixel_mask: Optional[np.ndarray] = None
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
        err_image : np.ndarray, optional
            误差图像，默认为None

        Returns
        -------
        Union[float, Tuple[float, float]]
            如果提供err_image，返回(背景值, 误差)；否则只返回背景值
        """
        if err_image is None:
            rho, intensity = calc_radial_profile(image, center, step=step, bad_pixel_mask=bad_pixel_mask,
                                                 start=fit_range[0], end=fit_range[1], method='median', median_method='std')
            sigma = None
        else:
            rho, intensity, errors = calc_radial_profile(image, center, step=step, bad_pixel_mask=bad_pixel_mask,
                                                         start=fit_range[0], end=fit_range[1],
                                                         image_error=err_image, method='median', median_method='std')
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
                                     p0=[1, 1, 0],
                                     sigma=sigma if sigma is not None else None,
                                     absolute_sigma=True if sigma is not None else False)
                a, b, c = popt
                
                if rho_target is not None:
                    # 使用目标距离处的拟合值作为背景
                    background = fit_function(rho_target, a, b, c)
                    
                    if err_image is not None:
                        # 使用一维插值计算目标距离处的误差
                        error_interp = interp1d(rho, errors, fill_value='extrapolate')
                        background_error = error_interp(rho_target)
                        return background, background_error
                else:
                    # 使用渐近值作为背景
                    background = c
                    
                    if err_image is not None:
                        # 使用参数c的标准差作为背景误差
                        background_error = np.sqrt(pcov[2,2])
                        return background, background_error
                return background
            else:
                raise ValueError("Unsupported fit function")
            
        except RuntimeError:
            raise RuntimeError("Failed to fit the radial profile")
        

def perform_photometry(image: np.ndarray,
                      background: float,
                      regions: Union[PixelRegion, List[PixelRegion], np.ndarray],
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
    err_image : np.ndarray, optional
        图像误差阵列
    background_err : float, optional
        背景误差值

    Returns
    -------
    Union[Union[float, List[float]], Tuple[Union[float, List[float]], Union[float, List[float]]]]
        如果提供了误差输入，返回(总流量数组, 误差数组)；否则只返回总流量数组
    """
    background_map = np.full_like(image, background)

    if image_err is not None and background_err is not None:
        background_err_map = np.full_like(image, background_err)
        net_flux_map, net_flux_error_map = ErrorPropagation.subtract((image, image_err), (background_map, background_err_map))
        net_flux = RegionStatistics.sum(net_flux_map, regions, combine_regions=False, mask=mask)
        def get_error(err_map):
            return np.sqrt(np.sum(err_map*err_map))
        net_flux_error = RegionStatistics.calculate_stats(net_flux_error_map, regions, func=get_error, combine_regions=False, mask=mask)
        return net_flux, net_flux_error

    else:
        net_flux = RegionStatistics.sum(image-background_map, regions, combine_regions=False, mask=mask)
        return net_flux

from sbpy.activity import phase_HalleyMarcus
import astropy.units as u
def convert_to_absolute_mag(apparent_mag, r_h, delta, alpha):
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
    # 计算5log(r_h * delta)
    distance_term = 5 * np.log10(r_h * delta)
    
    # 相角改正
    # 这里使用线性相角改正作为示例
    # 实际使用中可能需要更复杂的相角改正函数
    phase_correction = phase_HalleyMarcus(alpha * u.deg)    # 简化的相角改正
    
    # R(1,1,0) = m - 5log(r_h * delta) - φ(α)
    absolute_mag = apparent_mag - distance_term - phase_correction # TODO: check the phase correction
    
    return absolute_mag

def flux_to_st(flux):
    """Flux转换到ST星等
    flux: erg/s/cm2/A
    """
    return -2.5 * np.log10(flux) - 21.10

def flux_to_ab(flux):
    """Flux转换到AB星等
    flux: erg/s/cm2/Hz
    """
    return -2.5 * np.log10(flux) - 48.6

def st_to_ab(st_mag, photplam):
    """ST星等转换到AB星等
    st_mag: ST星等
    """
    return st_mag - 5*np.log10(photplam) + 18.692

class FluxConverter:
    @staticmethod
    def convert_flux_units(flux: u.Quantity, wavelength: u.Quantity, to_unit):
        """
        在不同的flux单位之间转换
        
        Parameters
        ----------
        flux : Quantity
            输入的flux值，必须包含单位
        wavelength : Quantity
            波长，必须包含单位
        to_unit : Unit
            目标单位 (如 su.FLAM, su.FNU, su.PHOTLAM)
            PHOTLAM: photon/s/cm2/A
            PHOTNU: photon/s/cm2/Hz
            FLAM: erg/s/cm2/A
            FNU: erg/s/cm2/Hz
        """
        return convert_flux(wavelength, flux, to_unit)

    @staticmethod
    def countrate_to_flux(count_rate, photflam, target_unit=su.FLAM, wavelength=None, bandpass=None, source_spectrum=None):
        """
        将计数率转换为flux
        
        Parameters
        ----------
        count_rate : float
            每秒计数率
        photflam : float
            inverse sensitivity (erg/s/cm²/Å per count/s)
        target_unit : Unit
            目标flux单位
        wavelength : Quantity, optional
            波长，当目标单位不是FLAM时需要提供
        bandpass : str or SpectralElement, optional
            滤光片
        source_spectrum : SourceSpectrum, optional
            源光谱
        """
        if source_spectrum is None:
            correction = 1
        else:
            bandpass = stsyn.band(bandpass)
            obs = Observation(source_spectrum, bandpass)
            correction = 1.0 / obs.effstim(flux_unit=u.FLAM)
        flux_flam = count_rate * photflam * su.FLAM * correction
        
        if wavelength is None and bandpass is not None:
            bandpass = stsyn.band(bandpass)
            wavelength = bandpass.pivot()
        if target_unit != su.FLAM:
            return convert_flux(wavelength, flux_flam, target_unit)
            
        return flux_flam

    @staticmethod
    def flux_to_stmag(flux: u.Quantity, wavelength: u.Quantity):
        """
        将flux转换为ST星等
        
        Parameters
        ----------
        flux : Quantity
            输入的flux，必须包含单位
        wavelength : Quantity
            波长，必须包含单位
        """
            
        return convert_flux(wavelength, flux, u.STmag)

    @staticmethod
    def flux_to_abmag(flux: u.Quantity, wavelength: u.Quantity):
        """
        将flux转换为AB星等
        
        Parameters
        ----------
        flux : Quantity
            输入的flux，必须包含单位
        wavelength : Quantity
            波长，必须包含单位
        """
            
        return convert_flux(wavelength, flux, u.ABmag)

    @staticmethod
    def flux_to_vegamag(flux: u.Quantity, wavelength: u.Quantity):
        """
        将flux转换为VEGA星等
        
        Parameters
        ----------
        flux : Quantity
            输入的flux，必须包含单位
        wavelength : Quantity
            波长
        """
        vega = SourceSpectrum.from_vega()
        
        return convert_flux(wavelength, flux, su.VEGAMAG, vegaspec=vega)

    @staticmethod
    def stmag_to_abmag(stmag: Union[float, u.Quantity], wavelength: u.Quantity):
        """
        ST星等转换为AB星等
        
        Parameters
        ----------
        stmag : float
            ST星等
        wavelength : Quantity
            波长
        """
        if isinstance(stmag, float):
            stmag = stmag * u.STmag
        # 先转换为FLAM
        flux = convert_flux(wavelength, stmag, su.FLAM)
        # 再转换为AB星等
        return FluxConverter.flux_to_abmag(flux, wavelength)

    @staticmethod
    def stmag_to_vegamag(stmag: Union[float, u.Quantity], wavelength: u.Quantity):
        """
        ST星等转换为VEGA星等
        
        Parameters
        ----------
        stmag : float
            ST星等
        wavelength : Quantity
            波长
        """
        if isinstance(stmag, float):
            stmag = stmag * u.STmag
        # 先转换为FLAM
        flux = convert_flux(wavelength, stmag, su.FLAM)
        # 再转换为VEGA星等
        return FluxConverter.flux_to_vegamag(flux, wavelength)

    @staticmethod
    def abmag_to_stmag(abmag: Union[float, u.Quantity], wavelength: u.Quantity):
        """
        AB星等转换为ST星等
        
        Parameters
        ----------
        abmag : float
            AB星等
        wavelength : Quantity
            波长
        """
        if isinstance(abmag, float):
            abmag = abmag * u.ABmag
        # 先转换为FLAM
        flux = convert_flux(wavelength, abmag, su.FLAM)
        # 再转换为ST星等
        return FluxConverter.flux_to_stmag(flux, wavelength)

    @staticmethod
    def abmag_to_vegamag(abmag: Union[float, u.Quantity], wavelength: u.Quantity):
        """
        AB星等转换为VEGA星等
        
        Parameters
        ----------
        abmag : float
            AB星等
        wavelength : Quantity
        """
        if isinstance(abmag, float):
            abmag = abmag * u.ABmag
        
        # 先转换为FLAM
        flux = convert_flux(wavelength, abmag, su.FLAM)
        # 再转换为VEGA星等
        return FluxConverter.flux_to_vegamag(flux, wavelength)

    @staticmethod
    def vegamag_to_stmag(vegamag: Union[float, u.Quantity], wavelength: u.Quantity):
        """
        VEGA星等转换为ST星等
        
        Parameters
        ----------
        vegamag : float
            VEGA星等
        wavelength : Quantity
            波长
        """
        if isinstance(vegamag, float):
            vegamag = vegamag * u.VEGAMAG
        
        vega = SourceSpectrum.from_vega()
        
        # 先转换为FLAM
        flux = convert_flux(wavelength, vegamag, su.FLAM, vegaspec=vega)
        # 再转换为ST星等
        return FluxConverter.flux_to_stmag(flux, wavelength)

    @staticmethod
    def vegamag_to_abmag(vegamag: Union[float, u.Quantity], wavelength: u.Quantity):
        """
        VEGA星等转换为AB星等
        
        Parameters
        ----------
        vegamag : float
            VEGA星等
        wavelength : Quantity
            波长
        """
        if isinstance(vegamag, float):
            vegamag = vegamag * u.VEGAMAG
        
        vega = SourceSpectrum.from_vega()
        
        # 先转换为FLAM
        flux = convert_flux(wavelength, vegamag, su.FLAM, vegaspec=vega)
        # 再转换为AB星等
        return FluxConverter.flux_to_abmag(flux, wavelength)

