from typing import Union, List, Optional, Tuple
import numpy as np
from regions import PixelRegion, CirclePixelRegion, PixCoord
from scipy.optimize import curve_fit
from uvotimgpy.base.region import RegionStatistics, RegionConverter
from uvotimgpy.utils.image_operation import calc_radial_profile
from uvotimgpy.base.math_tools import ErrorPropagation
from scipy.interpolate import interp1d


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
            value, error = ErrorPropagation.median((valid_data, valid_errors))
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
                                                 start=fit_range[0], end=fit_range[1],)
            sigma = None
        else:
            rho, intensity, errors = calc_radial_profile(image, center, step=step, bad_pixel_mask=bad_pixel_mask,
                                                         start=fit_range[0], end=fit_range[1],
                                                         image_error=err_image)
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

from synphot import SourceSpectrum, SpectralElement
from synphot.models import ConstFlux1D
from synphot import units

def convert_mags(ab_mag, bandpass='johnson_r'):
    """
    转换AB星等到其他系统
    
    参数:
    ab_mag: float, AB星等
    bandpass: str, 滤光片名称 ('r', 'g', 等)
    
    返回:
    dict: 不同星等系统的结果
    """
    # 创建AB系统中的源
    flux_density = 10 ** (-0.4 * (ab_mag + 48.6))
    sp = SourceSpectrum(ConstFlux1D, amplitude=flux_density * units.FLAM)
    
    # 加载滤光片
    bp = SpectralElement.from_filter(bandpass)
    
    # 计算不同系统的星等
    ab = -2.5 * np.log10(sp.integrate_filter(bp)) - 48.6
    vega = sp.to_vega(bp)
    
    return {'ab_mag': ab, 'vega_mag': vega}