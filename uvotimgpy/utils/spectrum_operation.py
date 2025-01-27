from typing import Union, List, Optional, Tuple
from astropy import units as u
import numpy as np
from synphot import SourceSpectrum, SpectralElement, Empirical1D, Observation
from sbpy.spectroscopy import SpectralGradient, Reddening
from sbpy.units import hundred_nm
import stsynphot as stsyn
from synphot.units import convert_flux
from astropy import constants as const
import synphot.units as su


def obtain_reddening(reddening_percent, wave, wave0):
    gradient = SpectralGradient(reddening_percent * u.percent / hundred_nm, 
                               wave=wave,
                               wave0=wave0
                               )
    reddening = Reddening(gradient)
    return reddening

class ReddeningSpectrum:
    @staticmethod
    def create_wave_grid(wave_range=[5000, 6000]*u.AA, num_points=1000):
        """
        创建波长网格
        
        Parameters
        ----------
        wave_range : array-like Quantity
            保持输入单位
        num_points : int
            波长网格点数
            
        Returns
        -------
        wave_grid : astropy.Quantity
            波长网格，返回Quantity
        """
        wave_min, wave_max = wave_range
        wave_max = wave_max.to(wave_min.unit)
        return np.linspace(wave_min.value, wave_max.value, num_points) * wave_min.unit

    @staticmethod
    def linear_reddening(reddening_percent, wave=None, wave0=None, wave_grid_range=[5000, 6000]*u.AA, wave_grid=None):
        """
        线性红化模型
        
        Parameters
        ----------
        reddening_percent : float
            红化百分比 (%/100nm)
        wave : array-like Quantity, e.g., [5200, 5800] * u.AA
        wave0 : Quantity, optional
            归一化波长点
        wave_grid_range : array-like Quantity, optional
            用于计算的波长网格范围
        wave_grid : Quantity, optional
            直接提供的波长网格
            
        Returns
        -------
        SpectralElement
            红化传输函数
        """
        if wave_grid is None:
            wave_grid = ReddeningSpectrum.create_wave_grid(wave_grid_range)
            
        reddening = obtain_reddening(reddening_percent, wave, wave0)
        red_factors = reddening(wave_grid.to(u.um))
            
        return SpectralElement(Empirical1D, 
                             points=wave_grid, 
                             lookup_table=red_factors)

    @staticmethod
    def piecewise_reddening(reddening_percents, breakpoints=None, wave0=None, wave_grid_range=[5000, 6000]*u.AA, wave_grid=None):
        """
        分段线性红化

        Parameters
        ----------
        reddening_percents : array-like
            每段的红化百分比
        breakpoints : array-like Quantity
            分段点波长，e.g., [5200, 5500, 5800] * u.AA
            长度应该比reddening_percents多1，用于定义分段区间
        wave0 : Quantity, optional
            首段的归一化波长点
        wave_grid_range : array-like Quantity, optional
            用于计算的波长网格范围
        wave_grid : Quantity, optional
            直接提供的波长网格

        Returns
        -------
        SpectralElement
            红化传输函数
        """
        if wave_grid is None:
            wave_grid = ReddeningSpectrum.create_wave_grid(wave_grid_range)
            wave_unit = wave_grid.unit
        else:
            wave_unit = wave_grid_range.unit

        if breakpoints is None:
            raise ValueError("breakpoints parameter must be provided for piecewise reddening")

        if len(breakpoints) != len(reddening_percents) + 1:
            raise ValueError("Length of breakpoints must be equal to length of reddening_percents + 1")
        
        breakpoints = breakpoints.to(wave_unit)

        if wave0 is None:
            wave0 = breakpoints[0]  # 如果未指定wave0，使用第一个分段点

        red_factors = np.ones(len(wave_grid))

        # 处理小于第一个分段点的部分
        mask = (wave_grid < breakpoints[0])
        if np.any(mask):
            # 使用第一段的红化率延伸
            first_segment = ReddeningSpectrum.linear_reddening(
                reddening_percents[0],
                wave=[wave_grid[mask][0].value, breakpoints[0].value]*wave_unit,
                wave0=wave0,
                wave_grid=wave_grid[mask]
            )
            red_factors[mask] = first_segment(wave_grid[mask]).value

        # 处理各个分段
        current_wave0 = wave0  # 初始归一化点
        current_factor = 1.0   # 初始透过率

        for i in range(len(breakpoints)-1):
            mask = ((wave_grid >= breakpoints[i]) & (wave_grid < breakpoints[i+1]))
            if np.any(mask):
                segment = ReddeningSpectrum.linear_reddening(
                    reddening_percents[i],
                    wave=[breakpoints[i].value, breakpoints[i+1].value]*wave_unit,
                    wave0=current_wave0,
                    wave_grid=wave_grid[mask]
                )
                red_factors[mask] = current_factor * segment(wave_grid[mask]).value

                # 更新下一段的起始条件
                current_wave0 = breakpoints[i+1]
                current_factor = red_factors[mask][-1]

        # 处理大于最后一个分段点的部分
        mask = (wave_grid >= breakpoints[-1])
        if np.any(mask):
            last_segment = ReddeningSpectrum.linear_reddening(
                reddening_percents[-1],
                wave=[breakpoints[-1].value, wave_grid[mask][-1].value]*wave_unit,
                wave0=current_wave0,
                wave_grid=wave_grid[mask]
            )
            red_factors[mask] = current_factor * last_segment(wave_grid[mask]).value

        return SpectralElement(Empirical1D, 
                             points=wave_grid, 
                             lookup_table=red_factors)
    
    @staticmethod
    def custom_reddening():
        """
        自定义红化曲线
        """
        # placeholder
        pass

class SolarSpectrum:
    @staticmethod
    def from_model(model_name='k93models'):
        solar_spectrum = stsyn.grid_to_spec(model_name, 5777, 0, 4.44)
        return solar_spectrum
    
    @staticmethod
    def from_file(file_path):
        pass

def scale_spectrum(spectrum_to_be_scaled, wavelength, flux_target=None, spectrum_target=None):
    if flux_target is None and spectrum_target is not None:
        flux_target = spectrum_target(wavelength)
    flux_to_be_scaled = spectrum_to_be_scaled(wavelength)

    flux_target = convert_flux(wavelength, flux_target, flux_to_be_scaled.unit)
    
    scale_factor = flux_target/flux_to_be_scaled
    scaled_spectrum = spectrum_to_be_scaled*scale_factor
    return scaled_spectrum

def create_bandpass(wave: u.Quantity, thru: Union[u.Quantity, np.ndarray]) -> SpectralElement:
    '''
    Create SpectralElement from wavelength and throughput arrays
    '''
    # 确保thru是无量纲的
    if isinstance(thru, u.Quantity):
        thru = thru.value
        
    return SpectralElement(Empirical1D, points=wave, lookup_table=thru, fill_value=0)

def create_spectrum(wave: u.Quantity, fluxd: u.Quantity) -> SourceSpectrum:
    '''
    Create SourceSpectrum from wavelength and flux arrays
    '''
    return SourceSpectrum(Empirical1D, points=wave, lookup_table=fluxd, fill_value=0)

def format_bandpass(bandpass: Union[str, SpectralElement]):
    if isinstance(bandpass, str):
        bandpass = stsyn.band(bandpass)
    return bandpass

def calculate_flux(source_spectrum: SourceSpectrum, bandpass: Union[str, SpectralElement], area: u.Quantity = None, unit=su.FLAM):
    """
    计算通过滤光片的总flux (erg/s(/cm2))
    
    参数:
    source_spectrum: 源光谱对象 (SourceSpectrum)
    bandpass: 通带对象 (SpectralElement)
    
    返回:
    integrated_flux: 积分后的总flux
    """
    bandpass = format_bandpass(bandpass)

    sp_filtered = source_spectrum * bandpass
    flux = sp_filtered.integrate(flux_unit=unit)
    if area is not None:
        flux = flux * area
    return flux

def calculate_count_rate(source_spectrum: SourceSpectrum, bandpass: Union[str, SpectralElement], area: u.Quantity = None):
    """
    计算每秒光子数(count rate)
    
    参数:
    source_spectrum: 源光谱对象 (SourceSpectrum)
    bandpass: 通带对象 (SpectralElement)
    
    返回:
    count_rate: 每秒光子数
    """
    return calculate_flux(source_spectrum, bandpass, area, unit=su.PHOTLAM)


class TypicalWaveFluxd:
    @staticmethod
    def pivot_wave(bandpass: Union[str, SpectralElement]):
        """
        pivot wave只由bandpass决定，方便lambda与nu的转换
        PHOTFLAM就是为了将count rate转换为pivot wave处的fluxd

        需要一个函数，用来将pivot_wave处的fluxd转换为其他波长处的fluxd，这样可以对比不同的fluxd计算得到的afrho（或星等）
        """
        bandpass = format_bandpass(bandpass)
        return bandpass.pivot()
    
    @staticmethod
    def average_wave(bandpass: Union[str, SpectralElement]):
        '''
        https://synphot.readthedocs.io/en/latest/synphot/formulae.html
        Koornneef et al. 1986 (page 836).
        '''
        bandpass = format_bandpass(bandpass)
        return bandpass.avgwave()
    
    @staticmethod
    def effective_wave_photon_weighted(source_spectrum: SourceSpectrum, bandpass: Union[str, SpectralElement]):
        '''
        https://synphot.readthedocs.io/en/latest/synphot/formulae.html
        Tokunaga & Vacca (2005PASP..117..421T)
        λ_eff = ∫ F_λ P_λ λ² dλ / ∫ F_λ P_λ λ dλ
        '''
        bandpass = format_bandpass(bandpass)
        obs = Observation(source_spectrum, bandpass, force='taper')
        return obs.effective_wavelength()
    
    @staticmethod
    def effective_wave_energy_weighted(source_spectrum: SourceSpectrum, bandpass: Union[str, SpectralElement]):
        '''
        Tokunaga & Vacca (2005PASP..117..421T)
        λ_eff = ∫ F_λ P_λ λ dλ / ∫ F_λ P_λ dλ
        '''
        bandpass = format_bandpass(bandpass)
        h = const.h.to(u.erg * u.s)
        c = const.c.to(u.cm / u.s)
        factor_per_photon = h * c / u.ph
        numerator = calculate_count_rate(source_spectrum, bandpass) * factor_per_photon
        denominator = calculate_flux(source_spectrum, bandpass)
        return (numerator / denominator).to(u.AA)
    
    @staticmethod
    def isophotal_fluxd(source_spectrum: SourceSpectrum, bandpass: Union[str, SpectralElement]):
        ''' 
        Tokunaga & Vacca (2005PASP..117..421T)
        fluxd = ∫λF_λ(λ)S(λ)dλ / ∫λS(λ)dλ
        这样的fluxd的平谱得到的photons number（积分面积）与实际的fluxd通过整个波段的photons number（积分面积）是相等的
        '''
        bandpass = format_bandpass(bandpass)
        #wave = bandpass.waveset
        #modified_band = bandpass(wave) * wave
        #new_bandpass = SpectralElement(Empirical1D, points=wave, lookup_table=modified_band)

        # 分子：∫ λF_λ(λ)S(λ)dλ
        #numerator = (source_spectrum * new_bandpass).integrate()

        # 分母：∫ λS(λ)dλ
        #denominator = new_bandpass.integrate()

        obs = Observation(source_spectrum, bandpass, force='taper')
        return obs.effstim(flux_unit=su.FLAM)
    
    def isoflux_fluxd(source_spectrum: SourceSpectrum, bandpass: Union[str, SpectralElement]):
        '''
        fluxd = ∫ F_λ(λ)S(λ)dλ / ∫ S(λ)dλ
        这样的fluxd的平谱得到的总通量（积分面积）与实际的fluxd通过整个波段的总通量（积分面积）是相等的
        根据这个特性，用这个fluxd计算得到的afrho（或星等）与用总通量计算得到的afrho（或星等）是相等的
        '''
        bandpass = format_bandpass(bandpass)

        # 分子：∫ F_λ(λ)S(λ)dλ
        numerator = calculate_flux(source_spectrum, bandpass) # TODO: check units
        # 分母：∫ S(λ)dλ
        denominator = bandpass.equivwidth() # bandpass.equivwidth() = bandpass.integrate()
        return numerator / denominator
