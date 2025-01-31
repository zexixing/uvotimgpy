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
from sbpy.calib import Sun


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
    #def from_model(model_name='k93models'):
    #    solar_spectrum = stsyn.grid_to_spec(model_name, 5777, 0, 4.44)
    #    return solar_spectrum
    def from_model():
        sun = Sun.from_default()
        return create_spectrum(sun.wave, sun.fluxd.to(su.FLAM))
    
    @staticmethod
    def from_file(file_path):
        pass

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
        numerator = calculate_flux(source_spectrum, bandpass)
        # 分母：∫ S(λ)dλ
        denominator = bandpass.equivwidth() # bandpass.equivwidth() = bandpass.integrate()
        return numerator / denominator

class FluxdConverter:
    @staticmethod
    def convert_fluxd_units(flux: u.Quantity, wavelength: u.Quantity, to_unit):
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
    def countrate_to_fluxd_pivot(count_rate, photflam, target_unit=su.FLAM, wavelength=None, bandpass=None, source_spectrum=None, area=None):
        """
        将计数率转换为fluxd (erg/s/cm²/Å)
        
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
        if source_spectrum is not None:
            bandpass = format_bandpass(bandpass)
            pivot_wave = bandpass.pivot()
            fluxd_at_pivot_in_theory = source_spectrum(pivot_wave, flux_unit=su.FLAM)
            count_rate_in_theory = calculate_count_rate(source_spectrum, bandpass, area=area)
            photflam = fluxd_at_pivot_in_theory / count_rate_in_theory
            photflam = photflam.value
        fluxd = count_rate * photflam * su.FLAM
        
        if target_unit != su.FLAM:
            if wavelength is None and bandpass is not None:
                bandpass = stsyn.band(bandpass)
                wavelength = bandpass.pivot()
            return convert_flux(wavelength, fluxd, target_unit)
            
        return fluxd

    @staticmethod
    def fluxd1_to_fluxd2(fluxd1: u.Quantity, wavelength1: u.Quantity, wavelength2: u.Quantity, source_spectrum: SourceSpectrum):
        """
        将wavelength1处的fluxd转换为wavelength2处的fluxd
        """
        fluxd1_in_theory = source_spectrum(wavelength1, flux_unit=fluxd1.unit)
        fluxd2_in_theory = source_spectrum(wavelength2, flux_unit=fluxd1.unit)
        factor = fluxd2_in_theory / fluxd1_in_theory
        return fluxd1 * factor
    
    @staticmethod
    def flux1_to_flux2_filter(flux1: u.Quantity, bandpass1: u.Quantity, bandpass2: u.Quantity, source_spectrum: SourceSpectrum):
        """
        将通过bandpass1得到的flux转换为通过bandpass2得到的flux
        """
        flux1_in_theory = calculate_flux(source_spectrum, bandpass1)
        flux2_in_theory = calculate_flux(source_spectrum, bandpass2)
        factor = flux2_in_theory / flux1_in_theory
        return flux1 * factor
    
    @staticmethod
    def fluxd_to_stmag(fluxd: u.Quantity, wavelength: u.Quantity):
        """
        将fluxd转换为ST星等
        
        Parameters
        ----------
        fluxd : Quantity
            输入的flux，必须包含单位
        wavelength : Quantity
            波长，必须包含单位
        """
            
        return convert_flux(wavelength, fluxd, u.STmag)

    @staticmethod
    def fluxd_to_abmag(fluxd: u.Quantity, wavelength: u.Quantity):
        """
        将fluxd转换为AB星等
        
        Parameters
        ----------
        fluxd : Quantity
            输入的flux，必须包含单位
        wavelength : Quantity
            波长，必须包含单位
        """
            
        return convert_flux(wavelength, fluxd, u.ABmag)

    @staticmethod
    def fluxd_to_vegamag(fluxd: u.Quantity, wavelength: u.Quantity):
        """
        将fluxd转换为VEGA星等
        
        Parameters
        ----------
        fluxd : Quantity
            输入的fluxd，必须包含单位
        wavelength : Quantity
            波长
        """
        vega = SourceSpectrum.from_vega()
        
        return convert_flux(wavelength, fluxd, su.VEGAMAG, vegaspec=vega)

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
        fluxd = convert_flux(wavelength, stmag, su.FLAM)
        # 再转换为AB星等
        return FluxdConverter.fluxd_to_abmag(fluxd, wavelength)

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
        fluxd = convert_flux(wavelength, stmag, su.FLAM)
        # 再转换为VEGA星等
        return FluxdConverter.fluxd_to_vegamag(fluxd, wavelength)

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
        fluxd = convert_flux(wavelength, abmag, su.FLAM)
        # 再转换为ST星等
        return FluxdConverter.fluxd_to_stmag(fluxd, wavelength)

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
        fluxd = convert_flux(wavelength, abmag, su.FLAM)
        # 再转换为VEGA星等
        return FluxdConverter.fluxd_to_vegamag(fluxd, wavelength)

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
        fluxd = convert_flux(wavelength, vegamag, su.FLAM, vegaspec=vega)
        # 再转换为ST星等
        return FluxdConverter.fluxd_to_stmag(fluxd, wavelength)

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
        fluxd = convert_flux(wavelength, vegamag, su.FLAM, vegaspec=vega)
        # 再转换为AB星等
        return FluxdConverter.fluxd_to_abmag(fluxd, wavelength)

class SpectrumScaler:
    @staticmethod
    def by_fluxd(spectrum_to_be_scaled, wavelength, fluxd_target):
        fluxd_to_be_scaled = spectrum_to_be_scaled(wavelength)
        
        fluxd_target = convert_flux(wavelength, fluxd_target, fluxd_to_be_scaled.unit)
        
        scale_factor = fluxd_target/fluxd_to_be_scaled
        scaled_spectrum = spectrum_to_be_scaled*scale_factor
        return scaled_spectrum
    
    @staticmethod
    def by_flux(spectrum_to_be_scaled, bandpass, flux_target):
        flux_to_be_scaled = calculate_flux(spectrum_to_be_scaled, bandpass)
        if isinstance(flux_target, u.Quantity) and flux_target.unit == su.FLAM:
            scale_factor = flux_target/flux_to_be_scaled
            scaled_spectrum = spectrum_to_be_scaled*scale_factor
            return scaled_spectrum
        else:
            raise ValueError("flux_target must be a Quantity with unit of FLAM")
