from typing import Union, List, Optional, Tuple
from astropy import units as u
from astropy.io import fits
import os
import warnings
import numpy as np
from synphot import SourceSpectrum, SpectralElement, Empirical1D, Observation, BaseUnitlessSpectrum
from sbpy.units import hundred_nm
import stsynphot as stsyn
from synphot.units import convert_flux
from astropy import constants as const
import synphot.units as su
from sbpy.calib import Sun
from synphot.specio import read_fits_spec
from stsynphot.config import conf
from uvotimgpy.config import paths
from uvotimgpy.base.math_tools import ErrorPropagation

def create_spectrum(wave: u.Quantity, spec: u.Quantity, ifunit=True, **kwargs) -> Union[SourceSpectrum, BaseUnitlessSpectrum]:
    '''
    Create SourceSpectrum from wavelength and flux arrays
    '''
    if ifunit:
        fill_value = kwargs.pop('fill_value', 0)
        return SourceSpectrum(Empirical1D, points=wave, lookup_table=spec, fill_value=fill_value, **kwargs)
    else:
        fill_value = kwargs.pop('fill_value', np.nan)
        return BaseUnitlessSpectrum(Empirical1D, points=wave, lookup_table=spec, fill_value=fill_value, **kwargs)

class SolarSpectrum:
    @staticmethod
    #def from_model(model_name='k93models'):
    #    solar_spectrum = stsyn.grid_to_spec(model_name, 5777, 0, 4.44)
    #    return solar_spectrum
    def from_model():
        sun = Sun.from_default()
        return create_spectrum(sun.wave, sun.fluxd.to(su.FLAM), True)
    
    @staticmethod
    def from_colina96():
        colina96_path = paths.get_subpath(paths.package_uvotimgpy, 'auxil', 'sun_1A.txt')
        colina96 = np.loadtxt(colina96_path)
        return create_spectrum(colina96[:, 0] * u.AA, colina96[:, 1] * su.FLAM, True)
    
def read_calspec(file_name, file_path=None):
    if file_path is None:
        conf_path = conf.rootdir
        calspec_path = os.path.join(conf_path, 'calspec', file_name+'.fits')
    else:
        calspec_path = os.path.join(file_path, file_name+'.fits')
    hdul = fits.open(calspec_path)
    wave = hdul[1].data['WAVELENGTH']
    sfluxd = hdul[1].data['FLUX']
    hdul.close()
    diff = np.diff(wave)
    monotonic = np.all(diff > 0) or np.all(diff < 0)
    if monotonic:
        sp = create_spectrum(wave, sfluxd, True)
        return sp
    else:
        sort_idx = np.argsort(wave)
        wave_sorted = wave[sort_idx]
        sfluxd_sorted = sfluxd[sort_idx]
        unique_idx = np.unique(wave_sorted, return_index=True)[1]
        wave_unique = wave_sorted[unique_idx] * u.AA
        sfluxd_unique = sfluxd_sorted[unique_idx] * su.FLAM
        return create_spectrum(wave_unique, sfluxd_unique, True)

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
    erg/s (power): area is not None; 
        area is 1~cm2 for bandpass from effective area (bandpass has no area unit)
    erg/s/cm2 (flux): area is None
    
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
    def linear_reddening(reddening_percent, wave=None, wave0=None, wave_grid_range=[1000, 10000]*u.AA, num_points=3000, wave_grid=None, 
                         reddening_defination='r', bp1=None, bp2=None, return_a_b=False):
        """
        线性红化模型
        
        Parameters
        ----------
        reddening_percent : float
            红化百分比 (%/100nm)
        wave : array-like Quantity, e.g., [5200, 5800] * u.AA
        wave0 : Quantity, optional
            center of wave
        wave_grid_range : array-like Quantity, optional
            用于计算的波长网格范围
        wave_grid : Quantity, optional
            直接提供的波长网格
        reddening_defination : str, optional
            'r' for reddening defined from two points on the reflectance spectrum that is created from flux density (erg/s/cm2/A),
            'flux' for reddening defined from flux ratio (erg/s/cm2 or erg/s),
            'cr' for reddening defined from countrate ratio.
            default is 'r'.
            
        Returns
        -------
        SpectralElement
            红化传输函数
        """
        if wave_grid is None:
            wave_grid = ReddeningSpectrum.create_wave_grid(wave_grid_range, num_points)
            
        #reddening = obtain_reddening(reddening_percent, wave, wave0)
        #red_factors = reddening(wave_grid)
        wave_grid_unit = wave_grid.unit
        if reddening_percent == 0:
            red_factors = np.ones(len(wave_grid))
        else:
            # r(lambda) = a * (lambda + b)
            # a can be set as arbitrary value, e.g., 1
            a = 1 / wave_grid_unit
            if reddening_defination == 'r':
                if wave0 is None:
                    wave0 = (wave[0].to(wave_grid_unit) + wave[1].to(wave_grid_unit))/2
                b = (1000*u.AA).to(wave_grid_unit) * 100*u.percent/(reddening_percent*u.percent) - wave0
            else:
                if bp1 is None or bp2 is None:
                    raise ValueError("bp1 and bp2 must be provided for flux reddening defination")
                wave1 = TypicalWaveSfluxd.average_wave(bp1)
                wave2 = TypicalWaveSfluxd.average_wave(bp2)
                sun = SolarSpectrum.from_model()
                if reddening_defination == 'flux':
                    # wave_weighted_by_solar_flux: int(wave*f_sun*bp*dwave) / int(f_sun*bp*dwave)
                    wave1_weighted_by_solar_flux = TypicalWaveSfluxd.effective_wave_energy_weighted(sun, bp1)
                    wave2_weighted_by_solar_flux = TypicalWaveSfluxd.effective_wave_energy_weighted(sun, bp2)
                    # set alias
                    t1 = wave1_weighted_by_solar_flux
                    t2 = wave2_weighted_by_solar_flux
                    b = ((t2 - t1) / (wave2 - wave1)) * ((100*u.percent) / (reddening_percent*u.percent)) * (1000*u.AA) - (t2+t1)/2
                elif reddening_defination == 'cr':
                    # wave_weighted_by_solar_countrate: int(wave*wave*f_sun*bp*dwave) / int(wave*f_sun*bp*dwave)
                    wave1_weighted_by_solar_countrate = TypicalWaveSfluxd.effective_wave_photon_weighted(sun, bp1)
                    wave2_weighted_by_solar_countrate = TypicalWaveSfluxd.effective_wave_photon_weighted(sun, bp2)
                    # set alias
                    t1 = wave1_weighted_by_solar_countrate
                    t2 = wave2_weighted_by_solar_countrate
                    b = ((t2 - t1) / (wave2 - wave1)) * ((100*u.percent) / (reddening_percent*u.percent)) * (1000*u.AA) - (t2+t1)/2
            
            red_factors = a * (wave_grid + b)
        if return_a_b:
            return a, b
        else:
            #return SpectralElement(Empirical1D, 
            #                      points=wave_grid, 
            #                      lookup_table=red_factors,
            #                      keep_neg = True)
            return create_spectrum(wave_grid, red_factors, ifunit=False, keep_neg=True)
        

    @staticmethod
    def piecewise_reddening(reddening_percents, breakpoints=None, wave_grid_range=[1000, 10000]*u.AA, num_points=3000, wave_grid=None,
                            reddening_defination='r', bp_list=None):
        """
        分段线性红化

        Parameters
        ----------
        reddening_percents : array-like
            每段的红化百分比
        breakpoints : array-like Quantity
            分段点波长，e.g., [5200, 5500, 5800] * u.AA
            长度应该比reddening_percents多1，用于定义分段区间
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
            if wave_grid_range is None:
                wave_grid_range = [breakpoints[0].value, breakpoints[-1].value]*u.AA
            wave_grid = ReddeningSpectrum.create_wave_grid(wave_grid_range, num_points)
            wave_unit = wave_grid.unit
        else:
            wave_unit = wave_grid_range.unit

        if breakpoints is None:
            raise ValueError("breakpoints parameter must be provided for piecewise reddening")

        if len(breakpoints) != len(reddening_percents) + 1:
            raise ValueError("Length of breakpoints must be equal to length of reddening_percents + 1")
        
        breakpoints = breakpoints.to(wave_unit)

        insert_positions = np.searchsorted(wave_grid.value, breakpoints.value)
        wave_grid = np.insert(wave_grid.value, insert_positions, breakpoints.value)
        wave_grid = np.unique(wave_grid) * wave_unit

        red_factors = np.ones(len(wave_grid))


        # 处理第一个分段
        if reddening_defination != 'r':
            warnings.warn("reddening_defination is based on countrate or flux, the breakpoints should be average wavelengths of the filters")
        mask = (wave_grid <= breakpoints[1])
        if np.any(mask):
            segment = ReddeningSpectrum.linear_reddening(
                reddening_percents[0],
                wave=[breakpoints[0].value, breakpoints[1].value]*wave_unit,
                wave_grid=wave_grid[mask],
                reddening_defination=reddening_defination,
                bp1=bp_list[0],
                bp2=bp_list[1]
            )
            start = segment(breakpoints[0]).value
            red_factors[mask] = segment(wave_grid[mask]).value
            end = segment(breakpoints[1]).value

        # 处理中间各个分段
        for i in range(len(breakpoints)-1)[1:-1]:
            mask = ((wave_grid >= breakpoints[i]) & (wave_grid <= breakpoints[i+1]))
            if np.any(mask):
                segment = ReddeningSpectrum.linear_reddening(
                    reddening_percents[i],
                    wave=[breakpoints[i].value, breakpoints[i+1].value]*wave_unit,
                    wave_grid=wave_grid[mask],
                    reddening_defination=reddening_defination,
                    bp1=bp_list[i],
                    bp2=bp_list[i+1]
                )
                start = segment(breakpoints[i]).value
                current_factor = end / start
                red_factors[mask] = current_factor * segment(wave_grid[mask]).value

                # 更新下一段的起始条件
                end = current_factor * segment(breakpoints[i+1]).value

        # 处理最后一个分段点的部分
        mask = (wave_grid >= breakpoints[-2])
        if np.any(mask):
            last_segment = ReddeningSpectrum.linear_reddening(
                reddening_percents[-1],
                wave=[breakpoints[-2].value, breakpoints[-1].value]*wave_unit,
                wave_grid=wave_grid[mask],
                reddening_defination=reddening_defination,
                bp1=bp_list[-2],
                bp2=bp_list[-1]
            )
            start = last_segment(breakpoints[-2]).value
            current_factor = end / start
            red_factors[mask] = current_factor * last_segment(wave_grid[mask]).value

        #return SpectralElement(Empirical1D, 
        #                     points=wave_grid, 
        #                     lookup_table=red_factors,
        #                     keep_neg = True)
        return create_spectrum(wave_grid, red_factors, ifunit=False, keep_neg=True)
    
    @staticmethod
    def custom_reddening(wave_grid, reddening_percents):
        """
        wave_grid: array-like Quantity
            波长网格
        reddening_percents: array
            红化百分比
        """
        # TODO: The current codes are wrong, to be updated
        if len(wave_grid.value) != len(reddening_percents):
            raise ValueError("Length of wave_grid must be equal to length of reddening_percents")
        
        red_factors = (1000*u.AA * 100*u.percent/(reddening_percents*u.percent))* (1/u.AA)
        #return SpectralElement(Empirical1D, 
        #                       points=wave_grid, 
        #                       lookup_table=red_factors,
        #                       keep_neg = True)
        return create_spectrum(wave_grid, red_factors, ifunit=False, keep_neg=True)

class ReddeningCalculator:
    """
    There are mainly 3 types of reddening definitions:
    1. defined with reflectance at two wavelengths
    2. defined with flux observed through two filters
    3. defined with countrate observed through two filters

    the methods below can be classified into these types:
    Type 1: from_reflectance_spectrum, from_sfluxd
    Type 2: from_flux, from_flux_source_spectrum, from_color (when the color is derived with magnitudes, which are measured with fluxes through filters)
    Type 3: from_countrate, from_countrate_source_spectrum
    from_afrho may be Type 2 or Type 3, depending on the definition of afrho
    """
    @staticmethod
    def basic_function(r1: Union[u.Quantity, float], r2: Union[u.Quantity, float], wave1: u.Quantity, wave2: u.Quantity,):
        wave1 = wave1.to(u.AA)
        wave2 = wave2.to(u.AA)
        reddening = ((r2 - r1) / (wave2 - wave1)) / ((r2 + r1) / 2) * 1000 * u.AA * 100 * u.percent
        return reddening
        
    @staticmethod
    def basic_function_err(r1: Union[u.Quantity, float], r2: Union[u.Quantity, float], wave1: u.Quantity, wave2: u.Quantity,
                           r1_err: Union[u.Quantity, None], r2_err: Union[u.Quantity, None]):
        if r1_err is None:
            r1_err = 0*r1
        if r2_err is None:
            r2_err = 0*r2
        wave1 = wave1.to(u.AA)
        wave2 = wave2.to(u.AA)
        factor1 = 2*(2/np.abs(wave2-wave1))*1000*u.AA*100*u.percent
        factor2 = 1/(r1+r2)**2
        factor3 = np.sqrt((r2*r1_err)**2 + (r1*r2_err)**2)
        reddening_err = factor1*factor2*factor3
        reddening = ((r2 - r1) / (wave2 - wave1)) / ((r2 + r1) / 2) * 1000 * u.AA * 100 * u.percent
        return reddening, reddening_err
    
    # reddening defined with r
    @staticmethod
    def from_reflectance_spectrum(reflectance_spectrum: BaseUnitlessSpectrum, wave1: u.Quantity, wave2: u.Quantity,
                                  reflectance_spectrum_err: Optional[BaseUnitlessSpectrum] = None):
        r1 = reflectance_spectrum(wave1)
        r2 = reflectance_spectrum(wave2)
        if reflectance_spectrum_err is None:
            reddening = ReddeningCalculator.basic_function(r1, r2, wave1, wave2)
            return reddening
        else:
            r1_err = reflectance_spectrum_err(wave1, flux_unit=su.FLAM)
            r2_err = reflectance_spectrum_err(wave2, flux_unit=su.FLAM)
            reddening, reddening_err = ReddeningCalculator.basic_function_err(r1, r2, wave1, wave2,
                                                                              r1_err, r2_err)
            return reddening, reddening_err
        
    @staticmethod
    def from_sfluxd(sfluxd1: u.Quantity, sfluxd2: u.Quantity, wave1: u.Quantity, wave2: u.Quantity, solar_spectrum: Union[SolarSpectrum, SourceSpectrum],
                   sfluxd_err1: Optional[u.Quantity] = None, sfluxd_err2: Optional[u.Quantity] = None,
                   solar_spectrum_err: Optional[SourceSpectrum] = None):
        sfluxd1_sun = solar_spectrum(wave1, flux_unit=sfluxd1.unit)
        sfluxd2_sun = solar_spectrum(wave2, flux_unit=sfluxd1.unit)
        if sfluxd_err1 is None and sfluxd_err2 is None and solar_spectrum_err is None:
            r1 = sfluxd1/sfluxd1_sun
            r2 = sfluxd2/sfluxd2_sun
            reddening = ReddeningCalculator.basic_function(r1, r2, wave1, wave2)
            return reddening
        else:
            if solar_spectrum_err is not None:
                sfluxd1_sun_err = solar_spectrum_err(wave1, flux_unit=sfluxd1.unit)
                sfluxd2_sun_err = solar_spectrum_err(wave2, flux_unit=sfluxd1.unit)
            else:
                sfluxd1_sun_err = 0*sfluxd1_sun
                sfluxd2_sun_err = 0*sfluxd2_sun
            if sfluxd_err1 is None: sfluxd_err1 = 0*sfluxd1
            if sfluxd_err2 is None: sfluxd_err2 = 0*sfluxd2
            r1, r1_err = ErrorPropagation.divide(sfluxd1, sfluxd_err1, sfluxd1_sun, sfluxd1_sun_err)
            r2, r2_err = ErrorPropagation.divide(sfluxd2, sfluxd_err2, sfluxd2_sun, sfluxd2_sun_err)
            if isinstance(r1, u.Quantity):
                r1 = r1.value
                r2 = r2.value
            if isinstance(r1_err, u.Quantity):
                r1_err = r1_err.value
                r2_err = r2_err.value
            reddening, reddening_err = ReddeningCalculator.basic_function_err(r1, r2, wave1, wave2,
                                                                              r1_err, r2_err)
            return reddening, reddening_err
        
    # reddening defined with flux
    @staticmethod
    def from_flux(flux1: u.Quantity, flux2: u.Quantity, solar_spectrum: Union[SolarSpectrum, SourceSpectrum],
                  bandpass1: Union[str, SpectralElement], bandpass2: Union[str, SpectralElement],
                  flux1_err: Optional[u.Quantity] = None,
                  flux2_err: Optional[u.Quantity] = None,
                  solar_spectrum_err: Optional[SourceSpectrum] = None,
                  area: Optional[u.Quantity] = None):
        bandpass1 = format_bandpass(bandpass1)
        bandpass2 = format_bandpass(bandpass2)
        wave1 = TypicalWaveSfluxd.average_wave(bandpass1) # TODO: to be checked
        wave2 = TypicalWaveSfluxd.average_wave(bandpass2) # TODO: to be checked
        if flux1_err is None and flux2_err is None and solar_spectrum_err is None:
            solar_flux1 = calculate_flux(solar_spectrum, bandpass1, area)
            r1 = flux1/solar_flux1
            solar_flux2 = calculate_flux(solar_spectrum, bandpass2, area)
            r2 = flux2/solar_flux2
            reddening = ReddeningCalculator.basic_function(r1, r2, wave1, wave2)
            return reddening
        else:
            solar_flux1 = calculate_flux(solar_spectrum, bandpass1, area)
            solar_flux2 = calculate_flux(solar_spectrum, bandpass2, area)
            if flux1_err is None:
                flux1_err = 0*flux1
            if flux2_err is None:
                flux2_err = 0*flux2
            if solar_spectrum_err is not None:
                solar_flux1_err = calculate_flux(solar_spectrum_err, bandpass1, area)
                solar_flux2_err = calculate_flux(solar_spectrum_err, bandpass2, area)
            else:
                solar_flux1_err = 0*solar_flux1
                solar_flux2_err = 0*solar_flux2
            r1, r1_err = ErrorPropagation.divide(flux1, flux1_err, solar_flux1, solar_flux1_err)
            r2, r2_err = ErrorPropagation.divide(flux2, flux2_err, solar_flux2, solar_flux2_err)
             
            if isinstance(r1, u.Quantity):
                r1 = r1.value
                r2 = r2.value
            if isinstance(r1_err, u.Quantity):
                r1_err = r1_err.value
                r2_err = r2_err.value
            reddening, reddening_err = ReddeningCalculator.basic_function_err(r1, r2, wave1, wave2,
                                                                              r1_err, r2_err) 
            return reddening, reddening_err
        
    @staticmethod
    def from_flux_source_spectrum(source_spectrum: SourceSpectrum, solar_spectrum: Union[SolarSpectrum, SourceSpectrum],
                                  bandpass1: Union[str, SpectralElement], bandpass2: Union[str, SpectralElement],
                                  source_spectrum_err: Optional[SourceSpectrum] = None,
                                  solar_spectrum_err: Optional[SourceSpectrum] = None,
                                  ):
        bandpass1 = format_bandpass(bandpass1)
        bandpass2 = format_bandpass(bandpass2)
        #wave1 = TypicalWaveSfluxd.average_wave(bandpass1) # TODO: to be checked
        #wave2 = TypicalWaveSfluxd.average_wave(bandpass2) # TODO: to be checked

        flux1 = calculate_flux(source_spectrum, bandpass1, area=None)
        flux2 = calculate_flux(source_spectrum, bandpass2, area=None)
        flux1_err = calculate_flux(source_spectrum_err, bandpass1, area=None)
        flux2_err = calculate_flux(source_spectrum_err, bandpass2, area=None)
        return ReddeningCalculator.from_flux(flux1, flux2, solar_spectrum, bandpass1, bandpass2, flux1_err, flux2_err, solar_spectrum_err, area=None)

        #if source_spectrum_err is None and solar_spectrum_err is None:
        #    r1 = calculate_flux(source_spectrum, bandpass1)/calculate_flux(solar_spectrum, bandpass1)
        #    r2 = calculate_flux(source_spectrum, bandpass2)/calculate_flux(solar_spectrum, bandpass2)
        #    reddening = ReddeningCalculator.basic_function(r1, r2, wave1, wave2)
        #    return reddening
        #else:
        #    flux1 = calculate_flux(source_spectrum, bandpass1)
        #    solar_flux1 = calculate_flux(solar_spectrum, bandpass1)
        #    flux2 = calculate_flux(source_spectrum, bandpass2)
        #    solar_flux2 = calculate_flux(solar_spectrum, bandpass2)
        #    if source_spectrum_err is not None:
        #        flux1_err = calculate_flux(source_spectrum_err, bandpass1)
        #        flux2_err = calculate_flux(source_spectrum_err, bandpass2)
        #    else:
        #        flux1_err = 0*flux1
        #        flux2_err = 0*flux2
        #    if solar_spectrum_err is not None:
        #        solar_flux1_err = calculate_flux(solar_spectrum_err, bandpass1)
        #        solar_flux2_err = calculate_flux(solar_spectrum_err, bandpass2)
        #    else:
        #        solar_flux1_err = 0*solar_flux1
        #        solar_flux2_err = 0*solar_flux2
        #    
        #    r1, r1_err = ErrorPropagation.divide(flux1, flux1_err, solar_flux1, solar_flux1_err)
        #    r2, r2_err = ErrorPropagation.divide(flux2, flux2_err, solar_flux2, solar_flux2_err)
        #     
        #    # TODO: the r1 here is float instead of u.Quantity
        #    reddening, reddening_err = ReddeningCalculator.basic_function_err(r1, r2, wave1, wave2,
        #                                                                      r1_err, r2_err) 
        #    return reddening, reddening_err

    # reddening defined with flux: flux -> mag -> color
    @staticmethod
    def from_color(source_color: Union[u.Quantity, float], solar_color: Union[u.Quantity, float], wave1: u.Quantity, wave2: u.Quantity,
                   source_color_err: Optional[Union[u.Quantity, float]] = None, solar_color_err: Optional[Union[u.Quantity, float]] = None):
        if isinstance(source_color, u.Quantity):
            source_color = source_color.value
        if isinstance(solar_color, u.Quantity):
            solar_color = solar_color.value
        r1 = np.power(10, -0.4*solar_color)
        r2 = np.power(10, -0.4*source_color)
        if source_color_err is None and solar_color_err is None:
            reddening = ReddeningCalculator.basic_function(r1, r2, wave1, wave2)
            return reddening
        else:
            if source_color_err is None:
                source_color_err = 0
            elif isinstance(source_color_err, u.Quantity):
                source_color_err = source_color_err.value
            if solar_color_err is None:
                solar_color_err = 0
            elif isinstance(solar_color_err, u.Quantity):
                solar_color_err = solar_color_err.value
            r1_err = 0.4*np.log(10)*np.power(10, -0.4*solar_color)*solar_color_err
            r2_err = 0.4*np.log(10)*np.power(10, -0.4*source_color)*source_color_err
            reddening, reddening_err = ReddeningCalculator.basic_function_err(r1, r2, wave1, wave2,
                                                                              r1_err, r2_err)
            return reddening, reddening_err
        
    # reddening defined with flux
    @staticmethod
    def from_countrate(countrate1: float, countrate2: float, solar_spectrum: Union[SolarSpectrum, SourceSpectrum],
                       bandpass1: Union[str, SpectralElement], bandpass2: Union[str, SpectralElement], area: u.Quantity,
                       countrate1_err: Optional[float] = None,
                       countrate2_err: Optional[float] = None,
                       solar_spectrum_err: Optional[SourceSpectrum] = None,):
        bandpass1 = format_bandpass(bandpass1)
        bandpass2 = format_bandpass(bandpass2)
        wave1 = TypicalWaveSfluxd.average_wave(bandpass1) # TODO: to be checked
        wave2 = TypicalWaveSfluxd.average_wave(bandpass2) # TODO: to be checked
        if countrate1_err is None and countrate2_err is None and solar_spectrum_err is None:
            solar_countrate1 = calculate_count_rate(solar_spectrum, bandpass1, area).value
            r1 = countrate1/solar_countrate1
            solar_countrate2 = calculate_count_rate(solar_spectrum, bandpass2, area).value
            r2 = countrate2/solar_countrate2
            reddening = ReddeningCalculator.basic_function(r1, r2, wave1, wave2)
            return reddening
        else:
            solar_countrate1 = calculate_count_rate(solar_spectrum, bandpass1, area).value
            solar_countrate2 = calculate_count_rate(solar_spectrum, bandpass2, area).value
            if countrate1_err is None:
                countrate1_err = 0*countrate1
            if countrate2_err is None:
                countrate2_err = 0*countrate2
            if solar_spectrum_err is not None:
                solar_countrate1_err = calculate_count_rate(solar_spectrum_err, bandpass1, area).value
                solar_countrate2_err = calculate_count_rate(solar_spectrum_err, bandpass2, area).value
            else:
                solar_countrate1_err = 0*solar_countrate1
                solar_countrate2_err = 0*solar_countrate2
            r1, r1_err = ErrorPropagation.divide(countrate1, countrate1_err, solar_countrate1, solar_countrate1_err)
            r2, r2_err = ErrorPropagation.divide(countrate2, countrate2_err, solar_countrate2, solar_countrate2_err)
             
            reddening, reddening_err = ReddeningCalculator.basic_function_err(r1, r2, wave1, wave2,
                                                                              r1_err, r2_err) 
            return reddening, reddening_err
        
    @staticmethod
    def from_countrate_source_spectrum(source_spectrum: SourceSpectrum, solar_spectrum: Union[SolarSpectrum, SourceSpectrum],
                                       bandpass1: Union[str, SpectralElement], bandpass2: Union[str, SpectralElement],area: u.Quantity,
                                       source_spectrum_err: Optional[SourceSpectrum] = None,
                                       solar_spectrum_err: Optional[SourceSpectrum] = None):
        bandpass1 = format_bandpass(bandpass1)
        bandpass2 = format_bandpass(bandpass2)
        #wave1 = TypicalWaveSfluxd.average_wave(bandpass1) # TODO: to be checked
        #wave2 = TypicalWaveSfluxd.average_wave(bandpass2) # TODO: to be checked

        countrate1 = calculate_count_rate(source_spectrum, bandpass1, area=area)
        countrate2 = calculate_count_rate(source_spectrum, bandpass2, area=area)
        countrate1_err = calculate_count_rate(source_spectrum_err, bandpass1, area=area)
        countrate2_err = calculate_count_rate(source_spectrum_err, bandpass2, area=area)
        return ReddeningCalculator.from_countrate(countrate1, countrate2, solar_spectrum, bandpass1, bandpass2, area, countrate1_err, countrate2_err, solar_spectrum_err)

    # reddening defined with flux or countrate, depending on the definition of afrho
    @staticmethod
    def from_afrho(afrho1: u.Quantity, afrho2: u.Quantity, wave1: u.Quantity, wave2: u.Quantity,
                   afrho1_err: Optional[u.Quantity] = None, afrho2_err: Optional[u.Quantity] = None):
        if afrho1_err is None and afrho2_err is None:
            reddening = ReddeningCalculator.basic_function(afrho1, afrho2, wave1, wave2)
            return reddening
        else:
            reddening, reddening_err = ReddeningCalculator.basic_function_err(afrho1, afrho2, wave1, wave2,
                                                                              afrho1_err, afrho2_err)
            return reddening, reddening_err
        
class TypicalWaveSfluxd:
    @staticmethod
    def pivot_wave(bandpass: Union[str, SpectralElement]):
        """
        pivot wave只由bandpass决定，方便lambda与nu的转换
        PHOTFLAM就是为了将count rate转换为pivot wave处的sfluxd

        需要一个函数，用来将pivot_wave处的sfluxd转换为其他波长处的sfluxd，这样可以对比不同的sfluxd计算得到的afrho（或星等）
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
    def isophotal_sfluxd(source_spectrum: SourceSpectrum, bandpass: Union[str, SpectralElement]):
        ''' 
        Tokunaga & Vacca (2005PASP..117..421T)
        sfluxd = ∫λF_λ(λ)S(λ)dλ / ∫λS(λ)dλ
        这样的sfluxd的平谱得到的photons number（积分面积）与实际的sfluxd通过整个波段的photons number（积分面积）是相等的
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
    
    def isoflux_sfluxd(source_spectrum: SourceSpectrum, bandpass: Union[str, SpectralElement]):
        '''
        sfluxd = ∫ F_λ(λ)S(λ)dλ / ∫ S(λ)dλ
        这样的sfluxd的平谱得到的总通量（积分面积）与实际的sfluxd通过整个波段的总通量（积分面积）是相等的
        根据这个特性，用这个sfluxd计算得到的afrho（或星等）与用总通量计算得到的afrho（或星等）是相等的
        '''
        bandpass = format_bandpass(bandpass)

        # 分子：∫ F_λ(λ)S(λ)dλ
        numerator = calculate_flux(source_spectrum, bandpass)
        # 分母：∫ S(λ)dλ
        denominator = bandpass.equivwidth() # bandpass.equivwidth() = bandpass.integrate()
        return numerator / denominator

class FluxConverter:
    @staticmethod
    def convert_sfluxd_units(flux: u.Quantity, wavelength: u.Quantity, to_unit):
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
    def countrate_to_sfluxd_pivot(count_rate, photflam, target_unit=su.FLAM, wavelength=None, bandpass=None, source_spectrum=None, area=None):
        """
        将计数率转换为sfluxd (erg/s/cm²/Å)
        
        Parameters
        ----------
        count_rate : float
            每秒计数率
        photflam : float
            inverse sensitivity (erg/s/cm²/Å per count/s)
        target_unit : Unit
            目标sfluxd单位
        wavelength : Quantity, optional
            波长，当目标单位不是FLAM时需要提供
        bandpass : str or SpectralElement, optional
            滤光片
        source_spectrum : SourceSpectrum, optional
            源光谱
        """
        if source_spectrum is not None:
            # f689m: 修正前：3.7868471e-19； 修正后：3.750848994323741e-19；相差：1.5%
            # f487n: 修正前：5.9840053e-18； 修正后：5.551361941365817e-18；相差：7.2%
            # f845m: 修正前：4.596043e-19； 修正后：4.523694736174325e-19；相差：1.6%
            # f350lp: 修正前：5.2244e-20； 修正后：6.328435318410915e-20；相差：22%
            bandpass = format_bandpass(bandpass)
            pivot_wave = bandpass.pivot()
            sfluxd_at_pivot_in_theory = source_spectrum(pivot_wave, flux_unit=su.FLAM)
            count_rate_in_theory = calculate_count_rate(source_spectrum, bandpass, area=area)
            photflam = sfluxd_at_pivot_in_theory / count_rate_in_theory
            photflam = photflam.value
        sfluxd = count_rate * photflam * su.FLAM
        
        if target_unit != su.FLAM:
            if wavelength is None and bandpass is not None:
                bandpass = stsyn.band(bandpass)
                wavelength = bandpass.pivot()
            return convert_flux(wavelength, sfluxd, target_unit)
            
        return sfluxd

    @staticmethod
    def countrate_to_flux(count_rate, bandpass, source_spectrum, area, result_unit=u.erg/u.s/u.cm**2):
        """
        将计数率转换为flux
        area: u.Quantity
            area is 1~cm2 for bandpass from effective area (bandpass has no area unit)
        erg/s/cm2: flux
        erg/s: power
        """
        count_rate_in_theory = calculate_count_rate(source_spectrum, bandpass, area=area)
        if result_unit == u.erg/u.s/u.cm**2:
            flux_in_theory = calculate_flux(source_spectrum, bandpass, area=None)
        elif result_unit == u.erg/u.s:
            flux_in_theory = calculate_flux(source_spectrum, bandpass, area=area)
        factor = flux_in_theory / count_rate_in_theory
        return count_rate*(u.ph/u.s) * factor
    
    @staticmethod
    def sfluxd1_to_sfluxd2(sfluxd1: u.Quantity, wavelength1: u.Quantity, wavelength2: u.Quantity, source_spectrum: SourceSpectrum):
        """
        将wavelength1处的sfluxd转换为wavelength2处的sfluxd
        """
        sfluxd1_in_theory = source_spectrum(wavelength1, flux_unit=sfluxd1.unit)
        sfluxd2_in_theory = source_spectrum(wavelength2, flux_unit=sfluxd1.unit)
        factor = sfluxd2_in_theory / sfluxd1_in_theory
        return sfluxd1 * factor
    
    @staticmethod
    def flux1_to_flux2(flux1: u.Quantity, bandpass1: u.Quantity, bandpass2: u.Quantity, source_spectrum: SourceSpectrum):
        """
        将通过bandpass1得到的flux转换为通过bandpass2得到的flux
        """
        flux1_in_theory = calculate_flux(source_spectrum, bandpass1)
        flux2_in_theory = calculate_flux(source_spectrum, bandpass2)
        factor = flux2_in_theory / flux1_in_theory
        return flux1 * factor
    
    @staticmethod
    def sfluxd_to_stmag(sfluxd: u.Quantity, wavelength: u.Quantity,
                       sfluxd_err:Optional[u.Quantity] = None):
        """
        将sfluxd转换为ST星等
        sfluxd_stmag = -2.5*np.log10(sfluxd) - 21.10

        Parameters
        ----------
        sfluxd : Quantity
            输入的flux，必须包含单位
        wavelength : Quantity
            波长，必须包含单位
        """
        sfluxd_stmag = convert_flux(wavelength, sfluxd, u.STmag)
        if sfluxd_err is None:
            return sfluxd_stmag
        else:
            sfluxd_stmag_err = (2.5/(sfluxd*np.log(10))*sfluxd_err).value * u.STmag
            return sfluxd_stmag, sfluxd_stmag_err

    @staticmethod
    def sfluxd_to_abmag(sfluxd: u.Quantity, wavelength: u.Quantity,
                       sfluxd_err:Optional[u.Quantity] = None):
        """
        将sfluxd转换为AB星等
        sfluxd_abmag = -2.5*np.log10(sfluxd) - 48.60

        Parameters
        ----------
        sfluxd : Quantity
            输入的spectral flux density，必须包含单位
        wavelength : Quantity
            波长，必须包含单位
        """
        sfluxd_abmag = convert_flux(wavelength, sfluxd, u.ABmag)
        if sfluxd_err is None:
            return sfluxd_abmag
        else:
            sfluxd_abmag_err = (2.5/(sfluxd*np.log(10))*sfluxd_err).value * u.ABmag
            return sfluxd_abmag, sfluxd_abmag_err

    @staticmethod
    def sfluxd_to_vegamag(sfluxd: u.Quantity, wavelength: u.Quantity,
                         sfluxd_err:Optional[u.Quantity] = None):
        """
        将sfluxd转换为VEGA星等
        
        Parameters
        ----------
        sfluxd : Quantity
            输入的sfluxd，必须包含单位
        wavelength : Quantity
            波长

        sfluxd_vega = vega(bp.pivot(), flux_unit=su.FLAM)
        vegamag = -2.5*np.log10(sfluxd/sfluxd_vega)
        """
        vega = SourceSpectrum.from_vega()
        vegamag = convert_flux(wavelength, sfluxd, su.VEGAMAG, vegaspec=vega)
        if sfluxd_err is None:
            return vegamag
        else:
            vegamag_err = (2.5/(sfluxd*np.log(10))*sfluxd_err).value * su.VEGAMAG
            return vegamag, vegamag_err
    
    @staticmethod
    def flux_to_vegamag(flux: u.Quantity, bandpass: Union[str, SpectralElement], area: u.Quantity = None,
                        flux_err:Optional[u.Quantity] = None):
        """
        if flux's unit is erg/s/cm2 (flux), area needs to be None,
        if flux's unit is erg/s (power), area needs to be provided
            area is 1~cm2 for bandpass from effective area (bandpass has no area unit)
        """
        vega = SourceSpectrum.from_vega()
        flux_vega = calculate_flux(vega, bandpass, area=area)
        #flux_ratio = flux/flux_vega
        #if isinstance(flux_ratio, u.Quantity):
        #    flux_ratio = flux_ratio.value
        #vegamag = -2.5*np.log10(flux_ratio) * su.VEGAMAG
        vegamag = -2.5*np.log10(flux/flux_vega).value * su.VEGAMAG
        if flux_err is None:
            return vegamag
        else:
            vegamag_err = (2.5/(flux*np.log(10))*flux_err).value * su.VEGAMAG
            return vegamag, vegamag_err

    @staticmethod
    def stmag_to_abmag(stmag: Union[float, u.Quantity], wavelength: u.Quantity,
                       stmag_err:Optional[Union[float, u.Quantity]] = None):
        """
        ST星等转换为AB星等
        
        Parameters
        ----------
        stmag : float
            ST星等
        wavelength : Quantity
            波长
        """
        #if isinstance(stmag, float):
        #    stmag = stmag * u.STmag
        ## 先转换为FLAM
        #sfluxd = convert_flux(wavelength, stmag, su.FLAM)
        #if stmag_err is not None:
        #    if isinstance(stmag_err, u.Quantity):
        #        stmag_err = stmag_err.value
        #    sfluxd_err = (np.log(10)/2.5)*np.power(10, -(stmag.value+21.10)/2.5)*stmag_err * su.FLAM
        #else:
        #    sfluxd_err = None
        ## 再转换为AB星等
        #return FluxConverter.sfluxd_to_abmag(sfluxd, wavelength, sfluxd_err)
        if isinstance(stmag, u.Quantity):
            stmag = stmag.value
        abmag = (stmag - 27.5) * u.ABmag #+ 21.1 - 48.6
        if stmag_err is None:
            return abmag
        else:
            abmag_err = stmag_err
            if isinstance(abmag_err, float):
                abmag_err = abmag_err * u.ABmag
            return abmag, abmag_err

    @staticmethod
    def stmag_to_vegamag(stmag: Union[float, u.Quantity], wavelength: u.Quantity,
                         stmag_err:Optional[Union[float, u.Quantity]] = None):
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
        sfluxd = convert_flux(wavelength, stmag, su.FLAM)
        if stmag_err is not None:
            if isinstance(stmag_err, u.Quantity):
                stmag_err = stmag_err.value
            sfluxd_err = (np.log(10)/2.5)*np.power(10, -(stmag.value+21.10)/2.5)*stmag_err * su.FLAM
        else:
            sfluxd_err = None
        # 再转换为VEGA星等
        return FluxConverter.sfluxd_to_vegamag(sfluxd, wavelength, sfluxd_err)

    @staticmethod
    def abmag_to_stmag(abmag: Union[float, u.Quantity], wavelength: u.Quantity,
                       abmag_err:Optional[Union[float, u.Quantity]] = None):
        """
        AB星等转换为ST星等
        
        Parameters
        ----------
        abmag : float
            AB星等
        wavelength : Quantity
            波长
        """
        #if isinstance(abmag, float):
        #    abmag = abmag * u.ABmag
        ## 先转换为FLAM
        #sfluxd = convert_flux(wavelength, abmag, su.FLAM)
        #
        #if abmag_err is not None:
        #    if isinstance(abmag_err, u.Quantity):
        #        abmag_err = abmag_err.value
        #    sfluxd_err = (np.log(10)/2.5)*np.power(10, -(abmag.value+48.60)/2.5)*abmag_err * su.FLAM
        #else:
        #    sfluxd_err = None
        #
        ## 再转换为ST星等
        #return FluxConverter.sfluxd_to_stmag(sfluxd, wavelength, sfluxd_err)
        if isinstance(abmag, u.Quantity):
            abmag = abmag.value
        stmag = (abmag + 27.5) * u.STmag #- 21.1 + 48.6
        if abmag_err is None:
            return stmag
        else:
            stmag_err = abmag_err
            if isinstance(stmag_err, float):
                stmag_err = stmag_err * u.STmag
            return stmag, stmag_err


    @staticmethod
    def abmag_to_vegamag(abmag: Union[float, u.Quantity], wavelength: u.Quantity,
                         abmag_err:Optional[Union[float, u.Quantity]] = None):
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
        sfluxd = convert_flux(wavelength, abmag, su.FLAM)

        if abmag_err is not None:
            if isinstance(abmag_err, u.Quantity):
                abmag_err = abmag_err.value
            sfluxd_err = (np.log(10)/2.5)*np.power(10, -(abmag.value+48.60)/2.5)*abmag_err * su.FLAM
        else:
            sfluxd_err = None

        # 再转换为VEGA星等
        return FluxConverter.sfluxd_to_vegamag(sfluxd, wavelength, sfluxd_err)

    @staticmethod
    def vegamag_to_stmag(vegamag: Union[float, u.Quantity], wavelength: u.Quantity,
                         vegamag_err:Optional[Union[float, u.Quantity]] = None):
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
        sfluxd = convert_flux(wavelength, vegamag, su.FLAM, vegaspec=vega)

        if vegamag_err is not None:
            sfluxd_vega = vega(wavelength, flux_unit=su.FLAM)
            if isinstance(vegamag_err, u.Quantity):
                vegamag_err = vegamag_err.value
            sfluxd_err = 0.4*sfluxd_vega*np.log(10)*np.power(10, -0.4*vegamag.value)*vegamag_err
        else:
            sfluxd_err = None
        # 再转换为ST星等
        return FluxConverter.sfluxd_to_stmag(sfluxd, wavelength, sfluxd_err)

    @staticmethod
    def vegamag_to_abmag(vegamag: Union[float, u.Quantity], wavelength: u.Quantity,
                         vegamag_err:Optional[Union[float, u.Quantity]] = None):
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
        sfluxd = convert_flux(wavelength, vegamag, su.FLAM, vegaspec=vega)

        if vegamag_err is not None:
            sfluxd_vega = vega(wavelength, flux_unit=su.FLAM)
            if isinstance(vegamag_err, u.Quantity):
                vegamag_err = vegamag_err.value
            sfluxd_err = 0.4*sfluxd_vega*np.log(10)*np.power(10, -0.4*vegamag.value)*vegamag_err
        else:
            sfluxd_err = None
        # 再转换为AB星等
        return FluxConverter.sfluxd_to_abmag(sfluxd, wavelength, sfluxd_err)


class SpectrumScaler:
    @staticmethod
    def by_sfluxd(spectrum_to_be_scaled, wavelength, sfluxd_target):
        sfluxd_to_be_scaled = spectrum_to_be_scaled(wavelength)
        
        sfluxd_target = convert_flux(wavelength, sfluxd_target, sfluxd_to_be_scaled.unit)
        
        scale_factor = sfluxd_target/sfluxd_to_be_scaled
        scaled_spectrum = spectrum_to_be_scaled*scale_factor
        return scaled_spectrum
    
    @staticmethod
    def by_flux(spectrum_to_be_scaled, bandpass, flux_target, area=None):
        """
        if flux_target's unit is erg/s/cm2 (flux), area needs to be None,
        if flux_target's unit is erg/s (power), area needs to be provided
            area is 1~cm2 for bandpass from effective area (bandpass has no area unit)
        """
        flux_to_be_scaled = calculate_flux(spectrum_to_be_scaled, bandpass, area=area)
        if isinstance(flux_target, u.Quantity) and flux_target.unit == flux_to_be_scaled.unit:
            scale_factor = flux_target/flux_to_be_scaled
            scaled_spectrum = spectrum_to_be_scaled*scale_factor
            return scaled_spectrum
        else:
            raise ValueError("(check area) flux_target must be a Quantity with the same unit as flux_to_be_scaled")