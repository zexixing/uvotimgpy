from astropy import units as u
import numpy as np
from synphot import SpectralElement, Empirical1D
from sbpy.spectroscopy import SpectralGradient, Reddening
from sbpy.units import hundred_nm
import stsynphot as stsyn

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