from astropy.time import Time
import numpy as np
import re
from datetime import datetime
from uvotimgpy.base.file_and_table import parse_date_string
from uvotimgpy.uvot_analysis.activity import get_g_factor
from uvotimgpy.base.math_tools import UnitConverter
from uvotimgpy.uvot_analysis.activity import create_vectorial_model, RatioCalculator_V_UV
from astropy import units as u
from typing import Tuple

def create_magnitude_calculator(coefficients, time_intervals=None):
    """
    创建一个计算magnitude的函数
    
    参数:
        coefficients: tuple 或 list of tuples
            - 单个公式: (H, G) 如 (12.0, 17.0)
            - 多个公式: [(H1, G1), (H2, G2), ...] 
        time_intervals: None 或 list of str
            - None: 不分段，使用单个公式
            - list: 时间分段点，支持多种格式:
                    ['2025-10-10', '2026-01-08', '2026-06-27']
                    ['2025 Oct. 10', '2026 Jan. 8', '2026 June 27']
                    ['2025 Oct 10', '2026 Jan 8', '2026 Jun 27']
    
    返回:
        计算magnitude的函数
    """
    
    # 处理输入参数
    if time_intervals is None:
        # 不分段的情况
        if isinstance(coefficients, tuple) and len(coefficients) == 2:
            coeff_list = [coefficients]
        else:
            raise ValueError("不分段时，coefficients应该是单个tuple (H, G)")
        time_boundaries = None
    else:
        # 分段的情况
        if not isinstance(coefficients, list):
            raise ValueError("分段时，coefficients应该是list of tuples")
        
        coeff_list = coefficients
        
        # 解析各种格式的时间字符串
        time_boundaries = []
        for t in time_intervals:
            try:
                parsed_time = parse_date_string(t)
                time_boundaries.append(parsed_time)
            except ValueError as e:
                raise ValueError(f"解析时间 '{t}' 失败: {e}")
        
        # 检查数量匹配
        if len(coeff_list) != len(time_boundaries) + 1:
            raise ValueError(f"需要{len(time_boundaries)+1}个系数组，但提供了{len(coeff_list)}个")
    
    def calculate_magnitude(rh_list, delta_list, date_list):
        """
        计算magnitude: m = H + 5*log10(delta) + G*log10(rh)
        
        参数:
            rh_list: array-like, 日心距离 r（AU）
            delta_list: array-like, 地心距离 d（AU）  
            date_list: list of str, 日期时间字符串
                      支持格式: '2022-01-01T00:00:00.000', '2022-01-01', '2022 Jan 1'等
        
        返回:
            numpy array, 计算得到的magnitude值
        """
        # 转换为numpy数组
        rh = np.asarray(rh_list)
        delta = np.asarray(delta_list)
        
        # 解析日期列表
        times = []
        for date_str in date_list:
            try:
                # 首先尝试Time直接解析
                times.append(Time(date_str))
            except:
                # 如果失败，使用自定义解析
                times.append(parse_date_string(date_str))
        times = Time(times)
        
        # 初始化结果数组
        magnitude = np.zeros_like(rh, dtype=float)
        
        if time_boundaries is None:
            # 不分段，使用单一公式
            H, G = coeff_list[0]
            magnitude = H + 5 * np.log10(delta) + G * np.log10(rh)
        else:
            # 分段计算
            for i in range(len(times)):
                t = times[i]
                
                # 确定使用哪个公式
                if t < time_boundaries[0]:
                    H, G = coeff_list[0]
                elif t >= time_boundaries[-1]:
                    H, G = coeff_list[-1]
                else:
                    # 在中间某个区间
                    for j in range(len(time_boundaries) - 1):
                        if time_boundaries[j] <= t < time_boundaries[j + 1]:
                            H, G = coeff_list[j + 1]
                            break
                
                # 计算magnitude
                magnitude[i] = H + 5 * np.log10(delta[i]) + G * np.log10(rh[i])
        
        return list(magnitude)
    
    # 辅助方法
    def get_formula_info():
        """返回公式配置信息"""
        info = []
        if time_boundaries is None:
            H, G = coeff_list[0]
            info.append(f"m = {H} + 5*log(d) + {G}*log(r)  [all times]")
        else:
            for i, (H, G) in enumerate(coeff_list):
                if i == 0:
                    info.append(f"m = {H:4.1f} + 5*log(d) + {G:4.1f}*log(r)  (before {time_boundaries[0].iso[:10]})")
                elif i == len(coeff_list) - 1:
                    info.append(f"m = {H:4.1f} + 5*log(d) + {G:4.1f}*log(r)  (after {time_boundaries[-1].iso[:10]})")
                else:
                    info.append(f"m = {H:4.1f} + 5*log(d) + {G:4.1f}*log(r)  ({time_boundaries[i-1].iso[:10]} to {time_boundaries[i].iso[:10]})")
        return "\n".join(info)
    
    # 给返回的函数添加属性
    calculate_magnitude.get_formula_info = get_formula_info
    calculate_magnitude.coefficients = coeff_list
    calculate_magnitude.time_boundaries = time_boundaries
    
    return calculate_magnitude

class ExposureCalculator:
    def __init__(self, target_dict, reference_dict, aperture, snr=3):
        """
        初始化曝光时间计算器
        
        Parameters:
        -----------
        reference_dict : dict
            参考源的字典，包含 'm_v' 和 'cr_v' 等键
        target_dict : dict
            目标源的字典，包含 'm_v', 'rh', 'delta', 'rhv' 等键
        aperture : float or tuple
            孔径大小（km），可以是单个值或(内径, 外径)元组
        snr : float
            信噪比要求，默认为3
        """
        self.reference_dict = reference_dict
        self.target_dict = target_dict
        self.aperture = aperture
        self.snr = snr
        
        # 计算像素相关参数
        km_per_pixel = UnitConverter.arcsec_to_km(1*1.004, target_dict['delta'])
        self.x = 3
        
        if isinstance(aperture, Tuple):
            radius_outer = aperture[1]/km_per_pixel
            radius_inner = aperture[0]/km_per_pixel
            self.pixel_number = (radius_outer**2 - radius_inner**2)*np.pi
        else:
            radius = aperture/km_per_pixel
            self.pixel_number = radius**2*np.pi
        
        # 背景计数率
        self.uw1_bkg_countrate = 0.003503 * self.pixel_number
        self.uw1_bkg_countrate_err = 6.475e-04 * np.sqrt(self.pixel_number)
        self.v_bkg_countrate = 0.02226 * self.pixel_number
        self.v_bkg_countrate_err = 0.002021 * np.sqrt(self.pixel_number)
        
        # 添加beta作为实例变量（从RatioCalculator获取）
        self.beta = RatioCalculator_V_UV.dust_countrate_ratio_from_reddening(
            reddening=10, 
            obs_time=Time('2025-01-01T00:00:00.000')
        )
    
    def get_qwater(self, m_v, delta):
        """计算水的产生率"""
        m_h = m_v - 5*np.log10(delta)
        qwater = np.power(10, 30.675 - 0.2453*m_h)
        return qwater
    
    def qwater_to_oh_number_in_aperture(self, qwater, r_h, delta, aperture):
        """计算孔径内的OH分子数"""
        vm = create_vectorial_model(r_h)
        
        if isinstance(aperture, float) or isinstance(aperture, int):
            number_model = vm.total_number(aperture*u.km)
        elif isinstance(aperture, Tuple):
            number_model = vm.total_number(aperture[1]*u.km) - vm.total_number(aperture[0]*u.km)
        
        number_aperture = number_model * qwater/(vm.base_q)
        return number_aperture
    
    def get_oh_countrate(self, oh_number, rhv, rh, delta):
        """计算OH的计数率"""
        g_1au_value = get_g_factor(rhv)
        g_value = g_1au_value / np.power(rh, 2)
        luminosity = oh_number * g_value
        delta_cm = UnitConverter.au_to_km(delta) * 1000 * 100
        emission_flux = luminosity / (4 * np.pi * np.power(delta_cm, 2))
        factor = 1/(1.6368359501510164e-12)
        countrate = emission_flux * factor
        return countrate
    
    def get_v_dust_countrate(self, m_v, m_v_ref, cr_v_ref):
        """计算V波段尘埃计数率"""
        f_to_fref = np.power(10, -0.4*(m_v - m_v_ref))
        cr_to_crref = f_to_fref
        cr_in_aperture = cr_v_ref * cr_to_crref
        return cr_in_aperture
    
    def get_cr_uw1(self, oh_countrate, v_dust_countrate, uw1_bkg_countrate):
        """计算UW1波段总计数率"""
        cr_uw1 = oh_countrate + self.beta*v_dust_countrate + uw1_bkg_countrate
        return cr_uw1
    
    def get_cr_v(self, v_dust_countrate, v_bkg_countrate):
        """计算V波段总计数率"""
        cr_v = v_dust_countrate + v_bkg_countrate
        return cr_v
    
    def get_exposure_time(self, x, cr_uw1, cr_v, cr_oh, cr_uw1_bkg_err, cr_v_bkg_err, snr):
        """计算所需的曝光时间"""
        numerator = cr_uw1/x + (self.beta**2)*cr_v
        a = (cr_uw1_bkg_err**2) + (self.beta**2)*(cr_v_bkg_err**2)
        denominator = (cr_oh/snr)**2 - a
        
        if denominator <= 0:
            print('best SNR = '+f'{cr_oh/np.sqrt(a):.2f}')
            raise ValueError("无法达到所需的信噪比，分母为负值或零")
        
        exposure = numerator/denominator

        return exposure
    
    def calculate_exposure_time(self):
        """主方法：计算曝光时间"""
        target_dict = self.target_dict
        reference_dict = self.reference_dict
        
        # 计算水的产生率
        qwater = self.get_qwater(target_dict['m_v'], target_dict['delta'])
        
        # 计算孔径内的OH数量
        oh_number_in_aperture = self.qwater_to_oh_number_in_aperture(
            qwater, 
            target_dict['rh'], 
            target_dict['delta'], 
            self.aperture
        )
        
        # 计算OH计数率
        oh_countrate = self.get_oh_countrate(
            oh_number_in_aperture, 
            target_dict['rhv'], 
            target_dict['rh'], 
            target_dict['delta']
        )
    
        # 计算V波段尘埃计数率
        v_dust_countrate = self.get_v_dust_countrate(
            target_dict['m_v'], 
            reference_dict['m_v'], 
            reference_dict['cr_v']
        )
        
        # 计算总计数率
        print(qwater, oh_number_in_aperture,
              oh_countrate, v_dust_countrate, 
              self.uw1_bkg_countrate, self.v_bkg_countrate)
        print(self.uw1_bkg_countrate_err, self.v_bkg_countrate_err)
        cr_uw1 = self.get_cr_uw1(oh_countrate, v_dust_countrate, self.uw1_bkg_countrate)
        cr_v = self.get_cr_v(v_dust_countrate, self.v_bkg_countrate)
        
        # 计算曝光时间
        exposure = self.get_exposure_time(
            self.x, 
            cr_uw1, 
            cr_v, 
            oh_countrate, 
            self.uw1_bkg_countrate_err, 
            self.v_bkg_countrate_err, 
            self.snr
        )
        
        return exposure


if __name__ == "__main__":
    #target_j3 = {'m_v': 14.1, 'rh': 4.0, 'delta': 3.47, 'rhv': -3.52, 'date': '2026-08-11'} #12.92
    #
    #target_e1 = {'m_v': 15.8, 'rh': 3.05, 'delta': 2.85, 'rhv': -21.78, 'date': '2025-07-24'}
    #
    #mag_calc_e1 = create_magnitude_calculator((10.5, 10.0))
    #print(mag_calc_e1([target_e1['rh']], [target_e1['delta']], [target_e1['date']]))


    #km = UnitConverter.arcsec_to_km(1, target_e1['delta'])
    #aperture = (36000, 40000) #km
    #print(40000/2067.022189836139)
    #print(36000/2067.022189836139)
    #
    #print((779.62 - 681.45)/196.6)

    target_t5 = {'m_v': 13.67, 'rh': 4.0, 'delta': 3.2, 'rhv': -4.05, 'date': '2027-01-04'} #12.05
    aperture = (36000, 40000) #km
    target_e1 = {'m_v': 14.1, 'rh': 1.9, 'delta': 2.45, 'rhv': 25.5, 'date': '2026-04-28'}
    target_e1 = {'m_v': 17.4, 'rh': 4.0, 'delta': 4.25, 'rhv': 19.5, 'date': '2026-10-09'}
    aperture = 40000
    reference_dict = {'m_v': 15.8, 'cr_v': 0.49934, 'rh':3.05, 'delta':2.85, 'rhv':-21.76, 'date':'2025-07-24'}
    etc = ExposureCalculator(target_dict=target_e1, reference_dict=reference_dict, aperture=aperture, snr=3)
    t_v = etc.calculate_exposure_time()
    t_uw1 = 3*t_v
    print(t_v, t_uw1, (t_v+t_uw1)/1600)

    