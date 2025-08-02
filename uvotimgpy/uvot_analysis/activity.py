from typing import Dict, Optional, Union, List, Tuple, Callable, Any
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, differential_evolution, approx_fprime
import warnings
import astropy.units as u
from sbpy.activity.gas import VectorialModel
from sbpy.data import Phys
from synphot import SpectralElement
import synphot.units as su
import csv
import re

from uvotimgpy.config import paths
from uvotimgpy.base.math_tools import UnitConverter
from uvotimgpy.utils.image_operation import DistanceMap
from uvotimgpy.utils.spectrum_operation import ReddeningSpectrum, SolarSpectrum, calculate_flux, calculate_count_rate, ReddeningCalculator, read_OH_spectrum, FluxConverter
from uvotimgpy.base.filters import get_effective_area

def transform_reddening_from_other_papers(reddening, bp1, bp2, measure_method, return_reddened_spectrum=False,
                                          bp1_my=None, bp2_my=None, area=None, my_method='countrate'):
    """
    transform the reddening from other papers to the reddening in our waveband.

    Parameters:
    -----------
    reddening: measured by other papers;
    bp1, bp2: the filters used to measure the reddening in other papers;
    measure_method: 'flux' or 'countrate';
    bp1_my, bp2_my: the filters used in our waveband.

    Returns:
    --------
    reddening: the reddening in our waveband.
    """
    if measure_method == 'flux' or 'countrate':
        reddening_spectrum = ReddeningSpectrum.linear_reddening(reddening, reddening_defination=measure_method, bp1=bp1, bp2=bp2)
    else:
        raise ValueError(f"Not supported measure_method for now: {measure_method}")
    
    gray_dust = SolarSpectrum.from_model()
    reddened_dust = gray_dust*reddening_spectrum

    if return_reddened_spectrum:
        return reddened_dust
    else:
        if my_method == 'countrate':
            reddening_my = ReddeningCalculator.from_countrate_source_spectrum(reddened_dust, gray_dust, bp1_my, bp2_my, area, None, None)
        elif my_method == 'flux':
            reddening_my = ReddeningCalculator.from_flux_source_spectrum(reddened_dust, gray_dust, bp1_my, bp2_my, None, None)
        else:
            raise ValueError(f"Not supported my_method for now: {my_method}")
        return reddening_my

class RatioCalculator_V_UV:
    """
    for swift image subtraction
    """
    @staticmethod
    def reddening_correction(reddening):
        """
        To get (1+t)/(1-t), where t = (Lambda_2-Lambda_1)*R/(2*1000A*100%)

        Obtained with steps below:
            bp_uv = get_effective_area('uvw1', transmission=True, bandpass=True)
            bp_v = get_effective_area('v', transmission=True, bandpass=True)
            average_wave_uv = TypicalWaveSfluxd.average_wave(bp_uv).to(u.AA).value
            average_wave_v = TypicalWaveSfluxd.average_wave(bp_v).to(u.AA).value
        """
        average_wave_uv = 2616.72 # 2616.728247883734 A for UVW1
        average_wave_v = 5429.57 # 5429.569566258718 A for V
        t = (average_wave_uv - average_wave_v)*reddening/(2*1000*100)
        return (1+t)/(1-t)

    @staticmethod
    def dust_countrate_ratio_from_reddening_CR(reddening, gray_dust_spectrum='sun'):
        """
        To get CountRate(UV,dust)/CountRate(V,dust) for swift image subtraction.
        Equals to (1+t)/(1-t) * CountRate(UV,gray dust)/CountRate(V,gray dust)
        This is suitable for reddening defined from countrates, 
            or is an approximation for reddening defined from fluxes.

        To get gray_dust_countrate_ratio
        Obtained with steps below:
            bp_uv = get_effective_area('uvw1', transmission=True, bandpass=True)
            bp_v = get_effective_area('v', transmission=True, bandpass=True)
            sun = SolarSpectrum.from_model()
            area = np.pi*15*15*u.cm**2
            countrate_uv_gray_dust = calculate_count_rate(sun, bp_uv, area)
            countrate_v_gray_dust = calculate_count_rate(sun, bp_v, area)
            gray_dust_countrate_ratio = countrate_uv_gray_dust/countrate_v_gray_dust
        """
        if gray_dust_spectrum == 'sun':
            # gray_dust_countrate_ratio = countrate_uv_gray_dust/countrate_v_gray_dust
            gray_dust_countrate_ratio = 0.09020447142191476 # sun = SolarSpectrum.from_model()
            # gray_dust_countrate_ratio = 0.09276191549759981 # sun = SolarSpectrum.from_colina96()
        else:
            raise ValueError(f"Not supported for now: {gray_dust_spectrum}")

        correction_factor = RatioCalculator_V_UV.reddening_correction(reddening)
        return correction_factor * gray_dust_countrate_ratio
    
    @staticmethod
    def dust_flux_ratio_from_reddening_flux(reddening, gray_dust_spectrum='sun'):
        """
        To get Flux(UV,dust)/Flux(V,dust) for swift image subtraction.
        Equals to (1+t)/(1-t) * Flux(UV,gray dust)/Flux(V,gray dust)
        This is suitable for reddening defined from fluxes, 
            or is an approximation for reddening defined from countrates.

        To get gray_dust_flux_ratio
        Obtained with steps below:
            bp_uv = get_effective_area('uvw1', transmission=True, bandpass=True)
            bp_v = get_effective_area('v', transmission=True, bandpass=True)
            sun = SolarSpectrum.from_model()
            flux_uv_gray_dust = calculate_flux(sun, bp_uv)
            flux_v_gray_dust = calculate_flux(sun, bp_v)
            gray_dust_flux_ratio = flux_uv_gray_dust/flux_v_gray_dust
        """
        if gray_dust_spectrum == 'sun':
            # gray_dust_flux_ratio = flux_uv_gray_dust/flux_v_gray_dust
            gray_dust_flux_ratio = 0.15373407817907703 # sun = SolarSpectrum.from_model()
            # gray_dust_flux_ratio = 0.15838868851932458 # sun = SolarSpectrum.from_colina96()
        else:
            raise ValueError(f"Not supported for now: {gray_dust_spectrum}")

        correction_factor = RatioCalculator_V_UV.reddening_correction(reddening)
        return correction_factor * gray_dust_flux_ratio
    
    @staticmethod
    def dust_countrate_ratio_from_reddening_flux(reddening, bp1=None, bp2=None, gray_dust_spectrum='sun'):
        """
        To get CountRate(UV,dust)/CountRate(V,dust) for swift image subtraction.
        Equals to (1+t)/(1-t) * CountRate(UV,gray dust)/CountRate(V,gray dust)
        This is suitable for reddening defined from flux.

        bp1 and bp2 are the two filters used to measure the reddening. They needs to be SpectralElement objects.
        """
        if gray_dust_spectrum == 'sun':
            gray_dust_spectrum = SolarSpectrum.from_model()
            # gray_dust_flux_ratio = flux_uv_gray_dust/flux_v_gray_dust
            gray_dust_flux_ratio = 0.15373407817907703 # sun = SolarSpectrum.from_model()
            # gray_dust_flux_ratio = 0.15838868851932458 # sun = SolarSpectrum.from_colina96()
        else:
            raise ValueError(f"Not supported for now: {gray_dust_spectrum}")
        bp_uv = get_effective_area('uvw1', transmission=True, bandpass=True)
        bp_v = get_effective_area('v', transmission=True, bandpass=True)
        if bp1 is None and bp2 is None:
            bp1 = bp_uv
            bp2 = bp_v
        reddening_spectrum = ReddeningSpectrum.linear_reddening(reddening, wave_grid_range=[1000, 9000]*u.AA, num_points=1000, 
                                                                wave_grid=None, reddening_defination='flux', bp1=bp1, bp2=bp2)
        reddened_dust_spectrum = gray_dust_spectrum * reddening_spectrum
        cr2flux_uvw1 = calculate_flux(reddened_dust_spectrum, bp_uv)/calculate_count_rate(reddened_dust_spectrum, bp_uv)
        cr2flux_v = calculate_flux(reddened_dust_spectrum, bp_v)/calculate_count_rate(reddened_dust_spectrum, bp_v)
        dust_countrate_ratio = (cr2flux_v/cr2flux_uvw1) * gray_dust_flux_ratio * RatioCalculator_V_UV.reddening_correction(reddening)
        return dust_countrate_ratio

def countrate_to_emission_flux_for_oh(countrate: Union[float, np.ndarray], 
                                      countrate_err: Union[float, np.ndarray]=None,
                                      ):
    """
    Convert countrate measurements to flux OH. Default is for Swift UVOT UVW1.
    The flux is the true flux emitted by coma, instead of observed flux. Unit: erg/s/cm2.
    The flux and countrate are both for the region of measurement (which perhaps is an aperture or even a pixel).

    The factor is obtained with steps below:
        bp = get_effective_area('uvw1', transmission=True, bandpass=True)
        area = np.pi*15*15*u.cm**2
        oh_spectrum = read_OH_spectrum()
        count_rate_in_theory = calculate_count_rate(oh_spectrum, bp, area=area)
        emission_flux_in_theory = oh_spectrum.integrate(flux_unit=su.FLAM)
        factor = emission_flux_in_theory.value/count_rate_in_theory.value 
    """
    factor = 1.2750922625672172e-12
    flux = countrate * factor
    if countrate_err is not None:
        flux_err = countrate_err * factor
        return flux, flux_err
    else:
        return flux

def emission_flux_to_total_number(emission_flux: Union[float, np.ndarray],  
                                  rh: float, delta: float,
                                  rhv: float,
                                  emission_flux_err: Union[float, np.ndarray]=None):
    """
    Convert flux measurements to molecular total number using OH fluorescence g-factors.
    
    Parameters
    ----------
    flux : float or np.ndarray
        Observed flux in erg/s/cm2
    rh : float
        Heliocentric distance in AU
    delta : float
        Distance to observer in AU
    rhv : float
        Radial velocity in km/s (relative to Sun)
    flux_err : float or np.ndarray
        Uncertainty in flux measurements, default is None
        
    Returns
    -------
    total_number : float or np.ndarray
        Molecular total number
    total_number_err : float or np.ndarray
        Uncertainty in molecular total number
        
    Notes
    -----
    This function uses OH fluorescence g-factors from Schleicher's thesis to convert
    observed fluxes to molecular total number. The g-factors are interpolated
    based on the radial velocity and scaled for heliocentric distance.
    
    The luminosity is calculated as:
    L = flux * 4pi * delta^2
    
    Where delta is the observer distance converted to cm.
    
    The g-factor is scaled as:
    g = g_1AU(rv) / r^2
    
    Where r is the heliocentric distance.
    """
    # Load g-factor file
    package_path = paths.package_uvotimgpy
    g_file_path = paths.get_subpath(package_path, 'auxil', 'fluorescenceOH.txt')
    
    if not g_file_path.exists():
        raise FileNotFoundError(f"G-factor file not found: {g_file_path}")
    
    # Read g-factor data (skip first 3 rows which are comments)
    g_data = np.loadtxt(g_file_path, skiprows=3)
    
    # Extract data columns
    helio_v_list = g_data[:, 0]  # Radial velocity in km/s
    # Sum the three fluorescence bands (0-0, 1-0, 1-1); Unit: erg/s/molecule
    g_1au_list = (g_data[:, 1] + g_data[:, 2] + g_data[:, 3]) * 1e-16
    
    # Create interpolation function for g-factors
    g_1au_interp = interp1d(helio_v_list, g_1au_list, kind='linear', fill_value='extrapolate')
    
    # Convert observer distance from AU to cm
    # AU -> km -> cm
    delta_cm = UnitConverter.au_to_km(delta) * 1000 * 100
    
    # Calculate luminosity: L = flux * 4 * pi * delta^2
    luminosity = emission_flux * 4 * np.pi * np.power(delta_cm, 2)
    if emission_flux_err is not None:
        luminosity_err = emission_flux_err * 4 * np.pi * np.power(delta_cm, 2)
    
    # Calculate scaled g-factor: g = g_1AU(rv) / r^2
    g_1au_value = g_1au_interp(rhv)
    g_value = g_1au_value / np.power(rh, 2)
    
    # Calculate molecular number: N = L / g
    total_number = luminosity / g_value
    if emission_flux_err is None:
        return total_number
    else:
        total_number_err = luminosity_err / g_value
        return total_number, total_number_err


def create_vectorial_model(
    r_h: Union[float, u.Quantity],
    base_q: Union[float, u.Quantity] = 1e28,
    parent_params: Optional[Dict] = None,
    fragment_params: Optional[Dict] = None,
    time_q: Optional[Dict] = None,
    grid_params: Optional[Dict] = None,
    print_progress: bool = True
    ) -> VectorialModel:
    """
    创建一个矢量模型。
    
    Parameters
    ----------
    r_h : float or astropy.units.Quantity
        日心距离 (如果是float则单位为AU)
    base_q : float or astropy.units.Quantity
        基础产生率 (如果是float则单位为1/s)
    parent_params : dict, optional
        母分子参数 (默认: H2O参数)
    fragment_params : dict, optional
        碎片分子参数 (默认: OH参数)
    time_q : dict, optional
        时变产生率: {'q': [values], 't': [times]}
    grid_params : dict, optional
        网格参数
    print_progress : bool
        是否打印进度
        
    Returns
    -------
    VectorialModel
        创建的矢量模型实例
    """
    
    # 转换输入为适当的单位
    if not isinstance(r_h, u.Quantity):
        r_h = r_h * u.au
    if not isinstance(base_q, u.Quantity):
        base_q = base_q * (1/u.s)
    
    # 默认H2O母分子参数
    if parent_params is None:
        parent_params = {
            'tau_d': 86000 * u.s,
            'tau_T': 86000 * 0.93 * u.s,
            'v_outflow': 0.85 * u.km/u.s,
            'sigma': 3.0e-16 * u.cm**2
        }
    
    # 默认OH碎片参数
    if fragment_params is None:
        fragment_params = {
            'tau_T': 129000 * u.s,
            'v_photo': 1.05 * u.km/u.s
        }
    
    # 根据日心距离缩放
    r_h_val = r_h.to(u.au).value
    parent_params = parent_params.copy()
    fragment_params = fragment_params.copy()
    
    if 'tau_d' in parent_params:
        parent_params['tau_d'] *= r_h_val**2
    if 'tau_T' in parent_params:
        parent_params['tau_T'] *= r_h_val**2
    if 'v_outflow' in parent_params:
        parent_params['v_outflow'] /= np.sqrt(r_h_val)
    if 'tau_T' in fragment_params:
        fragment_params['tau_T'] *= r_h_val**2
    
    # 转换为Phys对象
    parent_phys = Phys.from_dict(parent_params)
    fragment_phys = Phys.from_dict(fragment_params)
    
    # 网格参数
    grid_defaults = {
        'radial_points': 50,#200,
        'angular_points': 50,#100,
        'radial_substeps': 100,
        'parent_destruction_level': 0.99,
        'fragment_destruction_level': 0.95,
        'max_fragment_lifetimes': 4
    }
    
    if grid_params is not None:
        grid_defaults.update(grid_params)
    
    # 如果提供了time_q，创建q_t函数
    q_t_func = None
    if time_q is not None:
        q_values = time_q['q']
        t_values = time_q['t']
        
        # 如果需要，转换为适当的单位
        if not hasattr(q_values, 'unit'):
            q_values = q_values * (1/u.s)
        if not hasattr(t_values, 'unit'):
            t_values = t_values * u.day
        
        # 创建时间依赖函数
        q_vals = q_values.to(1/u.s).value
        t_vals = t_values.to(u.s).value
        base_q_val = base_q.to(1/u.s).value
        
        def q_t_func(t):
            # t是以秒为单位的过去时间
            if t < 0 or t > t_vals[0]:
                return 0.0
            
            # 查找所属区间
            for i in range(len(t_vals) - 1):
                if t <= t_vals[i] and t > t_vals[i + 1]:
                    return q_vals[i] - base_q_val
            
            # 最后一个区间
            if t <= t_vals[-1]:
                return q_vals[-1] - base_q_val
            
            return 0.0
    
    # 创建模型
    model_kwargs = {
        'base_q': base_q,
        'parent': parent_phys,
        'fragment': fragment_phys,
        'print_progress': print_progress,
        **grid_defaults
    }
    
    if q_t_func is not None:
        model_kwargs['q_t'] = q_t_func
    
    return VectorialModel(**model_kwargs)

def save_vectorial_model_to_csv(vm, filename, comments=None):
    """
    Save VectorialModel parameters and column density data to CSV file.
    
    Parameters
    ----------
    vm : VectorialModel
        The VectorialModel instance to save
    filename : str
        Output CSV filename
    """
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header with metadata
        writer.writerow(['# VectorialModel Parameters and Results'])
        writer.writerow(['# Generated from sbpy VectorialModel'])
        writer.writerow(['#'])
        
        # Basic parameters
        writer.writerow(['# === Basic Parameters ==='])
        writer.writerow([f'# base_q (1/s): {vm.base_q}'])
        writer.writerow([f'# epsilon_max (radians): {vm.epsilon_max}'])
        writer.writerow([f'# d_alpha (radians): {vm.d_alpha}'])
        writer.writerow([f'# time_limit (s): {vm.time_limit}'])
        #writer.writerow([f'# print_progress: {vm.print_progress}'])
        writer.writerow(['#'])
        
        # Parent parameters
        writer.writerow(['# === Parent Parameters ==='])
        writer.writerow([f'# parent.v_outflow (m/s): {vm.parent.v_outflow}'])
        writer.writerow([f'# parent.tau_d (s): {vm.parent.tau_d}'])
        writer.writerow([f'# parent.tau_T (s): {vm.parent.tau_T}'])
        writer.writerow([f'# parent.sigma (m²): {vm.parent.sigma}'])
        writer.writerow(['#'])
        
        # Fragment parameters
        writer.writerow(['# === Fragment Parameters ==='])
        writer.writerow([f'# fragment.v_photo (m/s): {vm.fragment.v_photo}'])
        writer.writerow([f'# fragment.tau_T (s): {vm.fragment.tau_T}'])
        writer.writerow(['#'])
        
        # Grid parameters
        writer.writerow(['# === Grid Parameters ==='])
        writer.writerow([f'# grid.radial_points: {vm.grid.radial_points}'])
        writer.writerow([f'# grid.angular_points: {vm.grid.angular_points}'])
        writer.writerow([f'# grid.radial_substeps: {vm.grid.radial_substeps}'])
        writer.writerow(['#'])
        
        # Model parameters
        writer.writerow(['# === Model Parameters ==='])
        writer.writerow([f'# model_params.parent_destruction_level: {vm.model_params.parent_destruction_level}'])
        writer.writerow([f'# model_params.fragment_destruction_level: {vm.model_params.fragment_destruction_level}'])
        writer.writerow([f'# model_params.max_fragment_lifetimes: {vm.model_params.max_fragment_lifetimes}'])
        writer.writerow(['#'])
        
        # Result parameters (with units)
        writer.writerow(['# === Result Parameters ==='])
        writer.writerow([f'# collision_sphere_radius (m): {vm.vmr.collision_sphere_radius.value}'])
        writer.writerow([f'# max_grid_radius (m): {vm.vmr.max_grid_radius.value}'])
        writer.writerow([f'# coma_radius (m): {vm.vmr.coma_radius.value}'])
        writer.writerow([f'# t_perm_flow (d): {vm.vmr.t_perm_flow.value}'])
        writer.writerow([f'# num_fragments_theory: {vm.vmr.num_fragments_theory}'])
        writer.writerow([f'# num_fragments_grid: {vm.vmr.num_fragments_grid}'])
        writer.writerow(['#'])
        
        # Result parameters (values only, for reference)
        writer.writerow(['# === Result Parameters (values only) ==='])
        writer.writerow([f'# collision_sphere_radius_value (m): {vm.vmr.collision_sphere_radius.value}'])
        writer.writerow([f'# max_grid_radius_value (m): {vm.vmr.max_grid_radius.value}'])
        writer.writerow([f'# coma_radius_value (m): {vm.vmr.coma_radius.value}'])
        writer.writerow([f'# t_perm_flow_value (d): {vm.vmr.t_perm_flow.value}'])
        writer.writerow(['#'])
        
        # Data section header
        writer.writerow(['# === Density Data ==='])
        writer.writerow([f'# rho units: {vm.vmr.column_density_grid.to(u.km).unit}'])
        writer.writerow([f'# column_density units: {vm.vmr.column_density.to(1/u.cm**2).unit}'])
        writer.writerow([f'# volume_density units: {vm.vmr.volume_density.to(1/u.cm**3).unit}'])
        writer.writerow(['#'])

        if comments is not None:
            writer.writerow(['# === Comments ==='])
            writer.writerow([f'# {comments}'])
            writer.writerow(['#'])
        
        # Column headers
        writer.writerow(['rho', 'column_density', 'volume_density'])
        
        # Write data
        grid_values = vm.vmr.column_density_grid.to(u.km).value
        column_density_values = vm.vmr.column_density.to(1/u.cm**2).value
        volume_density_values = vm.vmr.volume_density.to(1/u.cm**3).value
        
        for grid_val, column_density_val, volume_density_val in zip(grid_values, column_density_values, volume_density_values):
            writer.writerow([grid_val, column_density_val, volume_density_val])

def read_vectorial_model_csv(filename):
    """
    Read VectorialModel CSV file and return a simple dictionary.
    
    Parameters
    ----------
    filename : str
        Path to the CSV file
        
    Returns
    -------
    dict
        Dictionary with all parameters and data arrays
    """
    result = {}
    
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        data_section = False
        data_rows = []
        data_units = {}
        column_names = []
        
        for row in reader:
            if not row:
                continue
                
            line = row[0] if len(row) == 1 else ','.join(row)
            
            # Parse parameter lines (comments with : )
            if line.startswith('#') and ':' in line and not 'units:' in line and not '===' in line:
                # Extract parameter name and value
                content = line[2:].strip()  # Remove '# '
                key_part, value_part = content.split(':', 1)
                
                # Extract parameter name (remove units in parentheses)
                key = re.sub(r'\s*\([^)]*\)', '', key_part).strip()
                value_str = value_part.strip()
                
                # Extract units from key
                unit_match = re.search(r'\(([^)]+)\)', key_part)
                unit_str = unit_match.group(1) if unit_match else None
                
                # Convert value
                try:
                    if '.' in value_str or 'e' in value_str.lower():
                        value = float(value_str)
                    else:
                        value = int(value_str)
                    
                    # Add units if available
                    if unit_str:
                        try:
                            value = value * u.Unit(unit_str)
                        except:
                            pass  # Keep as scalar if unit parsing fails
                            
                except ValueError:
                    value = value_str  # Keep as string
                
                result[key] = value
            
            # Parse data units
            elif 'units:' in line:
                # Extract column name and unit
                parts = line.split('units:', 1)
                if len(parts) == 2:
                    column_part = parts[0].replace('#', '').strip()
                    unit_part = parts[1].strip()
                    data_units[column_part] = unit_part
            
            # Check for data section start (any line that doesn't start with #)
            elif data_section:
                if not line.startswith('#'):
                    try:
                        data_row = [float(x) for x in row]
                        data_rows.append(data_row)
                    except ValueError:
                        continue
            
            # Find column headers (first non-comment line after units)
            elif not line.startswith('#') and not data_section:
                column_names = [col.strip() for col in row]
                data_section = True
        
        # Convert data to arrays with units
        if data_rows and column_names:
            data_array = np.array(data_rows)
            
            for i, col_name in enumerate(column_names):
                # Get unit for this column
                unit_str = data_units.get(col_name)
                
                if unit_str:
                    try:
                        unit = u.Unit(unit_str)
                        # Create quantity array with units
                        result[col_name] = data_array[:, i] * unit
                    except:
                        # If unit parsing fails, store as plain array
                        result[col_name] = data_array[:, i]
                else:
                    # No units specified, store as plain array
                    result[col_name] = data_array[:, i]
    
    return result

class ColumnDensityProfile:
    """
    获取柱密度剖面的工具类。
    
    提供从VectorialModel或CSV文件获取柱密度剖面的静态方法。
    """
    
    @staticmethod
    def from_model(
        model,  # VectorialModel
        rho: Optional[Union[Tuple[float, float, float], List[float], np.ndarray]] = None,
        limit_rho: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从VectorialModel获取柱密度剖面。
        
        Parameters
        ----------
        model : VectorialModel
            矢量模型实例
        rho : tuple, list, or array, optional
            如果是元组: (start_km, stop_km, step_km)
            如果是列表/数组: rho值，单位为km
            如果为None: 使用模型的网格点
        limit_rho : bool, optional
            是否限制rho范围在collision_sphere_radius到max_grid_radius之间
        
        Returns
        -------
        tuple
            (rho_km, column_densities_cm2) - 都是无单位的numpy数组
        """
        
        # 处理rho输入
        if rho is None:
            rho_meters = model.vmr.column_density_grid.value
            rho_km = rho_meters / 1000
        elif isinstance(rho, tuple) and len(rho) == 3:
            start, stop, step = rho
            rho_km = np.arange(start, stop + step, step)
            rho_meters = rho_km * 1000  # km转m
        else:
            rho_km = np.array(rho)
            rho_meters = rho_km * 1000  # km转m
        
        # 直接使用内部插值函数
        # 这比为每个点调用column_density()要快得多
        if rho is None:
            column_densities_m2 = model.vmr.column_density.value
        else:
            column_densities_m2 = model.vmr.column_density_interpolation(rho_meters)
        
        # 从m^-2转换为cm^-2
        column_densities_cm2 = column_densities_m2 * 1e-4  # m^-2转cm^-2
        
        # 应用限制
        if limit_rho:
            rho_km, column_densities_cm2 = ColumnDensityProfile._apply_rho_limits(
                rho_km, column_densities_cm2, 
                model.vmr.max_grid_radius.to(u.km).value, 
                model.vmr.collision_sphere_radius.to(u.km).value
            )
        
        return rho_km, column_densities_cm2
    
    @staticmethod
    def from_file(
        filename: str,
        rho: Optional[Union[Tuple[float, float, float], List[float], np.ndarray]] = None,
        limit_rho: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从CSV文件获取柱密度剖面。
        
        Parameters
        ----------
        filename : str
            CSV文件路径
        rho : tuple, list, or array, optional
            如果是元组: (start_km, stop_km, step_km)
            如果是列表/数组: rho值，单位为km
            如果为None: 使用文件中的网格点
        limit_rho : bool, optional
            是否限制rho范围在collision_sphere_radius到max_grid_radius之间
        
        Returns
        -------
        tuple
            (rho_km, column_densities_cm2) - 都是无单位的numpy数组
        """
        # 读取数据
        data = read_vectorial_model_csv(filename)
        
        # 处理rho输入
        if rho is None:
            # 使用文件中的网格数据，直接转换单位
            rho_km = data['rho'].to(u.km).value
            column_densities_cm2 = data['column_density'].to(1/u.cm**2).value
        
        elif isinstance(rho, tuple) and len(rho) == 3:
            start, stop, step = rho
            rho_km = np.arange(start, stop + step, step)
            
            # 需要插值，先转换原始数据单位
            file_rho_km = data['rho'].to(u.km).value
            file_cd_cm2 = data['column_density'].to(1/u.cm**2).value
            column_densities_cm2 = ColumnDensityProfile._interpolate_from_data(
                file_rho_km, file_cd_cm2, rho_km
            )
        
        else:
            rho_km = np.array(rho)
            
            # 需要插值，先转换原始数据单位
            file_rho_km = data['rho'].to(u.km).value
            file_cd_cm2 = data['column_density'].to(1/u.cm**2).value
            column_densities_cm2 = ColumnDensityProfile._interpolate_from_data(
                file_rho_km, file_cd_cm2, rho_km
            )
        
        # 应用限制
        if limit_rho:
            # 从数据中获取限制值，直接转换单位
            max_grid_radius_km = data['max_grid_radius'].to(u.km).value
            collision_sphere_radius_km = data['collision_sphere_radius'].to(u.km).value
            
            rho_km, column_densities_cm2 = ColumnDensityProfile._apply_rho_limits(
                rho_km, column_densities_cm2, max_grid_radius_km, collision_sphere_radius_km
            )
        
        return rho_km, column_densities_cm2
    
    @staticmethod
    def _interpolate_from_data(
        file_rho_km: np.ndarray, 
        file_cd_cm2: np.ndarray, 
        rho_km: np.ndarray
    ) -> np.ndarray:
        """
        插值获取柱密度。
        
        Parameters
        ----------
        file_rho_km : np.ndarray
            原始网格数据，单位为km，无单位
        file_cd_cm2 : np.ndarray
            原始柱密度数据，单位为1/cm^2，无单位
        rho_km : np.ndarray
            需要插值的rho值，单位为km，无单位
        
        Returns
        -------
        np.ndarray
            插值后的柱密度，单位为1/cm^2，无单位
        """
        from scipy.interpolate import CubicSpline
        
        # 创建插值函数
        interpolator = CubicSpline(file_rho_km, file_cd_cm2, bc_type="natural")
        
        # 插值
        return interpolator(rho_km)
    
    @staticmethod
    def _apply_rho_limits(
        rho_km: np.ndarray, 
        column_densities_cm2: np.ndarray, 
        max_grid_radius_km: float,
        collision_sphere_radius_km: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用rho范围限制。
        
        Parameters
        ----------
        rho_km : np.ndarray
            rho值，单位为km，无单位
        column_densities_cm2 : np.ndarray
            柱密度值，单位为1/cm^2，无单位
        max_grid_radius_km : float
            最大网格半径，单位为km，无单位
        collision_sphere_radius_km : float
            碰撞球半径，单位为km，无单位
        
        Returns
        -------
        tuple
            限制后的(rho_km, column_densities_cm2)，都无单位
        """
        # 应用限制
        mask = (rho_km >= collision_sphere_radius_km) & (rho_km <= max_grid_radius_km)
        
        return rho_km[mask], column_densities_cm2[mask]


def build_column_density_image(rho_km, column_densities_cm2, center, 
                               km_per_pixel=None, 
                               pixel_scale=None, delta=None, 
                               oversampling_factor=1,
                               fill_value=np.nan) -> np.ndarray:
    """
    根据距离-柱密度数据构建2D图像
    
    Parameters:
    -----------
    rho_km : array_like
        距离彗核的距离数组 (km)
    column_densities_cm2 : array_like
        对应的柱密度数组 (cm^-2)
    center : tuple
        中心坐标 (row, col)，即图像中彗核的位置
    km_per_pixel : float, optional
        每像素对应的公里数
    pixel_scale : float, optional
        像素尺度 (arcsec/pixel)，当km_per_pixel为None时使用
    delta : float, optional
        距离 (au)，当km_per_pixel为None时使用
    oversampling_factor : int, optional
        必须是奇数
        过采样因子，用于提高图像质量
        把本身的1*1个像素扩展为oversampling_factor*oversampling_factor个像素
        新的中心是(row * factor + (factor-1)/2, col * factor + (factor-1)/2)
        新的km_per_pixel是km_per_pixel / factor
    fill_value : float or str, optional
        插值范围外的填充值，可以是数值、'extrapolate'或np.nan (默认: np.nan)
    
    Returns:
    --------
    numpy.ndarray
        2D柱密度图像，shape为(2*center[0]+1, 2*center[1]+1)
    """
    
    # 参数验证
    rho_km = np.asarray(rho_km)
    column_densities_cm2 = np.asarray(column_densities_cm2)
    
    # 用户输入验证 - 使用具体的异常类型
    if len(rho_km) != len(column_densities_cm2):
        raise ValueError("rho_km和column_densities_cm2长度必须相同")
    
    if not all(isinstance(x, (int, np.integer)) and x >= 0 for x in center):
        raise ValueError("center坐标必须是非负整数")
    
    if km_per_pixel is None:
        if pixel_scale is None or delta is None:
            raise ValueError("必须提供km_per_pixel，或者同时提供pixel_scale和delta")
        # 计算km_per_pixel
        arcsec = UnitConverter.pixel_to_arcsec(1.0, pixel_scale)
        km_per_pixel = UnitConverter.arcsec_to_km(arcsec, delta)

    # 确保rho_km是单调递增的
    if not np.all(np.diff(rho_km) > 0):
        # 如果不是单调递增，先排序
        sort_indices = np.argsort(rho_km)
        rho_sorted = rho_km[sort_indices]
        column_densities_sorted = column_densities_cm2[sort_indices]
    else:
        rho_sorted = rho_km
        column_densities_sorted = column_densities_cm2
    
    # 创建插值函数
    interp_func = interp1d(rho_sorted, column_densities_sorted, 
                          kind='linear', bounds_error=False, 
                          fill_value=fill_value)
    
    # 确定图像尺寸
    center_row, center_col = center

    # oversampling
    if oversampling_factor > 1:
        if not isinstance(oversampling_factor, int) or oversampling_factor % 2 == 0:
            raise ValueError("oversampling_factor必须是奇数")
        center_row = center_row * oversampling_factor + (oversampling_factor-1)/2
        center_col = center_col * oversampling_factor + (oversampling_factor-1)/2
        km_per_pixel = km_per_pixel / oversampling_factor
        warnings.warn(f'oversampling_factor={oversampling_factor},\n \
                       new center = ({center_row}, {center_col}),\n \
                       new km_per_pixel={km_per_pixel}')
    
    empty_image = np.zeros((2 * center_row + 1, 2 * center_col + 1))
    mapper = DistanceMap(empty_image, (center_row, center_col))
    pixel_distances = mapper.get_distance_map
    
    # 转换为物理距离（km）
    physical_distances = pixel_distances * km_per_pixel
    
    # 向量化插值
    image = interp_func(physical_distances)
    
    return image

class TotalNumberCalculator:
    @staticmethod
    def from_model(
        model: VectorialModel,
        radius: u.Quantity, # with u.km
    ) -> float:
        return model.total_number(radius)
    
    @staticmethod
    def from_image(
        image: np.ndarray,
        mask: np.ndarray,
        km_per_pixel=None, 
        pixel_scale=None, delta=None, 
    ) -> float:
        """
        Calculate the total number of OH molecules within a given radius.

        Parameters
        ----------
        image : np.ndarray
            The image of the OH column density. unit: cm^-2
        """
        if km_per_pixel is None:
            if pixel_scale is None or delta is None:
                raise ValueError("必须提供km_per_pixel，或者同时提供pixel_scale和delta")
            # 计算km_per_pixel
            arcsec = UnitConverter.pixel_to_arcsec(1.0, pixel_scale)
            km_per_pixel = UnitConverter.arcsec_to_km(arcsec, delta)

        area_per_pixel = (km_per_pixel * 1000 * 100)**2 # cm2
        number_per_pixel = image * area_per_pixel
        total_number = number_per_pixel[mask].sum()
        return total_number
    
    @staticmethod # TODO
    def from_profile():
        pass

def oh_countrate_to_column_density(countrate_per_pixel, pixel_scale, delta, r, rv, countrate_per_pixel_err=None): #TODO
    # cnts/s/pixel
    arcsec = UnitConverter.pixel_to_arcsec(1.0, pixel_scale)
    km_per_pixel = UnitConverter.arcsec_to_km(arcsec, delta)
    cm_per_pixel = km_per_pixel * 1000 * 100
    cm2_per_pixel = cm_per_pixel**2
    countrate_per_cm2 = countrate_per_pixel / cm2_per_pixel
    if countrate_per_pixel_err is not None:
        countrate_per_cm2_err = countrate_per_pixel_err / cm2_per_pixel
        emission_flux, emission_flux_err = countrate_to_emission_flux_for_oh(countrate_per_cm2, countrate_per_cm2_err)
        number, number_err = emission_flux_to_total_number(emission_flux, r, delta, rv, emission_flux_err)
        return number, number_err
    else:
        emission_flux = countrate_to_emission_flux_for_oh(countrate_per_cm2)
        number = emission_flux_to_total_number(emission_flux, r, delta, rv)
    return number

def scale_from_total_number(total_number_data: float, 
                            total_number_model: float, 
                            base_q: float) -> float:
    return base_q * (total_number_data / total_number_model)

class OHProfileFitter:
    """
    用于拟合OH profile的类，同时优化reddening和q_factor两个参数。
    """
    
    def __init__(
        self,
        rho: np.ndarray,
        countrate_uw1: np.ndarray,
        countrate_v: np.ndarray,
        column_density_model: dict,
        reddening_func: Callable[[float], float],
        oh_countrate_to_column_density_factor: float = 1.0,
        countrate_uw1_err: Optional[np.ndarray] = None,
        countrate_v_err: Optional[np.ndarray] = None,
    ):
        """
        初始化拟合器。
        
        Parameters
        ----------
        rho : np.ndarray
            距离数组
        countrate_uw1 : np.ndarray
            UW1波段的亮度profile (unit: count/s)
        countrate_v : np.ndarray
            V波段的亮度profile (unit: count/s)
        column_density_model : dict
            {'rho': np.ndarray, 'column_density': np.ndarray, 'base_q': float}
        reddening_func : Callable
            红化函数，接受reddening参数并返回修正因子
        oh_countrate_to_column_density_factor : float
            从OH亮度到柱密度的转换因子
        countrate_uw1_err : Optional[np.ndarray]
            UW1 profile的误差
        countrate_v_err : Optional[np.ndarray]
            V profile的误差
        """
        # 验证输入
        assert len(rho) == len(countrate_uw1) == len(countrate_v), \
            "距离和profile数组长度必须相同"
        assert 'rho' in column_density_model and 'column_density' in column_density_model and 'base_q' in column_density_model, \
            "column_density_model必须包含'rho', 'column_density'和'base_q'"
        
        # 存储数据
        self.rho = rho
        self.countrate_uw1 = countrate_uw1
        self.countrate_v = countrate_v
        self.rho_model = column_density_model['rho']
        self.colden_model = column_density_model['column_density']
        self.base_q = column_density_model['base_q']
        self.reddening_func = reddening_func
        self.oh_factor = oh_countrate_to_column_density_factor
        
        # 处理误差
        if countrate_uw1_err is None:
            self.countrate_uw1_err = np.ones_like(countrate_uw1) * 0.01 * np.max(countrate_uw1)
        else:
            self.countrate_uw1_err = countrate_uw1_err
            
        if countrate_v_err is None:
            self.countrate_v_err = np.ones_like(countrate_v) * 0.01 * np.max(countrate_v)
        else:
            self.countrate_v_err = countrate_v_err
        
        # 创建插值函数
        self._create_interpolator()
    
    def _create_interpolator(self):
        """创建柱密度模型的插值函数"""
        self.colden_interp = interp1d(
            self.rho_model, 
            self.colden_model, 
            kind='linear', 
            fill_value='extrapolate'
        )
    
    def calculate_oh_error(self, reddening: float) -> np.ndarray:
        """计算OH profile的误差传播"""
        f_red = self.reddening_func(reddening)
        oh_err = np.sqrt(self.countrate_uw1_err**2 + (f_red * self.countrate_v_err)**2)
        return oh_err
    
    def column_density_model_func(self, rho: np.ndarray, q_factor: float) -> np.ndarray:
        """计算模型柱密度"""
        return self.colden_interp(rho) * q_factor
    
    def residual_function(self, params: np.ndarray) -> float:
        """计算残差平方和（标量）"""
        reddening, q_factor = params
        
        # 计算OH profile
        f_red = self.reddening_func(reddening)
        countrate_oh = self.countrate_uw1 - self.countrate_v * f_red
        oh_err = self.calculate_oh_error(reddening)
        
        # 转换为柱密度
        column_density_oh = countrate_oh * self.oh_factor
        column_density_err = oh_err * self.oh_factor
        
        # 计算模型
        try:
            column_density_model = self.column_density_model_func(self.rho, q_factor)
        except Exception:
            return np.inf
        
        # 计算加权残差
        residuals = (column_density_oh - column_density_model) / column_density_err
        
        # 去除无效值
        valid = np.isfinite(residuals)
        if not np.any(valid):
            return np.inf
        
        return np.sum(residuals[valid]**2)
    
    def residual_vector(self, params: np.ndarray) -> np.ndarray:
        """计算残差向量（用于least_squares）"""
        reddening, q_factor = params
        
        # 计算OH profile
        f_red = self.reddening_func(reddening)
        countrate_oh = self.countrate_uw1 - self.countrate_v * f_red
        oh_err = self.calculate_oh_error(reddening)
        
        # 转换为柱密度
        column_density_oh = countrate_oh * self.oh_factor
        column_density_err = oh_err * self.oh_factor
        
        # 计算模型
        try:
            column_density_model = self.column_density_model_func(self.rho, q_factor)
        except Exception:
            return np.full_like(self.rho, 1e10)
        
        # 返回加权残差向量
        residuals = (column_density_oh - column_density_model) / column_density_err
        
        # 处理无效值
        residuals[~np.isfinite(residuals)] = 1e10
        
        return residuals
    
    def _fit_least_squares(
        self, 
        initial_guess: Tuple[float, float],
        bounds: Tuple[list, list]
    ) -> Dict[str, Any]:
        """使用least_squares方法拟合"""
        result = least_squares(
            self.residual_vector,
            initial_guess,
            bounds=bounds,
            max_nfev=5000
        )
        
        if not result.success:
            warnings.warn(f"最小二乘拟合未收敛: {result.message}")
        
        reddening_fit, q_factor_fit = result.x
        
        # 估计误差
        try:
            J = result.jac
            n_data = len(self.rho)
            n_params = 2
            sigma_squared = result.cost / (n_data - n_params) if n_data > n_params else 1.0
            
            try:
                cov = np.linalg.inv(J.T @ J) * sigma_squared
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(J.T @ J) * sigma_squared
            
            pcov = cov
            perr = np.sqrt(np.diag(np.abs(cov)))
            reddening_err, q_factor_err = perr
            
        except Exception as e:
            warnings.warn(f"误差估计失败: {e}")
            reddening_err = 0.1 * abs(reddening_fit)
            q_factor_err = 0.1 * abs(q_factor_fit)
            pcov = None
        
        return {
            'x': result.x,
            'success': result.success,
            'pcov': pcov,
            'errors': (reddening_err, q_factor_err)
        }
    
    def _fit_minimize(
        self, 
        bounds: list
    ) -> Dict[str, Any]:
        """使用differential_evolution方法拟合"""
        result = differential_evolution(
            self.residual_function,
            bounds,
            seed=42,
            maxiter=1000,
            popsize=15
        )
        
        if not result.success:
            warnings.warn(f"优化未收敛: {result.message}")
        
        reddening_fit, q_factor_fit = result.x
        
        # 估计误差
        try:
            eps = np.sqrt(np.finfo(float).eps)
            hess = np.zeros((2, 2))
            
            for i in range(2):
                def grad_i(x):
                    return approx_fprime(x, self.residual_function, eps)[i]
                
                hess[i] = approx_fprime(result.x, grad_i, eps)
            
            pcov = 0.5 * np.linalg.inv(hess)
            perr = np.sqrt(np.diag(np.abs(pcov)))
            reddening_err, q_factor_err = perr
            
        except Exception as e:
            warnings.warn(f"误差估计失败: {e}")
            reddening_err = 0.1 * abs(reddening_fit)
            q_factor_err = 0.1 * abs(q_factor_fit)
            pcov = None
        
        return {
            'x': result.x,
            'success': result.success,
            'pcov': pcov,
            'errors': (reddening_err, q_factor_err)
        }
    
    def _calculate_fit_statistics(
        self, 
        reddening_fit: float, 
        q_factor_fit: float
    ) -> Dict[str, Any]:
        """计算拟合统计量"""
        # 计算最终结果
        f_red_fit = self.reddening_func(reddening_fit)
        countrate_oh_best = self.countrate_uw1 - self.countrate_v * f_red_fit
        column_density_best_model = self.column_density_model_func(self.rho, q_factor_fit)
        
        # 计算卡方
        oh_err_fit = self.calculate_oh_error(reddening_fit)
        column_density_oh_final = countrate_oh_best * self.oh_factor
        column_density_err_final = oh_err_fit * self.oh_factor
        
        residuals = (column_density_oh_final - column_density_best_model) / column_density_err_final
        valid = np.isfinite(residuals)
        chi2 = np.sum(residuals[valid]**2)
        n_data = np.sum(valid)
        n_params = 2
        reduced_chi2 = chi2 / (n_data - n_params) if n_data > n_params else np.nan
        
        return {
            'countrate_oh_best': countrate_oh_best,
            'column_density_best_model': column_density_best_model,
            'chi2': chi2,
            'reduced_chi2': reduced_chi2,
            'n_data_points': n_data
        }
    
    @staticmethod
    def fit(
        rho: np.ndarray,
        countrate_uw1: np.ndarray,
        countrate_v: np.ndarray,
        column_density_model: dict,
        reddening_func: Callable[[float], float],
        reddening_bounds: Tuple[float, float] = (0.0, 1.0),
        true_q_bounds: Tuple[float, float] = (1e26, 1e30),
        countrate_uw1_err: Optional[np.ndarray] = None,
        countrate_v_err: Optional[np.ndarray] = None,
        oh_countrate_to_column_density_factor: float = 1.0,
        initial_guess: Optional[Tuple[float, float]] = None,
        method: str = 'least_squares',
    ) -> Dict[str, Any]:
        """
        执行OH profile拟合的静态方法。
        
        Parameters
        ----------
        [参数说明与原函数相同]
        
        Returns
        -------
        Dict[str, Any]
            包含拟合结果的字典
        """
        # 创建拟合器实例
        fitter = OHProfileFitter(
            rho=rho,
            countrate_uw1=countrate_uw1,
            countrate_v=countrate_v,
            column_density_model=column_density_model,
            reddening_func=reddening_func,
            oh_countrate_to_column_density_factor=oh_countrate_to_column_density_factor,
            countrate_uw1_err=countrate_uw1_err,
            countrate_v_err=countrate_v_err
        )
        
        # 准备边界
        base_q = column_density_model['base_q']
        reddening_bounds = list(reddening_bounds)
        true_q_bounds = list(true_q_bounds)
        
        # 转换为q_factor bounds
        q_factor_bounds = [true_q_bounds[0]/base_q, true_q_bounds[1]/base_q]
        
        # 检查是否有固定参数（边界相同）
        reddening_fixed = np.isclose(reddening_bounds[0], reddening_bounds[1], rtol=1e-10)
        q_factor_fixed = np.isclose(q_factor_bounds[0], q_factor_bounds[1], rtol=1e-10)
        
        # 设置初始猜测值
        if initial_guess is None:
            # 对于固定参数，使用固定值；否则使用中点
            reddening_init = reddening_bounds[0] if reddening_fixed else (reddening_bounds[0] + reddening_bounds[1]) / 2
            q_factor_init = q_factor_bounds[0] if q_factor_fixed else np.sqrt(q_factor_bounds[0] * q_factor_bounds[1])
            initial_guess = (reddening_init, q_factor_init)
        
        # 处理固定参数的情况
        if reddening_fixed and q_factor_fixed:
            # 两个参数都固定，直接计算结果
            warnings.warn("两个参数都被固定，无法进行拟合优化")
            reddening_fit = reddening_bounds[0]
            q_factor_fit = q_factor_bounds[0]
            reddening_err = 0.0
            q_factor_err = 0.0
            pcov = np.zeros((2, 2))
            success = True
            method_used = 'fixed'
        elif reddening_fixed or q_factor_fixed:
            # 一个参数固定，转为一维优化问题
            if reddening_fixed:
                # 固定reddening，优化q_factor
                def objective_1d(q_factor):
                    return fitter.residual_function([reddening_bounds[0], q_factor])
                
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(
                    objective_1d,
                    bounds=(q_factor_bounds[0], q_factor_bounds[1]),
                    method='bounded'
                )
                
                reddening_fit = reddening_bounds[0]
                q_factor_fit = result.x
                reddening_err = 0.0
                q_factor_err = 0.1 * abs(q_factor_fit)  # 简单误差估计
                pcov = None
                success = result.success
                method_used = f'{method}_1d'
            else:
                # 固定q_factor，优化reddening
                def objective_1d(reddening):
                    return fitter.residual_function([reddening, q_factor_bounds[0]])
                
                from scipy.optimize import minimize_scalar
                result = minimize_scalar(
                    objective_1d,
                    bounds=(reddening_bounds[0], reddening_bounds[1]),
                    method='bounded'
                )
                
                reddening_fit = result.x
                q_factor_fit = q_factor_bounds[0]
                reddening_err = 0.1 * abs(reddening_fit)  # 简单误差估计
                q_factor_err = 0.0
                pcov = None
                success = result.success
                method_used = f'{method}_1d'
        else:
            # 正常的二维优化
            # 执行拟合
            if method == 'least_squares':
                try:
                    fit_result = fitter._fit_least_squares(
                        initial_guess,
                        bounds=(
                            [reddening_bounds[0], q_factor_bounds[0]],
                            [reddening_bounds[1], q_factor_bounds[1]]
                        )
                    )
                    method_used = method
                except Exception as e:
                    warnings.warn(f"least_squares失败: {e}，尝试使用minimize方法")
                    method = 'minimize'
                    fit_result = fitter._fit_minimize([reddening_bounds, q_factor_bounds])
                    method_used = 'minimize'
            
            elif method == 'minimize':
                fit_result = fitter._fit_minimize([reddening_bounds, q_factor_bounds])
                method_used = method
            
            else:
                raise ValueError(f"未知的方法: {method}")
            
            # 提取结果
            reddening_fit, q_factor_fit = fit_result['x']
            reddening_err, q_factor_err = fit_result['errors']
            pcov = fit_result['pcov']
            success = fit_result['success']
        
        # 计算统计量
        stats = fitter._calculate_fit_statistics(reddening_fit, q_factor_fit)
        
        # 判断结果是否在边界上（相对容差1%或绝对容差0.01）
        def is_at_boundary(value, lower, upper, rtol=0.01, atol=0.01):
            """判断值是否在边界上"""
            return np.isclose(value, lower, rtol=rtol, atol=atol) or \
                   np.isclose(value, upper, rtol=rtol, atol=atol)
        
        at_boundary = {
            'reddening': is_at_boundary(reddening_fit, reddening_bounds[0], reddening_bounds[1]),
            'q_factor': is_at_boundary(q_factor_fit, q_factor_bounds[0], q_factor_bounds[1]),
            'true_q': is_at_boundary(base_q * q_factor_fit, true_q_bounds[0], true_q_bounds[1])
        }
        
        # 返回结果
        return {
            'reddening': reddening_fit,
            'reddening_err': reddening_err,
            'q_factor': q_factor_fit,
            'q_factor_err': q_factor_err,
            'true_q': base_q * q_factor_fit,
            'true_q_err': base_q * q_factor_err,
            'chi2': stats['chi2'],
            'reduced_chi2': stats['reduced_chi2'],
            'covariance': pcov,
            'countrate_oh_best': stats['countrate_oh_best'],
            'column_density_best_model': stats['column_density_best_model'],
            'n_data_points': stats['n_data_points'],
            'success': success,
            'method': method_used,
            'at_boundary': at_boundary,
            'parameters_fixed': {
                'reddening': reddening_fixed,
                'q_factor': q_factor_fixed
            }
        }


# 使用示例
if __name__ == "__main__":
    # 直接调用静态方法
    #vm = create_vectorial_model(
    #    r_h=2*u.au,
    #    base_q=1e28,
    #)
    #print(vm.vmr.column_density_grid)
    #rho_model, column_density_model = get_column_density_profile(vm, (1.0, 100.0, 1.0))
    #uw1_data = column_density_model/1e12 * 10
    #v_data = column_density_model/1e12 * 100
    #result = OHProfileFitter.fit(
    #    rho=np.arange(1.0, 100.0+1.0, 1.0),
    #    countrate_uw1=uw1_data,
    #    countrate_v=v_data,
    #    column_density_model={
    #        'rho': rho_model,
    #        'column_density': column_density_model,
    #        'base_q': 1e28
    #    },
    #    reddening_func=lambda r: 10**(-0.4 * r),
    #    reddening_bounds=(1.0, 1.0),
    #    true_q_bounds=(1e27, 1e29),
    #    countrate_uw1_err=None,
    #    countrate_v_err=None,
    #    oh_countrate_to_column_density_factor=1e12,
    #    method='least_squares'
    #)
    #
    #print(f"Reddening: {result['reddening']:.3f} ± {result['reddening_err']:.3f}")
    #print(f"True Q: {result['true_q']:.2e} ± {result['true_q_err']:.2e}")
    #print(result)


# Example usage:
    project_path = paths.projects
    project_3i_path = paths.get_subpath(project_path, 'C_2025N1')
    vm_file = paths.get_subpath(project_3i_path, 'docs', 'vectorial_model_results.csv')
    data = read_vectorial_model_csv(vm_file)
    
    # Access parameters
    base_q = data['base_q']  # with units
    parent_velocity = data['parent.v_outflow']  # with units
    radial_points = data['grid.radial_points']  # no units

    # Access data arrays (whatever columns exist in the file)
    for key in data.keys():
        if not hasattr(data[key], 'unit'):  # It's a quantity array
            print(f"{key}: {data[key]}")
    print(hasattr(data['collision_sphere_radius'], 'unit'))