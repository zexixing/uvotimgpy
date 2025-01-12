import os
from astropy.io import fits
from datetime import datetime
from astropy.time import Time
import numpy as np
from typing import Union, Tuple, List, Optional

def process_astropy_table(data_table, output_path=None, save_format='csv'):
    """
    Process an Astropy table by either saving it to a file or printing it to console.

    Parameters:
    data_table (astropy.table.Table): The Astropy table to process.
    output_path (str, optional): Path for the output file. If None, the table will be printed to console. For saving, absolute is recommended.
    save_format (str, optional): Output file format if saving. Default is 'csv'.
    """
    if output_path is None:
        data_table.pprint(max_lines=-1, max_width=-1)
    else:
        save_astropy_table(data_table, output_path, save_format)

def save_astropy_table(data_table, output_path, save_format='csv'):
    """
    Save an Astropy table to a file.

    Parameters:
    data_table (astropy.table.Table): The Astropy table to save.
    output_path (str): Path for the output file. Absolute path is recommended.
    save_format (str, optional): Output file format. Default is 'csv'.
    """
    if not os.path.isabs(output_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_output_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'output', output_path))
    else:
        full_output_path = output_path
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    data_table.write(full_output_path, format=save_format, overwrite=True)
    print(f"Data saved to: {full_output_path}")

def save_stacked_fits(save_path: str, obs_list: list, 
                     stacked_image: np.ndarray, target_position: Union[Tuple, List],
                     stacked_error: np.ndarray = None, script_name: str = None):
    """
    将stacked image保存为fits文件
    
    Parameters
    ----------
    save_path : str
        保存路径
    obs_list : list
        观测列表
    stacked_image : np.ndarray
        叠加后的图像
    para_dict : dict
        参数字典
    stacked_error : np.ndarray, optional
        叠加后的误差图像
    """
    # 创建主header
    primary_hdu = fits.PrimaryHDU()
    primary_header = primary_hdu.header
    
    # 获取时间信息并转换为Time对象
    times_obs = Time([obs['DATE_OBS'] for obs in obs_list])
    times_end = Time([obs['DATE_END'] for obs in obs_list])
    
    # 获取最早的开始时间和最晚的结束时间
    first_date_obs = times_obs.min()
    last_date_end = times_end.max()
    
    # 计算中间时间
    midtime = Time(first_date_obs.jd + (last_date_end.jd - first_date_obs.jd) / 2, format='jd')
    
    # 转换为ISO格式字符串用于保存在header中
    first_date_obs_str = first_date_obs.iso
    last_date_end_str = last_date_end.iso
    midtime_iso = midtime.iso
    
    # 获取文件名列表
    file_names = [obs['file_name'] for obs in obs_list]
    
    # 计算总曝光时间
    total_exptime = sum(obs['EXPTIME'] for obs in obs_list)
    
    # 获取滤光片
    filt = obs_list[0]['FILTER']  # 假设所有观测使用相同的滤光片
    
    # 计算平均rh, delta, elongation
    avg_rh = np.mean([obs['r'] for obs in obs_list])
    avg_delta = np.mean([obs['delta'] for obs in obs_list])
    avg_elong = np.mean([obs['elong'] for obs in obs_list])
    
    # 获取目标位置
    target_pos = target_position

    # 获取当前文件名
    if script_name is None:
        script_name = __file__
    
    # 写入header信息
    primary_header['DATE_OBS'] = (first_date_obs_str, 'Start time of first observation')
    primary_header['DATE_END'] = (last_date_end_str, 'End time of last observation')
    primary_header['MIDTIME'] = (midtime_iso, 'Middle time between first and last observation')
    primary_header['FILELIST'] = (', '.join(f'{file_name}' for file_name in file_names), 'List of files used in stacking')
    primary_header['EXPTIME'] = (total_exptime, 'Total exposure time [s]')
    primary_header['BUNIT'] = ('ELECTRONS/S', 'Physical unit of array values')
    primary_header['XPIXEL'] = (target_pos[0], 'Target X position in Python coordinates')
    primary_header['YPIXEL'] = (target_pos[1], 'Target Y position in Python coordinates')
    primary_header['DS9XPIX'] = (target_pos[0] + 1, 'Target X position in DS9 coordinates')
    primary_header['DS9YPIX'] = (target_pos[1] + 1, 'Target Y position in DS9 coordinates')
    primary_header['FILTER'] = (filt, 'Filter used in observations')
    primary_header['R'] = (avg_rh, 'Average heliocentric distance [AU]')
    primary_header['DELTA'] = (avg_delta, 'Average geocentric distance [AU]')
    primary_header['ELONG'] = (avg_elong, 'Average solar elongation [deg]')
    primary_header['CREATED'] = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                'File creation time (local)')
    primary_header['HISTORY'] = f'Created by {script_name}'
    
    # 注明extension信息
    primary_header['EXT1NAME'] = ('STACKED_IMAGE', 'Name of extension 1')
    if stacked_error is not None:
        primary_header['EXT2NAME'] = ('STACKED_ERROR', 'Name of extension 2')
    
    # 创建extension
    image_hdu = fits.ImageHDU(data=stacked_image, name='STACKED_IMAGE')
    
    # 创建HDU列表
    hdul = fits.HDUList([primary_hdu, image_hdu])
    
    # 如果有误差图像，添加到第三个extension
    if stacked_error is not None:
        error_hdu = fits.ImageHDU(data=stacked_error, name='STACKED_ERROR')
        hdul.append(error_hdu)
    
    # 保存文件
    #save_path = save_path[:-3]
    hdul.writeto(save_path, overwrite=True)
