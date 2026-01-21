import os
from astropy.io import fits
from datetime import datetime
from astropy.time import Time
import numpy as np
from typing import Union, Tuple, List
from uvotimgpy.base.file_and_table import compress_fits, get_caller_filename, ephemeris_keywords

def save_stacked_fits(images_to_save: dict, 
                      save_path: str, obs_list: list, 
                      target_position: Union[Tuple, List],
                      script_name: str = None, 
                      bkg_list: list = None, bkg_error_list: list = None, 
                      comment: str = None,
                      other_header_info: dict = None,
                      compressed: bool = False,
                      stack_unit: str = 'counts/s',
                      hst: bool = False):
    """
    将stacked image保存为fits文件
    
    Parameters
    ----------
    save_path : str
        保存路径
    obs_list : list
        观测列表
    target_position : Union[Tuple, List]
        目标位置, (column, row)
    images_to_save : dict
        要保存的图像字典，key为extension名称，value为对应的2d array
        例如: {'STACKED_IMAGE': stacked_image, 'STACKED_ERROR': stacked_error, 'STACKED_IMAGE_OVERLAP': stacked_image_overlap}
    script_name : str, optional
        脚本名称
    bkg_list : list, optional
        背景值列表
    bkg_error_list : list, optional
        背景误差值列表
    keyword_dict : dict, optional
        存储用来查找相关信息的关键词
        例如: swift: {'file': 'OBSID', 'exp':'EXPOSURE'}, 
              hst: {'file': 'file_name', 'exp':'EXPTIME'}
    compressed : bool, optional
        是否压缩文件
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
    first_date_obs_str = first_date_obs_str.replace(' ', 'T')
    last_date_end_str = last_date_end_str.replace(' ', 'T')
    midtime_iso = midtime_iso.replace(' ', 'T')
    
    # 计算总曝光时间
    if hst:
        total_exptime = sum(obs['EXPTIME'] for obs in obs_list)
    else:
        total_exptime = sum(obs['EXPOSURE'] for obs in obs_list)
    
    # 获取滤光片
    filt = obs_list[0]['FILTER']  # 假设所有观测使用相同的滤光片
    
    # 获取目标位置
    target_pos = target_position
    
    # 写入header信息
    if hst:
        primary_header['YEAR'] = (obs_list[0]['date'], 'Observation year')
    primary_header['DATE_OBS'] = (first_date_obs_str, 'Start time of first observation')
    primary_header['DATE_END'] = (last_date_end_str, 'End time of last observation')
    primary_header['MIDTIME'] = (midtime_iso, 'Middle time between first and last observation')
    primary_header['EXPTIME'] = (total_exptime, 'Total exposure time [s]')
    primary_header['BUNIT'] = (stack_unit, 'Physical unit of array values')
    primary_header['COLPIXEL'] = (target_pos[0], 'Target X position in Python coordinates')
    primary_header['ROWPIXEL'] = (target_pos[1], 'Target Y position in Python coordinates')
    primary_header['DS9XPIX'] = (target_pos[0] + 1, 'Target X position in DS9 coordinates')
    primary_header['DS9YPIX'] = (target_pos[1] + 1, 'Target Y position in DS9 coordinates')
    primary_header['FILTER'] = (filt, 'Filter used in observations')

    # 计算平均rh, delta, elongation
    key_list = obs_list[0].keys()
    ephemeris_keywords_list = ephemeris_keywords()
    ephemeris_keywords_list.extend(['Sky_motion', 'Sky_mot_PA'])
    orbital_keywords = list(set(key_list) & set(ephemeris_keywords_list))
    orbital_keywords = [x for x in orbital_keywords if x not in ['RA', 'DEC', 'ra', 'dec']]
    for key in orbital_keywords:
        avg_value = np.mean([obs[key] for obs in obs_list])
        if key == 'RA*cos(Dec)_rate':
            primary_header['RA_RATE'] = (avg_value, f'Average RA RATE')
        else:
            primary_header[key] = (avg_value, f'Average {key}')

    if bkg_list is not None:
        primary_header['BKG'] = (', '.join(f'{bkg}' for bkg in bkg_list), 'Background values for each image; ctns/s/pixel')
    if bkg_error_list is not None:
        primary_header['BKG_ERR'] = (', '.join(f'{bkg_err}' for bkg_err in bkg_error_list), 'Background error values for each image; ctns/s/pixel')
    
    if other_header_info is not None:
        for key, value in other_header_info.items():
            primary_header[key] = value
    
    primary_header['CREATED'] = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                'File creation time (local)')
    if script_name is None:
        script_name = get_caller_filename()
    primary_header['HISTORY'] = f'Created by {script_name}'
    primary_header['HISTORY'] = f'Created by Zexi Xing'
    
    # 循环处理extension信息
    for ext_num, ext_name in enumerate(images_to_save.keys(), 1):
        primary_header[f'EXT{ext_num}NAME'] = (ext_name, f'Name of extension {ext_num}')
    
    primary_header.add_comment('Orbital values have the same units as default on JPL Horizons ('+', '.join(orbital_keywords)+').')
    if hst: # HST
        file_list = sorted(set([obs['file_name'] for obs in obs_list]))
        primary_header.add_comment('FILE LIST: '+ ', '.join(f'{file_name}' for file_name in file_list))
    else: # Swift
        primary_header.add_comment('EXP LIST: '+ ', '.join(f'{obs["OBSID"]}'[-4:]+f'_{obs["EXT_NO"]}' for obs in obs_list))
    primary_header.add_comment(comment)
    # 创建HDU列表，从primary开始
    hdul = fits.HDUList([primary_hdu])
    
    # 循环创建各个extension
    for ext_name, image_data in images_to_save.items():
        if image_data is not None:
            if image_data.dtype == bool:
                image_hdu = fits.CompImageHDU(data=image_data.astype(np.uint8), compression_type='RICE_1', name=ext_name)
            else:
                image_hdu = fits.ImageHDU(data=image_data, name=ext_name)
            hdul.append(image_hdu)
    
    # 保存文件
    hdul.writeto(save_path, overwrite=True)
    if compressed:
        compress_fits(save_path)

def save_cleaned_fits(images_to_save: dict,
                      save_path: str, obs: dict, 
                      target_position: Union[Tuple, List],
                      script_name: str = None, comment: str = None,
                      other_header_info: dict = None,
                      compressed: bool = True):
    """
    将cleaned image保存为fits文件
    
    Parameters
    ----------
    save_path : str
        保存路径
    obs : dict
        观测信息字典
    target_position : Union[Tuple, List]
        目标位置
    images_to_save : dict
        要保存的图像字典，key为extension名称，value为对应的2d array
        例如: {'CLEANED_IMAGE': cleaned_image, 'UNCLEANED_IMAGE': uncleaned_image, 
               'CLEANED_ERROR': cleaned_error, 'WHT': wht}
    script_name : str, optional
        脚本名称
    compressed : bool, optional
        是否压缩文件，默认True
    comment : str, optional
        注释信息
    """
    primary_hdu = fits.PrimaryHDU()
    primary_header = primary_hdu.header

    for key in obs.keys():
        primary_header[key] = f'{obs[key]}'
    
    target_pos = target_position
    
    primary_header['BUNIT'] = ('COUNTS/S', 'Physical unit of array values')
    primary_header['COLPIXEL'] = (target_pos[0], 'Target column position in Python coordinates')
    primary_header['ROWPIXEL'] = (target_pos[1], 'Target row position in Python coordinates')
    primary_header['DS9XPIX'] = (target_pos[0] + 1, 'Target X position in DS9 coordinates')
    primary_header['DS9YPIX'] = (target_pos[1] + 1, 'Target Y position in DS9 coordinates')

    if other_header_info is not None:
        for key, value in other_header_info.items():
            primary_header[key] = value

    primary_header['CREATED'] = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                'File creation time (local)')
    if script_name is None:
        script_name = get_caller_filename()
    primary_header['HISTORY'] = f'Created by {script_name}'
    primary_header['HISTORY'] = f'Created by Zexi Xing'

    # 循环处理extension信息
    for ext_num, ext_name in enumerate(images_to_save.keys(), 1):
        primary_header[f'EXT{ext_num}NAME'] = (ext_name, f'Name of extension {ext_num}')
    
    primary_header.add_comment(comment)
    
    # 创建HDU列表，从primary开始
    hdul = fits.HDUList([primary_hdu])
    
    # 循环创建各个extension
    for ext_name, image_data in images_to_save.items():
        if image_data is not None:
            image_hdu = fits.ImageHDU(data=image_data, name=ext_name)
            hdul.append(image_hdu)

    hdul.writeto(save_path, overwrite=True)
    if compressed:
        compress_fits(save_path)

