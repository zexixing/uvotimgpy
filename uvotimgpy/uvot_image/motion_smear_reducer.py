import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons
from typing import List, Tuple, Optional, Union
from sbpy.data import Ephem
from astropy.nddata import block_reduce
import pathlib
from uvotimgpy.utils.image_operation import align_images, stack_images, DS9Converter, bin_image
from uvotimgpy.base.math_tools import icrf_to_fk5

def read_event_file(evt_file_path: Union[str, pathlib.Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]:
    """
    读取事件文件
    
    Parameters
    ----------
    evt_file_path : str
        事件文件路径
    
    Returns
    -------
    tuple
        (time_list, x_list, y_list, header)
    """
    with fits.open(evt_file_path) as hdul:
        data = hdul[1].data
        header = hdul[1].header
        
        time_list = data['TIME']
        x_list = data['X'] 
        y_list = data['Y']
    # data X <-> DS9 x <-> col
    # data Y <-> DS9 y <-> row
    #x_min = header['TLMIN6'] x_min = 1
    #y_min = header['TLMIN7'] y_min = 1
    col_list = x_list - 1 # y_min
    row_list = y_list - 1 # x_min
    return time_list, col_list, row_list, header

def time_histogram(evt_file_path: Union[str, pathlib.Path]):
    data = fits.getdata(evt_file_path,ext=1)
    times = data['TIME']
    plt.hist(times, bins=100)
    plt.xlabel('Time (s)')
    plt.ylabel('Count rate')
    plt.show()

def create_wcs_from_header(header: fits.Header) -> WCS:
    """
    从header创建WCS对象
    
    Parameters
    ----------
    header : fits.Header
        FITS头文件
        
    Returns
    -------
    WCS
        WCS对象
    """
    wcs_obj = WCS(naxis=2)
    wcs_obj.wcs.crpix = [header['TCRPX6'], header['TCRPX7']]
    wcs_obj.wcs.cdelt = [header['TCDLT6'], header['TCDLT7']]
    wcs_obj.wcs.crval = [header['TCRVL6'], header['TCRVL7']]
    wcs_obj.wcs.ctype = [header['TCTYP6'], header['TCTYP7']]
    return wcs_obj

def ttime_to_utctime(ttime: float, mjdrefi: float, mjdreff: float) -> Time:
    mjd = mjdrefi + mjdreff + ttime / 86400
    return Time(mjd, format='mjd', scale='tt').utc

def slice_time(header: fits.Header, 
               group_number: int) -> Tuple[List[Time], np.ndarray]:
    """
    Parameters
    ----------
    header : fits.Header
        FITS头文件
    group_number : int
        分组数量
        
    Returns
    -------
    tuple
        (date_list, time_array) - 时间点列表和时间数组
    """
    time_start = header['TSTART']
    time_end = header['TSTOP']
    mjdrefi = header['MJDREFI']
    mjdreff = header['MJDREFF']
    
    time_utc = []
    time_raw = []
    
    time_duration = time_end - time_start
    for i in range(group_number + 1):
        new_time_float = time_start + (i / group_number) * time_duration
        new_time_time = ttime_to_utctime(new_time_float, mjdrefi, mjdreff)
        time_raw.append(new_time_float)
        time_utc.append(new_time_time)
    return time_utc, time_raw

def get_ephemeris_batch(times, target_id, location, orbital_keywords, batch_size):
    """
    分批获取历表数据的辅助函数
    Parameters
    ----------
    times : array-like
        时间点列表
    orbital_keywords : list
        需要获取的轨道参数关键字列表
    batch_size : int
        每批处理的时间点数量
    Returns
    -------
    Ephem 
        合并后的历表数据，格式与直接调用eph[orbital_keywords]相同
    """
    results = []  # 存储所有批次的结果
    # 分批处理
    for i in range(0, len(times), batch_size):
        try:
            batch_times = times[i:min(i + batch_size, len(times))]
            eph = Ephem.from_horizons(target_id, location=location, epochs=batch_times)
            results.append(eph)
        except Exception as e:
            if "Ambiguous target name" in str(e):
                print(f"请提供准确的目标ID。错误: {str(e)}")
            else:
                print(f"错误: {str(e)}")
            raise
    # 如果只有一批数据，直接返回
    if len(results) == 1:
        return results[0][orbital_keywords]
    # 使用sbpy的vstack合并结果
    final_eph = results[0]
    for eph in results[1:]:
        final_eph.vstack(eph)
    return final_eph[orbital_keywords]

def get_target_positions(time_utc: List[Time], target_id: Union[str, int], wcs: WCS) -> Tuple[List[float], List[float]]:
    """
    获取彗星在各时间段的位置
    
    Parameters
    ----------
    date_list : List[Time]
        时间点列表
    target_id : str or int
        彗星的JPL Horizons ID
    wcs : WCS
        WCS对象
        
    Returns
    -------
    tuple
        (x_positions, y_positions) - 彗星的像素坐标列表
    """
    num_segments = len(time_utc) - 1
    col_positions = []
    row_positions = []
    middle_time_list = []
    for i in range(num_segments):
        start_time = time_utc[i]
        end_time = time_utc[i + 1]
        middle_time = start_time + 0.5 * (end_time - start_time)
        middle_time_list.append(middle_time.jd)
    eph = get_ephemeris_batch(Time(middle_time_list, format='jd'), target_id, '@swift', ['RA', 'DEC'], 50)
    #obj = Horizons(id=target_id, location='@swift', epochs=middle_time_list)
    #eph = obj.ephemerides()
    ra_list = eph['RA'].value
    dec_list = eph['DEC'].value
    ra_list, dec_list = icrf_to_fk5(ra_list, dec_list)

    for ra, dec in zip(ra_list, dec_list):
        x, y = wcs.wcs_world2pix(ra, dec, 1) # starting from 1 -> get the position in the event list
        # can be tested with header['TCRPX6'], header['TCRPX7'] and header['TCRVL6'], header['TCRVL7']
        # ra = 37.90958, dec = -20.7373 <-> 2000.5, 2000.5
        col, row = DS9Converter.ds9_to_coords(x, y)
        col_positions.append(col)
        row_positions.append(row)
    return col_positions, row_positions

def create_image_from_events(events_col: np.ndarray, events_row: np.ndarray) -> np.ndarray:
    """
    从事件创建图像
    
    Parameters
    ----------
    events_col, events_row : np.ndarray
        事件的坐标
    center_position: Tuple[float, float]
        图像中心位置 (col, row)
        
    Returns
    -------
    np.ndarray
        创建的图像
    """
    # 4000, 4000 = hdr['TLMAX6'], hdr['TLMAX7']
    # 0, 0 = hdr['TLMIN6'], hdr['TLMIN7']
    # img_shape = (4000-0, 4000-0)
    img_shape = (4000, 4000) # col, row
    edges_col = np.arange(0-0.5, img_shape[0]-1 + 0.5 + 1, 1) # img_shape[0]-1 is the max col in python; + 0.5 is the edge; +1 is for np.arange
    edges_row = np.arange(0-0.5, img_shape[1]-1 + 0.5 + 1, 1) 

    image, _, _ = np.histogram2d(events_col, events_row, bins=(edges_col, edges_row))
    image = image.T
    
    return image

def save_single_evt_image(evt_file_path: Union[str, pathlib.Path],
                          output_path: Union[str, pathlib.Path]):
    """
    保存单个事件图像
    """
    _, col_list, row_list, header = read_event_file(evt_file_path)
    image = create_image_from_events(col_list, row_list)
    wcs = create_wcs_from_header(header)
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header.extend(header)
    primary_hdu.header.update(wcs.to_header())
    hdu_img = fits.ImageHDU(image, name='IMAGE')
    hdu_img.header.extend(header)
    hdu_img.header.update(wcs.to_header())
    hdul = fits.HDUList([primary_hdu, hdu_img])
    hdul.writeto(output_path, overwrite=True)
    print(f'Saved to {output_path}')

def reduce_motion_smearing(evt_file_path: Union[str, pathlib.Path],
                           exp_file_path: Union[str, pathlib.Path],
                           target_coord: Tuple[float, float],
                           group_number: int,
                           target_id: Union[str, int],
                           stack_method: str = 'sum',
                           binby2: bool = False,) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    减少运动拖尾的主要函数
    
    Parameters
    ----------
    evt_file_path : str
        事件文件路径
    target_id : str or int
        彗星的JPL Horizons ID
    group_number : int, optional
        分组数量
    stack_method : str
        叠加方法：'median' (用count_rate) 或 'sum' (用count)
    target_coord : tuple, optional
        col, row
        
    Returns
    -------
    tuple
        (stacked_image, stacked_err, stacked_exp, processing_info) - 叠加后的图像、误差图、曝光图和处理信息
    """
    # 读取文件和exp
    time_list, col_list, row_list, header = read_event_file(evt_file_path)
    #total_exposure = header['EXPOSURE']
    exp_data = fits.getdata(exp_file_path, ext=1)
    exp_header = fits.getheader(exp_file_path, ext=1)
    total_exposure = np.max(exp_data)
        
    # 创建WCS
    wcs = create_wcs_from_header(header)
    wcs_exp = WCS(exp_header)
    
    # 分割时间
    time_utc, time_raw = slice_time(header, group_number)
        
    # 获取彗星位置
    target_col_list, target_row_list = get_target_positions(time_utc, target_id, wcs)
    target_col_list_exp, target_row_list_exp = get_target_positions(time_utc, target_id, wcs_exp)

    if binby2:
        target_col_list = [col // 2 for col in target_col_list]
        target_row_list = [row // 2 for row in target_row_list]
        target_col_list_exp = [col // 2 for col in target_col_list_exp]
        target_row_list_exp = [row // 2 for row in target_row_list_exp]
        exp_data = block_reduce(exp_data, block_size=2, func=np.nanmean)
    
    # 为每个时间段创建图像
    num_segments = len(time_utc) - 1
    event_nums_total = len(time_list)
    segment_elapsed_time = time_raw[1] - time_raw[0]
    segment_images = []
    segment_errs = []
    segment_exp_maps = []
    target_positions = []
    target_positions_exp = []
    event_nums = []
    segment_exp_times = []
    
    for i in range(num_segments):
        # 过滤事件
        mask = (time_list >= time_raw[i]) & (time_list < time_raw[i+1])
        events_col = col_list[mask]
        events_row = row_list[mask]
        event_nums.append(len(events_col))
        exp_time_ratio = len(events_col)/event_nums_total
        segment_exp_times.append(exp_time_ratio*total_exposure)
            
        # 创建图像
        segment_image = create_image_from_events(events_col, events_row)
        if binby2:
            segment_image = block_reduce(segment_image, block_size=2, func=np.nansum)
        segment_images.append(segment_image)
        segment_errs.append(np.sqrt(segment_image))
        target_positions.append((target_col_list[i], target_row_list[i]))
        target_positions_exp.append((target_col_list_exp[i], target_row_list_exp[i]))
        segment_exp_maps.append(exp_data*exp_time_ratio)
    event_nums_ratios = [num/np.min(event_nums) for num in event_nums]
    #print('Event number in each segment: '+', '.join(f'{x}' for x in event_nums))
    #print('Ratios of the event numbers: '+', '.join(f'{x:.2f}' for x in event_nums_ratios))
    # 使用现有的对齐功能
    aligned_images, aligned_errs = align_images(segment_images, target_positions, 
                                                target_coord,
                                                fill_value=np.nan,
                                                image_err=segment_errs)
    
    aligned_exp_map = align_images(segment_exp_maps, target_positions_exp, 
                                   target_coord,
                                   fill_value=0)
    
    # 根据叠加模式选择叠加方法
    if stack_method == 'median':
        with np.errstate(divide='ignore', invalid='ignore'):
            aligned_images = [img/exp for img, exp in zip(aligned_images, aligned_exp_map)]
            aligned_errs = [err/exp for err, exp in zip(aligned_errs, aligned_exp_map)]
    
    stacked_image, stacked_err = stack_images(aligned_images, method=stack_method, image_err=aligned_errs, verbose=False)
    stacked_exp = stack_images(aligned_exp_map, method='sum', verbose=False)
    stacked_image[np.isinf(stacked_image)] = np.nan
    stacked_err[np.isinf(stacked_err)] = np.nan
    stacked_image[stacked_exp/np.max(stacked_exp) < 0.99] = np.nan
    stacked_err[stacked_exp/np.max(stacked_exp) < 0.99] = np.nan
    if binby2:
        platescale = 0.502*2
    else:
        platescale = 0.502
    # 处理信息
    processing_info = {
        'num_segments': num_segments,
        'stack_method': stack_method,
        'total_exposure': total_exposure,
        'segment_exposures': total_exposure/group_number,
        'target_coord': target_coord,
        'binby2': binby2,
        'platescale': platescale,
        'event_nums': event_nums,
        'event_nums_ratios': event_nums_ratios,
        'segment_exp_times': segment_exp_times,
        'segment_elapsed_time':segment_elapsed_time
    }
    
    return stacked_image, stacked_err, stacked_exp, processing_info, header, aligned_images, aligned_errs, aligned_exp_map

def bin_evt_image(image, image_err=None, image_exp=None, processing_info=None):
    """
    对事件图像进行2x2 binning
    """
    if image_err is None:
        binned_image = bin_image(image, block_size=2, method='sum')
        binned_err = None
    else:
        binned_image, binned_err = bin_image(image, block_size=2, image_err=image_err, method='sum')
    if image_exp is not None:
        binned_exp = bin_image(image_exp, block_size=2, method='mean')
    else:
        binned_exp = None
    if processing_info is not None:
        col, row = processing_info['target_coord']
        new_col = col // 2
        new_row = row // 2
        processing_info['target_coord'] = (new_col, new_row)
    return binned_image, binned_err, binned_exp, processing_info

def save_smear_reduced_result(stacked_image: np.ndarray, stacked_err: np.ndarray,
                              exposure_map: np.ndarray, 
                              output_path: str, evt_hdr: fits.Header, 
                              processing_info: dict,
                              individual_images_to_save: dict,
                              image_unit_save: str = 'count'):
    """
    保存减少拖尾后的结果到FITS文件
    
    Parameters
    ----------
    stacked_image : np.ndarray
        叠加后的图像
    exposure_map : np.ndarray
        曝光图
    output_path : str
        输出文件路径
    header : fits.Header, optional
        原始FITS头文件
    processing_info : dict, optional
        处理信息字典
    individual_images_to_save : dict
        dict = {'IMAGE': [], 'ERROR': [], 'EXPOSURE': []}
    """
    # 创建header
    primary_hdu = fits.PrimaryHDU()
    hdr = primary_hdu.header
    
    if evt_hdr is not None:
        # 计算中间时间
        date_start = Time(evt_hdr['DATE-OBS'])
        date_end = Time(evt_hdr['DATE-END'])
        mid_time = date_start + 0.5 * (date_end - date_start)
        hdr['MID_TIME'] = str(mid_time.isot)
        
    hdr['PLATESCL'] = (0.502, 'arcsec/pixel')
    
    # 添加处理信息到header
    if processing_info is not None:
        hdr['NSEGMENT'] = (processing_info['num_segments'], 'Total number of time segments')
        hdr['STACKMTH'] = (processing_info['stack_method'], 'Stacking method')
        stack_method = processing_info['stack_method']
        # 如果是count模式，添加特殊标记
        if image_unit_save == 'count':
            hdr['BUNIT'] = ('count', 'Image units are counts (not count rate)')
        elif image_unit_save == 'count/s':
            hdr['BUNIT'] = ('count/s', 'Image units are count rate')
        else:
            raise ValueError(f"Invalid image unit: {image_unit_save}")
        if processing_info['binby2']:
            hdr['BINNED'] = ('True', 'Image is binned by 2x2')
        else:
            hdr['BINNED'] = 'False'
        hdr['PLATESCL'] = (processing_info['platescale'], 'Platescale in arcsec/pixel')
        hdr['TOTEXP'] = (processing_info['total_exposure'], 'Total exposure time')
        hdr['SEGELAP'] = (processing_info['segment_elapsed_time'], 'Segment elapsed time')
        hdr['COLPIXEL'] = (processing_info['target_coord'][0], 'Target X position in Python coordinates')
        hdr['ROWPIXEL'] = (processing_info['target_coord'][1], 'Target Y position in Python coordinates')
        hdr['DS9XPIX'] = (processing_info['target_coord'][0] + 1, 'Target X position in DS9 coordinates')
        hdr['DS9YPIX'] = (processing_info['target_coord'][1] + 1, 'Target Y position in DS9 coordinates')
        hdr.add_comment('Event number in each segment: '+', '.join(f'{x}' for x in processing_info['event_nums']))
        #hdr.add_comment('Ratio of event number relative to the minimum: '+', '.join(f'{x:.2f}' for x in processing_info['event_nums_ratios']))
        hdr.add_comment('Exposure time (s) in each segment: '+', '.join(f'{x:.2f}' for x in processing_info['segment_exp_times']))
    hdr['HISTORY'] = f'Created by Zexi Xing'
    hdr['REDUCER'] = 'motion_smear_reducer.py'
    # 创建HDU
    if stack_method != 'sum' and image_unit_save == 'count':
        stacked_image = stacked_image * exposure_map
        stacked_err = stacked_err * exposure_map
    elif stack_method == 'sum' and image_unit_save == 'count/s':
        stacked_image = stacked_image / exposure_map
        stacked_err = stacked_err / exposure_map
    else:
        pass
    hdu_img = fits.ImageHDU(stacked_image, name='IMAGE')
    hdu_err = fits.ImageHDU(stacked_err, name='ERROR')
    hdu_exp = fits.ImageHDU(exposure_map, name='EXPOSURE')
    hdr[f'EXT{1}NAME'] = ('IMAGE', f'Name of extension {1}')
    hdr[f'EXT{2}NAME'] = ('ERROR', f'Name of extension {2}')
    hdr[f'EXT{3}NAME'] = ('EXPOSURE', f'Name of extension {3}')
    hdul = fits.HDUList([primary_hdu, hdu_img, hdu_err, hdu_exp])

    if individual_images_to_save is not None:
        for key, value in individual_images_to_save.items():
            num_images = len(value)
            for i in range(num_images):
                hdu_img = fits.ImageHDU(value[i], name=f'{key}_{i}')
                ext_name_key = f'EXT{len(hdul)+1}NAME'
                if len(ext_name_key) > 8:
                    ext_name_key = ext_name_key[:8]
                hdr[ext_name_key] = (f'{key}_{i}', f'Name of extension {len(hdul)}')
                hdul.append(hdu_img)
    hdul.writeto(output_path, overwrite=True)
    print(f'Saved to {output_path}')

def reduce_smear(evt_file_path, exp_file_path, target_id, target_coord, group_number, stack_method, 
                 output_path = None, binby2=True, save_individual_images=False, image_unit_save='count'):
    """
    Reduce motion smear for a single observation
    Parameters
    ----------
    evt_file_path : str
        Event file path
    exp_file_path : str
        Exposure file path
    target_id : str
        Target ID
    target_coord : tuple
        Target coordinate
    group_number : int
        Group number
    stack_method : str
        Stack method, 'sum' or 'median'
    output_path : str
        Output file path
    binby2 : bool
        Whether to bin by 2x2
    save_individual_images : bool
        Whether to save individual images
    """
    stacked_image, stacked_err, stacked_exp, processing_info, header, aligned_images, aligned_errs, aligned_exp_map = \
        reduce_motion_smearing(evt_file_path, exp_file_path, target_coord, group_number, target_id, stack_method, binby2=binby2)
    if output_path is None:
        return stacked_image, stacked_err, stacked_exp, processing_info, header, aligned_images, aligned_errs, aligned_exp_map
    else:
        if save_individual_images:
            individual_images_to_save = {'IMAGE': aligned_images, 'ERROR': aligned_errs, 'EXPOSURE': aligned_exp_map}
        else:
            individual_images_to_save = None
        save_smear_reduced_result(stacked_image, stacked_err, stacked_exp, output_path, header, \
                                  processing_info, individual_images_to_save, image_unit_save=image_unit_save)

# 使用示例
import matplotlib.pyplot as plt
if __name__ == "__main__":
    """
    使用示例
    """
    #evt_file_path = '/Users/zexixing/Downloads/sw00094421002uvvw1po_uf.evt.gz'
    #exp_file_path = '/Users/zexixing/Downloads/sw00094421002uvv_ex.img.gz'
    #target_id = '90000548'
    #target_coord = (500, 500) 
    #group_number=2
    #stack_method='sum'
    #stacked_image, stacked_err, stacked_exp, processing_info, header, aligned_images, aligned_errs, aligned_exp_map = \
    #    reduce_motion_smearing(evt_file_path, exp_file_path, target_coord, group_number, target_id, stack_method, binby2=True)
    ##plt.imshow(stacked_err, origin='lower', vmax=20, vmin=0)
    ##plt.plot(processing_info['target_coord'][0], processing_info['target_coord'][1], 'ro')
    ##plt.show()
    ##plt.imshow(stacked_exp, origin='lower', vmax=418, vmin=0)
    ##plt.show()
    ##binned_image, binned_err, binned_exp, processing_info2 = bin_evt_image(stacked_image, stacked_err, stacked_exp, processing_info)
    ##plt.imshow(binned_err, origin='lower', vmax=2, vmin=0)
    ##plt.plot(processing_info2['target_coord'][0], processing_info2['target_coord'][1], 'ro')
    ##plt.show()
    #output_path = '/Users/zexixing/Downloads/test.fits'
    #individual_images_to_save = {'IMAGE': aligned_images, 'ERROR': aligned_errs, 'EXPOSURE': aligned_exp_map}
    #save_smear_reduced_result(stacked_image, stacked_err, stacked_exp, output_path, header, processing_info, individual_images_to_save)
    ##time_histogram(evt_file_path)
    evt_file_path = '/Users/zexixing/Library/CloudStorage/OneDrive-Personal/ZexiWork/data/Swift/C_2025N1/05000549001/uvot/event/sw05000549001uw1w1po_uf.evt.gz'  
    output_path = '/Users/zexixing/Downloads/test.fits'
    save_single_evt_image(evt_file_path, output_path)