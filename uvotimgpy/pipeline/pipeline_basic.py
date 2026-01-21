import os
import shutil
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import math
from regions import PixelRegion
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.nddata import block_reduce
from astroquery.skyview import SkyView
from synphot import SourceSpectrum, SpectralElement

import matplotlib.pyplot as plt

from uvotimgpy.config import paths
from uvotimgpy.query import SkyImageFetcher
from uvotimgpy.base.math_tools import UnitConverter
from uvotimgpy.base.file_and_table import TableConverter, target_name_converter
from uvotimgpy.base.visualizer import multi_show, draw_direction_compass, MaskInspector, draw_scalebar
from uvotimgpy.base.instruments import normalize_filter_name, get_effective_area
from uvotimgpy.utils.image_operation import DS9Converter, crop_image, stack_images
from uvotimgpy.utils.spectrum_operation import SolarSpectrum
from uvotimgpy.uvot_file.file_organization import ObservationLogLoader, ObservationLogger
from uvotimgpy.uvot_file.file_io import save_stacked_fits
from uvotimgpy.uvot_image.motion_smear_reducer import reduce_smear
from uvotimgpy.uvot_image.star_cleaner import StarCleaner, save_starmask, delete_starmask, save_filled
from uvotimgpy.uvot_image.image_correction import correct_offset_in_image, correct_coi_loss_in_image, get_coi_loss_map


# ===================== 工具函数 =====================
def is_path_like(string):
    """判断字符串是否看起来像路径"""
    try:
        # 尝试创建Path对象
        #path = Path(string)
        
        # 检查是否包含路径分隔符或看起来像路径
        if '/' in string or '\\' in string:
            return True
        
        # 检查是否有文件扩展名
        #if path.suffix:
        #    return True
            
        # 检查是否是已知的路径模式
        #if string in ['.', '..', '~']:
        #    return True
            
        return False
    except:
        return False

def load_path(path_or_name: Union[str, Path, None], parent_path: Union[str, Path, None] = None) -> Path:
    """统一的路径加载函数"""
    if isinstance(path_or_name, str) and not is_path_like(path_or_name):
        if parent_path is None:
            raise ValueError("Parent path is required when path is ")
        else:
            return paths.get_subpath(parent_path, path_or_name)
    if isinstance(path_or_name, str) and is_path_like(path_or_name):
        return Path(path_or_name)
    elif isinstance(path_or_name, Path):
        return path_or_name
    else:
        raise ValueError("Invalid path_or_name")
    
def table_to_list(table: Union[pd.DataFrame, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if isinstance(table, pd.DataFrame):
        return TableConverter.df_to_dict_list(table)
    elif isinstance(table, List[Dict[str, Any]]):
        return table
    else:
        raise ValueError("Invalid table")
    
def table_to_df(table: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table
    elif isinstance(table, List[Dict[str, Any]]):
        return TableConverter.dict_list_to_df(table)
    else:
        raise ValueError("Invalid table")
    
def get_obs_path(data_path, obsid: str, filt: str, return_type: str = 'sk', datatype: str = 'image') -> Path:
    filt_filename = normalize_filter_name(filt, output_format='filename')
    obs_path = paths.get_subpath(data_path, f'{obsid}')
    if return_type == 'sk':
        return paths.get_subpath(obs_path, 'uvot', 'image', f'sw{obsid}{filt_filename}_sk.img.gz')
    elif return_type == 'exp':
        return paths.get_subpath(obs_path, 'uvot', 'image', f'sw{obsid}{filt_filename}_ex.img.gz')
    elif return_type == 'evt':
        if datatype == 'image':
            evt_file_path = None
        else:
            evt_file_path = paths.get_subpath(obs_path, 'uvot', 'event', f'sw{obsid}{filt_filename}w1po_uf.evt.gz')
            if not os.path.exists(evt_file_path):
                evt_file_path = paths.get_subpath(obs_path, 'uvot', 'event', f'sw{obsid}{filt_filename}wupo_uf.evt.gz')
                if not os.path.exists(evt_file_path):
                    print(f"Event file not found: {evt_file_path}")
                    evt_file_path = None
            return evt_file_path
    else:
        raise ValueError("Invalid return_type")

# ===================== 准备环节 =====================
@dataclass
class BasicInfo:
    """基本信息"""
    instrument: str
    data_folder_name: str
    target_simplified_name: str
    target_full_name: str
    project_path: Path
    project_docs_path: Path
    #project_name: str
    data_path: Path
    evt_to_img_folder_path: Path
    alignment_folder_path: Path
    cleaned_folder_path: Path
    stacked_folder_path: Path
    evt_to_img_name_style: str
    alignment_name_style: str
    cleaned_name_style: str
    stacked_name_style: str
    target_id: str
    obsid_initial: str
    obsid_final: str
    epoch_name: str
    orbital_keywords: List[str]
    observation_log_path: Path
    observation_info: Dict[str, Any]
    filt_filename_v: str = 'uvv'
    filt_filename_uw1: str = 'uw1'
    filt_display_v: str = 'V'
    filt_display_uw1: str = 'UVW1'
    obs_time: Optional[Time] = None
    effective_wave_uv: u.Quantity = 3325.72*u.AA
    effective_wave_v: u.Quantity = 5437.83*u.AA
    central_wave: u.Quantity = 4381.775*u.AA
    area: u.Quantity = np.pi*15*15 * u.cm*u.cm
    target_coord: Tuple[Union[int, float], Union[int, float]] = (1000, 1000)
    sun: SourceSpectrum = SolarSpectrum.from_model()

    def __post_init__(self):
        """在初始化后自动计算 obs_time"""
        if self.obs_time is None:
            self.obs_time = Time(self.observation_info['mid_time'])
        self.bandpass_v: SpectralElement = get_effective_area(self.filt_filename_v, transmission=True, bandpass=True, obs_time=self.obs_time)
        self.bandpass_uw1: SpectralElement = get_effective_area(self.filt_filename_uw1, transmission=True, bandpass=True, obs_time=self.obs_time)
        self.filt_dict = {
                'uvv': {
                    'filename': 'uvv',
                    'display': 'V',
                    'bandpass': self.bandpass_v,
                },
                'uw1': {
                    'filename': 'uw1',
                    'display': 'UVW1',
                    'bandpass': self.bandpass_uw1,
                },
            }
        self.km_per_arcsec = UnitConverter.arcsec_to_km(1.0, self.observation_info['delta'])
    
class DataPreparation:
    """数据准备相关功能"""
    
    def __init__(self, target_name: str, project_path_or_name: Union[str, Path], target_id: Union[str, int], 
                 obsid_initial: Optional[str] = None, obsid_final: Optional[str] = None, epoch_name: Optional[str] = None,
                 data_path: Union[str, Path, None] = None,
                 evt_to_img_folder: Optional[Union[str, Path]] = 'evt_to_img', evt_to_img_name_style: Optional[str] = None,
                 alignment_folder: Optional[Union[str, Path]] = 'alignment', alignment_name_style: Optional[str] = None,
                 cleaned_folder: Optional[Union[str, Path]] = 'cleaned', cleaned_name_style: Optional[str] = None,
                 stacked_folder: Optional[Union[str, Path]] = 'stacked', stacked_name_style: Optional[str] = None,
                 ):
        """设置基本信息"""
        name_dict = target_name_converter(target_name)
        self.instrument = 'Swift'
        self.data_folder_name = name_dict['data_folder_name']
        self.target_simplified_name = name_dict['target_simplified_name']
        self.target_full_name = name_dict['target_full_name']
        self.project_path = load_path(project_path_or_name, parent_path=paths.projects)
        self.project_docs_path = paths.get_subpath(self.project_path, 'docs')
        #self.project_name = self.project_path.name

        if data_path is None:
            self.data_path = paths.get_subpath(paths.data, self.instrument, self.data_folder_name)
        else:
            self.data_path = load_path(data_path, parent_path=paths.get_subpath(paths.data, self.instrument))

        self.target_id = str(target_id)
        self.observation_log_path = None
        self.observation_loader = None
        self.all_observations_df = None
        self.observations_df = None
        self.observations_v_df = None
        self.observations_uw1_df = None
        self.observations_df_epoch = None
        self.observations_v_df_epoch = None
        self.observations_uw1_df_epoch = None
        #self.observations_list = None
        #self.observations_v_list = None
        #self.observations_uw1_list = None
        self.obsid_initial = obsid_initial
        self.obsid_final = obsid_final
        self.epoch_name = epoch_name
        self.delete_list_v = []
        self.delete_list_uw1 = []
        self.observation_info = {}
        # 存储其他可能的参数
        # target_name, 
        self.orbital_keywords = ['RA', 'DEC', 'RA*cos(Dec)_rate', 'DEC_rate', 'delta', 'r', 'r_rate', \
                                 'elong', 'alpha', 'sunTargetPA', 'velocityPA', 'Sky_motion', 'Sky_mot_PA']
        self.evt_to_img_folder_path = load_path(evt_to_img_folder, self.project_path)
        self.alignment_folder_path = load_path(alignment_folder, self.project_path)
        self.cleaned_folder_path = load_path(cleaned_folder, self.project_path)
        self.stacked_folder_path = load_path(stacked_folder, self.project_path)
        self.evt_to_img_name_style = '{obsid}_{ext_no}_{filt_filename}.fits' if evt_to_img_name_style is None else evt_to_img_name_style
        self.alignment_name_style = '{obsid}_{ext_no}_{filt_filename}.fits' if alignment_name_style is None else alignment_name_style
        self.cleaned_name_style = '{obsid}_{ext_no}_{filt_filename}.fits' if cleaned_name_style is None else cleaned_name_style
        self.stacked_name_style = '{epoch_name}_{filt_filename}.fits' if stacked_name_style is None else stacked_name_style

    def create_observation_log(self, output_path_or_name: Union[str, Path], orbital_keywords = None) -> pd.DataFrame:
        """创建observation log"""
        # 实现创建逻辑
        logger = ObservationLogger(self.data_folder_name,data_root_path=paths.get_subpath(paths.data, self.instrument), target_alternate=self.target_id)
        if orbital_keywords is not None:
            self.orbital_keywords = orbital_keywords
        orbital_keywords_to_get_log = [keyword for keyword in self.orbital_keywords if keyword not in ['Sky_motion', 'Sky_mot_PA']]
        self.observation_log_path = load_path(output_path_or_name, parent_path=self.project_docs_path)
        logger.process_data(output_path=self.observation_log_path, orbital_keywords=orbital_keywords_to_get_log)
    
    def load_observation_log(self, log_path_or_name: Union[str, Path, None] = None) -> pd.DataFrame: # TODO: sort by OBSID
        """读入observation log"""
        if log_path_or_name is not None:
            self.observation_log_path = load_path(log_path_or_name, parent_path=self.project_docs_path)
        self.observation_loader = ObservationLogLoader(self.observation_log_path)
        self.all_observations_df = self.observation_loader.get_all_data()
        self.observations_df = self.observation_loader.where(f'OBSID >= "{self.obsid_initial}"', f'OBSID <= "{self.obsid_final}"')
        self.observations_v_df = self.observation_loader.where('FILTER == "V"', df=self.observations_df)
        self.observations_uw1_df = self.observation_loader.where('FILTER == "UVW1"', df=self.observations_df)
        self.observations_df_epoch = self.observations_df.copy()
        self.observations_v_df_epoch = self.observations_v_df.copy()
        self.observations_uw1_df_epoch = self.observations_uw1_df.copy()
        #self.observations_list = TableConverter.df_to_dict_list(self.observations_df)
        #self.observations_v_list = TableConverter.df_to_dict_list(self.observations_v_df)
        #self.observations_uw1_list = TableConverter.df_to_dict_list(self.observations_uw1_df)

    def delete_observations(self, delete_list_v: Union[List[str], None] = None, delete_list_uw1: Union[List[str], None] = None):
        # delete_list_v = ['05000525001_1', '05000525001_2', '05000525002_1'], obsid_extension
        if delete_list_v is None:
            delete_list_v = self.delete_list_v
        else:
            self.delete_list_v = list(dict.fromkeys(self.delete_list_v + delete_list_v)) # merge and remove duplicates
        if len(delete_list_v) > 0:
            criteria_v = []
            for obsid_extension in delete_list_v:
                obsid = obsid_extension.split('_')[0]
                extension = obsid_extension.split('_')[1]
                criteria = f'not (OBSID == "{obsid}" and EXT_NO == {extension})'
                criteria_v.append(criteria)
            criteria_v = tuple(criteria_v)
            self.observations_v_df = self.observation_loader.where(*criteria_v, df=self.observations_v_df)
            self.observations_df = self.observation_loader.where(*criteria_v, df=self.observations_df)
            #self.observations_v_list = TableConverter.df_to_dict_list(self.observations_v_df)
        if delete_list_uw1 is None:
            delete_list_uw1 = self.delete_list_uw1
        else:
            self.delete_list_uw1 = list(dict.fromkeys(self.delete_list_uw1 + delete_list_uw1))
        if len(delete_list_uw1) > 0:
            criteria_uw1 = []
            for obsid_extension in delete_list_uw1:
                obsid = obsid_extension.split('_')[0]
                extension = obsid_extension.split('_')[1]
                criteria = f'not (OBSID == "{obsid}" and EXT_NO == {extension})'
                criteria_uw1.append(criteria)
            criteria_uw1 = tuple(criteria_uw1)
            self.observations_uw1_df = self.observation_loader.where(*criteria_uw1, df=self.observations_uw1_df)
            self.observations_df = self.observation_loader.where(*criteria_uw1, df=self.observations_df)
            #self.observations_uw1_list = TableConverter.df_to_dict_list(self.observations_uw1_df)
        #self.observations_list = TableConverter.df_to_dict_list(self.observations_df)
        
    def display_observations(self, observations: Union[pd.DataFrame, List[Dict[str, Any]]],  
                             radius=20, vrange=None, max_cols=4, binby2=False, image_unit_show='count',
                             compare_with_dss: bool = False, scale: float = 1.004, dss_scale: Optional[float] = None):
        """
        展示所有筛选出的observation图像
        dss_scale: arcsec/pixel in dss image; default makes 100 pixels in radius
        """
        # 实现展示逻辑
        image_list = []
        xrange_list = []
        yrange_list = []
        title_list = []
        target_position_range = []
        mean_sunTargetPA = observations['sunTargetPA'].mean()
        mean_velocityPA = observations['velocityPA'].mean()
        observations = table_to_list(observations)
        if compare_with_dss:
            vrange_list = []
        for i, obs in enumerate(observations):
            target_col, target_row = DS9Converter.ds9_to_coords(obs['x_pixel'], obs['y_pixel'])
            img_path = get_obs_path(self.data_path, obs['OBSID'], obs['FILTER'], return_type='sk', datatype=obs['DATATYPE'])
            img = fits.getdata(img_path, ext = obs['EXT_NO'])
            if binby2:
                img = block_reduce(img, block_size=2, func=np.nanmean)
                target_col = target_col // 2
                target_row = target_row // 2
            if image_unit_show == 'count/s':
                image_list.append(img/obs['EXPOSURE'])
            elif image_unit_show == 'count':
                image_list.append(img)
            else:
                raise ValueError(f"Invalid image_unit_show: {image_unit_show}")
            xrange = (target_col - radius, target_col + radius)
            yrange = (target_row - radius, target_row + radius)
            xrange_list.append(xrange)
            yrange_list.append(yrange)
            title_list.append(f'{obs["OBSID"]} {obs["EXT_NO"]} {obs["FILTER"]}')
            target_position_range.append((target_col, target_row))
            if compare_with_dss:
                if vrange is None:
                    vrange_list.append((None, None))
                elif isinstance(vrange, list) and isinstance(vrange[0], tuple):
                    vrange_list.append(vrange[i])
                else:
                    vrange_list.append(vrange)
                if dss_scale is None:
                    dss_scale = radius*scale/100
                dss_img = SkyImageFetcher.from_hips(obs['RA'], obs['DEC'], radius_arcsec=radius*scale, scale=dss_scale)
                image_list.append(dss_img)
                xrange_list.append((None, None))
                yrange_list.append((None, None))
                vrange_list.append((None, None))
                dss_pixel = int(radius*scale/dss_scale)
                target_position_range.append((dss_pixel, dss_pixel))
                title_list.append(f'{obs["OBSID"]} {obs["EXT_NO"]} DSS')
        if compare_with_dss:
            vrange = vrange_list
        fig, axes = multi_show(image_list, max_cols=max_cols, vrange=vrange, xrange=xrange_list, yrange=yrange_list, 
                               title_list=title_list, target_position_range=target_position_range)

        # draw direction compass
        #if len(image_list) > max_cols:
        #    ax0 = axes[0][0]
        #else:
        #    ax0 = axes[0]
        ax0 = axes[0][0]
        draw_direction_compass(ax0, 
                               directions={'N': 0, 'E': 90, 'v': mean_velocityPA-180, '☉': mean_sunTargetPA-180},
                               colors='white',
                               position=(0.15, 0.85), 
                               arrow_length=0.08, 
                               text_offset=0.02, 
                               fontsize=10)
        plt.show(block=True)
        plt.close()
    
    def open_in_ds9(self, observations: Optional[pd.DataFrame] = None, **kwargs):
        """用ds9打开图像"""
        if observations is None:
            observations = self.filtered_observations
        # 实现ds9打开逻辑n
        pass

    def get_observation_info(self, observations: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
        """获取observation list的统计信息"""
        self.observation_info = {}
        if observations is None and self.observations_v_df is not None:
            self.observation_info['v_exposure_time'] = self.observations_v_df['EXPOSURE'].sum()
            self.observation_info['v_elapsed_time'] = self.observations_v_df['TELAPSE'].sum()
        if observations is None and self.observations_uw1_df is not None:
            self.observation_info['uw1_exposure_time'] = self.observations_uw1_df['EXPOSURE'].sum()
            self.observation_info['uw1_elapsed_time'] = self.observations_uw1_df['TELAPSE'].sum()
        if observations is None:
            observations = self.observations_df
        else:
            observations = table_to_df(observations)
        # 计算各种统计信息
        start_time = Time(observations['DATE_OBS'].min())
        end_time = Time(observations['DATE_END'].max())
        mid_time = start_time + (end_time - start_time) / 2
        self.observation_info['start_time'] = start_time.isot
        self.observation_info['end_time'] = end_time.isot
        self.observation_info['mid_time'] = mid_time.isot
        self.observation_info['total_exposure_time'] = observations['EXPOSURE'].sum()
        self.observation_info['total_elapsed_time'] = observations['TELAPSE'].sum()
        for keyword in self.orbital_keywords:
            self.observation_info[f'{keyword}'] = observations[keyword].mean()
    
    def get_basic_info(self) -> BasicInfo:
        """获取基本信息"""
        self.get_observation_info()
        return BasicInfo(instrument=self.instrument, 
                         data_folder_name=self.data_folder_name, 
                         target_simplified_name=self.target_simplified_name, 
                         target_full_name=self.target_full_name, 
                         project_path=self.project_path, 
                         project_docs_path=self.project_docs_path, 
                         #project_name=self.project_name, 
                         data_path=self.data_path, 
                         evt_to_img_folder_path=self.evt_to_img_folder_path,
                         alignment_folder_path=self.alignment_folder_path,
                         cleaned_folder_path=self.cleaned_folder_path,
                         stacked_folder_path=self.stacked_folder_path,
                         evt_to_img_name_style=self.evt_to_img_name_style,
                         alignment_name_style=self.alignment_name_style,
                         cleaned_name_style=self.cleaned_name_style,
                         stacked_name_style=self.stacked_name_style,
                         target_id=self.target_id, 
                         obsid_initial=self.obsid_initial, 
                         obsid_final=self.obsid_final, 
                         epoch_name=self.epoch_name, 
                         orbital_keywords=self.orbital_keywords, 
                         observation_log_path=self.observation_log_path, 
                         observation_info=self.observation_info)

# ===================== 数据Clean =====================
class DataCleaningIndividual:
    """数据清理相关功能"""
    
    def __init__(self, obs: Dict[str, Any], basic_info: BasicInfo, 
                 target_coord: Optional[Tuple[float, float]] = None, verbose: Optional[bool] = True,
                 evt_to_img_folder: Optional[Union[str, Path]] = None, evt_to_img_name_style: Optional[str] = None,
                 alignment_folder: Optional[Union[str, Path]] = None, alignment_name_style: Optional[str] = None,
                 cleaned_folder: Optional[Union[str, Path]] = None, cleaned_name_style: Optional[str] = None
                 ):
        self.obs = obs
        self.obsid = obs['OBSID']
        self.ext_no = obs['EXT_NO']
        self.datatype = obs['DATATYPE'] # 'image' or 'event'
        if self.datatype == 'image' and 'grism' not in obs['FILTER']:
            self.scale = 1.004
        elif self.datatype == 'event':
            self.scale = 0.502
        elif 'grism' in obs['FILTER']:
            self.scale = 0.58
        else:
            self.scale = None
        self.filt_filename = normalize_filter_name(obs['FILTER'], output_format='filename')
        self.sk_coord_ds9 = (obs['x_pixel'], obs['y_pixel'])
        self.sk_coord_py = DS9Converter.ds9_to_coords(obs['x_pixel'], obs['y_pixel']) # col, row
        self.basic_info = basic_info
        if target_coord is None:
            target_coord = basic_info.target_coord
        self.target_coord_py = target_coord
        self.target_coord_ds9 = DS9Converter.coords_to_ds9(target_coord[0], target_coord[1])
        # quality flags
        # 0: no correction, 1: partial correction, 2: full correction 3: No need for correction
        self.smearing_correction = 0
        self.alignment = 0
        self.coincidence_loss_correction = 0
        self.offset_correction = 0
        self.star_removal = 0
        self.verbose = verbose
        self.mask = None
        # method tracking
        self.smearing_correction_stack_method = None
        self.star_mask_steps = []
        self.star_fill_steps = []
        # path record
        self.evt_file_path = get_obs_path(self.basic_info.data_path, self.obsid, self.obs['FILTER'], return_type='evt', datatype=self.datatype)
        self.exp_file_path = get_obs_path(self.basic_info.data_path, self.obsid, self.obs['FILTER'], return_type='exp', datatype=self.datatype)
        self.sk_file_path = get_obs_path(self.basic_info.data_path, self.obsid, self.obs['FILTER'], return_type='sk', datatype=self.datatype)
        self.evt_to_img_folder_path = self.basic_info.evt_to_img_folder_path if evt_to_img_folder is None else load_path(evt_to_img_folder, self.basic_info.project_path)
        self.alignment_folder_path = self.basic_info.alignment_folder_path if alignment_folder is None else load_path(alignment_folder, self.basic_info.project_path)
        self.cleaned_folder_path = self.basic_info.cleaned_folder_path if cleaned_folder is None else load_path(cleaned_folder, self.basic_info.project_path)
        evt_to_img_name_style = self.basic_info.evt_to_img_name_style if evt_to_img_name_style is None else evt_to_img_name_style
        alignment_name_style = self.basic_info.alignment_name_style if alignment_name_style is None else alignment_name_style
        cleaned_name_style = self.basic_info.cleaned_name_style if cleaned_name_style is None else cleaned_name_style
        self.evt_to_img_name = evt_to_img_name_style.format(obsid=self.obsid, ext_no=self.ext_no, filt_filename=self.filt_filename)
        self.alignment_name = alignment_name_style.format(obsid=self.obsid, ext_no=self.ext_no, filt_filename=self.filt_filename)
        self.cleaned_name = cleaned_name_style.format(obsid=self.obsid, ext_no=self.ext_no, filt_filename = self.filt_filename)
        self.evt_to_img_path = paths.get_subpath(self.evt_to_img_folder_path, self.evt_to_img_name)
        self.alignment_path = paths.get_subpath(self.alignment_folder_path, self.alignment_name)
        self.cleaned_path = paths.get_subpath(self.cleaned_folder_path, self.cleaned_name)


    def remove_motion_smearing(self, longest_elapsed_time: float = 30, stack_method = 'sum', binby2 = True):
        """去除motion造成的smearing"""
        t_elapsed = self.obs['TELAPSE']
        group_number = math.ceil(t_elapsed/longest_elapsed_time)
        if self.verbose:
            print(f'Processing observation: {self.obsid}; group number: {group_number}; elapsed time per group: {t_elapsed/group_number};')
        self.smearing_correction_stack_method = stack_method
        reduce_smear(self.evt_file_path, self.exp_file_path, self.basic_info.target_id, self.target_coord_py, group_number, \
                     self.smearing_correction_stack_method, self.evt_to_img_path, binby2=binby2, save_individual_images=False, image_unit_save='count')
        self.smearing_correction = 2
        if binby2:
            self.scale = 1.004
        if self.verbose:
            print(f'Smearing correction applied, saved to {self.evt_to_img_path}')
    
    def align_image(self):
        """图像对齐到target"""
        # 对 event: （如果变了target_coord，否则直接粘过来）打开fits，移动所有的extension；更新header[0]里的中心坐标位置
        # 对 image: 对sk和exp各自移动，放到一个fits里；header直接贴过来，在header[0]里更新中心坐标位置
        if self.datatype == 'event':
            shutil.move(self.evt_to_img_path, self.alignment_path)
            self.alignment = 2
            if self.verbose:
                print(f'Event-mode image moved to {self.alignment_path}')
        elif self.datatype == 'image':
            # 对 image: 对sk和exp各自移动，放到一个fits里；header直接贴过来，在header[0]里更新中心坐标位置
            with fits.open(self.sk_file_path, mode='readonly') as hdul:
                sk_hdu = hdul[self.ext_no].copy()
            with fits.open(self.exp_file_path, mode='readonly') as hdul:
                exp_hdu = hdul[self.ext_no].copy()
            sk_hdu.data = crop_image(sk_hdu.data, self.sk_coord_py, self.target_coord_py, fill_value=np.nan)
            sk_hdu.name = 'IMAGE'
            exp_hdu.data = crop_image(exp_hdu.data, self.sk_coord_py, self.target_coord_py, fill_value=0)
            exp_hdu.name = 'EXPOSURE'
            w = WCS(naxis=2)
            w.wcs.crpix = [self.target_coord_ds9[0], self.target_coord_ds9[1]]
            w.wcs.crval = [self.obs['RA'], self.obs['DEC']]
            w.wcs.cdelt = [sk_hdu.header['CDELT1'], sk_hdu.header['CDELT2']]
            w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            w.wcs.cunit = ['deg', 'deg']
            w.to_header(relax=True)
            sk_hdu.header.update(w.to_header())
            exp_hdu.header.update(w.to_header())
            sk_hdu.data[exp_hdu.data/np.max(exp_hdu.data) < 0.99] = np.nan
            error_data = np.sqrt(sk_hdu.data)
            error_hdu = fits.ImageHDU(error_data, name='ERROR')
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header['BUNIT'] = 'count'
            primary_hdu.header['COLPIXEL'] = (self.target_coord_py[0], 'Target X position in Python coordinates')
            primary_hdu.header['ROWPIXEL'] = (self.target_coord_py[1], 'Target Y position in Python coordinates')
            primary_hdu.header['DS9XPIX'] = (self.target_coord_ds9[0], 'Target X position in DS9 coordinates')
            primary_hdu.header['DS9YPIX'] = (self.target_coord_ds9[1], 'Target Y position in DS9 coordinates')
            primary_hdu.header['TARG_RA'] = (self.obs['RA'], 'Degrees')
            primary_hdu.header['TARG_DEC'] = (self.obs['DEC'], 'Degrees')
            primary_hdu.header[f'EXT{1}NAME'] = (sk_hdu.name, f'Name of extension {1}')
            primary_hdu.header[f'EXT{2}NAME'] = (error_hdu.name, f'Name of extension {2}')
            primary_hdu.header[f'EXT{3}NAME'] = (exp_hdu.name, f'Name of extension {3}')
            hdul = fits.HDUList([primary_hdu, sk_hdu, error_hdu, exp_hdu])
            hdul.writeto(self.alignment_path, overwrite=True)
            self.alignment = 2
            if self.verbose:
                print(f'Image-mode image aligned to target, saved to {self.alignment_path}')
        else:
            print('Invalid observation mode') 

    def create_cleaned_image(self):
        """创建cleaned image"""
        if self.alignment == 2:
            shutil.copyfile(self.alignment_path, self.cleaned_path)
            try:
                with fits.open(self.cleaned_path, mode='update') as hdul:
                    primary_hdu = hdul[0]
                    primary_hdu.header['COICORR'] = (False, 'Coincidence loss correction not applied')
                    primary_hdu.header['STARMASK'] = (False, 'Star mask not applied')
                    primary_hdu.header['STARFILL'] = (False, 'Star fill not applied')
                    primary_hdu.header['OFFCORR'] = (False, 'Offset correction not applied')
                    # 遍历所有 extension，找到 EXTNAME 为 'IMAGE' 的那个
                    #image_found = False
                    #for hdu in hdul[1:]:  # 从第一个 extension 开始
                    #    if hdu.name and hdu.name.upper() == 'IMAGE':
                    #        hdu.name = 'IMAGE'
                    #        image_found = True
                    #        primary_hdu.header[f'EXT{1}NAME'] = (hdu.name, f'Name of extension {1}')
                    #        if self.verbose:
                    #            print(f'Alignment image copied to cleaned image {self.cleaned_path}')
                    #        break
                    #if not image_found:
                    #    raise ValueError(f"No 'IMAGE' extension found in {self.alignment_path}")
            except Exception as e:
                if os.path.exists(self.cleaned_path):
                    os.remove(self.cleaned_path)
                raise e
        else:
            print("Do alignment first")
        
    def correct_offset(self, box_size: Tuple[int, int] = (41, 41), plot: bool = False,
                       save: bool = False,  img_path: Union[str, Path] = None):
        """
        修正offset
        box_size: width, height
        """
        if img_path is None:
            img_path = self.cleaned_path
        correct_offset_in_image(img_path = img_path, img_extension = 'IMAGE', target_coord = self.target_coord_py, 
                                datatype=self.datatype, box_size=box_size, plot=plot, save=save, verbose=self.verbose)
        self.offset_correction = 2

    def correct_coincidence_loss(self, plot: bool = False, save: bool = False):
        """修正coincidence loss"""
        correct_coi_loss_in_image(img_path = self.cleaned_path, img_extension = 'IMAGE', scale = self.scale, func = 'poole2008', 
                                  plot = plot, save = save, verbose = self.verbose)
        self.coincidence_loss_correction = 2
    
    def identify_stars(self, 
                     identify_method: str = 'sigma_clip', # 'sigma_clip', 'manual'
                     identify_paras: Dict[str, Any] = None,
                     focus_region: Union[np.ndarray, PixelRegion, None] = None, 
                     exclude_region: Union[np.ndarray, PixelRegion, None] = None,
                     plot: bool = False,
                     plot_vrange: Union[Tuple[float, float], None] = None,
                     save: bool = False,
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        'sigma_clip':
        identify_paras = {'sigma': float = 3, 'maxiters': int = 3, 'tile_size': int = None, 'area_size': None, \
                'expand_shrink_paras': {'radius': float = 1, 'method': 'expand' or 'shrink', 'speend': 'normal' or 'fast'}
                }
        'manual':
        identify_paras = {'target_coord': tuple(int, int) = (1000, 1000), 'radius': int = 50, \
                        'vmin': float = 0, 'vmax': float = 10, 
                        'region_plot': Union[PixelRegion, List[PixelRegion]] = None}
        """
        self.star_mask_steps, self.mask = save_starmask(img_path = self.cleaned_path, 
                                                        img_extension = 'IMAGE',
                                                        star_mask_steps = self.star_mask_steps,
                                                        identify_method = identify_method,
                                                        identify_paras = identify_paras,
                                                        focus_region = focus_region,
                                                        exclude_region = exclude_region,
                                                        plot = plot,
                                                        plot_vrange = plot_vrange,
                                                        save = save,
                                                        verbose = self.verbose,
                                                        )
        if save:
            self.star_removal = 1
    
    def delete_mask_extension(self):
        delete_starmask(img_path = self.cleaned_path, verbose = self.verbose)
        self.star_removal = 0
    
    def fill_stars(self, 
                   mask: Union[np.ndarray, None] = None,
                   fill_method: str = 'neighbors', 
                   fill_paras: Dict[str, Any] = None,
                   focus_region: Union[np.ndarray, PixelRegion, None] = None,
                   exclude_region: Union[np.ndarray, PixelRegion, None] = None,
                   plot: bool = False,
                   plot_vrange: Union[Tuple[float, float], None] = None,
                   save: bool = False,
                   ):
        """
        'neighbors':
        fill_paras = {'radius': int = 4, 'method': str = 'nearest', 'mean_or_median': str = 'mean'} # 'median_filter', 'uniform_filter'
        'tile_median':
        fill_paras = {'tile_size': int = 40}
        'rings':
        fill_paras = {'center': tuple(int, int), 'step': float, 'method': str = 'median', \
                      'start': float = None, 'end': float = None} # 'median', 'mean'
        """
        self.star_fill_steps = save_filled(img_path = self.cleaned_path, 
                                           img_extension = 'IMAGE',
                                           mask = mask,
                                           star_fill_steps=self.star_fill_steps,
                                           fill_method = fill_method,
                                           fill_paras = fill_paras,
                                           focus_region = focus_region,
                                           exclude_region = exclude_region,
                                           plot = plot,
                                           plot_vrange = plot_vrange,
                                           save = save,
                                           verbose = self.verbose,
                                           )
        if save:
            self.star_removal = 2

    def restore_fillment(self, coi_loss_corr: bool = True):
        with fits.open(self.alignment_path, mode='readonly') as hdul:
            img = hdul['IMAGE'].data.copy()
            exp = hdul['EXPOSURE'].data.copy()
        if coi_loss_corr:
            coi_map = get_coi_loss_map(img/exp, self.scale, 'poole2008')
            img_original = img*coi_map
        else:
            img_original = img
        with fits.open(self.cleaned_path, mode='update') as hdul:
            hdul['IMAGE'].data = img_original
            hdul[0].header['STARFILL'] = (False, 'Star fill not applied')
            hdul[0].header['FILLHIST'] = ''
        if self.verbose:
            print('Star fill restored.')
        

    def display_cleaned_images(self, vrange = None, radius = 20, max_cols = 4):
        """
        展示clean好的图像
        # 原始图像，mask, clean好的图像，exposure map，error map, coicidence loss map, 
        """
        # read data
        image_dict = {}
        vrange_list_default = []
        with fits.open(self.alignment_path, mode='readonly') as hdul:
            original_img = hdul['IMAGE'].data.copy()
        image_dict['original'] = original_img
        original_vrange = (np.min(original_img), np.max(original_img)*0.1)
        vrange_list_default.append(original_vrange)
        with fits.open(self.cleaned_path, mode='readonly') as hdul:
            cleaned_img = hdul['IMAGE'].data.copy()
            exp = hdul['EXPOSURE'].data.copy()
            err = hdul['ERROR'].data.copy()
            image_dict['cleaned'] = cleaned_img
            vrange_list_default.append(original_vrange)
            image_dict['expsure'] = exp
            vrange_list_default.append((np.min(exp), np.max(exp)))
            image_dict['error'] = err
            vrange_list_default.append((np.min(err), np.max(err)*0.1))
            if 'COICORR' in hdul:
                image_dict['coicorr'] = hdul['COICORR'].data.copy()
                vrange_list_default.append((np.min(image_dict['coicorr']), np.max(image_dict['coicorr'])))
            if 'STARMASK' in hdul:
                mask = hdul['STARMASK'].data.copy()
                image_dict['identified'] = mask
                vrange_list_default.append((0,1))
        if vrange is None:
            vrange = vrange_list_default
        xrange = (self.target_coord_py[0] - radius, self.target_coord_py[0] + radius)
        yrange = (self.target_coord_py[1] - radius, self.target_coord_py[1] + radius)
        fig, axes = multi_show(list(image_dict.values()), max_cols=max_cols, vrange=vrange, \
                               xrange=xrange, yrange=yrange, title_list=list(image_dict.keys()),
                               target_position_range = self.target_coord_py)
        fig.suptitle(f'{self.obsid} {self.ext_no} {self.filt_filename}')
        plt.show(block=True)
        plt.close()
        
class DataCleaningMultiple:
    def __init__(self, observations: Union[pd.DataFrame, List[Dict[str, Any]]], basic_info: BasicInfo, 
                 observations_all: Union[pd.DataFrame, List[Dict[str, Any]], None] = None, target_coord = (1000, 1000), verbose = True,
                 evt_to_img_folder: Optional[Union[str, Path]] = None, evt_to_img_name_style: Optional[str] = None,
                 alignment_folder: Optional[Union[str, Path]] = None, alignment_name_style: Optional[str] = None,
                 cleaned_folder: Optional[Union[str, Path]] = None, cleaned_name_style: Optional[str] = None,
                 stacked_folder: Optional[Union[str, Path]] = None, stacked_name_style: Optional[str] = None
                 ):
        """
        evt_to_img_name_style example: '{obsid}_{ext_no}_{filt_filename}.fits'
        """
        self.verbose = verbose
        filter_unique = table_to_df(observations)['FILTER'].unique()
        if len(filter_unique) == 1:
            self.filt_filename = normalize_filter_name(filter_unique[0], output_format='filename')
        else:
            self.filt_filename = 'Multiple'
            if verbose:
                print("Multiple filter filenames found, using Multiple")
        datatype_unique = table_to_df(observations)['DATATYPE'].unique()
        if len(datatype_unique) == 1:
            self.datatype = datatype_unique[0]
        else:
            self.datatype = 'Multiple'
            if verbose:
                print("Multiple datatypes found, using Multiple")
        if self.datatype == 'image' and 'grism' not in self.filt_filename:
            self.scale = 1.004
        elif self.datatype == 'event':
            self.scale = 0.502
        elif 'grism' in self.filt_filename:
            self.scale = 0.58
        elif self.datatype == 'Multiple':
            self.scale = 'Multiple'
        else:
            self.scale = None
        self.observations = table_to_list(observations)
        if observations_all is not None:
            self.observations_all = table_to_list(observations_all)
        else:
            self.observations_all = self.observations.copy()
        self.basic_info = basic_info
        self.epoch_name = basic_info.epoch_name
        self.target_coord_py = target_coord
        self.target_coord_ds9 = DS9Converter.coords_to_ds9(target_coord[0], target_coord[1])
        # 0: no correction, 1: partial correction, 2: full correction 3: No need for correction
        self.image_stack = 0
        self.star_removal = 0
        self.offset_correction = 0
        self.stack_method = None
        self.stack_unit = None
        self.star_mask_steps = []
        self.star_fill_steps = []
        self.evt_to_img_folder_path = self.basic_info.evt_to_img_folder_path if evt_to_img_folder is None else load_path(evt_to_img_folder, self.basic_info.project_path)
        self.alignment_folder_path = self.basic_info.alignment_folder_path if alignment_folder is None else load_path(alignment_folder, self.basic_info.project_path)
        self.cleaned_folder_path = self.basic_info.cleaned_folder_path if cleaned_folder is None else load_path(cleaned_folder, self.basic_info.project_path)
        self.stacked_folder_path = self.basic_info.stacked_folder_path if stacked_folder is None else load_path(stacked_folder, self.basic_info.project_path)
        self.evt_to_img_name_style = self.basic_info.evt_to_img_name_style if evt_to_img_name_style is None else evt_to_img_name_style
        self.alignment_name_style = self.basic_info.alignment_name_style if alignment_name_style is None else alignment_name_style
        self.cleaned_name_style = self.basic_info.cleaned_name_style if cleaned_name_style is None else cleaned_name_style
        self.stacked_name_style = self.basic_info.stacked_name_style if stacked_name_style is None else stacked_name_style
        self.stacked_name = self.stacked_name_style.format(epoch_name=self.epoch_name, filt_filename=self.filt_filename)
        self.stacked_path = paths.get_subpath(self.stacked_folder_path, self.stacked_name)
        self.path_dict = dict(
            evt_to_img_folder = self.evt_to_img_folder_path,
            evt_to_img_name_style = self.evt_to_img_name_style,
            alignment_folder = self.alignment_folder_path,
            alignment_name_style = self.alignment_name_style,
            cleaned_folder = self.cleaned_folder_path,
            cleaned_name_style = self.cleaned_name_style
        )
        
    def basic_loop(self, longest_elapsed_time: float = 30, smearing_correction_stack_method: str = 'sum',):
        for obs in self.observations_all:
            obs_cleaning = DataCleaningIndividual(obs, self.basic_info, self.target_coord_py,
                                                  verbose = self.verbose,
                                                  **self.path_dict
                                                  )
            if obs_cleaning.datatype == 'event':
                obs_cleaning.remove_motion_smearing(longest_elapsed_time=longest_elapsed_time,
                                                    stack_method = smearing_correction_stack_method,)
            obs_cleaning.align_image()
            if obs in self.observations:
                obs_cleaning.create_cleaned_image()
                obs_cleaning.correct_coincidence_loss(save = True)

    def offset_correction_loop(self, observations: Union[pd.DataFrame, List[Dict[str, Any]],None], box_size: Tuple[int, int] = (41, 41), plot: bool = False):
        if observations is None:
            observations = self.observations
        else:
            observations = table_to_list(observations)
        for obs in observations:
            obs_cleaning = DataCleaningIndividual(obs, self.basic_info, self.target_coord_py,
                                                  verbose = self.verbose,
                                                  **self.path_dict
                                                  )
            obs_cleaning.correct_offset(box_size = box_size, plot = plot)

    def stack(self, observations: Union[pd.DataFrame, List[Dict[str, Any]],None]= None, 
                     stack_method: str = 'sum', compressed: bool = False, comment = None, image_type: str = 'cleaned'):
        if observations is None:
            observations = self.observations
            filt_filename = self.filt_filename
            datatype = self.datatype
        else:
            filter_unique = table_to_df(observations)['FILTER'].unique()
            if len(filter_unique) == 1:
                filt_filename = normalize_filter_name(filter_unique[0], output_format='filename')
            else:
                filt_filename = 'Multiple'
            datatype_unique = table_to_df(observations)['DATATYPE'].unique()
            if len(datatype_unique) == 1:
                datatype = datatype_unique[0]
            else:
                datatype = 'Multiple'
            observations = table_to_list(observations)
        if filt_filename == 'Multiple':
            raise ValueError("Multiple filter filenames found!")
        if datatype == 'Multiple':
            print("Multiple datatypes found!")
        exp_list = []
        img_list = []
        err_list = []
        mask_list = []
        for obs in observations:
            if image_type == 'cleaned':
                cleaned_name = self.cleaned_name_style.format(obsid=obs['OBSID'], ext_no=obs['EXT_NO'], filt_filename=filt_filename)
                cleaned_path = paths.get_subpath(self.cleaned_folder_path, cleaned_name)
                image_path = cleaned_path
            elif image_type == 'aligned':
                aligned_name = self.alignment_name_style.format(obsid=obs['OBSID'], ext_no=obs['EXT_NO'], filt_filename=filt_filename)
                aligned_path = paths.get_subpath(self.alignment_folder_path, aligned_name)
                image_path = aligned_path
            with fits.open(image_path, mode='readonly') as hdul:
                img = hdul['IMAGE'].data.copy()
                exp = hdul['EXPOSURE'].data.copy()
                if image_type == 'aligned':
                    cr = img/exp
                    coi_loss_map = get_coi_loss_map(cr, scale=self.scale, func = 'poole2008')
                    img = img*coi_loss_map
                err = hdul['ERROR'].data.copy()
                img[exp/np.max(exp) < 0.99] = np.nan
                err[exp/np.max(exp) < 0.99] = np.nan
                img_list.append(img)
                exp_list.append(exp)
                err_list.append(err)
                if 'STARMASK' in hdul:
                    mask = hdul['STARMASK'].data.copy()
                    mask_list.append(mask)
        stacked_img, stacked_err = stack_images(images = img_list, method=stack_method, image_err = err_list, 
                                                median_err_params = {'method': 'mean', 'mask': True}, input_unit = 'count', exposure_list = exp_list
                                                )
        stacked_exp = np.sum(exp_list, axis=0)
        if stack_method == 'mean' or stack_method == 'median':
            self.stack_unit = 'count/s'
        elif stack_method == 'sum':
            self.stack_unit = 'count'
        if len(mask_list) == 0:
            image_dict = {'IMAGE': stacked_img, 'ERROR': stacked_err, 'EXPOSURE':stacked_exp}
        else:
            stacked_mask = (np.sum(mask_list, axis=0) > 0).astype(int)
            image_dict = {'IMAGE': stacked_img, 'ERROR': stacked_err, 'EXPOSURE':stacked_exp, 'STARMASK': stacked_mask}
        if comment is None:
            comment = 'No background subtraction'
        other_header_info={'EPOCH': self.epoch_name, 'DATATYPE': datatype, 'STACKMET': stack_method}
        save_stacked_fits(images_to_save=image_dict, save_path=self.stacked_path, obs_list = observations, target_position=self.target_coord_py,
                          script_name='pipeline_basic.py', compressed=compressed, comment=comment, other_header_info=other_header_info, stack_unit = self.stack_unit)
        if self.verbose:
            print(f'Stacked image saved to {self.stacked_path}')

    def correct_offset(self, box_size: Tuple[int, int] = (41, 41), plot: bool = False, save = False) -> np.ndarray:
        """
        修正offset
        box_size: width, height
        """
        with fits.open(self.stacked_path, mode='readonly') as hdul:
            datatype = hdul[0].header['DATATYPE']
        correct_offset_in_image(img_path = self.stacked_path, img_extension = 'IMAGE', target_coord = self.target_coord_py, 
                                datatype=datatype, box_size=box_size, plot=plot, save=save, verbose=self.verbose)
        self.offset_correction = 2

    def identify_stars(self, 
                     identify_method: str = 'sigma_clip', # 'sigma_clip', 'manual'
                     identify_paras: Dict[str, Any] = None,
                     focus_region: Union[np.ndarray, PixelRegion, None] = None, 
                     exclude_region: Union[np.ndarray, PixelRegion, None] = None,
                     plot: bool = False,
                     plot_vrange: Union[Tuple[float, float], None] = None,
                     save: bool = False,
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        'sigma_clip':
        identify_paras = {'sigma': float = 3, 'maxiters': int = 3, \
            'tile_size': int = None, 'area_size': None, \
                'expand_shrink_paras': None}
        'manual':
        identify_paras = {{'target_coord': tuple(int, int) = (1000, 1000), 'radius': int = 50, \
                        'vmin': float = 0, 'vmax': float = 10, 
                        'region_plot': Union[PixelRegion, List[PixelRegion]] = None}
        """
        self.star_mask_steps, self.mask = save_starmask(img_path = self.stacked_path, 
                                                        img_extension = 'IMAGE',
                                                        star_mask_steps = self.star_mask_steps,
                                                        identify_method = identify_method,
                                                        identify_paras = identify_paras,
                                                        focus_region = focus_region,
                                                        exclude_region = exclude_region,
                                                        plot = plot,
                                                        plot_vrange = plot_vrange,
                                                        save = save,
                                                        verbose = self.verbose,
                                                        )
        if save:
            self.star_removal = 1
    
    def delete_mask_extension(self):
        delete_starmask(img_path = self.stacked_path, verbose = self.verbose)
        self.star_removal = 0
    
    def fill_stars(self, 
                   mask: Union[np.ndarray, None] = None,
                   fill_method: str = 'neighbors', 
                   fill_paras: Dict[str, Any] = None, 
                   focus_region: Union[np.ndarray, PixelRegion, None] = None,
                   exclude_region: Union[np.ndarray, PixelRegion, None] = None,
                   plot: bool = False,
                   plot_vrange: Union[Tuple[float, float], None] = None,
                   save: bool = False,
                   ):
        """
        'neighbors':
        fill_paras = {'radius': int = 4, 'method': str = 'nearest', 'mean_or_median': str = 'mean'} # 'median_filter', 'uniform_filter'
        'tile_median':
        fill_paras = {'tile_size': int = 40}
        'rings':
        fill_paras = {'center': tuple(int, int), 'step': float, 'method': str = 'median', \
                      'start': float = None, 'end': float = None} # 'median', 'mean'
        """
        self.star_fill_steps = save_filled(img_path = self.stacked_path, 
                                           img_extension = 'IMAGE',
                                           star_fill_steps=self.star_fill_steps,
                                           mask = mask,
                                           fill_method = fill_method,
                                           fill_paras = fill_paras,
                                           focus_region = focus_region,
                                           exclude_region = exclude_region,
                                           plot = plot,
                                           plot_vrange = plot_vrange,
                                           save = save,
                                           verbose = self.verbose,
                                           )
        if save:
            self.star_removal = 2

    def restore_fillment(self):
        pass

    def display_all_observations(self, observations: Union[pd.DataFrame, List[Dict[str, Any]],None] = None, 
                       radius = 20, vrange=None, max_cols = 4, image_unit_show='count'):
        if observations is None:
            observations = self.observations
        else:
            observations = table_to_list(observations)
        image_list = []
        title_list = []
        target_col = self.target_coord_py[0]
        target_row = self.target_coord_py[1]
        xrange = (target_col - radius, target_col + radius)
        yrange = (target_row - radius, target_row + radius)
        for obs in observations:
            filt_filename = normalize_filter_name(obs['FILTER'], output_format='filename')
            cleaned_name = self.cleaned_name_style.format(obsid=obs['OBSID'], ext_no=obs['EXT_NO'], filt_filename=filt_filename)
            cleaned_path = paths.get_subpath(self.cleaned_folder_path, cleaned_name)
            with fits.open(cleaned_path, mode='readonly') as hdul:
                img = hdul['IMAGE'].data.copy()
            if image_unit_show == 'count/s':
                img = img/obs['EXPOSURE']
            if image_unit_show == 'count':
                pass
            else:
                raise ValueError(f"Invalid image_unit_show: {image_unit_show}")
            image_list.append(img)
            title_list.append(f'{obs["OBSID"]} {obs["EXT_NO"]} {obs["FILTER"]}')
        fig, axes = multi_show(image_list, max_cols=max_cols, vrange=vrange, xrange=xrange, yrange=yrange, title_list=title_list)
        # draw direction compass
        if len(image_list) > max_cols:
            ax0 = axes[0][0]
        else:
            ax0 = axes[0]
        draw_direction_compass(ax0, 
                               directions={'N': 0, 'E': 90, 'v': self.basic_info.observation_info['velocityPA']-180, '☉': self.basic_info.observation_info['sunTargetPA']-180},
                               colors='white',
                               position=(0.15, 0.85), 
                               arrow_length=0.08, 
                               text_offset=0.02, 
                               fontsize=10)
        plt.show(block=True)
        plt.close()

def display_stacked_images(project_path: Union[str, Path] = None,
                           stacked_folder: Optional[Union[str, Path]] = 'stacked', 
                           stacked_name_style: Optional[str] = None, 
                           epoch_name: Optional[str] = None, 
                           target_coord_py: Tuple[float, float] = (1000, 1000), 
                           radius=20,
                           vrange_v=(0, 1.0), 
                           vrange_uw1=(0, 0.1), 
                           image_unit_show='count',
                           compass_paras = None,
                           scalebar_paras = None,
                           observation_time = None,
                           save_path = None):
    stacked_folder_path = load_path(stacked_folder, project_path)
    if stacked_name_style is None:
        stacked_name_style = '{epoch_name}_{filt_filename}.fits'
    filt_filename_v = normalize_filter_name('v', output_format='filename')
    filt_filename_uw1 = normalize_filter_name('uw1', output_format='filename')
    stacked_name_v = stacked_name_style.format(epoch_name=epoch_name, filt_filename=filt_filename_v)
    stacked_name_uw1 = stacked_name_style.format(epoch_name=epoch_name, filt_filename=filt_filename_uw1)
    stacked_path_v = paths.get_subpath(stacked_folder_path, stacked_name_v)
    stacked_path_uw1 = paths.get_subpath(stacked_folder_path, stacked_name_uw1)
    
    with fits.open(stacked_path_v, mode='readonly') as hdul:
        img_v = hdul['IMAGE'].data.copy()
    with fits.open(stacked_path_uw1, mode='readonly') as hdul:
        img_uw1 = hdul['IMAGE'].data.copy()
    sunTargetPA_v = fits.getheader(stacked_path_v, ext=0)['sunTargetPA']
    velocityPA_v = fits.getheader(stacked_path_v, ext=0)['velocityPA']
    exposure_v = fits.getheader(stacked_path_v, ext=0)['EXPTIME']
    unit_v = fits.getheader(stacked_path_v, ext=0)['BUNIT']
    sunTargetPA_uw1 = fits.getheader(stacked_path_uw1, ext=0)['sunTargetPA']
    velocityPA_uw1 = fits.getheader(stacked_path_uw1, ext=0)['velocityPA']
    exposure_uw1 = fits.getheader(stacked_path_uw1, ext=0)['EXPTIME']
    unit_uw1 = fits.getheader(stacked_path_uw1, ext=0)['BUNIT']
    
    if image_unit_show == 'count' and unit_v == 'count/s':
        img_v = img_v * exposure_v
    elif image_unit_show == 'count/s' and unit_v == 'count':
        img_v = img_v / exposure_v
    
    if image_unit_show == 'count' and unit_uw1 == 'count/s':
        img_uw1 = img_uw1 * exposure_uw1
    elif image_unit_show == 'count/s' and unit_uw1 == 'count':
        img_uw1 = img_uw1 / exposure_uw1
    
    xrange = (target_coord_py[0] - radius, target_coord_py[0] + radius)
    yrange = (target_coord_py[1] - radius, target_coord_py[1] + radius)
    fig, (ax_v, ax_uw1) = plt.subplots(1, 2, figsize=(12, 6))
    if compass_paras is None:
        compass_paras = {'directions': {'N': 0, 'E': 90, 'v': velocityPA_v-180, '☉': sunTargetPA_v-180},
                         'colors': 'white',
                         'position': (0.17, 0.83), 
                         'arrow_length': 0.08,
                         'arrow_width': 0.1,
                         'headwidth': 4,
                         'headlength': 3,
                         'text_offset': 0.02, 
                         'fontsize': 10}
    if scalebar_paras is None:
        scalebar_paras = {'length': 100, 
                          'label_top': r'$20\,000~$'+'km', 
                          'label_bottom': r"45$^{\prime\prime}$", 
                          'position': (0.85, 0.1), 
                          'color': 'white', 
                          'linewidth': 1, 
                          'text_offset': 5}
    ax_v.imshow(img_v, vmin=vrange_v[0], vmax=vrange_v[1], origin='lower', cmap='viridis')
    ax_v.plot(target_coord_py[0], target_coord_py[1], 'rx', markersize=5)
    if observation_time is not None:
        text_v = 'V\n'+r'$Swift$/UVOT'+'\n'+f'{observation_time}'
    else:
        text_v = 'V\n'+r'$Swift$/UVOT'
    ax_v.text(0.95, 0.03, text_v, 
              transform=ax_v.transAxes,ha='right', va='bottom',color='white')
    ax_v.set_xlim(xrange[0], xrange[1])
    ax_v.set_ylim(yrange[0], yrange[1])
    for spine in ax_v.spines.values():
        spine.set_visible(False)
    draw_direction_compass(ax_v, 
                           directions=compass_paras.get('directions', {'N': 0, 'E': 90, 'v': velocityPA_v-180, '☉': sunTargetPA_v-180}),
                           colors=compass_paras.get('colors', 'white'),
                           position=compass_paras.get('position', (0.17, 0.83)), 
                           arrow_length=compass_paras.get('arrow_length', 0.08),
                           arrow_width=compass_paras.get('arrow_width', 0.1),
                           headwidth=compass_paras.get('headwidth', 4),
                           headlength=compass_paras.get('headlength', 3),
                           text_offset=compass_paras.get('text_offset', 0.02), 
                           fontsize=compass_paras.get('fontsize', 10))
    draw_scalebar(ax_v,
                  length=scalebar_paras.get('length', 100),
                  label_top=scalebar_paras.get('label_top', r'$20\,000~$'+'km'),
                  label_bottom=scalebar_paras.get('label_bottom', r"45$^{\prime\prime}$"),
                  position=scalebar_paras.get('position', (0.85, 0.1)),
                  color=scalebar_paras.get('color', 'white'),
                  linewidth=scalebar_paras.get('linewidth', 1),
                  text_offset=scalebar_paras.get('text_offset', 5),
                  )
    ax_v.tick_params(axis='both', which='major', direction='in', colors='white', length=3, width=0.5,
                     top=True, bottom=True, left=True, right=True,
                     labelleft=False, labelbottom=False, labeltop=False, labelright=False)
    ax_uw1.imshow(img_uw1, vmin=vrange_uw1[0], vmax=vrange_uw1[1], origin='lower', cmap='viridis')
    ax_uw1.plot(target_coord_py[0], target_coord_py[1], 'rx', markersize=5)
    if observation_time is not None:
        text_uw1 = 'UVW1\n'+r'$Swift$/UVOT'+'\n'+f'{observation_time}'
    else:
        text_uw1 = 'UVW1\n'+r'$Swift$/UVOT'
    ax_uw1.text(0.95, 0.03, text_uw1, 
                transform=ax_uw1.transAxes,ha='right', va='bottom',color='white')
    ax_uw1.set_xlim(xrange[0], xrange[1])
    ax_uw1.set_ylim(yrange[0], yrange[1])
    for spine in ax_uw1.spines.values():
        spine.set_visible(False)
    draw_direction_compass(ax_uw1, 
                           directions=compass_paras.get('directions', {'N': 0, 'E': 90, 'v': velocityPA_uw1-180, '☉': sunTargetPA_uw1-180}),
                           colors=compass_paras.get('colors', 'white'),
                           position=compass_paras.get('position', (0.17, 0.83)), 
                           arrow_length=compass_paras.get('arrow_length', 0.08), 
                           arrow_width=compass_paras.get('arrow_width', 0.1),
                           headwidth=compass_paras.get('headwidth', 4),
                           headlength=compass_paras.get('headlength', 3),
                           text_offset=compass_paras.get('text_offset', 0.02), 
                           fontsize=compass_paras.get('fontsize', 10))
    draw_scalebar(ax_uw1,
                  length=scalebar_paras.get('length', 100),
                  label_top=scalebar_paras.get('label_top', r'$20\,000~$'+'km'),
                  label_bottom=scalebar_paras.get('label_bottom', r"45$^{\prime\prime}$"),
                  position=scalebar_paras.get('position', (0.85, 0.1)),
                  color=scalebar_paras.get('color', 'white'),
                  linewidth=scalebar_paras.get('linewidth', 1),
                  text_offset=scalebar_paras.get('text_offset', 5),
                  )
    ax_uw1.tick_params(axis='both', which='major', direction='in', colors='white', length=3, width=0.5, 
                       top=True, bottom=True, left=True, right=True,
                       labelleft=False, labelbottom=False, labeltop=False, labelright=False)
    for line in ax_v.xaxis.get_ticklines() + ax_v.yaxis.get_ticklines():
        line.set_alpha(0.5)
    for line in ax_uw1.xaxis.get_ticklines() + ax_uw1.yaxis.get_ticklines():
        line.set_alpha(0.5)
    fig.subplots_adjust(wspace=0.05, hspace=0)
    plt.tight_layout(pad=0.1)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show(block=True)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f'Stacked images saved to {save_path}')
    plt.close()