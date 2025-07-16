import os
from astropy.io import fits
from datetime import datetime
from astropy.time import Time
import numpy as np
from typing import Union, Tuple, List, Dict, Any, Optional
from astropy.table import Table
import subprocess
from pathlib import Path
import inspect
import os
import ipynbname
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.units import Quantity


def save_astropy_table(data_table, output_path, verbose=True):
    """
    Save an Astropy table to a file.

    Parameters:
    data_table (astropy.table.Table): The Astropy table to save.
    output_path (str): Absolute path for the output file. ecsv recommended.
    """
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if isinstance(data_table, Table):
        data_table.write(output_path, overwrite=True, delimiter=',')
    if verbose:
        file_format = str(output_path).split('.')[-1]
        print(f"Data saved to: {output_path}, format: {file_format}")

def show_or_save_astropy_table(data_table, output_path=None):
    """
    Process an Astropy table by either saving it to a file or printing it to console.

    Parameters:
    data_table (astropy.table.Table): The Astropy table to process.
    output_path (str, optional): Path for the output file. If None, the table will be printed to console. For saving, absolute is recommended.
    """
    if output_path is None:
        data_table.pprint(max_lines=-1, max_width=-1)
    else:
        save_astropy_table(data_table, output_path)

def load_table(table_path, type='astropy_table', verbose=True):
    if verbose:
        file_format = str(table_path).split('.')[-1]
        print(f"Loading data from: {table_path}, format: {file_format}")
    file_path = Path(table_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == '.csv' and type == 'df':
        # CSV文件
        dtype_dict = {'OBSID': str} if 'OBSID' in pd.read_csv(file_path, nrows=0).columns else {}
        df = pd.read_csv(file_path, dtype=dtype_dict)
        return df
    elif type == 'astropy_table' and file_path.suffix.lower() in ['.ecsv', '.csv']:
        # ECSV文件 (astropy格式)
        table = Table.read(file_path)
        return table 
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

def is_dict_list(obj) -> bool:
    return (
        isinstance(obj, list)
        and all(isinstance(item, dict) for item in obj)
    )

def is_list_dict(obj) -> bool:
    return (
        isinstance(obj, dict)
        and all(isinstance(v, list) for v in obj.values())
        and len(set(len(v) for v in obj.values())) <= 1  # 可选：所有列表长度一致
    )


class TableConverter:
    @staticmethod
    def df_to_astropy_table(df: pd.DataFrame) -> Table:
        """
        Convert a DataFrame to an Astropy Table.
        Assumes each value in the DataFrame is either:
        - a Quantity (e.g., [1*u.km, 2*u.km]) [RECOMMENDED]
        - or a normal scalar
        """
        table = Table()
        for col in df.columns:
            col_data = df[col]
            if isinstance(col_data.iloc[0], Quantity):
                table[col] = u.Quantity(col_data.tolist())
            else:
                table[col] = col_data.tolist()
        return table

    @staticmethod
    def astropy_table_to_df(table: Table) -> pd.DataFrame:
        """Convert Astropy Table to DataFrame. Each value will be Quantity if the column has units."""
        data = {}
        for col in table.colnames:
            col_data = table[col]
            if hasattr(col_data, 'unit') and col_data.unit is not None:
                data[col] = [val * col_data.unit for val in col_data]
            else:
                data[col] = col_data.tolist()
        return pd.DataFrame(data)

    @staticmethod
    def astropy_table_to_dict_list(table: Table) -> List[Dict[str, Any]]:
        """Convert Astropy Table to list of dicts. Values preserve units as Quantities."""
        rows = []
        for row in table:
            row_dict = {}
            for col in table.colnames:
                val = row[col]
                if hasattr(table[col], 'unit') and table[col].unit is not None:
                    row_dict[col] = val * table[col].unit
                else:
                    row_dict[col] = val
            rows.append(row_dict)
        return rows

    @staticmethod
    def dict_list_to_astropy_table(dict_list: List[Dict[str, Any]]) -> Table:
        """Convert list of dicts to Astropy Table. Quantities are recognized and units applied."""
        if not dict_list:
            return Table()
        columns = {}
        for key in dict_list[0].keys():
            col_values = [d[key] for d in dict_list]
            if isinstance(col_values[0], Quantity):
                columns[key] = u.Quantity(col_values)
            else:
                columns[key] = col_values
        return Table(columns)

    @staticmethod
    def astropy_table_to_list_dict(table: Table) -> Dict[str, List[Any]]:
        """Convert Astropy Table to dict of lists. Preserves units using Quantities."""
        result = {}
        for col in table.colnames:
            col_data = table[col]
            if hasattr(col_data, 'unit') and col_data.unit is not None:
                result[col] = [val * col_data.unit for val in col_data]
            else:
                result[col] = col_data.tolist()
        return result

    @staticmethod
    def list_dict_to_astropy_table(list_dict: Dict[str, List[Any]]) -> Table:
        """Convert dict of lists to Astropy Table. Quantities are recognized and units applied."""
        if not list_dict:
            return Table()
        lengths = {len(v) for v in list_dict.values()}
        if len(lengths) != 1:
            raise ValueError("All lists must have the same length.")
        columns = {}
        for key, values in list_dict.items():
            if isinstance(values[0], Quantity):
                columns[key] = u.Quantity(values)
            else:
                columns[key] = values
        return Table(columns)
    
    @staticmethod
    def df_to_dict_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert a DataFrame to a list of dictionaries."""
        if df.empty:
            return []
        return df.to_dict(orient='records')
    
    @staticmethod
    def dict_list_to_df(dict_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert a list of dictionaries to a DataFrame."""
        if not dict_list:
            return pd.DataFrame()
        return pd.DataFrame(dict_list)
    
    @staticmethod
    def dict_list_to_list_dict(dict_list: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not dict_list:
            return {}
        keys = dict_list[0].keys()
        return {key: [d[key] for d in dict_list] for key in keys}

    @staticmethod
    def list_dict_to_dict_list(list_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        if not list_dict:
            return []
        lengths = {len(v) for v in list_dict.values()}
        if len(lengths) != 1:
            raise ValueError("All lists in the dictionary must have the same length.")
        keys = list_dict.keys()
        length = next(iter(lengths))
        return [{key: list_dict[key][i] for key in keys} for i in range(length)]
    

def expand_quantity_in_dict(data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    遍历 dict 中所有列（list），若某列是 Quantity，则将其展开为元素为 Quantity 的列表。
    例：[1, 2, 3]*u.km -> [1*u.km, 2*u.km, 3*u.km]
    Parameters:
        data: dict of lists，列名对应值列表
    Returns:
        dict: 新的 dict，所有 Quantity 类型已被展开
    """
    result = {}
    for key, values in data.items():
        if isinstance(values, Quantity):
            result[key] = [v * values.unit for v in values.value]
        else:
            result[key] = values
    return result
    

def compress_quantity_in_dict(data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    将 dict 中所有列（list）中元素均为 Quantity 的，压缩为单个 Quantity 列。
    例：[1*u.km, 2*u.km, 3*u.km] → [1, 2, 3]*u.km
    Parameters:
        data: dict of lists，列名对应值列表
    Returns:
        dict: 新的 dict，Quantity 列被压缩为统一 Quantity
    """
    result = {}
    for key, values in data.items():
        if isinstance(values, list) and values and isinstance(values[0], Quantity):
            try:
                result[key] = u.Quantity(values)
            except Exception:
                # 有可能单位不一致，如 [1*u.km, 2*u.m]，则不压缩
                result[key] = values
        else:
            result[key] = values
    return result

class TableColumnManager:
    def __init__(self, table_like):
        self._original = table_like
        self.table = self._parse_table()

    def _parse_table(self):
        """解析 self._original 成 astropy Table"""
        if isinstance(self._original, str) and os.path.isfile(self._original):
            return load_table(self._original)
        elif isinstance(self._original, pd.DataFrame):
            return TableConverter.df_to_astropy_table(self._original)
        elif isinstance(self._original, list) and is_dict_list(self._original):
            return TableConverter.dict_list_to_astropy_table(self._original)
        elif isinstance(self._original, dict) and is_list_dict(self._original):
            return TableConverter.list_dict_to_astropy_table(self._original)
        elif isinstance(self._original, Table):
            return self._original
        else:
            raise ValueError("Unsupported table format.")

    def _save(self):
        """如果输入是路径，则保存到原路径"""
        ext = str(self._original).split('.')[-1].lower()
        if ext in ['csv', 'ecsv']:
            save_astropy_table(self.table, self._original, verbose=False)
            print(f"💾 Table saved to {self._original}")
        else:
            raise ValueError(f"Cannot determine format from extension: '{ext}'")

    def _get_table(self):
        if isinstance(self._original, pd.DataFrame):
            return TableConverter.astropy_table_to_df(self.table)
        elif isinstance(self._original, list) and is_dict_list(self._original):
            return TableConverter.astropy_table_to_dict_list(self.table)
        elif isinstance(self._original, dict) and is_list_dict(self._original):
            return TableConverter.astropy_table_to_list_dict(self.table)
        elif isinstance(self._original, Table):
            return self.table
        else:
            raise ValueError("Unsupported table format.")
        #return self.table

    @staticmethod
    def add_columns(table_like, new_columns: dict, after_column: str = None):
        manager = TableColumnManager(table_like)
        table = manager.table
        n_rows = len(table)

        wrong_length = [
            col for col, val in new_columns.items() if len(val) != n_rows
        ]
        if wrong_length:
            raise ValueError(f"❌ Length mismatch: {wrong_length} (table has {n_rows} rows)")

        insert_at = len(table.columns)
        if after_column:
            if after_column not in table.colnames:
                raise ValueError(f"❌ Column '{after_column}' not found.")
            insert_at = table.colnames.index(after_column) + 1

        for col_name, col_data in new_columns.items():
            table.add_column(col_data, name=col_name, index=insert_at)
            insert_at += 1

        print(f"✅ Added columns: {list(new_columns.keys())}")
        return manager._save() if isinstance(table_like, str) else manager._get_table()

    @staticmethod
    def remove_columns(table_like, columns_to_remove):
        manager = TableColumnManager(table_like)
        table = manager.table

        if isinstance(columns_to_remove, str):
            columns_to_remove = [columns_to_remove]

        missing = [col for col in columns_to_remove if col not in table.colnames]
        if missing:
            raise ValueError(f"❌ These columns do not exist: {missing}")

        table.remove_columns(columns_to_remove)
        print(f"✅ Removed columns: {columns_to_remove}")
        return manager._save() if isinstance(table_like, str) else manager._get_table()
    
def get_caller_filename():
    """
    获取调用者的文件名，支持普通Python脚本和Jupyter Notebook
    
    Returns
    -------
    str
        调用者的文件名
        - 对于普通Python脚本: 返回脚本文件名 (如 'script.py')
        - 对于Jupyter Notebook: 返回notebook文件名 (如 'analysis.ipynb')
        - 如果无法获取: 返回默认名称(Unknown)
    """
    
    frame = inspect.currentframe()
    try:
        # 获取调用者的frame (上一级调用栈)
        caller_frame = frame.f_back
        caller_filename = caller_frame.f_code.co_filename
        
        # 处理jupyter notebook的情况
        if caller_filename.startswith('<ipython-input-') or caller_filename == '<stdin>':
            try:
                # 方法1: 尝试从IPython获取notebook名称
                notebook_path = ipynbname.path()
                return os.path.basename(notebook_path)
            except ImportError:
                try:
                    # 方法2: 尝试从环境变量获取
                    if 'JPY_SESSION_NAME' in os.environ:
                        return os.environ['JPY_SESSION_NAME'] + '.ipynb'
                    else:
                        return "Unknown"
                except:
                    return "Unknown"
            except:
                return "Unknown"
        else:
            # 普通Python脚本
            return os.path.basename(caller_filename)
    finally:
        del frame

def compress_fits(fits_file, remove_original=True):
    fits_file = Path(fits_file)
    compressed_path = fits_file.with_name(fits_file.name + ".fz")
    if compressed_path.exists():
        os.remove(compressed_path)
    try:
        subprocess.run(["fpack", str(fits_file)], check=True)
        if remove_original:
            os.remove(fits_file)
    except subprocess.CalledProcessError as e:
        print("Compression failed:", e)

def save_stacked_fits(images_to_save: dict, 
                      save_path: str, obs_list: list, 
                      target_position: Union[Tuple, List],
                      script_name: str = None, 
                      bkg_list: list = None, bkg_error_list: list = None, 
                      comment: str = None,
                      other_header_info: dict = None,
                      compressed: bool = False):
    """
    将stacked image保存为fits文件
    
    Parameters
    ----------
    save_path : str
        保存路径
    obs_list : list
        观测列表
    target_position : Union[Tuple, List]
        目标位置
    images_to_save : dict
        要保存的图像字典，key为extension名称，value为对应的2d array
        例如: {'STACKED_IMAGE': stacked_image, 'STACKED_ERROR': stacked_error, 'STACKED_IMAGE_OVERLAP': stacked_image_overlap}
    script_name : str, optional
        脚本名称
    bkg_list : list, optional
        背景值列表
    bkg_error_list : list, optional
        背景误差值列表
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
    
    # 获取文件名列表
    try:
        file_names = [obs['file_name'] for obs in obs_list]
    except:
        file_names = None
    try:
        obs_ids = sorted(set([obs['OBSID'] for obs in obs_list]))
    except:
        obs_ids = None
    
    # 计算总曝光时间
    try:    
        total_exptime = sum(obs['EXPTIME'] for obs in obs_list)
    except:
        total_exptime = sum(obs['EXPOSURE'] for obs in obs_list)
    
    # 获取滤光片
    filt = obs_list[0]['FILTER']  # 假设所有观测使用相同的滤光片
    
    # 计算平均rh, delta, elongation
    avg_rh = np.mean([obs['r'] for obs in obs_list])
    avg_delta = np.mean([obs['delta'] for obs in obs_list])
    avg_elong = np.mean([obs['elong'] for obs in obs_list])
    avg_alpha = np.mean([obs['alpha'] for obs in obs_list])
    avg_ra_rate = np.mean([obs['RA*cos(Dec)_rate'] for obs in obs_list])
    avg_dec_rate = np.mean([obs['DEC_rate'] for obs in obs_list])
    avg_sunTargetPA = np.mean([obs['sunTargetPA'] for obs in obs_list])
    avg_velocityPA = np.mean([obs['velocityPA'] for obs in obs_list])
    
    # 获取目标位置
    target_pos = target_position
    
    # 写入header信息
    primary_header['DATE_OBS'] = (first_date_obs_str, 'Start time of first observation')
    primary_header['DATE_END'] = (last_date_end_str, 'End time of last observation')
    primary_header['MIDTIME'] = (midtime_iso, 'Middle time between first and last observation')
    try:
        primary_header['YEAR'] = (obs_list[0]['date'], 'Observation year')
    except:
        pass
    if file_names is not None:
        primary_header['FILELIST'] = (', '.join(f'{file_name}' for file_name in file_names), 'List of files used in stacking')
    primary_header['EXPTIME'] = (total_exptime, 'Total exposure time [s]')
    primary_header['BUNIT'] = ('COUNTS/S', 'Physical unit of array values')
    primary_header['COLPIXEL'] = (target_pos[0], 'Target X position in Python coordinates')
    primary_header['ROWPIXEL'] = (target_pos[1], 'Target Y position in Python coordinates')
    primary_header['DS9XPIX'] = (target_pos[0] + 1, 'Target X position in DS9 coordinates')
    primary_header['DS9YPIX'] = (target_pos[1] + 1, 'Target Y position in DS9 coordinates')
    primary_header['FILTER'] = (filt, 'Filter used in observations')
    primary_header['R'] = (avg_rh, 'Average heliocentric distance [AU]')
    primary_header['DELTA'] = (avg_delta, 'Average geocentric distance [AU]')
    primary_header['ELONG'] = (avg_elong, 'Average solar elongation [deg]')
    primary_header['ALPHA'] = (avg_alpha, 'Average phase angle [deg]')
    primary_header['RA_RATE'] = (avg_ra_rate, 'Average RA rate [deg/s]')
    primary_header['DEC_RATE'] = (avg_dec_rate, 'Average Dec rate [deg/s]')
    primary_header['sunTargetPA'] = (avg_sunTargetPA, 'Position angle of sun-to-target [deg]')
    primary_header['velocityPA'] = (avg_velocityPA, 'Position angle of velocity vector [deg]')

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
    
    primary_header.add_comment('EXP LIST: '+ ', '.join(f'{obs["OBSID"]}'[-4:]+f'_{obs["EXT_NO"]}' for obs in obs_list))
    primary_header.add_comment(comment)
    # 创建HDU列表，从primary开始
    hdul = fits.HDUList([primary_hdu])
    
    # 循环创建各个extension
    for ext_name, image_data in images_to_save.items():
        if image_data is not None:
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
    
    primary_header['BUNIT'] = ('ELECTRONS/S', 'Physical unit of array values')
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




